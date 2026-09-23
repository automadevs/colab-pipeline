#!/usr/bin/env python3
"""Opt-in E2E check for a private Hugging Face repository.

Does NOT embed tokens. Reads only from the environment:

  HF_TOKEN              required for private access
  HF_PRIVATE_TEST_REPO  e.g. owner/private-repo
  HF_PRIVATE_TEST_FILE  optional path inside the repo (small file preferred)
  HF_PRIVATE_TEST_REV   optional revision (default: main)

If credentials/repo are missing, prints an explicit NOT RUN status and exits 0
so CI/local suites are not marked failed. Never prints the token value.

Usage:
  python scripts/e2e_hf_private_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

NOT_RUN_REASON = "no suitable private test repository/credential available"


def _report_not_run(detail: str) -> int:
    print("HF private repository E2E: NOT RUN")
    print(f"Reason: {NOT_RUN_REASON}")
    print(f"Detail: {detail}")
    return 0


def main() -> int:
    token = os.environ.get("HF_TOKEN") or ""
    repo = (os.environ.get("HF_PRIVATE_TEST_REPO") or "").strip()
    file_path = (os.environ.get("HF_PRIVATE_TEST_FILE") or "").strip()
    revision = (os.environ.get("HF_PRIVATE_TEST_REV") or "main").strip() or "main"

    if not token:
        return _report_not_run("HF_TOKEN is unset")
    if not repo:
        return _report_not_run("HF_PRIVATE_TEST_REPO is unset")

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from kaggle_dataset_manager import (  # noqa: E402
        cleanup_hf_cache,
        configure_hf_cache,
        download_hf_file,
        _hf_file_size,
        _hf_list_repo_files,
    )

    cache_dir = configure_hf_cache()
    try:
        print("[E2E] HF private check starting (mocked=False)")
        print(f"[E2E] repo={repo} revision={revision}")
        print(f"[E2E] HF_HOME={os.environ.get('HF_HOME')}")
        print(f"[E2E] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
        print(f"[E2E] HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")

        files = _hf_list_repo_files(repo, revision=revision, token=token)
        if not files:
            print("[E2E] FAIL: metadata list returned no files")
            return 1

        target = file_path or next(
            (name for name in files if name.lower().endswith((".txt", ".json", ".md", ".safetensors", ".bin"))),
            files[0],
        )
        print(f"[E2E] metadata ok ({len(files)} file(s)); downloading: {target}")
        size = _hf_file_size(repo, target, revision=revision, token=token)
        print(f"[E2E] reported size={size}")

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            item = download_hf_file(
                repo_id=repo,
                file_path=target,
                category="loras",
                staging_dir=staging,
                hf_token=token,
                revision=revision,
                base_model="e2e",
            )
            local = staging / item.path
            if not local.exists() or local.stat().st_size <= 0:
                print("[E2E] FAIL: downloaded file missing or empty")
                return 1
            print(f"[E2E] download ok: {item.path} size={item.size} sha256={item.sha256}")

        print("HF private repository E2E: PASS")
        return 0
    except Exception as exc:
        # Never echo env values; exception text is already redacted by HF helpers when possible.
        print(f"[E2E] FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        cleanup_hf_cache(cache_dir)


if __name__ == "__main__":
    sys.exit(main())
