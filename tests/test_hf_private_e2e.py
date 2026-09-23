"""Opt-in live HF private E2E (mocked=False when credentials exist).

Never stores tokens. Skips with an explicit NOT RUN message when
HF_TOKEN / HF_PRIVATE_TEST_REPO are unavailable.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "e2e_hf_private_check.py"


class HfPrivateE2ETests(unittest.TestCase):
    def test_private_hf_e2e_or_explicit_not_run(self):
        """Live private HF when env is set; otherwise document NOT RUN (not a fake pass)."""
        env = os.environ.copy()
        # Do not inject tokens here — only pass through what the operator already set.
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        output = (result.stdout or "") + (result.stderr or "")
        has_creds = bool(env.get("HF_TOKEN") and env.get("HF_PRIVATE_TEST_REPO"))

        if not has_creds:
            self.assertEqual(result.returncode, 0, msg=output)
            self.assertIn("HF private repository E2E: NOT RUN", output)
            self.assertIn("no suitable private test repository/credential available", output)
            print(output)
            return

        # Live path: must actually pass; failures are real failures.
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("HF private repository E2E: PASS", output)
        self.assertIn("mocked=False", output)
        self.assertNotIn(env["HF_TOKEN"], output)


if __name__ == "__main__":
    unittest.main()
