import contextlib
import io
import json
import hashlib
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kaggle_dataset_manager import (
    CATEGORIES,
    DatasetFile,
    DownloadFailure,
    DownloadOutcome,
    ResolutionFailure,
    ResolutionOutcome,
    ResolvedArtifact,
    build_dataset_path,
    classify_resolved_artifacts,
    collect_dataset_edits,
    collect_input_queue,
    compare_states,
    configure_hf_cache,
    cleanup_hf_cache,
    download_input_queue,
    download_resolved_queue,
    format_hf_resolution_error,
    format_size,
    guess_category,
    manifest_payload,
    normalize_hf_file_path,
    parse_current_files,
    parse_air,
    parse_input,
    normalize_category,
    classify_civitai_type,
    classify_resource_type,
    parse_size,
    print_download_failure_summary,
    print_resolution_summary,
    publish,
    publish_staged_state,
    read_manifest,
    redact_secrets,
    resolve_queue_metadata,
    queue_contains_checkpoint,
    render_preview,
    retry,
    validate_air,
    validate_civitai_url,
    validate_hf_repo_id,
    _expected_sha256,
    download_with_civitai_cli,
    is_hf_input,
    parse_hf_input,
    validate_hf_input,
    resolve_hf_input,
    download_hf_file,
)
import kaggle_dataset_manager


class DatasetManagerTests(unittest.TestCase):
    def setUp(self):
        self.old = DatasetFile("checkpoints/old.safetensors", 100, "oldhash")
        self.same = DatasetFile("vae/same.safetensors", 200, "samehash")
        self.new = DatasetFile("loras/new.safetensors", 300, "newhash")

    def test_category_path(self):
        self.assertEqual(build_dataset_path("diffusion_models", "model.safetensors"), "diffusion_models/model.safetensors")
        self.assertEqual(build_dataset_path("text_encoders", "/tmp/clip.safetensors"), "text_encoders/clip.safetensors")
        with self.assertRaises(ValueError):
            build_dataset_path("invalid", "model.safetensors")

    def test_parse_air(self):
        parsed = parse_air("urn:air:krea2:lora:civitai:2761113@3139172+3019297")
        self.assertEqual(parsed["base_model"], "krea2")
        self.assertEqual(parsed["type"], "lora")
        self.assertEqual(parsed["model_id"], "2761113")
        self.assertEqual(parsed["version_id"], "3139172")
        self.assertEqual(parsed["file_id"], "3019297")

    def test_type_classification_and_aliases(self):
        self.assertEqual(normalize_category("diffusion"), "diffusion_models")
        self.assertEqual(normalize_category("text_encoder"), "text_encoders")
        self.assertEqual(classify_civitai_type("lora"), "loras")
        self.assertEqual(classify_civitai_type("textual_inversion"), "embeddings")
        self.assertEqual(classify_civitai_type("text_encoder"), "text_encoders")
        self.assertEqual(normalize_category("video"), "video_models")
        self.assertEqual(classify_civitai_type("checkpoint", input_fn=lambda _: "2"), "diffusion_models")

    def test_file_hash_metadata(self):
        self.assertEqual(_expected_sha256({"hashes": {"SHA256": "ABC123"}}), "abc123")
        self.assertIsNone(_expected_sha256({"hashes": {"CRC32": "x"}}))

    def test_civitai_cli_download_command_and_atomic_validation(self):
        payload = b"abc"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "loras" / "model.safetensors"

            def fake_run(command, env, **kwargs):
                output = Path(command[command.index("--out") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                self.assertEqual(env["CIVITAI_TOKEN"], "secret-token")
                return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

            with patch("kaggle_dataset_manager.ensure_civitai_cli", return_value="civitai") as ensure, patch(
                "kaggle_dataset_manager.subprocess.run", side_effect=fake_run
            ) as run:
                result = download_with_civitai_cli(
                    "3139172",
                    "3019297",
                    destination,
                    "secret-token",
                    expected_size=len(payload),
                    expected_sha256=expected_hash,
                )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name(destination.name + ".part").exists())
            ensure.assert_called_once_with()
            command = run.call_args.args[0]
            self.assertEqual(command[0:4], ["civitai", "download", "3139172", "--file"])
            self.assertIn("3019297", command)
            self.assertIn("--out", command)
            self.assertEqual(Path(command[command.index("--out") + 1]), destination)
            self.assertNotIn(".part", command[command.index("--out") + 1])

    def test_civitai_cli_error_redacts_token_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model.safetensors"

            def fake_run(*args, **kwargs):
                partial = destination.with_name(destination.name + ".part")
                partial.write_bytes(b"partial")
                return type("Result", (), {"returncode": 1, "stderr": "bad secret-token", "stdout": ""})()

            with patch("kaggle_dataset_manager.ensure_civitai_cli", return_value="civitai"), patch(
                "kaggle_dataset_manager.subprocess.run", side_effect=fake_run
            ):
                with self.assertRaises(RuntimeError) as context:
                    download_with_civitai_cli("1", "2", destination, "secret-token")

            self.assertIn("bad ***REDACTED***", str(context.exception))
            self.assertNotIn("secret-token", str(context.exception))
            self.assertTrue(destination.with_name(destination.name + ".part").exists())

    def test_cli_version_is_validated_and_reported(self):
        with patch("kaggle_dataset_manager.shutil.which", return_value="civitai"), patch(
            "kaggle_dataset_manager.subprocess.run"
        ) as run:
            run.return_value = type("Result", (), {"returncode": 0, "stdout": "civitai v0.1.104", "stderr": ""})()
            self.assertEqual(kaggle_dataset_manager.ensure_civitai_cli(), "civitai")

        self.assertEqual(run.call_args.args[0], ["civitai", "--version"])

    def test_shared_module_has_no_manager_cli_entrypoint(self):
        import kaggle_dataset_manager
        self.assertFalse(hasattr(kaggle_dataset_manager, "run_manager"))

    def test_manifest(self):
        item = DatasetFile(self.new.path, self.new.size, self.new.sha256, "air", "1", "2", "3", "urn:air:krea2:lora:civitai:1@2+3", "krea2")
        payload = manifest_payload("automamermaid/comfydocs", {item.path: item})
        self.assertEqual(payload["dataset"], "automamermaid/comfydocs")
        self.assertEqual(payload["files"][0]["sha256"], "newhash")
        self.assertEqual(payload["files"][0]["dataset_path"], self.new.path)
        self.assertEqual(payload["files"][0]["category"], "loras")
        self.assertEqual(payload["files"][0]["air"], "urn:air:krea2:lora:civitai:1@2+3")

    def test_add_remove_move_unchanged(self):
        current = {self.old.path: self.old, self.same.path: self.same}
        desired = {
            "checkpoints/renamed.safetensors": DatasetFile("checkpoints/renamed.safetensors", 100, "oldhash"),
            self.same.path: self.same,
            self.new.path: self.new,
        }
        changes = compare_states(current, desired)
        self.assertEqual(changes.moved, (("checkpoints/old.safetensors", "checkpoints/renamed.safetensors"),))
        self.assertEqual([item.path for item in changes.added], [self.new.path])
        self.assertEqual(changes.removed, ())
        self.assertEqual([item.path for item in changes.unchanged], [self.same.path])

    def test_sha_and_size(self):
        self.assertEqual(parse_size("11.94 GB"), int(11.94 * 1024 ** 3))
        self.assertEqual(parse_size("150 MB"), 150 * 1024 ** 2)
        self.assertEqual(format_size(1024 ** 3), "1.00 GB")
        current = parse_current_files([{"path": "vae/a.safetensors", "size": "300 MB"}])
        self.assertEqual(current["vae/a.safetensors"].size, 300 * 1024 ** 2)

    def test_current_state_size_with_separate_unit(self):
        current = parse_current_files([{"path": "checkpoints/a.safetensors", "size": "11.94 GB"}])
        self.assertEqual(current["checkpoints/a.safetensors"].size, int(11.94 * 1024 ** 3))

    def test_preview_contains_change_sections(self):
        current = {self.old.path: self.old}
        desired = {self.new.path: self.new}
        changes = compare_states(current, desired)
        preview = render_preview("automamermaid/comfydocs", current, desired, changes)
        for marker in ("DATASET CURRENT", "DESIRED", "CHANGES", "ADD", "REMOVE", "MOVE", "UNCHANGED"):
            self.assertIn(marker, preview)

    def test_publish_requires_explicit_call_and_writes_metadata(self):
        class FakePopen:
            def __init__(self, command, **kwargs):
                self.command = command
                self.stdout = iter(["version 7\n"])

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            with patch("kaggle_dataset_manager.subprocess.run") as run, patch(
                "kaggle_dataset_manager.subprocess.Popen", side_effect=lambda command, **kwargs: FakePopen(command)
            ) as popen:
                run.return_value.returncode = 1
                run.return_value.stdout = ""
                result = publish("automamermaid/comfydocs", Path(tmp), "test update")
                self.assertEqual(result, "version 7")
                metadata = json.loads((Path(tmp) / "dataset-metadata.json").read_text())
                self.assertEqual(metadata["id"], "automamermaid/comfydocs")
                command = popen.call_args.args[0]
                self.assertIn("version", command)
                self.assertIn("-p", command)
                self.assertNotIn("--delete-old-versions", command)

    def test_validate_air(self):
        parsed, error = validate_air("urn:air:krea2:lora:civitai:2761113@3139172+3019297")
        self.assertIsNone(error)
        self.assertEqual(parsed, {"model_id": 2761113, "version_id": 3139172})

        parsed, error = validate_air("urn:air:krea2:lora:civitai:0@3139172")
        self.assertIsNone(parsed)
        self.assertIn("> 0", error)

        parsed, error = validate_air("not-an-air")
        self.assertIsNone(parsed)
        self.assertIn("AIR inválido", error)

    def test_validate_civitai_url(self):
        self.assertIsNone(validate_civitai_url("https://civitai.com/models/2761113?modelVersionId=3139172"))
        self.assertIsNone(validate_civitai_url("https://civitai.com/api/download/models/3139172?fileId=3019297"))
        self.assertIsNotNone(validate_civitai_url("https://example.com/models/1"))
        self.assertIsNotNone(validate_civitai_url("https://civitai.com/user/foo"))
        self.assertIsNotNone(validate_civitai_url("not-a-url"))

    def test_retry_recovers_from_transient_errors(self):
        calls = []

        @retry(max_attempts=3, delay=2, backoff=2)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.URLError("boom")
            return "ok"

        with patch("kaggle_dataset_manager.time.sleep") as sleep:
            self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_retry_gives_up_after_max_attempts(self):
        calls = []

        @retry(max_attempts=3, delay=2, backoff=2)
        def always_fails():
            calls.append(1)
            raise ConnectionError("down")

        with patch("kaggle_dataset_manager.time.sleep"):
            with self.assertRaises(ConnectionError):
                always_fails()
        self.assertEqual(len(calls), 3)

    def test_retry_does_not_catch_permanent_errors(self):
        calls = []

        @retry(max_attempts=3, delay=2, backoff=2)
        def bad_input():
            calls.append(1)
            raise ValueError("permanent")

        with patch("kaggle_dataset_manager.time.sleep"):
            with self.assertRaises(ValueError):
                bad_input()
        self.assertEqual(len(calls), 1)

    def _fake_info(self, value):
        return {
            "model": {"type": "lora", "name": "m"},
            "version": {"id": 2, "name": "v", "baseModel": "krea2"},
            "file": {"id": 5, "name": "f.safetensors"},
            "model_id": "1",
            "air": {"air": value, "type": "lora", "base_model": "krea2"},
        }

    def test_queue_collects_all_entries_before_downloading(self):
        events = []
        entries = ["urn:air:krea2:lora:civitai:1@2", "https://civitai.com/models/3?modelVersionId=4"]
        inputs = iter([*entries, "done"])

        def fake_input(prompt=""):
            self.assertEqual(events, [], "resolução/download não pode ocorrer durante a coleta")
            return next(inputs)

        def fake_resolve(value, token, input_fn):
            events.append(f"resolve:{value}")
            return [self._fake_info(value)]

        def fake_download(info, category, staging_dir, token, source_url=None, air=None):
            events.append(f"download:{source_url}")
            return DatasetFile(f"loras/{len(events)}.safetensors", 1, f"hash{len(events)}")

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_resolve), patch.object(
            kaggle_dataset_manager, "download_civitai_file", side_effect=fake_download
        ), patch.object(kaggle_dataset_manager, "classify_civitai_type", return_value="loras"):
            queue = download_input_queue(Path("/tmp"), "token", input_fn=fake_input)

        self.assertEqual(len(queue), 2)
        self.assertEqual(
            events,
            [
                f"resolve:{entries[0]}",
                f"resolve:{entries[1]}",
                f"download:{entries[0]}",
                f"download:{entries[1]}",
            ],
        )

    def test_queue_skips_invalid_entries(self):
        inputs = iter(["not-an-air", "urn:air:x:lora:civitai:0@1", "done"])
        with patch.object(kaggle_dataset_manager, "resolve_civitai_input") as resolve:
            queue = download_input_queue(Path("/tmp"), "token", input_fn=lambda prompt="": next(inputs))
        self.assertEqual(queue, [])
        resolve.assert_not_called()

    def test_download_input_queue_aborts_when_resolution_fails(self):
        """1 falha de resolução aborta a fila ANTES de qualquer download (item 10)."""
        entries = ["urn:air:krea2:lora:civitai:1@2", "urn:air:krea2:lora:civitai:3@4"]
        inputs = iter([*entries, "done"])

        def fake_resolve(value, token, input_fn):
            if value == entries[0]:
                raise RuntimeError("falha permanente")
            return [self._fake_info(value)]

        downloads = []

        def fake_download(info, category, staging_dir, token, source_url=None, air=None):
            downloads.append(source_url)
            return DatasetFile("loras/ok.safetensors", 1, "hash")

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_resolve), patch.object(
            kaggle_dataset_manager, "download_civitai_file", side_effect=fake_download
        ), patch.object(kaggle_dataset_manager, "classify_civitai_type", return_value="loras"):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with self.assertRaises(RuntimeError) as ctx:
                    download_input_queue(Path("/tmp"), "token", input_fn=lambda prompt="": next(inputs))

        self.assertIn("nenhum download foi iniciado", str(ctx.exception))
        self.assertIn("INPUT RESOLUTION SUMMARY", stdout.getvalue())
        self.assertEqual(downloads, [])

    def test_download_input_queue_aborts_when_download_fails(self):
        """1 falha de download aborta SEM retornar fila parcial (publicação transacional)."""
        entries = ["urn:air:krea2:lora:civitai:1@2", "urn:air:krea2:lora:civitai:3@4"]
        inputs = iter([*entries, "done"])

        def fake_resolve(value, token, input_fn):
            return [self._fake_info(value)]

        downloads = []

        def fake_download(info, category, staging_dir, token, source_url=None, air=None):
            downloads.append(source_url)
            if source_url == entries[1]:
                raise RuntimeError("falha permanente de download")
            return DatasetFile("loras/ok.safetensors", 1, "hash")

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_resolve), patch.object(
            kaggle_dataset_manager, "download_civitai_file", side_effect=fake_download
        ), patch.object(kaggle_dataset_manager, "classify_civitai_type", return_value="loras"):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with self.assertRaises(RuntimeError) as ctx:
                    download_input_queue(Path("/tmp"), "token", input_fn=lambda prompt="": next(inputs))

        self.assertIn("nenhuma alteração no dataset foi publicada", str(ctx.exception))
        output = stdout.getvalue()
        self.assertIn("[DOWNLOAD FAILED]", output)
        self.assertIn("2 arquivo(s) resolvido(s)", output)
        self.assertIn("1 arquivo(s) baixado(s)", output)
        self.assertIn("1 arquivo(s) com falha", output)
        self.assertEqual(downloads, entries)  # ambos tentados; o lote ainda falha

    def test_classify_checkpoint_with_preset_destination_skips_prompt(self):
        def boom(prompt=""):
            raise AssertionError(f"input_fn não deve ser chamado: {prompt}")

        self.assertEqual(
            classify_civitai_type("checkpoint", input_fn=boom, checkpoint_destination="diffusion_models"),
            "diffusion_models",
        )
        self.assertEqual(
            classify_civitai_type("checkpoint", input_fn=boom, checkpoint_destination="checkpoints"),
            "checkpoints",
        )

    def test_queue_contains_checkpoint(self):
        self.assertTrue(queue_contains_checkpoint([{"resource_type": "lora"}, {"resource_type": "Checkpoint"}]))
        self.assertTrue(queue_contains_checkpoint([{"resource_type": "checkpoint"}]))
        self.assertFalse(queue_contains_checkpoint([{"resource_type": "lora"}]))
        self.assertFalse(queue_contains_checkpoint([]))

    def test_download_resolved_queue_with_preset_checkpoint_has_no_prompts(self):
        entry = {
            "value": "urn:air:sdxl:checkpoint:civitai:1@2",
            "index": 1,
            "total": 1,
            "info": {
                "model": {"type": "Checkpoint", "name": "m"},
                "version": {"id": 2, "name": "v", "baseModel": "sdxl"},
                "file": {"id": 5, "name": "f.safetensors"},
                "model_id": "1",
                "air": {"air": "urn:air:sdxl:checkpoint:civitai:1@2", "type": "checkpoint", "base_model": "sdxl"},
            },
            "resource_type": "checkpoint",
        }

        def fake_download(info, category, staging_dir, token, source_url=None, air=None):
            return DatasetFile(f"{category}/f.safetensors", 1, "hash")

        def boom(prompt=""):
            raise AssertionError(f"nenhum prompt pode ocorrer durante o download: {prompt}")

        with patch.object(kaggle_dataset_manager, "download_civitai_file", side_effect=fake_download):
            outcome = download_resolved_queue(
                [entry], Path("/tmp"), "token", input_fn=boom, checkpoint_destination="diffusion_models"
            )

        self.assertTrue(outcome.ok)
        self.assertEqual([item.path for item in outcome.items], ["diffusion_models/f.safetensors"])

    def test_collect_dataset_edits_collects_without_applying(self):
        inputs = iter(["s", "remove checkpoints/old.safetensors", "move a.safetensors b.safetensors", "bogus a b", "done"])
        with patch.object(
            kaggle_dataset_manager,
            "kaggle_files",
            return_value=[{"path": "checkpoints/old.safetensors", "size": "100 B"}],
        ):
            edits = collect_dataset_edits("owner/dataset", input_fn=lambda prompt="": next(inputs))
        self.assertEqual(
            edits,
            [("remove", "checkpoints/old.safetensors"), ("move", "a.safetensors", "b.safetensors")],
        )

    def test_collect_dataset_edits_declined_returns_empty(self):
        with patch.object(
            kaggle_dataset_manager,
            "kaggle_files",
            return_value=[{"path": "checkpoints/old.safetensors", "size": "100 B"}],
        ):
            edits = collect_dataset_edits("owner/dataset", input_fn=lambda prompt="": "n")
        self.assertEqual(edits, [])

    def test_publish_applies_pending_edits_without_prompting(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            new_file = staging / "loras" / "new.safetensors"
            new_file.parent.mkdir(parents=True)
            new_file.write_bytes(b"abc")
            prompts = []

            def fake_input(prompt=""):
                prompts.append(prompt)
                if "modificar" in prompt:
                    raise AssertionError("pergunta de edição não deve ocorrer em publish_staged_state")
                if "Publicar" in prompt:
                    return "n"
                return ""

            edits = [
                ("remove", "checkpoints/old.safetensors"),
                ("move", "loras/new.safetensors", "loras/renamed.safetensors"),
            ]
            with patch.object(
                kaggle_dataset_manager,
                "kaggle_files",
                return_value=[{"path": "checkpoints/old.safetensors", "size": "100 B"}],
            ), patch.object(kaggle_dataset_manager, "publish", return_value="ok") as pub:
                result = publish_staged_state("owner/dataset", staging, input_fn=fake_input, pending_edits=edits)

            self.assertIsNone(result)
            pub.assert_not_called()
            self.assertFalse((staging / "loras" / "new.safetensors").exists())
            self.assertTrue((staging / "loras" / "renamed.safetensors").exists())
            manifest = json.loads((staging / "dataset-manifest.json").read_text())
            paths = [entry["path"] for entry in manifest["files"]]
            self.assertIn("loras/renamed.safetensors", paths)
            self.assertNotIn("checkpoints/old.safetensors", paths)
            self.assertFalse(any("modificar" in prompt for prompt in prompts))

    # =========================================================================
    # TESTES HUGGING FACE - DETECÇÃO E PARSING
    # =========================================================================

    def test_is_hf_input_detects_hf_prefix(self):
        """Detecta prefixo hf:"""
        self.assertTrue(is_hf_input("hf:org/repo"))
        self.assertTrue(is_hf_input("hf:org/repo/path/to/file.safetensors"))
        self.assertTrue(is_hf_input("HF:org/repo"))  # Case insensitive

    def test_is_hf_input_detects_hf_url(self):
        """Detecta URL huggingface.co"""
        self.assertTrue(is_hf_input("https://huggingface.co/org/repo/resolve/main/file.safetensors"))
        self.assertTrue(is_hf_input("https://huggingface.co/org/repo/blob/main/file.safetensors"))

    def test_is_hf_input_rejects_civitai_and_other(self):
        """Rejeita Civitai e entradas não-HF"""
        self.assertFalse(is_hf_input("urn:air:krea2:lora:civitai:1@2"))
        self.assertFalse(is_hf_input("https://civitai.com/models/123"))
        self.assertFalse(is_hf_input("not-an-air"))

    def test_parse_hf_input_hf_prefix_with_file(self):
        """Parseia hf:org/repo/path/to/arquivo.safetensors"""
        repo_id, file_path, revision = parse_hf_input("hf:myorg/myrepo/path/to/model.safetensors")
        self.assertEqual(repo_id, "myorg/myrepo")
        self.assertEqual(file_path, "path/to/model.safetensors")
        self.assertEqual(revision, "main")

    def test_parse_hf_input_hf_prefix_without_file(self):
        """Parseia hf:org/repo (sem arquivo)"""
        repo_id, file_path, revision = parse_hf_input("hf:myorg/myrepo")
        self.assertEqual(repo_id, "myorg/myrepo")
        self.assertIsNone(file_path)
        self.assertEqual(revision, "main")

    def test_parse_hf_input_url_resolve(self):
        """Parseia URL /resolve/main/arquivo"""
        repo_id, file_path, revision = parse_hf_input(
            "https://huggingface.co/org/repo/resolve/main/model.safetensors"
        )
        self.assertEqual(repo_id, "org/repo")
        self.assertEqual(file_path, "model.safetensors")
        self.assertEqual(revision, "main")

    def test_parse_hf_input_url_blob(self):
        """Parseia URL /blob/main/arquivo"""
        repo_id, file_path, revision = parse_hf_input(
            "https://huggingface.co/org/repo/blob/dev/path/to/file.safetensors"
        )
        self.assertEqual(repo_id, "org/repo")
        self.assertEqual(file_path, "path/to/file.safetensors")
        self.assertEqual(revision, "dev")

    def test_validate_hf_input_accepts_valid(self):
        """Valida entrada HF válida"""
        self.assertIsNone(validate_hf_input("hf:org/repo"))
        self.assertIsNone(validate_hf_input("https://huggingface.co/org/repo/resolve/main/f.safetensors"))

    def test_validate_hf_input_rejects_invalid(self):
        """Rejeita entrada HF inválida"""
        error = validate_hf_input("not-hf")
        self.assertIsNotNone(error)
        self.assertIn("não é HF", error)

    def test_validate_hf_input_rejects_malformed_repo(self):
        """Rejeita hf: sem org/repo"""
        error = validate_hf_input("hf:single")
        self.assertIsNotNone(error)
        self.assertIn("hf:org/repo", error)

    def test_parse_hf_input_accepts_double_slash_with_file(self):
        """Tolera hf://org/repo/arquivo (barra dupla digitada por engano)"""
        repo_id, file_path, revision = parse_hf_input(
            "hf://myorg/myrepo/path/to/model.safetensors"
        )
        self.assertEqual(repo_id, "myorg/myrepo")
        self.assertEqual(file_path, "path/to/model.safetensors")
        self.assertEqual(revision, "main")

    def test_parse_hf_input_accepts_double_slash_repo_only(self):
        """Tolera hf://org/repo (repo inteiro)"""
        repo_id, file_path, revision = parse_hf_input("hf://myorg/myrepo")
        self.assertEqual(repo_id, "myorg/myrepo")
        self.assertIsNone(file_path)
        self.assertEqual(revision, "main")

    def test_validate_hf_input_accepts_double_slash(self):
        """hf://org/repo/arquivo passa na validação da coleta"""
        self.assertIsNone(validate_hf_input("hf://myorg/myrepo/model.safetensors"))
        self.assertIsNone(validate_hf_input("hf://myorg/myrepo"))

    def test_validate_hf_input_rejects_empty_segments(self):
        """Rejeita hf:, hf:/ e hf:// sem org/repo na fase de coleta"""
        for value in ("hf:", "hf:/", "hf://"):
            with self.subTest(value=value):
                error = validate_hf_input(value)
                self.assertIsNotNone(error)
                self.assertIn("hf:org/repo", error)

    def test_collect_input_queue_accepts_double_slash_hf(self):
        """Entrada hf://... digitada pelo usuário é aceita na coleta"""
        entries = ["hf://myorg/myrepo/model.safetensors", "done"]
        inputs = iter(entries)
        queue = collect_input_queue(input_fn=lambda prompt="": next(inputs))
        self.assertEqual(queue, ["hf://myorg/myrepo/model.safetensors"])

    def test_collect_input_queue_retries_malformed_hf(self):
        """Entrada HF malformada é rejeitada e a coleta continua"""
        entries = ["hf:single", "hf:org/repo/arquivo.safetensors", "done"]
        inputs = iter(entries)
        queue = collect_input_queue(input_fn=lambda prompt="": next(inputs))
        self.assertEqual(queue, ["hf:org/repo/arquivo.safetensors"])

    # =========================================================================
    # TESTES HUGGING FACE - CLASSIFICAÇÃO E COLETA
    # =========================================================================

    def test_classify_resource_type_auto_detects_known_types(self):
        """Mapeia tipos conhecidos sem prompt"""
        self.assertEqual(classify_resource_type("lora"), "loras")
        self.assertEqual(classify_resource_type("vae"), "vae")
        self.assertEqual(classify_resource_type("text_encoder"), "text_encoders")

    def test_classify_resource_type_always_asks_for_empty(self):
        """SEMPRE pergunta quando tipo é vazio (HF genérico)"""
        def fake_input(prompt=""):
            if "Categoria" in prompt:
                return "loras"
            return ""
        result = classify_resource_type("", input_fn=fake_input)
        self.assertEqual(result, "loras")

    def test_collect_input_queue_accepts_hf_and_civitai(self):
        """Coleta aceita Civitai e HF na mesma sessão"""
        entries = ["urn:air:krea2:lora:civitai:1@2", "hf:org/repo/arquivo.safetensors", "done"]
        inputs = iter(entries)
        queue = collect_input_queue(input_fn=lambda prompt="": next(inputs))
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue[0], "urn:air:krea2:lora:civitai:1@2")
        self.assertEqual(queue[1], "hf:org/repo/arquivo.safetensors")

    def test_resolve_hf_input_with_file_asks_category_and_base_model(self):
        """resolve_hf_input SEMPRE pergunta categoria e base_model"""
        prompts_captured = []
        def fake_input(prompt=""):
            prompts_captured.append(prompt)
            if "Categoria" in prompt:
                return "loras"
            if "Base model" in prompt:
                return "sdxl"
            return ""

        with patch.object(kaggle_dataset_manager, "_hf_file_size", return_value=1000), \
             patch.object(kaggle_dataset_manager, "_hf_list_repo_files", return_value=[]):
            result = resolve_hf_input("hf:org/repo/model.safetensors", input_fn=fake_input)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["repo_id"], "org/repo")
        self.assertEqual(result[0]["file_path"], "model.safetensors")
        self.assertEqual(result[0]["category"], "loras")
        self.assertEqual(result[0]["base_model"], "sdxl")
        self.assertIsNone(result[0]["air"])
        self.assertEqual(result[0]["source"], "hf")
        self.assertTrue(any("Categoria" in p for p in prompts_captured))
        self.assertTrue(any("Base model" in p for p in prompts_captured))

    # =========================================================================
    # TESTES HUGGING FACE - RESOLUÇÃO E ROTEAMENTO
    # =========================================================================

    def test_resolve_queue_metadata_routes_civitai_to_civitai(self):
        """resolve_queue_metadata roteia Civitai para resolve_civitai_input"""
        pending = ["urn:air:krea2:lora:civitai:1@2"]

        def fake_resolve_civitai(value, token, input_fn):
            return [{"model": {"type": "lora"}, "version": {"id": 2}, "file": {}, "air": {"type": "lora"}}]

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_resolve_civitai):
            outcome = resolve_queue_metadata(pending, "token", input_fn=lambda p="": "")

        self.assertEqual(len(outcome.artifacts), 1)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(outcome.artifacts[0].provider, "civitai")
        self.assertEqual(outcome.artifacts[0].source, "civitai")
        self.assertEqual(outcome.artifacts[0].resource_type, "lora")

    def test_resolve_queue_metadata_routes_hf_to_hf(self):
        """resolve_queue_metadata roteia HF para resolve_hf_input"""
        pending = ["hf:org/repo/file.safetensors"]

        def fake_resolve_hf(value, hf_token, input_fn):
            input_fn("Categoria: ")
            input_fn("Base model: ")
            return [{
                "repo_id": "org/repo",
                "file_path": "file.safetensors",
                "category": "loras",
                "base_model": "sdxl",
                "source": "hf",
            }]

        with patch.object(kaggle_dataset_manager, "resolve_hf_input", side_effect=fake_resolve_hf):
            outcome = resolve_queue_metadata(pending, "token", input_fn=lambda p="": "", hf_token="hf_token")

        self.assertEqual(len(outcome.artifacts), 1)
        self.assertEqual(outcome.failures, [])
        self.assertEqual(outcome.artifacts[0].source, "hf")
        self.assertEqual(outcome.artifacts[0].provider, "huggingface")
        self.assertEqual(outcome.artifacts[0].category, "loras")
        self.assertEqual(outcome.artifacts[0].base_model, "sdxl")

    def test_resolve_queue_metadata_hf_failure_does_not_stop_others(self):
        """Item HF inexistente na Fase 1 não interrompe os demais itens"""
        pending = [
            "urn:air:krea2:lora:civitai:1@2",
            "hf:org/missing/file.safetensors",
            "hf:org/repo/outro.safetensors",
        ]
        civitai_info = [{"model": {"type": "lora"}, "version": {"id": 2}, "file": {}, "air": {"type": "lora"}}]

        def fake_resolve_hf(value, hf_token=None, input_fn=input):
            if "missing" in value:
                raise RuntimeError("Repositório não encontrado: org/missing")
            return [{"repo_id": "org/repo", "file_path": "outro.safetensors", "revision": "main", "source": "hf"}]

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", return_value=civitai_info), \
             patch.object(kaggle_dataset_manager, "resolve_hf_input", side_effect=fake_resolve_hf):
            outcome = resolve_queue_metadata(pending, "token", input_fn=lambda p="": "", hf_token="tok")

        self.assertEqual([artifact.source for artifact in outcome.artifacts], ["civitai", "hf"])
        self.assertEqual(outcome.artifacts[1].file_path, "outro.safetensors")
        self.assertEqual(len(outcome.failures), 1)
        failure = outcome.failures[0]
        self.assertEqual(failure.original_input, "hf:org/missing/file.safetensors")
        self.assertEqual(failure.provider, "huggingface")
        self.assertIn("Repositório não encontrado", failure.technical)
        # Mensagem contextual do item 14 (nunca só o erro técnico)
        self.assertIn("Falha ao obter metadata do Hugging Face", failure.reason)
        self.assertIn("nenhum download foi iniciado", failure.reason)

    # =========================================================================
    # TESTES HUGGING FACE - FASES (SEM DOWNLOAD NA FASE 1) E RESILIÊNCIA
    # =========================================================================

    def test_queue_two_phases_no_download_during_collection(self):
        """Fase 1 (coleta/resolução) não baixa nada, mesmo misturando Civitai e HF"""
        pending = ["urn:air:krea2:lora:civitai:1@2", "hf:org/repo/file.safetensors"]
        civitai_info = [{"model": {"type": "lora"}, "version": {"id": 2}, "file": {}, "air": {"type": "lora"}}]

        def fake_resolve_hf(value, hf_token=None, input_fn=input):
            input_fn("Categoria ")
            input_fn("Base model ")
            return [{"repo_id": "org/repo", "file_path": "file.safetensors", "revision": "main", "source": "hf"}]

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", return_value=civitai_info), \
             patch.object(kaggle_dataset_manager, "resolve_hf_input", side_effect=fake_resolve_hf), \
             patch.object(kaggle_dataset_manager, "classify_civitai_type") as classify_mock, \
             patch.object(kaggle_dataset_manager, "download_civitai_file") as civitai_download, \
             patch.object(kaggle_dataset_manager, "download_hf_file") as hf_download:
            outcome = resolve_queue_metadata(pending, "token", input_fn=lambda p="": "", hf_token="tok")

            civitai_download.assert_not_called()
            hf_download.assert_not_called()
            classify_mock.assert_not_called()

        self.assertEqual([artifact.source for artifact in outcome.artifacts], ["civitai", "hf"])



    def test_download_resolved_queue_routes_hf_without_prompting(self):
        """Fase 2 roteia origem HF sem perguntar nada (categoria já veio da Fase 1)"""
        resolved = [
            {
                "value": "hf:org/repo/file.safetensors",
                "index": 1,
                "total": 1,
                "info": {
                    "repo_id": "org/repo",
                    "file_path": "file.safetensors",
                    "revision": "dev",
                    "category": "loras",
                    "base_model": "sdxl",
                    "source_url": "https://huggingface.co/org/repo/resolve/dev/file.safetensors",
                },
                "resource_type": None,
                "source": "hf",
            }
        ]
        item = DatasetFile(
            "loras/file.safetensors",
            10,
            "hash",
            "https://huggingface.co/org/repo/resolve/dev/file.safetensors",
            None,
            None,
            None,
            None,
            "sdxl",
            "org/repo",
            "dev",
            "file.safetensors",
        )

        def explode(prompt=""):
            raise AssertionError(f"prompt inesperado na Fase 2: {prompt}")

        with patch.object(kaggle_dataset_manager, "download_hf_file", return_value=item) as hf_download, \
             patch.object(kaggle_dataset_manager, "classify_civitai_type") as classify_mock:
            outcome = download_resolved_queue(resolved, Path("staging"), "token", input_fn=explode, hf_token="tok")

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.items, [item])
        classify_mock.assert_not_called()
        kwargs = hf_download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "org/repo")
        self.assertEqual(kwargs["file_path"], "file.safetensors")
        self.assertEqual(kwargs["category"], "loras")
        self.assertEqual(kwargs["revision"], "dev")
        self.assertEqual(kwargs["base_model"], "sdxl")

    def test_download_resolved_queue_continues_attempts_but_marks_batch_failed(self):
        """Falha no meio: demais itens ainda são tentados; outcome.ok=False (transacional)."""
        civitai_entry = {
            "value": "urn:air:krea2:lora:civitai:1@2",
            "index": 1,
            "total": 3,
            "info": {"model": {"type": "lora", "name": "m"}, "version": {"id": 2}, "file": {}, "air": {"type": "lora"}},
            "resource_type": "lora",
            "source": "civitai",
        }

        def hf_entry(repo_id, index):
            return {
                "value": f"hf:{repo_id}/f.safetensors",
                "index": index,
                "total": 3,
                "info": {
                    "repo_id": repo_id,
                    "file_path": "f.safetensors",
                    "revision": "main",
                    "category": "loras",
                    "base_model": "sdxl",
                    "source_url": "https://huggingface.co/x",
                },
                "resource_type": None,
                "source": "hf",
            }

        civitai_item = DatasetFile("loras/civ.safetensors", 1, "h1")
        hf_item = DatasetFile("loras/f.safetensors", 2, "h2", None, None, None, None, None, "sdxl", "org/repo", "main", "f.safetensors")
        calls = []

        def fake_download_hf_file(repo_id, file_path, category, staging_dir, hf_token=None, revision="main", source_url=None, base_model=None):
            calls.append(repo_id)
            if repo_id == "org/missing":
                raise RuntimeError("Repositório não encontrado: org/missing")
            return hf_item

        with patch.object(kaggle_dataset_manager, "download_civitai_file", return_value=civitai_item), \
             patch.object(kaggle_dataset_manager, "download_hf_file", side_effect=fake_download_hf_file), \
             patch.object(kaggle_dataset_manager, "classify_civitai_type", return_value="loras"):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                outcome = download_resolved_queue(
                    [civitai_entry, hf_entry("org/missing", 2), hf_entry("org/repo", 3)],
                    Path("staging"),
                    "token",
                    input_fn=lambda p="": "",
                    hf_token="tok",
                )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.items, [civitai_item, hf_item])
        self.assertEqual(len(outcome.failures), 1)
        self.assertEqual(outcome.resolved_count, 3)
        self.assertEqual(calls, ["org/missing", "org/repo"])
        output = stdout.getvalue()
        self.assertIn("[ERROR] Download falhou para [2/3]", output)
        self.assertIn("org/missing", output)
        self.assertIn("Repositório não encontrado", output)

    def test_print_download_failure_summary_format(self):
        outcome = DownloadOutcome(
            items=[DatasetFile("loras/a.safetensors", 1, "h")],
            failures=[DownloadFailure("hf:org/missing/f.safetensors", 2, 7, reason="boom")],
            resolved_count=7,
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_download_failure_summary(outcome)
        text = stdout.getvalue()
        self.assertIn("[DOWNLOAD FAILED]", text)
        self.assertIn("7 arquivo(s) resolvido(s)", text)
        self.assertIn("1 arquivo(s) baixado(s)", text)
        self.assertIn("1 arquivo(s) com falha", text)
        self.assertIn("Nenhuma alteração no dataset foi publicada.", text)

    # =========================================================================
    # TESTES HUGGING FACE - DOWNLOAD E MANIFEST
    # =========================================================================

    def test_download_hf_file_builds_datasetfile_with_hf_fields(self):
        """download_hf_file devolve DatasetFile com campos HF preenchidos e Civitai vazios"""
        source_url = "https://huggingface.co/org/repo/resolve/dev/nested/file.safetensors"

        def fake_hf_hub_download(repo_id, filename, revision="main", token=None, local_dir=None):
            target = Path(local_dir) / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"hf-bytes")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            with patch.object(kaggle_dataset_manager, "_hf_hub_download", side_effect=fake_hf_hub_download):
                item = download_hf_file(
                    "org/repo",
                    "nested/file.safetensors",
                    "loras",
                    staging,
                    hf_token="tok",
                    revision="dev",
                    source_url=source_url,
                    base_model="sdxl",
                )
            self.assertTrue((staging / "loras" / "file.safetensors").exists())

        self.assertEqual(item.path, "loras/file.safetensors")
        self.assertEqual(item.size, 8)
        self.assertEqual(item.sha256, hashlib.sha256(b"hf-bytes").hexdigest())
        self.assertEqual(item.source, source_url)
        self.assertEqual(item.base_model, "sdxl")
        self.assertEqual(item.hf_repo_id, "org/repo")
        self.assertEqual(item.hf_revision, "dev")
        self.assertEqual(item.hf_file_path, "nested/file.safetensors")
        self.assertIsNone(item.air)
        self.assertIsNone(item.civitai_model_id)
        self.assertIsNone(item.civitai_version_id)
        self.assertIsNone(item.civitai_file_id)

    def test_manifest_includes_hf_fields_and_roundtrips(self):
        """Manifest expõe os campos HF e relê sem perda"""
        hf_item = DatasetFile(
            "loras/hf.safetensors",
            10,
            "hash",
            "https://huggingface.co/org/repo/resolve/main/hf.safetensors",
            None,
            None,
            None,
            None,
            "sdxl",
            "org/repo",
            "main",
            "hf.safetensors",
        )
        payload = manifest_payload("owner/dataset", {"loras/hf.safetensors": hf_item})
        entry = payload["files"][0]
        self.assertEqual(entry["hf_repo_id"], "org/repo")
        self.assertEqual(entry["hf_revision"], "main")
        self.assertEqual(entry["hf_file_path"], "hf.safetensors")
        self.assertIsNone(entry["air"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset-manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            parsed = read_manifest(path)

        self.assertEqual(parsed["loras/hf.safetensors"].hf_repo_id, "org/repo")
        self.assertEqual(parsed["loras/hf.safetensors"].hf_file_path, "hf.safetensors")

    def test_manifest_hf_entry_has_required_fields_and_no_secrets(self):
        """Manifest pós-download: provider/repo/path/revision/filename/size/category/base_model/hash; sem tokens."""
        token = "hf_secret_token_should_never_appear"
        hf_item = DatasetFile(
            "loras/hf.safetensors",
            10,
            "abc123hash",
            "https://huggingface.co/org/repo/resolve/main/nested/hf.safetensors",
            None,
            None,
            None,
            None,
            "sdxl",
            "org/repo",
            "main",
            "nested/hf.safetensors",
        )
        payload = manifest_payload("owner/dataset", {"loras/hf.safetensors": hf_item})
        blob = json.dumps(payload)
        entry = payload["files"][0]
        self.assertEqual(entry["filename"], "hf.safetensors")
        self.assertEqual(entry["category"], "loras")
        self.assertEqual(entry["size"], 10)
        self.assertEqual(entry["sha256"], "abc123hash")
        self.assertEqual(entry["base_model"], "sdxl")
        self.assertEqual(entry["hf_repo_id"], "org/repo")
        self.assertEqual(entry["hf_revision"], "main")
        self.assertEqual(entry["hf_file_path"], "nested/hf.safetensors")
        self.assertEqual(entry["source"], "https://huggingface.co/org/repo/resolve/main/nested/hf.safetensors")
        self.assertNotIn(token, blob)
        self.assertNotIn("HF_TOKEN", blob)
        self.assertNotIn("token=", blob.lower())
        self.assertNotIn("authorization", blob.lower())

    def test_read_manifest_accepts_legacy_entries_without_hf_fields(self):
        """Manifest antigo (sem campos HF) continua válido"""
        legacy = {
            "dataset": "owner/dataset",
            "files": [
                {
                    "dataset_path": "loras/old.safetensors",
                    "filename": "old.safetensors",
                    "category": "loras",
                    "path": "loras/old.safetensors",
                    "size": 5,
                    "sha256": "abc",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset-manifest.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            parsed = read_manifest(path)

        item = parsed["loras/old.safetensors"]
        self.assertEqual(item.size, 5)
        self.assertIsNone(item.hf_repo_id)
        self.assertIsNone(item.hf_revision)
        self.assertIsNone(item.hf_file_path)

    # =========================================================================
    # TESTES HUGGING FACE - TRADUÇÃO DE ERROS DA LIB huggingface_hub
    # =========================================================================

    @staticmethod
    def _fake_hf_module(hf_api_cls):
        import types

        fake = types.ModuleType("huggingface_hub")
        fake.HfApi = hf_api_cls
        fake.hf_hub_download = lambda **kwargs: kwargs
        fake.get_hf_file_metadata = lambda *args, **kwargs: None
        return fake

    def test_hf_file_size_reads_sibling_size(self):
        """_hf_file_size lê RepoSibling.rfilename/size com files_metadata=True (1.31.0)"""
        calls = {}

        class Sibling:
            def __init__(self, rfilename, size):
                self.rfilename = rfilename  # attr REAL do RepoSibling (não 'filename')
                self.size = size

        class Info:
            siblings = [Sibling("a.safetensors", 111), Sibling("alvo.safetensors", 222)]

        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, files_metadata=False, **kwargs):
                calls["files_metadata"] = files_metadata
                return Info()

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            size = kaggle_dataset_manager._hf_file_size("org/repo", "alvo.safetensors", token="tok")
        self.assertEqual(size, 222)
        self.assertTrue(calls["files_metadata"])

    def test_hf_file_size_translates_gated_repo_error(self):
        """GatedRepoError vira RuntimeError com mensagem sobre HF_TOKEN"""
        class GatedRepoError(Exception):
            pass

        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, **kwargs):
                raise GatedRepoError("gated")

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_file_size("org/repo", "f.safetensors")
        self.assertIn("gated", str(ctx.exception))
        self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_hf_hub_download_translates_repository_not_found(self):
        """RepositoryNotFoundError no download vira RuntimeError claro"""
        class RepositoryNotFoundError(Exception):
            pass

        def fake_hf_hub_download(**kwargs):
            raise RepositoryNotFoundError("404")

        fake = self._fake_hf_module(object)
        fake.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_hub_download("org/missing", "f.safetensors")
        self.assertIn("Repositório não encontrado", str(ctx.exception))

    def test_hf_hub_download_translates_entry_not_found(self):
        """EntryNotFoundError no download vira RuntimeError claro"""
        class EntryNotFoundError(Exception):
            pass

        def fake_hf_hub_download(**kwargs):
            raise EntryNotFoundError("missing entry")

        fake = self._fake_hf_module(object)
        fake.hf_hub_download = fake_hf_hub_download
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_hub_download("org/repo", "arquivo.safetensors")
        self.assertIn("Arquivo não encontrado", str(ctx.exception))


    # =========================================================================
    # TESTES HUGGING FACE - FORMATOS, VALIDAÇÃO E PARSER (itens 4/9/16)
    # =========================================================================

    def test_parse_hf_input_supported_formats(self):
        """Todos os formatos suportados normalizam para (repo_id, file_path, revision)"""
        cases = {
            "hf:owner/repo": ("owner/repo", None, "main"),
            "hf:owner/repo/file.safetensors": ("owner/repo", "file.safetensors", "main"),
            "hf://owner/repo/path/to/file.safetensors": ("owner/repo", "path/to/file.safetensors", "main"),
            "https://huggingface.co/owner/repo/resolve/main/file.safetensors": (
                "owner/repo", "file.safetensors", "main",
            ),
            "https://huggingface.co/owner/repo/blob/v2/file.safetensors": (
                "owner/repo", "file.safetensors", "v2",
            ),
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(parse_hf_input(value), expected)

    def test_parse_hf_input_rejects_invalid_forms(self):
        """Casos inválidos (item 16): estrutura, traversal cru e percent-encoded"""
        invalid = (
            "hf:",
            "hf://",
            "hf://owner",
            "hf://owner/",
            "hf://owner/repo/../../secret",
            "hf://owner/repo/%2e%2e/secret.safetensors",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_hf_input(value)

    def test_normalize_hf_file_path_validation(self):
        """Normaliza separadores e rejeita traversal/absoluto/extensão/URL"""
        self.assertEqual(normalize_hf_file_path("a//b/./c.safetensors"), "a/b/c.safetensors")
        self.assertEqual(normalize_hf_file_path("dir\\model.safetensors"), "dir/model.safetensors")
        invalid = (
            "",
            "../x.safetensors",
            "a/../../x.safetensors",
            "%2e%2e/x.safetensors",
            "/abs/x.safetensors",
            "C:\\abs\\x.safetensors",
            "x.txt",
            "sem_extensao",
            "https://evil.example/x.safetensors",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_hf_file_path(value)

    def test_validate_hf_repo_id_rejects_malformed(self):
        """repo_id deve ser owner/repo com charset do Hub"""
        self.assertIsNone(validate_hf_repo_id("owner/repo"))
        self.assertIsNone(validate_hf_repo_id("my-org/my_repo.1"))
        for bad in ("owner", "a/b/c", "own er/repo", "-bad/repo", "bad-/repo", "../.."):
            with self.subTest(bad=bad):
                self.assertIsNotNone(validate_hf_repo_id(bad))

    def test_parse_input_routes_providers_and_rejects_done(self):
        """Parser roteia providers; 'done' nunca vira artefato (item 11)"""
        hf = parse_input("hf:org/repo/f.safetensors", 1)
        self.assertEqual(hf.provider, "huggingface")
        self.assertEqual(hf.repo_id, "org/repo")
        self.assertEqual(hf.file_path, "f.safetensors")
        self.assertEqual(hf.state, "PARSED")
        air = parse_input("urn:air:krea2:lora:civitai:1@2", 2)
        self.assertEqual(air.provider, "civitai")
        url = parse_input("https://civitai.com/models/3", 3)
        self.assertEqual(url.provider, "civitai")
        for bad in ("done", "", "not-an-air", "urn:air:x:lora:civitai:0@1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_input(bad)

    # =========================================================================
    # TESTES HUGGING FACE - METADATA COM API MOCKADA (itens 6/16)
    # =========================================================================

    def test_hf_list_repo_files_uses_dedicated_api(self):
        """_hf_list_repo_files usa HfApi.list_repo_files (list[str]) e repassa o token"""
        calls = {}

        class HfApi:
            def list_repo_files(self, repo_id=None, revision=None, token=None):
                calls.update(repo_id=repo_id, revision=revision, token=token)
                return ["a.safetensors", "README.md"]

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            files = kaggle_dataset_manager._hf_list_repo_files("org/repo", revision="dev", token="hf_secret")
        self.assertEqual(files, ["a.safetensors", "README.md"])
        self.assertEqual(calls["repo_id"], "org/repo")
        self.assertEqual(calls["revision"], "dev")
        self.assertEqual(calls["token"], "hf_secret")  # repo privado funciona com HF_TOKEN

    def test_hf_list_repo_files_translates_gated_error(self):
        """GatedRepoError na listagem vira mensagem clara sobre HF_TOKEN"""
        class GatedRepoError(Exception):
            pass

        class HfApi:
            def list_repo_files(self, repo_id=None, revision=None, token=None):
                raise GatedRepoError("403")

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_list_repo_files("org/gated")
        self.assertIn("gated", str(ctx.exception))
        self.assertIn("HF_TOKEN", str(ctx.exception))


    def test_hf_file_size_file_not_found(self):
        """Arquivo ausente do repo -> FileNotFoundError com nome do arquivo"""
        class Info:
            siblings = []

        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, files_metadata=False, **kwargs):
                return Info()

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(FileNotFoundError) as ctx:
                kaggle_dataset_manager._hf_file_size("org/repo", "ausente.safetensors")
        self.assertIn("ausente.safetensors", str(ctx.exception))

    def test_hf_file_size_falls_back_to_get_hf_file_metadata(self):
        """Sem size nos siblings -> fallback get_hf_file_metadata (sem round-trips extras no caso normal)"""
        import types

        class Sibling:
            rfilename = "f.safetensors"
            size = None

        class Info:
            siblings = [Sibling()]

        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, files_metadata=False, **kwargs):
                return Info()

            def hf_hub_url(self, repo_id, filename, revision=None):
                return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

        fake = types.ModuleType("huggingface_hub")
        fake.HfApi = HfApi
        fake.hf_hub_download = lambda **kwargs: kwargs
        fake.get_hf_file_metadata = lambda url, token=None: types.SimpleNamespace(size=987)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            size = kaggle_dataset_manager._hf_file_size("org/repo", "f.safetensors", token="tok")
        self.assertEqual(size, 987)

    def test_hf_file_size_redacts_token_on_network_error(self):
        """Erro de rede não pode vazar o HF_TOKEN na mensagem (item 7)"""
        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, files_metadata=False, **kwargs):
                raise ConnectionError(f"proxy rejeitou para {token}")

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_file_size("org/repo", "f.safetensors", token="hf_super_secret")
        message = str(ctx.exception)
        self.assertNotIn("hf_super_secret", message)
        self.assertIn("***REDACTED***", message)

    def test_hf_file_size_unexpected_api_response(self):
        """Resposta inesperada da API vira RuntimeError contextual (não AttributeError cru)"""
        class HfApi:
            def model_info(self, repo_id, revision=None, token=None, files_metadata=False, **kwargs):
                return object()  # sem .siblings

        fake = self._fake_hf_module(HfApi)
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaises(RuntimeError) as ctx:
                kaggle_dataset_manager._hf_file_size("org/repo", "f.safetensors")
        message = str(ctx.exception)
        self.assertIn("Erro ao obter tamanho do arquivo HF", message)
        # O detalhe técnico fica na mensagem, mas SEMPRE embrulhado em contexto
        self.assertNotIsInstance(ctx.exception, AttributeError)

    def test_resolve_hf_repo_only_lists_filters_and_asks(self):
        """Caso A: repo sem arquivo -> lista filtrada, re-pergunta até válida, categoria e base_model"""
        prompts = []
        answers = iter(["README.md", "2"])  # 1ª inválida (fora da lista), 2ª = nested/b.safetensors

        def fake_input(prompt=""):
            prompts.append(prompt)
            if "Escolha" in prompt:
                return next(answers)
            if "Categoria" in prompt:
                return "loras"
            if "Base model" in prompt:
                return "sdxl"
            return ""

        files = [".gitattributes", "README.md", "a.safetensors", "nested/b.safetensors"]
        with patch.object(kaggle_dataset_manager, "_hf_list_repo_files", return_value=files), \
             patch.object(kaggle_dataset_manager, "_hf_file_size", return_value=1234):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = resolve_hf_input("hf:org/repo", input_fn=fake_input)

        self.assertEqual(sum(1 for p in prompts if "Escolha" in p), 2)  # re-perguntou
        self.assertEqual(result[0]["file_path"], "nested/b.safetensors")
        self.assertEqual(result[0]["category"], "loras")
        self.assertEqual(result[0]["base_model"], "sdxl")
        self.assertEqual(result[0]["size"], 1234)
        listing = stdout.getvalue().split("Arquivos de modelo em org/repo:")[1].split("[WARN]")[0]
        self.assertIn("a.safetensors", listing)
        self.assertNotIn("README.md", listing)  # filtrado por extensão

    def test_resolve_hf_repo_only_without_model_files_fails(self):
        """Repo sem arquivos de modelo -> erro claro na Fase 1 (não baixa nada)"""
        with patch.object(kaggle_dataset_manager, "_hf_list_repo_files", return_value=["README.md", ".gitattributes"]):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_hf_input("hf:org/repo", input_fn=lambda p="": "")
        self.assertIn("Nenhum arquivo de modelo", str(ctx.exception))


    def test_print_resolution_summary_with_failures_reports_no_action(self):
        """Resumo com falha (item 10): FAILED + linguagem de aborto, sem 'Ready'"""
        outcome = ResolutionOutcome(
            artifacts=[ResolvedArtifact("civitai", "urn:air:krea2:lora:civitai:1@2", 1, 2, "f.safetensors", category="loras")],
            failures=[ResolutionFailure(
                "hf://org/missing/f.safetensors", 2, "huggingface",
                reason="Falha ao obter metadata do Hugging Face.\n  Input:\n    hf://org/missing/f.safetensors",
            )],
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_resolution_summary(outcome)
        out = stdout.getvalue()
        self.assertIn("FAILED:", out)
        self.assertIn("hf://org/missing/f.safetensors", out)
        self.assertIn("1/2 inputs resolved", out)
        self.assertIn("1 input(s) FAILED.", out)
        self.assertIn("Nenhum arquivo foi modificado.", out)
        self.assertIn("Nenhum download foi iniciado.", out)
        self.assertNotIn("Ready for download.", out)

    def test_classify_resolved_artifacts_preset_checkpoint_no_prompt(self):
        """Classificação entre resumo e download: checkpoint com destino pré-respondido não pergunta"""
        hf_art = ResolvedArtifact(
            "huggingface", "hf:o/r/f.safetensors", 1, 2, "f.safetensors",
            category="vae", repo_id="o/r", file_path="f.safetensors", revision="main",
        )
        ckpt = ResolvedArtifact(
            "civitai", "urn:air:sdxl:checkpoint:civitai:1@2", 2, 2, "c.safetensors",
            resource_type="checkpoint",
        )

        def boom(prompt=""):
            raise AssertionError(f"não deve perguntar: {prompt}")

        classify_resolved_artifacts([hf_art, ckpt], input_fn=boom, checkpoint_destination="diffusion_models")
        self.assertEqual(ckpt.category, "diffusion_models")
        self.assertEqual(ckpt.destination, "diffusion_models/c.safetensors")
        self.assertEqual(ckpt.state, "READY_TO_DOWNLOAD")
        self.assertEqual(hf_art.destination, "vae/f.safetensors")
        self.assertEqual(hf_art.state, "READY_TO_DOWNLOAD")

    def test_guess_category_pure_mapping(self):
        """guess_category: puro, retorna None quando exige interação"""
        self.assertEqual(guess_category("lora"), "loras")
        self.assertEqual(guess_category("Checkpoint"), None)
        self.assertEqual(guess_category(""), None)
        self.assertEqual(guess_category("desconhecido"), None)

    def test_redact_secrets_and_format_hf_error(self):
        """Redação de credenciais e mensagem contextual HF (itens 7/14)"""
        message = redact_secrets("falha com hf_abc123xyz e tok456", ["hf_abc123xyz", "tok456"])
        self.assertNotIn("hf_abc123xyz", message)
        self.assertNotIn("tok456", message)
        self.assertIn("***REDACTED***", message)

        formatted = format_hf_resolution_error("hf://o/r/f.safetensors", "o/r", "f.safetensors", "erro X")
        self.assertIn("Input:", formatted)
        self.assertIn("hf://o/r/f.safetensors", formatted)
        self.assertIn("Repo:", formatted)
        self.assertIn("o/r", formatted)
        self.assertIn("Erro técnico:", formatted)
        self.assertIn("erro X", formatted)
        self.assertIn("nenhum download foi iniciado", formatted)

    # =========================================================================
    # TESTES DE CACHE/StAGING HF (item 8)
    # =========================================================================

    def test_configure_and_cleanup_hf_cache(self):
        """configure_hf_cache redireciona HF_HOME etc. e cleanup restaura/remove"""
        kvars = kaggle_dataset_manager.HF_CACHE_ENV_VARS
        original = {var: os.environ.get(var) for var in kvars}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache_dir = configure_hf_cache(Path(tmp) / "hf_cache")
                self.assertEqual(os.environ.get("HF_HOME"), str(cache_dir))
                self.assertEqual(os.environ.get("HF_HUB_CACHE"), str(cache_dir))
                self.assertTrue(cache_dir.is_dir())
                cleanup_hf_cache(cache_dir)
                self.assertFalse(cache_dir.exists())
            for var, value in original.items():
                self.assertEqual(os.environ.get(var), value)
        finally:
            cleanup_hf_cache(None)

    def test_download_hf_file_removes_local_cache(self):
        """.cache/huggingface do hf_hub_download é removido do staging após o download"""
        def fake_hf_hub_download(repo_id, filename, revision="main", token=None, local_dir=None):
            target = Path(local_dir) / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"hf-bytes")
            cache_blob = Path(local_dir) / ".cache" / "huggingface" / "download.lock"
            cache_blob.parent.mkdir(parents=True, exist_ok=True)
            cache_blob.write_text("cache")
            return str(target)

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            with patch.object(kaggle_dataset_manager, "_hf_hub_download", side_effect=fake_hf_hub_download):
                item = download_hf_file("org/repo", "nested/f.safetensors", "loras", staging, hf_token="t")
            self.assertTrue((staging / "loras" / "f.safetensors").exists())
            self.assertFalse((staging / "loras" / ".cache").exists())
            self.assertEqual(item.path, "loras/f.safetensors")

    def test_publish_staged_state_ignores_hidden_cache_files(self):
        """Arquivos ocultos (.cache) não entram no manifest e são limpos antes do publish"""
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            (staging / "loras").mkdir()
            (staging / "loras" / "ok.safetensors").write_bytes(b"abc")
            hidden = staging / ".cache" / "junk.bin"
            hidden.parent.mkdir()
            hidden.write_bytes(b"junk")

            def fake_input(prompt=""):
                return "n" if "Publicar" in prompt else ""

            with patch.object(kaggle_dataset_manager, "kaggle_files", return_value=[]), \
                 patch.object(kaggle_dataset_manager, "publish", return_value="ok"):
                result = publish_staged_state("owner/ds", staging, input_fn=fake_input)

            self.assertIsNone(result)
            manifest = json.loads((staging / "dataset-manifest.json").read_text())
            paths = [entry["path"] for entry in manifest["files"]]
            self.assertIn("loras/ok.safetensors", paths)
            self.assertFalse(any(".cache" in path for path in paths))
            self.assertFalse((staging / ".cache").exists())


    # =========================================================================
    # TESTES DE FLUXO GLOBAL - 7 INPUTS (item 16)
    # =========================================================================

    @staticmethod
    def _seven_pending():
        return [
            "urn:air:krea2:lora:civitai:1@2",
            "hf:org/alpha/f.safetensors",
            "urn:air:krea2:vae:civitai:3@4",
            "hf:org/beta/f.safetensors",
            "urn:air:krea2:lora:civitai:5@6",
            "hf:org/gamma/f.safetensors",
            "urn:air:krea2:text_encoders:civitai:7@8",
        ]

    @staticmethod
    def _hf_flow_info(value):
        """Dict de info no formato devolvido por resolve_hf_input (1 arquivo)."""
        repo_id = value[len("hf:"):].rsplit("/", 1)[0]
        return {
            "repo_id": repo_id,
            "file_path": "f.safetensors",
            "revision": "main",
            "filename": "f.safetensors",
            "category": "loras",
            "base_model": "sdxl",
            "size": 10,
            "source_url": f"https://huggingface.co/{repo_id}/resolve/main/f.safetensors",
            "air": None,
            "source": "hf",
        }

    def _resolve_seven(self, hf_fail=(), civ_fail=()):
        """Resolve os 7 inputs com providers mockados; nenhum download pode ocorrer."""
        pending = self._seven_pending()

        def fake_civ(value, token, input_fn=input):
            if value in civ_fail:
                raise RuntimeError("civitai indisponível")
            return [self._fake_info(value)]

        def fake_hf(value, hf_token=None, input_fn=input):
            if any(marker in value for marker in hf_fail):
                raise RuntimeError("Repositório não encontrado: org")
            return [self._hf_flow_info(value)]

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_civ), \
             patch.object(kaggle_dataset_manager, "resolve_hf_input", side_effect=fake_hf), \
             patch.object(kaggle_dataset_manager, "download_civitai_file") as civ_download, \
             patch.object(kaggle_dataset_manager, "download_hf_file") as hf_download:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                outcome = resolve_queue_metadata(pending, "tok", input_fn=lambda p="": "", hf_token="t")
            civ_download.assert_not_called()
            hf_download.assert_not_called()
        return outcome

    def test_resolution_flow_7_of_7_success(self):
        """7 inputs / 7 sucesso: outcome completo e ok"""
        outcome = self._resolve_seven()
        self.assertEqual(len(outcome.artifacts), 7)
        self.assertEqual(outcome.failures, [])
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.total, 7)

    def test_resolution_flow_7_with_1_failure(self):
        """7 inputs / 1 falha: registra a falha sem derrubar os demais (sem download)"""
        outcome = self._resolve_seven(hf_fail=("beta",))
        self.assertEqual(len(outcome.artifacts), 6)
        self.assertEqual(len(outcome.failures), 1)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.failures[0].original_input, "hf:org/beta/f.safetensors")

    def test_resolution_flow_7_with_2_failures(self):
        """7 inputs / 2 falhas (1 HF + 1 Civitai)"""
        outcome = self._resolve_seven(hf_fail=("beta",), civ_fail=("urn:air:krea2:vae:civitai:3@4",))
        self.assertEqual(len(outcome.artifacts), 5)
        self.assertEqual(len(outcome.failures), 2)
        self.assertFalse(outcome.ok)

    def test_resolution_flow_7_all_failures(self):
        """7 inputs / todos falham: outcome sem artefatos -> orquestrador aborta"""
        pending = self._seven_pending()

        def boom(*args, **kwargs):
            raise RuntimeError("falha total")

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=boom), \
             patch.object(kaggle_dataset_manager, "resolve_hf_input", side_effect=boom):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                outcome = resolve_queue_metadata(pending, "tok", input_fn=lambda p="": "", hf_token="t")
        self.assertEqual(outcome.artifacts, [])
        self.assertEqual(len(outcome.failures), 7)
        self.assertFalse(outcome.ok)

    def test_print_resolution_summary_success(self):
        """Resumo de sucesso (item 17): formato, contagem e 'Ready for download'"""
        artifact = ResolvedArtifact(
            "huggingface", "hf:org/repo/f.safetensors", 1, 1, "f.safetensors",
            size_bytes=2048, category="loras", base_model="sdxl",
            repo_id="org/repo", file_path="f.safetensors", revision="main",
            destination="loras/f.safetensors", state="READY_TO_DOWNLOAD",
        )
        outcome = ResolutionOutcome(artifacts=[artifact])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_resolution_summary(outcome)
        out = stdout.getvalue()
        self.assertIn("INPUT RESOLUTION SUMMARY", out)
        self.assertIn("[1] Hugging Face", out)
        self.assertIn("Repo: org/repo", out)
        self.assertIn("Category: loras", out)
        self.assertIn("1/1 inputs resolved", out)
        self.assertIn("Ready for download.", out)


if __name__ == "__main__":
    unittest.main()
