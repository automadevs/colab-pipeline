import json
import hashlib
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
    build_dataset_path,
    compare_states,
    download_input_queue,
    format_size,
    manifest_payload,
    parse_current_files,
    parse_air,
    normalize_category,
    classify_civitai_type,
    parse_size,
    publish,
    render_preview,
    retry,
    validate_air,
    validate_civitai_url,
    _expected_sha256,
    download_with_civitai_cli,
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
                f"download:{entries[0]}",
                f"resolve:{entries[1]}",
                f"download:{entries[1]}",
            ],
        )

    def test_queue_skips_invalid_entries(self):
        inputs = iter(["not-an-air", "urn:air:x:lora:civitai:0@1", "done"])
        with patch.object(kaggle_dataset_manager, "resolve_civitai_input") as resolve:
            queue = download_input_queue(Path("/tmp"), "token", input_fn=lambda prompt="": next(inputs))
        self.assertEqual(queue, [])
        resolve.assert_not_called()

    def test_queue_continues_after_item_failure(self):
        entries = ["urn:air:krea2:lora:civitai:1@2", "urn:air:krea2:lora:civitai:3@4"]
        inputs = iter([*entries, "done"])

        def fake_resolve(value, token, input_fn):
            if value == entries[0]:
                raise RuntimeError("falha permanente")
            return [self._fake_info(value)]

        def fake_download(info, category, staging_dir, token, source_url=None, air=None):
            return DatasetFile("loras/ok.safetensors", 1, "hash")

        with patch.object(kaggle_dataset_manager, "resolve_civitai_input", side_effect=fake_resolve), patch.object(
            kaggle_dataset_manager, "download_civitai_file", side_effect=fake_download
        ), patch.object(kaggle_dataset_manager, "classify_civitai_type", return_value="loras"):
            queue = download_input_queue(Path("/tmp"), "token", input_fn=lambda prompt="": next(inputs))

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].path, "loras/ok.safetensors")


if __name__ == "__main__":
    unittest.main()
