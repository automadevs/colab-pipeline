import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kaggle_dataset_manager import (
    CATEGORIES,
    DatasetFile,
    build_dataset_path,
    compare_states,
    format_size,
    manifest_payload,
    parse_current_files,
    parse_air,
    normalize_category,
    classify_civitai_type,
    parse_size,
    publish,
    render_preview,
    _expected_sha256,
)


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
        with tempfile.TemporaryDirectory() as tmp:
            with patch("kaggle_dataset_manager.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "version 7"
                result = publish("automamermaid/comfydocs", Path(tmp), "test update")
                self.assertEqual(result, "version 7")
                metadata = json.loads((Path(tmp) / "dataset-metadata.json").read_text())
                self.assertEqual(metadata["id"], "automamermaid/comfydocs")
                command = run.call_args.args[0]
                self.assertIn("version", command)
                self.assertIn("-p", command)
                self.assertNotIn("--delete-old-versions", command)


if __name__ == "__main__":
    unittest.main()
