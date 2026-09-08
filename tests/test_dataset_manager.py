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
    parse_size,
    publish,
    render_preview,
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

    def test_manifest(self):
        payload = manifest_payload("automamermaid/comfydocs", {self.new.path: self.new})
        self.assertEqual(payload["dataset"], "automamermaid/comfydocs")
        self.assertEqual(payload["files"][0]["sha256"], "newhash")

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
