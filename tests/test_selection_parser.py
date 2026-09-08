import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from kaggle_sync import parse_model_selection, select_dataset_files


class SelectionParserTests(unittest.TestCase):
    PATHS = ["checkpoints/a.safetensors", "loras/b.safetensors", "vae/c.safetensors", "loras/d.safetensors", "clip/e.safetensors", "vae/f.safetensors", "loras/g.safetensors", "vae/h.safetensors"]

    def test_single_index(self):
        self.assertEqual(parse_model_selection("1", self.PATHS), [self.PATHS[0]])

    def test_comma_list(self):
        self.assertEqual(parse_model_selection("1,3,5", self.PATHS), [self.PATHS[0], self.PATHS[2], self.PATHS[4]])

    def test_range(self):
        self.assertEqual(parse_model_selection("1-4", self.PATHS), self.PATHS[:4])

    def test_mixed_selection(self):
        self.assertEqual(parse_model_selection("1,3-5,8", self.PATHS), [self.PATHS[0], *self.PATHS[2:5], self.PATHS[7]])

    def test_all(self):
        self.assertEqual(parse_model_selection("all", self.PATHS), self.PATHS)

    def test_invalid_inputs(self):
        for value in ("", "0", "999", "abc", "1,999", "1,1", "4-2", "1,,2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_model_selection(value, self.PATHS)

    def test_no_candidates(self):
        with self.assertRaises(ValueError):
            parse_model_selection("1", [])

    @patch("kaggle_sync.get_dataset_files_details")
    def test_single_candidate_auto_select(self, get_details):
        get_details.return_value = [{"path": self.PATHS[0], "name": "a.safetensors", "category": "checkpoints", "size": "11.94 GB"}]
        self.assertEqual(select_dataset_files("dataset", auto_select_single=True), [self.PATHS[0]])

    @patch("kaggle_sync.get_dataset_files_details")
    def test_invalid_input_repeats_until_valid(self, get_details):
        get_details.return_value = [
            {"path": self.PATHS[0], "name": "a.safetensors", "category": "checkpoints", "size": "1 GB"},
            {"path": self.PATHS[1], "name": "b.safetensors", "category": "loras", "size": "2 MB"},
        ]
        inputs = iter(["", "1,999", "1,2"])
        selected = select_dataset_files("dataset", auto_select_single=False, input_fn=lambda _: next(inputs))
        self.assertEqual(selected, self.PATHS[:2])


if __name__ == "__main__":
    unittest.main()