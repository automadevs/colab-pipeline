"""Testes do orquestrador do notebook 00 (resolução do dataset alvo e CLI mínima).

Cobrem a falha reportada em runtime: o módulo não pode exigir
KAGGLE_USERNAME/KAGGLE_DATASET_NAME no import, e sem dataset alvo o pipeline deve
abortar com mensagem acionável ANTES de clonar o repositório.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import master_pipeline


class MasterPipelineDatasetTests(unittest.TestCase):
    def test_import_does_not_require_dataset_env(self):
        """Importar o módulo não pode falhar sem KAGGLE_USERNAME/KAGGLE_DATASET_NAME."""
        with patch.dict(os.environ, {}, clear=True):
            module = importlib.reload(master_pipeline)
            self.assertIsNone(module.DATASET)
        importlib.reload(master_pipeline)

    def test_resolve_dataset_name_prefers_override(self):
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "u", "KAGGLE_DATASET_NAME": "d"}, clear=True):
            self.assertEqual(master_pipeline.resolve_dataset_name("owner/nome"), "owner/nome")
            self.assertEqual(master_pipeline.resolve_dataset_name(), "u/d")

    def test_resolve_dataset_name_error_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                master_pipeline.resolve_dataset_name()
        message = str(ctx.exception)
        self.assertIn("KAGGLE_DATASET_NAME", message)
        self.assertIn("Notebook access", message)
        self.assertIn("--dataset", message)

    def test_parse_args_dataset_override(self):
        self.assertEqual(master_pipeline.parse_args(["--dataset", "owner/nome"]).dataset, "owner/nome")
        self.assertIsNone(master_pipeline.parse_args([]).dataset)

    def test_main_fails_fast_without_dataset_and_does_not_clone(self):
        """Sem dataset alvo, main() aborta antes do setup do repositório (sem side effects)."""
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(master_pipeline, "setup_repo") as setup:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = master_pipeline.main([])
        self.assertEqual(code, 1)
        setup.assert_not_called()
        self.assertIn("Dataset Kaggle não resolvido", stdout.getvalue())

    def test_main_uses_dataset_override(self):
        """--dataset resolve o alvo sem depender de env."""
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(master_pipeline, "setup_repo", side_effect=RuntimeError("sentinel")):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = master_pipeline.main(["--dataset", "owner/nome"])
        self.assertEqual(code, 1)
        output = stdout.getvalue()
        self.assertIn("Dataset alvo: owner/nome", output)
        self.assertIn("Falha ao preparar o repositório", output)


if __name__ == "__main__":
    unittest.main()
