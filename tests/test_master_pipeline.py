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


    def test_main_aborts_before_dataset_edits_on_resolution_failure(self):
        """Com falha de resolução, main() imprime o resumo e aborta SEM edições/downloads (item 10)."""
        import kaggle_dataset_manager
        from kaggle_dataset_manager import ResolutionFailure, ResolutionOutcome, ResolvedArtifact

        outcome = ResolutionOutcome(
            artifacts=[ResolvedArtifact("civitai", "urn:air:krea2:lora:civitai:1@2", 1, 2, "f.safetensors")],
            failures=[
                ResolutionFailure(
                    "hf://org/missing/f.safetensors",
                    2,
                    "huggingface",
                    reason="Falha ao obter metadata do Hugging Face.\n  Input:\n    hf://org/missing/f.safetensors",
                )
            ],
        )
        with patch.dict(os.environ, {"CIVITAI_TOKEN": "tok", "HF_TOKEN": "hf"}, clear=False), \
             patch.object(master_pipeline, "setup_repo"), \
             patch.object(master_pipeline, "inspect_environment"), \
             patch.object(master_pipeline, "ensure_kaggle_auth", return_value=True), \
             patch.object(kaggle_dataset_manager, "get_secret", return_value="tok"), \
             patch.object(kaggle_dataset_manager, "collect_input_queue", return_value=["urn:air:krea2:lora:civitai:1@2"]), \
             patch.object(kaggle_dataset_manager, "resolve_queue_metadata", return_value=outcome), \
             patch.object(kaggle_dataset_manager, "collect_dataset_edits", side_effect=AssertionError("edições não devem rodar")) as edits, \
             patch.object(kaggle_dataset_manager, "download_resolved_queue", side_effect=AssertionError("download não deve iniciar")) as downloads:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = master_pipeline.main(["--dataset", "owner/nome"])

        self.assertEqual(code, 1)
        edits.assert_not_called()
        downloads.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("INPUT RESOLUTION SUMMARY", output)
        self.assertIn("FAILED:", output)
        self.assertIn("Nenhum download foi iniciado", output)
        self.assertIn("abortando SEM modificar o dataset", output)
        self.assertNotIn("Você deseja modificar", output)  # nunca pergunta edição com falha pendente


class MasterPipelineTransactionalDownloadTests(unittest.TestCase):
    """Publicação transacional: publish só com 100% dos downloads OK (mocked)."""

    def _artifacts(self, n: int):
        from kaggle_dataset_manager import ResolvedArtifact

        return [
            ResolvedArtifact("civitai", f"urn:air:krea2:lora:civitai:{i}@{i}", i, n, f"f{i}.safetensors")
            for i in range(1, n + 1)
        ]

    def _run_main_after_resolution(self, download_outcome, *, n_resolved: int = 7):
        import kaggle_dataset_manager
        from kaggle_dataset_manager import ResolutionOutcome

        artifacts = self._artifacts(n_resolved)
        outcome = ResolutionOutcome(artifacts=artifacts, failures=[])
        with patch.dict(os.environ, {"CIVITAI_TOKEN": "tok", "HF_TOKEN": "hf"}, clear=False), \
             patch.object(master_pipeline, "setup_repo"), \
             patch.object(master_pipeline, "inspect_environment"), \
             patch.object(master_pipeline, "ensure_kaggle_auth", return_value=True), \
             patch.object(kaggle_dataset_manager, "get_secret", return_value="tok"), \
             patch.object(kaggle_dataset_manager, "collect_input_queue", return_value=["urn:air:x"]), \
             patch.object(kaggle_dataset_manager, "resolve_queue_metadata", return_value=outcome), \
             patch.object(kaggle_dataset_manager, "queue_contains_checkpoint", return_value=False), \
             patch.object(kaggle_dataset_manager, "classify_resolved_artifacts"), \
             patch.object(kaggle_dataset_manager, "print_resolution_summary"), \
             patch.object(kaggle_dataset_manager, "collect_dataset_edits", return_value=[]), \
             patch.object(kaggle_dataset_manager, "download_resolved_queue", return_value=download_outcome), \
             patch.object(kaggle_dataset_manager, "write_manifest") as write_manifest, \
             patch.object(kaggle_dataset_manager, "publish_staged_state") as publish, \
             patch.object(master_pipeline, "publish_via_kagglehub") as hub:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = master_pipeline.main(["--dataset", "owner/nome"])
        return code, stdout.getvalue(), write_manifest, publish, hub

    def test_main_publishes_when_all_downloads_succeed(self):
        """7/7 resolução + 7/7 downloads → publish chamado."""
        from kaggle_dataset_manager import DatasetFile, DownloadOutcome

        items = [DatasetFile(f"loras/f{i}.safetensors", i, f"h{i}") for i in range(1, 8)]
        download_outcome = DownloadOutcome(items=items, failures=[], resolved_count=7)
        code, output, write_manifest, publish, hub = self._run_main_after_resolution(download_outcome)
        self.assertEqual(code, 0)
        write_manifest.assert_called_once()
        publish.assert_called_once()
        hub.assert_not_called()
        self.assertIn("[SUCCESS]", output)

    def test_main_skips_publish_on_partial_download_failure(self):
        """7/7 resolução + 6/7 downloads → publish NÃO chamado, exit != 0."""
        from kaggle_dataset_manager import DatasetFile, DownloadFailure, DownloadOutcome

        items = [DatasetFile(f"loras/f{i}.safetensors", i, f"h{i}") for i in range(1, 7)]
        failures = [DownloadFailure("hf:org/missing/f.safetensors", 7, 7, reason="not found")]
        download_outcome = DownloadOutcome(items=items, failures=failures, resolved_count=7)
        code, output, write_manifest, publish, hub = self._run_main_after_resolution(download_outcome)
        self.assertNotEqual(code, 0)
        write_manifest.assert_not_called()
        publish.assert_not_called()
        hub.assert_not_called()
        self.assertIn("[DOWNLOAD FAILED]", output)
        self.assertIn("7 arquivo(s) resolvido(s)", output)
        self.assertIn("6 arquivo(s) baixado(s)", output)
        self.assertIn("1 arquivo(s) com falha", output)
        self.assertIn("Nenhuma alteração no dataset foi publicada.", output)

    def test_main_skips_publish_when_all_downloads_fail(self):
        """7/7 resolução + 0/7 downloads → publish NÃO chamado, exit != 0."""
        from kaggle_dataset_manager import DownloadFailure, DownloadOutcome

        failures = [
            DownloadFailure(f"hf:org/missing/f{i}.safetensors", i, 7, reason="fail")
            for i in range(1, 8)
        ]
        download_outcome = DownloadOutcome(items=[], failures=failures, resolved_count=7)
        code, output, write_manifest, publish, hub = self._run_main_after_resolution(download_outcome)
        self.assertNotEqual(code, 0)
        write_manifest.assert_not_called()
        publish.assert_not_called()
        hub.assert_not_called()
        self.assertIn("[DOWNLOAD FAILED]", output)
        self.assertIn("0 arquivo(s) baixado(s)", output)
        self.assertIn("7 arquivo(s) com falha", output)

    def test_main_single_download_failure_leaves_dataset_unchanged(self):
        """1/1 download falha → dataset inalterado (publish nunca chamado)."""
        from kaggle_dataset_manager import DownloadFailure, DownloadOutcome

        download_outcome = DownloadOutcome(
            items=[],
            failures=[DownloadFailure("hf:org/x/f.safetensors", 1, 1, reason="boom")],
            resolved_count=1,
        )
        code, output, write_manifest, publish, hub = self._run_main_after_resolution(
            download_outcome, n_resolved=1
        )
        self.assertNotEqual(code, 0)
        write_manifest.assert_not_called()
        publish.assert_not_called()
        hub.assert_not_called()
        self.assertIn("Nenhuma alteração no dataset foi publicada.", output)


if __name__ == "__main__":
    unittest.main()
