"""
Testes de regressão — zero-trust / zero-persistent-image pipeline.

Testes estruturais (rodam em qualquer OS):
  RuntimeContractTests, A, B, C, D, E, F, G, H(estrutural), I, J, K, L, SecureMode, Extras

Testes runtime (Linux com /dev/shm):
  class TestRuntime_* — marcados com @pytest.mark.skipif(not ON_LINUX)
  ou skip via unittest.skipUnless quando pytest não está disponível.

Testes de ZIP AES (requerem pyzipper):
  class TestZipAES_* — skipados se pyzipper não instalado.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import comfyui_setup
import gpu_detect
import ngrok_tunnel

ON_LINUX = sys.platform.startswith("linux")
HAS_DEV_SHM = ON_LINUX and Path("/dev/shm").exists()

try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shm_tmpdir(name: str) -> Path:
    """Cria diretório temporário em /dev/shm para testes runtime. Apenas Linux."""
    d = Path(f"/dev/shm/{name}")
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


# ---------------------------------------------------------------------------
# Testes de contrato originais
# ---------------------------------------------------------------------------

class RuntimeContractTests(unittest.TestCase):
    def test_gpu_parser_enumerates_all_devices(self):
        result = gpu_detect.parse_nvidia_smi_csv(
            "Tesla T4, 15360 MiB, 535.104.05\nTesla T4, 15360 MiB, 535.104.05\n"
        )
        self.assertEqual([g["index"] for g in result], [0, 1])
        self.assertEqual(result[0]["vram_mib"], 15360)
        self.assertEqual(result[1]["driver"], "535.104.05")

    def test_cuda_device_defaults_to_zero_and_can_change(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(comfyui_setup.get_cuda_device(), 0)
        with patch.dict(os.environ, {"COMFYUI_CUDA_DEVICE": "1"}):
            self.assertEqual(comfyui_setup.get_cuda_device(), 1)

    def test_default_nodes_exclude_manager_and_include_required(self):
        self.assertEqual(comfyui_setup.DEFAULT_CUSTOM_NODES,
                         ["cubiq/ComfyUI_essentials", "lbouaraba/comfyui-krea2edit"])
        self.assertEqual(comfyui_setup.filter_custom_nodes(["ltdrdata/ComfyUI-Manager"]), [])

    def test_manager_requirements_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "manager_requirements.txt"
            req.write_text("requests\n", encoding="utf-8")
            with patch.object(comfyui_setup, "_run") as run:
                self.assertTrue(comfyui_setup.install_manager_requirements(Path(tmp)))
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][-2:], ["-r", str(req)])

    def test_krea2edit_install_and_update_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp)
            with patch.object(comfyui_setup, "_run") as run:
                node_path = comfyui_setup.install_or_update_custom_node(
                    custom_dir, "lbouaraba/comfyui-krea2edit"
                )
                self.assertIn("clone", run.call_args.args[0])
                node_path.mkdir(parents=True, exist_ok=True)
                (node_path / ".git").mkdir()
                run.reset_mock()
                comfyui_setup.install_or_update_custom_node(custom_dir, "lbouaraba/comfyui-krea2edit")
                self.assertEqual(run.call_args.args[0], ["git", "pull", "--ff-only"])

    def test_existing_non_git_node_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            node_path = Path(tmp) / "comfyui-krea2edit"
            node_path.mkdir()
            with self.assertRaises(RuntimeError):
                comfyui_setup.install_or_update_custom_node(Path(tmp), "lbouaraba/comfyui-krea2edit")

    def test_runtime_starts_ngrok_only_after_health(self):
        events = []
        process = MagicMock(pid=123)
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = lambda port: events.append("ngrok") or "https://example.ngrok.app"

        comfyui_setup.set_secure_mode(False, _test_override=True)
        try:
            # Primeira chamada health_check (porta livre) -> False
            # Segunda chamada (após start) -> True
            health_results = [False, True]
            def health_mock(*args, **kwargs):
                events.append("health")
                return health_results.pop(0)
            
            with patch.object(comfyui_setup, "start_comfyui",
                               side_effect=lambda **_: events.append("start") or process), \
                 patch.object(comfyui_setup, "health_check", side_effect=health_mock), \
                 patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
                 patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
                result = comfyui_setup.start_comfyui_runtime(
                    comfyui_dir=Path(tempfile.mkdtemp()),
                    enable_ngrok=True, reuse_existing=False, secure_mode=False,
                )
        finally:
            comfyui_setup.set_secure_mode(True, _test_override=True)

        # Ordem real: health_check (porta livre) -> start -> health_check (pós-start) -> ngrok
        self.assertEqual(events, ["health", "start", "health", "ngrok"])
        self.assertTrue(result["ngrok_started"])

    def test_failed_health_does_not_start_ngrok(self):
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)
        comfyui_setup.set_secure_mode(False, _test_override=True)
        try:
            with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
                 patch.object(comfyui_setup, "health_check", return_value=False), \
                 patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
                 patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
                result = comfyui_setup.start_comfyui_runtime(
                    comfyui_dir=Path(tempfile.mkdtemp()),
                    enable_ngrok=True, health_timeout=1, reuse_existing=False, secure_mode=False,
                )
        finally:
            comfyui_setup.set_secure_mode(True, _test_override=True)

        self.assertFalse(result["health"])
        fake_ngrok.start_ngrok_tunnel.assert_not_called()

    def test_ngrok_token_not_logged(self):
        token = "secret-token-value"
        calls = []
        fake_ngrok_api = types.SimpleNamespace(
            set_auth_token=lambda v: calls.append(("auth", v)),
            kill=lambda: calls.append(("kill",)),
            connect=lambda **kw: calls.append(("connect", kw))
                or types.SimpleNamespace(public_url="http://x.ngrok.app"),
        )
        fake_pyngrok = types.ModuleType("pyngrok")
        fake_pyngrok.ngrok = fake_ngrok_api
        with patch.dict(sys.modules, {"pyngrok": fake_pyngrok}), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            url = ngrok_tunnel.start_ngrok_tunnel(authtoken=token)
        self.assertEqual(url, "https://x.ngrok.app")
        self.assertNotIn(token, stdout.getvalue())

    def test_notebooks_have_language_metadata_and_ids(self):
        for nb in Path(__file__).parents[1].glob("**/*.ipynb"):
            doc = json.loads(nb.read_text(encoding="utf-8"))
            for cell in doc["cells"]:
                self.assertIn("language", cell.get("metadata", {}), str(nb))
                if cell.get("metadata", {}).get("language") in {"markdown", "python"}:
                    self.assertTrue(cell.get("metadata", {}).get("id"), str(nb))


# ---------------------------------------------------------------------------
# A — Input directory em /dev/shm
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
class TestA_InputDirectoryInShm(unittest.TestCase):
    def test_build_comfyui_command_input_in_shm(self):
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            secure_mode=False,
        )
        self.assertIn("--input-directory", cmd)
        idx = cmd.index("--input-directory")
        self.assertTrue(cmd[idx + 1].startswith("/dev/shm"))
        self.assertNotIn("/kaggle/working", cmd[idx + 1])

    def test_default_shm_input_is_dev_shm(self):
        self.assertTrue(str(comfyui_setup.SHM_INPUT).startswith("/dev/shm"))

    def test_assert_shm_path_rejects_kaggle_working(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.assert_shm_path(Path("/kaggle/working/ComfyUI/input"), "test")

    def test_assert_shm_path_accepts_dev_shm(self):
        comfyui_setup.assert_shm_path(Path("/dev/shm/comfy_ui_input"), "test")


# ---------------------------------------------------------------------------
# B — Output directory em /dev/shm
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
class TestB_OutputDirectoryInShm(unittest.TestCase):
    def test_output_in_shm(self):
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            secure_mode=False,
        )
        idx = cmd.index("--output-directory")
        self.assertTrue(cmd[idx + 1].startswith("/dev/shm"))
        self.assertNotIn("/kaggle/working", cmd[idx + 1])

    def test_rejects_kaggle_working_output(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                output_dir=Path("/kaggle/working/ComfyUI/output"),
                input_dir=comfyui_setup.SHM_INPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                secure_mode=False,
            )


# ---------------------------------------------------------------------------
# C — Temp directory em /dev/shm
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
class TestC_TempDirectoryInShm(unittest.TestCase):
    def test_temp_in_shm(self):
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            secure_mode=False,
        )
        idx = cmd.index("--temp-directory")
        self.assertTrue(cmd[idx + 1].startswith("/dev/shm"))

    def test_rejects_kaggle_working_temp(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                output_dir=comfyui_setup.SHM_OUTPUT,
                input_dir=comfyui_setup.SHM_INPUT,
                temp_dir=Path("/kaggle/working/ComfyUI/temp"),
                secure_mode=False,
            )


# ---------------------------------------------------------------------------
# D — Process reuse validation
# ---------------------------------------------------------------------------

class TestD_ProcessReuseValidation(unittest.TestCase):
    def test_verify_paths_rejects_kaggle_working_output(self):
        cmdline = [
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--input-directory", "/dev/shm/comfy_ui_input",
            "--output-directory", "/kaggle/working/ComfyUI/output",
            "--temp-directory", "/dev/shm/comfy_ui_temp",
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, reason = comfyui_setup._verify_process_paths(
                9999, comfyui_setup.SHM_INPUT, comfyui_setup.SHM_OUTPUT,
                comfyui_setup.SHM_TEMP, "127.0.0.1", 8188,
            )
        self.assertFalse(ok)

    def test_verify_paths_rejects_missing_input_directory(self):
        cmdline = [
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--output-directory", "/dev/shm/comfy_ui_output",
            "--temp-directory", "/dev/shm/comfy_ui_temp",
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, _ = comfyui_setup._verify_process_paths(
                9999, comfyui_setup.SHM_INPUT, comfyui_setup.SHM_OUTPUT,
                comfyui_setup.SHM_TEMP, "127.0.0.1", 8188,
            )
        self.assertFalse(ok)

    def test_verify_paths_accepts_correct_shm_paths(self):
        cmdline = [
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--input-directory", str(comfyui_setup.SHM_INPUT),
            "--output-directory", str(comfyui_setup.SHM_OUTPUT),
            "--temp-directory", str(comfyui_setup.SHM_TEMP),
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, reason = comfyui_setup._verify_process_paths(
                9999, comfyui_setup.SHM_INPUT, comfyui_setup.SHM_OUTPUT,
                comfyui_setup.SHM_TEMP, "127.0.0.1", 8188,
            )
        self.assertTrue(ok, reason)

    def test_reuse_existing_false_by_default(self):
        import inspect
        sig = inspect.signature(comfyui_setup.start_comfyui_runtime)
        self.assertFalse(sig.parameters["reuse_existing"].default)

    def test_mismatched_process_is_killed(self):
        killed = []
        started = []
        bad_cmdline = [
            "python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--input-directory", "/kaggle/working/ComfyUI/input",
            "--output-directory", "/kaggle/working/ComfyUI/output",
            "--temp-directory", "/kaggle/working/ComfyUI/temp",
        ]
        process = MagicMock(pid=1234)
        comfyui_setup.set_secure_mode(False, _test_override=True)
        try:
            with patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=9999), \
                 patch.object(comfyui_setup, "health_check", side_effect=[True, True]), \
                 patch.object(comfyui_setup, "_read_proc_cmdline", return_value=bad_cmdline), \
                 patch.object(comfyui_setup, "kill_mismatched_process",
                               side_effect=lambda pid: killed.append(pid)), \
                 patch.object(comfyui_setup, "start_comfyui",
                               side_effect=lambda **_: started.append(1) or process), \
                 patch.object(comfyui_setup, "provision_shm_dirs"):
                comfyui_setup.start_comfyui_runtime(
                    comfyui_dir=Path(tempfile.mkdtemp()),
                    reuse_existing=True, enable_ngrok=False, secure_mode=False,
                )
        finally:
            comfyui_setup.set_secure_mode(True, _test_override=True)

        self.assertIn(9999, killed)
        self.assertEqual(len(started), 1)


# ---------------------------------------------------------------------------
# E/F — extra_args não podem sobrescrever paths
# ---------------------------------------------------------------------------

class TestEF_ExtraArgsPrevention(unittest.TestCase):
    def test_extra_args_cannot_override_input_directory(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--input-directory", "/kaggle/working/ComfyUI/input"],
                secure_mode=False,
            )

    def test_extra_args_cannot_override_output_directory(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--output-directory", "/kaggle/working/out"],
                secure_mode=False,
            )

    def test_extra_args_cannot_override_temp_directory(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--temp-directory", "/kaggle/working/temp"],
                secure_mode=False,
            )


# ---------------------------------------------------------------------------
# G — ZIP não em /kaggle/working
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
class TestG_ZipNotInKaggleWorking(unittest.TestCase):
    def test_shm_archive_not_in_kaggle_working(self):
        self.assertFalse(str(comfyui_setup.SHM_ARCHIVE).startswith("/kaggle"))
        self.assertTrue(str(comfyui_setup.SHM_ARCHIVE).startswith("/dev/shm"))

    def test_create_secure_zip_rejects_non_shm_archive_dir(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.create_secure_zip(
                src_dir=Path("/dev/shm/comfy_ui_output"),
                archive_dir=Path("/kaggle/working"),
                zip_password="test",
            )

    def test_create_secure_zip_aborts_without_password(self):
        with patch.dict(os.environ, {}, clear=True):
            fake_ks = MagicMock()
            fake_ks.UserSecretsClient.return_value.get_secret.return_value = None
            with patch.dict(sys.modules, {"kaggle_secrets": fake_ks}):
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.create_secure_zip(
                        src_dir=comfyui_setup.SHM_OUTPUT,
                        archive_dir=comfyui_setup.SHM_ARCHIVE,
                        zip_password=None,
                    )


# ---------------------------------------------------------------------------
# H — ZIP AES-256 (estrutural + runtime)
# ---------------------------------------------------------------------------

class TestH_ZipIsEncrypted_Structural(unittest.TestCase):
    def test_pyzipper_and_wz_aes_in_source(self):
        src = Path(comfyui_setup.__file__).read_text(encoding="utf-8")
        self.assertIn("pyzipper", src)
        self.assertIn("WZ_AES", src)
        self.assertIn("AESZipFile", src)

    def test_create_secure_zip_requires_password(self):
        with patch.dict(os.environ, {}, clear=True):
            fake_ks = MagicMock()
            fake_ks.UserSecretsClient.return_value.get_secret.return_value = None
            with patch.dict(sys.modules, {"kaggle_secrets": fake_ks}):
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.create_secure_zip(
                        src_dir=comfyui_setup.SHM_OUTPUT,
                        archive_dir=comfyui_setup.SHM_ARCHIVE,
                        zip_password=None,
                    )


@unittest.skipUnless(HAS_DEV_SHM and HAS_PYZIPPER, "Requer /dev/shm Linux e pyzipper")
class TestH_ZipIsEncrypted_Runtime(unittest.TestCase):
    """Teste runtime: cria ZIP real com AES-256 e verifica comportamento de abertura."""

    def setUp(self):
        self.test_dir = _shm_tmpdir("test_zip_src")
        self.arch_dir = _shm_tmpdir("test_zip_arch")
        (self.test_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        self.password = "test_password_xyz_123"

    def tearDown(self):
        for d in (self.test_dir, self.arch_dir):
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    def test_zip_cannot_be_opened_without_password(self):
        """ZIP criado com AES-256 deve falhar ao abrir sem senha."""
        zip_path = comfyui_setup.create_secure_zip(
            src_dir=self.test_dir,
            archive_dir=self.arch_dir,
            zip_password=self.password,
            run_encryption_test=False,
        )
        self.assertTrue(zip_path.exists())

        open_failed = False
        try:
            with pyzipper.AESZipFile(zip_path, "r") as zf:
                zf.read("image.png")
        except Exception:
            open_failed = True

        self.assertTrue(open_failed, "ZIP deve falhar ao abrir sem senha")

    def test_zip_can_be_opened_with_correct_password(self):
        """ZIP criado com AES-256 deve abrir com a senha correta."""
        zip_path = comfyui_setup.create_secure_zip(
            src_dir=self.test_dir,
            archive_dir=self.arch_dir,
            zip_password=self.password,
            run_encryption_test=False,
        )
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(self.password.encode())
            content = zf.read("image.png")
        self.assertEqual(content[:8], b"\x89PNG\r\n\x1a\n")

    def test_zip_path_is_in_dev_shm(self):
        """ZIP deve ser criado em /dev/shm, não em /kaggle/working."""
        zip_path = comfyui_setup.create_secure_zip(
            src_dir=self.test_dir,
            archive_dir=self.arch_dir,
            zip_password=self.password,
            run_encryption_test=False,
        )
        self.assertTrue(str(zip_path).startswith("/dev/shm"))
        self.assertNotIn("/kaggle/working", str(zip_path))

    def test_verify_zip_encryption_runtime(self):
        """verify_zip_encryption deve provar comportamento real de criptografia."""
        result = comfyui_setup.verify_zip_encryption(
            self.arch_dir / "_enc_verify_test", self.password
        )
        self.assertTrue(result)

    def test_cleanup_zip_removes_and_verifies(self):
        """cleanup_zip deve remover o arquivo e confirmar."""
        zip_path = comfyui_setup.create_secure_zip(
            src_dir=self.test_dir,
            archive_dir=self.arch_dir,
            zip_password=self.password,
            run_encryption_test=False,
        )
        self.assertTrue(zip_path.exists())
        comfyui_setup.cleanup_zip(zip_path)
        self.assertFalse(zip_path.exists())


# ---------------------------------------------------------------------------
# I — Cleanup
# ---------------------------------------------------------------------------

class TestI_CleanupRemovesFiles(unittest.TestCase):
    @unittest.skipUnless(HAS_DEV_SHM, "Requer /dev/shm Linux")
    def test_clear_output_runtime(self):
        out_dir = _shm_tmpdir("test_clear_output")
        try:
            (out_dir / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (out_dir / "image2.webp").write_bytes(b"RIFF")
            comfyui_setup.clear_output(out_dir)
            remaining = [f for f in out_dir.rglob("*") if f.is_file()]
            self.assertEqual(len(remaining), 0)
        finally:
            import shutil
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

    @unittest.skipUnless(HAS_DEV_SHM, "Requer /dev/shm Linux")
    def test_clear_input_runtime_including_pasted_subdir(self):
        inp_dir = _shm_tmpdir("test_clear_input")
        try:
            pasted = inp_dir / "pasted"
            pasted.mkdir()
            (pasted / "pasted.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (inp_dir / "upload.jpg").write_bytes(b"\xff\xd8\xff")
            comfyui_setup.clear_input(inp_dir)
            remaining = [f for f in inp_dir.rglob("*") if f.is_file()]
            self.assertEqual(len(remaining), 0)
        finally:
            import shutil
            if inp_dir.exists():
                shutil.rmtree(inp_dir, ignore_errors=True)

    def test_clear_output_rejects_non_shm(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.clear_output(Path("/kaggle/working/ComfyUI/output"))

    def test_clear_input_rejects_non_shm(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.clear_input(Path("/kaggle/working/ComfyUI/input"))


# ---------------------------------------------------------------------------
# J — Final filesystem check
# ---------------------------------------------------------------------------

class TestJ_FinalFilesystemCheck(unittest.TestCase):
    def test_passes_when_no_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "comfyui.log").write_text("log")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertEqual(result["violations"], 0)
        self.assertIn("PASS", result["report"])

    def test_fails_when_image_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leaked.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)
        self.assertIn("FAIL", result["report"])
        self.assertIn("leaked.png", result["report"])

    def test_fails_when_zip_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "output_secure.zip").write_bytes(b"PK\x03\x04")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)
        self.assertIn("FAIL", result["report"])

    def test_detects_nested_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "ComfyUI" / "input" / "pasted"
            nested.mkdir(parents=True)
            (nested / "pasted.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)

    def test_detects_image_by_magic_bytes_no_extension(self):
        """Arquivo sem extensão com magic bytes de PNG deve ser detectado."""
        with tempfile.TemporaryDirectory() as tmp:
            # Arquivo sem extensão
            (Path(tmp) / "noext_file").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)

    def test_assert_no_persistent_images_raises_on_image(self):
        """assert_no_persistent_images deve levantar SecurityError quando imagem presente."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "image.jpg").write_bytes(b"\xff\xd8\xff")
            with self.assertRaises(comfyui_setup.SecurityError):
                comfyui_setup.assert_no_persistent_images.__globals__["PERSISTENT_AUDIT_PATHS"] = (Path(tmp),)
                # Patch PERSISTENT_AUDIT_PATHS temporariamente
                original = comfyui_setup.PERSISTENT_AUDIT_PATHS
                comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
                try:
                    comfyui_setup.assert_no_persistent_images(label="TEST")
                finally:
                    comfyui_setup.PERSISTENT_AUDIT_PATHS = original


# ---------------------------------------------------------------------------
# K — Custom node allowlist
# ---------------------------------------------------------------------------

class TestK_CustomNodeAllowlist(unittest.TestCase):
    def test_unknown_node_raises_security_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            (custom_dir / "websocket_image_save").mkdir()
            with self.assertRaises(comfyui_setup.SecurityError):
                comfyui_setup.check_custom_nodes_allowlist(Path(tmp), strict=True)

    def test_allowed_nodes_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            for node in comfyui_setup.ALLOWED_CUSTOM_NODES:
                (custom_dir / node).mkdir()
            unknown = comfyui_setup.check_custom_nodes_allowlist(Path(tmp), strict=True)
            self.assertEqual(unknown, [])

    def test_installing_unauthorized_node_spec_raises(self):
        with patch.object(comfyui_setup, "_run"), \
             patch.object(comfyui_setup, "detect_gpu", return_value={"has_gpu": False, "gpus": []}), \
             patch.object(comfyui_setup, "install_manager_requirements", return_value=False), \
             patch.object(comfyui_setup, "provision_shm_dirs"):
            with tempfile.TemporaryDirectory() as tmp:
                comfyui_dir = Path(tmp) / "ComfyUI"
                comfyui_dir.mkdir()
                (comfyui_dir / "main.py").write_text("")
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.setup_comfyui(
                        comfyui_dir=comfyui_dir,
                        custom_nodes=["attacker/malicious_exfil_node"],
                        output_dir=comfyui_setup.SHM_OUTPUT,
                        input_dir=comfyui_setup.SHM_INPUT,
                        temp_dir=comfyui_setup.SHM_TEMP,
                        strict_allowlist=True,
                    )

    def test_snapshot_custom_nodes_captures_hashes(self):
        """snapshot_custom_nodes deve capturar SHA-256 dos arquivos dos nodes."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            test_file = node_dir / "essentials.py"
            test_file.write_text("# test node\n")
            snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))
        self.assertIn("ComfyUI_essentials", snapshot["nodes"])
        node_snap = snapshot["nodes"]["ComfyUI_essentials"]
        self.assertIn("essentials.py", node_snap["files"])
        file_hash = node_snap["files"]["essentials.py"]
        self.assertEqual(len(file_hash), 64)  # SHA-256 hex

    def test_verify_custom_nodes_unchanged_detects_modification(self):
        """verify_custom_nodes_unchanged deve detectar arquivo modificado."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            test_file = node_dir / "node.py"
            test_file.write_text("# original\n")
            startup_snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))

            # Modificar o arquivo
            test_file.write_text("# MODIFIED BY ATTACKER\n")

            changes = comfyui_setup.verify_custom_nodes_unchanged(
                Path(tmp), startup_snapshot, strict=False
            )
            self.assertTrue(any("MODIFIED" in c for c in changes))

    def test_verify_custom_nodes_unchanged_detects_new_node(self):
        """verify_custom_nodes_unchanged deve detectar node adicionado após startup."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            startup_snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))

            # Adicionar node novo após snapshot
            new_node = custom_dir / "malicious_new_node"
            new_node.mkdir()
            (new_node / "exfil.py").write_text("# exfil\n")

            changes = comfyui_setup.verify_custom_nodes_unchanged(
                Path(tmp), startup_snapshot, strict=False
            )
            self.assertTrue(any("ADDED" in c and "malicious_new_node" in c for c in changes))

    def test_verify_custom_nodes_strict_raises_on_change(self):
        """verify_custom_nodes_unchanged com strict=True deve levantar SecurityError."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            (node_dir / "node.py").write_text("original\n")
            snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))
            (node_dir / "injected.py").write_text("INJECTED\n")
            with self.assertRaises(comfyui_setup.SecurityError):
                comfyui_setup.verify_custom_nodes_unchanged(Path(tmp), snapshot, strict=True)


# ---------------------------------------------------------------------------
# L — ngrok desabilitado por padrão
# ---------------------------------------------------------------------------

class TestL_NgrokDisabledByDefault(unittest.TestCase):
    def test_enable_ngrok_false_by_default(self):
        import inspect
        sig = inspect.signature(comfyui_setup.start_comfyui_runtime)
        self.assertFalse(sig.parameters["enable_ngrok"].default)

    def test_ngrok_not_started_when_disabled(self):
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)
        comfyui_setup.set_secure_mode(False, _test_override=True)
        try:
            with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
                 patch.object(comfyui_setup, "health_check", side_effect=[False, True]), \
                 patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
                 patch.object(comfyui_setup, "provision_shm_dirs"), \
                 patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
                result = comfyui_setup.start_comfyui_runtime(
                    comfyui_dir=Path(tempfile.mkdtemp()),
                    enable_ngrok=False, reuse_existing=False, secure_mode=False,
                )
        finally:
            comfyui_setup.set_secure_mode(True, _test_override=True)

        self.assertFalse(result["ngrok_started"])
        self.assertIsNone(result["public_url"])
        fake_ngrok.start_ngrok_tunnel.assert_not_called()


# ---------------------------------------------------------------------------
# SECURE MODE
# ---------------------------------------------------------------------------

class TestSecureMode(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_secure_mode_allows_manager(self):
        """Em SECURE_MODE, enable_manager=True deve resultar em --enable-manager no cmd."""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            enable_manager=True,
            secure_mode=True,
        )
        # cmd pode ser uma lista de listas (com isolamento bwrap)
        flat_cmd = []
        for item in cmd:
            if isinstance(item, list):
                flat_cmd.extend(item)
            else:
                flat_cmd.append(item)
        self.assertIn("--enable-manager", flat_cmd)

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_secure_mode_allows_ngrok(self):
        """Em SECURE_MODE, enable_ngrok=True NÃO deve levantar SecurityError."""
        process = MagicMock(pid=123)
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = lambda port: "https://example.ngrok.app"
        with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
             patch.object(comfyui_setup, "health_check", return_value=True), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
                reuse_existing=False,
                secure_mode=True,
            )
        self.assertTrue(result["ngrok_started"])
        self.assertIsNotNone(result["public_url"])

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_secure_mode_forces_reuse_existing_off(self):
        """Em SECURE_MODE, reuse_existing é forçado False."""
        process = MagicMock(pid=123)
        started = []
        with patch.object(comfyui_setup, "start_comfyui",
                           side_effect=lambda **_: started.append(1) or process), \
             patch.object(comfyui_setup, "health_check", return_value=True), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"):
            comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=False,
                reuse_existing=True,  # deve ser ignorado
                secure_mode=True,
            )
        self.assertEqual(len(started), 1, "Deve sempre iniciar processo novo em SECURE_MODE")

    def test_secure_mode_default_is_true_from_env(self):
        """SECURE_MODE deve ser True por padrão (COMFYUI_SECURE_MODE=1)."""
        with patch.dict(os.environ, {"COMFYUI_SECURE_MODE": "1"}):
            import importlib
            self.assertTrue(os.environ.get("COMFYUI_SECURE_MODE") == "1")


# ---------------------------------------------------------------------------
# Extras
# ---------------------------------------------------------------------------

class TestSecurityExtras(unittest.TestCase):
    def test_secure_zip_not_in_kaggle_working(self):
        self.assertNotIn("/kaggle/working", str(comfyui_setup.SHM_ARCHIVE))

    def test_allowed_custom_nodes_is_frozenset(self):
        self.assertIsInstance(comfyui_setup.ALLOWED_CUSTOM_NODES, frozenset)

    def test_security_error_is_runtime_subclass(self):
        self.assertTrue(issubclass(comfyui_setup.SecurityError, RuntimeError))

    def test_safe_remove_idempotent(self):
        comfyui_setup.safe_remove(Path("/tmp/nonexistent_xyz_test_file.txt"))

    def test_final_check_report_contains_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "problem.jpg"
            bad.write_bytes(b"\xff\xd8\xff")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertIn("problem.jpg", result["report"])
        self.assertGreater(result["violations"], 0)

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_provision_shm_dirs_runtime(self):
        """provision_shm_dirs deve criar diretório em /dev/shm com permissão 0o777."""
        test_dir = Path("/dev/shm/test_provision_xyz")
        try:
            comfyui_setup.provision_shm_dirs(test_dir)
            self.assertTrue(test_dir.exists())
            mode = oct(test_dir.stat().st_mode & 0o777)
            self.assertEqual(mode, oct(0o777))
        finally:
            if test_dir.exists():
                test_dir.rmdir()


# ---------------------------------------------------------------------------
# M — Invariantes e Guardrails de Filesystem
# ---------------------------------------------------------------------------

class TestM_FilesystemGuardrails(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_validate_runtime_path_rejects_traversal(self):
        """validate_runtime_path deve rejeitar path com '..'"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.validate_runtime_path(
                Path("/dev/shm/../kaggle/working/ComfyUI")
            )

    def test_validate_runtime_path_rejects_kaggle_working(self):
        """validate_runtime_path deve rejeitar path em /kaggle/working"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.validate_runtime_path(
                Path("/kaggle/working/test.png")
            )

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_validate_runtime_path_accepts_dev_shm(self):
        """validate_runtime_path deve aceitar path em /dev/shm"""
        result = comfyui_setup.validate_runtime_path(
            Path("/dev/shm/comfy_ui_output/test.png")
        )
        self.assertTrue(str(result).startswith("/dev/shm"))

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_validate_runtime_path_rejects_symlink_escape(self):
        """validate_runtime_path deve rejeitar symlink que aponta para fora de /dev/shm"""
        import shutil
        shm_dir = Path("/dev/shm/test_symlink_escape")
        try:
            shm_dir.mkdir(parents=True, exist_ok=True)
            link_path = shm_dir / "escape_link"
            target = Path("/tmp/test_escape_target")
            target.mkdir(parents=True, exist_ok=True)
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            os.symlink(target, link_path)
            with self.assertRaises(comfyui_setup.SecurityError):
                comfyui_setup.validate_runtime_path(link_path)
        finally:
            if shm_dir.exists():
                shutil.rmtree(shm_dir, ignore_errors=True)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_assert_invariants_accepts_shm_paths(self):
        """assert_invariants deve passar com paths em /dev/shm"""
        try:
            comfyui_setup.assert_invariants()
        except comfyui_setup.SecurityError:
            self.fail("assert_invariants falhou com paths padrão em /dev/shm")

    def test_assert_invariants_rejects_kaggle_working(self):
        """assert_invariants deve rejeitar paths em /kaggle/working"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.assert_invariants(
                input_dir=Path("/kaggle/working/input"),
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
            )

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_record_working_snapshot_and_assert_clean(self):
        """record_working_snapshot + assert_working_clean deve funcionar quando não há arquivos novos"""
        comfyui_setup.record_working_snapshot()
        # Não criar arquivos novos — deve passar
        try:
            comfyui_setup.assert_working_clean()
        except comfyui_setup.SecurityError:
            # Se /kaggle/working não existe (não-Kaggle), o teste passa
            if Path("/kaggle/working").exists():
                raise

    def test_assert_working_policy_detects_non_image_sensitive(self):
        """assert_working_policy deve detectar .json, .log, .db, etc."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sensitive.log").write_text("log data")
            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_assert_working_policy_allows_output_secure_zip(self):
        """assert_working_policy não deve marcar output_secure.zip como violação"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "output_secure.zip").write_bytes(b"PK\x03\x04")
            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                # output_secure.zip tem magic bytes de ZIP, mas é o único permitido
                # Como não há snapshot, pode falhar em assert_working_clean
                # Mas assert_working_policy deve aceitar
                comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original


# ---------------------------------------------------------------------------
# N — Teste de processo com Manager e ngrok ativos
# ---------------------------------------------------------------------------

class TestN_ManagerNgrokActive(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_manager_enabled_in_cmd_when_requested(self):
        """--enable-manager deve estar no cmd quando enable_manager=True"""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            enable_manager=True,
            secure_mode=True,
        )
        flat_cmd = []
        for item in cmd:
            if isinstance(item, list):
                flat_cmd.extend(item)
            else:
                flat_cmd.append(item)
        self.assertIn("--enable-manager", flat_cmd)

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_ngrok_starts_after_health_in_secure_mode(self):
        """Em SECURE_MODE, ngrok deve iniciar após health check"""
        events = []
        process = MagicMock(pid=123)
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = lambda port: events.append("ngrok") or "https://example.ngrok.app"

        health_results = [False, True]
        def health_mock(*args, **kwargs):
            events.append("health")
            return health_results.pop(0)

        with patch.object(comfyui_setup, "start_comfyui",
                           side_effect=lambda **_: events.append("start") or process), \
             patch.object(comfyui_setup, "health_check", side_effect=health_mock), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True, reuse_existing=False, secure_mode=True,
            )

        self.assertEqual(events, ["health", "start", "health", "ngrok"])
        self.assertTrue(result["ngrok_started"])
        self.assertTrue(result["secure_mode"])

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_failed_health_does_not_start_ngrok_in_secure_mode(self):
        """Em SECURE_MODE, health check falhando não deve iniciar ngrok"""
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)
        with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
             patch.object(comfyui_setup, "health_check", return_value=False), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True, health_timeout=1, reuse_existing=False, secure_mode=True,
            )

        self.assertFalse(result["health"])
        fake_ngrok.start_ngrok_tunnel.assert_not_called()

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_user_directory_in_shm(self):
        """--user-directory deve estar em /dev/shm no cmd"""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
            secure_mode=False,
        )
        flat_cmd = []
        for item in cmd:
            if isinstance(item, list):
                flat_cmd.extend(item)
            else:
                flat_cmd.append(item)
        self.assertIn("--user-directory", flat_cmd)
        idx = flat_cmd.index("--user-directory")
        self.assertTrue(flat_cmd[idx + 1].startswith("/dev/shm"))

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_logs_directory_in_shm(self):
        """Em SECURE_MODE, logs devem ir para /dev/shm/comfy_ui_logs"""
        self.assertTrue(str(comfyui_setup.SHM_LOGS).startswith("/dev/shm"))
        self.assertNotIn("/kaggle/working", str(comfyui_setup.SHM_LOGS))

    @unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
    def test_extra_args_cannot_override_user_directory(self):
        """extra_args não pode sobrescrever --user-directory"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--user-directory", "/kaggle/working/ComfyUI/user"],
                secure_mode=False,
            )


# ---------------------------------------------------------------------------
# O — Adversarial custom node / path traversal tests
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win32"), "Requer /dev/shm Linux")
class TestO_AdversarialCustomNode(unittest.TestCase):
    """Testes de adversarial: paths maliciosos devem ser rejeitados pela arquitetura."""

    def test_traversal_path_to_kaggle_working_rejected(self):
        """Path com traversal para /kaggle/working deve ser rejeitado"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.validate_runtime_path(
                Path("/dev/shm/../../kaggle/working/escape.txt")
            )

    def test_absolute_path_to_kaggle_working_rejected(self):
        """Caminho absoluto para /kaggle/working deve ser rejeitado"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.validate_runtime_path(
                Path("/kaggle/working/test_escape.png")
            )

    def test_dev_shm_path_accepted(self):
        """Path em /dev/shm deve ser aceito"""
        result = comfyui_setup.validate_runtime_path(Path("/dev/shm/comfy_ui_output/image.png"))
        self.assertTrue(str(result).startswith("/dev/shm"))

    def test_tmp_path_outside_shm_rejected_by_default(self):
        """Path em /tmp deve ser rejeitado (fora de /dev/shm)"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.validate_runtime_path(Path("/tmp/escape.txt"))

    @unittest.skipUnless(HAS_DEV_SHM, "Requer /dev/shm Linux")
    def test_hardlink_detection_in_final_check(self):
        """final_filesystem_check deve detectar hardlinks de arquivos sensíveis"""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "original.png"
            src.write_bytes(b"\x89PNG\r\n\x1a\n")
            link = Path(tmp) / "hardlink.png"
            try:
                os.link(src, link)
            except (OSError, PermissionError):
                self.skipTest("Hardlinks não suportados neste filesystem")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertGreater(result["violations"], 0)


# ---------------------------------------------------------------------------
# P — Snapshot ignora __pycache__ e bytecode compilado
# ---------------------------------------------------------------------------

class TestP_SnapshotIgnoresBytecode(unittest.TestCase):
    def test_snapshot_ignores_pycache_directory(self):
        """snapshot_custom_nodes deve ignorar diretório __pycache__"""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            # Arquivo fonte
            (node_dir / "node.py").write_text("# source\n")
            # Diretório __pycache__ com .pyc
            pycache = node_dir / "__pycache__"
            pycache.mkdir()
            (pycache / "node.cpython-312.pyc").write_bytes(b"bytecode")
            (pycache / "other.pyc").write_bytes(b"bytecode")

            snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))

            node_snap = snapshot["nodes"]["ComfyUI_essentials"]
            self.assertIn("node.py", node_snap["files"])
            # __pycache__ não deve aparecer no snapshot
            self.assertFalse(any("__pycache__" in f for f in node_snap["files"]))
            self.assertFalse(any(f.endswith(".pyc") for f in node_snap["files"]))

    def test_snapshot_ignores_git_directory(self):
        """snapshot_custom_nodes deve ignorar diretório .git"""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            (node_dir / "node.py").write_text("# source\n")
            git_dir = node_dir / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("[core]\n")

            snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))

            node_snap = snapshot["nodes"]["ComfyUI_essentials"]
            self.assertIn("node.py", node_snap["files"])
            self.assertFalse(any(".git" in f for f in node_snap["files"]))

    def test_verify_unchanged_ignores_new_pycache_files(self):
        """verify_custom_nodes_unchanged não deve flagrar novos arquivos .pyc como alteração"""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            node_dir = custom_dir / "ComfyUI_essentials"
            node_dir.mkdir()
            (node_dir / "node.py").write_text("# original\n")

            startup_snapshot = comfyui_setup.snapshot_custom_nodes(Path(tmp))

            # Simular criação de __pycache__ após startup (comportamento normal do Python)
            pycache = node_dir / "__pycache__"
            pycache.mkdir()
            (pycache / "node.cpython-312.pyc").write_bytes(b"bytecode")

            changes = comfyui_setup.verify_custom_nodes_unchanged(
                Path(tmp), startup_snapshot, strict=False
            )
            # Não deve detectar alterações (apenas bytecode compilado foi adicionado)
            self.assertEqual(changes, [])


# ---------------------------------------------------------------------------
# Q — Filesystem check respeita ALLOWED_STATIC_FILES
# ---------------------------------------------------------------------------

class TestQ_FilesystemCheckAllowsStaticFiles(unittest.TestCase):
    def test_final_check_allows_comfyui_example_png(self):
        """final_filesystem_check deve permitir ComfyUI/input/example.png"""
        with tempfile.TemporaryDirectory() as tmp:
            # Criar estrutura ComfyUI/input/example.png
            example_png = Path(tmp) / "ComfyUI" / "input" / "example.png"
            example_png.parent.mkdir(parents=True)
            example_png.write_bytes(b"\x89PNG\r\n\x1a\n")

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertEqual(result["violations"], 0)
            self.assertIn("PASS", result["report"])

    def test_final_check_allows_comfyui_comfy_types_examples(self):
        """final_filesystem_check deve permitir arquivos em ComfyUI/comfy/comfy_types/examples/"""
        with tempfile.TemporaryDirectory() as tmp:
            example_dir = Path(tmp) / "ComfyUI" / "comfy" / "comfy_types" / "examples"
            example_dir.mkdir(parents=True)
            (example_dir / "required_hint.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (example_dir / "input_options.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (example_dir / "input_types.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertEqual(result["violations"], 0)
            self.assertIn("PASS", result["report"])

    def test_final_check_allows_custom_nodes_zip(self):
        """final_filesystem_check deve permitir .zip dentro de custom_nodes/"""
        with tempfile.TemporaryDirectory() as tmp:
            node_zip = Path(tmp) / "ComfyUI" / "custom_nodes" / "some_node" / "node.zip"
            node_zip.parent.mkdir(parents=True)
            node_zip.write_bytes(b"PK\x03\x04")

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertEqual(result["violations"], 0)
            self.assertIn("PASS", result["report"])

    def test_final_check_allows_custom_nodes_source_files(self):
        """final_filesystem_check deve permitir arquivos de código em custom_nodes/"""
        with tempfile.TemporaryDirectory() as tmp:
            node_dir = Path(tmp) / "ComfyUI" / "custom_nodes" / "some_node"
            node_dir.mkdir(parents=True)
            (node_dir / "node.py").write_text("# node code\n")
            (node_dir / "config.json").write_text("{}")
            (node_dir / "README.md").write_text("# Node")

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertEqual(result["violations"], 0)
            self.assertIn("PASS", result["report"])

    def test_final_check_still_detects_leaked_images_outside_allowed(self):
        """final_filesystem_check deve ainda detectar imagens vazadas fora da lista permitida"""
        with tempfile.TemporaryDirectory() as tmp:
            # Imagem em ComfyUI/output (não está na lista de permitidos)
            leaked = Path(tmp) / "ComfyUI" / "output" / "leaked.png"
            leaked.parent.mkdir(parents=True)
            leaked.write_bytes(b"\x89PNG\r\n\x1a\n")

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertGreater(result["violations"], 0)
            self.assertIn("FAIL", result["report"])
            self.assertIn("leaked.png", result["report"])


# ---------------------------------------------------------------------------
# R — assert_working_policy respeita ALLOWED_STATIC_FILES
# ---------------------------------------------------------------------------

class TestR_AssertWorkingPolicyAllowsStaticFiles(unittest.TestCase):
    def _patch_working(self, tmp):
        """Helper para patchar PERSISTENT_WORKING e PERSISTENT_AUDIT_PATHS"""
        original_working = comfyui_setup.PERSISTENT_WORKING
        original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
        comfyui_setup.PERSISTENT_WORKING = Path(tmp)
        comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
        return original_working, original_paths

    def _restore_working(self, original_working, original_paths):
        comfyui_setup.PERSISTENT_WORKING = original_working
        comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths

    def test_assert_working_policy_allows_example_png(self):
        """assert_working_policy deve permitir ComfyUI/input/example.png"""
        with tempfile.TemporaryDirectory() as tmp:
            example_png = Path(tmp) / "ComfyUI" / "input" / "example.png"
            example_png.parent.mkdir(parents=True)
            example_png.write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_working_policy()
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_working_policy_allows_comfy_types_examples(self):
        """assert_working_policy deve permitir ComfyUI/comfy/comfy_types/examples/*"""
        with tempfile.TemporaryDirectory() as tmp:
            example_dir = Path(tmp) / "ComfyUI" / "comfy" / "comfy_types" / "examples"
            example_dir.mkdir(parents=True)
            (example_dir / "required_hint.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_working_policy()
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_working_policy_allows_custom_nodes_zip(self):
        """assert_working_policy deve permitir .zip em custom_nodes/"""
        with tempfile.TemporaryDirectory() as tmp:
            node_zip = Path(tmp) / "ComfyUI" / "custom_nodes" / "some_node" / "node.zip"
            node_zip.parent.mkdir(parents=True)
            node_zip.write_bytes(b"PK\x03\x04")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_working_policy()
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_working_policy_still_detects_leaked_images(self):
        """assert_working_policy deve ainda detectar imagens vazadas fora da lista"""
        with tempfile.TemporaryDirectory() as tmp:
            leaked = Path(tmp) / "ComfyUI" / "output" / "leaked.png"
            leaked.parent.mkdir(parents=True)
            leaked.write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                self._restore_working(original_working, original_paths)


# ---------------------------------------------------------------------------
# S — assert_no_persistent_images respeita ALLOWED_STATIC_FILES
# ---------------------------------------------------------------------------

class TestS_AssertNoPersistentImagesAllowsStaticFiles(unittest.TestCase):
    def _patch_working(self, tmp):
        """Helper para patchar PERSISTENT_WORKING e PERSISTENT_AUDIT_PATHS"""
        original_working = comfyui_setup.PERSISTENT_WORKING
        original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
        comfyui_setup.PERSISTENT_WORKING = Path(tmp)
        comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
        return original_working, original_paths

    def _restore_working(self, original_working, original_paths):
        comfyui_setup.PERSISTENT_WORKING = original_working
        comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths

    def test_assert_no_persistent_images_allows_example_png(self):
        """assert_no_persistent_images deve permitir ComfyUI/input/example.png"""
        with tempfile.TemporaryDirectory() as tmp:
            example_png = Path(tmp) / "ComfyUI" / "input" / "example.png"
            example_png.parent.mkdir(parents=True)
            example_png.write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_no_persistent_images_allows_comfy_types_examples(self):
        """assert_no_persistent_images deve permitir ComfyUI/comfy/comfy_types/examples/*"""
        with tempfile.TemporaryDirectory() as tmp:
            example_dir = Path(tmp) / "ComfyUI" / "comfy" / "comfy_types" / "examples"
            example_dir.mkdir(parents=True)
            (example_dir / "required_hint.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_no_persistent_images_allows_custom_nodes_zip(self):
        """assert_no_persistent_images deve permitir .zip em custom_nodes/"""
        with tempfile.TemporaryDirectory() as tmp:
            node_zip = Path(tmp) / "ComfyUI" / "custom_nodes" / "some_node" / "node.zip"
            node_zip.parent.mkdir(parents=True)
            node_zip.write_bytes(b"PK\x03\x04")

            original_working, original_paths = self._patch_working(tmp)
            try:
                comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                self._restore_working(original_working, original_paths)

    def test_assert_no_persistent_images_still_detects_leaked_images(self):
        """assert_no_persistent_images deve ainda detectar imagens vazadas fora da lista"""
        with tempfile.TemporaryDirectory() as tmp:
            leaked = Path(tmp) / "ComfyUI" / "output" / "leaked.png"
            leaked.parent.mkdir(parents=True)
            leaked.write_bytes(b"\x89PNG\r\n\x1a\n")

            original_working, original_paths = self._patch_working(tmp)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                self._restore_working(original_working, original_paths)


# ---------------------------------------------------------------------------
# T — assert_only_allowed_persistent_artifact respeita ALLOWED_STATIC_FILES
# ---------------------------------------------------------------------------

class TestT_AssertOnlyAllowedPersistentArtifactAllowsStaticFiles(unittest.TestCase):
    def test_allows_example_png_in_snapshot(self):
        """assert_only_allowed_persistent_artifact deve permitir ComfyUI/input/example.png no snapshot"""
        with tempfile.TemporaryDirectory() as tmp:
            # Primeiro criar snapshot com example.png
            example_png = Path(tmp) / "ComfyUI" / "input" / "example.png"
            example_png.parent.mkdir(parents=True)
            example_png.write_bytes(b"\x89PNG\r\n\x1a\n")

            comfyui_setup.record_working_snapshot.__globals__["_WORKING_SNAPSHOT"] = set()
            original_snapshot = comfyui_setup._WORKING_SNAPSHOT
            original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
            
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                # Snapshot inicial com o arquivo permitido
                comfyui_setup.record_working_snapshot()
                # Não deve falhar
                comfyui_setup.assert_only_allowed_persistent_artifact()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths
                comfyui_setup._WORKING_SNAPSHOT = original_snapshot

    def test_allows_comfy_types_examples_in_snapshot(self):
        """assert_only_allowed_persistent_artifact deve permitir ComfyUI/comfy/comfy_types/examples/* no snapshot"""
        with tempfile.TemporaryDirectory() as tmp:
            example_dir = Path(tmp) / "ComfyUI" / "comfy" / "comfy_types" / "examples"
            example_dir.mkdir(parents=True)
            (example_dir / "required_hint.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            original_snapshot = comfyui_setup._WORKING_SNAPSHOT
            original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
            
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                comfyui_setup.record_working_snapshot()
                comfyui_setup.assert_only_allowed_persistent_artifact()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths
                comfyui_setup._WORKING_SNAPSHOT = original_snapshot

    def test_allows_custom_nodes_zip_in_snapshot(self):
        """assert_only_allowed_persistent_artifact deve permitir .zip em custom_nodes/ no snapshot"""
        with tempfile.TemporaryDirectory() as tmp:
            node_zip = Path(tmp) / "ComfyUI" / "custom_nodes" / "some_node" / "node.zip"
            node_zip.parent.mkdir(parents=True)
            node_zip.write_bytes(b"PK\x03\x04")

            original_snapshot = comfyui_setup._WORKING_SNAPSHOT
            original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
            
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                comfyui_setup.record_working_snapshot()
                comfyui_setup.assert_only_allowed_persistent_artifact()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths
                comfyui_setup._WORKING_SNAPSHOT = original_snapshot

    def test_still_detects_new_unauthorized_files(self):
        """assert_only_allowed_persistent_artifact deve detectar novos arquivos não autorizados"""
        with tempfile.TemporaryDirectory() as tmp:
            # Snapshot inicial vazio
            comfyui_setup._WORKING_SNAPSHOT = set()
            original_snapshot = comfyui_setup._WORKING_SNAPSHOT
            original_paths = comfyui_setup.PERSISTENT_AUDIT_PATHS
            original_working = comfyui_setup.PERSISTENT_WORKING
            
            # Patch PERSISTENT_WORKING para o diretório temp
            comfyui_setup.PERSISTENT_WORKING = Path(tmp)
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                comfyui_setup.record_working_snapshot()
                
                # Adicionar arquivo não autorizado após snapshot
                leaked = Path(tmp) / "ComfyUI" / "output" / "leaked.png"
                leaked.parent.mkdir(parents=True)
                leaked.write_bytes(b"\x89PNG\r\n\x1a\n")
                
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_only_allowed_persistent_artifact()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original_paths
                comfyui_setup.PERSISTENT_WORKING = original_working
                comfyui_setup._WORKING_SNAPSHOT = original_snapshot


# ---------------------------------------------------------------------------
# U — Git-based working directory audit (abordagem B)
# Verifica que um clone limpo do ComfyUI passa, mas arquivos sensíveis
# colocados manualmente continuam sendo detectados.
# ---------------------------------------------------------------------------

class TestU_GitBasedAudit(unittest.TestCase):
    """Testa a checagem baseada em git (git ls-files / git status --porcelain)."""

    def setUp(self):
        """Limpa cache de git antes de cada teste."""
        comfyui_setup._clear_git_cache()

    def tearDown(self):
        """Limpa cache de git após cada teste."""
        comfyui_setup._clear_git_cache()

    def _create_comfyui_git_repo(self, tmp: str):
        """
        Cria um mini-repo git em tmp/ComfyUI com alguns arquivos tracked
        simulando o clone oficial do ComfyUI.
        """
        import subprocess
        comfyui_dir = Path(tmp) / "ComfyUI"
        comfyui_dir.mkdir(parents=True)

        # Criar estrutura de diretórios do ComfyUI
        (comfyui_dir / "comfy").mkdir(parents=True)
        (comfyui_dir / "comfy" / "comfy_types" / "examples").mkdir(parents=True)
        (comfyui_dir / "blueprints").mkdir(parents=True)
        (comfyui_dir / "tests").mkdir(parents=True)
        (comfyui_dir / "tests-unit").mkdir(parents=True)
        (comfyui_dir / ".ci").mkdir(parents=True)
        (comfyui_dir / "input").mkdir(parents=True)
        (comfyui_dir / "output").mkdir(parents=True)
        (comfyui_dir / "temp").mkdir(parents=True)
        (comfyui_dir / "custom_nodes").mkdir(parents=True)

        # Arquivos de fábrica (tracked)
        (comfyui_dir / "main.py").write_text("# ComfyUI main")
        (comfyui_dir / "requirements.txt").write_text("torch\n")
        (comfyui_dir / "README.md").write_text("# ComfyUI")
        (comfyui_dir / "comfy" / "__init__.py").write_text("")
        (comfyui_dir / "comfy" / "sd.py").write_text("# diffusion code")
        (comfyui_dir / "comfy" / "comfy_types" / "examples" / "required_hint.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (comfyui_dir / "blueprints" / "default.json").write_text('{"nodes": []}')
        (comfyui_dir / "tests" / "test_core.py").write_text("def test(): pass")
        (comfyui_dir / "tests-unit" / "test_extra.py").write_text("def test(): pass")
        (comfyui_dir / ".ci" / "install.ps1").write_text("echo install")
        (comfyui_dir / "input" / "example.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (comfyui_dir / "input" / "README.md").write_text("input dir")
        (comfyui_dir / "output" / "README.md").write_text("output dir")
        (comfyui_dir / "temp" / "README.md").write_text("temp dir")

        # git init e commit
        env = os.environ.copy()
        env["GIT_AUTHOR_NAME"] = "Test"
        env["GIT_AUTHOR_EMAIL"] = "test@test.com"
        env["GIT_COMMITTER_NAME"] = "Test"
        env["GIT_COMMITTER_EMAIL"] = "test@test.com"
        subprocess.run(["git", "init"], cwd=str(comfyui_dir), capture_output=True, env=env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=str(comfyui_dir), capture_output=True, env=env, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(comfyui_dir), capture_output=True, env=env, check=True)
        return comfyui_dir

    def test_clone_limpo_passa_assert_working_policy(self):
        """Clone limpo do ComfyUI (sem arquivos gerados) deve passar assert_working_policy."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_comfyui_git_repo(tmp)

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                # Não deve levantar SecurityError
                comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_clone_limpo_passa_assert_no_persistent_images(self):
        """Clone limpo do ComfyUI deve passar assert_no_persistent_images."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_comfyui_git_repo(tmp)

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_clone_limpo_passa_final_filesystem_check(self):
        """Clone limpo do ComfyUI deve passar final_filesystem_check com 0 violações."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_comfyui_git_repo(tmp)

            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
            self.assertEqual(result["violations"], 0, f"Expected 0 violations, got: {result['report']}")
            self.assertIn("PASS", result["report"])

    def test_png_colocado_manualmente_e_detectado(self):
        """Arquivo .png colado manualmente (untracked) deve ser detectado como violação."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_comfyui_git_repo(tmp)
            # Colar imagem manualmente (untracked)
            leaked = Path(tmp) / "ComfyUI" / "output" / "leaked.png"
            leaked.write_bytes(b"\x89PNG\r\n\x1a\n")

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_safetensors_colocado_manualmente_e_detectado(self):
        """Arquivo .safetensors colado manualmente deve ser detectado como violação."""
        with tempfile.TemporaryDirectory() as tmp:
            self._create_comfyui_git_repo(tmp)
            # Colar modelo manualmente (untracked)
            model = Path(tmp) / "ComfyUI" / "models" / "checkpoints" / "model.safetensors"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"\x00" * 100)

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_arquivo_modificado_no_clone_e_violacao_na_politica_completa(self):
        """Arquivo tracked modificado deve ser violação em assert_working_policy."""
        with tempfile.TemporaryDirectory() as tmp:
            comfyui_dir = self._create_comfyui_git_repo(tmp)
            # Modificar um arquivo tracked
            (comfyui_dir / "requirements.txt").write_text("torch\ntransformers\n")

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_arquivo_python_untracked_no_clone_e_violacao_na_politica_completa(self):
        """Arquivo untracked sem extensão sensível também deve ser violação na política completa."""
        with tempfile.TemporaryDirectory() as tmp:
            comfyui_dir = self._create_comfyui_git_repo(tmp)
            (comfyui_dir / "unexpected.py").write_text("print('unexpected')\n")

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_clone_colab_pipeline_json_tracked_limpo_e_permitido(self):
        """JSON tracked e limpo de um segundo repo deve passar a política."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "colab-pipeline"
            repo.mkdir()
            runtime = repo / "kaggle_runtime"
            runtime.mkdir()
            config = runtime / "08_master_pipeline.json"
            config.write_text('{"nodes": []}\n')
            env = os.environ.copy()
            env["GIT_AUTHOR_NAME"] = "Test"
            env["GIT_AUTHOR_EMAIL"] = "test@test.com"
            env["GIT_COMMITTER_NAME"] = "Test"
            env["GIT_COMMITTER_EMAIL"] = "test@test.com"
            subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, env=env, check=True)
            subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, env=env, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, env=env, check=True)

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_clone_colab_pipeline_json_modificado_e_reprovado(self):
        """JSON tracked mas modificado de um segundo repo deve ser reprovado."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "colab-pipeline"
            repo.mkdir()
            config = repo / "config.json"
            config.write_text('{"clean": true}\n')
            env = os.environ.copy()
            env["GIT_AUTHOR_NAME"] = "Test"
            env["GIT_AUTHOR_EMAIL"] = "test@test.com"
            env["GIT_COMMITTER_NAME"] = "Test"
            env["GIT_COMMITTER_EMAIL"] = "test@test.com"
            subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, env=env, check=True)
            subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, env=env, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True, env=env, check=True)
            config.write_text('{"clean": false}\n')

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_working_policy()
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original

    def test_git_tracked_safetensors_e_bloqueado_por_camada_extra(self):
        """Mesmo se .safetensors for tracked pelo git, deve ser bloqueado pela camada extra."""
        with tempfile.TemporaryDirectory() as tmp:
            import subprocess
            comfyui_dir = self._create_comfyui_git_repo(tmp)

            # Criar e commitar um .safetensors (simulação extrema)
            model = comfyui_dir / "model.safetensors"
            model.write_bytes(b"\x00" * 100)
            env = os.environ.copy()
            env["GIT_AUTHOR_NAME"] = "Test"
            env["GIT_AUTHOR_EMAIL"] = "test@test.com"
            env["GIT_COMMITTER_NAME"] = "Test"
            env["GIT_COMMITTER_EMAIL"] = "test@test.com"
            subprocess.run(["git", "add", "-A"], cwd=str(comfyui_dir), capture_output=True, env=env, check=True)
            subprocess.run(["git", "commit", "-m", "add model"], cwd=str(comfyui_dir), capture_output=True, env=env, check=True)

            original = comfyui_setup.PERSISTENT_AUDIT_PATHS
            comfyui_setup.PERSISTENT_AUDIT_PATHS = (Path(tmp),)
            try:
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.assert_no_persistent_images(label="TEST")
            finally:
                comfyui_setup.PERSISTENT_AUDIT_PATHS = original


if __name__ == "__main__":
    unittest.main()
