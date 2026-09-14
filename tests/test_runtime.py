"""
Testes de regressão de segurança — zero-trust / zero-persistent-image pipeline.

Cobre os 12 requisitos de segurança (A–L) além dos testes de contrato originais.

TESTES DE SEGURANÇA:
  A – Input directory sempre em /dev/shm
  B – Output directory sempre em /dev/shm
  C – Temp directory sempre em /dev/shm
  D – Servidor antigo com output em /kaggle/working NÃO pode ser reutilizado
  E – Upload/paste não cria arquivos em /kaggle/working
  F – Geração não cria imagens em /kaggle/working (output em /dev/shm)
  G – ZIP não é criado em /kaggle/working
  H – ZIP criado é criptografado (pyzipper AES-256, não ZIP_DEFLATED plaintext)
  I – Cleanup remove input/output/temp
  J – Filesystem final acusa FAIL se existir qualquer imagem em /kaggle/working
  K – Custom node não autorizado causa SecurityError
  L – ngrok NÃO inicia por padrão (enable_ngrok=False)
"""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import comfyui_setup
import gpu_detect
import ngrok_tunnel


# ---------------------------------------------------------------------------
# Testes de contrato originais (preservados)
# ---------------------------------------------------------------------------

class RuntimeContractTests(unittest.TestCase):
    def test_gpu_parser_enumerates_all_devices(self):
        result = gpu_detect.parse_nvidia_smi_csv(
            "Tesla T4, 15360 MiB, 535.104.05\nTesla T4, 15360 MiB, 535.104.05\n"
        )
        self.assertEqual([gpu["index"] for gpu in result], [0, 1])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["vram_mib"], 15360)
        self.assertEqual(result[1]["driver"], "535.104.05")

    def test_cuda_device_defaults_to_zero_and_can_change(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(comfyui_setup.get_cuda_device(), 0)
        with patch.dict(os.environ, {"COMFYUI_CUDA_DEVICE": "1"}):
            self.assertEqual(comfyui_setup.get_cuda_device(), 1)

    def test_default_nodes_exclude_manager_and_include_required_nodes(self):
        self.assertEqual(
            comfyui_setup.DEFAULT_CUSTOM_NODES,
            ["cubiq/ComfyUI_essentials", "lbouaraba/comfyui-krea2edit"],
        )
        self.assertEqual(
            comfyui_setup.filter_custom_nodes(["ltdrdata/ComfyUI-Manager"]), []
        )

    def test_manager_requirements_are_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            requirement = Path(tmp) / "manager_requirements.txt"
            requirement.write_text("requests\n", encoding="utf-8")
            with patch.object(comfyui_setup, "_run") as run:
                self.assertTrue(comfyui_setup.install_manager_requirements(Path(tmp)))
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][-2:], ["-r", str(requirement)])

    def test_krea2edit_install_and_update_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp)
            with patch.object(comfyui_setup, "_run") as run:
                node_path = comfyui_setup.install_or_update_custom_node(
                    custom_dir, "lbouaraba/comfyui-krea2edit"
                )
                self.assertEqual(node_path.name, "comfyui-krea2edit")
                self.assertIn("clone", run.call_args.args[0])

                node_path.mkdir(parents=True, exist_ok=True)
                (node_path / ".git").mkdir()

                run.reset_mock()
                comfyui_setup.install_or_update_custom_node(
                    custom_dir, "lbouaraba/comfyui-krea2edit"
                )
                self.assertEqual(run.call_args.args[0], ["git", "pull", "--ff-only"])

    def test_existing_non_git_node_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            node_path = Path(tmp) / "comfyui-krea2edit"
            node_path.mkdir()
            with self.assertRaises(RuntimeError):
                comfyui_setup.install_or_update_custom_node(
                    Path(tmp), "lbouaraba/comfyui-krea2edit"
                )

    def test_runtime_starts_ngrok_only_after_health(self):
        events = []
        process = MagicMock(pid=123)
        fake_ngrok = types.ModuleType("ngrok_tunnel")

        def start_tunnel(port):
            events.append("ngrok")
            return "https://example.ngrok.app"

        fake_ngrok.start_ngrok_tunnel = start_tunnel
        with patch.object(comfyui_setup, "start_comfyui",
                          side_effect=lambda **_: events.append("start") or process), \
             patch.object(comfyui_setup, "health_check",
                          side_effect=lambda *a, **kw: events.append("health") or True), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
                reuse_existing=False,
            )

        self.assertEqual(events, ["start", "health", "ngrok"])
        self.assertTrue(result["ngrok_started"])
        self.assertEqual(result["public_url"], "https://example.ngrok.app")

    def test_failed_health_does_not_start_ngrok(self):
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)
        with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
             patch.object(comfyui_setup, "health_check", return_value=False), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
                health_timeout=1,
                reuse_existing=False,
            )

        self.assertFalse(result["health"])
        fake_ngrok.start_ngrok_tunnel.assert_not_called()

    def test_ngrok_restarts_tunnel_without_logging_token(self):
        token = "secret-token-value"
        calls = []
        fake_ngrok_api = types.SimpleNamespace(
            set_auth_token=lambda value: calls.append(("auth", value)),
            kill=lambda: calls.append(("kill",)),
            connect=lambda **kwargs: calls.append(("connect", kwargs))
                or types.SimpleNamespace(public_url="http://public.ngrok.app"),
        )
        fake_pyngrok = types.ModuleType("pyngrok")
        fake_pyngrok.ngrok = fake_ngrok_api
        with patch.dict(sys.modules, {"pyngrok": fake_pyngrok}), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            url = ngrok_tunnel.start_ngrok_tunnel(authtoken=token)

        self.assertEqual(url, "https://public.ngrok.app")
        self.assertEqual([c[0] for c in calls], ["auth", "kill", "connect"])
        self.assertNotIn(token, stdout.getvalue())

    def test_notebooks_have_language_metadata_and_ids_for_existing_cells(self):
        for notebook in Path(__file__).parents[1].glob("**/*.ipynb"):
            document = json.loads(notebook.read_text(encoding="utf-8"))
            for cell in document["cells"]:
                self.assertIn("language", cell.get("metadata", {}), str(notebook))
                if cell.get("metadata", {}).get("language") in {"markdown", "python"}:
                    self.assertTrue(cell.get("metadata", {}).get("id"), str(notebook))


# ---------------------------------------------------------------------------
# TESTE A — Input directory sempre em /dev/shm
# ---------------------------------------------------------------------------
class TestA_InputDirectoryInShm(unittest.TestCase):
    def test_build_comfyui_command_includes_input_directory_in_shm(self):
        """build_comfyui_command deve incluir --input-directory apontando para /dev/shm/"""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
        )
        self.assertIn("--input-directory", cmd)
        idx = cmd.index("--input-directory")
        input_val = cmd[idx + 1]
        self.assertTrue(
            input_val.startswith("/dev/shm"),
            f"--input-directory deve estar em /dev/shm, mas é: {input_val}",
        )
        self.assertNotIn("/kaggle/working", input_val)

    def test_default_input_dir_is_shm(self):
        """O default de SHM_INPUT deve estar em /dev/shm"""
        self.assertTrue(str(comfyui_setup.SHM_INPUT).startswith("/dev/shm"))

    def test_assert_shm_path_rejects_kaggle_working(self):
        """assert_shm_path deve levantar SecurityError para /kaggle/working"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.assert_shm_path(Path("/kaggle/working/ComfyUI/input"), "test")

    def test_assert_shm_path_accepts_dev_shm(self):
        """assert_shm_path não deve levantar para /dev/shm"""
        # Não deve levantar
        comfyui_setup.assert_shm_path(Path("/dev/shm/comfy_ui_input"), "test")


# ---------------------------------------------------------------------------
# TESTE B — Output directory sempre em /dev/shm
# ---------------------------------------------------------------------------
class TestB_OutputDirectoryInShm(unittest.TestCase):
    def test_build_comfyui_command_output_in_shm(self):
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
        )
        self.assertIn("--output-directory", cmd)
        idx = cmd.index("--output-directory")
        val = cmd[idx + 1]
        self.assertTrue(val.startswith("/dev/shm"), f"--output-directory deve estar em /dev/shm: {val}")
        self.assertNotIn("/kaggle/working", val)

    def test_default_output_dir_is_shm(self):
        self.assertTrue(str(comfyui_setup.SHM_OUTPUT).startswith("/dev/shm"))

    def test_build_comfyui_command_rejects_kaggle_working_output(self):
        """build_comfyui_command deve levantar SecurityError se output_dir apontar para /kaggle/working"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                output_dir=Path("/kaggle/working/ComfyUI/output"),
                input_dir=comfyui_setup.SHM_INPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
            )


# ---------------------------------------------------------------------------
# TESTE C — Temp directory sempre em /dev/shm
# ---------------------------------------------------------------------------
class TestC_TempDirectoryInShm(unittest.TestCase):
    def test_build_comfyui_command_temp_dir_not_in_kaggle_working(self):
        """--temp-directory não deve apontar para /kaggle/working"""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
        )
        self.assertIn("--temp-directory", cmd)
        idx = cmd.index("--temp-directory")
        val = cmd[idx + 1]
        self.assertTrue(val.startswith("/dev/shm"), f"--temp-directory deve estar em /dev/shm: {val}")
        self.assertNotIn("/kaggle/working", val)

    def test_build_comfyui_command_rejects_kaggle_working_temp(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                output_dir=comfyui_setup.SHM_OUTPUT,
                input_dir=comfyui_setup.SHM_INPUT,
                temp_dir=Path("/kaggle/working/ComfyUI/temp"),
            )


# ---------------------------------------------------------------------------
# TESTE D — Processo antigo com output em /kaggle/working NÃO pode ser reutilizado
# ---------------------------------------------------------------------------
class TestD_ProcessReuseValidation(unittest.TestCase):
    def test_verify_process_paths_rejects_kaggle_working_output(self):
        """_verify_process_paths deve rejeitar processo com --output-directory em /kaggle/working"""
        cmdline = [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--input-directory", "/dev/shm/comfy_ui_input",
            "--output-directory", "/kaggle/working/ComfyUI/output",  # ERRADO
            "--temp-directory", "/dev/shm/comfy_ui_temp",
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, reason = comfyui_setup._verify_process_paths(
                pid=9999,
                expected_input=comfyui_setup.SHM_INPUT,
                expected_output=comfyui_setup.SHM_OUTPUT,
                expected_temp=comfyui_setup.SHM_TEMP,
                expected_host="127.0.0.1",
                expected_port=8188,
            )
        self.assertFalse(ok)
        self.assertIn("output-directory", reason.lower())

    def test_verify_process_paths_rejects_missing_input_directory(self):
        """_verify_process_paths deve rejeitar processo sem --input-directory"""
        cmdline = [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--output-directory", "/dev/shm/comfy_ui_output",
            "--temp-directory", "/dev/shm/comfy_ui_temp",
            # --input-directory ausente
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, reason = comfyui_setup._verify_process_paths(
                pid=9999,
                expected_input=comfyui_setup.SHM_INPUT,
                expected_output=comfyui_setup.SHM_OUTPUT,
                expected_temp=comfyui_setup.SHM_TEMP,
                expected_host="127.0.0.1",
                expected_port=8188,
            )
        self.assertFalse(ok)

    def test_verify_process_paths_accepts_correct_shm_paths(self):
        """_verify_process_paths deve aceitar processo com todos os paths corretos em /dev/shm"""
        cmdline = [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--input-directory", str(comfyui_setup.SHM_INPUT),
            "--output-directory", str(comfyui_setup.SHM_OUTPUT),
            "--temp-directory", str(comfyui_setup.SHM_TEMP),
        ]
        with patch.object(comfyui_setup, "_read_proc_cmdline", return_value=cmdline):
            ok, reason = comfyui_setup._verify_process_paths(
                pid=9999,
                expected_input=comfyui_setup.SHM_INPUT,
                expected_output=comfyui_setup.SHM_OUTPUT,
                expected_temp=comfyui_setup.SHM_TEMP,
                expected_host="127.0.0.1",
                expected_port=8188,
            )
        self.assertTrue(ok, reason)

    def test_reuse_existing_false_by_default(self):
        """start_comfyui_runtime deve ter reuse_existing=False como padrão"""
        import inspect
        sig = inspect.signature(comfyui_setup.start_comfyui_runtime)
        default = sig.parameters["reuse_existing"].default
        self.assertFalse(default, "reuse_existing deve ser False por padrão")

    def test_mismatched_process_is_killed_before_new_start(self):
        """Quando processo existente tem paths errados, deve ser morto antes de iniciar novo"""
        killed = []
        started = []

        bad_cmdline = [
            "python", "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--input-directory", "/kaggle/working/ComfyUI/input",  # ERRADO
            "--output-directory", "/kaggle/working/ComfyUI/output",
            "--temp-directory", "/kaggle/working/ComfyUI/temp",
        ]

        process = MagicMock(pid=1234)
        process.pid = 1234

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
                reuse_existing=True,
                enable_ngrok=False,
            )

        self.assertIn(9999, killed, "Processo com paths errados deve ser morto")
        self.assertEqual(len(started), 1, "Novo processo deve ser iniciado após matar o antigo")


# ---------------------------------------------------------------------------
# TESTE E — Upload/paste não cria arquivos em /kaggle/working
# ---------------------------------------------------------------------------
class TestE_PasteUploadNotInKaggleWorking(unittest.TestCase):
    def test_input_directory_arg_prevents_default_comfyui_input(self):
        """O comando deve conter --input-directory explícito, prevenindo o default /kaggle/working/ComfyUI/input"""
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
        )
        # --input-directory deve estar presente
        self.assertIn("--input-directory", cmd)
        # Seu valor não deve ser dentro de /kaggle/working
        idx = cmd.index("--input-directory")
        self.assertNotIn("/kaggle/working", cmd[idx + 1])

    def test_extra_args_cannot_override_input_directory(self):
        """extra_args não podem sobrescrever --input-directory (seria um bypass de segurança)"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--input-directory", "/kaggle/working/ComfyUI/input"],
            )


# ---------------------------------------------------------------------------
# TESTE F — Geração não cria imagens em /kaggle/working
# ---------------------------------------------------------------------------
class TestF_OutputNotInKaggleWorking(unittest.TestCase):
    def test_output_directory_not_in_kaggle_working(self):
        cmd = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            input_dir=comfyui_setup.SHM_INPUT,
            output_dir=comfyui_setup.SHM_OUTPUT,
            temp_dir=comfyui_setup.SHM_TEMP,
        )
        idx = cmd.index("--output-directory")
        self.assertNotIn("/kaggle/working", cmd[idx + 1])

    def test_extra_args_cannot_override_output_directory(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.build_comfyui_command(
                comfyui_dir=Path("/kaggle/working/ComfyUI"),
                input_dir=comfyui_setup.SHM_INPUT,
                output_dir=comfyui_setup.SHM_OUTPUT,
                temp_dir=comfyui_setup.SHM_TEMP,
                extra_args=["--output-directory", "/kaggle/working/ComfyUI/output"],
            )


# ---------------------------------------------------------------------------
# TESTE G — ZIP não é criado em /kaggle/working
# ---------------------------------------------------------------------------
class TestG_ZipNotInKaggleWorking(unittest.TestCase):
    def test_shm_archive_not_in_kaggle_working(self):
        self.assertNotIn("/kaggle/working", str(comfyui_setup.SHM_ARCHIVE))
        self.assertTrue(str(comfyui_setup.SHM_ARCHIVE).startswith("/dev/shm"))

    def test_create_secure_zip_aborts_if_archive_dir_not_in_shm(self):
        """create_secure_zip deve abortar se archive_dir não estiver em /dev/shm"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.create_secure_zip(
                src_dir=Path("/dev/shm/comfy_ui_output"),
                archive_dir=Path("/kaggle/working"),  # ERRADO
                zip_password="test",
            )

    def test_create_secure_zip_aborts_without_password(self):
        """create_secure_zip deve abortar se senha não estiver disponível"""
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            src = Path(tmp) / "output"
            src.mkdir()
            (src / "test.png").write_bytes(b"\x89PNG\r\n")

            with patch.dict(os.environ, {}, clear=True), \
                 patch("builtins.__import__", side_effect=lambda n, *a, **k:
                       (_ for _ in ()).throw(ImportError()) if n == "kaggle_secrets" else __import__(n, *a, **k)):
                with self.assertRaises(comfyui_setup.SecurityError):
                    comfyui_setup.create_secure_zip(
                        src_dir=src,
                        archive_dir=comfyui_setup.SHM_ARCHIVE,
                        zip_password=None,
                    )


# ---------------------------------------------------------------------------
# TESTE H — ZIP criado é criptografado (AES-256)
# ---------------------------------------------------------------------------
class TestH_ZipIsEncrypted(unittest.TestCase):
    def test_create_secure_zip_uses_pyzipper_aes(self):
        """create_secure_zip deve usar pyzipper.AESZipFile com WZ_AES, não zipfile.ZipFile plaintext"""
        import importlib

        # Verifica que create_secure_zip usa pyzipper, não zipfile
        import ast
        src = Path(comfyui_setup.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Buscar referência a pyzipper no AST
        pyzipper_refs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and "pyzipper" in str(node.value)
        ]
        self.assertTrue(len(pyzipper_refs) > 0, "create_secure_zip deve usar pyzipper")

        # Garantir que WZ_AES está presente no código
        self.assertIn("WZ_AES", src, "Criptografia deve usar WZ_AES (AES-256)")
        self.assertIn("AESZipFile", src, "Deve usar pyzipper.AESZipFile")

    def test_secure_zip_function_requires_password(self):
        """create_secure_zip sem senha deve sempre levantar SecurityError, nunca criar ZIP plaintext"""
        # Sem nenhuma fonte de senha
        with patch.dict(os.environ, {}, clear=True):
            try:
                import importlib
                # Mockar ausência do kaggle_secrets
                fake_ks = MagicMock()
                fake_ks.UserSecretsClient.return_value.get_secret.return_value = None
                with patch.dict(sys.modules, {"kaggle_secrets": fake_ks}):
                    with self.assertRaises(comfyui_setup.SecurityError):
                        comfyui_setup.create_secure_zip(
                            src_dir=Path("/dev/shm/comfy_ui_output"),
                            archive_dir=comfyui_setup.SHM_ARCHIVE,
                            zip_password=None,
                        )
            except Exception as e:
                # Se levantou SecurityError como esperado, ok
                if isinstance(e, comfyui_setup.SecurityError):
                    pass
                else:
                    raise


# ---------------------------------------------------------------------------
# TESTE I — Cleanup remove input/output/temp
# ---------------------------------------------------------------------------
class TestI_CleanupRemovesFiles(unittest.TestCase):
    def test_clear_output_removes_files_and_verifies(self):
        """clear_output deve remover todos os arquivos e verificar que o diretório está vazio"""
        with tempfile.TemporaryDirectory(prefix="/dev/shm/test_") as tmp:
            out_dir = Path(tmp) / "output"
            out_dir.mkdir()
            (out_dir / "image1.png").write_bytes(b"\x89PNG\r\n")
            (out_dir / "image2.webp").write_bytes(b"RIFF")

            comfyui_setup.clear_output(out_dir)

            remaining = list(out_dir.rglob("*")) if out_dir.exists() else []
            files = [f for f in remaining if f.is_file()]
            self.assertEqual(len(files), 0, f"clear_output deve remover todos os arquivos: {files}")

    def test_clear_input_removes_files_including_pasted_subdir(self):
        """clear_input deve remover imagens incluindo subdiretório pasted/"""
        with tempfile.TemporaryDirectory(prefix="/dev/shm/test_") as tmp:
            inp_dir = Path(tmp) / "input"
            pasted = inp_dir / "pasted"
            pasted.mkdir(parents=True)
            (pasted / "pasted_image.png").write_bytes(b"\x89PNG\r\n")
            (inp_dir / "uploaded.jpg").write_bytes(b"\xff\xd8\xff")

            comfyui_setup.clear_input(inp_dir)

            remaining = [f for f in inp_dir.rglob("*") if f.is_file()] if inp_dir.exists() else []
            self.assertEqual(len(remaining), 0, f"clear_input deve remover tudo: {remaining}")

    def test_clear_output_rejects_non_shm_directory(self):
        """clear_output deve rejeitar diretório fora de /dev/shm"""
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.clear_output(Path("/kaggle/working/ComfyUI/output"))

    def test_clear_input_rejects_non_shm_directory(self):
        with self.assertRaises(comfyui_setup.SecurityError):
            comfyui_setup.clear_input(Path("/kaggle/working/ComfyUI/input"))


# ---------------------------------------------------------------------------
# TESTE J — Filesystem final acusa FAIL se existir imagem em /kaggle/working
# ---------------------------------------------------------------------------
class TestJ_FinalFilesystemCheck(unittest.TestCase):
    def test_final_check_passes_when_no_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Apenas arquivos não-sensíveis
            (Path(tmp) / "comfyui.log").write_text("log content")
            (Path(tmp) / "scripts").mkdir()
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertEqual(result["violations"], 0)
        self.assertIn("STATUS: PASS", result["report"])

    def test_final_check_fails_when_image_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leaked_image.png").write_bytes(b"\x89PNG\r\n")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)
        self.assertIn("STATUS: FAIL", result["report"])
        self.assertIn("leaked_image.png", result["report"])

    def test_final_check_fails_when_zip_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "output_secure.zip").write_bytes(b"PK\x03\x04")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)
        self.assertIn("STATUS: FAIL", result["report"])

    def test_final_check_detects_nested_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "ComfyUI" / "input" / "pasted"
            nested.mkdir(parents=True)
            (nested / "pasted_image.webp").write_bytes(b"RIFF")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertGreater(result["violations"], 0)


# ---------------------------------------------------------------------------
# TESTE K — Custom node não autorizado causa SecurityError
# ---------------------------------------------------------------------------
class TestK_CustomNodeAllowlist(unittest.TestCase):
    def test_unknown_node_raises_security_error(self):
        """check_custom_nodes_allowlist com strict=True deve levantar para node desconhecido"""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            (custom_dir / "websocket_image_save").mkdir()  # Node não na allowlist

            with self.assertRaises(comfyui_setup.SecurityError):
                comfyui_setup.check_custom_nodes_allowlist(Path(tmp), strict=True)

    def test_allowed_nodes_pass_check(self):
        """Nodes na allowlist não devem causar erro"""
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom_nodes"
            custom_dir.mkdir()
            # Criar apenas nodes permitidos
            for node in comfyui_setup.ALLOWED_CUSTOM_NODES:
                (custom_dir / node).mkdir()

            unknown = comfyui_setup.check_custom_nodes_allowlist(Path(tmp), strict=True)
            self.assertEqual(unknown, [])

    def test_installing_unauthorized_node_spec_raises_security_error(self):
        """setup_comfyui deve rejeitar spec de custom node fora da allowlist"""
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
                        custom_nodes=["some-user/websocket_image_exfil"],
                        output_dir=comfyui_setup.SHM_OUTPUT,
                        input_dir=comfyui_setup.SHM_INPUT,
                        temp_dir=comfyui_setup.SHM_TEMP,
                        strict_allowlist=True,
                    )


# ---------------------------------------------------------------------------
# TESTE L — ngrok NÃO inicia por padrão
# ---------------------------------------------------------------------------
class TestL_NgrokDisabledByDefault(unittest.TestCase):
    def test_enable_ngrok_false_by_default(self):
        """start_comfyui_runtime deve ter enable_ngrok=False como padrão"""
        import inspect
        sig = inspect.signature(comfyui_setup.start_comfyui_runtime)
        default = sig.parameters["enable_ngrok"].default
        self.assertFalse(default, "enable_ngrok deve ser False por padrão")

    def test_ngrok_not_started_when_disabled(self):
        """ngrok não deve ser iniciado quando enable_ngrok=False"""
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)

        with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
             patch.object(comfyui_setup, "health_check", return_value=True), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=False,
                reuse_existing=False,
            )

        self.assertFalse(result["ngrok_started"])
        self.assertIsNone(result["public_url"])
        fake_ngrok.start_ngrok_tunnel.assert_not_called()

    def test_ngrok_with_enable_true_still_works(self):
        """ngrok deve funcionar quando explicitamente habilitado"""
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock(return_value="https://example.ngrok.app")
        process = MagicMock(pid=123)

        with patch.object(comfyui_setup, "start_comfyui", return_value=process), \
             patch.object(comfyui_setup, "health_check", return_value=True), \
             patch.object(comfyui_setup, "find_existing_comfyui_pid", return_value=None), \
             patch.object(comfyui_setup, "provision_shm_dirs"), \
             patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
                reuse_existing=False,
            )

        self.assertTrue(result["ngrok_started"])
        self.assertEqual(result["public_url"], "https://example.ngrok.app")


# ---------------------------------------------------------------------------
# Testes de integridade extra
# ---------------------------------------------------------------------------

class TestSecurityExtras(unittest.TestCase):
    def test_secure_zip_path_not_in_kaggle_working(self):
        """SHM_ARCHIVE não deve estar em /kaggle/working"""
        self.assertNotIn("/kaggle/working", str(comfyui_setup.SHM_ARCHIVE))

    def test_allowed_custom_nodes_is_frozenset(self):
        """ALLOWED_CUSTOM_NODES deve ser frozenset (imutável)"""
        self.assertIsInstance(comfyui_setup.ALLOWED_CUSTOM_NODES, frozenset)

    def test_security_error_is_runtime_error_subclass(self):
        """SecurityError deve ser subclasse de RuntimeError"""
        self.assertTrue(issubclass(comfyui_setup.SecurityError, RuntimeError))

    def test_safe_remove_is_idempotent(self):
        """safe_remove em arquivo inexistente não deve levantar"""
        comfyui_setup.safe_remove(Path("/tmp/nonexistent_test_file_xyz.txt"))

    def test_final_filesystem_check_report_contains_paths(self):
        """Relatório de FAIL deve conter o caminho do arquivo problemático"""
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "problem.jpg"
            bad_file.write_bytes(b"\xff\xd8\xff")
            result = comfyui_setup.final_filesystem_check(scan_root=Path(tmp), silent=True)
        self.assertIn("problem.jpg", result["report"])
        self.assertGreater(result["violations"], 0)


if __name__ == "__main__":
    unittest.main()
