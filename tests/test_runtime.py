import io
import json
import os
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

    def test_start_command_is_local_dynamic_vram_and_manager_enabled(self):
        command = comfyui_setup.build_comfyui_command(
            comfyui_dir=Path("/kaggle/working/ComfyUI"),
            output_dir=Path("/kaggle/working/ComfyUI/output"),
        )

        self.assertIn("--listen", command)
        self.assertEqual(command[command.index("--listen") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--cuda-device") + 1], "0")
        self.assertIn("--enable-manager", command)
        for flag in comfyui_setup.DISCOURAGED_VRAM_FLAGS:
            self.assertNotIn(flag, command)
        self.assertEqual(
            command[command.index("--output-directory") + 1],
            "/kaggle/working/ComfyUI/output",
        )

    def test_runtime_starts_ngrok_only_after_health(self):
        events = []
        process = MagicMock(pid=123)
        fake_ngrok = types.ModuleType("ngrok_tunnel")

        def start_tunnel(port):
            events.append("ngrok")
            return "https://example.ngrok.app"

        fake_ngrok.start_ngrok_tunnel = start_tunnel
        with patch.object(comfyui_setup, "start_comfyui", side_effect=lambda **_: events.append("start") or process), patch.object(
            comfyui_setup, "health_check", side_effect=lambda *args, **kwargs: events.append("health") or True
        ), patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
            )

        self.assertEqual(events, ["start", "health", "ngrok"])
        self.assertTrue(result["ngrok_started"])
        self.assertEqual(result["public_url"], "https://example.ngrok.app")

    def test_failed_health_does_not_start_ngrok(self):
        fake_ngrok = types.ModuleType("ngrok_tunnel")
        fake_ngrok.start_ngrok_tunnel = MagicMock()
        process = MagicMock(pid=123)
        with patch.object(comfyui_setup, "start_comfyui", return_value=process), patch.object(
            comfyui_setup, "health_check", return_value=False
        ), patch.dict(sys.modules, {"ngrok_tunnel": fake_ngrok}):
            result = comfyui_setup.start_comfyui_runtime(
                comfyui_dir=Path(tempfile.mkdtemp()),
                enable_ngrok=True,
                health_timeout=1,
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
        with patch.dict(sys.modules, {"pyngrok": fake_pyngrok}), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as stdout:
            url = ngrok_tunnel.start_ngrok_tunnel(authtoken=token)

        self.assertEqual(url, "https://public.ngrok.app")
        self.assertEqual([call[0] for call in calls], ["auth", "kill", "connect"])
        self.assertEqual(calls[-1][1]["addr"], "127.0.0.1:8188")
        self.assertNotIn(token, stdout.getvalue())

    def test_notebooks_have_language_metadata_and_ids_for_existing_cells(self):
        for notebook in Path(__file__).parents[1].glob("**/*.ipynb"):
            document = json.loads(notebook.read_text(encoding="utf-8"))
            for cell in document["cells"]:
                self.assertIn("language", cell.get("metadata", {}), str(notebook))
                if cell.get("metadata", {}).get("language") in {"markdown", "python"}:
                    self.assertTrue(cell.get("metadata", {}).get("id"), str(notebook))


if __name__ == "__main__":
    unittest.main()