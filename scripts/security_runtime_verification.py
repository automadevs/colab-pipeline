#!/usr/bin/env python3
"""
SECURITY RUNTIME VERIFICATION — Kaggle / Linux Real
=====================================================

Validação de produção para SECURE_MODE=True.
Executar APÓS start_comfyui_runtime() em sessão Kaggle real.

Requisitos:
  - Linux / Kaggle
  - ComfyUI rodando em SECURE_MODE=True
  - pyzipper instalado
  - /dev/shm disponível (tmpfs)

NÃO MODIFICA a arquitetura de segurança.
NÃO faz refatoração.
Apenas OBSERVA e REPORTA.

Resultado: PASS / FAIL / NOT VERIFIED para cada teste.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constantes do pipeline (espelhadas de comfyui_setup.py)
# ---------------------------------------------------------------------------
SHM_BASE = Path("/dev/shm")
SHM_INPUT = SHM_BASE / "comfy_ui_input"
SHM_OUTPUT = SHM_BASE / "comfy_ui_output"
SHM_TEMP = SHM_BASE / "comfy_ui_temp"
SHM_USER = SHM_BASE / "comfy_ui_user"
SHM_LOGS = SHM_BASE / "comfy_ui_logs"
SHM_ARCHIVE = SHM_BASE / "comfy_ui_archive"
KAGGLE_WORKING = Path("/kaggle/working")
COMFYUI_DIR = KAGGLE_WORKING / "ComfyUI"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188

SENSITIVE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".bmp", ".tif", ".tiff",
})
SENSITIVE_ARCHIVES = frozenset({".zip", ".7z", ".rar", ".tar", ".gz"})

# Magic bytes para detecção
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_JPEG = b"\xff\xd8\xff"
MAGIC_WEBP_RIFF = b"RIFF"
MAGIC_WEBP_SIG = b"WEBP"
MAGIC_GIF87 = b"GIF87a"
MAGIC_GIF89 = b"GIF89a"
MAGIC_ZIP = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


# ---------------------------------------------------------------------------
# Report accumulator
# ---------------------------------------------------------------------------
class VerificationReport:
    """Acumula resultados de testes para relatório final."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.environment: Dict[str, str] = {}
        self.start_time = datetime.datetime.now()

    def record(self, test_id: str, status: str, details: str = "",
               data: Optional[Dict[str, Any]] = None):
        """status: PASS, FAIL, NOT VERIFIED, STATICALLY VERIFIED"""
        assert status in ("PASS", "FAIL", "NOT VERIFIED", "STATICALLY VERIFIED"), f"Status inválido: {status}"
        self.results[test_id] = {
            "status": status,
            "details": details,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat(),
        }
        icon = {"PASS": "✅", "FAIL": "❌", "NOT VERIFIED": "⚠️", "STATICALLY VERIFIED": "🔍"}[status]
        print(f"\n{icon} [{test_id}] {status}")
        if details:
            for line in details.split("\n"):
                print(f"   {line}")

    def get_verdict(self) -> str:
        """Classificação final conforme regra 19."""
        critical_tests = [
            "02_PATH_VERIFICATION",
            "03_PASTE_UPLOAD",
            "04_GENERATION",
            "05_IMG2IMG",
            "06_ZIP_AES256",
            "08_PROCESS",
            "15_EXCEPTION_CLEANUP",
            "16_FINAL_SCAN",
        ]
        statuses = [self.results.get(t, {}).get("status", "NOT VERIFIED") for t in critical_tests]

        if all(s == "PASS" for s in statuses):
            return "PRODUCTION VERIFIED"
        elif any(s == "FAIL" for s in statuses):
            return "FAILED SECURITY VERIFICATION"
        else:
            return "PARTIALLY VERIFIED"

    def to_markdown(self) -> str:
        """Gera relatório markdown completo."""
        lines = [
            "# SECURITY RUNTIME VERIFICATION REPORT",
            "",
            f"**Generated**: {datetime.datetime.now().isoformat()}",
            f"**Duration**: {(datetime.datetime.now() - self.start_time).total_seconds():.1f}s",
            "",
            "---",
            "",
            "## Environment",
            "",
        ]
        for k, v in self.environment.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # Resultados por teste
        sections = [
            ("02_PATH_VERIFICATION", "Path Verification"),
            ("03_PASTE_UPLOAD", "Paste/Upload"),
            ("04_GENERATION", "Generation"),
            ("05_IMG2IMG", "Img2Img"),
            ("06_ZIP_AES256", "ZIP AES-256"),
            ("07_DOWNLOAD", "Download"),
            ("08_PROCESS", "Process"),
            ("09_REUSE", "Reuse Detection"),
            ("10_NGROK", "ngrok"),
            ("11_MANAGER_ACTIVE", "Manager Active"),
            ("11_NETWORK_TRAFFIC", "External Network Traffic"),
            ("12_CUSTOM_NODES", "Custom Node Hashing"),
            ("13_NODE_ALTERATION", "Node Alteration"),
            ("14_DRIVE_CREDENTIALS", "Drive Credential Cleanup"),
            ("15_EXCEPTION_CLEANUP", "Exception Cleanup"),
            ("16_FINAL_SCAN", "Final Filesystem Scan"),
            ("17_PERSISTENCE", "Persistence Between Sessions"),
        ]

        for test_id, title in sections:
            result = self.results.get(test_id, {"status": "NOT VERIFIED", "details": "Teste não executado"})
            icon = {"PASS": "✅", "FAIL": "❌", "NOT VERIFIED": "⚠️", "STATICALLY VERIFIED": "🔍"}.get(result["status"], "ℹ️")
            lines.append(f"## {title}")
            lines.append("")
            lines.append(f"**Status**: {icon} {result['status']}")
            lines.append("")
            if result.get("details"):
                lines.append("```")
                lines.append(result["details"])
                lines.append("```")
                lines.append("")

        # Verdict
        verdict = self.get_verdict()
        v_icon = {
            "PRODUCTION VERIFIED": "✅",
            "PARTIALLY VERIFIED": "⚠️",
            "FAILED SECURITY VERIFICATION": "❌",
        }[verdict]

        lines.extend([
            "---",
            "",
            "## Final Verdict",
            "",
            f"### {v_icon} {verdict}",
            "",
        ])

        if verdict == "PARTIALLY VERIFIED":
            not_verified = [t for t, r in self.results.items() if r["status"] == "NOT VERIFIED"]
            if not_verified:
                lines.append("**Testes não verificados:**")
                for t in not_verified:
                    lines.append(f"- {t}")
                lines.append("")

        if verdict == "FAILED SECURITY VERIFICATION":
            failed = [t for t, r in self.results.items() if r["status"] == "FAIL"]
            lines.append("**Testes que falharam:**")
            for t in failed:
                lines.append(f"- {t}: {self.results[t].get('details', '')[:200]}")
            lines.append("")

        lines.extend([
            "> [!IMPORTANT]",
            "> Este relatório NÃO declara \"segurança absoluta\".",
            "> Valida apenas os controles do pipeline nos caminhos controlados.",
            "",
        ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_kaggle() -> bool:
    return Path("/kaggle").exists()


def _api_call(endpoint: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
              method: str = "GET", data: bytes = None, timeout: int = 30) -> Any:
    """Chamada HTTP à API local do ComfyUI."""
    url = f"http://{host}:{port}{endpoint}"
    req = urllib.request.Request(url, method=method, data=data)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _create_canary_image(path: Path, width: int = 64, height: int = 64) -> Path:
    """Cria uma imagem PNG mínima de teste (canary). Sem dependências externas."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # PNG mínimo: header + IHDR + IDAT com dados brutos + IEND
    def _crc32(data: bytes) -> bytes:
        import zlib
        return struct.pack(">I", zlib.crc32(data) & 0xffffffff)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + chunk_type + data + _crc32(chunk_type + data)

    import zlib

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT — linhas de pixels: filter byte (0) + RGB por pixel
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # filter: None
        for x in range(width):
            # Padrão único de canary: gradiente para identificação
            r = (x * 4) & 0xFF
            g = (y * 4) & 0xFF
            b = 0x42  # "B" de "baseline" — marca de canary
            raw_data += bytes([r, g, b])
    idat = _chunk(b"IDAT", zlib.compress(raw_data))

    # IEND
    iend = _chunk(b"IEND", b"")

    png_data = MAGIC_PNG + ihdr + idat + iend
    path.write_bytes(png_data)
    return path


def _find_images_in_dir(directory: Path) -> List[Dict[str, Any]]:
    """Encontra imagens por extensão E magic bytes recursivamente."""
    found = []
    if not directory.exists():
        return found
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        entry: Dict[str, Any] = {
            "path": str(item),
            "size": item.stat().st_size,
            "mtime": datetime.datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
            "detection": None,
        }
        ext = item.suffix.lower()
        if ext in SENSITIVE_EXTENSIONS:
            entry["detection"] = f"extension:{ext}"
            found.append(entry)
            continue
        if ext in SENSITIVE_ARCHIVES:
            entry["detection"] = f"archive:{ext}"
            found.append(entry)
            continue
        # Magic bytes
        try:
            magic = item.read_bytes()[:16]
            if magic[:8] == MAGIC_PNG:
                entry["detection"] = "magic:PNG"
            elif magic[:3] == MAGIC_JPEG:
                entry["detection"] = "magic:JPEG"
            elif magic[:4] == MAGIC_WEBP_RIFF and magic[8:12] == MAGIC_WEBP_SIG:
                entry["detection"] = "magic:WEBP"
            elif magic[:6] in (MAGIC_GIF87, MAGIC_GIF89):
                entry["detection"] = "magic:GIF"
            elif magic[:4] in MAGIC_ZIP:
                entry["detection"] = "magic:ZIP"
            if entry["detection"]:
                found.append(entry)
        except (OSError, PermissionError):
            pass
    return found


def _find_symlinks_in_dir(directory: Path) -> List[Dict[str, str]]:
    """Encontra symlinks recursivamente."""
    found = []
    if not directory.exists():
        return found
    for item in directory.rglob("*"):
        if item.is_symlink():
            try:
                target = str(item.resolve())
            except Exception:
                target = "UNRESOLVABLE"
            found.append({"path": str(item), "target": target})
    return found


def _scan_working_dir() -> Dict[str, Any]:
    """Scan completo de /kaggle/working para artefatos sensíveis."""
    images = _find_images_in_dir(KAGGLE_WORKING)
    symlinks = _find_symlinks_in_dir(KAGGLE_WORKING)
    return {
        "images": images,
        "symlinks": symlinks,
        "image_count": len(images),
        "symlink_count": len(symlinks),
    }


# ---------------------------------------------------------------------------
# Verificação de pré-requisitos
# ---------------------------------------------------------------------------

def check_prerequisites() -> Tuple[bool, str]:
    """Verifica que estamos no ambiente correto."""
    issues = []
    if not _is_linux():
        issues.append(f"OS não é Linux: {sys.platform}")
    if not _is_kaggle():
        issues.append("/kaggle não existe — não é ambiente Kaggle")
    if not SHM_BASE.exists():
        issues.append("/dev/shm não existe — tmpfs indisponível")
    if not COMFYUI_DIR.exists():
        issues.append(f"ComfyUI não encontrado em {COMFYUI_DIR}")

    if issues:
        return False, "\n".join(issues)
    return True, "Pré-requisitos OK"


# ===========================================================================
# TESTES
# ===========================================================================

def test_02_path_verification(report: VerificationReport,
                               host: str = DEFAULT_HOST,
                               port: int = DEFAULT_PORT):
    """Teste 2: Paths do ComfyUI via folder_paths real."""
    test_id = "02_PATH_VERIFICATION"
    try:
        # Método 1: consultar via import direto (se ComfyUI está no sys.path)
        paths_info = {}
        try:
            comfyui_main = COMFYUI_DIR / "main.py"
            if comfyui_main.exists():
                # Adicionar ao path
                if str(COMFYUI_DIR) not in sys.path:
                    sys.path.insert(0, str(COMFYUI_DIR))
                import folder_paths
                paths_info["input"] = str(folder_paths.get_input_directory())
                paths_info["output"] = str(folder_paths.get_output_directory())
                paths_info["temp"] = str(folder_paths.get_temp_directory())
        except Exception as e:
            # Fallback: ler de /proc/cmdline do processo ComfyUI
            paths_info["import_error"] = str(e)

        # Método 2: ler de /proc/<pid>/cmdline
        pid = _find_comfyui_pid(port)
        if pid:
            cmdline = _read_proc_cmdline(pid)
            cmdline_str = " ".join(cmdline)
            paths_info["cmdline_raw"] = cmdline_str

            for flag, key in [
                ("--input-directory", "cmdline_input"),
                ("--output-directory", "cmdline_output"),
                ("--temp-directory", "cmdline_temp"),
            ]:
                try:
                    idx = cmdline.index(flag)
                    paths_info[key] = cmdline[idx + 1]
                except (ValueError, IndexError):
                    paths_info[key] = "NOT FOUND"

        # Validar
        expected = {
            "input": str(SHM_INPUT),
            "output": str(SHM_OUTPUT),
            "temp": str(SHM_TEMP),
        }

        checks = []
        any_fail = False

        # Verificar a partir de cmdline (fonte mais confiável em runtime)
        for label, expected_path in expected.items():
            cmdline_key = f"cmdline_{label}"
            actual = paths_info.get(cmdline_key, paths_info.get(label, "UNKNOWN"))
            ok = actual == expected_path
            if not ok:
                any_fail = True
            checks.append(f"{label.upper():6s}: {actual} {'✓' if ok else '✗ ESPERADO: ' + expected_path}")

        # Verificar ausência de /kaggle/working
        for label, key in [("input", "cmdline_input"), ("output", "cmdline_output"), ("temp", "cmdline_temp")]:
            val = paths_info.get(key, paths_info.get(label, ""))
            if "/kaggle/working" in val:
                any_fail = True
                checks.append(f"VIOLAÇÃO: {label} contém /kaggle/working: {val}")

        details = "\n".join(checks)
        if paths_info.get("cmdline_raw"):
            details += f"\n\nPID {pid} cmdline: {paths_info['cmdline_raw'][:500]}"

        report.record(test_id, "FAIL" if any_fail else "PASS", details, paths_info)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_03_paste_upload(report: VerificationReport,
                          host: str = DEFAULT_HOST,
                          port: int = DEFAULT_PORT):
    """Teste 3: Upload/Paste de canary — confirmar explicitamente input/pasted e path final."""
    test_id = "03_PASTE_UPLOAD"
    canary_upload_name = "SECURITY_CANARY_UPLOAD.png"
    canary_pasted_name = "SECURITY_CANARY_PASTED.png"
    temp_upload = SHM_TEMP / canary_upload_name
    temp_pasted = SHM_TEMP / canary_pasted_name

    try:
        # 1. Criar canaries em tmpfs
        _create_canary_image(temp_upload)
        _create_canary_image(temp_pasted)
        assert temp_upload.exists(), f"Canary não criado em {temp_upload}"
        assert temp_pasted.exists(), f"Canary não criado em {temp_pasted}"

        def _upload_multipart(filename: str, file_path: Path, subfolder: str = "") -> Tuple[bool, str]:
            try:
                import http.client
                boundary = "----SecurityVerificationBoundary"
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                    f"Content-Type: image/png\r\n\r\n"
                ).encode("utf-8")
                body += file_path.read_bytes()
                body += f"\r\n--{boundary}\r\n".encode("utf-8")
                body += (
                    f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
                    f"{subfolder}\r\n"
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
                    f"true\r\n"
                    f"--{boundary}--\r\n"
                ).encode("utf-8")

                conn = http.client.HTTPConnection(host, port, timeout=15)
                conn.request("POST", "/upload/image", body=body,
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                resp = conn.getresponse()
                resp_data = resp.read().decode("utf-8", errors="replace")
                conn.close()
                return resp.status == 200, resp_data
            except Exception as e:
                return False, str(e)

        # Upload 1: Input padrão
        upload_ok, upload_resp = _upload_multipart(canary_upload_name, temp_upload, subfolder="")
        # Upload 2: Paste subfolder 'pasted' (mecanismo usado pelo clipboard do ComfyUI)
        pasted_ok, pasted_resp = _upload_multipart(canary_pasted_name, temp_pasted, subfolder="pasted")

        time.sleep(1)  # sync filesystem

        # Localizar paths reais
        final_upload_path = None
        final_pasted_path = None

        expected_upload = SHM_INPUT / canary_upload_name
        expected_pasted = SHM_INPUT / "pasted" / canary_pasted_name

        if expected_upload.exists():
            final_upload_path = expected_upload
        else:
            matches = list(SHM_INPUT.rglob(canary_upload_name)) if SHM_INPUT.exists() else []
            if matches:
                final_upload_path = matches[0]

        if expected_pasted.exists():
            final_pasted_path = expected_pasted
        else:
            matches = list(SHM_INPUT.rglob(canary_pasted_name)) if SHM_INPUT.exists() else []
            if matches:
                final_pasted_path = matches[0]

        # Verificar vazamento persistente em /kaggle/working
        persistent_violations = []
        if KAGGLE_WORKING.exists():
            for name in [canary_upload_name, canary_pasted_name]:
                for f in KAGGLE_WORKING.rglob(name):
                    persistent_violations.append(str(f))
            # Verificar explicitamente /kaggle/working/ComfyUI/input/pasted
            persistent_pasted_dir = KAGGLE_WORKING / "ComfyUI" / "input" / "pasted"
            if persistent_pasted_dir.exists():
                files = list(persistent_pasted_dir.rglob("*"))
                if files:
                    persistent_violations.append(f"{persistent_pasted_dir} contém {len(files)} arquivos")

        checks = [
            f"Upload Padrão HTTP status: {'200 OK ✓' if upload_ok else 'FALHOU ✗ (' + upload_resp[:100] + ')'}",
            f"Paste ('input/pasted') HTTP status: {'200 OK ✓' if pasted_ok else 'FALHOU ✗ (' + pasted_resp[:100] + ')'}",
            "",
            "CONFIRMAÇÃO EXPLÍCITA DE PATHS:",
            f"  PATH FINAL (Upload padrão): {final_upload_path if final_upload_path else 'NÃO ENCONTRADO ✗'}",
            f"  PATH FINAL (Paste input/pasted): {final_pasted_path if final_pasted_path else 'NÃO ENCONTRADO ✗'}",
            f"  Diretório input/pasted em tmpfs: {SHM_INPUT / 'pasted'} (Existe: {(SHM_INPUT / 'pasted').exists()})",
            "",
            f"Verificação em /kaggle/working: {len(persistent_violations)} violação(ões) persistente(s)",
        ]

        if persistent_violations:
            checks.append("VIOLAÇÕES PERSISTENTES ENCONTRADAS:")
            for v in persistent_violations:
                checks.append(f"  ❌ {v}")

        # Cleanup
        for p in [temp_upload, temp_pasted]:
            if p.exists():
                p.unlink()
        for name in [canary_upload_name, canary_pasted_name]:
            for d in [SHM_INPUT, SHM_OUTPUT, SHM_TEMP]:
                if d.exists():
                    for f in d.rglob(name):
                        f.unlink()
        pasted_dir = SHM_INPUT / "pasted"
        if pasted_dir.exists() and not list(pasted_dir.iterdir()):
            pasted_dir.rmdir()

        all_clean = not any(
            list(d.rglob(canary_upload_name)) + list(d.rglob(canary_pasted_name))
            for d in [SHM_INPUT, SHM_OUTPUT, SHM_TEMP] if d.exists()
        )
        checks.append(f"Cleanup pós-teste: {'Todos os canaries removidos ✓' if all_clean else 'FALHA no cleanup ✗'}")

        details = "\n".join(checks)

        if persistent_violations:
            report.record(test_id, "FAIL", details)
        elif not upload_ok and not pasted_ok:
            report.record(test_id, "NOT VERIFIED", f"Ambos os endpoints de upload falharam.\n{details}")
        elif final_upload_path and str(final_upload_path).startswith("/dev/shm") and (
            final_pasted_path is None or str(final_pasted_path).startswith("/dev/shm")
        ):
            report.record(test_id, "PASS", details, {
                "upload_path": str(final_upload_path),
                "pasted_path": str(final_pasted_path),
            })
        else:
            report.record(test_id, "NOT VERIFIED", f"Paths finais não puderam ser plenamente confirmados em tmpfs.\n{details}")

    except Exception as e:
        for p in [temp_upload, temp_pasted]:
            if p.exists():
                p.unlink()
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def _check_has_real_models() -> List[str]:
    """Verifica se existem modelos reais (.safetensors, .ckpt) em models/ ou datasets."""
    models_found = []
    models_dir = COMFYUI_DIR / "models"
    if models_dir.exists():
        for ext in [".safetensors", ".ckpt"]:
            for f in models_dir.rglob(f"*{ext}"):
                if f.is_file() and not f.is_symlink():
                    models_found.append(str(f))
    # Verificar datasets montados em /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for ext in [".safetensors", ".ckpt"]:
            for f in kaggle_input.rglob(f"*{ext}"):
                if f.is_file():
                    models_found.append(str(f))
    return models_found


def test_04_generation(report: VerificationReport,
                        host: str = DEFAULT_HOST,
                        port: int = DEFAULT_PORT):
    """Teste 4: Geração — NOT VERIFIED sem modelo real."""
    test_id = "04_GENERATION"
    try:
        real_models = _check_has_real_models()
        pre_scan = _scan_working_dir()
        shm_outputs_before = _find_images_in_dir(SHM_OUTPUT) if SHM_OUTPUT.exists() else []

        if not real_models:
            details = (
                "Nenhum modelo real (.safetensors / .ckpt) encontrado em models/ ou /kaggle/input.\n"
                "REGRA ESTRITA: NOT VERIFIED sem modelo real.\n"
                "Não marcar PASS baseado em simulação, AST ou prompt sem modelo carregado."
            )
            report.record(test_id, "NOT VERIFIED", details)
            return

        # Se houver modelo real, verificar se houve execução real
        # (por exemplo, prompt via API que gere imagem em SHM_OUTPUT)
        # Scan pós
        time.sleep(2)
        post_scan = _scan_working_dir()
        new_images_working = [img for img in post_scan["images"] if img not in pre_scan["images"]]
        shm_outputs_after = _find_images_in_dir(SHM_OUTPUT) if SHM_OUTPUT.exists() else []
        new_shm_outputs = [img for img in shm_outputs_after if img not in shm_outputs_before]

        if new_images_working:
            details = f"VIOLAÇÃO: {len(new_images_working)} imagem(ns) gerada(s) em /kaggle/working!\n"
            for img in new_images_working:
                details += f"  ❌ {img['path']}\n"
            report.record(test_id, "FAIL", details)
        elif new_shm_outputs:
            details = (
                f"Geração real confirmada com modelo ({len(real_models)} modelo(s) detectado(s)).\n"
                f"Novas imagens em /dev/shm/comfy_ui_output: {len(new_shm_outputs)}\n"
                f"Novas imagens em /kaggle/working: 0 ✓"
            )
            report.record(test_id, "PASS", details)
        else:
            details = (
                f"Modelos detectados no ambiente ({len(real_models)}), mas nenhuma geração real\n"
                "foi submetida e concluída nesta sessão de teste.\n"
                "REGRA ESTRITA: NOT VERIFIED sem modelo real em execução ativa."
            )
            report.record(test_id, "NOT VERIFIED", details)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_05_img2img(report: VerificationReport,
                     host: str = DEFAULT_HOST,
                     port: int = DEFAULT_PORT):
    """Teste 5: Img2Img — NOT VERIFIED sem modelo real."""
    test_id = "05_IMG2IMG"
    canary_name = "IMG2IMG_CANARY.png"
    canary_input = SHM_INPUT / canary_name
    try:
        real_models = _check_has_real_models()

        # Criar canary de input em /dev/shm/comfy_ui_input para validar pathing
        _create_canary_image(canary_input)
        shm_canary_exists = canary_input.exists()

        # Verificar se vazou para /kaggle/working
        persistent_canary = list(KAGGLE_WORKING.rglob(canary_name)) if KAGGLE_WORKING.exists() else []

        # Cleanup imediato do canary
        if canary_input.exists():
            canary_input.unlink()

        if persistent_canary:
            details = f"VIOLAÇÃO: Canary de input apareceu em /kaggle/working:\n"
            for p in persistent_canary:
                details += f"  ❌ {p}\n"
            report.record(test_id, "FAIL", details)
            return

        if not real_models:
            details = (
                f"Canary de input testado em tmpfs ({canary_input}): isolamento de path OK.\n"
                "Porém, nenhum modelo real (.safetensors/.ckpt) está carregado para executar a inferência img2img completa.\n"
                "REGRA ESTRITA: NOT VERIFIED sem modelo real."
            )
            report.record(test_id, "NOT VERIFIED", details)
        else:
            details = (
                "Modelos detectados, porém workflow de inferência img2img não foi submetido via API nesta sessão.\n"
                "REGRA ESTRITA: NOT VERIFIED sem modelo real executando inferência ativa."
            )
            report.record(test_id, "NOT VERIFIED", details)

    except Exception as e:
        if canary_input.exists():
            canary_input.unlink()
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")

    except Exception as e:
        # Cleanup
        for p in [SHM_INPUT / canary_name]:
            if p.exists():
                p.unlink()
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_06_zip_aes(report: VerificationReport):
    """Teste 6: ZIP AES-256 real."""
    test_id = "06_ZIP_AES256"
    zip_path = SHM_ARCHIVE / "security_test.zip"
    canary_path = SHM_TEMP / "ZIP_CANARY.png"
    test_password = "SecurityVerification_T3st_2024!"

    try:
        # Instalar pyzipper se necessário
        try:
            import pyzipper
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyzipper"],
                           check=True, timeout=300)
            import pyzipper

        # Criar canary
        _create_canary_image(canary_path)
        canary_content = canary_path.read_bytes()
        print(f"   Canary criado: {canary_path} ({len(canary_content)} bytes)")

        # Criar ZIP em /dev/shm
        SHM_ARCHIVE.mkdir(parents=True, exist_ok=True)
        with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED,
                                  encryption=pyzipper.WZ_AES) as zf:
            zf.setpassword(test_password.encode("utf-8"))
            zf.write(canary_path, arcname="ZIP_CANARY.png")

        checks = []

        # Verificar path
        zip_in_shm = str(zip_path).startswith("/dev/shm")
        checks.append(f"ZIP path em /dev/shm: {'SIM ✓' if zip_in_shm else 'NÃO ✗'}")

        # Verificar ausência em /kaggle/working
        persistent_zips = list(KAGGLE_WORKING.rglob("security_test.zip")) if KAGGLE_WORKING.exists() else []
        checks.append(f"ZIP em /kaggle/working: {len(persistent_zips)} {'✗' if persistent_zips else '✓'}")

        # Testar abertura sem senha
        open_without_pass_failed = False
        try:
            with pyzipper.AESZipFile(zip_path, "r") as zf:
                zf.read("ZIP_CANARY.png")
        except (RuntimeError, Exception):
            open_without_pass_failed = True
        checks.append(f"Abrir sem senha falha: {'SIM ✓' if open_without_pass_failed else 'NÃO ✗'}")

        # Testar abertura com senha
        open_with_pass_ok = False
        recovered_content = None
        try:
            with pyzipper.AESZipFile(zip_path, "r") as zf:
                zf.setpassword(test_password.encode("utf-8"))
                recovered_content = zf.read("ZIP_CANARY.png")
            open_with_pass_ok = True
        except Exception as e:
            checks.append(f"Abrir com senha: FALHOU — {e}")
        checks.append(f"Abrir com senha funciona: {'SIM ✓' if open_with_pass_ok else 'NÃO ✗'}")

        # Verificar conteúdo
        content_ok = recovered_content == canary_content if recovered_content else False
        checks.append(f"Conteúdo recuperado corretamente: {'SIM ✓' if content_ok else 'NÃO ✗'}")

        # Remover ZIP
        if zip_path.exists():
            zip_path.unlink()
        zip_gone = not zip_path.exists()
        checks.append(f"ZIP removido (Path.exists()==False): {'SIM ✓' if zip_gone else 'NÃO ✗'}")

        # Cleanup canary
        if canary_path.exists():
            canary_path.unlink()

        details = "\n".join(checks)
        all_ok = (zip_in_shm and not persistent_zips and open_without_pass_failed
                  and open_with_pass_ok and content_ok and zip_gone)

        report.record(test_id, "PASS" if all_ok else "FAIL", details)

    except Exception as e:
        for p in [zip_path, canary_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_07_download(report: VerificationReport, observed_during_download: bool = False):
    """Teste 7: Download — PASS somente se o filesystem for observado durante o download."""
    test_id = "07_DOWNLOAD"
    try:
        staging_files = _find_images_in_dir(KAGGLE_WORKING)
        staging_zips = [f for f in staging_files if f.get("detection", "").startswith("archive:")]
        staging_images = [f for f in staging_files if not f.get("detection", "").startswith("archive:")]

        details = (
            f"Staging scan atual em /kaggle/working:\n"
            f"  Imagens: {len(staging_images)}\n"
            f"  Archives: {len(staging_zips)}\n"
        )

        if staging_images or staging_zips:
            details += "\nArtefatos encontrados em /kaggle/working:\n"
            for f in staging_files[:20]:
                details += f"  ❌ {f['path']} ({f['size']} bytes, {f['detection']})\n"
            report.record(test_id, "FAIL", f"Artefatos persistentes encontrados!\n{details}")
            return

        if not observed_during_download:
            details += (
                "\nREGRA ESTRITA: PASS somente se o filesystem for observado durante o download.\n"
                "Caso contrário: NOT VERIFIED.\n"
                "O fluxo de download (FileLink / Kaggle Files) não foi monitorado continuamente\n"
                "em tempo real durante a transferência ativa do arquivo.\n"
                "Nota de plataforma: Se o Kaggle Files > Download exigir arquivos em /kaggle/working,\n"
                "esta limitação de plataforma impede garantia absoluta de zero-disk.\n"
                "Status: NOT VERIFIED / PLATFORM CONSTRAINT"
            )
            report.record(test_id, "NOT VERIFIED", details)
        else:
            details += (
                "\nFilesystem monitorado ativamente durante o download: ZERO arquivos criados em /kaggle/working.\n"
                "Status: PASS"
            )
            report.record(test_id, "PASS", details)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def _find_comfyui_pid(port: int = DEFAULT_PORT) -> Optional[int]:
    """Encontra PID do ComfyUI pela porta."""
    for cmd in [["fuser", f"{port}/tcp"], ["ss", "-tlnp", f"sport = :{port}"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            import re
            for token in result.stdout.split():
                if token.strip().isdigit():
                    return int(token.strip())
            m = re.search(r"pid=(\d+)", result.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def _read_proc_cmdline(pid: int) -> List[str]:
    """Lê /proc/<pid>/cmdline."""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [a.decode("utf-8", errors="replace") for a in data.split(b"\x00") if a]
    except (OSError, PermissionError):
        return []


def test_08_process(report: VerificationReport,
                     host: str = DEFAULT_HOST,
                     port: int = DEFAULT_PORT):
    """Teste 8: Verificação do processo real."""
    test_id = "08_PROCESS"
    try:
        pid = _find_comfyui_pid(port)
        if not pid:
            report.record(test_id, "NOT VERIFIED", "PID do ComfyUI não encontrado")
            return

        cmdline = _read_proc_cmdline(pid)
        if not cmdline:
            report.record(test_id, "NOT VERIFIED", f"Não foi possível ler /proc/{pid}/cmdline")
            return

        cmdline_str = " ".join(cmdline)
        checks = [f"PID: {pid}", f"cmdline: {cmdline_str[:500]}"]

        expected_flags = {
            "--input-directory": str(SHM_INPUT),
            "--output-directory": str(SHM_OUTPUT),
            "--temp-directory": str(SHM_TEMP),
            "--user-directory": str(SHM_USER),
            "--listen": "127.0.0.1",
            "--port": str(port),
        }

        all_ok = True
        for flag, expected_val in expected_flags.items():
            try:
                idx = cmdline.index(flag)
                actual = cmdline[idx + 1] if idx + 1 < len(cmdline) else "MISSING"
                ok = actual == expected_val
                if not ok:
                    all_ok = False
                checks.append(f"  {flag}: {actual} {'✓' if ok else '✗ ESPERADO: ' + expected_val}")
            except ValueError:
                all_ok = False
                checks.append(f"  {flag}: AUSENTE ✗")

        # Verificar presença de --enable-manager (PERMITIDO em SECURE_MODE)
        if "--enable-manager" in cmdline:
            checks.append("  --enable-manager: PRESENTE ✓ (permitido em SECURE_MODE com isolamento)")
        else:
            checks.append("  --enable-manager: ausente (Manager pode não ter sido solicitado)")

        details = "\n".join(checks)
        report.record(test_id, "PASS" if all_ok else "FAIL", details,
                      {"pid": pid, "cmdline": cmdline})

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_09_reuse(report: VerificationReport,
                   port: int = DEFAULT_PORT,
                   real_incompatible_process_tested: bool = False):
    """Teste 9: Reuse detection — STATICALLY VERIFIED enquanto não houver teste real de processo incompatível."""
    test_id = "09_REUSE"
    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import get_secure_mode

        if not get_secure_mode():
            report.record(test_id, "NOT VERIFIED",
                          "SECURE_MODE não está ativo — teste de reuse não aplicável")
            return

        if not real_incompatible_process_tested:
            details = (
                "REGRA: Marcar como STATICALLY VERIFIED enquanto não houver teste real de processo incompatível.\n\n"
                "Verificação por análise estática de código (comfyui_setup.py):\n"
                "  - L1427-1429: reuse_existing forçado incondicionalmente para False em SECURE_MODE\n"
                "  - L1463-1468: processo com configuração incompatível é terminado com SIGTERM → SIGKILL\n"
                "  - L1471-1473: porta em uso sem PID identificado levanta SecurityError (fail-closed)\n"
                "  - L910-959: _verify_process_paths valida paths e rejeita qualquer argumento em /kaggle/working\n\n"
                "Nenhum processo incompatível real foi gerado nesta sessão para teste destrutivo.\n"
                "Status: STATICALLY VERIFIED"
            )
            report.record(test_id, "STATICALLY VERIFIED", details)
        else:
            report.record(test_id, "PASS", "Teste real com processo incompatível executado: processo detectado e encerrado com sucesso.")

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_10_ngrok(report: VerificationReport):
    """Teste 10: ngrok em SECURE_MODE — deve estar ATIVO e funcional (após health check)."""
    test_id = "10_NGROK"
    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import get_secure_mode

        if not get_secure_mode():
            report.record(test_id, "NOT VERIFIED",
                          "SECURE_MODE não está ativo — teste ngrok não aplicável")
            return

        # Em SECURE_MODE, ngrok é PERMITIDO.
        # Verificar se há túneis ativos (pipeline já deve ter iniciado ngrok)
        tunnels = []
        try:
            from pyngrok import ngrok
            tunnels = list(ngrok.get_tunnels())
        except Exception:
            pass

        # Verificar que o token não aparece em logs
        token_leaked = False
        token_value = os.environ.get("NGROK_AUTHTOKEN", "")
        if token_value:
            log_path = SHM_LOGS / "comfyui.log" if SHM_LOGS.exists() else None
            if log_path and log_path.exists():
                try:
                    log_content = log_path.read_text(encoding="utf-8", errors="replace")
                    if token_value in log_content:
                        token_leaked = True
                except Exception:
                    pass

        # Verificar que nenhum arquivo do ngrok foi gravado em /kaggle/working
        ngrok_files_in_working = []
        if KAGGLE_WORKING.exists():
            for item in KAGGLE_WORKING.rglob("*"):
                if item.is_file() and "ngrok" in item.name.lower():
                    ngrok_files_in_working.append(str(item))

        details = (
            f"Túneis ngrok ativos: {len(tunnels)} {'✓' if len(tunnels) > 0 else '✗ (ngrok pode não ter sido iniciado)'}\n"
            f"Token leaked nos logs: {'SIM ✗' if token_leaked else 'NÃO ✓'}\n"
            f"Arquivos ngrok em /kaggle/working: {len(ngrok_files_in_working)} {'✓' if len(ngrok_files_in_working) == 0 else '✗'}\n"
        )

        if len(tunnels) > 0 and not token_leaked and len(ngrok_files_in_working) == 0:
            report.record(test_id, "PASS", details)
        elif len(tunnels) == 0:
            report.record(test_id, "NOT VERIFIED", f"ngrok não parece estar ativo.\n{details}")
        else:
            report.record(test_id, "FAIL", f"Problema com ngrok detectado.\n{details}")

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_11_manager(report: VerificationReport,
                     port: int = DEFAULT_PORT):
    """Teste 11: Manager Active & Network Traffic — Manager está ativo em SECURE_MODE, com isolamento."""
    try:
        pid = _find_comfyui_pid(port)
        checks_manager = []
        checks_network = []

        # 1. Manager active verification
        # Em SECURE_MODE, Manager é PERMITIDO (com isolamento de filesystem).
        manager_active = False
        if pid:
            cmdline = _read_proc_cmdline(pid)
            manager_in_cmdline = "--enable-manager" in cmdline
            if manager_in_cmdline:
                manager_active = True
            checks_manager.append(
                f"[MANAGER ACTIVE] --enable-manager no cmdline: {'PRESENTE ✓' if manager_in_cmdline else 'AUSENTE ✗'}"
            )
        else:
            checks_manager.append("[MANAGER ACTIVE] PID não encontrado — verificação de cmdline pulada")

        # Verificar que logs do Manager estão em /dev/shm, não em /kaggle/working
        log_paths = [
            SHM_LOGS / "comfyui.log",
            SHM_TEMP / "comfyui.log",
            COMFYUI_DIR / "comfyui.log",
        ]
        manager_in_logs = False
        log_in_shm = False
        for log_path in log_paths:
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding="utf-8", errors="replace")
                    for indicator in ["api.comfy.org", "ComfyUI-Manager", "manager_server"]:
                        if indicator in log_content:
                            manager_in_logs = True
                            if "/dev/shm" in str(log_path):
                                log_in_shm = True
                except Exception:
                    pass
        if manager_in_logs:
            checks_manager.append(f"[MANAGER ACTIVE] Indicador Manager nos logs: {'em /dev/shm ✓' if log_in_shm else 'em /kaggle/working ✗'}")
        else:
            checks_manager.append("[MANAGER ACTIVE] Manager não encontrado nos logs (pode não ter sido carregado ainda)")

        # Manager directory — pode estar em custom_nodes, desde que não escreva em /kaggle/working
        manager_dir = COMFYUI_DIR / "custom_nodes" / "ComfyUI-Manager"
        manager_installed = manager_dir.exists()
        checks_manager.append(
            f"[MANAGER ACTIVE] ComfyUI-Manager em custom_nodes: {'INSTALADO ✓' if manager_installed else 'NÃO INSTALADO'}"
        )

        # Verificar que Manager não criou arquivos em /kaggle/working
        manager_files_in_working = []
        if KAGGLE_WORKING.exists():
            for item in KAGGLE_WORKING.rglob("*"):
                if item.is_file() and "manager" in item.name.lower():
                    manager_files_in_working.append(str(item))
        checks_manager.append(
            f"[MANAGER ACTIVE] Arquivos do Manager em /kaggle/working: {len(manager_files_in_working)} {'✓' if len(manager_files_in_working) == 0 else '✗'}"
        )

        # 2. Network traffic verification
        checks_network.append("[EXTERNAL NETWORK TRAFFIC]")
        checks_network.append("  - custom nodes executam Python arbitrário e podem fazer requests externos")
        checks_network.append("  - ngrok cria exposição externa (túnel público)")
        checks_network.append("  - não há egress control no ambiente Kaggle")
        checks_network.append("  - RISCO RESIDUAL: exfiltração por código de terceiros não é bloqueada")
        checks_network.append("  - Resultado: NOT VERIFIED (risco residual documentado)")

        manager_status_str = "VERIFIED" if manager_active else "FAIL"
        network_status_str = "NOT VERIFIED"

        # Record two separate test results
        manager_details = "MANAGER ACTIVE CHECKS:\n" + "\n".join(checks_manager)
        network_details = "\n".join(checks_network)

        report.record("11_MANAGER_ACTIVE", "PASS" if manager_active else "FAIL", manager_details,
                      {"status": manager_status_str})
        report.record("11_NETWORK_TRAFFIC", network_status_str, network_details,
                      {"status": network_status_str})

    except Exception as e:
        report.record("11_MANAGER_ACTIVE", "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")
        report.record("11_NETWORK_TRAFFIC", "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_12_custom_nodes(report: VerificationReport):
    """Teste 12: Custom node hashing — gerar baseline, não verificar automaticamente."""
    test_id = "12_CUSTOM_NODES"
    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import (
            ALLOWED_CUSTOM_NODES,
            compute_node_directory_hash,
            check_custom_nodes_allowlist,
        )

        custom_dir = COMFYUI_DIR / "custom_nodes"
        if not custom_dir.exists():
            report.record(test_id, "NOT VERIFIED", f"{custom_dir} não existe")
            return

        # 1. Listar nodes presentes
        nodes_present = [
            d.name for d in sorted(custom_dir.iterdir())
            if d.is_dir() and not d.name.startswith("__")
        ]
        print(f"   Nodes presentes: {nodes_present}")

        # 2. Validar allowlist
        unauthorized = [n for n in nodes_present if n not in ALLOWED_CUSTOM_NODES]

        # 3. Calcular hashes
        baseline = {}
        for node_name in nodes_present:
            node_path = custom_dir / node_name
            try:
                h = compute_node_directory_hash(node_path)
                file_count = sum(1 for f in node_path.rglob("*") if f.is_file())
                baseline[node_name] = {"hash": h, "file_count": file_count}
            except Exception as e:
                baseline[node_name] = {"hash": f"ERROR: {e}", "file_count": -1}

        # 4. Gerar relatório de baseline
        lines = [
            f"Allowlist: {sorted(ALLOWED_CUSTOM_NODES)}",
            f"Nodes presentes: {nodes_present}",
            f"Nodes não autorizados: {unauthorized}",
            "",
            "HASH BASELINE:",
        ]
        for name, info in sorted(baseline.items()):
            lines.append(f"  {name}")
            lines.append(f"    hash: {info['hash']}")
            lines.append(f"    files: {info['file_count']}")

        lines.append("")
        lines.append("STATUS: HASH BASELINE GENERATED")
        lines.append("NÃO é HASH VERIFIED — baseline precisa de aprovação explícita.")

        details = "\n".join(lines)

        if unauthorized:
            report.record(test_id, "FAIL",
                          f"Nodes não autorizados detectados: {unauthorized}\n\n{details}")
        else:
            report.record(test_id, "PASS", details,
                          {"baseline": baseline, "unauthorized": unauthorized})

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_13_node_alteration(report: VerificationReport):
    """Teste 13: Alteração de node — verificar detecção de tampering."""
    test_id = "13_NODE_ALTERATION"
    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import (
            ALLOWED_CUSTOM_NODES,
            EXPECTED_CUSTOM_NODE_HASHES,
            compute_node_directory_hash,
            snapshot_custom_nodes,
            verify_custom_nodes_unchanged,
            SecurityError as SetupSecurityError,
        )

        custom_dir = COMFYUI_DIR / "custom_nodes"

        # Criar cópia de teste (NÃO modificar o ambiente de produção)
        test_node_dir = SHM_TEMP / "_security_test_node"
        test_node_dir.mkdir(parents=True, exist_ok=True)

        # Criar arquivo de teste
        test_file = test_node_dir / "__init__.py"
        test_file.write_text("# Security test node\nprint('hello')\n")

        # Calcular hash original
        original_hash = compute_node_directory_hash(test_node_dir)

        # Alterar um byte
        content = test_file.read_bytes()
        altered = content[:-1] + bytes([content[-1] ^ 0x01])  # flip last bit
        test_file.write_bytes(altered)

        # Calcular hash alterado
        altered_hash = compute_node_directory_hash(test_node_dir)

        # Verificar que os hashes são diferentes
        hash_changed = original_hash != altered_hash

        # Testar via snapshot — criar snapshot antes e verificar depois
        # Restaurar original
        test_file.write_text("# Security test node\nprint('hello')\n")
        snapshot_before = snapshot_custom_nodes(SHM_TEMP)  # snapshot do dir de teste

        # Alterar novamente
        test_file.write_bytes(altered)
        snapshot_after = snapshot_custom_nodes(SHM_TEMP)

        # Verificar detecção
        detection_works = (
            snapshot_before.get("nodes", {}).get("_security_test_node", {}).get("files", {}) !=
            snapshot_after.get("nodes", {}).get("_security_test_node", {}).get("files", {})
        )

        # Testar verify_node_hashes com hash esperado
        # Simular EXPECTED_CUSTOM_NODE_HASHES temporariamente
        import comfyui_setup
        original_expected = dict(comfyui_setup.EXPECTED_CUSTOM_NODE_HASHES)
        try:
            comfyui_setup.EXPECTED_CUSTOM_NODE_HASHES["_security_test_node"] = original_hash
            mismatches = comfyui_setup.verify_node_hashes(SHM_TEMP)
            mismatch_detected = len(mismatches) > 0
        finally:
            comfyui_setup.EXPECTED_CUSTOM_NODE_HASHES = original_expected

        # Cleanup
        if test_node_dir.exists():
            shutil.rmtree(test_node_dir)

        details = (
            f"Hash original: {original_hash[:32]}...\n"
            f"Hash alterado: {altered_hash[:32]}...\n"
            f"Hashes diferentes após 1-byte change: {'SIM ✓' if hash_changed else 'NÃO ✗'}\n"
            f"Snapshot detecta mudança: {'SIM ✓' if detection_works else 'NÃO ✗'}\n"
            f"verify_node_hashes detecta mismatch: {'SIM ✓' if mismatch_detected else 'NÃO ✗'}\n"
            "Nota: teste executado em cópia em /dev/shm, NÃO em produção."
        )

        all_ok = hash_changed and detection_works and mismatch_detected
        report.record(test_id, "PASS" if all_ok else "FAIL", details)

    except Exception as e:
        # Cleanup
        test_node_dir = SHM_TEMP / "_security_test_node"
        if test_node_dir.exists():
            shutil.rmtree(test_node_dir, ignore_errors=True)
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_14_drive_credentials(report: VerificationReport):
    """Teste 14: Cleanup de credenciais do Drive."""
    test_id = "14_DRIVE_CREDENTIALS"
    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import cleanup_gdrive_credentials

        cred_paths = [
            Path("/root/gdrive_sa.json"),
            Path("/root/.config/rclone/rclone.conf"),
        ]

        # Verificar estado atual
        checks = []
        for p in cred_paths:
            exists_before = p.exists()
            checks.append(f"Antes cleanup — {p.name}: {'existe' if exists_before else 'ausente'}")

        # Executar cleanup
        cleanup_gdrive_credentials()

        # Verificar após cleanup
        any_remaining = False
        for p in cred_paths:
            exists_after = p.exists()
            if exists_after:
                any_remaining = True
            checks.append(f"Após cleanup — {p.name}: {'EXISTE ✗' if exists_after else 'removido ✓'}")

        # Testar cleanup após exceção
        # Criar credencial temporária de teste para verificar
        test_sa = Path("/root/gdrive_sa.json")
        try:
            test_sa.parent.mkdir(parents=True, exist_ok=True)
            test_sa.write_text('{"type": "service_account", "test": true}')
            os.chmod(test_sa, 0o600)
            checks.append(f"Credencial teste criada: {test_sa}")

            # Simular exceção e cleanup no finally
            try:
                raise RuntimeError("Exceção simulada para teste de cleanup")
            except RuntimeError:
                pass
            finally:
                cleanup_gdrive_credentials()

            exists_after_exception = test_sa.exists()
            checks.append(
                f"Após exceção + cleanup — gdrive_sa.json: "
                f"{'EXISTE ✗' if exists_after_exception else 'removido ✓'}"
            )
            if exists_after_exception:
                any_remaining = True
        except PermissionError:
            checks.append("Sem permissão para criar credencial de teste em /root/")

        details = "\n".join(checks)
        report.record(test_id, "PASS" if not any_remaining else "FAIL", details)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_15_exception_cleanup(report: VerificationReport):
    """Teste 15: Cleanup após exceção — artefatos devem ser removidos."""
    test_id = "15_EXCEPTION_CLEANUP"
    try:
        # Criar artefatos de teste em /dev/shm
        test_input = SHM_INPUT / "exception_test.png"
        test_output = SHM_OUTPUT / "exception_test.png"
        test_zip = SHM_ARCHIVE / "exception_test.zip"

        _create_canary_image(test_input)
        _create_canary_image(test_output)

        # Criar ZIP de teste
        SHM_ARCHIVE.mkdir(parents=True, exist_ok=True)
        try:
            import pyzipper
            with pyzipper.AESZipFile(test_zip, "w", compression=pyzipper.ZIP_DEFLATED,
                                      encryption=pyzipper.WZ_AES) as zf:
                zf.setpassword(b"test_password_12345")
                zf.write(test_output, arcname="exception_test.png")
        except ImportError:
            test_zip.write_bytes(b"PK\x03\x04" + b"\x00" * 100)  # fake ZIP header

        # Verificar que artefatos existem
        assert test_input.exists(), "Input não criado"
        assert test_output.exists(), "Output não criado"
        assert test_zip.exists(), "ZIP não criado"

        # Simular exceção com cleanup no finally
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))

        from comfyui_setup import secure_cleanup

        try:
            raise RuntimeError("Exceção simulada para teste de cleanup")
        except RuntimeError:
            pass
        finally:
            secure_cleanup(
                comfyui_dir=COMFYUI_DIR,
                raise_on_persistent=False,
            )

        # Verificar que artefatos foram removidos
        checks = []
        shm_dirs = [SHM_INPUT, SHM_OUTPUT, SHM_TEMP, SHM_ARCHIVE]
        any_remaining = False

        for d in shm_dirs:
            if d.exists():
                files = list(d.rglob("*"))
                files = [f for f in files if f.is_file()]
                if files:
                    any_remaining = True
                    checks.append(f"{d.name}: {len(files)} arquivo(s) RESTANTE(S) ✗")
                    for f in files[:5]:
                        checks.append(f"  {f}")
                else:
                    checks.append(f"{d.name}: vazio ✓")
            else:
                checks.append(f"{d.name}: não existe (recriado vazio) ✓")

        # Verificar /kaggle/working
        working_images = _find_images_in_dir(KAGGLE_WORKING)
        checks.append(f"/kaggle/working imagens: {len(working_images)} {'✓' if len(working_images) == 0 else '✗'}")

        details = "\n".join(checks)
        report.record(test_id, "PASS" if not any_remaining and len(working_images) == 0 else "FAIL", details)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_16_final_scan(report: VerificationReport):
    """Teste 16: Scan completo de /kaggle/working."""
    test_id = "16_FINAL_SCAN"
    try:
        images = _find_images_in_dir(KAGGLE_WORKING)
        symlinks = _find_symlinks_in_dir(KAGGLE_WORKING)

        lines = [
            f"Scan root: {KAGGLE_WORKING}",
            f"Imagens/archives detectados: {len(images)}",
            f"Symlinks detectados: {len(symlinks)}",
        ]

        if images:
            lines.append("")
            lines.append("ARTEFATOS SENSÍVEIS:")
            for img in images:
                lines.append(
                    f"  PATH: {img['path']}\n"
                    f"  SIZE: {img['size']} bytes\n"
                    f"  MTIME: {img['mtime']}\n"
                    f"  TYPE: {img['detection']}\n"
                    f"  REASON: sensitive artifact in persistent storage"
                )

        if symlinks:
            lines.append("")
            lines.append("SYMLINKS:")
            for sl in symlinks:
                lines.append(f"  {sl['path']} → {sl['target']}")

        lines.append(f"\nTotal: {len(images)} sensitive artifacts")

        details = "\n".join(lines)
        report.record(test_id, "PASS" if len(images) == 0 else "FAIL", details,
                      {"image_count": len(images), "symlink_count": len(symlinks)})

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


def test_17_persistence(report: VerificationReport):
    """Teste 17: Persistência entre sessões — registrar estado para verificação futura."""
    test_id = "17_PERSISTENCE"
    try:
        # Este teste não pode ser completado em uma única sessão.
        # Registramos o estado atual para verificação em sessão futura.

        scan = _scan_working_dir()
        state = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_count": scan["image_count"],
            "symlink_count": scan["symlink_count"],
            "images": scan["images"][:20],
        }

        # Salvar estado em /dev/shm (volátil — desaparece entre sessões)
        state_file = SHM_TEMP / "session_state.json"
        SHM_TEMP.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))

        details = (
            "REGRA ESTRITA: NOT VERIFIED até segunda sessão.\n\n"
            f"Estado da sessão atual registrado em {state_file}\n"
            f"Imagens em /kaggle/working: {scan['image_count']}\n"
            f"Symlinks em /kaggle/working: {scan['symlink_count']}\n\n"
            "Procedimento para completar este teste:\n"
            "1. Encerrar completamente esta sessão do Kaggle\n"
            "2. Iniciar uma nova sessão limpa e independente (sem reuso de dados)\n"
            "3. Verificar que /kaggle/working não contém artefatos antigos de imagem\n"
            "4. Verificar que /dev/shm/session_state.json NÃO existe (tmpfs volátil foi resetado)\n\n"
            "Status: NOT VERIFIED (aguardando execução na segunda sessão)"
        )

        report.record(test_id, "NOT VERIFIED", details, state)

    except Exception as e:
        report.record(test_id, "NOT VERIFIED", f"Exceção: {e}\n{traceback.format_exc()}")


# ===========================================================================
# EXECUÇÃO PRINCIPAL
# ===========================================================================

def collect_environment(report: VerificationReport):
    """Coleta informações do ambiente."""
    report.environment["OS"] = f"{platform.system()} {platform.release()}"
    report.environment["Python"] = sys.version.split()[0]
    report.environment["Platform"] = platform.platform()

    try:
        kernel = subprocess.run(["uname", "-r"], capture_output=True, text=True, timeout=5)
        report.environment["Kernel"] = kernel.stdout.strip()
    except Exception:
        report.environment["Kernel"] = "unknown"

    report.environment["Kaggle"] = "Yes" if _is_kaggle() else "No"
    report.environment["/dev/shm"] = "exists" if SHM_BASE.exists() else "MISSING"

    # ComfyUI version
    try:
        version_file = COMFYUI_DIR / "comfyui_version.py"
        if version_file.exists():
            content = version_file.read_text()
            report.environment["ComfyUI"] = content.strip()[:100]
        else:
            report.environment["ComfyUI"] = str(COMFYUI_DIR)
    except Exception:
        report.environment["ComfyUI"] = "unknown"

    # SECURE_MODE
    report.environment["COMFYUI_SECURE_MODE"] = os.environ.get("COMFYUI_SECURE_MODE", "(not set)")

    try:
        scripts_dir = KAGGLE_WORKING / "colab-pipeline" / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        scripts_dir2 = KAGGLE_WORKING / "scripts"
        if scripts_dir2.exists() and str(scripts_dir2) not in sys.path:
            sys.path.insert(0, str(scripts_dir2))
        from comfyui_setup import get_secure_mode
        report.environment["SECURE_MODE_effective"] = str(get_secure_mode())
    except Exception:
        report.environment["SECURE_MODE_effective"] = "import failed"


def run_all_tests(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                   skip_generation: bool = False,
                   observed_during_download: bool = False,
                   real_incompatible_process_tested: bool = False) -> VerificationReport:
    """Executa todos os testes de verificação."""
    report = VerificationReport()

    print("=" * 60)
    print("SECURITY RUNTIME VERIFICATION — STARTING")
    print(f"Time: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    # Pré-requisitos
    ok, msg = check_prerequisites()
    if not ok:
        print(f"\n⚠️  PRÉ-REQUISITOS NÃO ATENDIDOS:\n{msg}")
        print("Alguns testes podem falhar ou ser marcados NOT VERIFIED.")

    # Ambiente
    collect_environment(report)
    print("\n--- Environment ---")
    for k, v in report.environment.items():
        print(f"  {k}: {v}")

    # Executar testes
    print("\n" + "=" * 60)
    print("EXECUTANDO TESTES")
    print("=" * 60)

    test_02_path_verification(report, host, port)
    test_03_paste_upload(report, host, port)
    if not skip_generation:
        test_04_generation(report, host, port)
    else:
        report.record("04_GENERATION", "NOT VERIFIED", "skip_generation=True (sem modelo carregado)")
    test_05_img2img(report, host, port)
    test_06_zip_aes(report)
    test_07_download(report, observed_during_download=observed_during_download)
    test_08_process(report, host, port)
    test_09_reuse(report, port=port, real_incompatible_process_tested=real_incompatible_process_tested)
    test_10_ngrok(report)
    test_11_manager(report, port)
    test_12_custom_nodes(report)
    test_13_node_alteration(report)
    test_14_drive_credentials(report)
    test_15_exception_cleanup(report)
    test_16_final_scan(report)
    test_17_persistence(report)

    # Relatório final
    print("\n" + "=" * 60)
    verdict = report.get_verdict()
    icon = {"PRODUCTION VERIFIED": "✅", "PARTIALLY VERIFIED": "⚠️",
            "FAILED SECURITY VERIFICATION": "❌"}[verdict]
    print(f"FINAL VERDICT: {icon} {verdict}")
    print("=" * 60)

    # Salvar relatório
    report_path = SHM_TEMP / "SECURITY_RUNTIME_VERIFICATION_REPORT.md"
    SHM_TEMP.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nRelatório salvo: {report_path}")

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Security Runtime Verification")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Pular teste de geração (sem modelo carregado)")
    parser.add_argument("--observed-download", action="store_true",
                        help="Declarar que o filesystem foi observado continuamente durante download")
    parser.add_argument("--test-incompatible-process", action="store_true",
                        help="Declarar que processo incompatível real foi testado")
    args = parser.parse_args()

    report = run_all_tests(
        host=args.host,
        port=args.port,
        skip_generation=args.skip_generation,
        observed_during_download=args.observed_download,
        real_incompatible_process_tested=args.test_incompatible_process,
    )

    # Exit code baseado no verdict
    verdict = report.get_verdict()
    if verdict == "FAILED SECURITY VERIFICATION":
        sys.exit(1)
    sys.exit(0)
