#!/usr/bin/env python3
"""Setup do ComfyUI no Kaggle Notebook (SSD local) — zero-trust / zero-persistent-image.

Garantias de isolamento (fail-closed):
  INPUT  → /dev/shm/comfy_ui_input   (tmpfs volátil)
  OUTPUT → /dev/shm/comfy_ui_output  (tmpfs volátil)
  TEMP   → /dev/shm/comfy_ui_temp    (tmpfs volátil)

Nenhuma imagem toca /kaggle/working em nenhuma circunstância controlada pelo pipeline.
O ZIP de download é criado em /dev/shm/comfy_ui_archive/ e NUNCA em /kaggle/working.
Processo antigo só pode ser reutilizado após verificação de /proc/<pid>/cmdline.
reuse_existing padrão é False — sempre inicia processo novo salvo override explícito.

Ordem obrigatória de runtime: bootstrap → models → start → health → ngrok.
Manager integrado: manager_requirements.txt + --enable-manager (NÃO custom node).
VRAM: COMFYUI_CUDA_DEVICE=0 por padrão; DynamicVRAM do ComfyUI (sem highvram/lowvram).

LIMITES DESTE CÓDIGO:
  Não controla o host Kaggle, a plataforma, nem vulnerabilidades em dependências externas.
  Não garante isolamento contra acesso privilegiado do provedor ao host.
  Garante apenas prevenção de persistência acidental através dos caminhos controlados aqui.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_COMFYUI_DIR = Path("/kaggle/working/ComfyUI")
DEFAULT_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
ENV_CUDA_DEVICE = "COMFYUI_CUDA_DEVICE"
DEFAULT_CUDA_DEVICE = 0

# Diretórios voláteis (tmpfs) — ÚNICA localização válida para I/O de imagens
SHM_BASE = Path("/dev/shm")
SHM_INPUT = SHM_BASE / "comfy_ui_input"
SHM_OUTPUT = SHM_BASE / "comfy_ui_output"
SHM_TEMP = SHM_BASE / "comfy_ui_temp"
SHM_ARCHIVE = SHM_BASE / "comfy_ui_archive"  # ZIP temporário de download

# Custom nodes explicitamente permitidos (allowlist). Qualquer node fora desta lista
# que seja detectado em custom_nodes/ durante uma sessão segura causa ABORT.
ALLOWED_CUSTOM_NODES: frozenset[str] = frozenset([
    "ComfyUI_essentials",
    "comfyui-krea2edit",
    "ComfyUI-Krea2T-Enhancer",
    "rgthree-comfy",
])

# Apenas custom nodes instalados automaticamente pelo setup.
DEFAULT_CUSTOM_NODES = [
    "cubiq/ComfyUI_essentials",
    "lbouaraba/comfyui-krea2edit",
]

MODEL_CATEGORIES = [
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
]

# Flags de VRAM que NÃO devem ser adicionadas por padrão
DISCOURAGED_VRAM_FLAGS = {
    "--highvram", "--gpu-only", "--lowvram", "--novram",
    "--fast", "--reserve-vram",
}

# Extensões de imagem que NUNCA devem aparecer em /kaggle/working
SENSITIVE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".bmp", ".tif", ".tiff",
}
SENSITIVE_ARCHIVES = {".zip", ".7z", ".rar", ".tar", ".gz"}


# ---------------------------------------------------------------------------
# Helpers de segurança — fail-closed
# ---------------------------------------------------------------------------

def _security_abort(msg: str) -> None:
    """Imprime mensagem crítica e levanta SecurityError. Nunca continua silenciosamente."""
    print(f"\n{'='*60}")
    print("SECURITY VIOLATION — PIPELINE ABORTED")
    print(f"REASON: {msg}")
    print(f"{'='*60}\n")
    raise SecurityError(msg)


class SecurityError(RuntimeError):
    """Levantada quando uma violação de segurança é detectada. Não deve ser silenciada."""


def assert_shm_path(path: Path, label: str) -> None:
    """Garante que path está em /dev/shm. Aborta caso contrário."""
    path = Path(path)
    if not str(path).startswith("/dev/shm"):
        _security_abort(
            f"{label} está FORA de /dev/shm: {path}\n"
            f"Nenhuma imagem deve ser processada com {label} em disco persistente."
        )
    if "/kaggle/working" in str(path):
        _security_abort(
            f"{label} aponta para /kaggle/working: {path}\n"
            f"Isso viola o isolamento de zero-persistent-image."
        )


def provision_shm_dirs(*dirs: Path) -> None:
    """Cria diretórios em /dev/shm com permissões 0o777. Verifica que estão em tmpfs."""
    for d in dirs:
        assert_shm_path(d, f"diretório tmpfs '{d.name}'")
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o777)
    print(f"[INFO] tmpfs provisionado: {' | '.join(str(d) for d in dirs)}")


def safe_remove(path: Path) -> None:
    """Remove arquivo ou diretório de forma segura e idempotente. Verifica remoção."""
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()
    if path.exists():
        raise RuntimeError(f"[SECURITY] safe_remove falhou: {path} ainda existe após remoção")


def secure_cleanup(
    comfyui_dir: Optional[Path] = None,
    extra_dirs: Optional[list[Path]] = None,
) -> None:
    """
    Limpeza segura e completa de todos os artefatos voláteis.

    Apaga:
      - /dev/shm/comfy_ui_input
      - /dev/shm/comfy_ui_output
      - /dev/shm/comfy_ui_temp
      - /dev/shm/comfy_ui_archive
      - extra_dirs (passados explicitamente)

    Recria os diretórios vazios após limpeza (para que o processo ComfyUI
    ainda ativo não encontre paths ausentes).

    Verifica que /kaggle/working não contém imagens sensíveis.
    """
    print("\n[CLEANUP] Iniciando secure_cleanup...")

    shm_dirs = [SHM_INPUT, SHM_OUTPUT, SHM_TEMP, SHM_ARCHIVE]
    if extra_dirs:
        shm_dirs.extend(extra_dirs)

    for d in shm_dirs:
        if d.exists():
            # Conta arquivos antes de limpar
            files = list(d.rglob("*"))
            img_files = [f for f in files if f.is_file() and f.suffix.lower() in SENSITIVE_EXTENSIONS]
            if img_files:
                print(f"[CLEANUP] {d}: removendo {len(img_files)} imagem(ns)")
            safe_remove(d)
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o777)
        print(f"[CLEANUP] ✓ {d} limpo")

    gc.collect()
    print("[CLEANUP] GC coletado")

    # Verificação final: /kaggle/working não deve conter imagens
    persistent_check = final_filesystem_check(silent=True)
    if persistent_check["violations"] > 0:
        print(f"[CLEANUP] ⚠️  {persistent_check['violations']} artefato(s) sensível(is) "
              f"ainda encontrado(s) em /kaggle/working — ver relatório abaixo")
        print(persistent_check["report"])
    else:
        print("[CLEANUP] ✓ /kaggle/working: zero artefatos sensíveis")

    print("[CLEANUP] secure_cleanup concluído\n")


def clear_input(input_dir: Path = SHM_INPUT) -> None:
    """
    Apaga imagens, masks e uploads do diretório de input volátil.
    Verifica que o diretório está vazio após limpeza.
    """
    assert_shm_path(input_dir, "input_dir em clear_input")
    if input_dir.exists():
        for f in input_dir.rglob("*"):
            if f.is_file():
                safe_remove(f)
        # Apaga subdiretórios vazios (ex: pasted/)
        for sub in sorted(input_dir.iterdir(), reverse=True):
            if sub.is_dir():
                try:
                    sub.rmdir()  # só remove se vazio
                except OSError:
                    pass
    remaining = list(input_dir.rglob("*")) if input_dir.exists() else []
    files_remaining = [f for f in remaining if f.is_file()]
    if files_remaining:
        raise SecurityError(
            f"clear_input: {len(files_remaining)} arquivo(s) não removido(s) em {input_dir}:\n"
            + "\n".join(f"  {f}" for f in files_remaining[:10])
        )
    print(f"[CLEANUP] ✓ {input_dir} limpo (zero arquivos)")


def clear_output(output_dir: Path = SHM_OUTPUT) -> None:
    """
    Apaga imagens e artefatos do diretório de output volátil.
    Deve ser chamado APÓS coleta/zip, nunca antes.
    """
    assert_shm_path(output_dir, "output_dir em clear_output")
    if output_dir.exists():
        for f in output_dir.rglob("*"):
            if f.is_file():
                safe_remove(f)
    remaining = [f for f in output_dir.rglob("*") if f.is_file()] if output_dir.exists() else []
    if remaining:
        raise SecurityError(
            f"clear_output: {len(remaining)} arquivo(s) não removido(s) em {output_dir}"
        )
    print(f"[CLEANUP] ✓ {output_dir} limpo (zero arquivos)")


def final_filesystem_check(
    scan_root: Path = Path("/kaggle/working"),
    silent: bool = False,
) -> dict:
    """
    Verifica recursivamente que scan_root não contém imagens nem arquivos sensíveis.
    Retorna dict com 'violations' (int) e 'report' (str).
    Se silent=False, imprime o relatório.
    """
    img_found = []
    arch_found = []

    if scan_root.exists():
        for f in scan_root.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in SENSITIVE_EXTENSIONS:
                stat = f.stat()
                img_found.append({
                    "path": str(f),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            elif ext in SENSITIVE_ARCHIVES:
                stat = f.stat()
                arch_found.append({
                    "path": str(f),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })

    violations = len(img_found) + len(arch_found)

    lines = [
        "",
        "=== FINAL SECURITY FILESYSTEM CHECK ===",
        f"",
        f"{scan_root} image files  : {len(img_found)}",
        f"{scan_root} archives     : {len(arch_found)}",
        f"Sensitive artifacts detected: {violations}",
        "",
    ]

    if violations == 0:
        lines.append("STATUS: PASS ✅")
    else:
        lines.append("STATUS: FAIL ❌")
        lines.append("SECURITY VIOLATION DETECTED")
        lines.append("")
        if img_found:
            lines.append("Images found:")
            for item in img_found:
                import datetime
                ts = datetime.datetime.fromtimestamp(item["mtime"]).isoformat()
                lines.append(f"  {item['path']}  ({item['size']} bytes, mtime={ts})")
        if arch_found:
            lines.append("Archives found:")
            for item in arch_found:
                import datetime
                ts = datetime.datetime.fromtimestamp(item["mtime"]).isoformat()
                lines.append(f"  {item['path']}  ({item['size']} bytes, mtime={ts})")

    lines.append("")
    report = "\n".join(lines)

    if not silent:
        print(report)

    return {"violations": violations, "images": img_found, "archives": arch_found, "report": report}


def verify_comfyui_paths(comfyui_dir: Path) -> None:
    """
    Verificação de segurança pós-startup via folder_paths do ComfyUI.
    Aborta se INPUT/OUTPUT/TEMP não estiverem em /dev/shm.
    """
    print("\n=== COMFYUI SECURITY CONFIGURATION ===")
    try:
        # folder_paths só existe no contexto do ComfyUI importado
        import folder_paths  # type: ignore[import]

        input_dir = Path(folder_paths.get_input_directory())
        output_dir = Path(folder_paths.get_output_directory())
        temp_dir = Path(folder_paths.get_temp_directory())
    except ImportError:
        # folder_paths não disponível fora do processo ComfyUI —
        # verificação via /proc/<pid>/cmdline é feita em verify_process_paths
        print("[INFO] folder_paths não disponível neste contexto (esperado fora do processo ComfyUI)")
        print(f"INPUT  = {SHM_INPUT}  (configurado)")
        print(f"TEMP   = {SHM_TEMP}   (configurado)")
        print(f"OUTPUT = {SHM_OUTPUT} (configurado)")
        print("=== END SECURITY CONFIGURATION ===\n")
        return

    print(f"INPUT  = {input_dir}")
    print(f"TEMP   = {temp_dir}")
    print(f"OUTPUT = {output_dir}")
    print("=== END SECURITY CONFIGURATION ===\n")

    errors = []
    for label, path in [("INPUT", input_dir), ("TEMP", temp_dir), ("OUTPUT", output_dir)]:
        if not str(path).startswith("/dev/shm"):
            errors.append(f"{label} está fora de /dev/shm: {path}")
        if "/kaggle/working" in str(path):
            errors.append(f"{label} aponta para /kaggle/working: {path}")

    if errors:
        _security_abort(
            "Paths do ComfyUI não estão em /dev/shm:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Verificação de processo existente
# ---------------------------------------------------------------------------

def _read_proc_cmdline(pid: int) -> list[str]:
    """Lê /proc/<pid>/cmdline e retorna lista de argumentos. Retorna [] se indisponível."""
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [a.decode("utf-8", errors="replace") for a in data.split(b"\x00") if a]
    except (OSError, PermissionError):
        return []


def _verify_process_paths(
    pid: int,
    expected_input: Path,
    expected_output: Path,
    expected_temp: Path,
    expected_host: str,
    expected_port: int,
) -> tuple[bool, str]:
    """
    Verifica via /proc/<pid>/cmdline que o processo usa exatamente os paths esperados.
    Retorna (ok: bool, reason: str).
    """
    cmdline = _read_proc_cmdline(pid)
    if not cmdline:
        return False, f"Não foi possível ler /proc/{pid}/cmdline — processo inacessível ou inexistente"

    cmd_str = " ".join(cmdline)

    checks = [
        ("--input-directory",  str(expected_input),  "input-directory"),
        ("--output-directory", str(expected_output), "output-directory"),
        ("--temp-directory",   str(expected_temp),   "temp-directory"),
        ("--listen",           expected_host,        "host"),
        ("--port",             str(expected_port),   "port"),
    ]

    issues = []
    for flag, expected_val, label in checks:
        # Verifica que a flag está presente e seguida pelo valor esperado
        try:
            idx = cmdline.index(flag)
            actual_val = cmdline[idx + 1] if idx + 1 < len(cmdline) else ""
            if actual_val != expected_val:
                issues.append(f"{label}: esperado '{expected_val}', encontrado '{actual_val}'")
        except ValueError:
            issues.append(f"flag '{flag}' ausente no cmdline do processo {pid}")

    # Verificação extra: nenhum path deve ser /kaggle/working
    for arg in cmdline:
        if "/kaggle/working" in arg and arg not in {str(DEFAULT_COMFYUI_DIR), str(DEFAULT_COMFYUI_DIR / "models")}:
            issues.append(f"Argumento contém /kaggle/working de forma inesperada: {arg}")

    if issues:
        return False, (
            f"Processo PID={pid} tem configuração incorreta:\n"
            + "\n".join(f"  - {i}" for i in issues)
            + f"\ncmdline completo: {cmd_str}"
        )

    return True, f"Processo PID={pid} verificado: paths corretos"


def find_existing_comfyui_pid(port: int = DEFAULT_PORT) -> Optional[int]:
    """Tenta encontrar o PID do processo ComfyUI ouvindo na porta especificada."""
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5
        )
        pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
        if pids:
            return pids[0]
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5
        )
        import re
        m = re.search(r"pid=(\d+)", result.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return None


def kill_mismatched_process(pid: int) -> None:
    """Mata processo com configuração incorreta. Aguarda encerramento."""
    import signal
    print(f"[SECURITY] Matando processo com configuração incorreta: PID={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        # Verifica se ainda existe
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass
        print(f"[SECURITY] PID={pid} encerrado")
    except ProcessLookupError:
        print(f"[INFO] PID={pid} já não existe")
    except PermissionError:
        print(f"[WARN] Sem permissão para matar PID={pid} — pode ser necessário reiniciar o kernel")


# ---------------------------------------------------------------------------
# Custom nodes — allowlist
# ---------------------------------------------------------------------------

def check_custom_nodes_allowlist(comfyui_dir: Path, strict: bool = True) -> list[str]:
    """
    Verifica se há custom nodes fora da ALLOWED_CUSTOM_NODES allowlist.
    Se strict=True e houver nodes não autorizados, levanta SecurityError.
    Retorna lista de nodes desconhecidos encontrados.
    """
    custom_dir = Path(comfyui_dir) / "custom_nodes"
    if not custom_dir.exists():
        return []

    unknown = []
    for item in custom_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("__"):
            continue
        if item.name not in ALLOWED_CUSTOM_NODES:
            unknown.append(item.name)

    if unknown:
        msg = (
            f"Custom nodes NÃO AUTORIZADOS detectados em {custom_dir}:\n"
            + "\n".join(f"  - {n}" for n in unknown)
            + "\n\nAllowlist atual: " + ", ".join(sorted(ALLOWED_CUSTOM_NODES))
            + "\n\nAdicione o node à ALLOWED_CUSTOM_NODES em comfyui_setup.py "
            "ou remova-o antes de continuar."
        )
        if strict:
            _security_abort(msg)
        else:
            print(f"[WARN] {msg}")

    return unknown


# ---------------------------------------------------------------------------
# GPU / device
# ---------------------------------------------------------------------------

def get_cuda_device(default: int = DEFAULT_CUDA_DEVICE) -> int:
    """Lê COMFYUI_CUDA_DEVICE (default 0)."""
    raw = os.environ.get(ENV_CUDA_DEVICE, str(default))
    try:
        device = int(str(raw).strip())
        if device < 0:
            raise ValueError("negativo")
        return device
    except (TypeError, ValueError):
        print(f"[WARN] {ENV_CUDA_DEVICE}={raw!r} inválido; usando {default}")
        return default


def detect_gpu() -> dict:
    """Delegado para gpu_detect. Fallback mínimo se ausente."""
    try:
        from gpu_detect import detect_gpu as _detect
        return _detect()
    except ImportError:
        info = {"has_gpu": False, "gpu_count": 0, "gpus": [], "gpu_name": "Unknown", "vram_gb": 0}
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": props.name,
                        "vram_gb": props.total_memory / (1024 ** 3),
                        "cuda": torch.version.cuda,
                        "driver": None,
                    })
                info.update({
                    "has_gpu": True,
                    "gpu_count": len(gpus),
                    "gpus": gpus,
                    "gpu_name": gpus[0]["name"],
                    "vram_gb": gpus[0]["vram_gb"],
                })
                print(f"[INFO] GPU(s): {info['gpu_count']} — padrão device={get_cuda_device()}")
        except Exception as exc:
            print(f"[WARN] Detecção de GPU falhou: {exc}")
        return info


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd, cwd=None, timeout=900, check=True):
    print("[CMD]", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr[-4000:])
        if check:
            raise RuntimeError(f"Comando falhou ({result.returncode}): {' '.join(map(str, cmd))}")
    return result


# ---------------------------------------------------------------------------
# Custom node helpers
# ---------------------------------------------------------------------------

def parse_custom_node_spec(spec: str) -> tuple[str, str, str]:
    """Retorna (repo_url, branch, node_name) a partir de user/repo[@branch] ou URL."""
    if "@" in spec and not spec.startswith("http"):
        repo_part, branch = spec.rsplit("@", 1)
    elif spec.startswith("http") and "@" in spec.rsplit("/", 1)[-1]:
        repo_part, branch = spec.rsplit("@", 1)
    else:
        repo_part, branch = spec, "main"
    repo = repo_part if repo_part.startswith("http") else f"https://github.com/{repo_part}.git"
    node_name = repo.rstrip("/").split("/")[-1].removesuffix(".git")
    return repo, branch, node_name


def is_manager_custom_node_spec(spec: str) -> bool:
    lower = spec.lower()
    return "comfyui-manager" in lower or "comfyui_manager" in lower


def filter_custom_nodes(custom_nodes: Optional[list[str]]) -> list[str]:
    """Remove ComfyUI-Manager da lista."""
    if not custom_nodes:
        return []
    filtered = []
    for spec in custom_nodes:
        if is_manager_custom_node_spec(spec):
            print(f"[WARN] Ignorando '{spec}': Manager integrado via --enable-manager.")
            continue
        filtered.append(spec)
    return filtered


def install_or_update_custom_node(custom_dir: Path, spec: str) -> Path:
    """Instala ou atualiza um custom node de forma idempotente."""
    custom_dir = Path(custom_dir)
    custom_dir.mkdir(parents=True, exist_ok=True)
    repo, branch, node_name = parse_custom_node_spec(spec)
    node_path = custom_dir / node_name

    if node_path.exists():
        if not (node_path / ".git").exists():
            raise RuntimeError(
                f"Custom node existente não é um checkout Git: {node_path}."
            )
        print(f"[INFO] Atualizando custom node: {node_name}")
        _run(["git", "pull", "--ff-only"], cwd=node_path, timeout=300, check=False)
    else:
        print(f"[INFO] Clonando custom node: {node_name} (branch={branch})")
        _run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo, str(node_path)],
            timeout=900,
        )

    node_req = node_path / "requirements.txt"
    if node_req.exists():
        _run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req)],
            timeout=900,
            check=False,
        )
    return node_path


def install_manager_requirements(comfyui_dir: Path) -> bool:
    """Instala ComfyUI/manager_requirements.txt (Manager integrado)."""
    comfyui_dir = Path(comfyui_dir)
    mgr_req = comfyui_dir / "manager_requirements.txt"
    if not mgr_req.exists():
        print(f"[WARN] {mgr_req} não encontrado — Manager integrado pode não ativar")
        return False
    print(f"[INFO] Instalando Manager integrado: {mgr_req}")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(mgr_req)], timeout=900)
    return True


# ---------------------------------------------------------------------------
# extra_model_paths.yaml
# ---------------------------------------------------------------------------

def build_extra_model_paths_yaml(
    models_dir: Path,
    additional_roots: Optional[list[tuple[str, Path]]] = None,
) -> str:
    """Gera conteúdo do extra_model_paths.yaml."""
    lines = ["kaggle_models:", f"  base_path: {models_dir}"]
    for cat in MODEL_CATEGORIES:
        lines.append(f"  {cat}: {cat}")

    if additional_roots:
        for name, root in additional_roots:
            root_path = Path(root)
            if not root_path.is_dir():
                print(f"[WARN] Raiz adicional '{name}' não existe: {root}. Pulando.")
                continue
            lines.append(f"{name}:")
            lines.append(f"  base_path: {root_path}")
            for cat in MODEL_CATEGORIES:
                lines.append(f"  {cat}: {cat}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Setup principal do ComfyUI
# ---------------------------------------------------------------------------

def setup_comfyui(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    repo_url=DEFAULT_REPO_URL,
    custom_nodes=None,
    models_dir=None,
    output_dir=None,
    input_dir=None,
    temp_dir=None,
    drive_base=DEFAULT_DRIVE_BASE,
    enable_manager: bool = True,
    additional_model_roots: Optional[list[tuple[str, Path]]] = None,
    strict_allowlist: bool = True,
):
    """
    Instala/atualiza ComfyUI e configura custom nodes.

    input_dir, output_dir, temp_dir: DEVEM estar em /dev/shm.
    Se não fornecidos, usam SHM_INPUT, SHM_OUTPUT, SHM_TEMP.
    """
    comfyui_dir = Path(comfyui_dir)
    models_dir = Path(models_dir) if models_dir else comfyui_dir / "models"

    # Resolver e validar dirs de I/O — FAIL CLOSED
    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP

    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")

    gpu_info = detect_gpu()
    cuda_device = get_cuda_device()
    if gpu_info.get("has_gpu"):
        print(
            f"[INFO] GPUs={gpu_info.get('gpu_count', 0)}; "
            f"ComfyUI usará --cuda-device {cuda_device}. VRAM não é somada entre placas."
        )

    is_git_repo = (comfyui_dir / ".git").exists()
    has_main = (comfyui_dir / "main.py").exists()

    if not comfyui_dir.exists():
        _run(["git", "clone", "--depth", "1", repo_url, str(comfyui_dir)], timeout=900)
    elif is_git_repo:
        _run(["git", "pull", "--ff-only"], cwd=comfyui_dir, timeout=300, check=False)
    elif has_main:
        print(f"[WARN] {comfyui_dir} tem main.py sem .git — pulando clone/pull")
    else:
        raise RuntimeError(
            f"Diretório {comfyui_dir} existe mas não é um checkout Git válido do ComfyUI."
        )

    # Provisionar dirs voláteis
    provision_shm_dirs(input_dir, output_dir, temp_dir, SHM_ARCHIVE)

    req = comfyui_dir / "requirements.txt"
    if req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=900)

    if enable_manager:
        install_manager_requirements(comfyui_dir)

    # mkdir apenas em models_dir (gravável). NUNCA em additional_roots (podem ser read-only).
    for cat in MODEL_CATEGORIES:
        (models_dir / cat).mkdir(parents=True, exist_ok=True)

    nodes = filter_custom_nodes(
        custom_nodes if custom_nodes is not None else list(DEFAULT_CUSTOM_NODES)
    )
    if nodes:
        custom_dir = comfyui_dir / "custom_nodes"
        for spec in nodes:
            node_name = parse_custom_node_spec(spec)[2]
            if node_name not in ALLOWED_CUSTOM_NODES:
                _security_abort(
                    f"Tentativa de instalar custom node não autorizado: '{node_name}'\n"
                    f"Allowlist: {sorted(ALLOWED_CUSTOM_NODES)}"
                )
            install_or_update_custom_node(custom_dir, spec)

    # Verificar custom nodes pré-existentes contra allowlist
    check_custom_nodes_allowlist(comfyui_dir, strict=strict_allowlist)

    extra_paths = comfyui_dir / "extra_model_paths.yaml"
    yaml_content = build_extra_model_paths_yaml(models_dir, additional_model_roots)
    extra_paths.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] (Re)escrito {extra_paths}")

    print(f"[INFO] ComfyUI: {comfyui_dir}")
    print(f"[INFO] Models: {models_dir}")
    print(f"[SECURITY] INPUT  → {input_dir} (tmpfs)")
    print(f"[SECURITY] OUTPUT → {output_dir} (tmpfs)")
    print(f"[SECURITY] TEMP   → {temp_dir} (tmpfs)")
    print(f"[INFO] Manager integrado: {'sim' if enable_manager else 'não'}")
    print(f"[INFO] CUDA device padrão: {cuda_device}")
    if gpu_info.get("has_gpu"):
        for g in gpu_info.get("gpus") or []:
            print(f"  GPU[{g.get('index')}] {g.get('name')} — {g.get('vram_gb', 0):.1f} GB")
    else:
        print("[INFO] GPU: CPU only")
    return comfyui_dir


# ---------------------------------------------------------------------------
# build_comfyui_command — agora inclui --input-directory
# ---------------------------------------------------------------------------

def build_comfyui_command(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """
    Monta o comando de start do ComfyUI.

    SEGURANÇA:
      --input-directory  → /dev/shm/comfy_ui_input  (obrigatório)
      --output-directory → /dev/shm/comfy_ui_output (obrigatório)
      --temp-directory   → /dev/shm/comfy_ui_temp   (obrigatório)
      --listen           → 127.0.0.1 apenas
      Sem CORS global. Sem flags de VRAM desencorajadas.
    """
    comfyui_dir = Path(comfyui_dir)
    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP

    # Fail-closed: valida antes de montar o comando
    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")

    if cuda_device is None:
        cuda_device = get_cuda_device()

    cmd = [
        sys.executable,
        "main.py",
        "--listen",
        host,
        "--port",
        str(port),
        "--input-directory",
        str(input_dir),
        "--output-directory",
        str(output_dir),
        "--temp-directory",
        str(temp_dir),
        "--cuda-device",
        str(cuda_device),
    ]
    if enable_manager:
        cmd.append("--enable-manager")

    if extra_args:
        for arg in extra_args:
            if arg in DISCOURAGED_VRAM_FLAGS or arg.startswith("--reserve-vram"):
                print(f"[WARN] Flag de VRAM desencorajada no pipeline padrão: {arg}")
            if arg in {"--enable-cors-header", "--cors-origins"} or arg.startswith("--enable-cors"):
                print(f"[WARN] CORS global não é padrão neste pipeline: {arg}")
            # Bloquear override dos paths de segurança via extra_args
            for blocked_flag in ("--input-directory", "--output-directory", "--temp-directory"):
                if arg == blocked_flag:
                    _security_abort(
                        f"Tentativa de sobrescrever {blocked_flag} via extra_args. "
                        "Use os parâmetros input_dir/output_dir/temp_dir de build_comfyui_command."
                    )
        cmd.extend(extra_args)

    return cmd


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------

def tail_log_file(log_path: Path, n: int = 40) -> str:
    log_path = Path(log_path)
    if not log_path.exists():
        return f"(log inexistente: {log_path})"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(falha ao ler log: {exc})"


# ---------------------------------------------------------------------------
# start_comfyui
# ---------------------------------------------------------------------------

def start_comfyui(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra_args=None,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
):
    main_py = Path(comfyui_dir) / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"ComfyUI não encontrado em {comfyui_dir}")

    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP

    # Garantir que dirs existem e têm permissão correta
    provision_shm_dirs(input_dir, output_dir, temp_dir, SHM_ARCHIVE)

    cmd = build_comfyui_command(
        comfyui_dir=comfyui_dir,
        host=host,
        port=port,
        output_dir=output_dir,
        input_dir=input_dir,
        temp_dir=temp_dir,
        cuda_device=cuda_device,
        enable_manager=enable_manager,
        extra_args=extra_args,
    )

    log_path = Path(comfyui_dir) / "comfyui.log"
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=comfyui_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[INFO] ComfyUI iniciado PID={proc.pid}; log={log_path}")
    print(f"[INFO] Listen: {host}:{port}")
    print(f"[SECURITY] INPUT  → {input_dir} (tmpfs)")
    print(f"[SECURITY] OUTPUT → {output_dir} (tmpfs)")
    print(f"[SECURITY] TEMP   → {temp_dir} (tmpfs)")
    print(f"[INFO] CUDA device: {cuda_device if cuda_device is not None else get_cuda_device()}")
    print(f"[INFO] Manager: {'--enable-manager' if enable_manager else 'desligado'}")
    print("[INFO] VRAM: DynamicVRAM padrão do ComfyUI (sem highvram/lowvram/novram)")
    return proc


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

def health_check(host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: int = 60) -> bool:
    import urllib.request

    start = time.time()
    url = f"http://{host}:{port}/system_stats"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print("[INFO] Health check OK")
                    return True
        except Exception:
            time.sleep(2)
    print(f"[ERROR] Health check falhou após {timeout}s ({url})")
    return False


# ---------------------------------------------------------------------------
# start_comfyui_runtime — reuse_existing=False por padrão
# ---------------------------------------------------------------------------

def start_comfyui_runtime(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    extra_args=None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    enable_ngrok: bool = False,
    health_timeout: int = 90,
    health_host: str = "127.0.0.1",
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """
    Ordem obrigatória: start → health → ngrok.

    SEGURANÇA:
      reuse_existing=False por padrão — sempre inicia processo novo.
      Se reuse_existing=True, verifica /proc/<pid>/cmdline antes de aceitar.
      Se paths divergem, mata o processo antigo e inicia novo.
      Nunca aceita "porta 8188 responde HTTP 200" como prova de isolamento.
      enable_ngrok=False por padrão — requer opt-in explícito.
    """
    comfyui_dir = Path(comfyui_dir)
    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP
    log_path = comfyui_dir / "comfyui.log"

    # Fail-closed
    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")

    result: dict[str, Any] = {
        "ok": False,
        "proc": None,
        "public_url": None,
        "local_url": f"http://{health_host}:{port}",
        "log_path": str(log_path),
        "health": False,
        "ngrok_started": False,
        "reused_existing": False,
        "pid": None,
    }

    # Verificar se já há processo rodando na porta
    existing_pid = find_existing_comfyui_pid(port)
    port_in_use = health_check(health_host, port, timeout=2)

    if port_in_use and existing_pid:
        if reuse_existing:
            # Verificar via /proc/<pid>/cmdline
            ok, reason = _verify_process_paths(
                pid=existing_pid,
                expected_input=input_dir,
                expected_output=output_dir,
                expected_temp=temp_dir,
                expected_host=host,
                expected_port=port,
            )
            if ok:
                print(f"[INFO] {reason}")
                print(f"[INFO] Reutilizando processo verificado PID={existing_pid}")
                result["health"] = True
                result["ok"] = True
                result["reused_existing"] = True
                result["pid"] = existing_pid
            else:
                print(f"[SECURITY] {reason}")
                print("[SECURITY] Matando processo com configuração incorreta...")
                kill_mismatched_process(existing_pid)
                # Agora inicia processo novo
                port_in_use = False
        else:
            # reuse_existing=False — matar processo existente e iniciar novo
            print(f"[INFO] reuse_existing=False: matando processo existente PID={existing_pid}")
            kill_mismatched_process(existing_pid)
            port_in_use = False
    elif port_in_use and not existing_pid:
        # Porta em uso mas não conseguimos identificar o PID — não é seguro continuar
        _security_abort(
            f"Porta {port} está em uso mas não foi possível identificar o processo.\n"
            "Reinicie o kernel para garantir um estado limpo."
        )

    if not port_in_use:
        proc = start_comfyui(
            comfyui_dir=comfyui_dir,
            host=host,
            port=port,
            extra_args=extra_args,
            output_dir=output_dir,
            input_dir=input_dir,
            temp_dir=temp_dir,
            cuda_device=cuda_device,
            enable_manager=enable_manager,
        )
        result["proc"] = proc
        result["pid"] = proc.pid

        healthy = health_check(health_host, port, timeout=health_timeout)
        result["health"] = healthy
        if not healthy:
            print(f"[ERROR] ComfyUI falhou no health check. Log: {log_path}")
            print("======== ÚLTIMAS LINHAS DO LOG ========")
            print(tail_log_file(log_path, n=50))
            print("=======================================")
            print("[SECURITY] ngrok NÃO será iniciado (health check falhou)")
            return result

        result["ok"] = True

    if enable_ngrok:
        try:
            from ngrok_tunnel import start_ngrok_tunnel
            public_url = start_ngrok_tunnel(port=port)
            result["public_url"] = public_url
            result["ngrok_started"] = True
        except Exception as exc:
            msg = str(exc)
            try:
                from ngrok_tunnel import redact_secrets
                msg = redact_secrets(msg)
            except Exception:
                pass
            print(f"[WARN] ngrok não iniciado: {msg}")
    else:
        print("[INFO] ngrok desabilitado (enable_ngrok=False). Acesso somente via 127.0.0.1")

    print("=" * 60)
    print("COMFYUI READY — ZERO-PERSISTENT-IMAGE MODE")
    print(f"Local  : {result['local_url']}")
    print(f"Public : {result['public_url'] or '(ngrok desabilitado — acesso local apenas)'}")
    print(f"INPUT  : {input_dir} (tmpfs)")
    print(f"OUTPUT : {output_dir} (tmpfs)")
    print(f"TEMP   : {temp_dir} (tmpfs)")
    print(f"PID    : {result['pid']}")
    print("=" * 60)
    return result


# ---------------------------------------------------------------------------
# ZIP seguro
# ---------------------------------------------------------------------------

def create_secure_zip(
    src_dir: Path,
    zip_password: Optional[str] = None,
    archive_dir: Path = SHM_ARCHIVE,
    zip_name: str = "output.zip",
) -> Path:
    """
    Cria ZIP criptografado com AES-256 (via pyzipper) em /dev/shm/comfy_ui_archive/.

    NUNCA cria o ZIP em /kaggle/working.
    Se zip_password não fornecido, tenta SECRET_ZIP_PASSWORD env/kaggle secret.
    Aborta se senha indisponível (fail-closed).

    Retorna path do ZIP criado.
    """
    assert_shm_path(archive_dir, "archive_dir do ZIP")

    # Resolver senha
    if not zip_password:
        zip_password = os.environ.get("SECRET_ZIP_PASSWORD") or os.environ.get("ZIP_PASSWORD")
    if not zip_password:
        try:
            from kaggle_secrets import UserSecretsClient
            zip_password = UserSecretsClient().get_secret("SECRET_ZIP_PASSWORD")
        except Exception:
            pass
    if not zip_password:
        _security_abort(
            "ZIP_PASSWORD / SECRET_ZIP_PASSWORD não encontrado em env ou Kaggle Secrets.\n"
            "Configure o Secret 'SECRET_ZIP_PASSWORD' antes de criar o ZIP criptografado.\n"
            "Não será criado ZIP sem senha como fallback."
        )

    # Instalar pyzipper se necessário
    try:
        import pyzipper  # type: ignore[import]
    except ImportError:
        print("[INFO] Instalando pyzipper para ZIP AES-256...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "pyzipper"],
            check=True, timeout=300,
        )
        import pyzipper  # type: ignore[import]

    archive_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(archive_dir, 0o700)  # Restrito: só o processo dono
    zip_path = archive_dir / zip_name

    src_dir = Path(src_dir)
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    if not files:
        print("[WARN] Nenhum arquivo em src_dir para compactar.")
        return zip_path

    with pyzipper.AESZipFile(
        zip_path,
        "w",
        compression=pyzipper.ZIP_DEFLATED,
        encryption=pyzipper.WZ_AES,
    ) as zf:
        zf.setpassword(zip_password.encode("utf-8"))
        for f in files:
            zf.write(f, arcname=str(f.relative_to(src_dir)))

    size_mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"[SECURITY] ZIP AES-256 criado: {zip_path} ({size_mb:.1f} MB, {len(files)} arquivo(s))")
    print(f"[SECURITY] ZIP está em {archive_dir} (tmpfs) — NUNCA em /kaggle/working")
    return zip_path


def cleanup_zip(zip_path: Path) -> None:
    """Remove o ZIP e verifica a remoção. Idempotente."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        safe_remove(zip_path)
        print(f"[CLEANUP] ✓ ZIP removido: {zip_path}")
    else:
        print(f"[CLEANUP] ZIP já não existe: {zip_path}")


# ---------------------------------------------------------------------------
# credential cleanup
# ---------------------------------------------------------------------------

def cleanup_gdrive_credentials() -> None:
    """
    Apaga service account JSON e rclone config após uso.
    Chame depois de montar o Drive.
    """
    sa_path = Path("/root/gdrive_sa.json")
    rclone_conf = Path("/root/.config/rclone/rclone.conf")

    for p in (sa_path, rclone_conf):
        if p.exists():
            safe_remove(p)
            print(f"[SECURITY] Credencial removida: {p}")
        else:
            print(f"[INFO] Credencial já não existe: {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", default=str(DEFAULT_COMFYUI_DIR))
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--custom-nodes", nargs="*")
    parser.add_argument("--models-dir")
    parser.add_argument("--output-dir", help="Deve estar em /dev/shm/")
    parser.add_argument("--input-dir", help="Deve estar em /dev/shm/")
    parser.add_argument("--temp-dir", help="Deve estar em /dev/shm/")
    parser.add_argument("--drive-base", default=DEFAULT_DRIVE_BASE)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--ngrok", action="store_true", help="Após health OK, abrir túnel ngrok")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--no-manager", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Reutilizar processo existente SE verificado via cmdline")
    args = parser.parse_args()

    if args.cuda_device is not None:
        os.environ[ENV_CUDA_DEVICE] = str(args.cuda_device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    nodes = args.custom_nodes if args.custom_nodes is not None else list(DEFAULT_CUSTOM_NODES)
    comfyui = setup_comfyui(
        Path(args.comfyui_dir),
        args.repo_url,
        nodes,
        Path(args.models_dir) if args.models_dir else None,
        output_dir=output_dir,
        input_dir=input_dir,
        temp_dir=temp_dir,
        drive_base=args.drive_base,
        enable_manager=not args.no_manager,
    )
    if args.start:
        if args.ngrok or args.health_check:
            runtime = start_comfyui_runtime(
                comfyui_dir=comfyui,
                host=args.host,
                port=args.port,
                output_dir=output_dir,
                input_dir=input_dir,
                temp_dir=temp_dir,
                enable_manager=not args.no_manager,
                enable_ngrok=args.ngrok,
                health_timeout=90,
                reuse_existing=args.reuse_existing,
            )
            if not runtime["health"]:
                raise RuntimeError("Health check falhou")
            try:
                if runtime["proc"]:
                    runtime["proc"].wait()
            except KeyboardInterrupt:
                if runtime["proc"]:
                    runtime["proc"].terminate()
                try:
                    from ngrok_tunnel import stop_ngrok_tunnel
                    stop_ngrok_tunnel()
                except Exception:
                    pass
        else:
            proc = start_comfyui(
                comfyui,
                args.host,
                args.port,
                output_dir=output_dir,
                input_dir=input_dir,
                temp_dir=temp_dir,
                enable_manager=not args.no_manager,
            )
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()


if __name__ == "__main__":
    main()
