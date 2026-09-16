#!/usr/bin/env python3
"""Setup do ComfyUI no Kaggle Notebook — zero-trust / zero-persistent-image.

GARANTIAS (fail-closed, não absolutas):
  INPUT   → /dev/shm/comfy_ui_input   (tmpfs volátil)
  OUTPUT  → /dev/shm/comfy_ui_output  (tmpfs volátil)
  TEMP    → /dev/shm/comfy_ui_temp    (tmpfs volátil)
  USER    → /dev/shm/comfy_ui_user    (tmpfs volátil)
  LOGS    → /dev/shm/comfy_ui_logs    (tmpfs volátil)
  ARCHIVE → /dev/shm/comfy_ui_archive (tmpfs volátil)

Manager e ngrok são PERMITIDOS em SECURE_MODE:
  - Manager roda com state/downloads/custom nodes em /dev/shm
  - ngrok roda após health check, token via Kaggle Secrets
  - A segurança vem do isolamento de filesystem, não do bloqueio de funcionalidade

Nenhuma imagem controlada pelo pipeline toca /kaggle/working.
Qualquer tentativa de escrever imagem/ZIP em /kaggle/working levanta SecurityError.
reuse_existing=False é obrigatório em SECURE_MODE.

LIMITES EXPLÍCITOS (fora do controle deste código):
  - Infraestrutura Kaggle / acesso privilegiado do provedor ao host.
  - Vulnerabilidades em dependências externas (pyzipper, pyngrok, ComfyUI).
  - Snapshots automáticos da plataforma Kaggle do working directory.
  - Custom nodes e código instalado pelo Manager executam Python arbitrário.
  - ngrok cria exposição externa — não é uma barreira de segurança.
  - Custom nodes podem fazer requests externos e acessar dados em memória.

NÃO DECLARAR "segurança absoluta". Objetivo: prevenir persistência acidental
nos caminhos controlados pelo pipeline, com fail-closed e verificação contínua.
"""
from __future__ import annotations

import datetime
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_COMFYUI_DIR = Path("/kaggle/working/ComfyUI")
DEFAULT_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
ENV_CUDA_DEVICE = "COMFYUI_CUDA_DEVICE"
DEFAULT_CUDA_DEVICE = 0

# ---------------------------------------------------------------------------
# SECURE MODE — configuração global fail-closed
# ---------------------------------------------------------------------------
# Quando True (padrão no Kaggle Seguro):
#   - enable_manager PERMITIDO (rodando com isolamento de filesystem)
#   - enable_ngrok PERMITIDO (rodando após health check)
#   - reuse_existing forçado para False
#   - todos os paths mutáveis (input, output, temp, user, logs) redirecionados para /dev/shm
#   - escrita em /kaggle/working bloqueada exceto via secure_persistent_write()
# Lido do ambiente: COMFYUI_SECURE_MODE=1 → True; 0 → False.
# Padrão: "1".
_SECURE_MODE: bool = os.environ.get("COMFYUI_SECURE_MODE", "1") == "1"


def get_secure_mode() -> bool:
    return _SECURE_MODE


def set_secure_mode(value: bool, _test_override: bool = False) -> None:
    global _SECURE_MODE
    if not _test_override and value is False and _SECURE_MODE:
        _security_abort("SECURE_MODE é fail-closed e não pode ser desativado em produção.")
    _SECURE_MODE = value


# ---------------------------------------------------------------------------
# Diretórios voláteis (tmpfs) — ÚNICA localização válida para I/O mutável
# ---------------------------------------------------------------------------
SHM_BASE = Path("/dev/shm")
SHM_INPUT = SHM_BASE / "comfy_ui_input"
SHM_OUTPUT = SHM_BASE / "comfy_ui_output"
SHM_TEMP = SHM_BASE / "comfy_ui_temp"
SHM_USER = SHM_BASE / "comfy_ui_user"
SHM_LOGS = SHM_BASE / "comfy_ui_logs"
SHM_ARCHIVE = SHM_BASE / "comfy_ui_archive"

# ---------------------------------------------------------------------------
# Persistência controlada
# ---------------------------------------------------------------------------
PERSISTENT_WORKING = Path("/kaggle/working")
SECURE_PERSISTENT_FILE = PERSISTENT_WORKING / "output_secure.zip"


def secure_persistent_write(src_path: Path, dst_path: Path = SECURE_PERSISTENT_FILE) -> None:
    """
    Única função autorizada para gravar em /kaggle/working.
    Aceita APENAS o arquivo output_secure.zip.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if dst_path != SECURE_PERSISTENT_FILE:
        _security_abort(f"Escrita persistente proibida: {dst_path}. Apenas {SECURE_PERSISTENT_FILE} é permitido.")

    if not src_path.exists():
        _security_abort(f"Origem para escrita persistente não encontrada: {src_path}")

    # Garantir que a origem está em /dev/shm (não persistente)
    assert_shm_path(src_path, "origem do persist_write")

    print(f"[SECURITY] Gravando artefato persistente: {dst_path} (from {src_path})")
    try:
        shutil.copy2(src_path, dst_path)
        # Verificar integridade após cópia
        if dst_path.stat().st_size != src_path.stat().st_size:
            raise RuntimeError("Falha na integridade da cópia persistente")
    except Exception as e:
        _security_abort(f"Falha ao gravar arquivo persistente: {e}")


# ---------------------------------------------------------------------------
# Invariantes e Guardrails de Filesystem
# ---------------------------------------------------------------------------

# Snapshot do estado de /kaggle/working no início da sessão
_WORKING_SNAPSHOT: Optional[set] = None

# Extensões não-imagem que também são consideradas artefatos sensíveis
SENSITIVE_NON_IMAGE_EXTENSIONS = frozenset({
    ".json", ".txt", ".log", ".db", ".sqlite", ".cache",
    ".tmp", ".latent", ".pt", ".pth", ".safetensors", ".bin",
})


def record_working_snapshot() -> set:
    """
    Registra snapshot dos arquivos em /kaggle/working no início da sessão.
    Deve ser chamada antes de iniciar o ComfyUI.
    Retorna o conjunto de paths relativos encontrados.
    """
    global _WORKING_SNAPSHOT
    snapshot: set = set()
    working = PERSISTENT_WORKING
    if working.exists():
        for item in working.rglob("*"):
            if item.is_file():
                try:
                    rel = str(item.relative_to(working))
                    snapshot.add(rel)
                except ValueError:
                    snapshot.add(str(item))
    _WORKING_SNAPSHOT = snapshot
    print(f"[SECURITY] Working snapshot: {len(snapshot)} arquivo(s) registrado(s) em /kaggle/working")
    return snapshot


def assert_working_clean() -> None:
    """
    Verifica que /kaggle/working não tem novos arquivos desde o snapshot.
    Deve ser chamada antes da geração.
    """
    if _WORKING_SNAPSHOT is None:
        record_working_snapshot()
    current = set()
    working = PERSISTENT_WORKING
    if working.exists():
        for item in working.rglob("*"):
            if item.is_file():
                try:
                    rel = str(item.relative_to(working))
                    current.add(rel)
                except ValueError:
                    current.add(str(item))
    new_files = current - _WORKING_SNAPSHOT
    # output_secure.zip é permitido
    new_files.discard("output_secure.zip")
    if new_files:
        _security_abort(
            f"assert_working_clean: {len(new_files)} arquivo(s) novo(s) detectado(s) em /kaggle/working:\n"
            + "\n".join(f"  - {f}" for f in sorted(new_files)[:20])
        )
    print(f"[SECURITY] assert_working_clean: PASS (zero arquivos novos)")


def assert_working_policy() -> None:
    """
    Verifica que /kaggle/working contém apenas arquivos permitidos.
    Levanta SecurityError se encontrar qualquer artefato sensível.
    Diferente de assert_no_persistent_images, verifica também extensões não-imagem.
    """
    scan_roots = list(PERSISTENT_AUDIT_PATHS)
    comfyui_persistent_dirs = [
        Path("/kaggle/working/ComfyUI/input"),
        Path("/kaggle/working/ComfyUI/output"),
        Path("/kaggle/working/ComfyUI/temp"),
        Path("/kaggle/working/ComfyUI/user"),
    ]
    for d in comfyui_persistent_dirs:
        if d.exists() and d not in scan_roots:
            scan_roots.append(d)

    violations: List[Dict[str, Any]] = []

    for root in scan_roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            if not item.is_file():
                continue
            # output_secure.zip é o único artefato persistente permitido
            try:
                rel = item.relative_to(PERSISTENT_WORKING)
                if str(rel) == "output_secure.zip":
                    continue
            except ValueError:
                pass

            stat = item.stat()
            ext = item.suffix.lower()

            if ext in SENSITIVE_EXTENSIONS or ext in SENSITIVE_ARCHIVES:
                violations.append({
                    "path": str(item), "type": "extension", "ext": ext,
                    "size": stat.st_size, "mtime": stat.st_mtime,
                })
                continue

            if ext in SENSITIVE_NON_IMAGE_EXTENSIONS:
                violations.append({
                    "path": str(item), "type": "non_image_sensitive", "ext": ext,
                    "size": stat.st_size, "mtime": stat.st_mtime,
                })
                continue

            # Verificação por magic bytes
            try:
                magic = item.read_bytes()[:16]
                is_image = any([
                    magic[:8] == b"\x89PNG\r\n\x1a\n",
                    magic[:3] == b"\xff\xd8\xff",
                    magic[:4] == b"RIFF" and magic[8:12] == b"WEBP",
                    magic[:6] in (b"GIF87a", b"GIF89a"),
                    magic[:4] in (b"PK\x03\x04", b"PK\x05\x06"),
                ])
                if is_image:
                    violations.append({
                        "path": str(item), "type": "magic_bytes",
                        "magic": magic[:8].hex(), "size": stat.st_size,
                    })
            except (OSError, PermissionError):
                pass

    if violations:
        lines = [f"assert_working_policy: {len(violations)} violação(ões) de política em /kaggle/working:"]
        for v in violations:
            lines.append(f"  PATH : {v['path']}")
            lines.append(f"  TYPE : {v['type']}")
            lines.append(f"  SIZE : {v.get('size', '?')} bytes")
            lines.append("")
        _security_abort("\n".join(lines))
    print(f"[SECURITY] assert_working_policy: PASS (zero artefatos proibidos)")


def assert_only_allowed_persistent_artifact() -> None:
    """
    Verifica que o único arquivo persistente em /kaggle/working (além do snapshot inicial)
    é output_secure.zip. Levanta SecurityError se houver qualquer outro.
    """
    if _WORKING_SNAPSHOT is None:
        record_working_snapshot()
    current: set = set()
    working = PERSISTENT_WORKING
    if working.exists():
        for item in working.rglob("*"):
            if item.is_file():
                try:
                    rel = str(item.relative_to(working))
                    current.add(rel)
                except ValueError:
                    current.add(str(item))
    new_files = current - _WORKING_SNAPSHOT
    # output_secure.zip é permitido
    new_files.discard("output_secure.zip")
    if new_files:
        _security_abort(
            f"assert_only_allowed_persistent_artifact: {len(new_files)} arquivo(s) não autorizado(s) em /kaggle/working:\n"
            + "\n".join(f"  - {f}" for f in sorted(new_files)[:20])
            + "\nApenas output_secure.zip é permitido como artefato persistente."
        )
    print(f"[SECURITY] assert_only_allowed_persistent_artifact: PASS")


def validate_runtime_path(path: Path, allowed_roots: Optional[List[Path]] = None) -> Path:
    """
    Valida que um path de runtime está dentro de uma raiz permitida.
    Rejeita: path traversal (..), symlinks para fora da raiz, caminhos absolutos arbitrários.

    allowed_roots: lista de raízes permitidas. Default: [SHM_BASE].
    Retorna o path resolvido se válido.
    Levanta SecurityError se o path escapa da raiz.
    """
    path = Path(path)
    roots = allowed_roots or [SHM_BASE]

    # Rejeição de traversal '..'
    if ".." in path.parts:
        _security_abort(f"validate_runtime_path: traversal '..' detectado: {path}")

    # Verificação léxica: path deve começar com uma das raízes
    path_str = str(path)
    in_root = False
    matched_root = None
    for root in roots:
        root_str = str(root)
        try:
            common = os.path.commonpath([root_str, path_str])
        except ValueError:
            continue
        if common == root_str:
            in_root = True
            matched_root = root
            break

    if not in_root:
        _security_abort(
            f"validate_runtime_path: path fora das raízes permitidas: {path}\n"
            f"Raízes: {[str(r) for r in roots]}"
        )

    # Resolução simbólica — segue symlinks
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path

    resolved_str = str(resolved)

    # Rejeitar se resolve para /kaggle/working
    if "/kaggle/working" in resolved_str and PERSISTENT_WORKING not in roots:
        _security_abort(
            f"validate_runtime_path: path resolve para /kaggle/working: {path} → {resolved}"
        )

    # Verificar que resolved também está em uma raiz permitida
    resolved_in_root = False
    for root in roots:
        root_str = str(root)
        try:
            common = os.path.commonpath([root_str, resolved_str])
        except ValueError:
            continue
        if common == root_str:
            resolved_in_root = True
            break

    if not resolved_in_root:
        _security_abort(
            f"validate_runtime_path: path resolvido fora das raízes: {path} → {resolved}"
        )

    # Rejeitar symlink que aponta para fora
    if path.is_symlink():
        try:
            symlink_target = path.resolve(strict=False)
        except Exception:
            symlink_target = path
        target_str = str(symlink_target)
        target_in_root = False
        for root in roots:
            root_str = str(root)
            try:
                common = os.path.commonpath([root_str, target_str])
            except ValueError:
                continue
            if common == root_str:
                target_in_root = True
                break
        if not target_in_root:
            _security_abort(
                f"validate_runtime_path: symlink aponta para fora das raízes: {path} → {symlink_target}"
            )

    return resolved


def assert_invariants(
    input_dir: Path = SHM_INPUT,
    output_dir: Path = SHM_OUTPUT,
    temp_dir: Path = SHM_TEMP,
    user_dir: Path = SHM_USER,
    log_dir: Path = SHM_LOGS,
    archive_dir: Path = SHM_ARCHIVE,
) -> None:
    """
    Valida todos os diretórios efetivos usados pelo processo ComfyUI.
    Deve ser chamada antes do start.
    Levanta SecurityError se qualquer diretório mutável estiver fora de /dev/shm.
    """
    mutable_dirs = {
        "input": input_dir,
        "output": output_dir,
        "temp": temp_dir,
        "user": user_dir,
        "logs": log_dir,
        "archive": archive_dir,
    }
    for name, d in mutable_dirs.items():
        d = Path(d)
        validate_runtime_path(d, allowed_roots=[SHM_BASE])
        # Garantir que não aponta para /kaggle/working
        if str(d).startswith("/kaggle/working"):
            _security_abort(
                f"assert_invariants: {name} está em /kaggle/working: {d}\n"
                "Diretórios mutáveis devem estar em /dev/shm."
            )
    print(f"[SECURITY] assert_invariants: PASS — todos os dirs mutáveis em /dev/shm")


def assert_comfy_process_isolation(
    pid: Optional[int] = None,
    port: int = DEFAULT_PORT,
    input_dir: Path = SHM_INPUT,
    output_dir: Path = SHM_OUTPUT,
    temp_dir: Path = SHM_TEMP,
    user_dir: Path = SHM_USER,
    log_dir: Path = SHM_LOGS,
) -> Dict[str, Any]:
    """
    Verifica que o processo ComfyUI está isolado:
    - PID, cmdline, cwd obtidos e registrados
    - input/output/temp/user/log dirs todos em /dev/shm
    - nenhum caminho mutável em /kaggle/working

    Levanta SecurityError se qualquer verificação falhar.
    Retorna dict com informações do processo.
    """
    if pid is None:
        pid = find_existing_comfyui_pid(port)
    if not pid:
        _security_abort("assert_comfy_process_isolation: ComfyUI não encontrado na porta")

    cmdline = _read_proc_cmdline(pid)
    if not cmdline:
        _security_abort(f"assert_comfy_process_isolation: não foi possível ler /proc/{pid}/cmdline")

    # Verificar cwd
    cwd = None
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve(strict=False)
    except Exception:
        cwd = None

    # Paths esperados
    expected_paths = {
        "--input-directory": input_dir,
        "--output-directory": output_dir,
        "--temp-directory": temp_dir,
        "--user-directory": user_dir,
    }

    process_info: Dict[str, Any] = {
        "pid": pid,
        "cmdline": cmdline,
        "cwd": str(cwd) if cwd else None,
    }

    # Verificar cada flag no cmdline
    for flag, expected_dir in expected_paths.items():
        if flag in cmdline:
            try:
                idx = cmdline.index(flag)
                actual = cmdline[idx + 1] if idx + 1 < len(cmdline) else ""
            except (ValueError, IndexError):
                actual = ""
            process_info[flag] = actual
            if actual and not actual.startswith("/dev/shm"):
                _security_abort(
                    f"assert_comfy_process_isolation: {flag}={actual} não está em /dev/shm"
                )
            if actual and "/kaggle/working" in actual:
                _security_abort(
                    f"assert_comfy_process_isolation: {flag}={actual} aponta para /kaggle/working"
                )

    # Verificar que nenhum argumento contém /kaggle/working (exceto comfyui_dir/models)
    allowed_persistent_refs = {
        str(DEFAULT_COMFYUI_DIR),
        str(DEFAULT_COMFYUI_DIR / "models"),
        str(DEFAULT_COMFYUI_DIR / "extra_model_paths.yaml"),
    }
    for arg in cmdline:
        if "/kaggle/working" in arg and arg not in allowed_persistent_refs:
            _security_abort(
                f"assert_comfy_process_isolation: argumento suspeito em /kaggle/working: {arg}"
            )

    # Log dir não deve estar em cmdline mas deve estar em /dev/shm
    if log_dir and str(log_dir).startswith("/kaggle/working"):
        _security_abort(
            f"assert_comfy_process_isolation: log_dir={log_dir} está em /kaggle/working"
        )
    process_info["log_dir"] = str(log_dir)

    print(f"[SECURITY] assert_comfy_process_isolation: PASS — PID={pid} isolado")
    return process_info

# ---------------------------------------------------------------------------
# Custom nodes — allowlist imutável com expected file/directory hash
# ---------------------------------------------------------------------------
# ALLOWED_CUSTOM_NODES: frozenset imutável. Não modificável em runtime.
# Qualquer tentativa de instalar node fora desta allowlist → SecurityError.
ALLOWED_CUSTOM_NODES: frozenset[str] = frozenset([
    "ComfyUI_essentials",
    "comfyui-krea2edit",
    "ComfyUI-Krea2T-Enhancer",
    "rgthree-comfy",
])

# EXPECTED_CUSTOM_NODE_HASHES: hash SHA-256 esperado do diretório de cada node.
# Populado após primeira snapshot em setup_comfyui(). Se vazio, skip de hash estático.
# Verificado em startup via verify_custom_nodes_unchanged(strict=True).
# Para configurar: execute setup_comfyui() uma vez, capture o hash via compute_node_directory_hash(),
# e preencha este dict. Exemplo:
# EXPECTED_CUSTOM_NODE_HASHES = {
#     "ComfyUI_essentials": "a1b2c3d4...",
#     "comfyui-krea2edit": "e5f6g7h8...",
#     "ComfyUI-Krea2T-Enhancer": "i9j0k1l2...",
#     "rgthree-comfy": "m3n4o5p6...",
# }
EXPECTED_CUSTOM_NODE_HASHES: Dict[str, str] = {}

# api.comfy.org e outros domínios externos são bloqueados em SECURE_MODE
BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "api.comfy.org",
    "comfy-org.gho.io",
    "raw.githubusercontent.com",
})

DEFAULT_CUSTOM_NODES: List[str] = [
    "cubiq/ComfyUI_essentials",
    "lbouaraba/comfyui-krea2edit",
]

MODEL_CATEGORIES: List[str] = [
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
]

DISCOURAGED_VRAM_FLAGS = {
    "--highvram", "--gpu-only", "--lowvram", "--novram",
    "--fast", "--reserve-vram",
}

# Extensões de imagem/archive que NUNCA devem aparecer em /kaggle/working
SENSITIVE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".gif",
    ".bmp", ".tif", ".tiff",
})
SENSITIVE_ARCHIVES = frozenset({".zip", ".7z", ".rar", ".tar", ".gz"})

# Paths persistentes que NUNCA devem conter imagens
PERSISTENT_AUDIT_PATHS: Tuple[Path, ...] = (
    Path("/kaggle/working"),
)

# ---------------------------------------------------------------------------
# Security primitives
# ---------------------------------------------------------------------------


class SecurityError(RuntimeError):
    """Levantada quando uma violação de segurança é detectada. Não deve ser silenciada."""


def _security_abort(msg: str) -> None:
    """Imprime mensagem crítica e levanta SecurityError. Nunca continua silenciosamente."""
    banner = "=" * 60
    print(f"\n{banner}")
    print("SECURITY VIOLATION — PIPELINE ABORTED")
    print(f"REASON: {msg}")
    print(f"{banner}\n")
    raise SecurityError(msg)


def assert_shm_path(path: Path, label: str) -> None:
    """
    Garante que path está em /dev/shm e não atravessa symlinks para /kaggle/working.

    Uso de resolve(strict=False):
      - strict=False evita FileNotFoundError em paths inexistentes (durante setup,
        antes de mkdir). Resolvemos oque é possível e rejeitamos qualquer escape.

    commonpath:
      - Verifica containment POSIX-style no path original (funciona cross-platform
        sem depender de filesystem real). Fail-closed: mismatch → abort.

    Rejeição de symlink:
      - Se path.is_symlink() → o symlink deve resolver para dentro de /dev/shm.
      - Symlink que resolve para fora /dev/shm é abortado.

    Rejeição de traversal:
      - ".." presente em path.parts → abortado imediatamente.
      - Resolve segue links; se o resolved path sai de /dev/shm → abortado.

    Fail-closed: qualquer dúvida é abortada, nunca permissiva.
    """
    path = Path(path)
    path_str = str(path)
    shm_str = str(SHM_BASE)

    # /dev/shm obrigatório — commonpath no path original (não resolve)
    # No Windows, /dev/shm não existe, então usamos verificação léxica estrita
    try:
        common = os.path.commonpath([shm_str, path_str])
    except ValueError:
        _security_abort(
            f"{label} path fora de /dev/shm (commonpath falhou — drives divergentes): {path}"
        )
    if common != shm_str:
        _security_abort(
            f"{label} está FORA de /dev/shm: {path}\n"
            "Nenhuma imagem deve ser processada em disco persistente."
        )

    # Rejeição de traversal '..' (fail-closed)
    if ".." in path.parts:
        _security_abort(f"{label} contém traversal '..' no path: {path}")

    # Resolução simbólica — resolve(strict=False) resiste a symlink bypass
    # Em Linux real, resolve followa symlinks; se o target sai de /dev/shm, aborta.
    # No Windows, /dev/shm não existe; pulamos a verificação de resolved path
    # para paths que não existem, já que resolve() converteria para C:\dev\shm\...
    is_windows = os.name == "nt"
    path_exists = path.exists()

    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path

    resolved_str = str(resolved)

    # Rejeição de symlink bypass para /kaggle/working
    if "/kaggle/working" in resolved_str:
        _security_abort(
            f"{label} resolve para /kaggle/working via symlink/traversal: {path} → {resolved}\n"
            "Symlink bypass detectado."
        )

    # No Windows, se o path não existe, pulamos a verificação de resolved path
    # pois resolve() em Windows converte /dev/shm para C:\dev\shm\...
    if not (is_windows and not path_exists):
        # Rejeição de symlink cujo target sai de /dev/shm
        if path.is_symlink():
            try:
                symlink_target = path.resolve(strict=False)
            except Exception:
                symlink_target = path
            if str(symlink_target) != path_str and not str(symlink_target).startswith("/dev/shm"):
                _security_abort(
                    f"{label} é um symlink para fora de /dev/shm: {path} → {symlink_target}"
                )

        # Rejeição de traversal no *resolved* path (commonpath pós-resolve)
        try:
            resolved_common = os.path.commonpath([shm_str, resolved_str])
        except ValueError:
            resolved_common = ""
        if resolved_common != shm_str:
            _security_abort(
                f"{label} resolve fora de /dev/shm: {path} → {resolved}\n"
                "Traversal via symlink detectado."
            )


def assert_no_persistent_images(
    label: str = "",
    extra_paths: Optional[List[Path]] = None,
) -> None:
    """
    Verifica que NENHUMA imagem ou archive sensível existe em /kaggle/working.
    Deve ser chamada antes e depois de cada geração, antes de criar ZIP,
    após download e no cleanup.

    Levanta SecurityError imediatamente ao encontrar qualquer arquivo sensível.
    Verifica também symlinks, arquivos ocultos e arquivos sem extensão mas com
    magic bytes de imagem.
    """
    scan_roots = list(PERSISTENT_AUDIT_PATHS)
    # Adicionar subpaths críticos explicitamente
    comfyui_persistent_dirs = [
        Path("/kaggle/working/ComfyUI/input"),
        Path("/kaggle/working/ComfyUI/output"),
        Path("/kaggle/working/ComfyUI/temp"),
    ]
    for d in comfyui_persistent_dirs:
        if d.exists() and d not in scan_roots:
            scan_roots.append(d)
    if extra_paths:
        scan_roots.extend(extra_paths)

    violations: List[Dict[str, Any]] = []
    prefix = f"[{label}] " if label else ""

    for root in scan_roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            # Seguir symlinks para detectar bypass
            try:
                real = item.resolve()
            except Exception:
                real = item

            if item.is_symlink():
                # Symlink apontando para fora de /kaggle/working também é violação
                target = str(real)
                if not target.startswith("/kaggle/working") and not target.startswith("/dev/shm"):
                    # Symlink para local inesperado — registrar mas não abortar automaticamente
                    violations.append({
                        "path": str(item),
                        "type": "symlink",
                        "target": target,
                        "size": 0,
                        "mtime": item.lstat().st_mtime if item.exists() else 0,
                    })

            if not item.is_file():
                continue

            stat = item.stat()
            ext = item.suffix.lower()

            # Verificação por extensão
            if ext in SENSITIVE_EXTENSIONS or ext in SENSITIVE_ARCHIVES:
                violations.append({
                    "path": str(item),
                    "type": "extension",
                    "ext": ext,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
                continue

            # Verificação por magic bytes (arquivos sem extensão ou extensão disfarçada)
            try:
                magic = item.read_bytes()[:16]
                is_image = any([
                    magic[:8] == b"\x89PNG\r\n\x1a\n",   # PNG
                    magic[:3] == b"\xff\xd8\xff",          # JPEG
                    magic[:4] == b"RIFF" and magic[8:12] == b"WEBP",  # WEBP
                    magic[:6] in (b"GIF87a", b"GIF89a"),   # GIF
                    magic[:4] in (b"PK\x03\x04", b"PK\x05\x06"),  # ZIP
                ])
                if is_image:
                    violations.append({
                        "path": str(item),
                        "type": "magic_bytes",
                        "magic": magic[:8].hex(),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
            except (OSError, PermissionError):
                pass

    if violations:
        lines = [f"{prefix}SECURITY VIOLATION: {len(violations)} artefato(s) sensível(is) em armazenamento persistente:"]
        for v in violations:
            ts = datetime.datetime.fromtimestamp(v["mtime"]).isoformat() if v.get("mtime") else "unknown"
            lines.append(f"  PATH : {v['path']}")
            lines.append(f"  TYPE : {v['type']}")
            lines.append(f"  SIZE : {v.get('size', '?')} bytes")
            lines.append(f"  MTIME: {ts}")
            if v.get("target"):
                lines.append(f"  TARGET: {v['target']}")
            lines.append("")
        _security_abort("\n".join(lines))


# ---------------------------------------------------------------------------
# Custom node integrity — snapshot + hash
# ---------------------------------------------------------------------------

def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_node_directory_hash(node_path: Path) -> str:
    """
    Computa SHA-256 determinístico do diretório de um custom node.
    Combina: lista ordenada de (relative_path, sha256) de todos os arquivos.
    Usado para verificação de integridade de diretório.
    """
    h = hashlib.sha256()
    files = []
    for f in sorted(node_path.rglob("*")):
        if f.is_file() and not f.is_symlink():
            try:
                files.append((str(f.relative_to(node_path)), _hash_file(f)))
            except (OSError, PermissionError):
                files.append((str(f.relative_to(node_path)), "UNREADABLE"))
    for rel, fhash in files:
        h.update(rel.encode("utf-8"))
        h.update(fhash.encode("utf-8"))
    return h.hexdigest()


def get_expected_node_hash(node_name: str) -> Optional[str]:
    """Retorna o hash esperado do diretório do node, se configurado."""
    return EXPECTED_CUSTOM_NODE_HASHES.get(node_name)


def verify_node_hashes(comfyui_dir: Path) -> List[str]:
    """
    Verifica hashes de diretório dos custom nodes contra EXPECTED_CUSTOM_NODE_HASHES.
    Retorna lista de mismatches. Se EXPECTED_CUSTOM_NODE_HASHES estiver vazio, retorna [].
    """
    custom_dir = Path(comfyui_dir) / "custom_nodes"
    mismatches: List[str] = []
    if not custom_dir.exists() or not EXPECTED_CUSTOM_NODE_HASHES:
        return mismatches

    for item in sorted(custom_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("__"):
            continue
        if item.name not in EXPECTED_CUSTOM_NODE_HASHES:
            continue
        expected = EXPECTED_CUSTOM_NODE_HASHES[item.name]
        try:
            actual = compute_node_directory_hash(item)
        except (OSError, PermissionError):
            mismatches.append(f"UNREADABLE: {item.name}")
            continue
        if actual != expected:
            mismatches.append(
                f"HASH_MISMATCH: {item.name} (esperado={expected[:16]}…, atual={actual[:16]}…)"
            )
    return mismatches


def snapshot_custom_nodes(comfyui_dir: Path) -> Dict[str, Any]:
    """
    Cria snapshot dos custom nodes instalados: nome, path, lista de arquivos com SHA-256.
    Usado para detectar alterações posteriores ao startup.
    """
    custom_dir = Path(comfyui_dir) / "custom_nodes"
    snapshot: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "nodes": {},
    }
    if not custom_dir.exists():
        return snapshot

    for item in sorted(custom_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("__"):
            continue
        files: Dict[str, str] = {}
        for f in sorted(item.rglob("*")):
            if f.is_file():
                try:
                    files[str(f.relative_to(item))] = _hash_file(f)
                except (OSError, PermissionError):
                    files[str(f.relative_to(item))] = "UNREADABLE"
        snapshot["nodes"][item.name] = {
            "path": str(item),
            "file_count": len(files),
            "files": files,
        }
    return snapshot


def verify_custom_nodes_unchanged(
    comfyui_dir: Path,
    startup_snapshot: Dict[str, Any],
    strict: bool = True,
) -> List[str]:
    """
    Compara snapshot atual dos custom nodes com o snapshot do startup.
    Retorna lista de alterações detectadas.
    Se strict=True e houver alterações, levanta SecurityError.
    """
    current = snapshot_custom_nodes(comfyui_dir)
    changes: List[str] = []

    startup_nodes = set(startup_snapshot.get("nodes", {}).keys())
    current_nodes = set(current.get("nodes", {}).keys())

    # Nodes adicionados após startup
    for added in current_nodes - startup_nodes:
        if added not in ALLOWED_CUSTOM_NODES:
            changes.append(f"ADDED_UNKNOWN: {added}")
        else:
            changes.append(f"ADDED_ALLOWED: {added} (adicionado após startup)")

    # Nodes removidos (menos crítico, mas registrar)
    for removed in startup_nodes - current_nodes:
        changes.append(f"REMOVED: {removed}")

    # Nodes modificados (conteúdo alterado)
    for node_name in startup_nodes & current_nodes:
        s_files = startup_snapshot["nodes"][node_name].get("files", {})
        c_files = current["nodes"][node_name].get("files", {})

        for fpath, s_hash in s_files.items():
            c_hash = c_files.get(fpath)
            if c_hash is None:
                changes.append(f"DELETED_FILE: {node_name}/{fpath}")
            elif c_hash != s_hash and s_hash != "UNREADABLE":
                changes.append(f"MODIFIED: {node_name}/{fpath} ({s_hash[:8]}→{c_hash[:8]})")

        for fpath in set(c_files) - set(s_files):
            changes.append(f"NEW_FILE: {node_name}/{fpath}")

    if changes and strict:
        _security_abort(
            f"Custom nodes alterados após startup ({len(changes)} alteração(ões)):\n"
            + "\n".join(f"  - {c}" for c in changes)
            + "\nIsso pode indicar instalação via Manager durante sessão. Abortando."
        )

    # Verificar hashes estáticos esperados (se configurados)
    hash_mismatches = verify_node_hashes(comfyui_dir)
    if hash_mismatches and strict:
        _security_abort(
            f"Custom nodes com hash inesperado ({len(hash_mismatches)} mismatch(es)):\n"
            + "\n".join(f"  - {c}" for c in hash_mismatches)
            + "\nPossível tampering. Abortando."
        )

    return changes


def check_custom_nodes_allowlist(comfyui_dir: Path, strict: bool = True) -> List[str]:
    """
    Verifica se há custom nodes fora da ALLOWED_CUSTOM_NODES allowlist.
    Se strict=True e houver nodes não autorizados, levanta SecurityError.
    """
    custom_dir = Path(comfyui_dir) / "custom_nodes"
    if not custom_dir.exists():
        return []

    unknown = []
    for item in custom_dir.iterdir():
        if not item.is_dir() or item.name.startswith("__"):
            continue
        # Verificar symlinks para fora da allowlist
        if item.is_symlink():
            target = str(item.resolve())
            unknown.append(f"{item.name} (symlink→{target})")
            continue
        if item.name not in ALLOWED_CUSTOM_NODES:
            unknown.append(item.name)

    if unknown:
        msg = (
            f"Custom nodes NÃO AUTORIZADOS detectados em {custom_dir}:\n"
            + "\n".join(f"  - {n}" for n in unknown)
            + f"\nAllowlist: {sorted(ALLOWED_CUSTOM_NODES)}"
        )
        if strict:
            _security_abort(msg)
        else:
            print(f"[WARN] {msg}")

    return unknown


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def provision_shm_dirs(*dirs: Path) -> None:
    """Cria diretórios em /dev/shm com permissões 0o777."""
    for d in dirs:
        assert_shm_path(d, f"diretório tmpfs '{d.name}'")
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o777)
    print(f"[INFO] tmpfs provisionado: {' | '.join(str(d) for d in dirs)}")


def safe_remove(path: Path) -> None:
    """Remove arquivo ou diretório. Verifica remoção. Idempotente."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"[SECURITY] safe_remove falhou: {path} ainda existe após remoção")


def clear_input(input_dir: Path = SHM_INPUT) -> None:
    """Apaga imagens e uploads do diretório de input. Verifica que fica vazio."""
    assert_shm_path(input_dir, "input_dir em clear_input")
    if input_dir.exists():
        for f in sorted(input_dir.rglob("*"), reverse=True):
            if f.is_file() or f.is_symlink():
                safe_remove(f)
        for sub in sorted(input_dir.iterdir(), reverse=True):
            if sub.is_dir():
                try:
                    sub.rmdir()
                except OSError:
                    pass
    remaining = [f for f in input_dir.rglob("*") if f.is_file()] if input_dir.exists() else []
    if remaining:
        raise SecurityError(
            f"clear_input: {len(remaining)} arquivo(s) não removido(s) em {input_dir}:\n"
            + "\n".join(f"  {f}" for f in remaining[:10])
        )
    print(f"[CLEANUP] ✓ {input_dir} limpo (zero arquivos)")


def clear_output(output_dir: Path = SHM_OUTPUT) -> None:
    """Apaga imagens do diretório de output. Deve ser chamado APÓS ZIP. Verifica vazio."""
    assert_shm_path(output_dir, "output_dir em clear_output")
    if output_dir.exists():
        for f in sorted(output_dir.rglob("*"), reverse=True):
            if f.is_file() or f.is_symlink():
                safe_remove(f)
    remaining = [f for f in output_dir.rglob("*") if f.is_file()] if output_dir.exists() else []
    if remaining:
        raise SecurityError(
            f"clear_output: {len(remaining)} arquivo(s) não removido(s) em {output_dir}"
        )
    print(f"[CLEANUP] ✓ {output_dir} limpo (zero arquivos)")


def secure_cleanup(
    comfyui_dir: Optional[Path] = None,
    extra_dirs: Optional[List[Path]] = None,
    raise_on_persistent: bool = True,
    known_pids: Optional[List[int]] = None,
    comfyui_pid: Optional[int] = None,
) -> None:
    """
    Limpeza completa — robusta, fail-closed, idempotent.

    1. Limpa todos os dirs em /dev/shm (tmpfs): input, output, temp, user, logs, archive.
       Verifica que cada dir é tmpfs antes de limpar (fail-closed).
    2. Remove ZIPs em /dev/shm/comfy_ui_archive (limpeza parcial).
    3. Apaga credenciais (gdrive_sa.json, rclone.conf).
    4. Mata processo conhecido (comfyui_pid ou known_pids) — NUNCA mata PIDs arbitrários.
    5. Faz GC.
    6. Verifica /kaggle/working via final_filesystem_check.
    7. Se raise_on_persistent=True e encontrar violações, levanta SecurityError.

    Deve ser chamada em try/finally — garante limpeza mesmo em exceção ou KeyboardInterrupt.
    """
    print("\n[CLEANUP] Iniciando secure_cleanup...")

    shm_dirs = [SHM_INPUT, SHM_OUTPUT, SHM_TEMP, SHM_USER, SHM_LOGS, SHM_ARCHIVE]
    if extra_dirs:
        for d in extra_dirs:
            if d not in shm_dirs:
                shm_dirs.append(d)

    # 1-2. Limpar dirs tmpfs + ZIPs
    for d in shm_dirs:
        if not d.exists():
            continue
        try:
            assert_shm_path(d, f"shm_dir em secure_cleanup: {d.name}")
        except SecurityError:
            print(f"[CLEANUP] WARN: {d} não passou em assert_shm_path, tentando remover arquivos sensíveis")

        # Remover ZIPs explicitamente (limpeza parcial de arquivos)
        for zip_file in sorted(d.rglob("*")):
            if zip_file.is_file() and zip_file.suffix.lower() in SENSITIVE_ARCHIVES:
                try:
                    safe_remove(zip_file)
                    print(f"[CLEANUP] ✓ ZIP removido: {zip_file}")
                except Exception as e:
                    print(f"[CLEANUP] WARN: falha ao remover ZIP {zip_file}: {e}")

        # Remover imagens sensíveis explicitamente
        try:
            img_files = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in SENSITIVE_EXTENSIONS]
            if img_files:
                print(f"[CLEANUP] {d}: removendo {len(img_files)} imagem(ns)")
        except Exception:
            pass

        try:
            safe_remove(d)
        except Exception as e:
            print(f"[CLEANUP] WARN: falha ao remover {d}: {e}")
            try:
                if d.is_dir() and not d.is_symlink():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

        # Recriar dir limpo em /dev/shm
        try:
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o777)
        except Exception:
            pass
        print(f"[CLEANUP] ✓ {d} limpo")

    # 3. Limpar credenciais
    try:
        cleanup_gdrive_credentials()
    except Exception as e:
        print(f"[CLEANUP] WARN: falha ao limpar credenciais: {e}")

    # 4. Matar processo conhecido apenas (nunca PIDs arbitrários)
    pids_to_kill = []
    if comfyui_pid is not None:
        pids_to_kill.append(comfyui_pid)
    if known_pids:
        pids_to_kill.extend(known_pids)

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[CLEANUP] SIGTERM enviado ao PID conhecido={pid}")
            time.sleep(1)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
                print(f"[CLEANUP] SIGKILL enviado ao PID conhecido={pid}")
                time.sleep(0.5)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            print(f"[INFO] PID={pid} já não existe")
        except PermissionError:
            print(f"[WARN] Sem permissão para matar PID={pid}")
        except Exception as e:
            print(f"[CLEANUP] WARN: erro ao matar PID={pid}: {e}")

    # 5. GC
    gc.collect()
    print("[CLEANUP] GC coletado")

    # 6-7. Verificação final (fail-closed)
    try:
        result = final_filesystem_check(silent=True)
    except Exception as e:
        print(f"[CLEANUP] WARN: final_filesystem_check falhou: {e}")
        result = {"violations": 1, "report": str(e), "images": [], "archives": [], "symlinks": []}

    if result["violations"] > 0:
        msg = (
            f"secure_cleanup: {result['violations']} artefato(s) sensível(is) "
            f"encontrado(s) em /kaggle/working:\n{result['report']}"
        )
        if raise_on_persistent:
            _security_abort(msg)
        else:
            print(f"[CLEANUP] ⚠️  {msg}")
    else:
        print("[CLEANUP] ✓ /kaggle/working: zero artefatos sensíveis")

    print("[CLEANUP] secure_cleanup concluído\n")


def final_filesystem_check(
    scan_root: Path = Path("/kaggle/working"),
    silent: bool = False,
    check_comfyui_subdirs: bool = True,
) -> Dict[str, Any]:
    """
    Verifica recursivamente que scan_root não contém imagens/archives sensíveis.
    Verifica também: symlinks, arquivos ocultos, arquivos sem extensão com magic bytes.
    Verifica explicitamente ComfyUI/input, ComfyUI/output, ComfyUI/temp mesmo se vazios.

    Retorna dict com 'violations' (int), 'report' (str).
    """
    img_found = []
    arch_found = []
    symlink_found = []

    extra_report_lines = []

    # Subpaths críticos — verificar existência e conteúdo mesmo se vazios
    if check_comfyui_subdirs:
        comfyui_subdirs = [
            scan_root / "ComfyUI" / "input",
            scan_root / "ComfyUI" / "output",
            scan_root / "ComfyUI" / "temp",
        ]
        for d in comfyui_subdirs:
            if d.exists():
                contents = list(d.rglob("*"))
                files = [f for f in contents if f.is_file()]
                extra_report_lines.append(
                    f"  {d}: {'VAZIO' if not files else f'{len(files)} arquivo(s) — VERIFICAR'}"
                )

    if scan_root.exists():
        for item in scan_root.rglob("*"):
            # Symlinks
            if item.is_symlink():
                try:
                    target = str(item.resolve())
                except Exception:
                    target = "unresolvable"
                symlink_found.append({
                    "path": str(item),
                    "target": target,
                    "size": 0,
                    "mtime": item.lstat().st_mtime,
                })
                continue

            if not item.is_file():
                continue

            stat = item.stat()
            ext = item.suffix.lower()

            if ext in SENSITIVE_EXTENSIONS:
                img_found.append({"path": str(item), "size": stat.st_size, "mtime": stat.st_mtime, "type": "extension"})
                continue
            if ext in SENSITIVE_ARCHIVES:
                arch_found.append({"path": str(item), "size": stat.st_size, "mtime": stat.st_mtime, "type": "extension"})
                continue

            # Magic bytes para arquivos sem extensão ou extensão suspeita
            try:
                magic = item.read_bytes()[:16]
                is_img = any([
                    magic[:8] == b"\x89PNG\r\n\x1a\n",
                    magic[:3] == b"\xff\xd8\xff",
                    magic[:4] == b"RIFF" and magic[8:12] == b"WEBP",
                    magic[:6] in (b"GIF87a", b"GIF89a"),
                ])
                is_zip = magic[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
                if is_img:
                    img_found.append({"path": str(item), "size": stat.st_size, "mtime": stat.st_mtime, "type": "magic_bytes"})
                elif is_zip:
                    arch_found.append({"path": str(item), "size": stat.st_size, "mtime": stat.st_mtime, "type": "magic_bytes"})
            except (OSError, PermissionError):
                pass

    violations = len(img_found) + len(arch_found)

    lines = [
        "",
        "=== FINAL SECURITY FILESYSTEM CHECK ===",
        f"Scan root: {scan_root}",
        f"Image files    : {len(img_found)}",
        f"Archive files  : {len(arch_found)}",
        f"Symlinks found : {len(symlink_found)}",
        f"Violations     : {violations}",
        "",
    ]

    if extra_report_lines:
        lines.append("ComfyUI subdir status:")
        lines.extend(extra_report_lines)
        lines.append("")

    if violations == 0 and not symlink_found:
        lines.append("STATUS: PASS ✅")
    elif violations > 0:
        lines.append("STATUS: FAIL ❌")
        lines.append("SECURITY VIOLATION DETECTED")
        lines.append("")
        for item in img_found:
            ts = datetime.datetime.fromtimestamp(item["mtime"]).isoformat()
            lines.append(f"  IMAGE   {item['path']}  ({item['size']} bytes, {ts}, type={item['type']})")
        for item in arch_found:
            ts = datetime.datetime.fromtimestamp(item["mtime"]).isoformat()
            lines.append(f"  ARCHIVE {item['path']}  ({item['size']} bytes, {ts}, type={item['type']})")
    else:
        lines.append("STATUS: PASS ✅ (com symlinks — verificar manualmente)")

    if symlink_found:
        lines.append("")
        lines.append("Symlinks detectados (verificar manualmente):")
        for s in symlink_found:
            lines.append(f"  SYMLINK {s['path']} → {s['target']}")

    lines.append("")
    report = "\n".join(lines)

    if not silent:
        print(report)

    return {
        "violations": violations,
        "images": img_found,
        "archives": arch_found,
        "symlinks": symlink_found,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Credential cleanup
# ---------------------------------------------------------------------------

def cleanup_gdrive_credentials() -> None:
    """Apaga service account JSON e rclone config. Idempotente."""
    for p in (Path("/root/gdrive_sa.json"), Path("/root/.config/rclone/rclone.conf")):
        if p.exists():
            safe_remove(p)
            print(f"[SECURITY] Credencial removida: {p}")
        else:
            print(f"[INFO] Credencial já não existe: {p}")


# ---------------------------------------------------------------------------
# Process verification
# ---------------------------------------------------------------------------

def _read_proc_cmdline(pid: int) -> List[str]:
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
    expected_user: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Verifica via /proc/<pid>/cmdline que o processo usa exatamente os paths esperados."""
    cmdline = _read_proc_cmdline(pid)
    if not cmdline:
        return False, f"Não foi possível ler /proc/{pid}/cmdline"

    checks = [
        ("--input-directory",  str(expected_input),  "input-directory"),
        ("--output-directory", str(expected_output), "output-directory"),
        ("--temp-directory",   str(expected_temp),   "temp-directory"),
        ("--listen",           expected_host,        "host"),
        ("--port",             str(expected_port),   "port"),
    ]
    if expected_user:
        checks.append(("--user-directory", str(expected_user), "user-directory"))

    issues = []
    for flag, expected_val, label in checks:
        try:
            idx = cmdline.index(flag)
            actual_val = cmdline[idx + 1] if idx + 1 < len(cmdline) else ""
            if actual_val != expected_val:
                issues.append(f"{label}: esperado '{expected_val}', encontrado '{actual_val}'")
        except ValueError:
            issues.append(f"flag '{flag}' ausente no cmdline do PID={pid}")

    for arg in cmdline:
        if "/kaggle/working" in arg:
            # Verificar se é um path esperado (comfyui dir ou models dir)
            allowed_persistent = {
                str(DEFAULT_COMFYUI_DIR),
                str(DEFAULT_COMFYUI_DIR / "models"),
                str(DEFAULT_COMFYUI_DIR / "comfyui.log"),
                str(DEFAULT_COMFYUI_DIR / "extra_model_paths.yaml"),
            }
            if arg not in allowed_persistent:
                issues.append(f"Argumento suspeito em /kaggle/working: {arg}")

    if issues:
        return False, (
            f"PID={pid} tem configuração incorreta:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    return True, f"PID={pid} verificado: paths corretos em /dev/shm"


def find_existing_comfyui_pid(port: int = DEFAULT_PORT) -> Optional[int]:
    """Tenta encontrar o PID do processo ComfyUI na porta especificada."""
    for cmd in [["fuser", f"{port}/tcp"], ["ss", "-tlnp", f"sport = :{port}"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            import re
            # fuser output: "  1234"
            for token in result.stdout.split():
                if token.strip().isdigit():
                    return int(token.strip())
            # ss output: pid=1234
            m = re.search(r"pid=(\d+)", result.stdout)
            if m:
                return int(m.group(1))
        except Exception:
            continue
    return None


def kill_mismatched_process(pid: int) -> None:
    """Mata processo com configuração incorreta com SIGTERM → SIGKILL."""
    print(f"[SECURITY] Matando processo com configuração incorreta: PID={pid}")
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)  # verifica se ainda existe
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            pass
        print(f"[SECURITY] PID={pid} encerrado")
    except ProcessLookupError:
        print(f"[INFO] PID={pid} já não existe")
    except PermissionError:
        print(f"[WARN] Sem permissão para matar PID={pid} — reinicie o kernel")


# ---------------------------------------------------------------------------
# GPU / device
# ---------------------------------------------------------------------------

def get_cuda_device(default: int = DEFAULT_CUDA_DEVICE) -> int:
    raw = os.environ.get(ENV_CUDA_DEVICE, str(default))
    try:
        device = int(str(raw).strip())
        if device < 0:
            raise ValueError
        return device
    except (TypeError, ValueError):
        print(f"[WARN] {ENV_CUDA_DEVICE}={raw!r} inválido; usando {default}")
        return default


def detect_gpu() -> Dict[str, Any]:
    try:
        from gpu_detect import detect_gpu as _detect
        return _detect()
    except ImportError:
        info: Dict[str, Any] = {"has_gpu": False, "gpu_count": 0, "gpus": [], "gpu_name": "Unknown", "vram_gb": 0}
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({"index": i, "name": props.name,
                                 "vram_gb": props.total_memory / (1024 ** 3),
                                 "cuda": torch.version.cuda, "driver": None})
                info.update({"has_gpu": True, "gpu_count": len(gpus), "gpus": gpus,
                              "gpu_name": gpus[0]["name"], "vram_gb": gpus[0]["vram_gb"]})
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
# Custom node install helpers
# ---------------------------------------------------------------------------

def parse_custom_node_spec(spec: str) -> Tuple[str, str, str]:
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


def filter_custom_nodes(custom_nodes: Optional[List[str]]) -> List[str]:
    if not custom_nodes:
        return []
    filtered = []
    for spec in custom_nodes:
        if is_manager_custom_node_spec(spec):
            print(f"[INFO] Ignorando '{spec}': Manager integrado via --enable-manager.")
            continue
        filtered.append(spec)
    return filtered


def install_or_update_custom_node(custom_dir: Path, spec: str) -> Path:
    custom_dir = Path(custom_dir)
    custom_dir.mkdir(parents=True, exist_ok=True)
    repo, branch, node_name = parse_custom_node_spec(spec)
    node_path = custom_dir / node_name

    if node_path.exists():
        if not (node_path / ".git").exists():
            raise RuntimeError(f"Custom node existente não é um checkout Git: {node_path}.")
        print(f"[INFO] Atualizando custom node: {node_name}")
        _run(["git", "pull", "--ff-only"], cwd=node_path, timeout=300, check=False)
    else:
        print(f"[INFO] Clonando custom node: {node_name} (branch={branch})")
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo, str(node_path)], timeout=900)

    node_req = node_path / "requirements.txt"
    if node_req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req)], timeout=900, check=False)
    return node_path


def install_manager_requirements(comfyui_dir: Path) -> bool:
    comfyui_dir = Path(comfyui_dir)
    mgr_req = comfyui_dir / "manager_requirements.txt"
    if not mgr_req.exists():
        print(f"[WARN] {mgr_req} não encontrado")
        return False
    print(f"[INFO] Instalando Manager integrado: {mgr_req}")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(mgr_req)], timeout=900)
    return True


# ---------------------------------------------------------------------------
# extra_model_paths.yaml
# ---------------------------------------------------------------------------

def build_extra_model_paths_yaml(
    models_dir: Path,
    additional_roots: Optional[List[Tuple[str, Path]]] = None,
) -> str:
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
    user_dir=None,
    drive_base=DEFAULT_DRIVE_BASE,
    enable_manager: bool = True,
    additional_model_roots: Optional[List[Tuple[str, Path]]] = None,
    strict_allowlist: bool = True,
    secure_mode: Optional[bool] = None,
) -> Path:
    """
    Instala/atualiza ComfyUI e configura custom nodes.
    
    SECURE_MODE (Hardened Architecture):
      - Manager e ngrok são permitidos.
      - I/O redirecionado para /dev/shm.
    """
    effective_secure = secure_mode if secure_mode is not None else get_secure_mode()

    comfyui_dir = Path(comfyui_dir)
    models_dir = Path(models_dir) if models_dir else comfyui_dir / "models"

    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP
    user_dir = Path(user_dir) if user_dir else SHM_USER

    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")
    assert_shm_path(user_dir, "--user-directory")

    gpu_info = detect_gpu()
    cuda_device = get_cuda_device()

    is_git_repo = (comfyui_dir / ".git").exists()
    has_main = (comfyui_dir / "main.py").exists()

    if not comfyui_dir.exists():
        _run(["git", "clone", "--depth", "1", repo_url, str(comfyui_dir)], timeout=900)
    elif is_git_repo:
        _run(["git", "pull", "--ff-only"], cwd=comfyui_dir, timeout=300, check=False)
    elif has_main:
        print(f"[WARN] {comfyui_dir} tem main.py sem .git — pulando clone/pull")
    else:
        raise RuntimeError(f"Diretório {comfyui_dir} não é um checkout Git válido do ComfyUI.")

    provision_shm_dirs(input_dir, output_dir, temp_dir, user_dir, SHM_LOGS, SHM_ARCHIVE)

    req = comfyui_dir / "requirements.txt"
    if req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=900)

    if enable_manager:
        install_manager_requirements(comfyui_dir)

    for cat in MODEL_CATEGORIES:
        (models_dir / cat).mkdir(parents=True, exist_ok=True)

    # Custom nodes imutáveis no checkout Git
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

    check_custom_nodes_allowlist(comfyui_dir, strict=strict_allowlist)

    extra_paths = comfyui_dir / "extra_model_paths.yaml"
    yaml_content = build_extra_model_paths_yaml(models_dir, additional_model_roots)
    extra_paths.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] (Re)escrito {extra_paths}")

    print(f"[INFO] ComfyUI: {comfyui_dir} | Manager: {'sim' if enable_manager else 'não'}")
    print(f"[SECURITY] INPUT  → {input_dir}")
    print(f"[SECURITY] OUTPUT → {output_dir}")
    print(f"[SECURITY] TEMP   → {temp_dir}")
    print(f"[SECURITY] USER   → {user_dir}")
    print(f"[SECURITY] SECURE_MODE={effective_secure}")
    return comfyui_dir


# ---------------------------------------------------------------------------
# build_comfyui_command
# ---------------------------------------------------------------------------

def build_comfyui_command(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    user_dir: Optional[Path] = None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    extra_args: Optional[List[str]] = None,
    secure_mode: Optional[bool] = None,
) -> List[List[str]]:
    """
    Monta o comando de start do ComfyUI.
    Retorna uma lista de comandos (pode incluir wrapper de isolamento).
    """
    effective_secure = secure_mode if secure_mode is not None else get_secure_mode()

    comfyui_dir = Path(comfyui_dir)
    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP
    user_dir = Path(user_dir) if user_dir else SHM_USER

    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")
    assert_shm_path(user_dir, "--user-directory")

    if cuda_device is None:
        cuda_device = get_cuda_device()

    cmd = [
        sys.executable, "main.py",
        "--listen", host,
        "--port", str(port),
        "--input-directory", str(input_dir),
        "--output-directory", str(output_dir),
        "--temp-directory", str(temp_dir),
        "--user-directory", str(user_dir),
        "--cuda-device", str(cuda_device),
    ]
    if enable_manager:
        cmd.append("--enable-manager")

    if extra_args:
        blocked_flags = {"--input-directory", "--output-directory", "--temp-directory", "--user-directory"}
        for arg in extra_args:
            if arg in DISCOURAGED_VRAM_FLAGS:
                print(f"[WARN] Flag de VRAM desencorajada: {arg}")
            if arg in blocked_flags:
                _security_abort(
                    f"Tentativa de sobrescrever {arg} via extra_args. "
                    "Use os parâmetros nomeados correspondentes."
                )
        cmd.extend(extra_args)

    # -----------------------------------------------------------------------
    # Isolamento de Filesystem (Kaggle / Linux)
    # -----------------------------------------------------------------------
    if effective_secure and os.name != "nt":
        # Tentar isolamento real se disponível
        isolation_cmd = _get_isolation_wrapper(comfyui_dir, [input_dir, output_dir, temp_dir, user_dir, SHM_LOGS])
        if isolation_cmd:
            return [isolation_cmd + cmd]

    return [cmd]


def _get_isolation_wrapper(comfyui_dir: Path, mutable_paths: List[Path]) -> Optional[List[str]]:
    """
    Retorna prefixo de comando para isolamento (bubblewrap/unshare) se disponível.
    """
    # 1. Bubblewrap (melhor isolamento)
    if shutil.which("bwrap"):
        print("[SECURITY] Usando bubblewrap para isolamento de filesystem")
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--tmpfs", "/run",
            # Tornar /kaggle/working somente leitura para o processo
            "--ro-bind", "/kaggle/working", "/kaggle/working",
        ]
        # Adicionar binds de escrita para caminhos em /dev/shm
        for p in mutable_paths:
            p.mkdir(parents=True, exist_ok=True)
            bwrap_cmd.extend(["--bind", str(p), str(p)])
        return bwrap_cmd

    # 2. Unshare (mount namespace)
    if shutil.which("unshare"):
        print("[SECURITY] Usando unshare para isolamento de filesystem")
        # Nota: unshare requer privilégios ou user namespaces habilitados
        return ["unshare", "--mount", "--map-root-user"]

    print("[SECURITY] Isolamento real (bwrap/unshare) não disponível. Usando apenas restrição de paths.")
    return None


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
    user_dir: Optional[Path] = None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    secure_mode: Optional[bool] = None,
):
    main_py = Path(comfyui_dir) / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"ComfyUI não encontrado em {comfyui_dir}")

    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP
    provision_shm_dirs(input_dir, output_dir, temp_dir, SHM_ARCHIVE, SHM_LOGS, SHM_USER)

    cmd = build_comfyui_command(
        comfyui_dir=comfyui_dir, host=host, port=port,
        output_dir=output_dir, input_dir=input_dir, temp_dir=temp_dir,
        cuda_device=cuda_device, enable_manager=enable_manager,
        extra_args=extra_args, secure_mode=secure_mode,
    )

    effective_secure = secure_mode if secure_mode is not None else get_secure_mode()
    if effective_secure:
        log_path = SHM_LOGS / "comfyui.log"
    else:
        log_path = Path(comfyui_dir) / "comfyui.log"
    log_fh = open(log_path, "a", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=comfyui_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    print(f"[INFO] ComfyUI PID={proc.pid} | {host}:{port}")
    print(f"[SECURITY] INPUT  → {input_dir} | OUTPUT → {output_dir} | TEMP → {temp_dir}")
    print(f"[SECURITY] LOG    → {log_path}")
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
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print("[INFO] Health check OK")
                    return True
        except Exception:
            time.sleep(2)
    print(f"[ERROR] Health check falhou após {timeout}s ({url})")
    return False


# ---------------------------------------------------------------------------
# start_comfyui_runtime
# ---------------------------------------------------------------------------

def start_comfyui_runtime(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    user_dir: Optional[Path] = None,
    extra_args=None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = False,
    enable_ngrok: bool = False,
    health_timeout: int = 90,
    health_host: str = "127.0.0.1",
    reuse_existing: bool = False,
    secure_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Ordem obrigatória: start → health → ngrok.
    SECURE_MODE:
      - Manager PERMITIDO (com isolamento de filesystem — state/downloads em /dev/shm)
      - ngrok PERMITIDO (após health check, token via Kaggle Secrets)
      - reuse_existing forçado False
      - todos os paths mutáveis em /dev/shm
    """
    effective_secure = secure_mode if secure_mode is not None else get_secure_mode()

    # SECURE_MODE: apenas reuse_existing é forçado False.
    # Manager e ngrok são permitidos — a segurança vem do isolamento de filesystem,
    # não do bloqueio de funcionalidade.
    if effective_secure:
        if reuse_existing:
            print("[SECURITY] SECURE_MODE=True: reuse_existing forçado → False")
            reuse_existing = False

    comfyui_dir = Path(comfyui_dir)
    input_dir = Path(input_dir) if input_dir else SHM_INPUT
    output_dir = Path(output_dir) if output_dir else SHM_OUTPUT
    temp_dir = Path(temp_dir) if temp_dir else SHM_TEMP
    user_dir = Path(user_dir) if user_dir else SHM_USER
    if effective_secure:
        log_path = SHM_LOGS / "comfyui.log"
    else:
        log_path = comfyui_dir / "comfyui.log"

    assert_shm_path(input_dir, "--input-directory")
    assert_shm_path(output_dir, "--output-directory")
    assert_shm_path(temp_dir, "--temp-directory")
    assert_shm_path(user_dir, "--user-directory")

    result: Dict[str, Any] = {
        "ok": False, "proc": None, "public_url": None,
        "local_url": f"http://{health_host}:{port}",
        "log_path": str(log_path), "health": False,
        "ngrok_started": False, "reused_existing": False, "pid": None,
        "secure_mode": effective_secure,
    }

    existing_pid = find_existing_comfyui_pid(port)
    port_in_use = health_check(health_host, port, timeout=2)

    if port_in_use and existing_pid:
        if reuse_existing:
            ok, reason = _verify_process_paths(
                pid=existing_pid,
                expected_input=input_dir, expected_output=output_dir,
                expected_temp=temp_dir, expected_host=host, expected_port=port,
                expected_user=user_dir,
            )
            if ok:
                print(f"[INFO] {reason} — reutilizando PID={existing_pid}")
                result.update({"health": True, "ok": True, "reused_existing": True, "pid": existing_pid})
            else:
                print(f"[SECURITY] {reason}")
                kill_mismatched_process(existing_pid)
                port_in_use = False
        else:
            print(f"[INFO] reuse_existing=False: matando PID={existing_pid}")
            kill_mismatched_process(existing_pid)
            port_in_use = False
    elif port_in_use and not existing_pid:
        _security_abort(
            f"Porta {port} em uso mas PID não identificado. Reinicie o kernel."
        )

    if not port_in_use:
        proc = start_comfyui(
            comfyui_dir=comfyui_dir, host=host, port=port,
            extra_args=extra_args, output_dir=output_dir,
            input_dir=input_dir, temp_dir=temp_dir,
            user_dir=user_dir,
            cuda_device=cuda_device, enable_manager=enable_manager,
            secure_mode=effective_secure,
        )
        result["proc"] = proc
        result["pid"] = proc.pid

        healthy = health_check(health_host, port, timeout=health_timeout)
        result["health"] = healthy
        if not healthy:
            print(f"[ERROR] ComfyUI falhou. Log: {log_path}")
            print(tail_log_file(log_path, n=50))
            return result
        result["ok"] = True

    if enable_ngrok:
        try:
            from ngrok_tunnel import start_ngrok_tunnel
            public_url = start_ngrok_tunnel(port=port)
            result.update({"public_url": public_url, "ngrok_started": True})
        except Exception as exc:
            msg = str(exc)
            try:
                from ngrok_tunnel import redact_secrets
                msg = redact_secrets(msg)
            except Exception:
                pass
            print(f"[WARN] ngrok não iniciado: {msg}")
    else:
        print("[INFO] ngrok desabilitado. Acesso local: 127.0.0.1")

    print("=" * 60)
    print(f"COMFYUI READY | SECURE_MODE={effective_secure}")
    print(f"Local  : {result['local_url']}")
    print(f"Public : {result['public_url'] or '(ngrok OFF)'}")
    print(f"INPUT  : {input_dir} | OUTPUT : {output_dir} | TEMP : {temp_dir}")
    print(f"PID    : {result['pid']}")
    print("=" * 60)
    return result


# ---------------------------------------------------------------------------
# ZIP seguro — AES-256, apenas em /dev/shm
# ---------------------------------------------------------------------------

def verify_zip_encryption(zip_path: Path, password: str) -> bool:
    """
    Teste runtime de criptografia real:
    1. Cria ZIP de teste em /dev/shm com arquivo conhecido
    2. Tenta abrir sem senha — deve falhar
    3. Abre com senha correta — deve ter sucesso
    4. Verifica conteúdo
    5. Remove ZIP de teste
    Retorna True se criptografia está funcionando.
    """
    import tempfile
    test_dir = zip_path.parent / f".{zip_path.stem}_enc_test"
    test_zip = zip_path
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(test_dir, 0o700)
        test_content = b"ENCRYPTION_TEST_SENTINEL_12345"
        (test_dir / "test.bin").write_bytes(test_content)

        try:
            import pyzipper
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyzipper"],
                           check=True, timeout=300)
            import pyzipper

        with pyzipper.AESZipFile(test_zip, "w", compression=pyzipper.ZIP_DEFLATED,
                                 encryption=pyzipper.WZ_AES) as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.write(test_dir / "test.bin", arcname="test.bin")

        # Tentar abrir sem senha — deve falhar
        open_without_password_failed = False
        try:
            with pyzipper.AESZipFile(test_zip, "r") as zf:
                zf.read("test.bin")
        except (RuntimeError, Exception):
            open_without_password_failed = True

        if not open_without_password_failed:
            raise SecurityError("ZIP encryption test FAILED: arquivo aberto sem senha!")

        # Abrir com senha — deve funcionar
        with pyzipper.AESZipFile(test_zip, "r") as zf:
            zf.setpassword(password.encode("utf-8"))
            content = zf.read("test.bin")

        if content != test_content:
            raise SecurityError("ZIP encryption test FAILED: conteúdo divergente após decrypt!")

        print("[SECURITY] ✅ ZIP AES-256 encryption test: PASS (sem senha → falhou; com senha → OK)")
        return True

    finally:
        for p in (test_zip, test_dir):
            if p.exists():
                try:
                    safe_remove(p)
                except Exception:
                    pass


def create_secure_zip(
    src_dir: Path,
    zip_password: Optional[str] = None,
    archive_dir: Path = SHM_ARCHIVE,
    zip_name: str = "output.zip",
    run_encryption_test: bool = True,
) -> Path:
    """
    Cria ZIP AES-256 em /dev/shm/comfy_ui_archive/ — NUNCA em /kaggle/working.
    Senha obrigatória via SECRET_ZIP_PASSWORD (Kaggle Secret ou env).
    Aborta sem fallback se senha ausente.
    Opcionalmente roda verify_zip_encryption() antes de criar o ZIP real.
    """
    assert_shm_path(archive_dir, "archive_dir do ZIP")

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
            "SECRET_ZIP_PASSWORD não encontrado em env ou Kaggle Secrets.\n"
            "Não será criado ZIP sem senha."
        )

    # NUNCA passar a senha para logs
    print(f"[SECURITY] ZIP password: presente ({len(zip_password)} chars) — NÃO logada")

    try:
        import pyzipper
    except ImportError:
        print("[INFO] Instalando pyzipper...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyzipper"],
                       check=True, timeout=300)
        import pyzipper

    # Teste runtime de criptografia antes de criar ZIP real
    if run_encryption_test:
        assert_shm_path(archive_dir, "archive_dir para encryption test")
        archive_dir.mkdir(parents=True, exist_ok=True)
        test_zip_path = archive_dir / f".{zip_name}_enc_test"
        verify_zip_encryption(test_zip_path, zip_password)

    archive_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(archive_dir, 0o700)
    zip_path = archive_dir / zip_name

    src_dir = Path(src_dir)
    assert_shm_path(src_dir, "src_dir do ZIP")

    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    if not files:
        print("[WARN] Nenhum arquivo em src_dir para compactar.")
        return zip_path

    with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED,
                              encryption=pyzipper.WZ_AES) as zf:
        zf.setpassword(zip_password.encode("utf-8"))
        for f in files:
            zf.write(f, arcname=str(f.relative_to(src_dir)))

    size_mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"[SECURITY] ZIP AES-256: {zip_path} ({size_mb:.1f} MB, {len(files)} arquivo(s))")
    print(f"[SECURITY] ZIP em tmpfs APENAS — NUNCA em /kaggle/working")
    return zip_path


def cleanup_zip(zip_path: Path) -> None:
    """Remove ZIP e verifica remoção. Idempotente."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        safe_remove(zip_path)
        print(f"[CLEANUP] ✓ ZIP removido: {zip_path}")
    else:
        print(f"[CLEANUP] ZIP já não existe: {zip_path}")


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
    parser.add_argument("--ngrok", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--no-manager", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--no-secure-mode", action="store_true",
                        help="Desabilita SECURE_MODE (apenas para desenvolvimento/testes)")
    args = parser.parse_args()

    if args.cuda_device is not None:
        os.environ[ENV_CUDA_DEVICE] = str(args.cuda_device)
    if args.no_secure_mode:
        set_secure_mode(False)

    output_dir = Path(args.output_dir) if args.output_dir else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    nodes = args.custom_nodes if args.custom_nodes is not None else list(DEFAULT_CUSTOM_NODES)

    comfyui = setup_comfyui(
        Path(args.comfyui_dir), args.repo_url, nodes,
        Path(args.models_dir) if args.models_dir else None,
        output_dir=output_dir, input_dir=input_dir, temp_dir=temp_dir,
        drive_base=args.drive_base, enable_manager=not args.no_manager,
    )
    if args.start:
        runtime = start_comfyui_runtime(
            comfyui_dir=comfyui, host=args.host, port=args.port,
            output_dir=output_dir, input_dir=input_dir, temp_dir=temp_dir,
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


if __name__ == "__main__":
    main()
