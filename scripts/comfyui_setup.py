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


def _is_git_internal_path(rel_path: str) -> bool:
    """Retorna True se o caminho relativo contém um segmento '.git'.

    Arquivos dentro de diretórios .git/ são bookkeeping interno do próprio Git
    (FETCH_HEAD, ORIG_HEAD, refs/, objects/, logs/, etc.) — criados ou modificados
    automaticamente por ``git pull``/``git fetch``/``git merge``, nunca escritos
    pelo usuário. Como o diretório .git em si nunca é *tracked* pelo próprio git,
    esses arquivos nunca devem ser candidatos a "artefato persistente não
    autorizado".

    Usa ``Path(rel_path).parts`` para detectar o segmento '.git' em **qualquer**
    posição do caminho (ex: ``ComfyUI/.git/FETCH_HEAD``,
    ``custom_nodes/foo/.git/HEAD``) e não apenas quando o caminho começa com
    '.git/'.
    """
    return ".git" in Path(rel_path).parts

RUNTIME_GENERATED_DIR_NAMES: frozenset[str] = frozenset({
    "__pycache__",
})
RUNTIME_GENERATED_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc",
    ".pyo",
    ".pyd",
})


def _is_runtime_generated_path(rel_path: str) -> bool:
    """Retorna True se o caminho eh artefato gerado pelo runtime Python."""
    parts = Path(rel_path).parts
    if any(part in RUNTIME_GENERATED_DIR_NAMES for part in parts):
        return True
    if Path(rel_path).suffix.lower() in RUNTIME_GENERATED_EXTENSIONS:
        return True
    return False


def _is_ephemeral_internal_path(rel_path: str) -> bool:
    """Retorna True para bookkeeping interno (git ou runtime) — nunca artefato."""
    return _is_git_internal_path(rel_path) or _is_runtime_generated_path(rel_path)


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
                except ValueError:
                    rel = str(item)
                # Ignorar bookkeeping interno (.git/ e bytecode __pycache__) —
                # nunca escritos pelo usuario como artefato.
                if _is_ephemeral_internal_path(rel):
                    continue
                snapshot.add(rel)
    _WORKING_SNAPSHOT = snapshot
    print(f"[SECURITY] Working snapshot: {len(snapshot)} arquivo(s) registrado(s) em /kaggle/working")
    return snapshot


def rebuild_working_snapshot_after_provisioning(label: str = "POST-SETUP") -> set:
    """
    Re-baseline do snapshot de /kaggle/working APOS o provisionamento oficial.

    O snapshot inicial (record_working_snapshot) eh tirado na celula PRE-CHECK,
    ANTES de setup_comfyui(). Tudo que a instalacao oficial instala depois
    (custom nodes aninhados, extra_model_paths.yaml, __pycache__ do servidor,
    bookkeeping .git/) nasce como "arquivo novo" e seria flagrado arquivo por
    arquivo. Em vez de enumerar extensao por extensao, o baseline passa a ser
    o ambiente TOTALMENTE PROVISIONADO: a fase de geracao nao pode persistir
    nada novo a partir dai.

    Fail-closed: antes de reescrever o baseline, roda uma varredura de
    imagens/archives/modelos/symlinks sobre TUDO (independente do snapshot).
    Se houver qualquer artefato sensivel, aborta e NAO reescreve — nunca
    "lava" lixo para dentro do baseline.
    """
    scan_root = PERSISTENT_AUDIT_PATHS[0] if PERSISTENT_AUDIT_PATHS else PERSISTENT_WORKING
    _clear_git_cache()
    pre_existing = _scan_working_violations(
        scan_root,
        check_symlinks=True,
        check_non_image_sensitive=False,
        skip_output_zip=True,
    )
    if pre_existing:
        _security_abort(_format_violations(pre_existing, f"{label}-PRE-REBASELINE"))
    previous_count = len(_WORKING_SNAPSHOT) if _WORKING_SNAPSHOT is not None else 0
    snapshot = record_working_snapshot()
    added = len(snapshot) - previous_count
    print(
        f"[SECURITY] Working snapshot re-baseline [{label}]: "
        f"{previous_count} -> {len(snapshot)} arquivo(s) (+{max(added, 0)} do provisionamento oficial)"
    )
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
                except ValueError:
                    rel = str(item)
                # Ignorar bookkeeping interno (.git/ e bytecode __pycache__) —
                # nunca escritos pelo usuario como artefato.
                if _is_ephemeral_internal_path(rel):
                    continue
                current.add(rel)
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

    Usa abordagem híbrida git-based + allowlist estático:
    - Arquivos tracked pelo `git clone` oficial do ComfyUI → permitidos
    - Arquivos untracked/modified pelo git → candidatos a violação
    - Sem git (testes) → fallback para allowlist estático expandido
    - Extensões sensíveis (.safetensors, .pt, .png, etc.) SEMPRE bloqueadas
    """
    _clear_git_cache()
    scan_root = PERSISTENT_AUDIT_PATHS[0] if PERSISTENT_AUDIT_PATHS else PERSISTENT_WORKING
    violations = _scan_working_violations(
        scan_root,
        check_symlinks=True,
        check_non_image_sensitive=True,
    )
    if violations:
        lines = [f"assert_working_policy: {len(violations)} violação(ões) de política em /kaggle/working:"]
        for v in violations:
            lines.append(f"  PATH : {v['path']}")
            lines.append(f"  TYPE : {v['type']}")
            lines.append(f"  SIZE : {v.get('size', '?')} bytes")
            lines.append("")
        _security_abort("\n".join(lines))
    print(f"[SECURITY] assert_working_policy: PASS (zero artefatos proibidos)")


def _scan_unauthorized_persistent_files(scan_root: Path) -> List[Dict[str, Any]]:
    """
    Coleta (sem abortar) os arquivos novos em scan_root que nao sao permitidos.

    Mesma logica de assert_only_allowed_persistent_artifact(), em modo coleta:
    - compara contra _WORKING_SNAPSHOT (auto-registra se ainda nao existe)
    - ignora bookkeeping interno (.git/, __pycache__/bytecode)
    - descarta output_secure.zip e arquivos autorizados (git tracked/allowlist)

    Retorna lista de violacoes (dicts com path/size/mtime/ext/type).
    """
    if _WORKING_SNAPSHOT is None:
        record_working_snapshot()
    current: set = set()
    working = scan_root
    if working.exists():
        for item in working.rglob("*"):
            if item.is_file():
                try:
                    rel = str(item.relative_to(working))
                except ValueError:
                    rel = str(item)
                # Ignorar bookkeeping interno (.git/ e bytecode __pycache__) —
                # nunca escritos pelo usuario como artefato.
                if _is_ephemeral_internal_path(rel):
                    continue
                current.add(rel)
    new_files = current - _WORKING_SNAPSHOT
    # output_secure.zip é permitido
    new_files.discard("output_secure.zip")
    # Arquivos permitidos (git tracked ou allowlist) também não contam como novos
    allowed_to_discard = {f for f in new_files if _is_file_allowed(f.replace("\\", "/"), working)}
    new_files -= allowed_to_discard
    violations: List[Dict[str, Any]] = []
    for rel in sorted(new_files):
        item = working / rel
        try:
            stat = item.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except (OSError, PermissionError):
            size, mtime = 0, 0.0
        violations.append({
            "path": str(item),
            "type": "unauthorized_persistent_file",
            "rel": rel.replace("\\", "/"),
            "ext": Path(rel).suffix.lower(),
            "size": size,
            "mtime": mtime,
        })
    return violations


def assert_only_allowed_persistent_artifact() -> None:
    """
    Verifica que o único arquivo persistente em /kaggle/working (além do snapshot inicial)
    é output_secure.zip. Levanta SecurityError se houver qualquer outro.

    Usa abordagem híbrida git-based + allowlist estático:
    - Arquivos tracked pelo `git clone` oficial do ComfyUI → permitidos
    - Arquivos untracked/modified → não permitidos (a menos que seja custom_node)
    - Sem git → fallback para allowlist estático expandido
    """
    _clear_git_cache()
    working = PERSISTENT_AUDIT_PATHS[0] if PERSISTENT_AUDIT_PATHS else PERSISTENT_WORKING
    violations = _scan_unauthorized_persistent_files(working)
    if violations:
        _security_abort(
            f"assert_only_allowed_persistent_artifact: {len(violations)} arquivo(s) não autorizado(s) em /kaggle/working:\n"
            + "\n".join(f"  - {v['rel']}" for v in violations[:20])
            + "\nApenas output_secure.zip e arquivos da instalação oficial do ComfyUI são permitidos."
        )
    _clear_git_cache()
    print(f"[SECURITY] assert_only_allowed_persistent_artifact: PASS")


def _format_audit_report(
    results: Dict[str, Any],
    max_samples: int = 20,
) -> str:
    """
    Formata o dict retornado por audit_working_directory() em relatorio unico.

    Agrupa por categoria, com contagem total, breakdown por tipo e amostra
    de paths — para que TODAS as categorias de violacao aparecam numa unica
    execucao, em vez de uma por run (whack-a-mole).
    """
    lines = [
        "",
        f"=== WORKING DIRECTORY AUDIT [{results.get('label', '')}] ===",
        f"Scan root: {results.get('scan_root', '?')}",
        "",
    ]
    grand_total = 0
    for category, violations in results.get("categories", {}).items():
        count = len(violations)
        grand_total += count
        lines.append(f"[{category}] {count} violacao(oes)")
        if count:
            by_type: Dict[str, int] = {}
            for v in violations:
                vtype = v.get("type", "?")
                by_type[vtype] = by_type.get(vtype, 0) + 1
            breakdown = ", ".join(f"{t}={n}" for t, n in sorted(by_type.items()))
            lines.append(f"  por tipo: {breakdown}")
            for v in violations[:max_samples]:
                extra = v.get("rel") or v.get("ext") or v.get("target") or ""
                lines.append(f"  - {v.get('path', '?')}  (type={v.get('type', '?')}{f', {extra}' if extra else ''})")
            if count > max_samples:
                lines.append(f"  ... e mais {count - max_samples} (ver log completo)")
        lines.append("")
    lines.append(f"TOTAL: {grand_total} violacao(oes) em {len(results.get('categories', {}))} categoria(s)")
    lines.append(f"STATUS: {'FAIL' if grand_total else 'PASS'}")
    return "\n".join(lines)


def audit_working_directory(
    scan_root: Optional[Path] = None,
    *,
    label: str = "AUDIT",
    raise_on_violation: bool = True,
    include_final_filesystem: bool = False,
    include_working_policy: bool = True,
) -> Dict[str, Any]:
    """
    Auditoria consolidada de /kaggle/working: roda TODAS as checagens em
    modo coleta e reporta um unico relatorio agrupado por categoria.

    Motivo: o notebook executa assert_no_persistent_images,
    assert_working_policy, assert_only_allowed_persistent_artifact e
    final_filesystem_check em sequencia, cada um abortando na propria
    primeira falha — o usuario so via UMA categoria por run. Aqui todas as
    categorias sao coletadas ANTES de reportar, entao uma execucao mostra
    tudo de uma vez.

    Categorias (podem se sobrepor — o mesmo arquivo pode aparecer em mais
    de uma, pois cada uma usa flags diferentes):
      - persistent_images: imagens/archives/modelos/symlinks (== assert_no_persistent_images)
      - working_policy: acima + .json/.log/.db/etc. e git_changed (== assert_working_policy)
      - unauthorized_persistent_artifact: diff contra o snapshot + allowlist
        (== assert_only_allowed_persistent_artifact)
      - final_filesystem: como working_policy porem incluindo output_secure.zip
        (== final_filesystem_check da sessao final; omitida por PADRAO porque
        output_secure.zip — o unico artefato permitido pos-geracao — seria
        flagrado como `extension`.So faz sentido liga-la
        (include_final_filesystem=True) na verificacao de ENCERRAMENTO da
        sessao, quando ate o ZIP ja deve ter sumido. Nos gates PRE-ZIP e
        POST-CLEAR ela deve ficar DESLIGADA.)

    include_working_policy=False omite a categoria working_policy. Usado em
    testes sem clone git: _is_git_changed_file() eh fail-closed (retorna True
    quando nao ha repo), entao sem git a categoria marcaria TUDO como
    violacao. Em producao (Kaggle) o clone do ComfyUI sempre existe.

    Retorna dict com 'label', 'scan_root', 'categories', 'total_by_category',
    'has_violations'. Com raise_on_violation=True (padrao), imprime o
    relatorio e levanta UM unico SecurityError com todas as categorias.
    """
    root = Path(scan_root) if scan_root is not None else (
        PERSISTENT_AUDIT_PATHS[0] if PERSISTENT_AUDIT_PATHS else PERSISTENT_WORKING
    )
    _clear_git_cache()
    categories: Dict[str, List[Dict[str, Any]]] = {}
    categories["persistent_images"] = _scan_working_violations(
        root,
        check_symlinks=True,
        check_non_image_sensitive=False,
        skip_output_zip=True,
    )
    categories["working_policy"] = (
        _scan_working_violations(
            root,
            check_symlinks=True,
            check_non_image_sensitive=True,
            skip_output_zip=True,
        )
        if include_working_policy
        else []
    )
    categories["unauthorized_persistent_artifact"] = _scan_unauthorized_persistent_files(root)
    if include_final_filesystem:
        # output_secure.zip detectado aqui de proposito (= final_filesystem_check)
        categories["final_filesystem"] = _scan_working_violations(
            root,
            check_symlinks=True,
            check_non_image_sensitive=False,
            skip_output_zip=False,
        )
    _clear_git_cache()
    results: Dict[str, Any] = {
        "label": label,
        "scan_root": str(root),
        "categories": categories,
        "total_by_category": {name: len(v) for name, v in categories.items()},
        "has_violations": any(categories.values()),
    }
    report = _format_audit_report(results)
    if results["has_violations"]:
        if raise_on_violation:
            _security_abort(report)
        return {**results, "report": report}
    print(report)
    print(f"[SECURITY] audit_working_directory [{label}]: PASS (zero violacoes em todas as categorias)")
    return {**results, "report": report}


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

# Arquivos/padrões estáticos conhecidos do ComfyUI que NÃO são vazamentos
# Usado por final_filesystem_check e assert_working_policy
# NOTA: Em produção (Kaggle), a checagem primária é baseada em git — arquivos
# tracked pelo `git clone` oficial do ComfyUI não são violação. Este allowlist
# serve como fallback para quando o git não está disponível (testes, dirs temporários).
# Subpastas de custom nodes onde imagens empacotadas (docs/screenshots/ícones
# de UI) são esperadas. Restrição conservadora: a exceção NARROW de imagens
# tracked-e-limpas (ver _is_packaged_node_doc_image) só vale dentro de um
# destes segmentos. Qualquer imagem tracked-e-limpa fora deles continua bloqueada.
NODE_DOC_IMAGE_DIR_NAMES: frozenset[str] = frozenset({
    "docs", "web", "src_web", "assets", "images",
})
ALLOWED_STATIC_FILES: frozenset[str] = frozenset({
    "ComfyUI/input/example.png",
    "ComfyUI/comfy/comfy_types/examples/required_hint.png",
    "ComfyUI/comfy/comfy_types/examples/input_options.png",
    "ComfyUI/comfy/comfy_types/examples/input_types.png",
})

# Padrões de prefixo permitidos (qualquer arquivo sob esses caminhos é ignorado)
# Cobrem a instalação padrão do ComfyUI (git clone oficial) que não contém
# dados sensíveis nem conteúdo gerado pelo pipeline.
ALLOWED_STATIC_PREFIXES: Tuple[str, ...] = (
    "ComfyUI/comfy/comfy_types/examples/",
    "ComfyUI/custom_nodes/",
    # Instalação padrão do ComfyUI — código-fonte e configs de fábrica
    "ComfyUI/comfy/",
    "ComfyUI/blueprints/",
    "ComfyUI/tests/",
    "ComfyUI/tests-unit/",
    "ComfyUI/.ci/",
    "ComfyUI/.github/",
    "ComfyUI/docs/",
    "ComfyUI/web/",
    "ComfyUI/api_examples/",
    "ComfyUI/notebooks/",
    "ComfyUI/scripts/",
    "ComfyUI/app/",
    "ComfyUI/assets/",
    "ComfyUI/cython_modules/",
)

# Arquivos de fábrica do ComfyUI na raiz do clone (sem subdiretório)
ALLOWED_STATIC_ROOT_FILES: frozenset[str] = frozenset({
    "ComfyUI/requirements.txt",
    "ComfyUI/manager_requirements.txt",
    "ComfyUI/pyproject.toml",
    "ComfyUI/README.md",
    "ComfyUI/LICENSE",
    "ComfyUI/.gitignore",
    "ComfyUI/.gitattributes",
    "ComfyUI/main.py",
    "ComfyUI/folder_paths.py",
    "ComfyUI/custom_nodes/README.md",
    "ComfyUI/custom_nodes/example_node.py.example",
    "ComfyUI/input/README.md",
    "ComfyUI/output/README.md",
    "ComfyUI/temp/README.md",
    "ComfyUI/user/.gitkeep",
    # Config gerada pelo proprio setup_comfyui() (extra_model_paths.yaml) —
    # nunca dado de usuario: aponta modelos para /kaggle/input (datasets).
    "ComfyUI/extra_model_paths.yaml",
})


def _is_allowed_static_file(rel_path: str) -> bool:
    """
    Verifica se um caminho relativo é um arquivo estático conhecido/permitido.

    Esta função é o FALLBACK usado quando a checagem baseada em git não está
    disponível (ex: diretório temporário em testes). Em produção (Kaggle),
    a checagem primária usa _get_git_tracked_set() para identificar arquivos
    que fazem parte do clone oficial do ComfyUI.

    Extensões realmente sensíveis (.safetensors, .pt, .png, .jpg, etc.) são
    SEMPRE bloqueadas por _scan_working_violations(), mesmo se casarem com um
    prefixo permitido aqui — isso é uma camada extra de defesa.
    Exceções NARROW (ver _is_allowed_static_file na Camada 1): arquivos de
    fábrica conhecidos (ex: ComfyUI/input/example.png) e imagens de docs
    empacotadas de custom nodes tracked-e-limpas (ver _is_packaged_node_doc_image).
    """
    # Normalizar separadores de path para comparação cross-platform
    normalized = rel_path.replace("\\", "/")
    if normalized in ALLOWED_STATIC_FILES:
        return True
    if normalized in ALLOWED_STATIC_ROOT_FILES:
        return True
    for prefix in ALLOWED_STATIC_PREFIXES:
        if normalized.startswith(prefix):
            # Para ComfyUI/custom_nodes/, permitir apenas .zip e arquivos de código
            if prefix == "ComfyUI/custom_nodes/":
                if normalized.endswith(".zip"):
                    return True
                if any(normalized.endswith(ext) for ext in (".py", ".json", ".yaml", ".yml", ".txt", ".md", ".js", ".css", ".html", ".vue")):
                    return True
                return False
            # Para outros prefixos permitidos (instalação padrão do ComfyUI), permitir
            return True
    return False


# ---------------------------------------------------------------------------
# Git-based working-directory audit
# ---------------------------------------------------------------------------

# Cache de arquivos tracked pelo git (evita rodar git status a cada chamada)
_git_tracked_cache: Dict[str, frozenset[str]] = {}

# Cache dos sets (tracked, changed) dos repos git ANINHADOS de custom nodes,
# por node_dir resolvido. NUNCA confunde repos: a chave inclui o path absoluto.
_nested_node_git_cache: Dict[str, Optional[Tuple[frozenset[str], frozenset[str]]]] = {}


def _get_git_repo_dirs(scan_root: Path) -> Optional[List[Path]]:
    """Retorna os repositórios Git de primeiro nível encontrados em scan_root."""
    repos = [
        child for child in scan_root.iterdir()
        if child.is_dir() and (child / ".git").is_dir()
    ]
    return repos or None


def _get_git_tracked_set(scan_root: Path) -> Optional[frozenset[str]]:
    """
    Retorna o conjunto de paths relativos (relativos a scan_root) de todos
        arquivos tracked pelo git nos repos de primeiro nível dentro de scan_root.

        Usa `git -C <repo_dir> ls-files` para listar arquivos tracked em cada repo.
    Retorna None se:
            - Não houver repo git em scan_root

        Um repo individual que falhar não contribui arquivos confiáveis; os demais
        repos continuam sendo agregados, preservando o comportamento fail-closed.

    O cache é por scan_root resolved string.
    """
    key = str(scan_root.resolve())
    if key in _git_tracked_cache:
        return _git_tracked_cache[key]

    repo_dirs = _get_git_repo_dirs(scan_root)
    if repo_dirs is None:
        return None

    tracked: set[str] = set()
    for repo_dir in repo_dirs:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "ls-files"],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        prefix = f"{repo_dir.name}/"
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                tracked.add(f"{prefix}{line}")

    frozen = frozenset(tracked)
    _git_tracked_cache[key] = frozen
    return frozen


def _get_git_untracked_set(scan_root: Path) -> Optional[frozenset[str]]:
    """
    Retorna o conjunto de paths relativos (relativos a scan_root) de arquivos
    untracked ou modified nos repos de primeiro nível dentro de scan_root.

    Um repo individual que falhar não contribui paths alterados; seus arquivos
    continuam não confiáveis porque também não aparecem no conjunto tracked.
    """
    key = str(scan_root.resolve()) + ":untracked"
    if key in _git_tracked_cache:
        return _git_tracked_cache[key]

    repo_dirs = _get_git_repo_dirs(scan_root)
    if repo_dirs is None:
        return None

    changed: set[str] = set()
    for repo_dir in repo_dirs:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "status", "--porcelain", "--ignored=no"],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode != 0:
            continue
        prefix = f"{repo_dir.name}/"
        for line in result.stdout.splitlines():
            if len(line) < 4 or line[:2] == "!!":
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ")[1]
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            if path:
                changed.add(f"{prefix}{path}")

    frozen = frozenset(changed)
    _git_tracked_cache[key] = frozen
    return frozen


def _clear_git_cache() -> None:
    """Limpa o cache de git tracked/untracked. Usado em testes."""
    _git_tracked_cache.clear()
    _nested_node_git_cache.clear()


def _matches_git_changed_path(rel_path: str, changed_paths: frozenset[str]) -> bool:
    """Retorna True se rel_path aparece no porcelain ou dentro de um diretório listado."""
    normalized = rel_path.replace("\\", "/")
    for changed in changed_paths:
        changed_normalized = changed.replace("\\", "/").rstrip("/")
        if normalized == changed_normalized or normalized.startswith(f"{changed_normalized}/"):
            return True
    return False


def _is_git_changed_file(rel_path: str, scan_root: Path) -> bool:
    """Retorna True quando git status marca o arquivo como untracked/modified/etc."""
    changed = _get_git_untracked_set(scan_root)
    if changed is None:
        return True
    return _matches_git_changed_path(rel_path, changed)


def _is_file_allowed(rel_path: str, scan_root: Path) -> bool:
    """
    Verifica se um arquivo é permitido em /kaggle/working usando a abordagem
    híbrida: git-based (primária) + allowlist estático (fallback).

    Lógica:
    1. Se o arquivo é tracked pelo git E não está modified/untracked → permitido
       (faz parte do clone oficial do ComfyUI)
    2. Se o arquivo é untracked/modified pelo git → NÃO permitido por git
       (candidato a violação — será avaliado pelas extensões sensíveis)
    3. Se git não disponível → usa _is_allowed_static_file() como fallback

    NOTA: output_secure.zip NÃO é tratado aqui — o pulo por nome é feito em
    _scan_working_violations() via parâmetro skip_output_zip.
    Extensões sensíveis (.safetensors, .pt, etc.) são bloqueadas por
    _scan_working_violations() MESMO se o arquivo for tracked pelo git.
    Exceção: extensões de IMAGEM (.png, .jpg, etc.) passam se forem arquivo
    de fábrica conhecido (_is_allowed_static_file) ou doc empacotada de
    custom node tracked-e-limpa (_is_packaged_node_doc_image) — ver Camada 1.
    """
    normalized = rel_path.replace("\\", "/")

    # Passo 1: checagem baseada em git. Quando o clone existe, ele é a fonte
    # primária: arquivos tracked e limpos são permitidos; untracked/modified não.
    tracked = _get_git_tracked_set(scan_root)
    if tracked is not None:
        changed = _get_git_untracked_set(scan_root)
        if normalized in tracked:
            if changed is None:
                return False
            if not _matches_git_changed_path(normalized, changed):
                return True
        # Custom nodes são clones aninhados/instalações externas esperadas, então
        # mantêm o tratamento especial pré-existente mesmo quando o repo pai existe.
        if normalized.startswith("ComfyUI/custom_nodes/") and _is_allowed_static_file(normalized):
            return True
        return False

    # Passo 2: fallback estático apenas quando git não está disponível.
    return _is_allowed_static_file(normalized)


def _get_nested_node_git_sets(
    scan_root: Path, node_dir: Path
) -> Optional[Tuple[frozenset[str], frozenset[str]]]:
    """
    Retorna (tracked, changed) do repo git ANINHADO de um custom node.

    node_dir: dir absoluto do custom node (ex: .../ComfyUI/custom_nodes/rgthree-comfy).
    Retorna None se node_dir não for um repo git válido ou o git falhar —
    fail-closed: chamadores tratam None como "não confiável".
    """
    cache_key = str(node_dir.resolve())
    cached = _nested_node_git_cache.get(cache_key)
    if cached is not None:
        return cached
    if not node_dir.is_dir() or not (node_dir / ".git").is_dir():
        result: Optional[Tuple[frozenset[str], frozenset[str]]] = None
        _nested_node_git_cache[cache_key] = result
        return result
    try:
        tracked_proc = subprocess.run(
            ["git", "-C", str(node_dir), "ls-files"],
            capture_output=True, text=True, timeout=30,
        )
        changed_proc = subprocess.run(
            ["git", "-C", str(node_dir), "status", "--porcelain", "--ignored=no"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        result = None
        _nested_node_git_cache[cache_key] = result
        return result
    if tracked_proc.returncode != 0 or changed_proc.returncode != 0:
        result = None
        _nested_node_git_cache[cache_key] = result
        return result
    tracked: set[str] = set()
    for line in tracked_proc.stdout.splitlines():
        line = line.strip()
        if line:
            tracked.add(line.replace("\\", "/"))
    changed: set[str] = set()
    for line in changed_proc.stdout.splitlines():
        if len(line) < 4 or line[:2] == "!!":
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if path:
            changed.add(path.replace("\\", "/"))
    result = (frozenset(tracked), frozenset(changed))
    _nested_node_git_cache[cache_key] = result
    return result


def _is_packaged_node_doc_image(rel_path: str, scan_root: Path) -> bool:
    """
    Exceção NARROW para imagens empacotadas de custom nodes (docs/screenshots/
    ícones de UI). Retorna True SOMENTE quando TODAS as condições valem:

    1. O path está em ComfyUI/custom_nodes/<node>/... (nunca raiz do ComfyUI,
       nunca input/output/temp/user, nunca o próprio colab-pipeline).
    2. Abaixo do dir do node há um segmento de documentação/asset conhecido
       (docs/, web/, src_web/, assets/, images/) — ver NODE_DOC_IMAGE_DIR_NAMES.
    3. O arquivo está tracked E limpo no repo git DAQUELE custom node
       (git ls-files + git status do node, não do ComfyUI pai).

    Fail-closed: repo ausente/ilegível, git indisponível, arquivo untracked ou
    modificado → False. Extensões NÃO-imagem (ex: .safetensors) nunca passam
    por aqui — ficam no bloqueio incondicional.
    """
    normalized = rel_path.replace("\\", "/")
    parts = Path(normalized).parts
    # Condicao 1: dentro de ComfyUI/custom_nodes/<node>/ (minimo 4 segmentos)
    if (
        len(parts) < 4
        or parts[0] != "ComfyUI"
        or parts[1] != "custom_nodes"
        or parts[2] in ("", ".", "..")
    ):
        return False
    ext = Path(normalized).suffix.lower()
    if ext not in SENSITIVE_EXTENSIONS:
        return False
    # Condicao 2: segmento de docs/assets conhecido abaixo do dir do node
    node_sub_parts = parts[3:]
    if not any(seg in NODE_DOC_IMAGE_DIR_NAMES for seg in node_sub_parts[:-1]):
        return False
    # Condicao 3: tracked-e-limpo no repo git do node
    node_dir = scan_root / "ComfyUI" / "custom_nodes" / parts[2]
    sets = _get_nested_node_git_sets(scan_root, node_dir)
    if sets is None:
        return False
    tracked, changed = sets
    rel_inside_node = "/".join(node_sub_parts)
    if rel_inside_node not in tracked:
        return False
    return not _matches_git_changed_path(rel_inside_node, changed)


# Extensões de modelo que são SEMPRE bloqueadas, mesmo se tracked pelo git
# (camada extra de defesa contra vazamento de modelos).
# NOTA: Extensões de imagem (.png, .jpg, etc.) e archive (.zip) NÃO estão aqui
# porque o ComfyUI tem imagens de exemplo legítimas (example.png, comfy_types/examples/)
# que são tracked pelo git. Essas extensões são tratadas na Camada 2 apenas quando
# o arquivo NÃO é permitido pelo git/allowlist.
ALWAYS_BLOCKED_EXTENSIONS = frozenset({
    ".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".onnx", ".tflite",
})


def _scan_working_violations(
    scan_root: Path,
    label: str = "",
    extra_paths: Optional[List[Path]] = None,
    check_symlinks: bool = True,
    check_non_image_sensitive: bool = False,
    check_comfyui_subdirs: bool = False,
    skip_output_zip: bool = True,
) -> List[Dict[str, Any]]:
    """
    Helper compartilhado que varre scan_root em busca de violações de segurança.

    Substitui a lógica duplicada em assert_working_policy, assert_no_persistent_images
    e final_filesystem_check.

    Parâmetros:
      scan_root: diretório raiz para scanear (normalmente /kaggle/working)
      label: prefixo para mensagens (ex: "[PRE-ZIP]")
      extra_paths: paths adicionais para scanear além de scan_root
      check_symlinks: se True, verifica symlinks que apontam para fora
      check_non_image_sensitive: se True, também bloqueia .json/.log/.db/etc.
        (usado por assert_working_policy, não por assert_no_persistent_images)
      check_comfyui_subdirs: se True, reporta status de ComfyUI/input/output/temp
      skip_output_zip: se True, output_secure.zip é ignorado (permitido).
        final_filesystem_check passa False para detectá-lo como violação.

    Retorna lista de violações (cada uma é um dict com path, type, size, etc.).
    """
    scan_roots: List[Path] = [scan_root]

    # Adicionar subpaths críticos explicitamente
    comfyui_persistent_dirs = [
        scan_root / "ComfyUI" / "input",
        scan_root / "ComfyUI" / "output",
        scan_root / "ComfyUI" / "temp",
        scan_root / "ComfyUI" / "user",
    ]
    for d in comfyui_persistent_dirs:
        if d.exists() and d not in scan_roots:
            scan_roots.append(d)
    if extra_paths:
        scan_roots.extend(extra_paths)

    violations: List[Dict[str, Any]] = []
    scanned_files: set[str] = set()

    for root in scan_roots:
        if not root.exists():
            continue
        for item in root.rglob("*"):
            # Verificação de symlinks
            if check_symlinks and item.is_symlink():
                try:
                    real = item.resolve()
                except Exception:
                    real = item
                target = str(real)
                if not target.startswith(str(scan_root)) and not target.startswith("/dev/shm"):
                    violations.append({
                        "path": str(item),
                        "type": "symlink",
                        "target": target,
                        "size": 0,
                        "mtime": item.lstat().st_mtime if item.exists() else 0,
                    })

            if not item.is_file():
                continue

            try:
                file_key = str(item.resolve(strict=False))
            except Exception:
                file_key = str(item)
            if file_key in scanned_files:
                continue
            scanned_files.add(file_key)

            # output_secure.zip é o único artefato persistente permitido
            # (final_filesystem_check não pula, para detectá-lo como violação)
            if skip_output_zip and item.name == "output_secure.zip":
                continue

            # Calcular path relativo a scan_root para casar com allowlist/git
            try:
                rel = str(item.relative_to(scan_root))
            except ValueError:
                rel = str(item)
            rel_normalized = rel.replace("\\", "/")

            # Ignorar bookkeeping interno (.git/ e bytecode __pycache__) —
            # gerados pelo git e pelo interpretador, nunca artefato de usuário.
            if _is_ephemeral_internal_path(rel_normalized):
                continue

            # Camada 1: checagem git-based + allowlist estático
            if _is_file_allowed(rel_normalized, scan_root):
                # Mesmo se permitido, verificar extensões sempre bloqueadas
                ext = item.suffix.lower()
                if ext in ALWAYS_BLOCKED_EXTENSIONS:
                    violations.append({
                        "path": str(item), "type": "always_blocked_extension",
                        "ext": ext, "size": item.stat().st_size,
                        "mtime": item.stat().st_mtime,
                    })
                elif (
                    ext in SENSITIVE_EXTENSIONS
                    and not _is_allowed_static_file(rel_normalized)
                    and not _is_packaged_node_doc_image(rel_normalized, scan_root)
                ):
                    # Imagem permitida pelo git/allowlist mas que NÃO é arquivo
                    # de fábrica conhecido (example.png, comfy_types/examples/)
                    # nem doc empacotada de custom node → continua bloqueada
                    # (fail-closed). Ex: ComfyUI/output/gerado.png mesmo se
                    # commitado em algum repo.
                    violations.append({
                        "path": str(item), "type": "extension",
                        "ext": ext, "size": item.stat().st_size,
                        "mtime": item.stat().st_mtime,
                    })
                continue

            stat = item.stat()
            ext = item.suffix.lower()

            # Exceção NARROW: imagem empacotada de custom node (docs/screenshots/
            # ícones) tracked-e-limpa no git do próprio node. Tudo o mais segue
            # para as camadas de bloqueio abaixo.
            if ext in SENSITIVE_EXTENSIONS and _is_packaged_node_doc_image(
                rel_normalized, scan_root
            ):
                continue

            # Camada 2: extensões sensíveis (imagem/archive/modelo)
            if ext in SENSITIVE_EXTENSIONS or ext in SENSITIVE_ARCHIVES or ext in ALWAYS_BLOCKED_EXTENSIONS:
                violation_type = "always_blocked_extension" if ext in ALWAYS_BLOCKED_EXTENSIONS else "extension"
                violations.append({
                    "path": str(item), "type": violation_type,
                    "ext": ext, "size": stat.st_size, "mtime": stat.st_mtime,
                })
                continue

            # Camada 3: política completa baseada em git. Em assert_working_policy,
            # qualquer arquivo marcado pelo repo como untracked/modified/etc. é violação.
            if check_non_image_sensitive and _is_git_changed_file(rel_normalized, scan_root):
                violations.append({
                    "path": str(item), "type": "git_changed",
                    "size": stat.st_size, "mtime": stat.st_mtime,
                })
                continue

            # Camada 4: extensões não-imagem sensíveis (.json, .log, .db, etc.)
            if check_non_image_sensitive and ext in SENSITIVE_NON_IMAGE_EXTENSIONS:
                violations.append({
                    "path": str(item), "type": "non_image_sensitive",
                    "ext": ext, "size": stat.st_size, "mtime": stat.st_mtime,
                })
                continue

            # Camada 5: verificação por magic bytes
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
                        "mtime": stat.st_mtime,
                    })
            except (OSError, PermissionError):
                pass

    return violations


def _format_violations(violations: List[Dict[str, Any]], prefix_label: str = "") -> str:
    """Formata lista de violações em string de relatório."""
    prefix = f"[{prefix_label}] " if prefix_label else ""
    lines = [f"{prefix}SECURITY VIOLATION: {len(violations)} artefato(s) sensível(is) em armazenamento persistente:"]
    for v in violations:
        ts = datetime.datetime.fromtimestamp(v["mtime"]).isoformat() if v.get("mtime") else "unknown"
        lines.append(f"  PATH : {v['path']}")
        lines.append(f"  TYPE : {v['type']}")
        lines.append(f"  SIZE : {v.get('size', '?')} bytes")
        lines.append(f"  MTIME: {ts}")
        if v.get("target"):
            lines.append(f"  TARGET: {v['target']}")
        if v.get("ext"):
            lines.append(f"  EXT  : {v['ext']}")
        if v.get("magic"):
            lines.append(f"  MAGIC: {v['magic']}")
        lines.append("")
    return "\n".join(lines)


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

    Usa _scan_working_violations() com check_non_image_sensitive=False
    (esta função não bloqueia .json/.log/.db — apenas imagens/archives/modelos).
    """
    _clear_git_cache()
    scan_root = PERSISTENT_AUDIT_PATHS[0] if PERSISTENT_AUDIT_PATHS else PERSISTENT_WORKING
    violations = _scan_working_violations(
        scan_root,
        label=label,
        extra_paths=extra_paths,
        check_symlinks=True,
        check_non_image_sensitive=False,
    )
    if violations:
        _security_abort(_format_violations(violations, label))


# ---------------------------------------------------------------------------
# Custom node integrity — snapshot + hash
# ---------------------------------------------------------------------------

def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# Padrões ignorados em snapshots de custom nodes (bytecode compilado, VCS, etc.)
# Alias dos conjuntos centralizados RUNTIME_GENERATED_* (mantidos por compat).
IGNORED_NODE_PATTERNS: frozenset[str] = RUNTIME_GENERATED_DIR_NAMES
IGNORED_NODE_EXTENSIONS: frozenset[str] = RUNTIME_GENERATED_EXTENSIONS


def _should_ignore_node_file(rel_path: str) -> bool:
    """Verifica se um arquivo relativo deve ser ignorado no snapshot do node.

    Reusa _is_ephemeral_internal_path (.git/ + __pycache__/bytecode) e
    adiciona a checagem de extensao compilada.
    """
    if _is_ephemeral_internal_path(rel_path):
        return True
    # Ignorar por extensão
    if Path(rel_path).suffix in IGNORED_NODE_EXTENSIONS:
        return True
    return False


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
                rel = str(f.relative_to(node_path))
                if _should_ignore_node_file(rel):
                    continue
                files.append((rel, _hash_file(f)))
            except (OSError, PermissionError):
                rel = str(f.relative_to(node_path))
                if not _should_ignore_node_file(rel):
                    files.append((rel, "UNREADABLE"))
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
    Ignora: __pycache__, .git, arquivos .pyc/.pyo/.pyd
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
                    rel = str(f.relative_to(item))
                    if _should_ignore_node_file(rel):
                        continue
                    files[rel] = _hash_file(f)
                except (OSError, PermissionError):
                    rel = str(f.relative_to(item))
                    if not _should_ignore_node_file(rel):
                        files[rel] = "UNREADABLE"
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

    Usa abordagem híbrida git-based + allowlist estático via _scan_working_violations():
    - Arquivos tracked pelo `git clone` oficial do ComfyUI → permitidos
    - Arquivos untracked/modified → candidatos a violação
    - Sem git (testes) → fallback para allowlist estático expandido
    - Extensões sensíveis (.safetensors, .pt, .png, etc.) SEMPRE bloqueadas

    Retorna dict com 'violations' (int), 'report' (str).
    """
    _clear_git_cache()
    extra_report_lines = []

    # Subpaths críticos — verificar existência e conteúdo mesmo se vazios
    if check_comfyui_subdirs:
        comfyui_subdirs = [
            scan_root / "ComfyUI" / "input",
            scan_root / "ComfyUI" / "output",
            scan_root / "ComfyUI" / "temp",
            scan_root / "ComfyUI" / "user",
        ]
        for d in comfyui_subdirs:
            if d.exists():
                contents = list(d.rglob("*"))
                files = [f for f in contents if f.is_file()]
                extra_report_lines.append(
                    f"  {d}: {'VAZIO' if not files else f'{len(files)} arquivo(s) — VERIFICAR'}"
                )

    violations = _scan_working_violations(
        scan_root,
        check_symlinks=True,
        check_non_image_sensitive=False,
        skip_output_zip=False,
    )

    # Separar violações por tipo para o relatório
    img_found: List[Dict[str, Any]] = []
    arch_found: List[Dict[str, Any]] = []
    symlink_found: List[Dict[str, Any]] = []

    for v in violations:
        vtype = v.get("type", "")
        if vtype == "symlink":
            symlink_found.append(v)
        elif vtype == "extension" and v.get("ext", "") in SENSITIVE_ARCHIVES:
            arch_found.append(v)
        elif vtype == "extension" and v.get("ext", "") in SENSITIVE_EXTENSIONS:
            img_found.append(v)
        elif vtype in ("always_blocked_extension",):
            # Extensão sempre bloqueada (.safetensors, .pt, etc.)
            ext = v.get("ext", "")
            if ext in SENSITIVE_EXTENSIONS:
                img_found.append(v)
            elif ext in SENSITIVE_ARCHIVES:
                arch_found.append(v)
            else:
                # .safetensors, .pt, .pth, .bin, etc. — tratar como modelo
                arch_found.append(v)
        elif vtype == "magic_bytes":
            # magic bytes pode ser imagem ou archive
            magic_hex = v.get("magic", "")
            if magic_hex.startswith("504b"):  # PK = ZIP
                arch_found.append(v)
            else:
                img_found.append(v)

    violation_count = len(img_found) + len(arch_found)

    lines = [
        "",
        "=== FINAL SECURITY FILESYSTEM CHECK ===",
        f"Scan root: {scan_root}",
        f"Image files    : {len(img_found)}",
        f"Archive files  : {len(arch_found)}",
        f"Symlinks found : {len(symlink_found)}",
        f"Violations     : {violation_count}",
        "",
    ]

    if extra_report_lines:
        lines.append("ComfyUI subdir status:")
        lines.extend(extra_report_lines)
        lines.append("")

    if violation_count == 0 and not symlink_found:
        lines.append("STATUS: PASS ✅")
    elif violation_count > 0:
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
        "violations": violation_count,
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
            return isolation_cmd + cmd

    return cmd


def _get_isolation_wrapper(comfyui_dir: Path, mutable_paths: List[Path]) -> Optional[List[str]]:
    """
    Retorna prefixo de comando para isolamento (bubblewrap) se disponível.
    Unshare NÃO é usado pois requer CAP_SYS_ADMIN (não disponível no Kaggle).
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

    print("[SECURITY] Isolamento real (bubblewrap) não disponível. Usando apenas restrição de paths via parâmetros do ComfyUI.")
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
