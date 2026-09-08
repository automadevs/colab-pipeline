#!/usr/bin/env python3
"""
Sincronização de outputs, workflows, logs e metadata com Google Drive.
Suporta Kaggle (rclone + service account) e Colab (mount nativo).
O Google Drive é utilizado estritamente para persistência, backup e sync,
NUNCA como caminho crítico de geração do ComfyUI.
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
DEFAULT_LOCAL_OUTPUTS = Path("/kaggle/working/ComfyUI/output")
DEFAULT_LOCAL_WORKFLOWS = Path("/kaggle/working/ComfyUI/user/default/workflows")
DEFAULT_LOCAL_LOGS = Path("/kaggle/working/ComfyUI")
DEFAULT_LOCAL_METADATA = Path("/kaggle/working/ComfyUI/metadata")

CHUNK_SIZE = 1024 * 1024  # 1 MB chunk streaming


def detect_env() -> str:
    """Detecta se está rodando no Colab ou Kaggle."""
    if os.path.exists("/content"):
        return "colab"
    elif os.path.exists("/kaggle"):
        return "kaggle"
    return "unknown"


def calculate_sha256_stream(filepath: Path) -> str:
    """
    Calcula SHA-256 do arquivo em streaming/chunks de ~1 MB.
    Nunca carrega o arquivo inteiro em memória.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def setup_rclone_kaggle() -> bool:
    """Configura rclone no Kaggle usando service account do Secret."""
    rclone_conf = Path("/root/.config/rclone/rclone.conf")
    if rclone_conf.exists():
        return True

    sa_json = None
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        sa_json = client.get_secret("GDRIVE_SERVICE_ACCOUNT_JSON")
    except Exception:
        pass

    if not sa_json:
        sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")

    if not sa_json:
        print("[WARN] GDRIVE_SERVICE_ACCOUNT_JSON não encontrado nos Secrets/env")
        print("       Configure no Kaggle: Secrets → GDRIVE_SERVICE_ACCOUNT_JSON")
        return False

    sa_path = Path("/root/gdrive_sa.json")
    sa_path.write_text(sa_json)

    rclone_conf.parent.mkdir(parents=True, exist_ok=True)
    config_content = f"""[gdrive]
type = drive
scope = drive
service_account_file = {sa_path}
"""
    rclone_conf.write_text(config_content)
    print("[INFO] rclone configurado com service account")
    return True


def mount_drive_colab(drive_base: str = DEFAULT_DRIVE_BASE) -> Path:
    """Monta Google Drive no Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        drive_path = Path("/content/drive/MyDrive") / drive_base
        drive_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Google Drive montado em: {drive_path}")
        return drive_path
    except ImportError:
        raise RuntimeError("google.colab não disponível. Não está no Colab?")


def get_drive_path_kaggle(drive_base: str = DEFAULT_DRIVE_BASE) -> Path:
    """Retorna path montado via rclone no Kaggle."""
    mount_point = Path("/kaggle/working/gdrive")
    mount_point.mkdir(parents=True, exist_ok=True)

    # Verificar se já montado
    is_mounted = False
    try:
        if list(mount_point.iterdir()):
            is_mounted = True
    except Exception:
        is_mounted = False

    if not is_mounted:
        print("[INFO] Montando Google Drive via rclone...")
        result = subprocess.run([
            "rclone", "mount", "gdrive:",
            str(mount_point),
            "--vfs-cache-mode", "full",
            "--daemon"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            result = subprocess.run([
                "rclone", "lsd", "gdrive:"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Falha ao acessar Drive via rclone: {result.stderr.strip()}")

    drive_path = mount_point / drive_base
    drive_path.mkdir(parents=True, exist_ok=True)
    return drive_path


def get_drive_path(drive_base: str = DEFAULT_DRIVE_BASE, env: Optional[str] = None) -> Path:
    """Obtém path raiz do projeto no Google Drive baseado no ambiente."""
    if env is None:
        env = detect_env()

    if env == "colab":
        return mount_drive_colab(drive_base)
    elif env == "kaggle":
        if not setup_rclone_kaggle():
            raise RuntimeError("Falha ao configurar rclone no Kaggle")
        return get_drive_path_kaggle(drive_base)
    else:
        # Fallback local para desenvolvimento/testes
        test_dir = Path("/tmp/gdrive_test") if os.name != "nt" else Path(os.environ.get("TEMP", ".")) / "gdrive_test"
        p = test_dir / drive_base
        p.mkdir(parents=True, exist_ok=True)
        return p


def test_drive_connection(drive_base: str = DEFAULT_DRIVE_BASE, env: Optional[str] = None) -> Dict[str, Any]:
    """
    Operação 'test' que valida:
    - Configuração do rclone / mount
    - Acesso ao Drive
    - Existência/criação da estrutura Automa/ComfyUI (outputs, workflows, logs, metadata)
    - Capacidade de listar o diretório sem transferir arquivos
    """
    print("=" * 60)
    print("TESTE DE CONEXÃO COM GOOGLE DRIVE")
    print("=" * 60)
    res: Dict[str, Any] = {
        "status": "fail",
        "env": env or detect_env(),
        "drive_path": None,
        "subdirs": [],
        "error": None
    }
    try:
        drive_path = get_drive_path(drive_base=drive_base, env=env)
        res["drive_path"] = str(drive_path)
        print(f"[OK] Drive acessível em: {drive_path}")

        subdirs = ["outputs", "workflows", "logs", "metadata"]
        created = []
        for s in subdirs:
            p = drive_path / s
            p.mkdir(parents=True, exist_ok=True)
            created.append(s)
            print(f"  [OK] Estrutura verificada: {p}")

        res["subdirs"] = created
        res["status"] = "pass"
        print("[SUCCESS] Teste de conexão e estrutura do Drive concluído com sucesso.")
    except Exception as e:
        res["error"] = str(e)
        print(f"[ERROR] Falha no teste de conexão com o Drive: {e}")

    return res


def _sync_directory(
    src_dir: Path,
    dst_dir: Path,
    category: str,
    patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sincroniza arquivos de src_dir para dst_dir de forma estritamente idempotente.
    - Se arquivo não existe no destino: copia -> synced
    - Se tamanho é diferente: copia -> synced
    - Se mesmo tamanho e mtime idêntico com tolerância, verifica hash antes de pular.
    - Se mesmo tamanho + mesmo hash: pula -> skipped
    - Se mesmo tamanho + hash diferente: copia -> synced
    - Erros de leitura/cópia registrados em errors.
    """
    stats: Dict[str, Any] = {
        "category": category,
        "src": str(src_dir),
        "dst": str(dst_dir),
        "synced": 0,
        "skipped": 0,
        "errors": 0,
        "details": [],
    }

    if not src_dir.exists():
        return stats

    dst_dir.mkdir(parents=True, exist_ok=True)

    files_to_process: List[Path] = []
    try:
        for item in src_dir.rglob("*"):
            if item.is_file():
                if patterns:
                    matched = any(item.match(p) for p in patterns)
                    if not matched:
                        continue
                files_to_process.append(item)
    except Exception as e:
        stats["errors"] += 1
        stats["details"].append({
            "category": category,
            "file": str(src_dir),
            "status": "error",
            "error": f"Erro ao listar arquivos da origem: {e}"
        })
        return stats

    for src_file in files_to_process:
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel_path

        try:
            # 1. Destino inexistente -> SYNCED
            if not dst_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                size_mb = src_file.stat().st_size / (1024 ** 2)
                stats["synced"] += 1
                stats["details"].append({
                    "category": category,
                    "file": str(rel_path),
                    "local_path": str(src_file),
                    "remote_path": str(dst_file),
                    "status": "synced",
                    "reason": "new_file",
                    "size_mb": round(size_mb, 3),
                })
                continue

            src_stat = src_file.stat()
            dst_stat = dst_file.stat()

            # 2. Tamanho diferente -> SYNCED
            if src_stat.st_size != dst_stat.st_size:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                size_mb = src_file.stat().st_size / (1024 ** 2)
                stats["synced"] += 1
                stats["details"].append({
                    "category": category,
                    "file": str(rel_path),
                    "local_path": str(src_file),
                    "remote_path": str(dst_file),
                    "status": "synced",
                    "reason": "size_diff",
                    "size_mb": round(size_mb, 3),
                })
                continue

            # 3. Mesmo tamanho: validar SHA-256 via streaming para garantir igualdade
            # (Nunca assumir mesmo conteúdo apenas por mesmo tamanho)
            src_hash = calculate_sha256_stream(src_file)
            dst_hash = calculate_sha256_stream(dst_file)

            if src_hash == dst_hash:
                # Mesmo tamanho + mesmo hash -> SKIPPED
                size_mb = src_stat.st_size / (1024 ** 2)
                stats["skipped"] += 1
                stats["details"].append({
                    "category": category,
                    "file": str(rel_path),
                    "local_path": str(src_file),
                    "remote_path": str(dst_file),
                    "status": "skipped",
                    "reason": "identical_hash",
                    "size_mb": round(size_mb, 3),
                })
            else:
                # Mesmo tamanho + hash diferente -> SYNCED
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                size_mb = src_file.stat().st_size / (1024 ** 2)
                stats["synced"] += 1
                stats["details"].append({
                    "category": category,
                    "file": str(rel_path),
                    "local_path": str(src_file),
                    "remote_path": str(dst_file),
                    "status": "synced",
                    "reason": "hash_diff",
                    "size_mb": round(size_mb, 3),
                })

        except Exception as e:
            stats["errors"] += 1
            stats["details"].append({
                "category": category,
                "file": str(rel_path),
                "local_path": str(src_file),
                "remote_path": str(dst_file),
                "status": "error",
                "error": str(e),
            })

    return stats


def _resolve_paths(
    drive_path: Path,
    local_outputs: Path,
    local_workflows: Path,
    local_logs: Path,
    local_metadata: Path,
    action: str = "push",
) -> Dict[str, Dict[str, Any]]:
    """Mapeia cada categoria para suas localizações local e no Drive e filtros padrão."""
    # Para workflows: destino local e origem preferencial é ComfyUI/user/default/workflows.
    # No push, se a pasta padrão ainda não existir, busca subpasta 'workflows' em user.
    wf_local = local_workflows
    if action == "push" and not wf_local.exists():
        alt_wf = local_workflows.parent / "workflows"
        if alt_wf.exists():
            wf_local = alt_wf
        elif Path("/kaggle/working/ComfyUI/user").exists():
            wf_user = Path("/kaggle/working/ComfyUI/user")
            # Procura pasta workflows dentro de user
            candidates = list(wf_user.glob("**/workflows"))
            if candidates:
                wf_local = candidates[0]
            else:
                wf_local = wf_user

    return {
        "outputs": {
            "local": local_outputs,
            "drive": drive_path / "outputs",
            "patterns": ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.webm", "*.gif", "*.json"],
        },
        "workflows": {
            "local": wf_local,
            "drive": drive_path / "workflows",
            "patterns": ["*.json"],  # Workflows são arquivos JSON; evita configurações temporárias e cache
        },
        "logs": {
            "local": local_logs,
            "drive": drive_path / "logs",
            "patterns": ["*.log", "*.txt"],
        },
        "metadata": {
            "local": local_metadata,
            "drive": drive_path / "metadata",
            "patterns": ["*.json", "*.csv", "*.yaml", "*.yml"],
        },
    }


def sync_category(
    category: str,
    action: str = "push",
    local_outputs: Path = DEFAULT_LOCAL_OUTPUTS,
    local_workflows: Path = DEFAULT_LOCAL_WORKFLOWS,
    local_logs: Path = DEFAULT_LOCAL_LOGS,
    local_metadata: Path = DEFAULT_LOCAL_METADATA,
    drive_base: str = DEFAULT_DRIVE_BASE,
    env: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    drive_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Sincroniza uma categoria individual em 'push' (local -> Drive) ou 'pull' (Drive -> local).
    Categorias válidas: 'outputs', 'workflows', 'logs', 'metadata'.
    """
    if category not in {"outputs", "workflows", "logs", "metadata"}:
        raise ValueError(f"Categoria desconhecida: '{category}'. Use outputs, workflows, logs ou metadata.")

    if action not in {"push", "pull"}:
        raise ValueError(f"Ação inválida: '{action}'. Use 'push' ou 'pull'.")

    if drive_path is None:
        drive_path = get_drive_path(drive_base=drive_base, env=env)

    mapping = _resolve_paths(drive_path, local_outputs, local_workflows, local_logs, local_metadata, action=action)
    cat_cfg = mapping[category]

    use_patterns = patterns or cat_cfg["patterns"]

    if action == "push":
        src = cat_cfg["local"]
        dst = cat_cfg["drive"]
    else:  # pull
        src = cat_cfg["drive"]
        dst = cat_cfg["local"]

    if not src.exists():
        return {
            "category": category,
            "action": action,
            "src": str(src),
            "dst": str(dst),
            "synced": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
            "message": f"Diretório de origem não existe: {src}",
        }

    stats = _sync_directory(src, dst, category=category, patterns=use_patterns)
    stats["action"] = action
    return stats


def sync_outputs(
    action: str = "push",
    categories: Optional[List[str]] = None,
    local_outputs: Path = DEFAULT_LOCAL_OUTPUTS,
    local_workflows: Path = DEFAULT_LOCAL_WORKFLOWS,
    local_logs: Path = DEFAULT_LOCAL_LOGS,
    local_metadata: Path = DEFAULT_LOCAL_METADATA,
    drive_base: str = DEFAULT_DRIVE_BASE,
    env: Optional[str] = None,
    patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sincronização flexível e idempotente com Google Drive por categoria individual.
    - action: 'push' (local -> Drive) ou 'pull' (Drive -> local)
    - categories: lista de categorias a sincronizar.
      Se push: padrão é ['outputs', 'workflows', 'logs', 'metadata']
      Se pull: padrão é ['workflows'] (com suporte explícito para qualquer combinação: 'outputs', 'logs', 'metadata')
    """
    if action == "push":
        selected_categories = categories or ["outputs", "workflows", "logs", "metadata"]
    elif action == "pull":
        selected_categories = categories or ["workflows"]
    else:
        raise ValueError(f"Ação inválida: '{action}'. Use 'push' ou 'pull'.")

    print(f"[INFO] === SYNC GOOGLE DRIVE ({action.upper()}) ===")
    print(f"[INFO] Ambiente: {env or detect_env()}")
    print(f"[INFO] Categorias selecionadas: {selected_categories}")

    drive_path = get_drive_path(drive_base=drive_base, env=env)

    total_stats: Dict[str, Any] = {
        "action": action,
        "synced": 0,
        "skipped": 0,
        "errors": 0,
        "categories": {},
        "details": [],
    }

    for cat in selected_categories:
        res = sync_category(
            category=cat,
            action=action,
            local_outputs=local_outputs,
            local_workflows=local_workflows,
            local_logs=local_logs,
            local_metadata=local_metadata,
            drive_base=drive_base,
            env=env,
            patterns=patterns,
            drive_path=drive_path,
        )
        total_stats["synced"] += res["synced"]
        total_stats["skipped"] += res["skipped"]
        total_stats["errors"] += res["errors"]
        total_stats["categories"][cat] = {
            "synced": res["synced"],
            "skipped": res["skipped"],
            "errors": res["errors"],
        }
        total_stats["details"].extend(res.get("details", []))
        print(f"  [{cat}] {action.upper()}: {res['synced']} novos/modificados, {res['skipped']} inalterados, {res['errors']} erros")

    print(f"\n[INFO] === RESUMO GERAL ({action.upper()}) ===")
    print(f"  Total sincronizados: {total_stats['synced']}")
    print(f"  Total ignorados (mesmo tamanho + mesmo hash): {total_stats['skipped']}")
    print(f"  Total erros: {total_stats['errors']}")

    return total_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync com Google Drive (outputs, workflows, logs, metadata)")
    parser.add_argument("--action", choices=["push", "pull", "test"], default="push", help="Ação a executar")
    parser.add_argument("--categories", nargs="*", choices=["outputs", "workflows", "logs", "metadata"], help="Categorias específicas")
    parser.add_argument("--local-outputs", default=str(DEFAULT_LOCAL_OUTPUTS))
    parser.add_argument("--local-workflows", default=str(DEFAULT_LOCAL_WORKFLOWS))
    parser.add_argument("--local-logs", default=str(DEFAULT_LOCAL_LOGS))
    parser.add_argument("--local-metadata", default=str(DEFAULT_LOCAL_METADATA))
    parser.add_argument("--drive-base", default=DEFAULT_DRIVE_BASE)
    parser.add_argument("--env", choices=["colab", "kaggle"])
    parser.add_argument("--patterns", nargs="*", help="Filtros glob (ex: *.png *.json)")

    args = parser.parse_args()

    if args.action == "test":
        res = test_drive_connection(drive_base=args.drive_base, env=args.env)
        if res["status"] != "pass":
            sys.exit(1)
        sys.exit(0)

    try:
        stats = sync_outputs(
            action=args.action,
            categories=args.categories,
            local_outputs=Path(args.local_outputs),
            local_workflows=Path(args.local_workflows),
            local_logs=Path(args.local_logs),
            local_metadata=Path(args.local_metadata),
            drive_base=args.drive_base,
            env=args.env,
            patterns=args.patterns,
        )
        if stats["errors"] > 0:
            sys.exit(1)
        print("\n[SUCCESS] Operação concluída com sucesso.")
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()