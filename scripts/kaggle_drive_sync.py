#!/usr/bin/env python3
"""
Sync de outputs e arquivos leves para Google Drive.
Suporta Colab (mount nativo) e Kaggle (rclone + service account).
Uso: python kaggle_drive_sync.py [--action push|pull] [--local-dir DIR] [--drive-path PATH] [--env colab|kaggle]
"""

import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional

DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
DEFAULT_LOCAL_OUTPUTS = Path("/kaggle/working/ComfyUI/output")
DEFAULT_LOCAL_WORKFLOWS = Path("/kaggle/working/ComfyUI/user")


def detect_env() -> str:
    """Detecta se está rodando no Colab ou Kaggle."""
    if os.path.exists("/content"):
        return "colab"
    elif os.path.exists("/kaggle"):
        return "kaggle"
    return "unknown"


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


def setup_rclone_kaggle() -> bool:
    """Configura rclone no Kaggle usando service account do Secret."""
    rclone_conf = Path("/root/.config/rclone/rclone.conf")
    
    if rclone_conf.exists():
        print("[INFO] rclone já configurado")
        return True
    
    # Tentar obter service account do Secret do Kaggle
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
    
    # Salvar service account
    sa_path = Path("/root/gdrive_sa.json")
    sa_path.write_text(sa_json)
    
    # Configurar rclone
    rclone_conf.parent.mkdir(parents=True, exist_ok=True)
    config_content = f"""[gdrive]
type = drive
scope = drive
service_account_file = {sa_path}
"""
    rclone_conf.write_text(config_content)
    print("[INFO] rclone configurado com service account")
    return True


def get_drive_path_kaggle(drive_base: str = DEFAULT_DRIVE_BASE) -> Path:
    """Retorna path montado via rclone no Kaggle."""
    mount_point = Path("/kaggle/working/gdrive")
    mount_point.mkdir(parents=True, exist_ok=True)
    
    # Verificar se já montado
    if not list(mount_point.iterdir()):
        print("[INFO] Montando Google Drive via rclone...")
        result = subprocess.run([
            "rclone", "mount", "gdrive:",
            str(mount_point),
            "--vfs-cache-mode", "full",
            "--daemon"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            # Tentar sem daemon para ver erro
            result = subprocess.run([
                "rclone", "lsd", "gdrive:"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Falha ao acessar Drive via rclone: {result.stderr}")
    
    drive_path = mount_point / drive_base
    drive_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Google Drive acessível em: {drive_path}")
    return drive_path


def get_drive_path(drive_base: str = DEFAULT_DRIVE_BASE, env: str = None) -> Path:
    """Obtém path do Google Drive baseado no ambiente."""
    if env is None:
        env = detect_env()
    
    if env == "colab":
        return mount_drive_colab(drive_base)
    elif env == "kaggle":
        if not setup_rclone_kaggle():
            raise RuntimeError("Falha ao configurar rclone no Kaggle")
        return get_drive_path_kaggle(drive_base)
    else:
        raise RuntimeError(f"Ambiente não suportado: {env}")


def sync_push(local_dir: Path, drive_dir: Path, patterns: list = None) -> dict:
    """Envia arquivos locais para Drive (push)."""
    if not local_dir.exists():
        print(f"[WARN] Diretório local não existe: {local_dir}")
        return {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    
    drive_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    
    # Coletar arquivos locais
    local_files = []
    for f in local_dir.rglob("*"):
        if f.is_file():
            if patterns:
                matched = any(f.match(p) for p in patterns)
                if not matched:
                    continue
            local_files.append(f)
    
    print(f"[INFO] Enviando {len(local_files)} arquivos de {local_dir} → {drive_dir}")
    
    for src in local_files:
        rel = src.relative_to(local_dir)
        dest = drive_dir / rel
        
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Verificar se já existe e é idêntico (mtime + size)
            if dest.exists():
                src_stat = src.stat()
                dest_stat = dest.stat()
                if src_stat.st_size == dest_stat.st_size and abs(src_stat.st_mtime - dest_stat.st_mtime) < 2:
                    stats["skipped"] += 1
                    stats["details"].append({"file": str(rel), "status": "skipped", "size_mb": src_stat.st_size / (1024**2)})
                    continue
            
            shutil.copy2(src, dest)
            stats["synced"] += 1
            stats["details"].append({"file": str(rel), "status": "synced", "size_mb": src.stat().st_size / (1024**2)})
            
        except Exception as e:
            stats["errors"] += 1
            stats["details"].append({"file": str(rel), "status": "error", "error": str(e)})
            print(f"[ERROR] Falha ao copiar {rel}: {e}")
    
    return stats


def sync_pull(drive_dir: Path, local_dir: Path, patterns: list = None) -> dict:
    """Baixa arquivos do Drive para local (pull)."""
    if not drive_dir.exists():
        print(f"[WARN] Diretório no Drive não existe: {drive_dir}")
        return {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    
    drive_files = []
    for f in drive_dir.rglob("*"):
        if f.is_file():
            if patterns:
                matched = any(f.match(p) for p in patterns)
                if not matched:
                    continue
            drive_files.append(f)
    
    print(f"[INFO] Baixando {len(drive_files)} arquivos de {drive_dir} → {local_dir}")
    
    for src in drive_files:
        rel = src.relative_to(drive_dir)
        dest = local_dir / rel
        
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if dest.exists():
                src_stat = src.stat()
                dest_stat = dest.stat()
                if src_stat.st_size == dest_stat.st_size and abs(src_stat.st_mtime - dest_stat.st_mtime) < 2:
                    stats["skipped"] += 1
                    stats["details"].append({"file": str(rel), "status": "skipped", "size_mb": src_stat.st_size / (1024**2)})
                    continue
            
            shutil.copy2(src, dest)
            stats["synced"] += 1
            stats["details"].append({"file": str(rel), "status": "synced", "size_mb": src.stat().st_size / (1024**2)})
            
        except Exception as e:
            stats["errors"] += 1
            stats["details"].append({"file": str(rel), "status": "error", "error": str(e)})
            print(f"[ERROR] Falha ao copiar {rel}: {e}")
    
    return stats


def sync_outputs(
    action: str = "push",
    local_outputs: Path = DEFAULT_LOCAL_OUTPUTS,
    local_workflows: Path = DEFAULT_LOCAL_WORKFLOWS,
    drive_base: str = DEFAULT_DRIVE_BASE,
    env: str = None,
    patterns: list = None,
) -> dict:
    """
    Sync principal de outputs/workflows.
    action: "push" (local → Drive) ou "pull" (Drive → local)
    """
    print(f"[INFO] === DRIVE SYNC ({action.upper()}) ===")
    print(f"[INFO] Ambiente: {env or detect_env()}")
    print(f"[INFO] Drive base: {drive_base}")
    
    drive_path = get_drive_path(drive_base, env)
    
    # Subdiretórios no Drive
    drive_outputs = drive_path / "outputs"
    drive_workflows = drive_path / "workflows"
    drive_logs = drive_path / "logs"
    drive_metadata = drive_path / "metadata"
    
    all_stats = {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    
    if action == "push":
        # Outputs (imagens geradas)
        if local_outputs.exists():
            stats = sync_push(local_outputs, drive_outputs, patterns)
            print(f"  outputs: {stats['synced']} novos, {stats['skipped']} pulados, {stats['errors']} erros")
            all_stats["synced"] += stats["synced"]
            all_stats["skipped"] += stats["skipped"]
            all_stats["errors"] += stats["errors"]
            all_stats["details"].extend(stats["details"])
        
        # Workflows salvos pelo usuário
        if local_workflows.exists():
            stats = sync_push(local_workflows, drive_workflows, patterns)
            print(f"  workflows: {stats['synced']} novos, {stats['skipped']} pulados, {stats['errors']} erros")
            all_stats["synced"] += stats["synced"]
            all_stats["skipped"] += stats["skipped"]
            all_stats["errors"] += stats["errors"]
            all_stats["details"].extend(stats["details"])
        
        # Logs do ComfyUI (se existirem)
        comfy_logs = Path("/kaggle/working/ComfyUI/logs")
        if comfy_logs.exists():
            stats = sync_push(comfy_logs, drive_logs, patterns)
            print(f"  logs: {stats['synced']} novos, {stats['skipped']} pulados, {stats['errors']} erros")
            all_stats["synced"] += stats["synced"]
            all_stats["skipped"] += stats["skipped"]
            all_stats["errors"] += stats["errors"]
            all_stats["details"].extend(stats["details"])
            
    elif action == "pull":
        # Pull workflows (compartilhados/backup)
        stats = sync_pull(drive_workflows, local_workflows, patterns)
        print(f"  workflows: {stats['synced']} novos, {stats['skipped']} pulados, {stats['errors']} erros")
        all_stats["synced"] += stats["synced"]
        all_stats["skipped"] += stats["skipped"]
        all_stats["errors"] += stats["errors"]
        all_stats["details"].extend(stats["details"])
    else:
        raise ValueError(f"Ação inválida: {action}. Use 'push' ou 'pull'")
    
    print(f"\n[INFO] === RESUMO DRIVE SYNC ===")
    print(f"  Enviados/Baixados: {all_stats['synced']}")
    print(f"  Pulados: {all_stats['skipped']}")
    print(f"  Erros: {all_stats['errors']}")
    
    return all_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync Google Drive para outputs ComfyUI")
    parser.add_argument("--action", choices=["push", "pull"], default="push", help="Direção do sync")
    parser.add_argument("--local-outputs", default=str(DEFAULT_LOCAL_OUTPUTS), help="Diretório local de outputs")
    parser.add_argument("--local-workflows", default=str(DEFAULT_LOCAL_WORKFLOWS), help="Diretório local de workflows")
    parser.add_argument("--drive-base", default=DEFAULT_DRIVE_BASE, help="Pasta base no Google Drive")
    parser.add_argument("--env", choices=["colab", "kaggle"], help="Forçar ambiente (auto-detect por padrão)")
    parser.add_argument("--patterns", nargs="*", help="Padrões de arquivo (ex: *.png *.json)")
    
    args = parser.parse_args()
    
    try:
        stats = sync_outputs(
            action=args.action,
            local_outputs=Path(args.local_outputs),
            local_workflows=Path(args.local_workflows),
            drive_base=args.drive_base,
            env=args.env,
            patterns=args.patterns,
        )
        if stats["errors"] > 0:
            exit(1)
        print("\n[SUCCESS] Drive sync concluído")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()