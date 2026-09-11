#!/usr/bin/env python3
"""Orquestrador único Colab: setup do repositório -> inspeção -> download sequencial -> publicação.

Uso no Colab (célula única, ver colab_transfer/00_master_pipeline.ipynb):
    !python /content/colab-pipeline/scripts/master_pipeline.py

Código de saída: 0 em sucesso (ou cancelamento explícito da publicação), 1 em falha crítica.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_dataset_name(override: str | None = None) -> str:
    """Resolve o dataset Kaggle a partir de secrets/env ou override explícito.

    Prioridade:
    1. ``override`` se fornecido (CLI/parâmetro)
    2. Montagem dinâmica: f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_NAME}"

    Raises:
        ValueError: Se nem override nem as variáveis de ambiente estiverem configuradas.
    """
    if override:
        return override

    username = os.environ.get("KAGGLE_USERNAME")
    dataset_name = os.environ.get("KAGGLE_DATASET_NAME")

    if not username or not dataset_name:
        raise ValueError(
            "Dataset Kaggle não resolvido. Configure os secrets/env:\n"
            "  - KAGGLE_USERNAME: seu username Kaggle\n"
            "  - KAGGLE_DATASET_NAME: nome do dataset (ex: comfydocs)\n"
            "Ou passe o dataset explicitamente via --dataset \"owner/nome\"."
        )

    return f"{username}/{dataset_name}"


REPO_URL = "https://github.com/automadevs/colab-pipeline.git"
REPO_DIR = Path("/content/colab-pipeline")
SCRIPTS_DIR = REPO_DIR / "scripts"
DATASET = resolve_dataset_name()
STAGING_DIR = Path("/content/kaggle_staging")


def setup_repo() -> None:
    """Clona na primeira execução; nas seguintes apenas fast-forward."""
    if REPO_DIR.exists():
        print(f"[INFO] Repositório encontrado em {REPO_DIR}; atualizando (git pull --ff-only)...")
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        print(f"[INFO] Clonando {REPO_URL} -> {REPO_DIR} (depth 1)...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def ensure_kaggle_auth(get_secret) -> bool:
    """Cria ~/.kaggle/kaggle.json a partir dos Secrets do Colab ou de env, se necessário."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    username = get_secret("KAGGLE_USERNAME")
    key = get_secret("KAGGLE_KEY")
    if not username or not key:
        return False
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    os.chmod(kaggle_json, 0o600)
    return True


def inspect_environment() -> None:
    print(f"Python: {sys.version.split()[0]}")
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    kaggle_version = (result.stdout or result.stderr).strip()
    print(f"Kaggle CLI: {kaggle_version if result.returncode == 0 and kaggle_version else 'NÃO DISPONÍVEL'}")
    civitai_cli = shutil.which("civitai")
    print(f"Civitai CLI: {civitai_cli or 'não instalado (será instalado sob demanda)'}")
    print(f"Staging: {STAGING_DIR}")


def publish_via_kagglehub(dataset: str, staging_dir: Path, notes: str) -> None:
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
    if not hasattr(kagglehub, "dataset_upload"):
        raise RuntimeError("kagglehub.dataset_upload não disponível nesta versão")
    kagglehub.dataset_upload(handle=dataset, local_dataset_dir=str(staging_dir), version_notes=notes)
    print(f"[INFO] Fallback kagglehub concluído: {dataset}")


def main() -> int:
    print("=" * 60)
    print("MASTER PIPELINE: CIVITAI -> KAGGLE DATASET")
    print("=" * 60)

    print("\n[1/4] SETUP DO REPOSITÓRIO")
    try:
        setup_repo()
    except Exception as exc:
        print(f"[ERROR] Falha ao preparar o repositório: {exc}")
        return 1

    from kaggle_dataset_manager import (
        download_input_queue,
        get_secret,
        publish_staged_state,
        write_manifest,
    )

    print("\n[2/4] INSPEÇÃO DO AMBIENTE")
    inspect_environment()
    token = get_secret("CIVITAI_TOKEN") or get_secret("CIVITAI_API_KEY")
    if not token:
        print("[ERROR] CIVITAI_TOKEN não encontrado (Secrets do Colab ou variável de ambiente).")
        return 1
    print("CIVITAI_TOKEN: configurado")
    if not ensure_kaggle_auth(get_secret):
        print("[ERROR] KAGGLE_USERNAME/KAGGLE_KEY não configurados (Secrets do Colab ou env).")
        return 1
    print("Kaggle auth: configurado")

    print("\n[3/4] DOWNLOAD SEQUENCIAL (Civitai)")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    items = download_input_queue(STAGING_DIR, token)
    if not items:
        print("[ERROR] Nenhum arquivo novo no staging; nada a publicar.")
        return 1
    write_manifest(STAGING_DIR / "dataset-manifest.json", DATASET, {item.path: item for item in items})
    print(f"[INFO] {len(items)} arquivo(s) prontos no staging: {STAGING_DIR}")

    print("\n[4/4] PUBLICAÇÃO NO KAGGLE")
    try:
        result = publish_staged_state(DATASET, STAGING_DIR)
    except Exception as exc:
        print(f"[WARN] Publicação via Kaggle CLI falhou: {exc}")
        print("[INFO] Tentando fallback via kagglehub...")
        try:
            publish_via_kagglehub(DATASET, STAGING_DIR, "Update Dataset state (kagglehub fallback)")
        except Exception as fallback_exc:
            print(f"[ERROR] Fallback kagglehub também falhou: {fallback_exc}")
            return 1
    else:
        if result is None:
            print("[INFO] Publicação cancelada pelo usuário; dataset remoto inalterado.")

    print("\n[SUCCESS] Pipeline concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
