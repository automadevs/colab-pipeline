#!/usr/bin/env python3
"""Orquestrador único Colab: setup do repositório -> inspeção -> coleta/resolução de inputs -> download sequencial -> publicação.

Todas as perguntas interativas (fila AIR/URL/HF, destino único de checkpoints,
edições remove/move do dataset) acontecem antes de qualquer download; depois o
pipeline roda sem pausa até o preview final, notas da versão e confirmação de
publicação.

Fontes de entrada suportadas na fila: AIR (`urn:air:...`), URL Civitai e
Hugging Face (`hf:org/repo[/arquivo]` ou URL `huggingface.co`). Para HF a
categoria e o `base_model` são sempre perguntados na fase de coleta, pois não há
como inferi-los de um repo genérico.

Uso no Colab (célula única, ver colab_transfer/00_master_pipeline.ipynb):
    !python /content/colab-pipeline/scripts/master_pipeline.py
    !python /content/colab-pipeline/scripts/master_pipeline.py --dataset "owner/nome"

Variáveis de ambiente / Secrets do Colab (obrigatórias):
    CIVITAI_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_DATASET_NAME
Opcional: HF_TOKEN (repos Hugging Face privados/gated).

O dataset alvo é resolvido no início de main(): ``--dataset`` tem prioridade sobre
``KAGGLE_USERNAME``/``KAGGLE_DATASET_NAME``. Sem nenhum dos dois o pipeline aborta
com mensagem explicativa antes de clonar o repositório.

Código de saída: 0 em sucesso (ou cancelamento explícito da publicação), 1 em falha crítica.
"""
from __future__ import annotations

import argparse
import atexit
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
            "No Colab, confira se os Secrets existem, se estão com \"Notebook access\"\n"
            "habilitado e se foram injetados em os.environ (ver colab_transfer/00_master_pipeline.ipynb).\n"
            "Ou passe o dataset explicitamente via --dataset \"owner/nome\"."
        )

    return f"{username}/{dataset_name}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI mínima: permite sobrescrever o dataset alvo.

    Sem ``--dataset`` o alvo vem de ``KAGGLE_USERNAME``/``KAGGLE_DATASET_NAME``
    (variáveis de ambiente ou Secrets do Colab).
    """
    parser = argparse.ArgumentParser(
        description="Orquestrador Colab: Civitai/Hugging Face -> Kaggle Dataset",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help='Dataset Kaggle no formato "owner/nome" (default: KAGGLE_USERNAME/KAGGLE_DATASET_NAME)',
    )
    return parser.parse_args(argv)


REPO_URL = "https://github.com/automadevs/colab-pipeline.git"
REPO_DIR = Path("/content/colab-pipeline")
SCRIPTS_DIR = REPO_DIR / "scripts"
# Resolvido em main() (aceita --dataset ou KAGGLE_USERNAME/KAGGLE_DATASET_NAME).
DATASET: str | None = None
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
    try:
        # importlib.metadata NÃO importa o módulo (sem efeitos colaterais): as
        # constantes de cache do huggingface_hub precisam continuar configuráveis
        # via configure_hf_cache antes do primeiro import real (item 8).
        from importlib import metadata as importlib_metadata

        hf_status = importlib_metadata.version("huggingface_hub")
    except Exception:
        hf_status = "não instalado (será instalado sob demanda se houver input HF)"
    print(f"huggingface_hub: {hf_status}")
    print(f"Staging: {STAGING_DIR}")


def ensure_huggingface_hub() -> None:
    """Instala huggingface_hub sob demanda (mesmo padrão lazy usado no kagglehub)."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[INFO] huggingface_hub não encontrado; instalando (pip -q)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)


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


def main(argv: list[str] | None = None) -> int:
    print("=" * 60)
    print("MASTER PIPELINE: CIVITAI/HUGGING FACE -> KAGGLE DATASET")
    print("=" * 60)

    args = parse_args(argv)
    try:
        dataset = args.dataset or resolve_dataset_name()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    print(f"[INFO] Dataset alvo: {dataset}")

    print("\n[1/5] SETUP DO REPOSITÓRIO")
    try:
        setup_repo()
    except Exception as exc:
        print(f"[ERROR] Falha ao preparar o repositório: {exc}")
        return 1

    from kaggle_dataset_manager import (
        classify_resolved_artifacts,
        cleanup_hf_cache,
        collect_dataset_edits,
        collect_input_queue,
        configure_hf_cache,
        download_resolved_queue,
        get_secret,
        is_hf_input,
        print_download_failure_summary,
        print_resolution_summary,
        publish_staged_state,
        queue_contains_checkpoint,
        resolve_queue_metadata,
        write_manifest,
    )

    print("\n[2/5] INSPEÇÃO DO AMBIENTE")
    inspect_environment()
    token = get_secret("CIVITAI_TOKEN") or get_secret("CIVITAI_API_KEY")
    if not token:
        print("[ERROR] CIVITAI_TOKEN não encontrado (Secrets do Colab ou variável de ambiente).")
        return 1
    print("CIVITAI_TOKEN: configurado")
    hf_token = get_secret("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN: configurado")
    else:
        print("[WARN] HF_TOKEN não configurado; inputs Hugging Face privados/gated falharão (repos públicos funcionam).")
    if not ensure_kaggle_auth(get_secret):
        print("[ERROR] KAGGLE_USERNAME/KAGGLE_KEY não configurados (Secrets do Colab ou env).")
        return 1
    print("Kaggle auth: configurado")

    print("\n[3/5] COLETA E RESOLUÇÃO DE INPUTS")
    pending = collect_input_queue()
    if any(is_hf_input(value) for value in pending):
        # Cache HF em diretório temporário, configurado ANTES do primeiro import
        # de huggingface_hub (constantes congeladas na importação); removido ao
        # sair do processo, qualquer que seja o desfecho (item 8).
        hf_cache_dir = configure_hf_cache()
        atexit.register(cleanup_hf_cache, hf_cache_dir)
        try:
            ensure_huggingface_hub()
        except Exception as exc:
            print(f"[WARN] Não foi possível instalar huggingface_hub automaticamente: {exc}")
    outcome = resolve_queue_metadata(pending, token, hf_token=hf_token)

    if not outcome.artifacts:
        print_resolution_summary(outcome)
        print("[ERROR] Nenhum item válido resolvido; nada a publicar.")
        return 1
    if outcome.failures:
        # Item 10: com QUALQUER falha, mostra o relatório e finaliza com erro
        # ANTES de tocar no dataset (sem perguntas de edição, sem downloads).
        print_resolution_summary(outcome)
        print(
            f"[ERROR] {len(outcome.failures)} input(s) falharam na resolução; "
            "abortando SEM modificar o dataset e SEM iniciar downloads."
        )
        return 1

    checkpoint_destination = None
    if queue_contains_checkpoint(outcome.artifacts):
        choice = input("Checkpoint: 1=checkpoints/ 2=diffusion_models/: ").strip().lower()
        checkpoint_destination = {"1": "checkpoints", "2": "diffusion_models"}.get(choice, choice)
        print(f"[INFO] Destino de checkpoints do lote: {checkpoint_destination}")

    # Completa a classificação pendente (Civitai checkpoint/desconhecido) e
    # então exibe o INPUT RESOLUTION SUMMARY ANTES de qualquer pergunta de
    # edição do dataset (itens 11/17).
    classify_resolved_artifacts(outcome.artifacts, checkpoint_destination=checkpoint_destination)
    print_resolution_summary(outcome)

    try:
        pending_edits = collect_dataset_edits(dataset)
    except Exception as exc:
        print(f"[WARN] Não foi possível consultar o dataset remoto para edições: {exc}")
        pending_edits = []

    print("\n[4/5] DOWNLOAD SEQUENCIAL (Civitai/Hugging Face)")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    download_outcome = download_resolved_queue(
        outcome.artifacts,
        STAGING_DIR,
        token,
        checkpoint_destination=checkpoint_destination,
        hf_token=hf_token,
    )
    if download_outcome.failures:
        # Publicação transacional: qualquer falha de download impede publicar o lote.
        print_download_failure_summary(download_outcome)
        return 1
    items = download_outcome.items
    if not items:
        print("[ERROR] Nenhum arquivo novo no staging; nada a publicar.")
        return 1
    write_manifest(STAGING_DIR / "dataset-manifest.json", dataset, {item.path: item for item in items})
    print(f"[INFO] {len(items)} arquivo(s) prontos no staging: {STAGING_DIR}")

    print("\n[5/5] PUBLICAÇÃO NO KAGGLE")
    try:
        result = publish_staged_state(dataset, STAGING_DIR, pending_edits=pending_edits)
    except Exception as exc:
        print(f"[WARN] Publicação via Kaggle CLI falhou: {exc}")
        print("[INFO] Tentando fallback via kagglehub...")
        try:
            publish_via_kagglehub(dataset, STAGING_DIR, "Update Dataset state (kagglehub fallback)")
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
