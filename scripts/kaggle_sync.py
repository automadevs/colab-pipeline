#!/usr/bin/env python3
"""
Sincronização Kaggle Dataset → SSD local do Kaggle Notebook.
Uso: python kaggle_sync.py [--dataset DATASET] [--target-dir DIR] [--categories CAT1 CAT2 ...] [--model-name NAME]
"""

import os
import subprocess
import shutil
import hashlib
import tempfile
import argparse
from pathlib import Path

DEFAULT_DATASET = "automamermaid/comfydocs"
DEFAULT_TARGET_DIR = Path("/kaggle/working/ComfyUI/models")

# Categorias de modelos suportadas
MODEL_CATEGORIES = [
    "checkpoints",
    "diffusion_models",
    "loras",
    "vae",
    "text_encoders",
    "clip",
    "controlnet",
    "upscale_models",
    "video_models",
    "embeddings",
]


def calculate_sha256(filepath: Path) -> str:
    """Calcula SHA256 do arquivo."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_dataset_files(dataset: str) -> list:
    """Lista arquivos no dataset via Kaggle CLI."""
    result = subprocess.run(
        ["kaggle", "datasets", "files", dataset],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao listar dataset: {result.stderr}")

    # Parse output - formato típico: name, size, date
    files = []
    for line in result.stdout.strip().split('\n')[1:]:  # Skip header
        parts = line.split()
        if parts:
            files.append(parts[0])
    return files


def sync_dataset_to_local(
    dataset: str = DEFAULT_DATASET,
    target_dir: Path = DEFAULT_TARGET_DIR,
    categories: list = None,
    model_names: list = None,
    force: bool = False,
) -> dict:
    """
    Sincroniza modelos do Kaggle Dataset para SSD local.
    Retorna dict com stats: {synced, skipped, errors, details}
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Criar subdiretórios para cada categoria
    for cat in MODEL_CATEGORIES:
        (target_dir / cat).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] === SYNC KAGGLE DATASET → LOCAL ===")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Target: {target_dir}")
    if categories:
        print(f"[INFO] Categorias: {categories}")
    if model_names:
        print(f"[INFO] Modelos específicos: {model_names}")

    # Listar arquivos no dataset
    print("[INFO] Listando arquivos no dataset...")
    try:
        dataset_files = get_dataset_files(dataset)
        print(f"[INFO] Arquivos no dataset: {len(dataset_files)}")
        for f in dataset_files:
            print(f"  - {f}")
    except Exception as e:
        print(f"[WARN] Não foi possível listar: {e}")
        dataset_files = []

    stats = {"synced": 0, "skipped": 0, "errors": 0, "details": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"[INFO] Baixando dataset para {tmpdir}...")

        result = subprocess.run(
            ["kaggle", "datasets", "download", dataset, "-p", str(tmpdir), "--unzip"],
            capture_output=True, text=True, timeout=7200
        )

        if result.returncode != 0:
            raise RuntimeError(f"Falha ao baixar dataset: {result.stderr}")

        # Encontrar todos os arquivos baixados
        downloaded_files = list(tmpdir.rglob("*"))
        downloaded_files = [f for f in downloaded_files if f.is_file()]
        print(f"[INFO] Arquivos baixados: {len(downloaded_files)}")

        for src in downloaded_files:
            rel_path = src.relative_to(tmpdir)
            filename = src.name

            # Filtrar por categorias se especificado
            if categories:
                # Verificar se o arquivo está em uma das categorias
                # Estrutura esperada: categoria/arquivo ou arquivo na raiz
                in_category = False
                for cat in categories:
                    if str(rel_path).startswith(cat + "/") or (cat == "checkpoints" and "/" not in str(rel_path)):
                        in_category = True
                        break
                if not in_category:
                    continue

            # Filtrar por nomes específicos se especificado
            if model_names and filename not in model_names:
                continue

            # Determinar categoria de destino
            dest_category = "checkpoints"  # default
            for cat in MODEL_CATEGORIES:
                if str(rel_path).startswith(cat + "/"):
                    dest_category = cat
                    break

            dest = target_dir / dest_category / filename

            # Verificar se já existe e é válido
            if dest.exists() and not force:
                src_size = src.stat().st_size
                dest_size = dest.stat().st_size

                if src_size == dest_size and src_size > 0:
                    # Verificar hash para garantir integridade
                    src_hash = calculate_sha256(src)
                    dest_hash = calculate_sha256(dest)

                    if src_hash == dest_hash:
                        print(f"[INFO] ⏭️  Pulando {filename} (idêntico, {src_size/(1024**3):.2f} GB)")
                        stats["skipped"] += 1
                        stats["details"].append({"file": filename, "status": "skipped", "size_gb": src_size/(1024**3)})
                        continue
                    else:
                        print(f"[WARN] Hash diverge para {filename}, re-sincronizando")
                else:
                    print(f"[WARN] Tamanho diverge para {filename} (src={src_size}, dest={dest_size}), re-sincronizando")

            # Copiar arquivo
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 📥 Sincronizando {filename} → {dest_category}/ ({src.stat().st_size/(1024**3):.2f} GB)")
            shutil.copy2(src, dest)

            # Verificar cópia
            if dest.stat().st_size != src.stat().st_size:
                raise RuntimeError(f"Tamanho divergiu na cópia de {filename}")

            print(f"[INFO] ✅ {filename} sincronizado (SHA256: {calculate_sha256(dest)[:16]}...)")
            stats["synced"] += 1
            stats["details"].append({"file": filename, "status": "synced", "size_gb": src.stat().st_size/(1024**3), "category": dest_category})

    print(f"\n[INFO] === RESUMO ===")
    print(f"  Sincronizados: {stats['synced']}")
    print(f"  Pulados: {stats['skipped']}")
    print(f"  Erros: {stats['errors']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync Kaggle Dataset para SSD local")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--categories", nargs="*", choices=MODEL_CATEGORIES, help="Categorias para sincronizar")
    parser.add_argument("--model-names", nargs="*", help="Nomes específicos de modelos para sincronizar")
    parser.add_argument("--force", action="store_true", help="Forçar re-sync mesmo se arquivo existe")

    args = parser.parse_args()

    try:
        stats = sync_dataset_to_local(
            dataset=args.dataset,
            target_dir=Path(args.target_dir),
            categories=args.categories,
            model_names=args.model_names,
            force=args.force,
        )
        print(f"\n[SUCCESS] Sync concluído: {stats['synced']} novos, {stats['skipped']} pulados")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()