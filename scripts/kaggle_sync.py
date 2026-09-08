#!/usr/bin/env python3
"""Sincronização seletiva Kaggle Dataset -> SSD local do Kaggle Notebook.

A versão anterior baixava o Dataset inteiro e só depois filtrava os arquivos.
Esta versão lista os arquivos primeiro e usa `kaggle datasets download -f` para
baixar somente os arquivos selecionados.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_DATASET = "automamermaid/comfydocs"
DEFAULT_TARGET_DIR = Path("/kaggle/working/ComfyUI/models")
MODEL_CATEGORIES = [
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
]


def parse_model_selection(selection: str, candidate_paths: list[str]) -> list[str]:
    """Converte ``1``, ``1,3-5`` ou ``all`` em paths exatos do Dataset."""
    raw = str(selection or "").strip().lower()
    if not raw:
        raise ValueError("Entrada vazia. Informe números, ranges ou 'all'.")
    if not candidate_paths:
        raise ValueError("Nenhum candidato disponível para seleção.")
    if raw == "all":
        return list(candidate_paths)

    indexes: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("Sintaxe inválida: use números separados por vírgula.")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"Range inválido: {token}")
            start, end = (int(part.strip()) for part in parts)
            if start > end:
                raise ValueError(f"Range invertido: {token}")
            indexes.extend(range(start, end + 1))
        elif token.isdigit():
            indexes.append(int(token))
        else:
            raise ValueError(f"Índice inválido: {token}")

    if len(indexes) != len(set(indexes)):
        raise ValueError("Seleção contém índices duplicados.")
    if any(index < 1 or index > len(candidate_paths) for index in indexes):
        raise ValueError(f"Índice fora do intervalo: escolha entre 1 e {len(candidate_paths)}.")
    return [candidate_paths[index - 1] for index in indexes]


def select_dataset_files(
    dataset: str = DEFAULT_DATASET,
    preselected_categories: list[str] | None = None,
    auto_select_single: bool = True,
    input_fn=input,
) -> list[str]:
    """Lista modelos e aguarda seleção síncrona via stdin/input()."""
    try:
        details = get_dataset_files_details(dataset)
    except Exception as exc:
        print(f"[WARN] Falha ao obter detalhes dos arquivos: {exc}")
        details = []

    if details:
        candidates = [
            item for item in details
            if not preselected_categories or item["category"] in preselected_categories
        ] or details
        entries = [(item["path"], item["name"], item["category"], item["size"]) for item in candidates]
    else:
        raw_files = get_dataset_files(dataset)
        candidates = filter_dataset_files(raw_files, categories=preselected_categories) or raw_files
        entries = [(path, Path(path).name, _category_for_path(path), "N/A") for path in candidates]

    if not entries:
        raise ValueError("Nenhum arquivo candidato encontrado no Dataset.")

    print(f"\nModelos disponíveis em {dataset}:\n")
    for index, (path, name, category, size) in enumerate(entries, 1):
        print(f"[{index}] {name}")
        print(f"    {path}")
        print(f"    {category}/ | {size}")

    candidate_paths = [entry[0] for entry in entries]
    if len(candidate_paths) == 1 and auto_select_single:
        print("\nApenas um arquivo disponível. Selecionando automaticamente.")
        return candidate_paths

    print("\nSelecione os arquivos pelo número.")
    print("Exemplos: 1 | 1,3,5 | 1-4 | all")
    while True:
        try:
            return parse_model_selection(input_fn("\n> Seleção: "), candidate_paths)
        except ValueError as exc:
            print(f"[WARN] {exc}")


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_kaggle_files_output(stdout: str) -> list[str]:
    """Extrai nomes da saída de `kaggle datasets files` sem confundir cabeçalho."""
    files = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("-") or line.lower().startswith("name") or line.lower() == "files":
            continue
        # A saída tabular do CLI começa pelo nome/path do arquivo.
        parts = line.split()
        if parts and parts[0] not in {"File", "Name", "files"}:
            files.append(parts[0])
    return files


def _parse_kaggle_files_detailed(stdout: str) -> list[dict]:
    """Extrai informações detalhadas (path, size, category) da saída do CLI."""
    items = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("-") or line.lower().startswith("name") or line.lower() == "files":
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in {"File", "Name", "files"}:
            path = parts[0]
            size = parts[1]
            cat = _category_for_path(path)
            items.append({
                "path": path,
                "name": Path(path).name,
                "category": cat,
                "size": size,
            })
        elif len(parts) == 1 and parts[0] not in {"File", "Name", "files"}:
            path = parts[0]
            cat = _category_for_path(path)
            items.append({
                "path": path,
                "name": Path(path).name,
                "category": cat,
                "size": "N/A",
            })
    return items


def get_dataset_files(dataset: str = DEFAULT_DATASET) -> list[str]:
    result = subprocess.run(
        ["kaggle", "datasets", "files", dataset],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao listar dataset: {result.stderr.strip()}")
    return _parse_kaggle_files_output(result.stdout)


def get_dataset_files_details(dataset: str = DEFAULT_DATASET) -> list[dict]:
    """Retorna lista de dicionários com path, name, category e size formatado."""
    result = subprocess.run(
        ["kaggle", "datasets", "files", dataset],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao listar dataset: {result.stderr.strip()}")
    return _parse_kaggle_files_detailed(result.stdout)


def _category_for_path(rel_path: str) -> str:
    normalized = rel_path.replace("\\", "/").lstrip("./")
    for cat in MODEL_CATEGORIES:
        if normalized.startswith(cat + "/"):
            return cat
    # Arquivo na raiz continua sendo tratado como checkpoint por compatibilidade.
    return "checkpoints"


def filter_dataset_files(
    dataset_files: list[str],
    categories: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[str]:
    selected = []
    names = set(model_names or [])
    cats = set(categories or [])
    for path in dataset_files:
        category = _category_for_path(path)
        filename = Path(path).name
        if cats and category not in cats:
            continue
        if names and filename not in names and path not in names:
            continue
        selected.append(path)
    return selected


def _safe_extract_path(tmpdir: Path, dataset_file: str) -> Path:
    """Localiza o arquivo extraído sem confiar cegamente no caminho retornado."""
    expected = tmpdir / dataset_file
    if expected.is_file():
        return expected
    matches = list(tmpdir.rglob(Path(dataset_file).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Arquivo baixado não localizado: {dataset_file}")


def download_selected_file(dataset: str, dataset_file: str, tmpdir: Path) -> Path:
    """Baixa somente um arquivo do Dataset."""
    result = subprocess.run(
        [
            "kaggle", "datasets", "download", dataset,
            "-f", dataset_file,
            "-p", str(tmpdir),
            "--unzip",
        ],
        capture_output=True, text=True, timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Falha ao baixar '{dataset_file}': {result.stderr.strip() or result.stdout.strip()}"
        )
    return _safe_extract_path(tmpdir, dataset_file)


def sync_dataset_to_local(
    dataset: str = DEFAULT_DATASET,
    target_dir: Path = DEFAULT_TARGET_DIR,
    categories: list[str] | None = None,
    model_names: list[str] | None = None,
    selected_files: list[str] | None = None,
    force: bool = False,
) -> dict:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for cat in MODEL_CATEGORIES:
        (target_dir / cat).mkdir(parents=True, exist_ok=True)

    print("[INFO] === SYNC KAGGLE DATASET -> LOCAL ===")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Target: {target_dir}")

    dataset_files = get_dataset_files(dataset)
    print(f"[INFO] Arquivos disponíveis: {len(dataset_files)}")

    if selected_files is not None:
        wanted = set(selected_files)
        files_to_sync = [f for f in dataset_files if f in wanted]
        missing = wanted - set(files_to_sync)
        if missing:
            raise ValueError(f"Arquivos não encontrados no dataset: {sorted(missing)}")
    else:
        files_to_sync = filter_dataset_files(dataset_files, categories, model_names)

    print(f"[INFO] Arquivos selecionados: {len(files_to_sync)}")
    for f in files_to_sync:
        print(f"  - {f}")

    stats = {"synced": 0, "skipped": 0, "errors": 0, "details": []}
    if not files_to_sync:
        print("[INFO] Nada para sincronizar.")
        return stats

    for dataset_file in files_to_sync:
        filename = Path(dataset_file).name
        category = _category_for_path(dataset_file)
        dest = target_dir / category / filename

        try:
            if dest.exists() and not force and dest.stat().st_size > 0:
                size_gb = dest.stat().st_size / (1024 ** 3)
                print(f"[INFO] Pulando {filename}: já existe localmente ({size_gb:.2f} GB). Use --force para atualizar.")
                stats["skipped"] += 1
                stats["details"].append({"file": filename, "path": dataset_file, "status": "skipped", "size_gb": size_gb, "category": category})
                continue

            with tempfile.TemporaryDirectory(prefix="kaggle_sync_") as tmp:
                src = download_selected_file(dataset, dataset_file, Path(tmp))
                size_gb = src.stat().st_size / (1024 ** 3)
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Baixando {filename} -> {category}/ ({size_gb:.2f} GB)")
                shutil.copy2(src, dest)
                if dest.stat().st_size != src.stat().st_size:
                    raise RuntimeError(f"Tamanho divergiu na cópia de {filename}")
                digest = calculate_sha256(dest)
                print(f"[INFO] OK {filename} (SHA256: {digest[:16]}...)")
                stats["synced"] += 1
                stats["details"].append({"file": filename, "path": dataset_file, "status": "synced", "size_gb": size_gb, "category": category, "sha256": digest})
        except Exception as exc:
            stats["errors"] += 1
            stats["details"].append({"file": filename, "path": dataset_file, "status": "error", "category": category, "error": str(exc)})
            print(f"[ERROR] {dataset_file}: {exc}")
            raise

    print("\n[INFO] === RESUMO ===")
    print(f"  Sincronizados: {stats['synced']}")
    print(f"  Pulados: {stats['skipped']}")
    print(f"  Erros: {stats['errors']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync seletivo Kaggle Dataset -> SSD local")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--categories", nargs="*", choices=MODEL_CATEGORIES)
    parser.add_argument("--model-names", nargs="*", help="Nomes específicos")
    parser.add_argument("--selected-files", nargs="*", help="Paths exatos retornados por 'kaggle datasets files'")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Somente lista arquivos e não baixa")
    args = parser.parse_args()

    try:
        files = get_dataset_files(args.dataset)
        if args.list_only:
            print(json.dumps(files, ensure_ascii=False, indent=2))
            return
        stats = sync_dataset_to_local(
            dataset=args.dataset,
            target_dir=Path(args.target_dir),
            categories=args.categories,
            model_names=args.model_names,
            selected_files=args.selected_files,
            force=args.force,
        )
        print(f"\n[SUCCESS] Sync concluído: {stats['synced']} novos, {stats['skipped']} pulados")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
