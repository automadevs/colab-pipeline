#!/usr/bin/env python3
"""Independent Colab tool for building and publishing complete Kaggle Dataset state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

DEFAULT_DATASET = "automamermaid/comfydocs"
DEFAULT_STAGING = Path("/content/kaggle_dataset_manager")
CATEGORIES = (
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
)
MANIFEST_NAME = "dataset-manifest.json"
METADATA_NAME = "dataset-metadata.json"


@dataclass(frozen=True)
class DatasetFile:
    path: str
    size: int
    sha256: Optional[str] = None
    source: Optional[str] = None
    civitai_model_id: Optional[str] = None
    civitai_version_id: Optional[str] = None
    civitai_file_id: Optional[str] = None


@dataclass(frozen=True)
class ChangeSet:
    added: tuple[DatasetFile, ...] = ()
    removed: tuple[DatasetFile, ...] = ()
    moved: tuple[tuple[str, str], ...] = ()
    unchanged: tuple[DatasetFile, ...] = ()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def build_dataset_path(category: str, filename: str) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"Categoria inválida: {category}")
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("Nome de arquivo inválido")
    return f"{category}/{name}"


def parse_current_files(details: Iterable[dict[str, Any]]) -> dict[str, DatasetFile]:
    current: dict[str, DatasetFile] = {}
    for item in details:
        path = str(item.get("path") or item.get("name") or "").replace("\\", "/").lstrip("./")
        if not path:
            continue
        raw_size = item.get("size", 0)
        size = parse_size(raw_size)
        current[path] = DatasetFile(path=path, size=size)
    return current


def parse_size(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip().replace(",", "")
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(B|KB|KiB|MB|MiB|GB|GiB|TB|TiB)?$", text, re.I)
    if not match:
        return 0
    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    powers = {"b": 0, "kb": 1, "kib": 1, "mb": 2, "mib": 2, "gb": 3, "gib": 3, "tb": 4, "tib": 4}
    return int(number * (1024 ** powers[unit]))


def compare_states(current: dict[str, DatasetFile], desired: dict[str, DatasetFile]) -> ChangeSet:
    added = tuple(desired[path] for path in sorted(set(desired) - set(current)))
    removed = tuple(current[path] for path in sorted(set(current) - set(desired)))
    unchanged: list[DatasetFile] = []
    changed: list[DatasetFile] = []
    for path in sorted(set(current) & set(desired)):
        old, new = current[path], desired[path]
        if old.size == new.size and (not old.sha256 or not new.sha256 or old.sha256 == new.sha256):
            unchanged.append(new)
        else:
            changed.append(new)
    added = tuple((*added, *changed))

    moved: list[tuple[str, str]] = []
    remaining_added = list(added)
    remaining_removed = list(removed)
    for old in list(remaining_removed):
        match = next((new for new in remaining_added if old.size and old.size == new.size and old.sha256 and old.sha256 == new.sha256), None)
        if match:
            moved.append((old.path, match.path))
            remaining_removed.remove(old)
            remaining_added.remove(match)
    return ChangeSet(tuple(remaining_added), tuple(remaining_removed), tuple(moved), tuple(unchanged))


def manifest_payload(dataset: str, desired: dict[str, DatasetFile]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "files": [asdict(desired[path]) for path in sorted(desired)],
    }


def write_manifest(path: Path, dataset: str, desired: dict[str, DatasetFile]) -> None:
    Path(path).write_text(json.dumps(manifest_payload(dataset, desired), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, DatasetFile]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["path"]: DatasetFile(**item) for item in data.get("files", [])}


def render_preview(dataset: str, current: dict[str, DatasetFile], desired: dict[str, DatasetFile], changes: ChangeSet) -> str:
    lines = ["DATASET CURRENT", f"Dataset: {dataset}", f"Files: {len(current)}", f"Total: {format_size(sum(item.size for item in current.values()))}", "", "DESIRED", f"Files: {len(desired)}", f"Total: {format_size(sum(item.size for item in desired.values()))}", "", "CHANGES"]
    for title, entries, prefix in (("ADD", changes.added, "+"), ("REMOVE", changes.removed, "-"), ("UNCHANGED", changes.unchanged, "=")):
        lines.append(f"{title} ({len(entries)})")
        lines.extend(f"{prefix} {item.path} ({format_size(item.size)})" for item in entries)
    lines.append(f"MOVE ({len(changes.moved)})")
    lines.extend(f"~ {old} -> {new}" for old, new in changes.moved)
    return "\n".join(lines)


def resolve_civitai_url(url: str, token: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    match = re.search(r"/models/(\d+)", parsed.path)
    if not match:
        raise ValueError("URL Civitai inválida: esperado https://civitai.com/models/<id>")
    model_id = match.group(1)
    query = urllib.parse.parse_qs(parsed.query)
    requested_version = (query.get("modelVersionId") or [None])[0]
    request = urllib.request.Request(
        f"https://civitai.com/api/v1/models/{model_id}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "colab-pipeline-dataset-manager"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        model = json.load(response)
    versions = model.get("modelVersions") or []
    version = next((item for item in versions if str(item.get("id")) == str(requested_version)), versions[0] if versions else None)
    if not version:
        raise RuntimeError("Nenhuma versão Civitai encontrada")
    files = version.get("files") or []
    if not files:
        raise RuntimeError("A versão Civitai não possui arquivos")
    primary = next((item for item in files if item.get("primary")), files[0])
    return {"model": model, "version": version, "file": primary, "model_id": model_id}


def download_civitai_file(
    info: dict[str, Any],
    category: str,
    staging_dir: Path,
    token: str,
    source_url: Optional[str] = None,
) -> DatasetFile:
    file_info = info["file"]
    filename = Path(file_info.get("name") or "model.safetensors").name
    dataset_path = build_dataset_path(category, filename)
    destination = Path(staging_dir) / dataset_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int((file_info.get("sizeKB") or 0) * 1024)
    if destination.exists() and destination.stat().st_size > 0 and (not expected_size or destination.stat().st_size == expected_size):
        print(f"[SKIP] já presente no staging e com tamanho válido: {dataset_path}")
    else:
        url = file_info.get("downloadUrl") or f"https://civitai.com/api/download/models/{info['version']['id']}?fileId={file_info['id']}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "colab-pipeline-dataset-manager"})
        print(f"Download: {filename} -> {dataset_path}")
        with urllib.request.urlopen(request, timeout=7200) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    actual_size = destination.stat().st_size
    if actual_size == 0 or (expected_size and actual_size != expected_size):
        raise RuntimeError(f"Tamanho inválido para {dataset_path}: {actual_size} bytes; esperado {expected_size}")
    return DatasetFile(dataset_path, actual_size, sha256_file(destination), source_url, str(info["model_id"]), str(info["version"].get("id")), str(file_info.get("id")))


def kaggle_files(dataset: str) -> list[dict[str, Any]]:
    result = subprocess.run(["kaggle", "datasets", "files", dataset], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Falha ao consultar arquivos do Dataset")
    items = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() not in {"name", "file", "files"} and not parts[0].startswith("-"):
            size = " ".join(parts[1:3]) if len(parts) >= 3 else parts[1]
            items.append({"path": parts[0], "name": Path(parts[0]).name, "size": size})
    return items


def materialize_dataset_file(dataset: str, dataset_file: str, staging_dir: Path, cache_dir: Path) -> DatasetFile:
    destination = Path(staging_dir) / dataset_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Preservando arquivo existente via download seletivo: {dataset_file}")
        result = subprocess.run(["kaggle", "datasets", "download", dataset, "-f", dataset_file, "-p", str(cache_dir), "--unzip"], capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Falha ao preservar {dataset_file}")
        matches = list(cache_dir.rglob(Path(dataset_file).name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Arquivo preservado não localizado: {dataset_file}")
        shutil.copy2(matches[0], destination)
    return DatasetFile(dataset_file, destination.stat().st_size, sha256_file(destination))


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def fetch_dataset_metadata(dataset: str, destination: Path) -> dict[str, Any]:
    """Obtém metadata atual sem baixar o conteúdo dos arquivos."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "datasets", "metadata", dataset, "-p", str(destination)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    metadata_path = destination / METADATA_NAME
    if result.returncode == 0 and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"title": Path(dataset).name, "licenses": [{"name": "CC0-1.0"}]}


def publish(dataset: str, staging_dir: Path, notes: str, delete_old_versions: bool = False) -> str:
    with tempfile.TemporaryDirectory(prefix="kaggle_metadata_") as metadata_tmp:
        metadata = fetch_dataset_metadata(dataset, Path(metadata_tmp))
    metadata["id"] = dataset
    Path(staging_dir, METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    command = ["kaggle", "datasets", "version", "-p", str(staging_dir), "-m", notes, "-r", "zip"]
    if delete_old_versions:
        command.append("--delete-old-versions")
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Falha ao publicar Dataset")
    return result.stdout.strip()


def run_manager(dataset: str = DEFAULT_DATASET, staging_dir: Path = DEFAULT_STAGING, input_fn=input) -> None:
    token = get_secret("CIVITAI_TOKEN") or get_secret("CIVITAI_API_KEY")
    if not token:
        raise RuntimeError("CIVITAI_TOKEN não encontrado nos Secrets/env")
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = staging_dir / MANIFEST_NAME
    current = parse_current_files(kaggle_files(dataset))
    desired = dict(current)
    queue: list[DatasetFile] = []

    while True:
        url = input_fn("\nCivitai URL (done para finalizar): ").strip()
        if url.lower() == "done":
            break
        if not url:
            print("[WARN] URL vazia")
            continue
        info = resolve_civitai_url(url, token)
        print(f"Modelo: {info['model'].get('name', 'N/A')} | arquivo: {info['file'].get('name', 'N/A')} | tamanho: {format_size(int((info['file'].get('sizeKB') or 0) * 1024))}")
        category = input_fn(f"Categoria {CATEGORIES}: ").strip()
        item = download_civitai_file(info, category, staging_dir, token, source_url=url)
        previous = current.get(item.path)
        if previous and previous.size == item.size:
            print(f"[SKIP] já presente e com mesmo tamanho: {item.path}")
        elif previous:
            print(f"[REPLACE] conteúdo alterado: {item.path}")
        desired[item.path] = item
        queue.append(item)
        print(f"[{len(queue)}] adicionado à fila: {item.path}")

    while True:
        command = input_fn("\nComando remove <path>, move <old> <new> ou done: ").strip()
        if command.lower() == "done":
            break
        parts = command.split()
        if len(parts) == 2 and parts[0].lower() == "remove":
            desired.pop(parts[1], None)
            print(f"[REMOVE] marcado: {parts[1]}")
        elif len(parts) == 3 and parts[0].lower() == "move":
            old, new = parts[1], parts[2]
            if old not in desired:
                print(f"[WARN] path inexistente: {old}")
                continue
            item = desired.pop(old)
            desired[new] = DatasetFile(new, item.size, item.sha256, item.source, item.civitai_model_id, item.civitai_version_id, item.civitai_file_id)
            source = staging_dir / old
            target = staging_dir / new
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and not target.exists():
                shutil.move(str(source), str(target))
            print(f"[MOVE] {old} -> {new}")
        elif command:
            print("[WARN] Use remove <path>, move <old> <new> ou done")

    for path, item in list(desired.items()):
        if not (staging_dir / path).exists():
            desired[path] = materialize_dataset_file(dataset, path, staging_dir, staging_dir / ".cache")
    write_manifest(manifest_path, dataset, desired)
    changes = compare_states(current, desired)
    print("\n" + render_preview(dataset, current, desired, changes))
    notes = input_fn("\nNotas da versão: ").strip() or "Update Dataset state"
    if input_fn("Publicar esta nova versão? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
        print("[CANCEL] Dataset remoto não foi alterado.")
        return
    output = publish(dataset, staging_dir, notes)
    print("\n" + "=" * 60 + "\nDATASET UPDATE COMPLETE\n" + "=" * 60)
    print(f"Dataset: {dataset}\nVersion notes: {notes}\nAdded: {len(changes.added)}\nRemoved: {len(changes.removed)}\nMoved: {len(changes.moved)}\nUnchanged: {len(changes.unchanged)}")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Administração independente do Kaggle Dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--staging-dir", default=str(DEFAULT_STAGING))
    args = parser.parse_args()
    run_manager(args.dataset, Path(args.staging_dir))


if __name__ == "__main__":
    main()
