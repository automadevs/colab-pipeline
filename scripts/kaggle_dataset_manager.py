#!/usr/bin/env python3
"""Independent Colab tool for building and publishing complete Kaggle Dataset state."""
from __future__ import annotations

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
CATEGORY_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "diffusion": "diffusion_models",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "lora": "loras",
    "loras": "loras",
    "locon": "loras",
    "dora": "loras",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "textual_inversion": "embeddings",
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "clip": "clip",
    "controlnet": "controlnet",
    "vae": "vae",
    "video": "video_models",
    "video_models": "video_models",
    "upscaler": "upscale_models",
    "upscale_models": "upscale_models",
    "motion_module": "video_models",
}


@dataclass(frozen=True)
class DatasetFile:
    path: str
    size: int
    sha256: Optional[str] = None
    source: Optional[str] = None
    civitai_model_id: Optional[str] = None
    civitai_version_id: Optional[str] = None
    civitai_file_id: Optional[str] = None
    air: Optional[str] = None
    base_model: Optional[str] = None


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


def normalize_category(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CATEGORY_ALIASES:
        raise ValueError(f"Categoria inválida ou desconhecida: {value}")
    return CATEGORY_ALIASES[key]


def parse_air(value: str) -> dict[str, Any]:
    """Parseia urn:air:<base>:<type>:civitai:<model>@<version>[+<file>...]."""
    match = re.fullmatch(
        r"urn:air:([^:]+):([^:]+):civitai:(\d+)@(\d+)(?:\+([\d+]+))?",
        str(value or "").strip(),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("AIR inválido: use urn:air:<base>:<type>:civitai:<model>@<version>[+<file>]")
    files = match.group(5).split("+") if match.group(5) else []
    return {
        "air": value,
        "base_model": match.group(1),
        "type": match.group(2).lower(),
        "model_id": match.group(3),
        "version_id": match.group(4),
        "file_ids": files,
        "file_id": files[0] if files else None,
    }


def classify_civitai_type(resource_type: str, input_fn=input) -> str:
    resource_type = str(resource_type or "").lower()
    if resource_type in {"lora", "locon", "dora"}:
        return "loras"
    if resource_type == "vae":
        return "vae"
    if resource_type in {"text_encoder", "text_encoders"}:
        return "text_encoders"
    if resource_type == "clip":
        return "clip"
    if resource_type in {"textual_inversion", "embedding"}:
        return "embeddings"
    if resource_type == "controlnet":
        return "controlnet"
    if resource_type == "upscaler":
        return "upscale_models"
    if resource_type in {"motion_module", "video", "video_model"}:
        return "video_models"
    if resource_type == "checkpoint":
        choice = input_fn("Checkpoint: 1=checkpoints/ 2=diffusion_models/: ").strip().lower()
        return normalize_category({"1": "checkpoints", "2": "diffusion_models"}.get(choice, choice))
    print(f"[WARN] Tipo Civitai desconhecido: {resource_type}")
    return normalize_category(input_fn(f"Destino manual {CATEGORIES}: "))


def _expected_sha256(file_info: dict[str, Any]) -> Optional[str]:
    hashes = file_info.get("hashes") or {}
    if isinstance(hashes, dict):
        for key, value in hashes.items():
            if str(key).lower() in {"sha256", "sha-256"} and value:
                return str(value).lower()
    return None


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
        "files": [
            {
                "dataset_path": desired[path].path,
                "filename": Path(desired[path].path).name,
                "category": Path(desired[path].path).parent.name,
                **asdict(desired[path]),
            }
            for path in sorted(desired)
        ],
    }


def write_manifest(path: Path, dataset: str, desired: dict[str, DatasetFile]) -> None:
    Path(path).write_text(json.dumps(manifest_payload(dataset, desired), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, DatasetFile]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = set(DatasetFile.__dataclass_fields__)
    result = {}
    for item in data.get("files", []):
        normalized = dict(item)
        normalized["path"] = normalized.get("path") or normalized.get("dataset_path")
        result[normalized["path"]] = DatasetFile(**{key: value for key, value in normalized.items() if key in fields})
    return result


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


def resolve_civitai_input(value: str, token: str, input_fn=input) -> list[dict[str, Any]]:
    """Resolve AIR, página Civitai ou URL de download em uma ou mais filas."""
    value = str(value or "").strip()
    if value.lower().startswith("urn:air:"):
        air = parse_air(value)
        info = resolve_civitai_url(
            f"https://civitai.com/models/{air['model_id']}?modelVersionId={air['version_id']}",
            token,
        )
        selected = air["file_ids"]
        files = info["version"].get("files") or []
        if selected:
            files = [item for item in files if str(item.get("id")) in selected]
            if not files:
                raise ValueError("Nenhum file_id do AIR foi encontrado na versão Civitai")
        return [{**info, "air": air, "file": file_info} for file_info in files]
    if "/api/download/models/" in value:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(value).query)
        parsed = urllib.parse.urlparse(value)
        version_match = re.search(r"/models/(\d+)", parsed.path)
        version_id = query.get("modelVersionId", [None])[0] or (version_match.group(1) if version_match else None)
        file_id = query.get("fileId", [None])[0]
        if not version_id:
            raise ValueError("URL de download deve conter modelVersionId e fileId")
        request = urllib.request.Request(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "colab-pipeline-dataset-manager"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            version = json.load(response)
        model = version.get("model") or {}
        info = {"model": model, "version": version, "file": (version.get("files") or [{}])[0], "model_id": str(model.get("id", "unknown"))}
        if file_id:
            info["file"] = next((item for item in info["version"].get("files", []) if str(item.get("id")) == file_id), info["file"])
        info["air"] = None
        return [info]
    info = resolve_civitai_url(value, token)
    info["air"] = None
    return [info]


def download_civitai_file(
    info: dict[str, Any],
    category: str,
    staging_dir: Path,
    token: str,
    source_url: Optional[str] = None,
    air: Optional[dict[str, Any]] = None,
) -> DatasetFile:
    file_info = info["file"]
    filename = Path(file_info.get("name") or "model.safetensors").name
    dataset_path = build_dataset_path(category, filename)
    destination = Path(staging_dir) / dataset_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(float(file_info.get("sizeKB") or 0) * 1024)
    expected_hash = _expected_sha256(file_info)
    if destination.exists() and destination.stat().st_size > 0 and (not expected_size or destination.stat().st_size == expected_size) and (not expected_hash or sha256_file(destination) == expected_hash):
        print(f"[SKIP] já presente no staging e com tamanho válido: {dataset_path}")
    else:
        url = file_info.get("downloadUrl") or f"https://civitai.com/api/download/models/{info['version']['id']}?fileId={file_info['id']}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "colab-pipeline-dataset-manager"})
        partial = destination.with_name(destination.name + ".part")
        print(f"Download: {filename} -> {dataset_path}")
        if partial.exists():
            partial.unlink()
        with urllib.request.urlopen(request, timeout=7200) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size == 0:
            partial.unlink()
            raise RuntimeError(f"Download vazio para {dataset_path}")
        if expected_size and partial.stat().st_size != expected_size:
            partial.unlink()
            raise RuntimeError(f"Tamanho inválido para {dataset_path}: {partial.stat().st_size} bytes; esperado {expected_size}")
        if expected_hash and sha256_file(partial) != expected_hash:
            partial.unlink()
            raise RuntimeError(f"SHA256 inválido para {dataset_path}")
        partial.replace(destination)
    actual_size = destination.stat().st_size
    if actual_size == 0 or (expected_size and actual_size != expected_size):
        raise RuntimeError(f"Tamanho inválido para {dataset_path}: {actual_size} bytes; esperado {expected_size}")
    return DatasetFile(
        dataset_path,
        actual_size,
        sha256_file(destination),
        source_url,
        str(info["model_id"]),
        str(info["version"].get("id")),
        str(file_info.get("id")),
        (air or {}).get("air"),
        (air or {}).get("base_model"),
    )


def download_input_queue(
    staging_dir: Path,
    token: str,
    input_fn=input,
) -> list[DatasetFile]:
    """Fila síncrona AIR/URL para o notebook 02."""
    queue: list[DatasetFile] = []
    while True:
        value = input_fn("\nAIR/URL (done para finalizar): ").strip()
        if value.lower() == "done":
            return queue
        if not value:
            print("[WARN] Entrada vazia")
            continue
        for info in resolve_civitai_input(value, token, input_fn):
            air = info.get("air") or {}
            resource_type = air.get("type") or info["model"].get("type", "unknown")
            category = classify_civitai_type(resource_type, input_fn)
            file_info = info["file"]
            base_model = air.get("base_model") or info["version"].get("baseModel") or info["model"].get("baseModel")
            manifest_input = {
                **air,
                "air": air.get("air") or (value if value.lower().startswith("urn:air:") else None),
                "base_model": base_model,
            }
            print(
                f"Modelo: {info['model'].get('name', 'N/A')} | "
                f"versão: {info['version'].get('name', info['version'].get('id'))} | "
                f"arquivo: {file_info.get('name', 'N/A')} | "
                f"tipo: {resource_type} | base model: {base_model or 'N/A'}"
            )
            item = download_civitai_file(info, category, staging_dir, token, source_url=value, air=manifest_input)
            if any(existing.path == item.path and existing.size == item.size and existing.sha256 == item.sha256 for existing in queue):
                print(f"[SKIP] já presente na fila e idêntico: {item.path}")
                continue
            queue.append(item)
            print(f"[{len(queue)}] Download concluído: {item.path}")


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


def publish_staged_state(
    dataset: str,
    staging_dir: Path,
    input_fn=input,
) -> Optional[str]:
    """Constrói o estado completo a partir do staging e publica após preview."""
    staging_dir = Path(staging_dir)
    current = parse_current_files(kaggle_files(dataset))
    desired = dict(current)
    manifest_path = staging_dir / MANIFEST_NAME
    if manifest_path.exists():
        for path, item in read_manifest(manifest_path).items():
            desired[path] = item
    else:
        for path in sorted(p.relative_to(staging_dir).as_posix() for p in staging_dir.rglob("*") if p.is_file() and p.name not in {MANIFEST_NAME, METADATA_NAME}):
            file_path = staging_dir / path
            desired[path] = DatasetFile(path, file_path.stat().st_size, sha256_file(file_path))

    print("\nDATASET CURRENT")
    for item in current.values():
        print(f"= {item.path} ({format_size(item.size)})")
    print("\nNEW STAGED FILES")
    for path in sorted(set(desired) - set(current)):
        print(f"+ {path} ({format_size(desired[path].size)})")

    if input_fn("Você deseja modificar arquivos existentes? [s/N] ").strip().lower() in {"s", "sim", "y", "yes"}:
        while True:
            command = input_fn("Comando remove <path>, move <old> <new> ou done: ").strip()
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
                desired[new] = DatasetFile(new, item.size, item.sha256, item.source, item.civitai_model_id, item.civitai_version_id, item.civitai_file_id, item.air, item.base_model)
                source, target = staging_dir / old, staging_dir / new
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.exists() and not target.exists():
                    shutil.move(str(source), str(target))
                print(f"[MOVE] {old} -> {new}")
            else:
                print("[WARN] Comando inválido")

    for path in list(desired):
        if not (staging_dir / path).exists():
            desired[path] = materialize_dataset_file(dataset, path, staging_dir, staging_dir / ".cache")
    write_manifest(manifest_path, dataset, desired)
    changes = compare_states(current, desired)
    print("\n" + render_preview(dataset, current, desired, changes))
    notes = input_fn("Notas da versão: ").strip() or "Update Dataset state"
    if input_fn("Publicar? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
        print("[CANCEL] Nenhuma alteração remota foi feita.")
        return None
    return publish(dataset, staging_dir, notes)
