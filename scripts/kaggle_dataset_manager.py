#!/usr/bin/env python3
"""Independent Colab tool for building and publishing complete Kaggle Dataset state."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def resolve_dataset_name(override: Optional[str] = None) -> str:
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


DEFAULT_DATASET = resolve_dataset_name()
DEFAULT_STAGING = Path("/content/kaggle_dataset_manager")
CATEGORIES = (
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
)
MANIFEST_NAME = "dataset-manifest.json"
METADATA_NAME = "dataset-metadata.json"
CIVITAI_CLI_PACKAGE = "@civitai/cli@0.1.104"
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

RETRIABLE_ERRORS = (
    urllib.error.HTTPError,
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    subprocess.TimeoutExpired,
)


def retry(max_attempts: int = 3, delay: float = 2, backoff: float = 2) -> Callable:
    """Re-tenta erros transitórios de rede com backoff exponencial."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RETRIABLE_ERRORS as exc:
                    if attempt >= max_attempts:
                        raise
                    print(f"[WARN] {func.__name__}: {exc} (tentativa {attempt}/{max_attempts}); nova tentativa em {wait:.0f}s")
                    time.sleep(wait)
                    wait *= backoff

        return wrapper

    return decorator


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
    hf_repo_id: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_file_path: Optional[str] = None


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


def validate_air(value: str) -> tuple[Optional[dict[str, int]], Optional[str]]:
    """Valida um AIR: retorna ({"model_id", "version_id"}, None) ou (None, erro)."""
    try:
        parsed = parse_air(value)
    except ValueError as exc:
        return None, str(exc)
    model_id = int(parsed["model_id"])
    version_id = int(parsed["version_id"])
    if model_id <= 0 or version_id <= 0:
        return None, "AIR inválido: model_id e version_id devem ser inteiros > 0"
    return {"model_id": model_id, "version_id": version_id}, None


def validate_civitai_url(value: str) -> Optional[str]:
    """Retorna mensagem de erro quando a URL Civitai é inválida; None se válida."""
    parsed = urllib.parse.urlparse(str(value or "").strip())
    host = parsed.netloc.lower()
    if parsed.scheme not in {"http", "https"} or not (host == "civitai.com" or host.endswith(".civitai.com")):
        return "URL inválida: use https://civitai.com/models/<id> ou um link de download Civitai"
    if not (re.search(r"/models/\d+", parsed.path) or "/api/download/models/" in parsed.path):
        return "URL Civitai deve conter /models/<id> ou /api/download/models/<id>"
    return None


def is_hf_input(value: str) -> bool:
    """Detecta se a entrada é um Hugging Face repo/arquivo: 'hf:' prefixo ou URL huggingface.co."""
    value = str(value or "").strip().lower()
    if value.startswith("hf:"):
        return True
    parsed = urllib.parse.urlparse(value)
    host = parsed.netloc.lower()
    return host == "huggingface.co" or host.endswith(".huggingface.co")


def parse_hf_input(value: str) -> tuple[str, Optional[str], str]:
    """Parseia entrada HF e retorna (repo_id, file_path ou None, revision).

    Formatos:
    - hf:org/repo ou hf:org/repo/path/to/arquivo.safetensors
    - https://huggingface.co/org/repo/resolve/main/arquivo.safetensors
    - https://huggingface.co/org/repo/blob/main/arquivo.safetensors
    """
    value = str(value or "").strip()

    if value.lower().startswith("hf:"):
        # hf:org/repo/path/...
        hf_part = value[3:]  # Remove "hf:"
        parts = hf_part.split("/")
        if len(parts) < 2:
            raise ValueError("Formato HF inválido: use hf:org/repo ou hf:org/repo/path/to/arquivo")
        repo_id = f"{parts[0]}/{parts[1]}"
        file_path = "/".join(parts[2:]) if len(parts) > 2 else None
        return repo_id, file_path, "main"

    # URL format
    parsed = urllib.parse.urlparse(value)
    path_parts = [p for p in parsed.path.split("/") if p]

    # Expected: /org/repo/resolve/main/path/to/file ou /org/repo/blob/main/path/to/file
    if len(path_parts) < 2:
        raise ValueError("URL HF inválida: esperado /org/repo/resolve/rev/arquivo ou /org/repo/blob/rev/arquivo")

    repo_id = f"{path_parts[0]}/{path_parts[1]}"

    # Detectar resolve/blob
    if len(path_parts) >= 4 and path_parts[2] in {"resolve", "blob"}:
        revision = path_parts[3]
        file_path = "/".join(path_parts[4:]) if len(path_parts) > 4 else None
    else:
        # Fallback: tudo após /org/repo é caminho
        file_path = "/".join(path_parts[2:]) if len(path_parts) > 2 else None
        revision = "main"

    return repo_id, file_path, revision


def validate_hf_input(value: str) -> Optional[str]:
    """Valida entrada HF; retorna mensagem de erro ou None."""
    if not is_hf_input(value):
        return "Entrada não é HF (use hf:org/repo ou URL huggingface.co)"
    try:
        parse_hf_input(value)
        return None
    except ValueError as exc:
        return str(exc)


def _hf_api_available() -> bool:
    """Verifica se huggingface_hub está disponível."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def _hf_list_repo_files(repo_id: str, revision: str = "main", token: Optional[str] = None) -> list[str]:
    """Lista arquivos do repositório HF. Retorna nomes dos arquivos."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(
            "huggingface_hub não está instalado. Execute no Colab:\n"
            "  !pip install huggingface_hub\n"
            "ou inclua HF_TOKEN nos Secrets do Colab para repositórios privados."
        )
    try:
        api = HfApi()
        info = api.model_info(repo_id, revision=revision, token=token)
        return [f.filename for f in (info.siblings or [])]
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "RepositoryNotFoundError":
            raise RuntimeError(f"Repositório não encontrado: {repo_id}")
        elif exc_name == "GatedRepoError":
            raise RuntimeError(
                f"Repositório gated (acesso restrito): {repo_id}\n"
                f"Configure HF_TOKEN nos Secrets do Colab com acesso a este repositório."
            )
        raise RuntimeError(f"Erro ao listar arquivos HF: {exc}")


def _hf_file_size(repo_id: str, file_path: str, revision: str = "main", token: Optional[str] = None) -> int:
    """Obtém tamanho do arquivo HF em bytes."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub não está instalado (execute !pip install huggingface_hub no Colab)")
    try:
        api = HfApi()
        info = api.model_info(repo_id, revision=revision, token=token)
        found = False
        for file_info in (info.siblings or []):
            if file_info.filename == file_path:
                found = True
                if file_info.size:
                    return file_info.size or 0
                break
        # Fallback: metadata direta do arquivo (casos sem size nos siblings)
        from huggingface_hub import get_hf_file_metadata

        metadata = get_hf_file_metadata(api.hf_hub_url(repo_id, file_path, revision=revision), token=token)
        if metadata.size:
            return metadata.size
        if found:
            return 0
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path} em {repo_id}")
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "RepositoryNotFoundError":
            raise RuntimeError(f"Repositório não encontrado: {repo_id}")
        elif exc_name == "GatedRepoError":
            raise RuntimeError(f"Repositório gated; HF_TOKEN sem acesso: {repo_id}")
        raise RuntimeError(f"Erro ao obter tamanho do arquivo HF: {exc}")


def _hf_hub_download(repo_id: str, filename: str, revision: str = "main", token: Optional[str] = None, local_dir: Optional[str] = None) -> str:
    """Baixa arquivo HF via hf_hub_download. Retorna o caminho local."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub não está instalado")
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token, local_dir=local_dir)
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "RepositoryNotFoundError":
            raise RuntimeError(f"Repositório não encontrado: {repo_id}")
        elif exc_name == "GatedRepoError":
            raise RuntimeError(f"Repositório gated (acesso restrito): {repo_id}. Verifique HF_TOKEN.")
        elif exc_name == "EntryNotFoundError":
            raise RuntimeError(f"Arquivo não encontrado: {filename} em {repo_id}")
        raise RuntimeError(f"Erro ao baixar arquivo HF: {exc}")


def classify_resource_type(resource_type: str, input_fn=input, checkpoint_destination: Optional[str] = None) -> str:
    """Mapeia um resource_type para categoria do dataset, com fallback interativo.

    Para tipo vazio/desconhecido, SEMPRE pergunta ao usuário (usado para HF).
    """
    resource_type = str(resource_type or "").strip().lower()
    if not resource_type:
        # Tipo não disponível (HF genérico)
        return normalize_category(input_fn(f"Categoria {CATEGORIES}: "))
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
        if checkpoint_destination:
            return normalize_category(checkpoint_destination)
        choice = input_fn("Checkpoint: 1=checkpoints/ 2=diffusion_models/: ").strip().lower()
        return normalize_category({"1": "checkpoints", "2": "diffusion_models"}.get(choice, choice))
    print(f"[WARN] Tipo desconhecido: {resource_type}")
    return normalize_category(input_fn(f"Destino manual {CATEGORIES}: "))


def classify_civitai_type(resource_type: str, input_fn=input, checkpoint_destination: Optional[str] = None) -> str:
    """Mapeia o resource_type Civitai para a categoria do dataset.

    Para ``checkpoint``, ``checkpoint_destination`` pré-respondido evita o prompt
    interativo, permitindo decidir o destino uma única vez por lote.
    Delega a classify_resource_type para evitar duplicação de lógica.
    """
    return classify_resource_type(resource_type, input_fn=input_fn, checkpoint_destination=checkpoint_destination)


def _expected_sha256(file_info: dict[str, Any]) -> Optional[str]:
    hashes = file_info.get("hashes") or {}
    if isinstance(hashes, dict):
        for key, value in hashes.items():
            if str(key).lower() in {"sha256", "sha-256"} and value:
                return str(value).lower()
    return None


def ensure_civitai_cli() -> str:
    """Retorna o CLI oficial, instalando uma versão estável se necessário."""
    executable = shutil.which("civitai")
    if not executable:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("CLI civitai não encontrado e npm não está disponível")
        subprocess.run(
            [npm, "install", "--global", CIVITAI_CLI_PACKAGE],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        executable = shutil.which("civitai")
    if not executable:
        raise RuntimeError("civitai CLI não ficou disponível após a instalação")
    version = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    version_text = (version.stdout or version.stderr or "").strip()
    if not version_text:
        raise RuntimeError("civitai --version não retornou uma versão")
    print(f"Civitai CLI:\n{version_text}")
    return executable


@retry()
def download_with_civitai_cli(
    version_id: str,
    file_id: str,
    destination: Path,
    token: str,
    expected_size: int = 0,
    expected_sha256: Optional[str] = None,
) -> Path:
    """Baixa por version/file id e valida o destino final promovido pelo CLI."""
    if not version_id or not file_id:
        raise ValueError("version_id e file_id são obrigatórios para o download")
    if not token:
        raise ValueError("CIVITAI_TOKEN é obrigatório para o download")

    cli = ensure_civitai_cli()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        actual_size = destination.stat().st_size
        actual_hash = sha256_file(destination)
        if (not expected_size or actual_size == expected_size) and (not expected_sha256 or actual_hash.lower() == expected_sha256.lower()):
            print(f"[SKIP] arquivo válido já existe: {destination}")
            return destination

    environment = os.environ.copy()
    environment["CIVITAI_TOKEN"] = token
    environment["CIVITAI_NO_UPDATE_CHECK"] = "1"
    command = [
        cli,
        "download",
        str(version_id),
        "--file",
        str(file_id),
        "--out",
        str(destination),
        "--force",
        "--no-update-check",
    ]
    print(f"Version ID: {version_id}")
    print(f"File ID: {file_id}")
    print(f"Destination: {destination}")
    result = subprocess.run(
        command,
        env=environment,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        error_output = (result.stderr or "").replace(token, "***REDACTED***")
        raise RuntimeError(
            f"civitai download falhou ({result.returncode}): "
            f"{error_output[-2000:]}"
        )
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"civitai download não produziu o arquivo final: {destination}")

    actual_size = destination.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(f"Tamanho inválido: {actual_size}; esperado {expected_size}")
    actual_hash = sha256_file(destination)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        raise RuntimeError("SHA256 inválido para o arquivo baixado")
    return destination


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


def resolve_hf_input(value: str, hf_token: Optional[str] = None, input_fn=input) -> list[dict[str, Any]]:
    """Resolve entrada HF em uma ou mais filas.

    Retorna [{repo_id, file_path, revision, filename, category, base_model, size, source_url, air: None}].
    Se arquivo não for especificado no input, lista e pede ao usuário escolher.
    """
    value = str(value or "").strip()
    if not _hf_api_available():
        print(
            "[WARN] huggingface_hub não está instalado; instale antes da fase de download: "
            "!pip install huggingface_hub"
        )
    repo_id, file_path, revision = parse_hf_input(value)

    # Se não tem arquivo, listar e pedir
    if not file_path:
        files = _hf_list_repo_files(repo_id, revision=revision, token=hf_token)
        if not files:
            raise RuntimeError(f"Repositório vazio: {repo_id}")
        print(f"Arquivos em {repo_id}:")
        for idx, fname in enumerate(files, start=1):
            print(f"  {idx}: {fname}")
        choice = input_fn("Escolha o número do arquivo (ou nome completo): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                file_path = files[idx]
            else:
                file_path = choice  # Assume nome completo
        except ValueError:
            file_path = choice  # Assume nome completo

    # Obter tamanho
    size = _hf_file_size(repo_id, file_path, revision=revision, token=hf_token)
    filename = Path(file_path).name

    # SEMPRE perguntar categoria para HF
    category = classify_resource_type("", input_fn=input_fn)  # "" -> sempre pergunta

    # SEMPRE perguntar base_model para HF
    base_model = input_fn("Base model (ex: sdxl, krea2, etc): ").strip() or "unknown"

    # Imprimir no formato esperado
    print(
        f"Modelo: {repo_id} | "
        f"arquivo: {file_path} | "
        f"tipo: {category} | "
        f"base model: {base_model}"
    )

    # Source URL (resolve endpoint)
    source_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{file_path}"

    return [{
        "repo_id": repo_id,
        "file_path": file_path,
        "revision": revision,
        "filename": filename,
        "category": category,
        "base_model": base_model,
        "size": size,
        "source_url": source_url,
        "air": None,
        "source": "hf",
    }]


@retry()
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


def download_hf_file(
    repo_id: str,
    file_path: str,
    category: str,
    staging_dir: Path,
    hf_token: Optional[str] = None,
    revision: str = "main",
    source_url: Optional[str] = None,
    base_model: Optional[str] = None,
) -> DatasetFile:
    """Baixa arquivo HF e retorna DatasetFile."""
    destination_dir = Path(staging_dir) / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Usar hf_hub_download com local_dir
    try:
        downloaded_path = _hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            token=hf_token,
            local_dir=str(destination_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"Erro ao baixar {file_path} de {repo_id}: {exc}")

    # hf_hub_download retorna o caminho completo; normalizar para staging/category/filename
    downloaded_path = Path(downloaded_path)
    filename = downloaded_path.name
    dataset_path = build_dataset_path(category, filename)
    final_destination = Path(staging_dir) / dataset_path

    # Se o arquivo foi colocado num subdiretório, mover para a raiz da categoria
    if downloaded_path != final_destination and downloaded_path.exists():
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        if final_destination.exists():
            final_destination.unlink()
        shutil.move(str(downloaded_path), str(final_destination))

    # Validar tamanho e calcular hash
    if not final_destination.exists():
        raise RuntimeError(f"Arquivo não encontrado após download: {final_destination}")

    actual_size = final_destination.stat().st_size
    actual_hash = sha256_file(final_destination)

    print(f"[{dataset_path}] Tamanho: {format_size(actual_size)}, SHA256: {actual_hash[:16]}...")

    return DatasetFile(
        dataset_path,
        actual_size,
        actual_hash,
        source_url,
        None,  # civitai_model_id
        None,  # civitai_version_id
        None,  # civitai_file_id
        None,  # air
        base_model,
        repo_id,  # hf_repo_id
        revision,  # hf_revision
        file_path,  # hf_file_path
    )


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
        print(f"Download: {filename} -> {dataset_path}")
        download_with_civitai_cli(
            version_id=str(info["version"]["id"]),
            file_id=str(file_info["id"]),
            destination=destination,
            token=token,
            expected_size=expected_size,
            expected_sha256=expected_hash,
        )
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


def collect_input_queue(input_fn=input) -> list[str]:
    """Coleta pura da fila AIR/URL/HF: valida cada entrada até o usuário digitar 'done'."""
    pending: list[str] = []
    while True:
        value = input_fn("\nAIR/URL/HF (done para finalizar): ").strip()
        if value.lower() == "done":
            break
        if not value:
            print("[WARN] Entrada vazia")
            continue
        if value.lower().startswith("urn:air:"):
            _, error = validate_air(value)
        elif is_hf_input(value):
            error = validate_hf_input(value)
        else:
            error = validate_civitai_url(value)
        if error:
            print(f"[WARN] Entrada inválida: {error}")
            continue
        pending.append(value)
    print(f"[INFO] Lista fechada com {len(pending)} item(ns).")
    return pending


def resolve_queue_metadata(pending: list[str], token: str, input_fn=input, hf_token: Optional[str] = None) -> list[dict[str, Any]]:
    """Resolve os metadados Civitai/HF de cada item da fila sem baixar nada.

    Retorna uma entrada por arquivo resolvido, preservando os metadados resolvidos,
    para que a fase de download não precise chamar as APIs da Civitai/HF novamente.
    """
    if hf_token is None:
        hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        print("[WARN] HF_TOKEN não configurado; repositórios HF privados/gated falharão.")

    resolved: list[dict[str, Any]] = []
    failures = 0
    total = len(pending)
    for index, value in enumerate(pending, start=1):
        print(f"[INFO] Resolvendo metadados [{index}/{total}]: {value}")
        try:
            if value.lower().startswith("urn:air:") or (not is_hf_input(value)):
                # Civitai AIR ou URL
                infos = resolve_civitai_input(value, token, input_fn)
                source = "civitai"
            else:
                # Hugging Face
                infos = resolve_hf_input(value, hf_token=hf_token, input_fn=input_fn)
                source = "hf"
        except Exception as exc:
            failures += 1
            print(f"[ERROR] Falha ao resolver [{index}/{total}] {value}: {exc}")
            continue
        for info in infos:
            if source == "civitai":
                air = info.get("air") or {}
                resource_type = air.get("type") or info["model"].get("type", "unknown")
            else:
                # HF
                resource_type = None
            resolved.append({
                "value": value,
                "index": index,
                "total": total,
                "info": info,
                "resource_type": resource_type,
                "source": source,
            })
    if failures:
        print(f"[WARN] Resolução concluída com {failures} falha(s); {len(resolved)} arquivo(s) resolvido(s).")
    return resolved


def queue_contains_checkpoint(resolved: list[dict[str, Any]]) -> bool:
    """Detecta se há pelo menos um resource_type 'checkpoint' no lote inteiro resolvido."""
    return any(str(entry.get("resource_type") or "").strip().lower() == "checkpoint" for entry in resolved)


def download_resolved_queue(
    resolved: list[dict[str, Any]],
    staging_dir: Path,
    token: str,
    input_fn=input,
    checkpoint_destination: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> list[DatasetFile]:
    """Executa os downloads da fila já resolvida, sem nenhum prompt intermediário.

    ``checkpoint_destination`` é a resposta única do lote para itens do tipo
    checkpoint; quando None, classify_civitai_type mantém o comportamento
    interativo por item (compatibilidade).
    """
    if hf_token is None:
        hf_token = get_secret("HF_TOKEN")

    print("[INFO] Iniciando downloads...")
    queue: list[DatasetFile] = []
    failures = 0
    last_value: Optional[str] = None
    for entry in resolved:
        value = entry["value"]
        if value != last_value:
            print(f"--- [{entry['index']}/{entry['total']}] {value} ---")
            last_value = value
        info = entry["info"]
        source = entry.get("source", "civitai")  # Default para compatibilidade com mocks de teste antigos

        try:
            if source == "hf":
                # HF: categoria e base_model já perguntados na Fase 1
                repo_id = info["repo_id"]
                file_path = info["file_path"]
                category = info["category"]
                base_model = info["base_model"]
                revision = info.get("revision", "main")
                source_url = info.get("source_url")
                item = download_hf_file(
                    repo_id=repo_id,
                    file_path=file_path,
                    category=category,
                    staging_dir=staging_dir,
                    hf_token=hf_token,
                    revision=revision,
                    source_url=source_url,
                    base_model=base_model,
                )
            else:
                # Civitai
                air = info.get("air") or {}
                resource_type = entry["resource_type"]
                category = classify_civitai_type(resource_type, input_fn, checkpoint_destination=checkpoint_destination)
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
        except Exception as exc:
            failures += 1
            print(f"[ERROR] Download falhou para [{entry['index']}/{entry['total']}] {value}: {exc}")
            continue
    if failures:
        print(f"[WARN] Lote concluído com {failures} falha(s); {len(queue)} arquivo(s) na fila.")
    return queue


def download_input_queue(
    staging_dir: Path,
    token: str,
    input_fn=input,
    hf_token: Optional[str] = None,
) -> list[DatasetFile]:
    """Fila síncrona AIR/URL/HF: encadeia coleta -> resolução -> download.

    Mantida como atalho público equivalente; o orquestrador chama as etapas
    separadamente para antecipar todas as perguntas interativas.
    """
    pending = collect_input_queue(input_fn)
    resolved = resolve_queue_metadata(pending, token, input_fn, hf_token=hf_token)
    return download_resolved_queue(resolved, staging_dir, token, input_fn, hf_token=hf_token)


def kaggle_files(dataset: str) -> list[dict[str, Any]]:
    result = subprocess.run(["kaggle", "datasets", "files", dataset], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Falha ao consultar arquivos do Dataset")
    items = []
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].lower() not in {"name", "file", "files"} and not parts[0].startswith("-"):
            size = parts[1]
            items.append({"path": parts[0], "name": Path(parts[0]).name, "size": size})
    return items


def materialize_dataset_file(dataset: str, dataset_file: str, staging_dir: Path, cache_dir: Path) -> DatasetFile:
    destination = Path(staging_dir) / dataset_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Preservando arquivo existente via download seletivo: {dataset_file}")
        process = subprocess.Popen(
            ["kaggle", "datasets", "download", dataset, "-f", dataset_file, "-p", str(cache_dir), "--unzip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[kaggle] {line.rstrip()}", flush=True)
            return_code = process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError(f"Timeout preservando {dataset_file}")
        if return_code != 0:
            raise RuntimeError(f"Falha ao preservar {dataset_file} (exit {return_code})")
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
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            output.append(text)
            print(f"[kaggle] {text}", flush=True)
        return_code = process.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError("Timeout publicando Dataset")
    if return_code != 0:
        raise RuntimeError(f"Falha ao publicar Dataset (exit {return_code})")
    return "\n".join(output).strip()


def collect_dataset_edits(dataset: str, input_fn=input) -> list[tuple[str, ...]]:
    """Coleta edições remove/move sobre o estado atual do dataset remoto, sem aplicá-las.

    Mostra apenas a listagem "DATASET CURRENT" como contexto (os arquivos novos
    ainda não foram baixados neste ponto) e retorna as operações na ordem digitada:
    ("remove", path) ou ("move", old, new).
    """
    current = parse_current_files(kaggle_files(dataset))
    print("\nDATASET CURRENT")
    for item in current.values():
        print(f"= {item.path} ({format_size(item.size)})")

    edits: list[tuple[str, ...]] = []
    if input_fn("Você deseja modificar arquivos existentes? [s/N] ").strip().lower() in {"s", "sim", "y", "yes"}:
        while True:
            command = input_fn("Comando remove <path>, move <old> <new> ou done: ").strip()
            if command.lower() == "done":
                break
            parts = command.split()
            if len(parts) == 2 and parts[0].lower() == "remove":
                edits.append(("remove", parts[1]))
                print(f"[REMOVE] marcado: {parts[1]}")
            elif len(parts) == 3 and parts[0].lower() == "move":
                edits.append(("move", parts[1], parts[2]))
                print(f"[MOVE] {parts[1]} -> {parts[2]}")
            else:
                print("[WARN] Comando inválido")
    return edits


def publish_staged_state(
    dataset: str,
    staging_dir: Path,
    input_fn=input,
    pending_edits: Iterable[tuple[str, ...]] = (),
) -> Optional[str]:
    """Constrói o estado completo a partir do staging e publica após preview.

    ``pending_edits`` são as operações remove/move coletadas antecipadamente por
    collect_dataset_edits; são aplicadas aqui sobre o estado desejado já montado
    (staging + manifest), sem nenhum prompt adicional.
    """
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

    for edit in pending_edits:
        if len(edit) == 2 and edit[0].lower() == "remove":
            desired.pop(edit[1], None)
        elif len(edit) == 3 and edit[0].lower() == "move":
            old, new = edit[1], edit[2]
            if old not in desired:
                print(f"[WARN] path inexistente: {old}")
                continue
            item = desired.pop(old)
            desired[new] = DatasetFile(new, item.size, item.sha256, item.source, item.civitai_model_id, item.civitai_version_id, item.civitai_file_id, item.air, item.base_model, item.hf_repo_id, item.hf_revision, item.hf_file_path)
            source, target = staging_dir / old, staging_dir / new
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and not target.exists():
                shutil.move(str(source), str(target))

    print("\nDATASET CURRENT")
    for item in current.values():
        print(f"= {item.path} ({format_size(item.size)})")
    print("\nNEW STAGED FILES")
    for path in sorted(set(desired) - set(current)):
        print(f"+ {path} ({format_size(desired[path].size)})")

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
