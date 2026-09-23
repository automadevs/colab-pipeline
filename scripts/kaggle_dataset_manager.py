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
from dataclasses import asdict, dataclass, field
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
            "No Colab, confira se os Secrets existem, se estão com \"Notebook access\"\n"
            "habilitado e se foram injetados em os.environ (ver colab_transfer/00_master_pipeline.ipynb).\n"
            "Ou passe o dataset explicitamente via --dataset \"owner/nome\"."
        )

    return f"{username}/{dataset_name}"


try:
    DEFAULT_DATASET: Optional[str] = resolve_dataset_name()
except ValueError:
    # Sem KAGGLE_USERNAME/KAGGLE_DATASET_NAME o módulo continua importável (a suíte de
    # testes roda sem secrets); a publicação resolve o dataset explicitamente em runtime.
    DEFAULT_DATASET = None
DEFAULT_STAGING = Path("/content/kaggle_dataset_manager")
CATEGORIES = (
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
)
MANIFEST_NAME = "dataset-manifest.json"
METADATA_NAME = "dataset-metadata.json"
CIVITAI_CLI_PACKAGE = "@civitai/cli@0.1.104"
# Extenções aceitas para arquivos de pesos baixados do Hugging Face.
HF_ALLOWED_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".npz"}
# Variáveis de cache do huggingface_hub. As constantes HF_HOME/HF_HUB_CACHE são
# congeladas na importação do módulo: precisam ser definidas ANTES do primeiro
# import da lib (ver configure_hf_cache / inspect_environment no master_pipeline).
HF_CACHE_ENV_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
# Estados explícitos do fluxo de inputs (parser -> resolver -> downloader):
#   INPUT_COLLECTED (collect_input_queue) -> PARSED (parse_input)
#   -> RESOLVED/FAILED (resolve_queue_metadata) -> READY_TO_DOWNLOAD
#   (classify_resolved_artifacts/validação) -> DOWNLOADED (download_resolved_queue)
INPUT_STATES = ("INPUT_COLLECTED", "PARSED", "RESOLVED", "FAILED", "READY_TO_DOWNLOAD", "DOWNLOADED")
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


def validate_hf_repo_id(repo_id: str) -> Optional[str]:
    """Valida a estrutura de um repo_id HF (owner/repo); None quando válido.

    Espelha as regras do Hub (caracteres alfanuméricos + '-', '_', '.', sem
    começar/terminar com '-' ou '.', máx. 96) para falhar cedo com mensagem
    clara em vez do erro genérico da biblioteca.
    """
    parts = str(repo_id or "").split("/")
    if len(parts) != 2 or not all(parts):
        return f"repo_id HF inválido (esperado owner/repo): {repo_id!r}"
    for part in parts:
        if (
            len(part) > 96
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
            or part[0] in "-."
            or part[-1] in "-."
        ):
            return f"repo_id HF inválido: {repo_id!r}"
    return None


def normalize_hf_file_path(file_path: str) -> str:
    """Normaliza e valida um caminho de arquivo DENTRO de um repo HF.

    Levanta ValueError para: vazio, path traversal (inclusive percent-encoded
    como %2e%2e%2f), caminho absoluto (unix/windows), URL embutida e extensão
    fora de HF_ALLOWED_EXTENSIONS. Separaçõess '\\' são normalizadas para '/'.
    Retorna o caminho normalizado (relativo ao repo).
    """
    raw = str(file_path or "").strip()
    if not raw:
        raise ValueError("Caminho de arquivo HF vazio")
    candidates = [raw]
    decoded = urllib.parse.unquote(raw)
    if decoded != raw:
        candidates.append(decoded)
    normalized = ""
    for candidate in candidates:
        candidate = candidate.replace("\\", "/")
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", candidate):
            raise ValueError(f"Caminho HF inválido (URL não permitida): {file_path}")
        if candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
            raise ValueError(f"Caminho HF absoluto não permitido: {file_path}")
        segments = [segment for segment in candidate.split("/") if segment not in ("", ".")]
        if not segments:
            raise ValueError(f"Caminho de arquivo HF vazio: {file_path}")
        if any(segment == ".." for segment in segments):
            raise ValueError(f"Caminho HF com path traversal não permitido: {file_path}")
        normalized = "/".join(segments)
    extension = Path(normalized).suffix.lower()
    if extension not in HF_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(HF_ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Extensão HF não permitida: {extension or '(sem extensão)'}; use uma de: {allowed}"
        )
    return normalized


def parse_hf_input(value: str) -> tuple[str, Optional[str], str]:
    """Parser único de todas as entradas HF -> (repo_id, file_path|None, revision).

    Formatos (item 4; 'hf://NÃO' é scheme HTTP, só um atalho do prefixo 'hf:'):
    - hf:org/repo ou hf:org/repo/path/to/arquivo.safetensors (hf:// também aceito)
    - https://huggingface.co/org/repo/resolve/main/arquivo.safetensors
    - https://huggingface.co/org/repo/blob/main/arquivo.safetensors

    Puro: sem chamadas de rede e sem prompts. Valida repo_id e, quando houver
    arquivo, normaliza/valida o caminho (traversal/absoluto/extensão) via
    normalize_hf_file_path.
    """
    value = str(value or "").strip()

    if value.lower().startswith("hf:"):
        # hf:org/repo/path/... (também aceita hf://org/repo como atalho de URL)
        hf_part = value[3:]  # Remove "hf:"
        # Ignora segmentos vazios (ex.: "hf://org/repo" -> ["org", "repo"])
        parts = [p for p in hf_part.split("/") if p]
        if len(parts) < 2:
            raise ValueError("Formato HF inválido: use hf:org/repo ou hf:org/repo/path/to/arquivo")
        repo_id = f"{parts[0]}/{parts[1]}"
        file_path = "/".join(parts[2:]) if len(parts) > 2 else None
        revision = "main"
    else:
        # URL huggingface.co
        parsed = urllib.parse.urlparse(value)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) < 2:
            raise ValueError(
                "URL HF inválida: esperado /org/repo/resolve/rev/arquivo ou /org/repo/blob/rev/arquivo"
            )
        repo_id = f"{path_parts[0]}/{path_parts[1]}"
        # Detectar resolve/blob (preserva revisions diferentes de 'main')
        if len(path_parts) >= 4 and path_parts[2] in {"resolve", "blob"}:
            revision = path_parts[3]
            file_path = "/".join(path_parts[4:]) if len(path_parts) > 4 else None
        else:
            # Fallback: tudo após /org/repo é caminho
            file_path = "/".join(path_parts[2:]) if len(path_parts) > 2 else None
            revision = "main"

    repo_error = validate_hf_repo_id(repo_id)
    if repo_error:
        raise ValueError(repo_error)
    revision = revision or "main"
    if file_path is not None:
        file_path = normalize_hf_file_path(file_path)
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


def _sibling_name(sibling: Any) -> str:
    """Nome do arquivo de uma entrada de lista de arquivos do Hub.

    huggingface_hub 1.31.0 expõe ``RepoSibling.rfilename`` (o atributo
    ``filename`` NUNCA existiu na lib — origem do bug 'RepoSibling object has
    no attribute filename'). Aceitamos também variantes defensivas ('filename',
    'path' e dicts) para tolerar outras formas de resposta da API.
    """
    for attribute in ("rfilename", "filename", "path"):
        value = getattr(sibling, attribute, None)
        if value:
            return str(value)
    if isinstance(sibling, dict):
        for key in ("rfilename", "filename", "path"):
            if sibling.get(key):
                return str(sibling[key])
    return ""


def redact_secrets(text: Any, secrets: Iterable[Optional[str]]) -> str:
    """Remove trechos de credenciais de mensagens exibidas/logadas (item 7)."""
    result = str(text or "")
    for secret in secrets:
        if secret and len(secret) >= 4:  # evita remover substrings triviais
            result = result.replace(secret, "***REDACTED***")
    return result


_HF_CACHE_ENV_BACKUP: dict[str, Optional[str]] = {}


def configure_hf_cache(cache_root: Optional[Path] = None) -> Path:
    """Redireciona o cache do huggingface_hub para um diretório temporário.

    DEVE ser chamado ANTES do primeiro import de huggingface_hub: as constantes
    HF_HOME/HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE são congeladas na importação do
    módulo (por isso inspect_environment lê a versão via importlib.metadata,
    sem importar a lib). Os valores originais são preservados para cleanup_hf_cache.
    """
    cache_dir = Path(cache_root) if cache_root else Path(tempfile.mkdtemp(prefix="hf_cache_"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for variable in HF_CACHE_ENV_VARS:
        _HF_CACHE_ENV_BACKUP.setdefault(variable, os.environ.get(variable))
        os.environ[variable] = str(cache_dir)
    return cache_dir


def cleanup_hf_cache(cache_dir: Optional[Path]) -> None:
    """Remove o cache temporário HF e restaura as variáveis de ambiente."""
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)
    for variable, original in list(_HF_CACHE_ENV_BACKUP.items()):
        if original is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = original
        _HF_CACHE_ENV_BACKUP.pop(variable, None)


def clean_hf_local_cache(directory: Path) -> None:
    """Remove '<directory>/.cache' (metadata local que hf_hub_download cria em local_dir).

    Sem isso, arquivos de cache do Hugging Face terminariam no staging e seriam
    enviados junto do dataset (o staging é publicado como diretório inteiro).
    """
    cache_dir = Path(directory) / ".cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)


def _prune_empty_dirs(root: Path) -> None:
    """Remove subdiretórios vazios deixados pelo download (ex.: aninhamentos do repo)."""
    root = Path(root)
    if not root.is_dir():
        return
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if directory != root:
            try:
                directory.rmdir()
            except OSError:
                pass


def _hf_list_repo_files(repo_id: str, revision: str = "main", token: Optional[str] = None) -> list[str]:
    """Lista os arquivos do repo via HfApi.list_repo_files (API dedicada, list[str]).

    Usa a API correta para listagem em vez de percorrer os RepoSibling de
    model_info — compatível com huggingface_hub 1.31.0 e sem depender da
    estrutura dos siblings. Não baixa nada.
    """
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
        files = api.list_repo_files(repo_id=repo_id, revision=revision, token=token)
        return [str(name) for name in (files or [])]
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "RepositoryNotFoundError":
            raise RuntimeError(f"Repositório não encontrado: {repo_id}")
        elif exc_name == "GatedRepoError":
            raise RuntimeError(
                f"Repositório gated (acesso restrito): {repo_id}\n"
                f"Configure HF_TOKEN nos Secrets do Colab com acesso a este repositório."
            )
        raise RuntimeError(f"Erro ao listar arquivos HF: {redact_secrets(exc, [token])}")


def _hf_file_size(repo_id: str, file_path: str, revision: str = "main", token: Optional[str] = None) -> int:
    """Obtém tamanho do arquivo HF em bytes com UMA chamada de metadata ao repo.

    Estratégia consciente (item 6): ``model_info(..., files_metadata=True)``
    confirma em uma única chamada que o arquivo pertence ao repo e devolve o
    tamanho em ``RepoSibling.size`` — o campo 'size' só é populado quando
    files_metadata=True, e o nome do arquivo é 'rfilename' (não 'filename') em
    huggingface_hub 1.31.0 (ver _sibling_name). Quando o size não vem nos
    siblings, cai para get_hf_file_metadata (HEAD no arquivo) como fallback,
    evitando round-trips redundantes no caso normal.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError("huggingface_hub não está instalado (execute !pip install huggingface_hub no Colab)")
    try:
        api = HfApi()
        info = api.model_info(repo_id, revision=revision, token=token, files_metadata=True)
        sibling = None
        for candidate in (info.siblings or []):
            if _sibling_name(candidate) == file_path:
                sibling = candidate
                break
        if sibling is None:
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path} em {repo_id}")
        size = getattr(sibling, "size", None)
        if size:
            return int(size)
        # Fallback: metadata direta do arquivo (alguns itens vêm sem size)
        from huggingface_hub import get_hf_file_metadata

        metadata = get_hf_file_metadata(api.hf_hub_url(repo_id, file_path, revision=revision), token=token)
        if metadata and getattr(metadata, "size", None):
            return int(metadata.size)
        raise RuntimeError(f"Tamanho indisponível para {file_path} em {repo_id}")
    except FileNotFoundError:
        raise
    except Exception as exc:
        exc_name = type(exc).__name__
        if exc_name == "RepositoryNotFoundError":
            raise RuntimeError(f"Repositório não encontrado: {repo_id}")
        elif exc_name == "GatedRepoError":
            raise RuntimeError(f"Repositório gated; HF_TOKEN sem acesso: {repo_id}")
        raise RuntimeError(f"Erro ao obter tamanho do arquivo HF: {redact_secrets(exc, [token])}")


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


def guess_category(resource_type: str) -> Optional[str]:
    """Mapeamento PURO resource_type -> categoria; None quando exige interação.

    None cobre os casos que precisam do usuário: tipo vazio (HF genérico),
    'checkpoint' (depende do destino do lote) e tipos desconhecidos. Usado
    pelo resolução/resumo sem prompts; classify_resource_type cuida da
    interação.
    """
    resource_type = str(resource_type or "").strip().lower()
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
    return None


def classify_resource_type(resource_type: str, input_fn=input, checkpoint_destination: Optional[str] = None) -> str:
    """Mapeia um resource_type para categoria do dataset, com fallback interativo.

    Para tipo vazio/desconhecido, SEMPRE pergunta ao usuário (usado para HF).
    O mapeamento puro (sem prompts) vive em guess_category; 'checkpoint'
    respeita checkpoint_destination quando pré-respondido.
    """
    resource_type = str(resource_type or "").strip().lower()
    if not resource_type:
        # Tipo não disponível (HF genérico)
        return normalize_category(input_fn(f"Categoria {CATEGORIES}: "))
    if resource_type == "checkpoint":
        if checkpoint_destination:
            return normalize_category(checkpoint_destination)
        choice = input_fn("Checkpoint: 1=checkpoints/ 2=diffusion_models/: ").strip().lower()
        return normalize_category({"1": "checkpoints", "2": "diffusion_models"}.get(choice, choice))
    guessed = guess_category(resource_type)
    if guessed:
        return guessed
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


def _hf_is_candidate_file(name: str) -> bool:
    """Arquivo de modelo elegível para seleção (extensão permitida, sem segmentos ocultos)."""
    lowered = str(name or "").lower()
    parts = Path(lowered).parts
    if any(part.startswith(".") for part in parts):
        return False
    return Path(lowered).suffix in HF_ALLOWED_EXTENSIONS


def resolve_hf_input(value: str, hf_token: Optional[str] = None, input_fn=input) -> list[dict[str, Any]]:
    """Fase 1 (metadados) de uma entrada HF: validação/listagem/metadata + prompts.

    NÃO baixa nada (download só na Fase 2).

    Caso A — ``hf:owner/repo`` (sem arquivo):
      1) valida acesso ao repo (list_repo_files);
      2) lista os arquivos filtrando por extensão relevante;
      3) apresenta e pede qual arquivo baixar (re-PERGUNTANDO até válida);
      4) pede categoria; 5) pede base_model.
    Caso B — ``hf:owner/repo/caminho/arquivo``:
      1) valida o caminho (feito no parse: traversal/absoluto/extensão);
      2) confirma existência + obtém tamanho (model_info files_metadata=True);
      3) pede categoria; 4) pede base_model.

    Categoria e base_model são SEMPRE perguntados para HF — não existe
    auto-classificação a partir de um repo genérico.
    """
    value = str(value or "").strip()
    if not _hf_api_available():
        print(
            "[WARN] huggingface_hub não está instalado; instale antes da fase de download: "
            "!pip install huggingface_hub"
        )
    repo_id, file_path, revision = parse_hf_input(value)

    # ---- Caso A: só o repo -> listar e pedir o arquivo ----
    if not file_path:
        files = _hf_list_repo_files(repo_id, revision=revision, token=hf_token)
        candidates = sorted(name for name in files if _hf_is_candidate_file(name))
        if not candidates:
            raise RuntimeError(
                f"Nenhum arquivo de modelo em {repo_id}; extensões aceitas: "
                f"{', '.join(sorted(HF_ALLOWED_EXTENSIONS))}"
            )
        print(f"Arquivos de modelo em {repo_id}:")
        for position, name in enumerate(candidates, start=1):
            print(f"  {position}: {name}")
        while True:
            choice = input_fn("Escolha o número do arquivo (ou caminho completo): ").strip()
            selected = None
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                selected = candidates[int(choice) - 1]
            elif choice in candidates:
                selected = choice
            if selected:
                file_path = selected
                break
            print(f"[WARN] Opção inválida: {choice or '(vazio)'}; escolha um número da lista ou o caminho exato.")
        # Pertence ao repo e é seguro/extensionável (pertence por construção: veio do list_repo_files)
        file_path = normalize_hf_file_path(file_path)

    # ---- Caso B (e A já escolhido): existência + tamanho ----
    file_path = normalize_hf_file_path(file_path)  # revalida defensivamente
    size = _hf_file_size(repo_id, file_path, revision=revision, token=hf_token)
    if size <= 0:
        raise RuntimeError(f"Tamanho inválido (0 bytes) para {file_path} em {repo_id}")
    filename = Path(file_path).name

    # SEMPRE perguntar categoria para HF (sem fallback automático)
    category = classify_resource_type("", input_fn=input_fn)

    # SEMPRE perguntar base_model para HF (não inferível de repo genérico)
    base_model = input_fn("Base model (ex: sdxl, krea2, etc): ").strip() or "unknown"

    # Mesmo formato de print usado para Civitai (+ tamanho para o resumo)
    print(
        f"Modelo: {repo_id} | "
        f"arquivo: {file_path} | "
        f"tipo: {category} | "
        f"base model: {base_model} | "
        f"tamanho: {format_size(size)}"
    )

    # Source URL (endpoint resolve — nunca contém o token)
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

    # Defesa (itens 8/9): o destino final precisa permanecer DENTRO do staging
    try:
        final_destination.resolve().relative_to(Path(staging_dir).resolve())
    except ValueError:
        raise RuntimeError(f"Destino fora do staging bloqueado: {final_destination}")

    # Remove o '.cache/huggingface' que hf_hub_download cria em local_dir e os
    # subdiretórios vazios do aninhamento do repo — o staging deve conter
    # somente o arquivo do dataset (item 8: cache controlado, nunca publicado).
    clean_hf_local_cache(destination_dir)
    _prune_empty_dirs(destination_dir)

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
    """Coleta pura da fila AIR/URL/HF: valida cada entrada via parser até 'done'.

    A validação é delegada a parse_input (mesma regra usada na resolução);
    'done' fecha a lista e nunca vira artefato. Sem downloads nem outros prompts.
    """
    pending: list[str] = []
    while True:
        value = input_fn("\nAIR/URL/HF (done para finalizar): ").strip()
        if value.lower() == "done":
            break
        if not value:
            print("[WARN] Entrada vazia")
            continue
        try:
            parse_input(value, len(pending) + 1)
        except ValueError as exc:
            print(f"[WARN] Entrada inválida: {exc}")
            continue
        pending.append(value)
    print(f"[INFO] Lista fechada com {len(pending)} item(ns).")
    return pending


@dataclass
class ParsedInput:
    """Saída PURA do parser (sem rede/prompts): entrada normalizada por provider."""

    provider: str  # "civitai" | "huggingface"
    original_input: str
    index: int = 0
    repo_id: Optional[str] = None      # HF
    file_path: Optional[str] = None    # HF
    revision: Optional[str] = None     # HF
    state: str = "PARSED"


@dataclass
class ResolvedArtifact:
    """Saída da Fase 1 (metadata resolver): 1 arquivo pronto para download."""

    provider: str                      # "civitai" | "huggingface"
    original_input: str
    index: int
    total: int
    filename: str
    size_bytes: int = 0
    category: Optional[str] = None     # HF: sempre na Fase 1; Civitai: quando inferível
    base_model: Optional[str] = None
    destination: Optional[str] = None  # "categoria/arquivo" no dataset
    state: str = "RESOLVED"            # -> READY_TO_DOWNLOAD -> DOWNLOADED
    # Hugging Face
    repo_id: Optional[str] = None
    file_path: Optional[str] = None
    revision: Optional[str] = None
    source_url: Optional[str] = None
    # Civitai (metadados crus do provider para o downloader)
    info: Optional[dict[str, Any]] = None
    resource_type: Optional[str] = None
    air: Optional[dict[str, Any]] = None

    @property
    def source(self) -> str:
        """Compat com o formato antigo das filas: 'hf' para Hugging Face."""
        return "hf" if self.provider == "huggingface" else "civitai"


@dataclass
class ResolutionFailure:
    """Entrada que falhou na Fase 1; motiva o aborto antes de tocar o dataset."""

    original_input: str
    index: int
    provider: str = "unknown"
    reason: str = ""        # mensagem contextual (exibida no resumo)
    technical: str = ""     # erro técnico original, já redigido
    state: str = "FAILED"


@dataclass
class ResolutionOutcome:
    """Resultado da Fase 1: TODOS os artefatos + falhas (nunca aborta no meio)."""

    artifacts: list[ResolvedArtifact] = field(default_factory=list)
    failures: list[ResolutionFailure] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.artifacts) + len(self.failures)

    @property
    def ok(self) -> bool:
        return bool(self.artifacts) and not self.failures


@dataclass
class DownloadFailure:
    """Entrada que falhou na Fase 2; motiva o aborto antes de publicar o dataset."""

    original_input: str
    index: int
    total: int
    reason: str = ""


@dataclass
class DownloadOutcome:
    """Resultado da Fase 2: itens baixados + falhas (tenta todos; publica só se zero falhas)."""

    items: list[DatasetFile] = field(default_factory=list)
    failures: list[DownloadFailure] = field(default_factory=list)
    resolved_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures


def print_download_failure_summary(outcome: DownloadOutcome) -> None:
    """Relatório de falha transacional da Fase 2 (sem publicação parcial)."""
    print("\n[DOWNLOAD FAILED]")
    print(f"{outcome.resolved_count} arquivo(s) resolvido(s)")
    print(f"{len(outcome.items)} arquivo(s) baixado(s)")
    print(f"{len(outcome.failures)} arquivo(s) com falha")
    print("\nNenhuma alteração no dataset foi publicada.")


def format_hf_resolution_error(value: str, repo_id: Optional[str], file_path: Optional[str], technical: str) -> str:
    """Mensagem contextual de falha HF: o erro técnico nunca é a única explicação (item 14)."""
    return (
        "Falha ao obter metadata do Hugging Face.\n"
        f"  Input:\n    {value}\n"
        f"  Repo:\n    {repo_id or '(não identificado)'}\n"
        f"  Arquivo:\n    {file_path or '(repo inteiro)'}\n"
        f"  Erro técnico:\n    {technical}\n"
        "  Ação:\n    nenhum download foi iniciado."
    )


def parse_input(value: str, index: int = 0) -> ParsedInput:
    """Etapa PARSER do pipeline: entrada de texto -> ParsedInput (sem rede/prompts).

    Roteia por provider: HF (prefixo hf:/URL huggingface.co) vs Civitai
    (urn:air:/URL civitai.com). 'done' é exclusivamente o marcador de fim da
    coleta (nunca vira artefato); entradas não reconhecidas são erro.
    """
    value = str(value or "").strip()
    if not value:
        raise ValueError("Entrada vazia")
    if value.lower() == "done":
        raise ValueError("'done' é o marcador de fim da coleta, não um input")
    if is_hf_input(value):
        repo_id, file_path, revision = parse_hf_input(value)
        return ParsedInput("huggingface", value, index, repo_id, file_path, revision)
    if value.lower().startswith("urn:air:"):
        _, error = validate_air(value)
        if error:
            raise ValueError(error)
    else:
        error = validate_civitai_url(value)
        if error:
            raise ValueError(error)
    return ParsedInput("civitai", value, index)


def _build_artifact(parsed: ParsedInput, info: dict[str, Any], total: int) -> ResolvedArtifact:
    """Converte a resposta específica do provider em um ResolvedArtifact comum (item 15)."""
    if parsed.provider == "huggingface":
        filename = str(info.get("filename") or Path(str(info.get("file_path") or "model.safetensors")).name)
        category = info.get("category")
        return ResolvedArtifact(
            provider="huggingface",
            original_input=parsed.original_input,
            index=parsed.index,
            total=total,
            filename=filename,
            size_bytes=int(info.get("size") or 0),
            category=category,
            base_model=info.get("base_model"),
            destination=f"{category}/{filename}" if category else None,
            repo_id=info.get("repo_id") or parsed.repo_id,
            file_path=info.get("file_path") or parsed.file_path,
            revision=info.get("revision") or parsed.revision or "main",
            source_url=info.get("source_url"),
            info=info,
        )
    air = info.get("air") or {}
    model = info.get("model") or {}
    file_info = info.get("file") or {}
    resource_type = str(air.get("type") or model.get("type") or "unknown")
    # Mapeamento puro (sem prompt): checkpoint/desconhecido ficam para
    # classify_resolved_artifacts, depois que o destino do lote for conhecido.
    category = guess_category(resource_type)
    filename = Path(file_info.get("name") or "model.safetensors").name
    base_model = air.get("base_model") or (info.get("version") or {}).get("baseModel")
    return ResolvedArtifact(
        provider="civitai",
        original_input=parsed.original_input,
        index=parsed.index,
        total=total,
        filename=filename,
        size_bytes=int(float(file_info.get("sizeKB") or 0) * 1024),
        category=category,
        base_model=base_model,
        destination=f"{category}/{filename}" if category else None,
        info=info,
        resource_type=resource_type,
        air=air or None,
    )


def resolve_queue_metadata(pending: list[str], token: str, input_fn=input, hf_token: Optional[str] = None) -> ResolutionOutcome:
    """Fase 1 (metadata resolver): resolve TODOS os inputs sem baixar nada.

    Nunca interrompe no meio: cada entrada vira um ResolvedArtifact ou uma
    ResolutionFailure com mensagem contextual. O orquestrador consulta
    outcome.failures e ABORTA antes de qualquer pergunta de edição/download
    quando houver ao menos uma falha (item 10: sem dataset parcial).
    """
    if hf_token is None:
        hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        print("[WARN] HF_TOKEN não configurado; repositórios HF privados/gated falharão.")

    outcome = ResolutionOutcome()
    total = len(pending)
    for index, value in enumerate(pending, start=1):
        print(f"[INFO] Resolvendo metadados [{index}/{total}]: {value}")
        try:
            parsed = parse_input(value, index)
        except Exception as exc:
            technical = redact_secrets(exc, [token, hf_token])
            outcome.failures.append(ResolutionFailure(value, index, "unknown", str(technical), technical))
            print(f"[ERROR] Falha ao resolver [{index}/{total}] {value}: {technical}")
            continue
        try:
            if parsed.provider == "huggingface":
                infos = resolve_hf_input(value, hf_token=hf_token, input_fn=input_fn)
            else:
                infos = resolve_civitai_input(value, token, input_fn)
        except Exception as exc:
            technical = redact_secrets(exc, [token, hf_token])
            if parsed.provider == "huggingface":
                reason = format_hf_resolution_error(value, parsed.repo_id, parsed.file_path, technical)
            else:
                reason = (
                    "Falha ao resolver metadata Civitai.\n"
                    f"  Input:\n    {value}\n"
                    f"  Erro técnico:\n    {technical}\n"
                    "  Ação:\n    nenhum download foi iniciado."
                )
            outcome.failures.append(ResolutionFailure(value, index, parsed.provider, reason, technical))
            print(f"[ERROR] Falha ao resolver [{index}/{total}] {value}: {technical}")
            continue
        for info in infos:
            outcome.artifacts.append(_build_artifact(parsed, info, total))
    if outcome.failures:
        print(
            f"[WARN] Resolução com {len(outcome.failures)} falha(s); "
            f"{len(outcome.artifacts)} arquivo(s) resolvido(s)."
        )
    return outcome


def print_resolution_summary(outcome: ResolutionOutcome) -> None:
    """INPUT RESOLUTION SUMMARY: exibido ANTES das perguntas de edição do dataset (item 17)."""
    print()
    print("=" * 60)
    print("INPUT RESOLUTION SUMMARY")
    print("=" * 60)
    for position, artifact in enumerate(outcome.artifacts, start=1):
        label = "Hugging Face" if artifact.provider == "huggingface" else "Civitai"
        print(f"\n[{position}] {label}")
        if artifact.provider == "huggingface":
            print(f"    Repo: {artifact.repo_id}")
            print(f"    File: {artifact.file_path}")
            print(f"    Revision: {artifact.revision}")
        else:
            print(f"    Input: {artifact.original_input}")
            print(f"    File: {artifact.filename}")
        print(f"    Size: {format_size(artifact.size_bytes)}")
        print(f"    Category: {artifact.category or '(será perguntado)'}")
        print(f"    Base model: {artifact.base_model or 'N/A'}")
        if artifact.destination:
            print(f"    Destination: {artifact.destination}")
    if outcome.failures:
        print("\nFAILED:")
        for position, failure in enumerate(outcome.failures, start=1):
            print(f"\n{position}. {failure.original_input}")
            for line in str(failure.reason).splitlines():
                print(f"    {line}")
    print()
    print("=" * 60)
    print(f"{len(outcome.artifacts)}/{outcome.total} inputs resolved")
    if outcome.failures:
        print(f"{len(outcome.failures)} input(s) FAILED.")
        print("Nenhum arquivo foi modificado.")
        print("Nenhum download foi iniciado.")
    else:
        print("Ready for download.")
    print("=" * 60)
    print()


def classify_resolved_artifacts(
    artifacts: list[ResolvedArtifact],
    input_fn=input,
    checkpoint_destination: Optional[str] = None,
) -> None:
    """Preenche category/destination dos artefatos ainda sem classificação (item 11).

    Roda ENTRE a resolução e o resumo/edições — nenhuma pergunta acontece
    durante o download. HF já sai da Fase 1 com categoria perguntada; Civitai
    preenche aqui os tipos que exigem interação (checkpoint/desconhecido),
    respeitando o checkpoint_destination único do lote.
    """
    for artifact in artifacts:
        if artifact.category is None and artifact.provider == "civitai":
            artifact.category = classify_civitai_type(
                artifact.resource_type or "",
                input_fn=input_fn,
                checkpoint_destination=checkpoint_destination,
            )
        if artifact.category:
            artifact.destination = f"{artifact.category}/{artifact.filename}"
            artifact.state = "READY_TO_DOWNLOAD"


def queue_contains_checkpoint(resolved: Iterable[Any]) -> bool:
    """Detecta 'checkpoint' no lote (aceita ResolvedArtifact ou dicts legados)."""
    for entry in resolved:
        resource_type = entry.get("resource_type") if isinstance(entry, dict) else getattr(entry, "resource_type", None)
        if str(resource_type or "").strip().lower() == "checkpoint":
            return True
    return False


def _queue_entry_view(entry: Any) -> dict[str, Any]:
    """Normaliza a entrada da fila (ResolvedArtifact novo | dict legado) para leitura única."""
    if isinstance(entry, ResolvedArtifact):
        info = entry.info or {}
        return {
            "value": entry.original_input,
            "index": entry.index,
            "total": entry.total,
            "info": info,
            "resource_type": entry.resource_type,
            "provider": entry.provider,
            "category": entry.category or info.get("category"),
            "base_model": entry.base_model or info.get("base_model"),
            "artifact": entry,
        }
    info = entry.get("info") or {}
    source = entry.get("source", "civitai")  # Default para compatibilidade com mocks de teste antigos
    return {
        "value": entry.get("value"),
        "index": entry.get("index", 0),
        "total": entry.get("total", 0),
        "info": info,
        "resource_type": entry.get("resource_type"),
        "provider": "huggingface" if source == "hf" else "civitai",
        "category": info.get("category"),
        "base_model": info.get("base_model"),
        "artifact": None,
    }


def download_resolved_queue(
    resolved: list[Any],
    staging_dir: Path,
    token: str,
    input_fn=input,
    checkpoint_destination: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> DownloadOutcome:
    """Fase 2 (downloader): baixa a fila já resolvida, sem nenhum prompt intermediário.

    Aceita ResolvedArtifact (formato novo) ou dicts legados. Categorias já
    vêm resolvidas da Fase 1/classificação; ``classify_civitai_type`` aqui é
    apenas fallback para entradas legadas sem categoria. ``checkpoint_destination``
    é a resposta única do lote para itens do tipo checkpoint.

    Tenta todos os itens (retries por item permanecem inalterados). Qualquer
    falha permanente é registrada em ``DownloadOutcome.failures``; o chamador
    NÃO deve publicar o dataset se ``outcome.ok`` for falso (publicação
    transacional: zero falhas ou nenhuma alteração publicada).
    """
    if hf_token is None:
        hf_token = get_secret("HF_TOKEN")

    print("[INFO] Iniciando downloads...")
    queue: list[DatasetFile] = []
    failures: list[DownloadFailure] = []
    last_value: Optional[str] = None
    resolved_count = len(resolved)
    for entry in resolved:
        view = _queue_entry_view(entry)
        value = view["value"]
        index, total = view["index"], view["total"]
        if value != last_value:
            print(f"--- [{index}/{total}] {value} ---")
            last_value = value
        info = view["info"]
        provider = view["provider"]

        try:
            if provider == "huggingface":
                # HF: categoria e base_model já perguntados na Fase 1
                category = view["category"] or info["category"]
                item = download_hf_file(
                    repo_id=info["repo_id"],
                    file_path=info["file_path"],
                    category=category,
                    staging_dir=staging_dir,
                    hf_token=hf_token,
                    revision=info.get("revision", "main"),
                    source_url=info.get("source_url"),
                    base_model=view["base_model"],
                )
            else:
                # Civitai
                air = info.get("air") or {}
                resource_type = view["resource_type"]
                category = view["category"] or classify_civitai_type(
                    resource_type, input_fn, checkpoint_destination=checkpoint_destination
                )
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
            artifact = view["artifact"]
            if artifact is not None:
                artifact.state = "DOWNLOADED"
            print(f"[{len(queue)}] Download concluído: {item.path}")
        except Exception as exc:
            reason = str(exc)
            failures.append(DownloadFailure(value, index, total, reason=reason))
            print(f"[ERROR] Download falhou para [{index}/{total}] {value}: {exc}")
            continue
    return DownloadOutcome(items=queue, failures=failures, resolved_count=resolved_count)


def download_input_queue(
    staging_dir: Path,
    token: str,
    input_fn=input,
    hf_token: Optional[str] = None,
) -> list[DatasetFile]:
    """Fila síncrona AIR/URL/HF: coleta -> parse/resolução -> classificação -> download.

    Mantida como atalho público equivalente; o orquestrador chama as etapas
    separadamente para antecipar todas as perguntas interativas. Se qualquer
    entrada falhar na resolução, imprime o resumo e aborta ANTES de qualquer
    download (item 10: nada de dataset parcial). Se qualquer download falhar,
    imprime o resumo transacional e aborta SEM retornar itens parciais.
    """
    pending = collect_input_queue(input_fn)
    outcome = resolve_queue_metadata(pending, token, input_fn, hf_token=hf_token)
    if outcome.failures:
        print_resolution_summary(outcome)
        raise RuntimeError(
            f"Resolução falhou para {len(outcome.failures)} entrada(s); "
            "nenhum download foi iniciado e o dataset não foi modificado."
        )
    if not outcome.artifacts:
        return []
    classify_resolved_artifacts(outcome.artifacts, input_fn=input_fn)
    download_outcome = download_resolved_queue(
        outcome.artifacts, staging_dir, token, input_fn, hf_token=hf_token
    )
    if download_outcome.failures:
        print_download_failure_summary(download_outcome)
        raise RuntimeError(
            f"Download falhou para {len(download_outcome.failures)} arquivo(s); "
            "nenhuma alteração no dataset foi publicada."
        )
    return download_outcome.items


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
        for path in sorted(
            p.relative_to(staging_dir).as_posix()
            for p in staging_dir.rglob("*")
            if p.is_file()
            and p.name not in {MANIFEST_NAME, METADATA_NAME}
            # Nunca inclui caches ocultos (.cache do kaggle/hf_hub_download)
            and not any(part.startswith(".") for part in p.relative_to(staging_dir).parts)
        ):
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
    # Remove caches ocultos (.cache do materialize e do hf_hub_download) antes
    # de publicar: o staging é enviado como diretório inteiro (itens 8/13).
    for cache_dir in list(staging_dir.rglob(".cache")):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
    write_manifest(manifest_path, dataset, desired)
    changes = compare_states(current, desired)
    print("\n" + render_preview(dataset, current, desired, changes))
    notes = input_fn("Notas da versão: ").strip() or "Update Dataset state"
    if input_fn("Publicar? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
        print("[CANCEL] Nenhuma alteração remota foi feita.")
        return None
    return publish(dataset, staging_dir, notes)
