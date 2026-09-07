#!/usr/bin/env python3
"""
Download de modelos da Civitai para staging local.
Uso: python civitai_download.py [--model-name NAME] [--model-version-id ID] [--file-id ID] [--staging-dir DIR]
"""

import os
import subprocess
import hashlib
import argparse
from pathlib import Path

DEFAULT_MODEL_NAME = "lustifyNSFWCheckpoint_v10Krea2.safetensors"
DEFAULT_MODEL_VERSION_ID = "3112728"
DEFAULT_FILE_ID = "2997637"
EXPECTED_GB = 11.94


def calculate_sha256(filepath: Path) -> str:
    """Calcula SHA256 do arquivo."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    model_name: str = DEFAULT_MODEL_NAME,
    model_version_id: str = DEFAULT_MODEL_VERSION_ID,
    file_id: str = DEFAULT_FILE_ID,
    staging_dir: Path = Path("/content/kaggle_staging"),
    expected_gb: float = EXPECTED_GB,
    token: str = None,
) -> Path:
    """
    Baixa modelo da Civitai para staging.
    Retorna o Path do arquivo baixado.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    model_path = staging_dir / model_name

    # Verificar se já existe e é válido
    if model_path.exists():
        size = model_path.stat().st_size
        size_gb = size / (1024**3)
        print(f"[INFO] Arquivo já existe: {size:,} bytes ({size_gb:.2f} GB)")
        if abs(size_gb - expected_gb) < 0.5 and size > 0:
            print(f"[INFO] ✅ Tamanho válido, pulando download")
            print(f"[INFO] SHA256: {calculate_sha256(model_path)}")
            return model_path
        else:
            print(f"[WARN] Tamanho inválido ({size_gb:.2f} GB), removendo e rebaixando")
            model_path.unlink()

    # Obter token (Colab Secrets via userdata + fallback env)
    if token is None:
        token = os.environ.get("CIVITAI_TOKEN") or os.environ.get("CIVITAI_API_KEY")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get('CIVITAI_TOKEN')
        except (ImportError, userdata.NotebookAccessError):
            pass
    if not token:
        raise ValueError("CIVITAI_TOKEN não configurado nas variáveis de ambiente/Secrets")

    url = f"https://civitai.com/api/download/models/{model_version_id}?fileId={file_id}"
    print(f"[INFO] Baixando de: {url}")
    print(f"[INFO] Salvando em: {model_path}")

    # curl com retry, resume, follow redirects
    cmd = [
        "curl",
        "-L",                    # Follow redirects
        "-C", "-",               # Resume if partial
        "--retry", "3",          # Retry 3 times
        "--retry-delay", "5",    # Wait 5s between retries
        "--fail",                # Fail on HTTP errors
        "-H", f"Authorization: Bearer {token}",
        "-o", str(model_path),
        url
    ]

    print(f"[INFO] Executando: {' '.join(cmd[:-1])} [URL]")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    print(f"[INFO] Return code: {result.returncode}")
    if result.stdout:
        print(f"[INFO] STDOUT: {result.stdout[-500:]}")
    if result.stderr:
        print(f"[INFO] STDERR: {result.stderr[-500:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Download falhou com código {result.returncode}: {result.stderr}")

    # Validar tamanho
    size = model_path.stat().st_size
    size_gb = size / (1024**3)
    print(f"[INFO] ✅ Download concluído: {size:,} bytes ({size_gb:.2f} GB)")

    if size == 0:
        model_path.unlink()
        raise RuntimeError("Arquivo baixado tem 0 bytes")

    if abs(size_gb - expected_gb) > 0.5:
        print(f"[WARN] AVISO: Tamanho ({size_gb:.2f} GB) diverge do esperado ({expected_gb} GB)")
    else:
        print(f"[INFO] ✅ Tamanho compatível com esperado")

    print(f"[INFO] SHA256: {calculate_sha256(model_path)}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Download modelo da Civitai")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version-id", default=DEFAULT_MODEL_VERSION_ID)
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--staging-dir", default="/content/kaggle_staging")
    parser.add_argument("--expected-gb", type=float, default=EXPECTED_GB)
    parser.add_argument("--token", help="Token Civitai (opcional, usa env CIVITAI_TOKEN)")

    args = parser.parse_args()

    try:
        path = download_model(
            model_name=args.model_name,
            model_version_id=args.model_version_id,
            file_id=args.file_id,
            staging_dir=Path(args.staging_dir),
            expected_gb=args.expected_gb,
            token=args.token,
        )
        print(f"\n[SUCCESS] Modelo salvo em: {path}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()