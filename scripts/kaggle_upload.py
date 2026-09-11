#!/usr/bin/env python3
"""
Upload de modelos para Kaggle Dataset.
Uso: python kaggle_upload.py [--model-name NAME] [--staging-dir DIR] [--dataset DATASET] [--method cli|kagglehub]
"""

import os
import json
import subprocess
import tempfile
import shutil
import argparse
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


DEFAULT_DATASET = resolve_dataset_name()
DEFAULT_MODEL_NAME = "lustifyNSFWCheckpoint_v10Krea2.safetensors"


def upload_via_cli(
    model_path: Path,
    model_name: str,
    dataset: str,
    version_notes: str = None,
) -> bool:
    """Upload via Kaggle CLI oficial."""
    if version_notes is None:
        version_notes = f"Add {model_name}"

    print(f"[INFO] === UPLOAD VIA KAGGLE CLI ===")
    print(f"[INFO] Dataset: {dataset}")
    print(f"[INFO] Arquivo: {model_name} ({model_path.stat().st_size / (1024**3):.2f} GB)")

    # Verificar permissão
    print("[INFO] Verificando permissões no dataset...")
    result = subprocess.run(
        ["kaggle", "datasets", "metadata", dataset],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        if "403" in result.stderr:
            raise PermissionError(f"Sem permissão de leitura no dataset: {result.stderr}")
        raise RuntimeError(f"Dataset inacessível: {result.stderr}")
    print("[INFO] ✅ Dataset acessível")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copiar modelo
        dest = tmpdir / model_name
        print(f"[INFO] Copiando para diretório temporário...")
        shutil.copy2(model_path, dest)
        assert dest.stat().st_size == model_path.stat().st_size, "Tamanho divergiu na cópia!"

        # Baixar metadata atual
        metadata_file = tmpdir / "dataset-metadata.json"
        result = subprocess.run(
            ["kaggle", "datasets", "metadata", dataset, "-p", str(tmpdir)],
            capture_output=True, text=True
        )

        if result.returncode == 0 and metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"[INFO] Metadata atual carregado: {metadata.get('title', 'N/A')}")
            print(f"[INFO] Arquivos atuais no metadata: {len(metadata.get('resources', []))}")
        else:
            print("[INFO] Não foi possível baixar metadata, criando novo...")
            metadata = {
                "title": "comfydocs",
                "id": dataset,
                "licenses": [{"name": "CC0-1.0"}],
                "resources": []
            }
        
        # Garantir que o ID do dataset esteja correto (obrigatório para version)
        metadata["id"] = dataset

        # Adicionar arquivo se não existe
        existing_files = {r["path"] for r in metadata.get("resources", [])}
        if model_name not in existing_files:
            metadata.setdefault("resources", []).append({
                "path": model_name,
                "description": "Civitai checkpoint FP8 ~12GB"
            })
            print(f"[INFO] Adicionado {model_name} ao metadata")
        else:
            print(f"[INFO] {model_name} já está no metadata")

        # Salvar metadata atualizado
        metadata["id"] = dataset  # Garantir ID correto
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload - criar nova versão
        print("[INFO] Enviando nova versão...")
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(tmpdir),
            "-m", version_notes,
            "-r", "zip"
        ]
        print(f"[INFO] Comando: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
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
            return_code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError("Timeout publicando nova versão Kaggle")

        print(f"[INFO] Return code: {return_code}")
        combined_output = "\n".join(output)
        if return_code == 0:
            print("[INFO] ✅ SUCESSO: Nova versão criada via Kaggle CLI")
            return True
        elif "403" in combined_output:
            print("[ERROR] ❌ ERRO 403 DETECTADO")
            raise PermissionError("CLI retornou 403")
        else:
            raise RuntimeError(f"Falha no upload (exit {return_code}): {combined_output[-2000:]}")


def upload_via_kagglehub(
    model_path: Path,
    model_name: str,
    dataset: str,
    version_notes: str = None,
) -> bool:
    """Upload via kagglehub (fallback)."""
    if version_notes is None:
        version_notes = f"Add {model_name} via kagglehub"

    print(f"[INFO] === UPLOAD VIA KAGGLEHUB ===")

    try:
        import kagglehub
    except ImportError:
        print("[INFO] Instalando kagglehub...")
        subprocess.run(["pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub

    if not hasattr(kagglehub, 'dataset_upload'):
        raise NotImplementedError("kagglehub.dataset_upload não disponível nesta versão")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dest = tmpdir / model_name
        shutil.copy2(model_path, dest)

        print(f"[INFO] Enviando via kagglehub.dataset_upload...")
        result = kagglehub.dataset_upload(
            handle=dataset,
            local_dataset_dir=str(tmpdir),
            version_notes=version_notes
        )
        print(f"[INFO] ✅ Upload via kagglehub bem-sucedido: {result}")
        return True


def upload_model(
    model_name: str = DEFAULT_MODEL_NAME,
    staging_dir: Path = Path("/content/kaggle_staging"),
    dataset: str = DEFAULT_DATASET,
    method: str = "cli",
    version_notes: str = None,
) -> bool:
    """
    Upload modelo para Kaggle Dataset.
    Tenta CLI primeiro, fallback para kagglehub se 403.
    """
    model_path = staging_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    size = model_path.stat().st_size
    size_gb = size / (1024**3)
    print(f"[INFO] Modelo a enviar: {model_name} ({size:,} bytes / {size_gb:.2f} GB)")

    if size == 0:
        raise ValueError("Arquivo tem 0 bytes, não será enviado")

    if method == "cli":
        try:
            return upload_via_cli(model_path, model_name, dataset, version_notes)
        except PermissionError as e:
            print(f"[WARN] CLI falhou com 403: {e}")
            print("[INFO] Tentando fallback kagglehub...")
            return upload_via_kagglehub(model_path, model_name, dataset, version_notes)
    else:
        return upload_via_kagglehub(model_path, model_name, dataset, version_notes)


def main():
    parser = argparse.ArgumentParser(description="Upload modelo para Kaggle Dataset")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--staging-dir", default="/content/kaggle_staging")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--method", choices=["cli", "kagglehub"], default="cli")
    parser.add_argument("--version-notes", help="Notas da versão")

    args = parser.parse_args()

    try:
        upload_model(
            model_name=args.model_name,
            staging_dir=Path(args.staging_dir),
            dataset=args.dataset,
            method=args.method,
            version_notes=args.version_notes,
        )
        print("\n[SUCCESS] Upload concluído")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()