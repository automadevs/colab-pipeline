#!/usr/bin/env python3
"""
Detecção de GPU no Kaggle Notebook.
Uso: python gpu_detect.py
"""

import subprocess
import sys


def detect_gpu() -> dict:
    """Detecta GPU disponível de forma robusta."""
    info = {
        "has_gpu": False,
        "gpu_name": "Unknown",
        "vram_gb": 0,
        "cuda_version": None,
        "driver_version": None,
        "torch_cuda": False,
    }

    # 1. Tentar via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]  # Primeira GPU
            name, mem, driver = [x.strip() for x in line.split(", ")]
            info["has_gpu"] = True
            info["gpu_name"] = name
            info["vram_gb"] = int(mem.replace(" MiB", "")) / 1024
            info["driver_version"] = driver
            print(f"[INFO] GPU via nvidia-smi: {name} ({info['vram_gb']:.1f} GB VRAM, Driver {driver})")
    except Exception as e:
        print(f"[INFO] nvidia-smi não disponível: {e}")

    # 2. Tentar via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["torch_cuda"] = True
            info["cuda_version"] = torch.version.cuda
            device_props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = device_props.name
            info["vram_gb"] = device_props.total_memory / (1024**3)
            print(f"[INFO] GPU via PyTorch: {info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM, CUDA {info['cuda_version']})")
        else:
            print("[INFO] PyTorch: CUDA não disponível")
    except ImportError:
        print("[INFO] PyTorch não instalado")

    # 3. Tentar via TensorFlow
    if not info["has_gpu"]:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info["has_gpu"] = True
                info["gpu_name"] = str(gpus[0])
                print(f"[INFO] GPU via TensorFlow: {gpus[0]}")
        except ImportError:
            pass
        except Exception:
            pass

    # 4. Verificar variável de ambiente Kaggle
    kaggle_gpu = os.environ.get("KAGGLE_GPU")
    if kaggle_gpu:
        print(f"[INFO] KAGGLE_GPU: {kaggle_gpu}")

    return info


def print_gpu_summary(info: dict):
    """Imprime resumo amigável da GPU."""
    print("\n" + "=" * 50)
    print("RESUMO DE GPU")
    print("=" * 50)

    if info["has_gpu"]:
        print(f"✅ GPU DISPONÍVEL")
        print(f"   Nome: {info['gpu_name']}")
        print(f"   VRAM: {info['vram_gb']:.1f} GB")
        if info["cuda_version"]:
            print(f"   CUDA: {info['cuda_version']}")
        if info["driver_version"]:
            print(f"   Driver: {info['driver_version']}")
        if info["torch_cuda"]:
            print(f"   PyTorch CUDA: ✅")
    else:
        print(f"❌ GPU NÃO DETECTADA")
        print(f"   Rodando em CPU apenas")

    print("=" * 50)


def recommend_batch_size(vram_gb: float) -> dict:
    """Recomenda batch sizes baseados na VRAM."""
    if vram_gb >= 24:
        return {"sd15": 4, "sdxl": 2, "flux": 1}
    elif vram_gb >= 16:
        return {"sd15": 2, "sdxl": 1, "flux": 1}
    elif vram_gb >= 12:
        return {"sd15": 1, "sdxl": 1, "flux": 1}
    elif vram_gb >= 8:
        return {"sd15": 1, "sdxl": "offload", "flux": "offload"}
    else:
        return {"sd15": "offload", "sdxl": "offload", "flux": "offload"}


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Detectar GPU no Kaggle Notebook")
    parser.add_argument("--json", action="store_true", help="Output em JSON")
    parser.add_argument("--recommend", action="store_true", help="Mostrar recomendações de batch size")

    args = parser.parse_args()

    info = detect_gpu()

    if args.json:
        import json
        print(json.dumps(info, indent=2))
    else:
        print_gpu_summary(info)

        if args.recommend and info["has_gpu"]:
            recs = recommend_batch_size(info["vram_gb"])
            print(f"\nRECOMENDAÇÕES DE BATCH SIZE:")
            for model, bs in recs.items():
                print(f"  {model}: {bs}")

    # Exit code para uso em scripts
    exit(0 if info["has_gpu"] else 1)


if __name__ == "__main__":
    main()