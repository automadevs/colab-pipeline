#!/usr/bin/env python3
"""
Detecção de GPU no Kaggle Notebook.

Enumera TODAS as GPUs (ex.: T4x2 → gpu_count=2). Cada GPU tem VRAM própria —
NÃO some VRAMs. ComfyUI padrão usa COMFYUI_CUDA_DEVICE=0; a segunda GPU
permanece disponível para workflows especializados (SelectModelDevice etc.).

Uso: python gpu_detect.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Optional


def parse_nvidia_smi_csv(stdout: str) -> list[dict[str, Any]]:
    """
    Parseia saída CSV de nvidia-smi (uma linha por GPU).

    Esperado: name, memory.total, driver_version
    Ex.: "Tesla T4, 15360 MiB, 535.104.05"
    """
    gpus: list[dict[str, Any]] = []
    if not stdout or not stdout.strip():
        return gpus

    for raw in stdout.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        mem_token = parts[1].replace("MiB", "").replace("MB", "").strip().split()[0]
        try:
            vram_mib = int(float(mem_token))
        except ValueError:
            continue
        driver = parts[2] if len(parts) > 2 else None
        gpus.append({
            "index": len(gpus),
            "name": name,
            "vram_gb": round(vram_mib / 1024.0, 2),
            "vram_mib": vram_mib,
            "cuda": None,
            "driver": driver,
        })
    return gpus


def _gpus_from_torch() -> tuple[list[dict[str, Any]], Optional[str], bool]:
    """Retorna (gpus, cuda_version, torch_cuda_available)."""
    try:
        import torch
    except ImportError:
        return [], None, False

    if not torch.cuda.is_available():
        return [], getattr(torch.version, "cuda", None), False

    cuda_version = torch.version.cuda
    gpus: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        vram_mib = int(props.total_memory / (1024 ** 2))
        gpus.append({
            "index": index,
            "name": props.name,
            "vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "vram_mib": vram_mib,
            "cuda": cuda_version,
            "driver": None,
        })
    return gpus, cuda_version, True


def _merge_gpu_lists(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mescla listas por índice; primary tem prioridade, secondary preenche campos vazios."""
    if not primary:
        return list(secondary)
    if not secondary:
        return list(primary)

    count = max(len(primary), len(secondary))
    merged: list[dict[str, Any]] = []
    for i in range(count):
        a = primary[i] if i < len(primary) else {}
        b = secondary[i] if i < len(secondary) else {}
        merged.append({
            "index": i,
            "name": a.get("name") or b.get("name") or "Unknown",
            "vram_gb": a.get("vram_gb") if a.get("vram_gb") is not None else b.get("vram_gb", 0),
            "vram_mib": a.get("vram_mib") if a.get("vram_mib") is not None else b.get("vram_mib", 0),
            "cuda": a.get("cuda") or b.get("cuda"),
            "driver": a.get("driver") or b.get("driver"),
        })
    return merged


def detect_gpu() -> dict[str, Any]:
    """
    Detecta e enumera todas as GPUs disponíveis.

    Retorno:
      has_gpu, gpu_count, gpus[{index, name, vram_gb, vram_mib, cuda, driver}],
      mais campos de compatibilidade (gpu_name, vram_gb da GPU 0).
    """
    info: dict[str, Any] = {
        "has_gpu": False,
        "gpu_count": 0,
        "gpus": [],
        "gpu_name": "Unknown",
        "vram_gb": 0,
        "cuda_version": None,
        "driver_version": None,
        "torch_cuda": False,
    }

    smi_gpus: list[dict[str, Any]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            smi_gpus = parse_nvidia_smi_csv(result.stdout)
            print(f"[INFO] nvidia-smi: {len(smi_gpus)} GPU(s) enumerada(s)")
            for g in smi_gpus:
                print(
                    f"  [{g['index']}] {g['name']} — {g['vram_gb']:.1f} GB VRAM"
                    + (f", Driver {g['driver']}" if g.get("driver") else "")
                )
    except Exception as exc:
        print(f"[INFO] nvidia-smi não disponível: {exc}")

    torch_gpus, cuda_version, torch_cuda = _gpus_from_torch()
    info["torch_cuda"] = torch_cuda
    info["cuda_version"] = cuda_version
    if torch_gpus:
        print(f"[INFO] PyTorch: {len(torch_gpus)} GPU(s), CUDA {cuda_version}")
        for g in torch_gpus:
            print(f"  [{g['index']}] {g['name']} — {g['vram_gb']:.1f} GB VRAM")

    # Preferir nvidia-smi para nome/driver/VRAM; PyTorch preenche CUDA
    gpus = _merge_gpu_lists(smi_gpus, torch_gpus)

    # Fallback TensorFlow somente se nada encontrado
    if not gpus:
        try:
            import tensorflow as tf

            tf_gpus = tf.config.list_physical_devices("GPU")
            for index, dev in enumerate(tf_gpus):
                gpus.append({
                    "index": index,
                    "name": str(dev),
                    "vram_gb": 0,
                    "vram_mib": 0,
                    "cuda": None,
                    "driver": None,
                })
            if gpus:
                print(f"[INFO] TensorFlow: {len(gpus)} GPU(s)")
        except Exception:
            pass

    kaggle_gpu = os.environ.get("KAGGLE_GPU")
    if kaggle_gpu:
        print(f"[INFO] KAGGLE_GPU: {kaggle_gpu}")

    if gpus:
        # Propagar cuda_version para todas as entradas se ausente
        for g in gpus:
            if not g.get("cuda") and cuda_version:
                g["cuda"] = cuda_version
        info["has_gpu"] = True
        info["gpu_count"] = len(gpus)
        info["gpus"] = gpus
        info["gpu_name"] = gpus[0]["name"]
        info["vram_gb"] = gpus[0].get("vram_gb") or 0
        info["driver_version"] = gpus[0].get("driver")
        if info["cuda_version"] is None:
            info["cuda_version"] = gpus[0].get("cuda")

    return info


def print_gpu_summary(info: dict[str, Any]) -> None:
    """Imprime resumo amigável da(s) GPU(s)."""
    print("\n" + "=" * 50)
    print("RESUMO DE GPU")
    print("=" * 50)

    if info.get("has_gpu"):
        print("GPU DISPONIVEL")
        print(f"   Contagem: {info.get('gpu_count', 0)}")
        print("   Nota: VRAM por GPU (nao somar). Kaggle T4x2 = 2x ~15GB.")
        for g in info.get("gpus") or []:
            print(f"   [{g['index']}] {g['name']}")
            print(f"       VRAM: {g.get('vram_gb', 0):.1f} GB")
            if g.get("cuda"):
                print(f"       CUDA: {g['cuda']}")
            if g.get("driver"):
                print(f"       Driver: {g['driver']}")
        if info.get("torch_cuda"):
            print("   PyTorch CUDA: sim")
        print("   ComfyUI padrao: COMFYUI_CUDA_DEVICE=0 (GPU 0).")
        print("   GPU 1 permanece disponivel para workflows especializados.")
    else:
        print("GPU NAO DETECTADA")
        print("   Rodando em CPU apenas")

    print("=" * 50)


def recommend_batch_size(vram_gb: float) -> dict[str, Any]:
    """Recomenda batch sizes com base na VRAM de UMA GPU (nao some placas)."""
    if vram_gb >= 24:
        return {"sd15": 4, "sdxl": 2, "flux": 1}
    if vram_gb >= 16:
        return {"sd15": 2, "sdxl": 1, "flux": 1}
    if vram_gb >= 12:
        return {"sd15": 1, "sdxl": 1, "flux": 1}
    if vram_gb >= 8:
        return {"sd15": 1, "sdxl": "offload", "flux": "offload"}
    return {"sd15": "offload", "sdxl": "offload", "flux": "offload"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detectar GPU(s) no Kaggle Notebook")
    parser.add_argument("--json", action="store_true", help="Output em JSON")
    parser.add_argument("--recommend", action="store_true", help="Mostrar recomendações de batch size")
    args = parser.parse_args()

    info = detect_gpu()

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print_gpu_summary(info)
        if args.recommend and info.get("has_gpu"):
            # Recomenda com base na GPU 0 (dispositivo padrão do ComfyUI)
            vram = (info.get("gpus") or [{}])[0].get("vram_gb") or info.get("vram_gb") or 0
            recs = recommend_batch_size(float(vram))
            print("\nRECOMENDACOES DE BATCH SIZE (GPU 0):")
            for model, bs in recs.items():
                print(f"  {model}: {bs}")

    sys.exit(0 if info.get("has_gpu") else 1)


if __name__ == "__main__":
    main()
