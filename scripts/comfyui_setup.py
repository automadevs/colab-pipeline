#!/usr/bin/env python3
"""Setup do ComfyUI no Kaggle Notebook (SSD local)."""
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_COMFYUI_DIR = Path("/kaggle/working/ComfyUI")
DEFAULT_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
MODEL_CATEGORIES = ["checkpoints", "diffusion_models", "loras", "vae", "text_encoders", "clip", "controlnet", "upscale_models", "video_models", "embeddings"]


def detect_gpu() -> dict:
    info = {"has_gpu": False, "gpu_name": "Unknown", "vram_gb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[INFO] GPU detectada: {info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM)")
            return info
    except Exception as exc:
        print(f"[WARN] PyTorch GPU detection falhou: {exc}")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[0]
            name, mem = [x.strip() for x in line.split(",", 1)]
            info["has_gpu"] = True
            info["gpu_name"] = name
            info["vram_gb"] = int(mem.split()[0]) / 1024
            print(f"[INFO] GPU detectada via nvidia-smi: {name} ({info['vram_gb']:.1f} GB VRAM)")
    except Exception as exc:
        print(f"[WARN] nvidia-smi indisponível: {exc}")
    return info


def _run(cmd, cwd=None, timeout=900, check=True):
    print("[CMD]", " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr[-4000:])
        if check:
            raise RuntimeError(f"Comando falhou ({result.returncode}): {' '.join(map(str, cmd))}")
    return result


def setup_comfyui(comfyui_dir=DEFAULT_COMFYUI_DIR, repo_url=DEFAULT_REPO_URL, custom_nodes=None, models_dir=None, output_dir=None, drive_base=DEFAULT_DRIVE_BASE):
    comfyui_dir = Path(comfyui_dir)
    models_dir = Path(models_dir) if models_dir else comfyui_dir / "models"
    
    # O output do ComfyUI deve ser SEMPRE local no SSD (ex: /kaggle/working/ComfyUI/output).
    # O Google Drive NÃO fica no caminho crítico da geração e é usado apenas para persistência/sync.
    if output_dir is None:
        output_dir = comfyui_dir / "output"
    else:
        output_dir = Path(output_dir)
    
    gpu_info = detect_gpu()

    # Verificar estado do diretório ComfyUI
    is_git_repo = (comfyui_dir / ".git").exists()
    has_main = (comfyui_dir / "main.py").exists()

    if not comfyui_dir.exists():
        # Caso 1: Diretório não existe → git clone
        _run(["git", "clone", "--depth", "1", repo_url, str(comfyui_dir)], timeout=900)
    elif is_git_repo:
        # Caso 2: Diretório existe e é um repo Git válido → git pull --ff-only
        _run(["git", "pull", "--ff-only"], cwd=comfyui_dir, timeout=300, check=False)
    else:
        # Caso 3: Diretório existe mas NÃO é um checkout Git válido
        raise RuntimeError(
            f"Diretório {comfyui_dir} existe mas não é um checkout Git válido do ComfyUI "
            f"(falta .git ou main.py). Remova ou renomeie este diretório antes de executar o setup."
        )
    # Criar output_dir APÓS o bootstrap do repositório para evitar conflitos
    output_dir.mkdir(parents=True, exist_ok=True)

    req = comfyui_dir / "requirements.txt"
    if req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=900)

    for cat in MODEL_CATEGORIES:
        (models_dir / cat).mkdir(parents=True, exist_ok=True)

    if custom_nodes:
        custom_dir = comfyui_dir / "custom_nodes"
        custom_dir.mkdir(parents=True, exist_ok=True)
        for spec in custom_nodes:
            if "@" in spec:
                repo_part, branch = spec.rsplit("@", 1)
            else:
                repo_part, branch = spec, "main"
            repo = repo_part if repo_part.startswith("http") else f"https://github.com/{repo_part}.git"
            node_name = repo.rstrip("/").split("/")[-1].removesuffix(".git")
            node_path = custom_dir / node_name
            if node_path.exists():
                _run(["git", "pull", "--ff-only"], cwd=node_path, timeout=300, check=False)
            else:
                _run(["git", "clone", "--depth", "1", "--branch", branch, repo, str(node_path)], timeout=900)
            node_req = node_path / "requirements.txt"
            if node_req.exists():
                _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req)], timeout=900, check=False)

    # ComfyUI espera um mapping YAML, não uma lista de mappings.
    extra_paths = comfyui_dir / "extra_model_paths.yaml"
    if not extra_paths.exists():
        lines = ["kaggle_models:", f"  base_path: {models_dir}"]
        for cat in MODEL_CATEGORIES:
            lines.append(f"  {cat}: {cat}")
        extra_paths.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] Criado {extra_paths}")

    print(f"[INFO] ComfyUI: {comfyui_dir}")
    print(f"[INFO] Models: {models_dir}")
    print(f"[INFO] GPU: {gpu_info['gpu_name'] if gpu_info['has_gpu'] else 'CPU only'}")
    return comfyui_dir


def start_comfyui(comfyui_dir=DEFAULT_COMFYUI_DIR, host="0.0.0.0", port=8188, extra_args=None, output_dir=None):
    main_py = Path(comfyui_dir) / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"ComfyUI não encontrado em {comfyui_dir}")
    
    if output_dir is None:
        output_dir = Path(comfyui_dir) / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "--listen", host, "--port", str(port), "--output-directory", str(output_dir)]
    if extra_args:
        cmd.extend(extra_args)
    
    log_path = Path(comfyui_dir) / "comfyui.log"
    log = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(cmd, cwd=comfyui_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[INFO] ComfyUI iniciado PID={proc.pid}; log={log_path}")
    print(f"[INFO] Output directory (SSD local): {output_dir}")
    return proc


def health_check(host="127.0.0.1", port=8188, timeout=60):
    import urllib.request
    start = time.time()
    url = f"http://{host}:{port}/system_stats"
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print("[INFO] Health check OK")
                    return True
        except Exception:
            time.sleep(2)
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-dir", default=str(DEFAULT_COMFYUI_DIR))
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--custom-nodes", nargs="*")
    parser.add_argument("--models-dir")
    parser.add_argument("--output-dir", help="Diretório de outputs (padrão: ComfyUI/output no SSD local)")
    parser.add_argument("--drive-base", default=DEFAULT_DRIVE_BASE, help="Pasta base no Google Drive")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--health-check", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    comfyui = setup_comfyui(
        Path(args.comfyui_dir), 
        args.repo_url, 
        args.custom_nodes, 
        Path(args.models_dir) if args.models_dir else None,
        output_dir=output_dir,
        drive_base=args.drive_base
    )
    if args.start:
        proc = start_comfyui(comfyui, args.host, args.port, output_dir=output_dir)
        if args.health_check and not health_check("127.0.0.1", args.port, 90):
            raise RuntimeError("Health check falhou")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()

if __name__ == "__main__":
    main()
