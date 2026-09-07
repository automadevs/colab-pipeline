#!/usr/bin/env python3
"""
Setup do ComfyUI no Kaggle Notebook (SSD local).
Uso: python comfyui_setup.py [--comfyui-dir DIR] [--repo-url URL] [--custom-nodes LIST]
"""

import subprocess
import sys
import os
import json
from pathlib import Path

DEFAULT_COMFYUI_DIR = Path("/kaggle/working/ComfyUI")
DEFAULT_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"


def detect_gpu() -> dict:
    """Detecta GPU disponível."""
    info = {"has_gpu": False, "gpu_name": "Unknown", "vram_gb": 0}

    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU detectada: {info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM)")
        else:
            print("[WARN] CUDA não disponível")
    except ImportError:
        print("[INFO] PyTorch não instalado, tentando nvidia-smi...")
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                line = result.stdout.strip()
                name, mem = line.split(", ")
                info["has_gpu"] = True
                info["gpu_name"] = name
                info["vram_gb"] = int(mem.replace(" MiB", "")) / 1024
                print(f"[INFO] GPU detectada via nvidia-smi: {info['gpu_name']} ({info['vram_gb']:.1f} GB VRAM)")
        except Exception:
            print("[WARN] Não foi possível detectar GPU")

    return info


def setup_comfyui(
    comfyui_dir: Path = DEFAULT_COMFYUI_DIR,
    repo_url: str = DEFAULT_REPO_URL,
    custom_nodes: list = None,
    models_dir: Path = None,
) -> Path:
    """
    Instala/atualiza ComfyUI e configura diretórios.
    Retorna o caminho do diretório ComfyUI.
    """
    print(f"[INFO] === SETUP COMFYUI ===")
    print(f"[INFO] Diretório: {comfyui_dir}")

    # Detectar GPU
    gpu_info = detect_gpu()

    # Clonar ou atualizar repositório
    main_py = comfyui_dir / "main.py"
    if main_py.exists():
        print("[INFO] ComfyUI já instalado, atualizando...")
        result = subprocess.run(
            ["git", "-C", str(comfyui_dir), "pull"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[WARN] Git pull falhou: {result.stderr}")
    else:
        print("[INFO] Clonando ComfyUI...")
        result = subprocess.run(
            ["git", "clone", repo_url, str(comfyui_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Falha ao clonar ComfyUI: {result.stderr}")
        print("[INFO] ✅ ComfyUI clonado")

    # Instalar dependências
    requirements = comfyui_dir / "requirements.txt"
    if requirements.exists():
        print("[INFO] Instalando dependências Python...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"[WARN] Pip install teve avisos: {result.stderr}")
        print("[INFO] ✅ Dependências instaladas")

    # Criar estrutura de diretórios de modelos
    if models_dir is None:
        models_dir = comfyui_dir / "models"

    print("[INFO] Criando estrutura de diretórios de modelos...")
    for cat in [
        "checkpoints", "diffusion_models", "loras", "vae", 
        "text_encoders", "clip", "controlnet", "upscale_models", 
        "video_models", "embeddings"
    ]:
        (models_dir / cat).mkdir(parents=True, exist_ok=True)

    # Instalar custom nodes se especificados
    if custom_nodes:
        custom_nodes_dir = comfyui_dir / "custom_nodes"
        custom_nodes_dir.mkdir(parents=True, exist_ok=True)

        for node_spec in custom_nodes:
            # Formato: "repo_url" ou "repo_url@branch" ou "user/repo@branch"
            if "@" in node_spec:
                repo_part, branch = node_spec.rsplit("@", 1)
            else:
                repo_part = node_spec
                branch = "main"

            # Converter user/repo para URL GitHub se necessário
            if not repo_part.startswith("http"):
                if repo_part.startswith("github.com/"):
                    repo_url = f"https://{repo_part}.git"
                else:
                    repo_url = f"https://github.com/{repo_part}.git"
            else:
                repo_url = repo_part

            node_name = repo_url.split("/")[-1].replace(".git", "")
            node_path = custom_nodes_dir / node_name

            if node_path.exists():
                print(f"[INFO] Atualizando custom node: {node_name}...")
                result = subprocess.run(
                    ["git", "-C", str(node_path), "pull"],
                    capture_output=True, text=True
                )
            else:
                print(f"[INFO] Instalando custom node: {node_name}...")
                result = subprocess.run(
                    ["git", "clone", "--branch", branch, repo_url, str(node_path)],
                    capture_output=True, text=True
                )

            if result.returncode != 0:
                print(f"[WARN] Falha ao instalar {node_name}: {result.stderr}")
            else:
                # Instalar requirements do custom node se existir
                node_req = node_path / "requirements.txt"
                if node_req.exists():
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req)],
                        capture_output=True, text=True
                    )
                print(f"[INFO] ✅ Custom node: {node_name}")

    # Configurar extra_model_paths.yaml para apontar para models_dir externo
    extra_paths = comfyui_dir / "extra_model_paths.yaml"
    if not extra_paths.exists():
        print("[INFO] Criando extra_model_paths.yaml...")
        config = [{
            "base_path": str(models_dir),
            "checkpoints": "checkpoints",
            "loras": "loras",
            "vae": "vae",
            "controlnet": "controlnet",
            "embeddings": "embeddings",
            "upscale_models": "upscale_models",
        }]
        with open(extra_paths, "w") as f:
            import yaml
            yaml.dump(config, f)

    print(f"[INFO] ✅ ComfyUI pronto em {comfyui_dir}")
    print(f"[INFO] Modelos em {models_dir}")
    print(f"[INFO] GPU: {gpu_info['gpu_name'] if gpu_info['has_gpu'] else 'CPU only'}")

    return comfyui_dir


def start_comfyui(
    comfyui_dir: Path = DEFAULT_COMFYUI_DIR,
    host: str = "0.0.0.0",
    port: int = 8188,
    extra_args: list = None,
) -> subprocess.Popen:
    """
    Inicia ComfyUI em background.
    Retorna o processo Popen.
    """
    main_py = comfyui_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"ComfyUI não encontrado em {comfyui_dir}")

    cmd = [sys.executable, "main.py", "--listen", host, "--port", str(port)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[INFO] Iniciando ComfyUI: {' '.join(cmd)}")
    print(f"[INFO] Diretório de trabalho: {comfyui_dir}")

    # Mudar para diretório do ComfyUI
    os.chdir(comfyui_dir)

    # Iniciar em background
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(f"[INFO] ComfyUI iniciado (PID: {proc.pid})")
    print(f"[INFO] Acesse: http://{host}:{port}")

    return proc


def health_check(host: str = "127.0.0.1", port: int = 8188, timeout: int = 60) -> bool:
    """Verifica se ComfyUI está respondendo."""
    import urllib.request
    import time

    url = f"http://{host}:{port}/system_stats"
    print(f"[INFO] Verificando health check em {url}...")

    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            response = urllib.request.urlopen(req, timeout=5)
            if response.status == 200:
                print(f"[INFO] ✅ ComfyUI respondendo (status 200)")
                return True
        except Exception:
            pass
        time.sleep(2)

    print(f"[ERROR] Health check falhou após {timeout}s")
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Setup ComfyUI no Kaggle Notebook")
    parser.add_argument("--comfyui-dir", default=str(DEFAULT_COMFYUI_DIR))
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--custom-nodes", nargs="*", help="Lista de custom nodes (repo_url[@branch])")
    parser.add_argument("--models-dir", help="Diretório externo de modelos")
    parser.add_argument("--start", action="store_true", help="Iniciar ComfyUI após setup")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--health-check", action="store_true", help="Fazer health check após iniciar")

    args = parser.parse_args()

    try:
        models_dir = Path(args.models_dir) if args.models_dir else None
        comfyui_dir = setup_comfyui(
            comfyui_dir=Path(args.comfyui_dir),
            repo_url=args.repo_url,
            custom_nodes=args.custom_nodes,
            models_dir=models_dir,
        )

        if args.start:
            proc = start_comfyui(
                comfyui_dir=comfyui_dir,
                host=args.host,
                port=args.port,
            )

            if args.health_check:
                import time
                time.sleep(5)  # Dar tempo para iniciar
                if health_check(args.host, args.port):
                    print("[INFO] ✅ ComfyUI pronto e respondendo!")
                else:
                    print("[WARN] ComfyUI iniciou mas health check falhou")

            # Manter processo vivo se for o caso
            print("[INFO] Pressione Ctrl+C para parar")
            try:
                proc.wait()
            except KeyboardInterrupt:
                print("[INFO] Parando ComfyUI...")
                proc.terminate()
                proc.wait()

        print("\n[SUCCESS] Setup concluído")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()