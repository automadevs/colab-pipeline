#!/usr/bin/env python3
"""Setup do ComfyUI no Kaggle Notebook (SSD local).

Ordem obrigatória de runtime: bootstrap → models → start → health → ngrok.
Manager integrado: manager_requirements.txt + --enable-manager (NÃO custom node).
VRAM: COMFYUI_CUDA_DEVICE=0 por padrão; DynamicVRAM do ComfyUI (sem highvram/lowvram).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

DEFAULT_COMFYUI_DIR = Path("/kaggle/working/ComfyUI")
DEFAULT_REPO_URL = "https://github.com/comfyanonymous/ComfyUI.git"
DEFAULT_DRIVE_BASE = "Automa/ComfyUI"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
ENV_CUDA_DEVICE = "COMFYUI_CUDA_DEVICE"
DEFAULT_CUDA_DEVICE = 0

# Apenas custom nodes realmente necessários. ComfyUI-Manager NÃO entra aqui.
DEFAULT_CUSTOM_NODES = [
    "cubiq/ComfyUI_essentials",
    "lbouaraba/comfyui-krea2edit",
]

MODEL_CATEGORIES = [
    "checkpoints", "diffusion_models", "loras", "vae", "text_encoders",
    "clip", "controlnet", "upscale_models", "video_models", "embeddings",
]

# Flags de VRAM que NÃO devem ser adicionadas por padrão
DISCOURAGED_VRAM_FLAGS = {
    "--highvram", "--gpu-only", "--lowvram", "--novram",
    "--fast", "--reserve-vram",
}


def get_cuda_device(default: int = DEFAULT_CUDA_DEVICE) -> int:
    """Lê COMFYUI_CUDA_DEVICE (default 0). Troque para 1 via env se necessário."""
    raw = os.environ.get(ENV_CUDA_DEVICE, str(default))
    try:
        device = int(str(raw).strip())
        if device < 0:
            raise ValueError("negativo")
        return device
    except (TypeError, ValueError):
        print(f"[WARN] {ENV_CUDA_DEVICE}={raw!r} inválido; usando {default}")
        return default


def detect_gpu() -> dict:
    """Delegado para gpu_detect (enumera todas as GPUs). Fallback mínimo se ausente."""
    try:
        from gpu_detect import detect_gpu as _detect

        return _detect()
    except ImportError:
        info = {"has_gpu": False, "gpu_count": 0, "gpus": [], "gpu_name": "Unknown", "vram_gb": 0}
        try:
            import torch

            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": props.name,
                        "vram_gb": props.total_memory / (1024 ** 3),
                        "cuda": torch.version.cuda,
                        "driver": None,
                    })
                info.update({
                    "has_gpu": True,
                    "gpu_count": len(gpus),
                    "gpus": gpus,
                    "gpu_name": gpus[0]["name"],
                    "vram_gb": gpus[0]["vram_gb"],
                })
                print(f"[INFO] GPU(s): {info['gpu_count']} — padrão device={get_cuda_device()}")
        except Exception as exc:
            print(f"[WARN] Detecção de GPU falhou: {exc}")
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


def parse_custom_node_spec(spec: str) -> tuple[str, str, str]:
    """Retorna (repo_url, branch, node_name) a partir de user/repo[@branch] ou URL."""
    if "@" in spec and not spec.startswith("http"):
        repo_part, branch = spec.rsplit("@", 1)
    elif spec.startswith("http") and "@" in spec.rsplit("/", 1)[-1]:
        # URL@branch raro — tratar @ final
        repo_part, branch = spec.rsplit("@", 1)
    else:
        repo_part, branch = spec, "main"
    repo = repo_part if repo_part.startswith("http") else f"https://github.com/{repo_part}.git"
    node_name = repo.rstrip("/").split("/")[-1].removesuffix(".git")
    return repo, branch, node_name


def is_manager_custom_node_spec(spec: str) -> bool:
    """True se o spec apontar para ltdrdata/ComfyUI-Manager (não deve ser clonado)."""
    lower = spec.lower()
    return "comfyui-manager" in lower or "comfyui_manager" in lower


def filter_custom_nodes(custom_nodes: Optional[list[str]]) -> list[str]:
    """Remove ComfyUI-Manager da lista (Manager integrado via --enable-manager)."""
    if not custom_nodes:
        return []
    filtered = []
    for spec in custom_nodes:
        if is_manager_custom_node_spec(spec):
            print(
                f"[WARN] Ignorando '{spec}': use Manager integrado "
                "(manager_requirements.txt + --enable-manager), não custom node."
            )
            continue
        filtered.append(spec)
    return filtered


def install_or_update_custom_node(custom_dir: Path, spec: str) -> Path:
    """
    Instala ou atualiza um custom node de forma idempotente.
    Se o diretório já existe: git pull --ff-only (não clona por cima).
    """
    custom_dir = Path(custom_dir)
    custom_dir.mkdir(parents=True, exist_ok=True)
    repo, branch, node_name = parse_custom_node_spec(spec)
    node_path = custom_dir / node_name

    if node_path.exists():
        if not (node_path / ".git").exists():
            raise RuntimeError(
                f"Custom node existente não é um checkout Git: {node_path}. "
                "Não será clonado por cima do diretório."
            )
        print(f"[INFO] Atualizando custom node existente: {node_name}")
        _run(["git", "pull", "--ff-only"], cwd=node_path, timeout=300, check=False)
    else:
        print(f"[INFO] Clonando custom node: {node_name} (branch={branch})")
        _run(
            ["git", "clone", "--depth", "1", "--branch", branch, repo, str(node_path)],
            timeout=900,
        )

    node_req = node_path / "requirements.txt"
    if node_req.exists():
        _run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(node_req)],
            timeout=900,
            check=False,
        )
    return node_path


def install_manager_requirements(comfyui_dir: Path) -> bool:
    """Instala ComfyUI/manager_requirements.txt (Manager integrado)."""
    comfyui_dir = Path(comfyui_dir)
    mgr_req = comfyui_dir / "manager_requirements.txt"
    if not mgr_req.exists():
        print(f"[WARN] {mgr_req} não encontrado — Manager integrado pode não ativar")
        return False
    print(f"[INFO] Instalando Manager integrado: {mgr_req}")
    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(mgr_req)], timeout=900)
    return True


def setup_comfyui(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    repo_url=DEFAULT_REPO_URL,
    custom_nodes=None,
    models_dir=None,
    output_dir=None,
    drive_base=DEFAULT_DRIVE_BASE,
    enable_manager: bool = True,
):
    comfyui_dir = Path(comfyui_dir)
    models_dir = Path(models_dir) if models_dir else comfyui_dir / "models"

    # Output SEMPRE no SSD local. Drive NUNCA é --output-directory.
    if output_dir is None:
        output_dir = comfyui_dir / "output"
    else:
        output_dir = Path(output_dir)

    gpu_info = detect_gpu()
    cuda_device = get_cuda_device()
    if gpu_info.get("has_gpu"):
        print(
            f"[INFO] GPUs={gpu_info.get('gpu_count', 0)}; "
            f"ComfyUI usará --cuda-device {cuda_device} "
            f"(env {ENV_CUDA_DEVICE}). VRAM não é somada entre placas."
        )

    is_git_repo = (comfyui_dir / ".git").exists()
    has_main = (comfyui_dir / "main.py").exists()

    if not comfyui_dir.exists():
        _run(["git", "clone", "--depth", "1", repo_url, str(comfyui_dir)], timeout=900)
    elif is_git_repo:
        _run(["git", "pull", "--ff-only"], cwd=comfyui_dir, timeout=300, check=False)
    elif has_main:
        print(f"[WARN] {comfyui_dir} tem main.py sem .git — pulando clone/pull")
    else:
        raise RuntimeError(
            f"Diretório {comfyui_dir} existe mas não é um checkout Git válido do ComfyUI "
            f"(falta .git ou main.py). Remova ou renomeie este diretório antes de executar o setup."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    req = comfyui_dir / "requirements.txt"
    if req.exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=900)

    if enable_manager:
        install_manager_requirements(comfyui_dir)

    for cat in MODEL_CATEGORIES:
        (models_dir / cat).mkdir(parents=True, exist_ok=True)

    nodes = filter_custom_nodes(
        custom_nodes if custom_nodes is not None else list(DEFAULT_CUSTOM_NODES)
    )
    if nodes:
        custom_dir = comfyui_dir / "custom_nodes"
        for spec in nodes:
            install_or_update_custom_node(custom_dir, spec)

    extra_paths = comfyui_dir / "extra_model_paths.yaml"
    if not extra_paths.exists():
        lines = ["kaggle_models:", f"  base_path: {models_dir}"]
        for cat in MODEL_CATEGORIES:
            lines.append(f"  {cat}: {cat}")
        extra_paths.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] Criado {extra_paths}")

    print(f"[INFO] ComfyUI: {comfyui_dir}")
    print(f"[INFO] Models: {models_dir}")
    print(f"[INFO] Output (SSD local): {output_dir}")
    print(f"[INFO] Manager integrado: {'sim' if enable_manager else 'não'}")
    print(f"[INFO] CUDA device padrão: {cuda_device}")
    if gpu_info.get("has_gpu"):
        for g in gpu_info.get("gpus") or []:
            print(f"  GPU[{g.get('index')}] {g.get('name')} — {g.get('vram_gb', 0):.1f} GB")
    else:
        print("[INFO] GPU: CPU only")
    return comfyui_dir


def build_comfyui_command(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir=None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """
    Monta o comando de start do ComfyUI.

    Padrão: listen 127.0.0.1, --cuda-device N, --enable-manager, output local.
    Sem --highvram/--lowvram/--novram/--gpu-only/--fast/--reserve-vram.
    Sem CORS global (--enable-cors-header).
    """
    comfyui_dir = Path(comfyui_dir)
    if output_dir is None:
        output_dir = comfyui_dir / "output"
    else:
        output_dir = Path(output_dir)

    if cuda_device is None:
        cuda_device = get_cuda_device()

    cmd = [
        sys.executable,
        "main.py",
        "--listen",
        host,
        "--port",
        str(port),
        "--output-directory",
        str(output_dir),
        "--cuda-device",
        str(cuda_device),
    ]
    if enable_manager:
        cmd.append("--enable-manager")

    if extra_args:
        for arg in extra_args:
            if arg in DISCOURAGED_VRAM_FLAGS or arg.startswith("--reserve-vram"):
                print(f"[WARN] Flag de VRAM desencorajada no pipeline padrão: {arg}")
            if arg in {"--enable-cors-header", "--cors-origins"} or arg.startswith("--enable-cors"):
                print(f"[WARN] CORS global não é padrão neste pipeline: {arg}")
        cmd.extend(extra_args)
    return cmd


def tail_log_file(log_path: Path, n: int = 40) -> str:
    """Retorna as últimas N linhas de um log (ou mensagem se ausente)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return f"(log inexistente: {log_path})"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(falha ao ler log: {exc})"


def start_comfyui(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    extra_args=None,
    output_dir=None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
):
    main_py = Path(comfyui_dir) / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"ComfyUI não encontrado em {comfyui_dir}")

    if output_dir is None:
        output_dir = Path(comfyui_dir) / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_comfyui_command(
        comfyui_dir=comfyui_dir,
        host=host,
        port=port,
        output_dir=output_dir,
        cuda_device=cuda_device,
        enable_manager=enable_manager,
        extra_args=extra_args,
    )

    log_path = Path(comfyui_dir) / "comfyui.log"
    log = open(log_path, "a", buffering=1, encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=comfyui_dir, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"[INFO] ComfyUI iniciado PID={proc.pid}; log={log_path}")
    print(f"[INFO] Listen: {host}:{port}")
    print(f"[INFO] Output directory (SSD local): {output_dir}")
    print(f"[INFO] CUDA device: {cuda_device if cuda_device is not None else get_cuda_device()}")
    print(f"[INFO] Manager: {'--enable-manager' if enable_manager else 'desligado'}")
    print("[INFO] VRAM: DynamicVRAM padrão do ComfyUI (sem highvram/lowvram/novram)")
    return proc


def health_check(host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: int = 60) -> bool:
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
    print(f"[ERROR] Health check falhou após {timeout}s ({url})")
    return False


def start_comfyui_runtime(
    comfyui_dir=DEFAULT_COMFYUI_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_dir=None,
    extra_args=None,
    cuda_device: Optional[int] = None,
    enable_manager: bool = True,
    enable_ngrok: bool = True,
    health_timeout: int = 90,
    health_host: str = "127.0.0.1",
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """
    Ordem obrigatória: start → health → ngrok.
    Se health falhar: NÃO abre ngrok; mostra path do log e últimas linhas.
    """
    comfyui_dir = Path(comfyui_dir)
    log_path = comfyui_dir / "comfyui.log"
    result: dict[str, Any] = {
        "ok": False,
        "proc": None,
        "public_url": None,
        "local_url": f"http://{health_host}:{port}",
        "log_path": str(log_path),
        "health": False,
        "ngrok_started": False,
        "reused_existing": False,
    }

    if reuse_existing and health_check(health_host, port, timeout=2):
        print(f"[INFO] ComfyUI já está saudável em {result['local_url']}; reutilizando processo existente")
        result["health"] = True
        result["ok"] = True
        result["reused_existing"] = True
    else:
        proc = start_comfyui(
            comfyui_dir=comfyui_dir,
            host=host,
            port=port,
            extra_args=extra_args,
            output_dir=output_dir,
            cuda_device=cuda_device,
            enable_manager=enable_manager,
        )
        result["proc"] = proc

        healthy = health_check(health_host, port, timeout=health_timeout)
        result["health"] = healthy
        if not healthy:
            print(f"[ERROR] ComfyUI falhou no health check. Log: {log_path}")
            print("======== ÚLTIMAS LINHAS DO LOG ========")
            print(tail_log_file(log_path, n=50))
            print("=======================================")
            print("[INFO] ngrok NÃO será iniciado (health check falhou)")
            return result

        result["ok"] = True

    if enable_ngrok:
        try:
            from ngrok_tunnel import start_ngrok_tunnel

            public_url = start_ngrok_tunnel(port=port)
            result["public_url"] = public_url
            result["ngrok_started"] = True
        except Exception as exc:
            # redacao de token se ngrok_tunnel disponível
            msg = str(exc)
            try:
                from ngrok_tunnel import redact_secrets

                msg = redact_secrets(msg)
            except Exception:
                pass
            print(f"[WARN] ngrok não iniciado: {msg}")
            print("[INFO] ComfyUI local segue saudável; configure NGROK_AUTHTOKEN se precisar de URL pública")
    else:
        print("[INFO] ngrok desabilitado (enable_ngrok=False)")

    print("=" * 60)
    print("COMFYUI READY")
    print(f"Local : {result['local_url']}")
    print(f"Public: {result['public_url'] or '(ngrok desabilitado)'}")
    print(f"Output: {output_dir or comfyui_dir / 'output'}")
    print("=" * 60)
    return result


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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--ngrok", action="store_true", help="Após health OK, abrir túnel ngrok")
    parser.add_argument("--cuda-device", type=int, default=None, help="Override COMFYUI_CUDA_DEVICE")
    parser.add_argument("--no-manager", action="store_true", help="Não instalar/ativar Manager integrado")
    args = parser.parse_args()

    if args.cuda_device is not None:
        os.environ[ENV_CUDA_DEVICE] = str(args.cuda_device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    nodes = args.custom_nodes if args.custom_nodes is not None else list(DEFAULT_CUSTOM_NODES)
    comfyui = setup_comfyui(
        Path(args.comfyui_dir),
        args.repo_url,
        nodes,
        Path(args.models_dir) if args.models_dir else None,
        output_dir=output_dir,
        drive_base=args.drive_base,
        enable_manager=not args.no_manager,
    )
    if args.start:
        if args.ngrok or args.health_check:
            runtime = start_comfyui_runtime(
                comfyui_dir=comfyui,
                host=args.host,
                port=args.port,
                output_dir=output_dir,
                enable_manager=not args.no_manager,
                enable_ngrok=args.ngrok,
                health_timeout=90,
            )
            if not runtime["health"]:
                raise RuntimeError("Health check falhou")
            try:
                if runtime["proc"]:
                    runtime["proc"].wait()
            except KeyboardInterrupt:
                if runtime["proc"]:
                    runtime["proc"].terminate()
                try:
                    from ngrok_tunnel import stop_ngrok_tunnel

                    stop_ngrok_tunnel()
                except Exception:
                    pass
        else:
            proc = start_comfyui(
                comfyui,
                args.host,
                args.port,
                output_dir=output_dir,
                enable_manager=not args.no_manager,
            )
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()


if __name__ == "__main__":
    main()
