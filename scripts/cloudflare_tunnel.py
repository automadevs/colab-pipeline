#!/usr/bin/env python3
"""
Túnel Cloudflare (cloudflared) para expor ComfyUI no Kaggle sem CORS global.

Quick tunnels (*.trycloudflare.com): gratuitos, SEM conta e SEM token.
Credenciais de túneis nomeados (TUNNEL_TOKEN / CLOUDFLARE_TUNNEL_TOKEN), se
presentes no ambiente, nunca são impressas nos logs.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

SECRET_ENV_NAMES = ("TUNNEL_TOKEN", "CLOUDFLARE_TUNNEL_TOKEN")
DEFAULT_PORT = 8188

CLOUDFLARED_VERSION = "2026.9.3"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/"
    f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)
CLOUDFLARED_SHA256 = "77e26d8d900e0b8469f416239d14b5f296525fdf79fee6f511ef55609e3fbac2"

_PUBLIC_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_URL_TIMEOUT = 60

_CLOUDFLARED_PROC: Optional[subprocess.Popen] = None


def redact_secrets(text: str, secrets: Optional[list[Optional[str]]] = None) -> str:
    """Remove tokens sensíveis de strings de log/erro."""
    if not text:
        return text
    out = text
    candidates = list(secrets or [])
    for name in SECRET_ENV_NAMES:
        env_token = os.environ.get(name)
        if env_token:
            candidates.append(env_token)
    for secret in candidates:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "***REDACTED***")
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_cloudflared(dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(CLOUDFLARED_URL, timeout=120) as response:
            part.write_bytes(response.read())
        if _sha256_file(part) != CLOUDFLARED_SHA256:
            raise RuntimeError("sha256 do cloudflared não confere com a release pinada")
        part.chmod(0o755)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)


def ensure_cloudflared_installed() -> str:
    """Garante o binário cloudflared e devolve o path executável.

    Ordem: PATH do sistema → CLOUDFLARED_PATH → download pinado (com sha256)
    em /tmp, que é tmpfs no Kaggle e não deixa artefato persistente.
    """
    on_path = shutil.which("cloudflared")
    if on_path:
        return on_path
    env_path = os.environ.get("CLOUDFLARED_PATH", "")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        raise RuntimeError(
            f"cloudflared: plataforma sem binário pinado "
            f"({sys.platform}/{platform.machine()}); instale manualmente e "
            "aponte CLOUDFLARED_PATH."
        )
    dest = Path(tempfile.gettempdir()) / f"cloudflared-{CLOUDFLARED_VERSION}"
    if dest.exists() and _sha256_file(dest) == CLOUDFLARED_SHA256:
        return str(dest)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            print(f"[INFO] Baixando cloudflared {CLOUDFLARED_VERSION} (tentativa {attempt + 1}/3)...")
            _download_cloudflared(dest)
            return str(dest)
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"Falha ao baixar cloudflared: {redact_secrets(str(last_exc))}"
    ) from None


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def stop_cloudflare_tunnel() -> None:
    """Encerra o túnel cloudflared ativo (idempotente)."""
    global _CLOUDFLARED_PROC
    proc, _CLOUDFLARED_PROC = _CLOUDFLARED_PROC, None
    if proc is not None:
        _terminate(proc)
        print("[INFO] cloudflared encerrado")
    # Órfãos de sessões anteriores (best-effort; pkill não existe fora de Linux)
    try:
        subprocess.run(
            ["pkill", "-f", "cloudflared tunnel"],
            check=False,
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _wait_for_public_url(proc: subprocess.Popen, timeout: int) -> str:
    """Drena o stderr do cloudflared até a URL pública aparecer (sem ecoar logs).

    A saída do processo NUNCA é impressa: só a URL parseada sai daqui. Em caso
    de falha, o chamador deve aplicar redact_secrets na exceção.
    """
    import queue as queue_mod

    lines: "queue_mod.Queue[Optional[str]]" = queue_mod.Queue()

    def _drain() -> None:
        try:
            for line in proc.stderr:
                lines.put(line.rstrip())
        except Exception:
            pass
        lines.put(None)

    threading.Thread(target=_drain, daemon=True).start()
    tail: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue_mod.Empty:
            continue
        if line is None:
            break
        match = _PUBLIC_URL_RE.search(line)
        if match:
            return match.group(0)
        tail.append(line)
        del tail[:-20]
    context = "\n".join(tail[-10:])
    raise RuntimeError(
        f"cloudflared não expôs URL pública (saída: {context or 'vazia'})"
    )


def start_cloudflare_tunnel(
    port: int = DEFAULT_PORT,
    authtoken: Optional[str] = None,
    bind_tls: bool = True,
) -> str:
    """
    Sobe quick tunnel para 127.0.0.1:port e retorna a URL pública (https).

    Sempre encerra túneis anteriores antes de conectar. Nunca imprime credencial.

    ``authtoken``/``bind_tls`` existem por compatibilidade de assinatura com o
    antigo módulo ngrok: quick tunnels Cloudflare não usam credencial e são
    sempre TLS; um authtoken informado é ignorado (e jamais impresso).

    AVISO: o túnel cria exposição externa. Em SECURE_MODE, o pipeline garante
    que health_check() passou antes de chamar esta função.
    """
    global _CLOUDFLARED_PROC
    stop_cloudflare_tunnel()
    if authtoken:
        print("[WARN] authtoken ignorado: quick tunnels Cloudflare não usam credencial.")
    binary = ensure_cloudflared_installed()
    try:
        proc = subprocess.Popen(
            [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao iniciar cloudflared: {redact_secrets(str(exc), [authtoken])}"
        ) from None
    try:
        public_url = _wait_for_public_url(proc, _URL_TIMEOUT)
    except Exception as exc:
        _terminate(proc)
        raise RuntimeError(redact_secrets(str(exc), [authtoken])) from None
    _CLOUDFLARED_PROC = proc
    print(f"[INFO] cloudflare public_url: {public_url}")
    print(f"[INFO] ComfyUI local permanece em 127.0.0.1:{port} (sem CORS global)")
    return public_url


def get_active_tunnels() -> list[Any]:
    """Lista com o processo de túnel ativo (debug; não expõe credenciais)."""
    if _CLOUDFLARED_PROC is not None and _CLOUDFLARED_PROC.poll() is None:
        return [_CLOUDFLARED_PROC]
    return []


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gerenciar túnel Cloudflare para ComfyUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--start", action="store_true", help="Iniciar túnel")
    parser.add_argument("--stop", action="store_true", help="Parar túnel")
    args = parser.parse_args()

    if args.stop or not args.start:
        if args.stop:
            stop_cloudflare_tunnel()
            return
        if not args.start:
            parser.print_help()
            return

    url = start_cloudflare_tunnel(port=args.port)
    print(url)


if __name__ == "__main__":
    main()
