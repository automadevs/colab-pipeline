#!/usr/bin/env python3
"""
Túnel ngrok para expor ComfyUI no Kaggle sem CORS global.

Secret / env: NGROK_AUTHTOKEN
Nunca use getpass. Nunca imprima o token nos logs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Optional

SECRET_NAME = "NGROK_AUTHTOKEN"
DEFAULT_PORT = 8188


def redact_secrets(text: str, secrets: Optional[list[Optional[str]]] = None) -> str:
    """Remove tokens sensíveis de strings de log/erro."""
    if not text:
        return text
    out = text
    candidates = list(secrets or [])
    env_token = os.environ.get(SECRET_NAME)
    if env_token:
        candidates.append(env_token)
    for secret in candidates:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "***REDACTED***")
    return out


def resolve_ngrok_authtoken() -> Optional[str]:
    """
    Resolve NGROK_AUTHTOKEN via Kaggle Secrets ou variável de ambiente.
    Não usa getpass. Não loga o valor.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(SECRET_NAME)
        if token and str(token).strip():
            return str(token).strip()
    except Exception:
        pass

    env_token = os.environ.get(SECRET_NAME)
    if env_token and env_token.strip():
        return env_token.strip()
    return None


def ensure_pyngrok_installed() -> None:
    """Instala pyngrok via pip se ainda não estiver disponível."""
    try:
        import pyngrok  # noqa: F401
        return
    except ImportError:
        pass
    print("[INFO] Instalando pyngrok...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
        check=True,
        timeout=300,
    )


def stop_ngrok_tunnel() -> None:
    """Encerra todos os túneis ngrok ativos (idempotente)."""
    try:
        ensure_pyngrok_installed()
        from pyngrok import ngrok

        ngrok.kill()
        print("[INFO] ngrok.kill() executado")
    except Exception as exc:
        print(f"[INFO] ngrok.kill() (sem túnel ativo ou falha benigna): {redact_secrets(str(exc))}")


def start_ngrok_tunnel(
    port: int = DEFAULT_PORT,
    authtoken: Optional[str] = None,
    bind_tls: bool = True,
) -> str:
    """
    Cria túnel HTTP para a porta local do ComfyUI.

    Sempre chama ngrok.kill() antes de conectar.
    Retorna public_url. Nunca imprime o token.
    """
    token = authtoken or resolve_ngrok_authtoken()
    if not token:
        raise RuntimeError(
            f"{SECRET_NAME} não encontrado. Configure o Secret no Kaggle "
            f"ou a variável de ambiente {SECRET_NAME}."
        )

    ensure_pyngrok_installed()
    from pyngrok import ngrok

    try:
        ngrok.set_auth_token(token)
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao configurar autenticação ngrok: {redact_secrets(str(exc), [token])}"
        ) from None

    # Sempre matar túneis anteriores antes de criar um novo
    try:
        ngrok.kill()
    except Exception:
        pass

    try:
        tunnel = ngrok.connect(
            addr=f"127.0.0.1:{port}",
            proto="http",
            bind_tls=bind_tls,
        )
    except TypeError:
        # Compatibilidade com versões antigas do pyngrok
        tunnel = ngrok.connect(f"127.0.0.1:{port}", "http")
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao criar túnel ngrok: {redact_secrets(str(exc), [token])}"
        ) from None

    public_url = getattr(tunnel, "public_url", None) or str(tunnel)
    if public_url.startswith("http://") and bind_tls:
        # Preferir URL https quando disponível
        https_url = "https://" + public_url[len("http://") :]
        public_url = https_url

    print(f"[INFO] ngrok public_url: {public_url}")
    print(f"[INFO] ComfyUI local permanece em 127.0.0.1:{port} (sem CORS global)")
    return public_url


def get_active_tunnels() -> list[Any]:
    """Lista túneis ativos (para debug; não expõe token)."""
    ensure_pyngrok_installed()
    from pyngrok import ngrok

    try:
        return list(ngrok.get_tunnels())
    except Exception:
        return []


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gerenciar túnel ngrok para ComfyUI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--start", action="store_true", help="Iniciar túnel")
    parser.add_argument("--stop", action="store_true", help="Parar túneis")
    args = parser.parse_args()

    if args.stop or not args.start:
        if args.stop:
            stop_ngrok_tunnel()
            return
        if not args.start:
            parser.print_help()
            return

    url = start_ngrok_tunnel(port=args.port)
    print(url)


if __name__ == "__main__":
    main()
