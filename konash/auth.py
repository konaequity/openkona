"""Authentication flows for KONASH.

Together AI: Browser-open + paste + instant validation (no OAuth available).
HuggingFace: OAuth device flow (no manual copy) with manual fallback.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rich.console import Console

# ---------------------------------------------------------------------------
# HuggingFace OAuth Device Flow (RFC 8628)
# Register a public OAuth app at: https://huggingface.co/settings/applications/new
# Scopes: read-repos, write-repos  |  No client secret (public app)
# ---------------------------------------------------------------------------
HF_OAUTH_CLIENT_ID: Optional[str] = None  # Set after registering
HF_DEVICE_URL = "https://huggingface.co/oauth/device"
HF_TOKEN_URL = "https://huggingface.co/oauth/token"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_SCOPES = "read-repos write-repos"

# Well-known token locations
_HF_TOKEN_PATHS = [
    os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "token"),
    "~/.huggingface/token",
]

# ---------------------------------------------------------------------------
# Together AI
# ---------------------------------------------------------------------------
TOGETHER_KEYS_PAGE = "https://api.together.xyz/settings/api-keys"
HF_TOKENS_PAGE = "https://huggingface.co/settings/tokens"


def detect_hf_token() -> Optional[str]:
    """Check well-known locations for an existing HuggingFace token."""
    # Env vars
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val
    # Token files from huggingface-cli login
    for path in _HF_TOKEN_PATHS:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded):
            try:
                token = open(expanded).read().strip()
                if token:
                    return token
            except OSError:
                pass
    return None


def validate_together_key(key: str) -> bool:
    """Validate a Together AI API key with a 1-token completion."""
    try:
        payload = json.dumps({
            "model": "Qwen/Qwen3.5-9B",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        })
        req = urllib.request.Request(
            "https://api.together.xyz/v1/chat/completions",
            data=payload.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "konash",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def validate_hf_token(token: str) -> Optional[str]:
    """Validate a HuggingFace token. Returns username or None."""
    try:
        req = urllib.request.Request(
            HF_WHOAMI_URL,
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("name") or data.get("fullname")
    except Exception:
        return None


def hf_device_flow(console: Console) -> Optional[str]:
    """HuggingFace OAuth device flow.

    Opens browser, user clicks authorize, token flows back.
    Returns access_token or None if flow unavailable/failed.
    """
    if not HF_OAUTH_CLIENT_ID:
        return None

    try:
        data = urllib.parse.urlencode({
            "client_id": HF_OAUTH_CLIENT_ID,
            "scope": HF_SCOPES,
        }).encode()
        req = urllib.request.Request(HF_DEVICE_URL, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
    except Exception:
        return None

    device_code = result["device_code"]
    user_code = result["user_code"]
    verification_uri = result.get("verification_uri", "https://huggingface.co/device")
    interval = result.get("interval", 5)
    expires_in = result.get("expires_in", 900)

    console.print()
    console.print(f"    Your one-time code:  [bold cyan]{user_code}[/]")
    console.print()
    console.print("    A browser window will open.")
    console.print("    Log in to HuggingFace and enter the code above.")
    console.print()

    webbrowser.open(verification_uri)

    deadline = time.time() + expires_in
    with console.status("    [cyan]Waiting for authorization...", spinner="dots"):
        while time.time() < deadline:
            time.sleep(interval)
            try:
                body = urllib.parse.urlencode({
                    "client_id": HF_OAUTH_CLIENT_ID,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                }).encode()
                req = urllib.request.Request(HF_TOKEN_URL, data=body)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    token_result = json.loads(resp.read())
                if "access_token" in token_result:
                    return token_result["access_token"]
            except urllib.error.HTTPError as e:
                try:
                    err_body = json.loads(e.read())
                except Exception:
                    continue
                error = err_body.get("error", "")
                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval += 1
                    continue
                else:
                    return None
            except Exception:
                continue

    return None
