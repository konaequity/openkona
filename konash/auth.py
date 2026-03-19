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
OPENAI_KEYS_PAGE = "https://platform.openai.com/api-keys"

# ---------------------------------------------------------------------------
# Google AI (Gemini Embeddings)
# ---------------------------------------------------------------------------
GOOGLE_AI_KEYS_PAGE = "https://aistudio.google.com/app/apikey"


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


def validate_together_key(key: str) -> tuple:
    """Validate a Together AI API key. Returns (valid, error_message)."""
    try:
        # Use /v1/models endpoint — only requires auth, no credits needed
        req = urllib.request.Request(
            "https://api.together.xyz/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "konash",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, ""
    except urllib.error.HTTPError as e:
        code = e.code
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        if code == 401:
            return False, "Invalid API key"
        elif code == 403:
            return False, "Request blocked — try again in a minute"
        elif code == 429:
            return False, "Rate limited — wait a moment and try again"
        else:
            return False, f"HTTP {code}: {body[:100]}"
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"
    except Exception as e:
        return False, str(e)


def validate_google_key(key: str) -> bool:
    """Validate a Google API key with a minimal Gemini embedding call."""
    try:
        payload = json.dumps({
            "model": "models/gemini-embedding-2-preview",
            "content": {"parts": [{"text": "test"}]},
        })
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent?key={key}",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


SHADEFORM_KEYS_PAGE = "https://platform.shadeform.ai/settings/api"


def validate_shadeform_key(key: str) -> bool:
    """Validate a Shadeform API key by listing user instances (requires auth)."""
    try:
        req = urllib.request.Request(
            "https://api.shadeform.ai/v1/instances",
            headers={"X-API-KEY": key},
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


def validate_openai_key(key: str) -> tuple[bool, str]:
    """Validate an OpenAI API key. Returns (valid, error_message)."""
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10):
            return True, ""
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Invalid API key"
        if e.code == 429:
            return False, "Rate limited — wait a moment and try again"
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}"
    except Exception as e:
        return False, str(e)


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
