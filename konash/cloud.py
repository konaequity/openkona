"""Cloud GPU training via Shadeform API.

Finds the cheapest available GPU across 20+ providers, provisions it,
runs OAPL training, downloads the trained adapter, and tears down.
All with a single Shadeform API key — no per-provider accounts needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
import urllib.error
import urllib.parse
import urllib.request

_SHADEFORM_API = "https://api.shadeform.ai/v1"
_CONFIG_DIR = os.path.expanduser("~/.konash")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")
_SSH_KEY_PATH = os.path.join(_CONFIG_DIR, "shadeform_ssh_key")
_INSTANCE_STATE = os.path.join(_CONFIG_DIR, "active_instance.json")
_PROJECT_ROOT = Path(__file__).parent.parent
# Remote paths — use ~ so it resolves to whatever the SSH user's home is
_REMOTE_DIR = "~/konash"
_REMOTE_LOG = "~/konash/training.log"


# ---------------------------------------------------------------------------
# Shadeform HTTP helpers
# ---------------------------------------------------------------------------

def _get_shadeform_key() -> Optional[str]:
    """Resolve Shadeform API key from env or config."""
    key = os.environ.get("SHADEFORM_API_KEY")
    if key:
        return key
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            return json.load(f).get("shadeform_api_key")
    return None


def _shadeform_request(
    method: str, path: str, body: Optional[dict] = None, api_key: Optional[str] = None,
) -> dict:
    """Make a Shadeform API request."""
    key = api_key or _get_shadeform_key()
    if not key:
        raise RuntimeError("No Shadeform API key. Run konash train to set one up.")

    url = f"{_SHADEFORM_API}{path}"
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""

        # Handle specific errors with helpful messages
        try:
            error_data = json.loads(error_body)
            error_code = error_data.get("error_code", "")
        except (json.JSONDecodeError, TypeError):
            error_code = ""

        if error_code == "INSUFFICIENT_FUNDS":
            raise RuntimeError(
                "Your Shadeform account needs funds to launch a GPU.\n"
                "Add funds at: https://platform.shadeform.ai/settings/billing"
            ) from e

        raise RuntimeError(f"Shadeform API error {e.code}: {error_body}") from e


# ---------------------------------------------------------------------------
# SSH key management
# ---------------------------------------------------------------------------

def _ensure_ssh_key(api_key: str) -> str:
    """Auto-generate SSH keypair and upload to Shadeform. Returns key path."""
    if os.path.exists(_SSH_KEY_PATH):
        return _SSH_KEY_PATH

    os.makedirs(_CONFIG_DIR, exist_ok=True)

    # Generate keypair
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", _SSH_KEY_PATH, "-N", "", "-q"],
        check=True,
    )
    os.chmod(_SSH_KEY_PATH, 0o600)

    # Upload public key to Shadeform
    pub_key_path = _SSH_KEY_PATH + ".pub"
    with open(pub_key_path) as f:
        public_key = f.read().strip()

    result = _shadeform_request(
        "POST", "/sshkeys/add",
        {"name": "konash", "public_key": public_key},
        api_key=api_key,
    )
    key_id = result.get("id", "")

    # Set as default
    if key_id:
        try:
            _shadeform_request(
                "POST", f"/sshkeys/{key_id}/setdefault", api_key=api_key,
            )
        except Exception:
            pass  # Not critical

    # Save key ID in config
    config = {}
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            config = json.load(f)
    config["shadeform_ssh_key_id"] = key_id
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    return _SSH_KEY_PATH


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

# GLM 4.5 Air FP8 needs ~110 GB VRAM. These GPUs can fit it
# (with 2+ cards or a single large card). Ordered by preference:
# cheapest first, newest last.
_VLLM_COMPATIBLE_GPUS = ["H100", "H200", "B200", "B300", "A100_80G"]

# Model weight estimates (GB) for VRAM budgeting.
# Ordered longest key first so "GLM-4.5-Air-FP8" matches before "GLM-4.5-Air".
_MODEL_WEIGHTS_GB = [
    ("zai-org/GLM-4.5-Air-FP8", 110),  # FP8: 106B params × ~1 byte
    ("zai-org/GLM-4.5-Air", 212),      # BF16: 106B params × 2 bytes
    ("zai-org/GLM-4.5-FP8", 220),      # FP8: dense 212B params × ~1 byte
    ("zai-org/GLM-4.5", 424),          # BF16: dense 212B params × 2 bytes
]

# Safety margin: model weights + 30% for KV cache, activations, vLLM overhead
_VRAM_SAFETY_MARGIN = 1.3


def _model_weight_gb(model: str) -> int:
    """Look up estimated model weight memory in GB.

    Matches longest key first to avoid 'GLM-4.5-Air' shadowing
    'GLM-4.5-Air-FP8'. Returns 0 for unknown models.
    """
    model_lower = model.lower()
    for key, weight_gb in _MODEL_WEIGHTS_GB:
        if key.lower() in model_lower:
            return weight_gb
    return 0


def _min_vram_for_model(model: str) -> int:
    """Compute minimum total VRAM (GB) needed to serve a model via vLLM."""
    weight_gb = _model_weight_gb(model)
    if weight_gb > 0:
        return int(weight_gb * _VRAM_SAFETY_MARGIN)
    return 80


def _find_cheapest_gpu(
    gpu_type: str = "H100",
    num_gpus: int = 1,
    api_key: Optional[str] = None,
    min_vram_gb: int = 0,
    fallback_gpu_types: Optional[list] = None,
) -> dict:
    """Find the cheapest available GPU instance.

    Parameters
    ----------
    gpu_type : str
        Preferred GPU type.
    num_gpus : int
        Number of GPUs needed.
    api_key : str
        Shadeform API key.
    min_vram_gb : int
        Minimum total VRAM required (filters out instances too small).
    fallback_gpu_types : list[str] | None
        If preferred GPU is unavailable, try these in order.
        Defaults to ``_VLLM_COMPATIBLE_GPUS``.
    """
    gpu_types_to_try = [gpu_type]
    if fallback_gpu_types:
        gpu_types_to_try += [g for g in fallback_gpu_types if g != gpu_type]

    for gtype in gpu_types_to_try:
        try:
            data = _shadeform_request(
                "GET",
                f"/instances/types?gpu_type={gtype}&num_gpus={num_gpus}"
                f"&available=true&sort=price",
                api_key=api_key,
            )
        except RuntimeError:
            continue

        instances = data.get("instance_types", [])

        # Filter by minimum VRAM if specified
        if min_vram_gb > 0:
            instances = [
                inst for inst in instances
                if _instance_total_vram(inst) >= min_vram_gb
            ]

        if not instances:
            continue

        best = instances[0]
        # Find an available region
        for avail in best.get("availability", []):
            if avail.get("available"):
                best["_region"] = avail["region"]
                break

        return best

    tried = ", ".join(gpu_types_to_try)
    raise RuntimeError(
        f"No {num_gpus}x GPU available on Shadeform right now.\n"
        f"  Tried: {tried}\n"
        f"  Minimum VRAM: {min_vram_gb} GB\n"
        f"  Check availability at https://www.shadeform.ai/instances"
    )


def _instance_total_vram(instance: dict) -> int:
    """Extract total VRAM in GB from a Shadeform instance type dict."""
    config = instance.get("configuration", {})
    vram = int(config.get("vram_per_gpu_in_gb", 0) or 0)
    n = int(config.get("num_gpus", 1) or 1)
    return vram * n


def _launch_instance(
    cloud: str, region: str, shade_type: str, name: str,
    ssh_key_id: Optional[str] = None,
    os_options: Optional[list] = None,
    api_key: Optional[str] = None,
) -> str:
    """Launch a GPU instance. Returns instance ID."""
    # Pick best OS: prefer CUDA shade_os, fall back to ubuntu
    os_image = "ubuntu22.04_cuda12.8_shade_os"
    if os_options:
        for pref in ["ubuntu22.04_cuda12.8_shade_os", "ubuntu22.04_cuda12.2_shade_os",
                      "ubuntu24.04_cuda12.4_shade_os", "ubuntu22.04", "ubuntu24.04"]:
            if pref in os_options:
                os_image = pref
                break
        else:
            os_image = os_options[0]

    body: dict[str, Any] = {
        "cloud": cloud,
        "region": region,
        "shade_instance_type": shade_type,
        "shade_cloud": True,
        "name": name,
        "os": os_image,
    }
    if ssh_key_id:
        body["ssh_key_id"] = ssh_key_id

    result = _shadeform_request("POST", "/instances/create", body, api_key=api_key)
    instance_id = result.get("id", "")
    if not instance_id:
        raise RuntimeError(f"Failed to launch instance: {result}")
    return instance_id


def _poll_until_active(
    instance_id: str, timeout: int = 600, api_key: Optional[str] = None,
) -> dict:
    """Poll instance until active. Returns {ip, ssh_port, ssh_user}."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        data = _shadeform_request(
            "GET", f"/instances/{instance_id}/info", api_key=api_key,
        )
        status = data.get("status", "")
        if status == "active":
            return {
                "ip": data.get("ip", ""),
                "ssh_port": data.get("ssh_port", 22),
                "ssh_user": data.get("ssh_user", "shadeform"),
            }
        if status in ("error", "failed", "deleted"):
            raise RuntimeError(f"Instance {instance_id} failed with status: {status}")
        time.sleep(10)
    raise RuntimeError(f"Instance {instance_id} did not become active within {timeout}s")


def _delete_instance(instance_id: str, api_key: Optional[str] = None) -> None:
    """Delete a GPU instance."""
    try:
        _shadeform_request("POST", f"/instances/{instance_id}/delete", api_key=api_key)
    except Exception:
        pass  # Best effort


# ---------------------------------------------------------------------------
# Instance state persistence (for konash logs / konash stop)
# ---------------------------------------------------------------------------

def _save_instance_state(
    instance_id: str, ip: str, ssh_port: int = 22, ssh_user: str = "shadeform",
    hourly_price: float = 0,
) -> None:
    """Save active instance info for logs/stop commands."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    with open(_INSTANCE_STATE, "w") as f:
        json.dump({
            "instance_id": instance_id,
            "ip": ip,
            "ssh_port": ssh_port,
            "ssh_user": ssh_user,
            "ssh_key_path": _SSH_KEY_PATH,
            "hourly_price": hourly_price,
            "created_at": time.time(),
        }, f, indent=2)


def _load_instance_state() -> Optional[dict]:
    """Load active instance info."""
    if not os.path.exists(_INSTANCE_STATE):
        return None
    with open(_INSTANCE_STATE) as f:
        return json.load(f)


def _clear_instance_state() -> None:
    if os.path.exists(_INSTANCE_STATE):
        os.remove(_INSTANCE_STATE)


def list_instances(api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Return all non-deleted Shadeform instances for the current workspace."""
    data = _shadeform_request("GET", "/instances", api_key=api_key)
    instances = data.get("instances", [])
    return instances if isinstance(instances, list) else []


def get_instance_status_summary(api_key: Optional[str] = None) -> dict[str, Any]:
    """Summarize currently visible Shadeform instances plus local tracked state."""
    local_state = _load_instance_state()
    key = api_key or _get_shadeform_key()
    if not key:
        return {
            "configured": False,
            "running_count": 0,
            "instances": [],
            "local_instance": local_state,
            "error": "",
        }

    raw_instances = list_instances(api_key=key)
    visible_statuses = {
        "pending",
        "creating",
        "provisioning",
        "starting",
        "active",
        "stopping",
    }
    instances: list[dict[str, Any]] = []
    for inst in raw_instances:
        if not isinstance(inst, dict):
            continue
        status = str(inst.get("status", "")).lower()
        if status and status not in visible_statuses:
            continue
        config = inst.get("configuration", {})
        gpu_type = ""
        num_gpus = 0
        if isinstance(config, dict):
            gpu_type = str(config.get("gpu_type", "") or "")
            try:
                num_gpus = int(config.get("num_gpus", 0) or 0)
            except (TypeError, ValueError):
                num_gpus = 0
        hourly_cents = inst.get("hourly_price", 0)
        try:
            hourly_usd = float(hourly_cents) / 100.0
        except (TypeError, ValueError):
            hourly_usd = 0.0
        instances.append({
            "id": str(inst.get("id", "")),
            "name": str(inst.get("name", "") or "Unnamed instance"),
            "status": status or "unknown",
            "cloud": str(inst.get("cloud", "") or ""),
            "region": str(inst.get("region", "") or ""),
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "hourly_usd": hourly_usd,
            "ip": str(inst.get("ip", "") or ""),
        })

    active_like = {"pending", "creating", "provisioning", "starting", "active", "stopping"}
    running_count = sum(1 for inst in instances if inst["status"] in active_like)
    return {
        "configured": True,
        "running_count": running_count,
        "instances": instances,
        "local_instance": local_state,
        "error": "",
    }


# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]


def _ssh_cmd(
    ip: str, command: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> list:
    """Build an SSH command."""
    return [
        "ssh", *_SSH_OPTS,
        "-i", key_path, "-p", str(port),
        f"{user}@{ip}", command,
    ]


def _scp_upload(
    ip: str, local: str, remote: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Upload a file via SCP."""
    subprocess.run(
        ["scp", *_SSH_OPTS, "-i", key_path, "-P", str(port),
         local, f"{user}@{ip}:{remote}"],
        check=True, capture_output=True,
    )


def _scp_download(
    ip: str, remote: str, local: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Download files via SCP."""
    subprocess.run(
        ["scp", *_SSH_OPTS, "-i", key_path, "-P", str(port),
         "-r", f"{user}@{ip}:{remote}", local],
        check=True, capture_output=True,
    )


def _upload_codebase(
    ip: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Upload the KONASH codebase via tar pipe."""
    tar_cmd = [
        "tar", "czf", "-",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".konash",
        "-C", str(_PROJECT_ROOT), ".",
    ]
    ssh_cmd = [
        "ssh", *_SSH_OPTS,
        "-i", key_path, "-p", str(port),
        f"{user}@{ip}",
        f"mkdir -p {_REMOTE_DIR} && tar xzf - -C {_REMOTE_DIR}",
    ]
    tar = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_cmd, stdin=tar.stdout)
    tar.stdout.close()
    ssh.wait()
    tar.wait()
    if ssh.returncode != 0:
        raise RuntimeError("Failed to upload codebase")


def _setup_remote(
    ip: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Install KONASH dependencies on the remote machine."""
    setup_cmd = (
        f"cd {_REMOTE_DIR} && "
        "pip3 install numpy sentence-transformers datasets huggingface_hub torch "
        "unsloth peft accelerate 2>&1 | tail -3"
    )
    result = subprocess.run(
        _ssh_cmd(ip, setup_cmd, key_path, port, user),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote setup failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Remote training with progress streaming
# ---------------------------------------------------------------------------

def _run_remote_training(
    ip: str, training_cmd: str, env_vars: dict,
    key_path: str = _SSH_KEY_PATH, port: int = 22, user: str = "shadeform",
    verbose: bool = True,
) -> None:
    """Run training on the remote machine, streaming ##KONASH## progress markers."""
    _print = print if verbose else lambda *a, **k: None

    # Build env exports
    env_str = " ".join(f'{k}="{v}"' for k, v in env_vars.items())

    # Run under nohup so SSH disconnection doesn't kill training
    remote_cmd = (
        f"cd {_REMOTE_DIR} && "
        f"export PYTHONPATH=. UNSLOTH_VLLM_STANDBY=1 {env_str} && "
        f"nohup bash -c '{training_cmd} > {_REMOTE_LOG} 2>&1' &"
    )

    # Start training in background
    subprocess.run(
        _ssh_cmd(ip, remote_cmd, key_path, port, user),
        capture_output=True, timeout=30,
    )

    # Wait a moment for the process to start
    time.sleep(2)

    # Tail the log file and parse progress markers
    tail_cmd = f"tail -f {_REMOTE_LOG}"
    proc = subprocess.Popen(
        _ssh_cmd(ip, tail_cmd, key_path, port, user),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    phase_start = time.monotonic()

    for line in proc.stdout:
        line = line.rstrip()

        if "##KONASH:" in line:
            elapsed = time.monotonic() - phase_start
            marker = line.split("##KONASH:")[1].rstrip("#")

            if marker == "loading_data":
                _print(f"  [green]✓[/]  Instance ready")
                phase_start = time.monotonic()

            elif marker == "loading_model":
                _print(f"  Loading model...")
                phase_start = time.monotonic()

            elif marker.startswith("model_loaded:"):
                load_time = marker.split(":")[1]
                _print(f"  [green]✓[/]  Model loaded  [dim]{load_time}[/]")

            elif marker == "oapl_start":
                _print(f"  Training OAPL...")
                phase_start = time.monotonic()

            elif marker.startswith("oapl_done:"):
                elapsed = time.monotonic() - phase_start
                loss = marker.split("loss=")[1] if "loss=" in marker else "?"
                _print(f"  [green]✓[/]  OAPL complete  [dim]loss {loss}  {elapsed:.0f}s[/]")

            elif marker == "value_model_start":
                _print(f"  Training value model...")
                phase_start = time.monotonic()

            elif marker.startswith("value_model_done:"):
                elapsed = time.monotonic() - phase_start
                loss = marker.split("loss=")[1] if "loss=" in marker else "?"
                _print(f"  [green]✓[/]  Value model trained  [dim]loss {loss}  {elapsed:.0f}s[/]")

            elif marker == "complete":
                _print(f"  [green]✓[/]  GPU training complete")
                proc.terminate()
                break

    proc.wait()


# ---------------------------------------------------------------------------
# Just-in-time Shadeform setup
# ---------------------------------------------------------------------------

def _ensure_shadeform(verbose: bool = True) -> str:
    """Ensure Shadeform API key is configured. Returns the key."""
    _print = print if verbose else lambda *a, **k: None
    key = _get_shadeform_key()

    if key:
        return key

    _print()
    _print(
        "  OAPL training needs a GPU. Shadeform gives you access to\n"
        "  the cheapest GPUs across 20+ providers with a single API key."
    )
    _print()

    try:
        from rich.prompt import Prompt
        from konash.cli import _arrow_select
        from rich.console import Console
        _con = Console()

        idx = _arrow_select(_con, [
            {"label": "Open shadeform.ai", "hint": "Sign up and grab an API key (free, takes 30 seconds)"},
            {"label": "I already have a key", "hint": ""},
        ])
        _con.print()

        if idx == 0:
            import webbrowser
            from konash.auth import SHADEFORM_KEYS_PAGE
            webbrowser.open(SHADEFORM_KEYS_PAGE)
            _con.print("    [dim]Settings → API Keys → create a key[/]")
            _con.print()

        while True:
            key = Prompt.ask("    Paste your Shadeform API key")

            if not key:
                _con.print("    [dim]No key entered. Try again or Ctrl+C to cancel.[/]")
                continue

            from konash.auth import validate_shadeform_key
            if validate_shadeform_key(key):
                break
            _con.print("    [red]✗[/]  Invalid key — check and try again")
            _con.print()

        _con.print("    [green]✓[/]  Connected to Shadeform")

        # Save to config
        config = {}
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE) as f:
                config = json.load(f)
        config["shadeform_api_key"] = key
        os.makedirs(_CONFIG_DIR, exist_ok=True)
        with open(_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

    except ImportError:
        raise RuntimeError(
            "No Shadeform API key configured.\n"
            "Set SHADEFORM_API_KEY or run konash train."
        )

    return key


# ---------------------------------------------------------------------------
# Public API (same signatures as before)
# ---------------------------------------------------------------------------

def _provision_gpu(
    gpu: str, api_key: str, verbose: bool = True, num_gpus: int = 1,
    min_vram_gb: int = 0, fallback_gpu_types: Optional[list] = None,
) -> tuple:
    """Find, launch, and wait for a GPU instance.

    Handles insufficient funds by prompting the user to add balance
    and retrying — without losing their place in the training flow.

    Returns (instance_id, ip, port, user, price, ssh_key, gpu_specs).
    gpu_specs is a dict with gpu_type, num_gpus, vram_per_gpu_gb, total_vram_gb.
    """
    _print = print if verbose else lambda *a, **k: None

    ssh_key = _ensure_ssh_key(api_key)

    _print(f"  Finding cheapest {num_gpus}x {gpu}...")
    best = _find_cheapest_gpu(
        gpu, num_gpus=num_gpus, api_key=api_key,
        min_vram_gb=min_vram_gb, fallback_gpu_types=fallback_gpu_types,
    )
    price = best["hourly_price"] / 100
    provider = best["cloud"]
    region = best.get("_region", "")
    _print(f"  [green]✓[/]  {provider}  [dim]${price:.2f}/hr  {region}[/]")

    # Get SSH key ID
    _config = {}
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            _config = json.load(f)
    _ssh_key_id = _config.get("shadeform_ssh_key_id")

    # Launch with retry on recoverable errors
    max_retries = 3
    for attempt in range(max_retries):
        _print(f"  Launching instance...")
        try:
            instance_id = _launch_instance(
                provider, region, best["shade_instance_type"],
                f"konash-{int(time.time())}", _ssh_key_id,
                best.get("configuration", {}).get("os_options"),
                api_key,
            )
            break
        except RuntimeError as e:
            msg = str(e)
            if "needs funds" in msg or "INSUFFICIENT_FUNDS" in msg:
                _print()
                _print(f"  [yellow]{e}[/]")
                _print()
                try:
                    from rich.prompt import Prompt
                    Prompt.ask(
                        "  Press Enter once you've added funds",
                        default="",
                    )
                except (ImportError, EOFError):
                    input("  Press Enter once you've added funds: ")
                continue
            elif attempt < max_retries - 1:
                _print(f"  [yellow]Launch failed: {msg}[/]")
                _print(f"  [dim]Retrying with a different provider...[/]")
                # Try next cheapest GPU
                best = _find_cheapest_gpu(
                    gpu, num_gpus=num_gpus, api_key=api_key,
                    min_vram_gb=min_vram_gb,
                    fallback_gpu_types=fallback_gpu_types,
                )
                provider = best["cloud"]
                region = best.get("_region", "")
                continue
            raise

    _save_instance_state(instance_id, "", hourly_price=price)

    _print(f"  Waiting for instance to boot...")
    try:
        info = _poll_until_active(instance_id, api_key=api_key)
    except RuntimeError as e:
        _print(f"  [yellow]Boot failed: {e}[/]")
        _print(f"  [dim]Cleaning up and retrying...[/]")
        _delete_instance(instance_id, api_key)
        # Retry once with a fresh instance
        _print(f"  Launching new instance...")
        instance_id = _launch_instance(
            provider, region, best["shade_instance_type"],
            f"konash-{int(time.time())}", _ssh_key_id,
            best.get("configuration", {}).get("os_options"),
            api_key,
        )
        _save_instance_state(instance_id, "", hourly_price=price)
        _print(f"  Waiting for instance to boot...")
        info = _poll_until_active(instance_id, api_key=api_key)

    ip = info["ip"]
    port = info["ssh_port"]
    user = info["ssh_user"]
    _save_instance_state(instance_id, ip, port, user, price)

    # Extract GPU specs from Shadeform instance config
    config = best.get("configuration", {})
    _num_gpus = int(config.get("num_gpus", num_gpus) or num_gpus)
    _vram_per_gpu = int(config.get("vram_per_gpu_in_gb", 0) or 0)
    gpu_specs = {
        "gpu_type": str(config.get("gpu_type", gpu) or gpu),
        "num_gpus": _num_gpus,
        "vram_per_gpu_gb": _vram_per_gpu,
        "total_vram_gb": _vram_per_gpu * _num_gpus,
    }
    _print(
        f"  [green]✓[/]  Instance active  [dim]{ip}  "
        f"{gpu_specs['num_gpus']}x {gpu_specs['gpu_type']} "
        f"({gpu_specs['total_vram_gb']} GB)[/]"
    )

    return instance_id, ip, port, user, price, ssh_key, gpu_specs


def train_oapl_from_rollouts(
    *,
    rollouts_path: str,
    base_model: str = "zai-org/GLM-4.5-Air",
    checkpoint_dir: str = "~/.konash/projects/konash-run/checkpoints",
    learning_rate: float = 1e-6,
    cloud: Optional[str] = None,
    gpu: str = "H100",
    use_spot: bool = False,
    keep_alive: bool = False,
    verbose: bool = True,
) -> dict:
    """Run OAPL training on a cloud GPU from pre-generated rollouts."""
    _print = print if verbose else lambda *a, **k: None

    api_key = _ensure_shadeform(verbose)
    instance_id, ip, port, user, price, ssh_key, _gpu_specs = _provision_gpu(gpu, api_key, verbose)

    try:
        # Upload codebase
        _print(f"  Uploading codebase...")
        _upload_codebase(ip, ssh_key, port, user)
        _print(f"  [green]✓[/]  Code uploaded")

        # Install dependencies
        _print(f"  Installing dependencies (this takes a few minutes first time)...")
        _setup_remote(ip, ssh_key, port, user)
        _print(f"  [green]✓[/]  Dependencies installed")

        # Upload rollout data
        _print(f"  Uploading rollout data...")
        _scp_upload(ip, rollouts_path, f"{_REMOTE_DIR}/rollouts.json", ssh_key, port, user)
        _print(f"  [green]✓[/]  Rollouts uploaded")

        # Run training
        training_cmd = (
            f"python3 scripts/train_oapl_unsloth.py "
            f"--rollouts {_REMOTE_DIR}/rollouts.json "
            f"--lr {learning_rate} "
            f"--output {_REMOTE_DIR}/checkpoints"
        )
        env_vars = {"TOGETHER_API_KEY": os.environ.get("TOGETHER_API_KEY", "")}
        together_key = _get_together_key()
        if together_key:
            env_vars["TOGETHER_API_KEY"] = together_key

        _run_remote_training(ip, training_cmd, env_vars, ssh_key, port, user, verbose)

        # Download checkpoints
        _print(f"  Downloading trained model...")
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        _scp_download(ip, f"{_REMOTE_DIR}/checkpoints/", checkpoint_dir, ssh_key, port, user)
        _print(f"  [green]✓[/]  Model downloaded to {checkpoint_dir}")

    finally:
        if not keep_alive:
            gpu_time = time.monotonic() - (time.time() - _load_instance_state().get("created_at", time.time()))
            est_cost = max(gpu_time, 0) / 3600 * price
            _print(f"  Shutting down GPU...  [dim]~${est_cost:.2f} estimated[/]")
            _delete_instance(instance_id, api_key)
            _clear_instance_state()
            _print(f"  [green]✓[/]  GPU released")

    # Load training stats
    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "completed"}


def train_remote(
    *,
    corpus: str = "financebench",
    base_model: str = "zai-org/GLM-4.5-Air",
    checkpoint_dir: str = "~/.konash/projects/konash-run/checkpoints",
    iterations: int = 1,
    synthesis_calls: int = 1500,
    rollouts_per_example: int = 8,
    rollout_max_steps: int = 50,
    max_examples: Optional[int] = None,
    learning_rate: float = 1e-6,
    cloud: Optional[str] = None,
    gpu: str = "H100",
    use_spot: bool = False,
    push_to_hub: Optional[str] = None,
    keep_alive: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full KARL training pipeline on a cloud GPU."""
    _print = print if verbose else lambda *a, **k: None

    api_key = _ensure_shadeform(verbose)
    instance_id, ip, port, user, price, ssh_key, _gpu_specs = _provision_gpu(gpu, api_key, verbose)

    try:
        _print(f"  Uploading codebase...")
        _upload_codebase(ip, ssh_key, port, user)
        _print(f"  [green]✓[/]  Code uploaded")

        _print(f"  Installing dependencies...")
        _setup_remote(ip, ssh_key, port, user)
        _print(f"  [green]✓[/]  Dependencies installed")

        # For named corpora, download on remote. For local paths, upload.
        named_corpora = {"financebench", "browsecomp-plus", "qampari", "freshstack"}
        if corpus in named_corpora:
            from konash.benchmarks import get_dataset

            dl_cmd = f"cd {_REMOTE_DIR} && PYTHONPATH=. python3 -c \"from konash.download import download_{corpus.replace('-', '_')}; download_{corpus.replace('-', '_')}()\""
            subprocess.run(
                _ssh_cmd(ip, dl_cmd, ssh_key, port, user),
                capture_output=True, timeout=600,
            )
            dataset = get_dataset(corpus)
            corpus_path = f"/root/.konash/corpora/{dataset.root_dirname}/{dataset.content_subdir}"
        else:
            _print(f"  Uploading corpus...")
            _scp_upload(ip, corpus, f"{_REMOTE_DIR}/corpus/", ssh_key, port, user)
            corpus_path = f"{_REMOTE_DIR}/corpus"

        training_cmd = (
            f"python3 scripts/train_oapl_unsloth.py "
            f"--corpus {corpus_path} "
            f"--iterations {iterations} "
            f"--synthesis-calls {synthesis_calls} "
            f"--rollouts-per-example {rollouts_per_example} "
            f"--rollout-max-steps {rollout_max_steps} "
            f"--lr {learning_rate} "
            f"--output {_REMOTE_DIR}/checkpoints"
        )
        if max_examples is not None:
            training_cmd += f" --max-examples {max_examples}"
        env_vars = {}
        together_key = _get_together_key()
        if together_key:
            env_vars["TOGETHER_API_KEY"] = together_key

        _run_remote_training(ip, training_cmd, env_vars, ssh_key, port, user, verbose)

        _print(f"  Downloading trained model...")
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        _scp_download(ip, f"{_REMOTE_DIR}/checkpoints/", checkpoint_dir, ssh_key, port, user)
        _print(f"  [green]✓[/]  Model downloaded")

    finally:
        if not keep_alive:
            _print(f"  Shutting down GPU...")
            _delete_instance(instance_id, api_key)
            _clear_instance_state()
            _print(f"  [green]✓[/]  GPU released")

    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "completed"}


def tear_down(verbose: bool = True) -> None:
    """Tear down the active training instance."""
    _print = print if verbose else lambda *a, **k: None
    state = _load_instance_state()
    if not state:
        _print("  No active training instance.")
        return
    _print(f"  Shutting down {state.get('ip', '?')}...")
    _delete_instance(state["instance_id"])
    _clear_instance_state()
    _print(f"  [green]✓[/]  GPU released")


def stream_logs(verbose: bool = True) -> None:
    """Stream training logs from the active instance."""
    state = _load_instance_state()
    if not state:
        print("  No active training instance.")
        return
    ip = state["ip"]
    port = state.get("ssh_port", 22)
    user = state.get("ssh_user", "shadeform")
    key_path = state.get("ssh_key_path", _SSH_KEY_PATH)

    tail_cmd = f"tail -f {_REMOTE_LOG} 2>/dev/null || echo 'No training log yet.'"
    try:
        subprocess.run(
            _ssh_cmd(ip, tail_cmd, key_path, port, user),
            check=True,
        )
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

_VLLM_MODEL = "zai-org/GLM-4.5-Air"
_VLLM_PORT = 8000


def _install_vllm_remote(
    ip: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Install vLLM on the remote machine."""
    install_cmd = (
        "pip3 install vllm 2>&1 | tail -3"
    )
    result = subprocess.run(
        _ssh_cmd(ip, install_cmd, key_path, port, user),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"vLLM install failed: {result.stderr}")


def _start_vllm_server(
    ip: str, model: str = _VLLM_MODEL,
    tensor_parallel: int = 1,
    lora_rank: int = 16,
    key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Start vLLM server with sleep mode + LoRA support on the remote machine."""
    hf_token = os.environ.get("HF_TOKEN", "")
    env_str = (
        f'HF_TOKEN="{hf_token}" '
        f'VLLM_SERVER_DEV_MODE=1 '
        f'VLLM_ALLOW_RUNTIME_LORA_UPDATING=1'
    )

    serve_cmd = (
        f"export {env_str} && "
        f"nohup vllm serve {model} "
        f"--tensor-parallel-size {tensor_parallel} "
        f"--port {_VLLM_PORT} "
        f"--host 0.0.0.0 "
        f"--max-model-len 4096 "
        f"--enable-auto-tool-choice "
        f"--tool-call-parser hermes "
        f"--enable-sleep-mode "
        f"--enable-lora "
        f"--max-lora-rank {lora_rank} "
        f"> ~/vllm.log 2>&1 &"
    )
    subprocess.run(
        _ssh_cmd(ip, serve_cmd, key_path, port, user),
        capture_output=True, timeout=30,
    )


def _sleep_vllm_server(
    ip: str, key_path: str = _SSH_KEY_PATH,
    ssh_port: int = 22, user: str = "shadeform",
) -> None:
    """Put the remote vLLM server to sleep (offload weights to CPU)."""
    cmd = f"curl -s -X POST http://localhost:{_VLLM_PORT}/sleep?level=1"
    subprocess.run(
        _ssh_cmd(ip, cmd, key_path, ssh_port, user),
        capture_output=True, timeout=30,
    )


def _wake_vllm_server(
    ip: str, key_path: str = _SSH_KEY_PATH,
    ssh_port: int = 22, user: str = "shadeform",
) -> None:
    """Wake the remote vLLM server (reload weights from CPU)."""
    cmd = f"curl -s -X POST http://localhost:{_VLLM_PORT}/wake_up"
    subprocess.run(
        _ssh_cmd(ip, cmd, key_path, ssh_port, user),
        capture_output=True, timeout=120,
    )


def _load_lora_adapter_remote(
    ip: str, adapter_path: str, lora_name: str,
    key_path: str = _SSH_KEY_PATH,
    ssh_port: int = 22, user: str = "shadeform",
) -> bool:
    """Hot-load a LoRA adapter into the remote vLLM server."""
    payload = json.dumps({"lora_name": lora_name, "lora_path": adapter_path})
    cmd = (
        f"curl -s -X POST http://localhost:{_VLLM_PORT}/v1/load_lora_adapter "
        f"-H 'Content-Type: application/json' "
        f"-d '{payload}'"
    )
    result = subprocess.run(
        _ssh_cmd(ip, cmd, key_path, ssh_port, user),
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def _stop_vllm_server(
    ip: str, key_path: str = _SSH_KEY_PATH,
    port: int = 22, user: str = "shadeform",
) -> None:
    """Stop the vLLM server on the remote machine."""
    subprocess.run(
        _ssh_cmd(ip, "pkill -f 'vllm serve' || true", key_path, port, user),
        capture_output=True, timeout=10,
    )


def _wait_for_vllm_ready(
    ip: str, timeout: int = 600,
    ssh_port: int = 22, user: str = "shadeform",
    key_path: str = _SSH_KEY_PATH,
) -> None:
    """Poll the vLLM health endpoint until the server is ready."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        # Check via SSH curl since the port may not be publicly exposed
        check_cmd = (
            f"curl -s -o /dev/null -w '%{{http_code}}' "
            f"http://localhost:{_VLLM_PORT}/health 2>/dev/null || echo 000"
        )
        result = subprocess.run(
            _ssh_cmd(ip, check_cmd, key_path, ssh_port, user),
            capture_output=True, text=True, timeout=15,
        )
        code = result.stdout.strip()
        if code == "200":
            return
        time.sleep(10)
    raise RuntimeError(f"vLLM server did not become ready within {timeout}s")


def _probe_vllm_concurrency(
    ip: str, key_path: str = _SSH_KEY_PATH,
    ssh_port: int = 22, user: str = "shadeform",
) -> Optional[int]:
    """Probe a running vLLM server's Prometheus metrics for actual KV cache capacity.

    After vLLM loads the model and allocates KV cache, it reports the actual
    number of available GPU blocks via ``vllm:num_gpu_blocks``.  This is more
    accurate than the pre-load estimate because it accounts for:

    - Actual model weight memory (may differ from our estimate)
    - Embedding layers and other allocations
    - vLLM's internal overhead and reserved memory
    - ``gpu_memory_utilization`` setting

    Returns None if metrics are unavailable (estimate_concurrency is used instead).
    """
    check_cmd = (
        f"curl -s http://localhost:{_VLLM_PORT}/metrics 2>/dev/null"
    )
    try:
        result = subprocess.run(
            _ssh_cmd(ip, check_cmd, key_path, ssh_port, user),
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout:
            return None
    except Exception:
        return None

    text = result.stdout
    total_blocks = 0
    for line in text.split("\n"):
        if line.startswith("#"):
            continue
        # vllm:num_gpu_blocks{...} <value>
        if "num_gpu_blocks" in line:
            try:
                total_blocks = int(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass

    if total_blocks <= 0:
        return None

    # vLLM uses 16-token blocks by default.
    # For max_model_len=4096: 4096/16 = 256 blocks per sequence.
    # Use 80% of capacity for headroom.
    blocks_per_seq = 256  # 4096 tokens / 16 tokens per block
    max_seqs = int(total_blocks / blocks_per_seq * 0.8)
    return max(min(max_seqs, 256), 4)


def estimate_concurrency(
    gpu_specs: dict,
    model: str = _VLLM_MODEL,
    max_model_len: int = 4096,
) -> int:
    """Estimate optimal vLLM concurrency from Shadeform GPU specs.

    Uses the actual GPU memory reported by Shadeform to calculate how
    many concurrent sequences the KV cache can hold. This adapts
    automatically to any GPU type (H100, H200, B200, B300, etc.).

    Parameters
    ----------
    gpu_specs : dict
        From ``_provision_gpu``: gpu_type, num_gpus, vram_per_gpu_gb,
        total_vram_gb.
    model : str
        HuggingFace model ID. Used to estimate model weight memory.
    max_model_len : int
        Max sequence length configured for vLLM.

    Returns
    -------
    int
        Recommended number of concurrent requests.
    """
    total_vram_gb = gpu_specs.get("total_vram_gb", 0)
    num_gpus = gpu_specs.get("num_gpus", 1) or 1

    if total_vram_gb <= 0:
        # No VRAM info from Shadeform — conservative fallback
        return 32

    # Look up model weight memory. Falls back to 60% of total VRAM.
    model_memory_gb = _model_weight_gb(model)
    if model_memory_gb <= 0:
        model_memory_gb = total_vram_gb * 0.6

    # Remaining VRAM available for KV cache
    kv_cache_gb = max(total_vram_gb - model_memory_gb, 1)

    # KV cache memory per sequence:
    # GLM 4.5 Air: 72 layers, hidden=5120, 40 heads, head_dim=128
    # KV per layer per token = 2 * head_dim * num_kv_heads * dtype_bytes
    # MoE models use GQA with fewer KV heads than attention heads.
    # GLM 4.5 Air: 8 KV heads, 128 head_dim, FP16 KV cache = 2 bytes
    # Per token per layer: 2 (K+V) * 128 * 8 * 2 bytes = 4096 bytes = 4 KB
    # Per token all layers: 4 KB * 72 = 288 KB
    # Per sequence (max_model_len tokens): 288 KB * 4096 = 1.15 GB
    kv_per_seq_gb = 288 * max_model_len / (1024 * 1024)  # KB * tokens → GB

    if kv_per_seq_gb <= 0:
        return 32

    # Max concurrent sequences from KV cache capacity
    # Use 80% to leave headroom for activations and overhead
    max_seqs = int(kv_cache_gb * 0.8 / kv_per_seq_gb)

    # Clamp to reasonable bounds
    concurrency = max(min(max_seqs, 256), 4)

    return concurrency


def provision_vllm(
    *,
    gpu: str = "H100",
    num_gpus: int = 2,
    model: str = _VLLM_MODEL,
    verbose: bool = True,
) -> dict:
    """Provision a Shadeform GPU and start a vLLM server.

    Automatically falls back to other GPU types (H200, B200, B300) if
    the preferred type is unavailable. Ensures minimum VRAM for the model.

    Returns dict with instance_id, ip, ssh_port, ssh_user, ssh_key,
    vllm_base_url, hourly_price, gpu_specs, and concurrency.
    """
    _print = print if verbose else lambda *a, **k: None

    api_key = _ensure_shadeform(verbose)
    min_vram = _min_vram_for_model(model)
    instance_id, ip, ssh_port, user, price, ssh_key, gpu_specs = _provision_gpu(
        gpu, api_key, verbose, num_gpus=num_gpus,
        min_vram_gb=min_vram,
        fallback_gpu_types=_VLLM_COMPATIBLE_GPUS,
    )
    _print(f"  Minimum VRAM required: {min_vram} GB")

    _print(f"  Installing vLLM...")
    _install_vllm_remote(ip, ssh_key, ssh_port, user)
    _print(f"  [green]✓[/]  vLLM installed")

    tp_size = gpu_specs["num_gpus"] or num_gpus
    _print(f"  Starting vLLM server ({model}, tp={tp_size})...")
    _start_vllm_server(
        ip, model=model, tensor_parallel=tp_size,
        key_path=ssh_key, port=ssh_port, user=user,
    )

    _print(f"  Waiting for model to load (this takes a few minutes)...")
    _wait_for_vllm_ready(ip, ssh_port=ssh_port, user=user, key_path=ssh_key)

    # Compute concurrency from Shadeform specs as initial estimate,
    # then refine with vLLM's live metrics (actual KV cache blocks
    # after model loading, embedding allocation, etc.)
    concurrency = estimate_concurrency(gpu_specs, model)

    live_concurrency = _probe_vllm_concurrency(ip, ssh_key, ssh_port, user)
    if live_concurrency is not None:
        _print(
            f"  [green]✓[/]  vLLM ready  [dim]{ip}:{_VLLM_PORT}  "
            f"concurrency={live_concurrency} "
            f"(live, was {concurrency} estimated)[/]"
        )
        concurrency = live_concurrency
    else:
        _print(
            f"  [green]✓[/]  vLLM ready  [dim]{ip}:{_VLLM_PORT}  "
            f"concurrency={concurrency} (estimated)[/]"
        )

    return {
        "instance_id": instance_id,
        "ip": ip,
        "ssh_port": ssh_port,
        "ssh_user": user,
        "ssh_key": ssh_key,
        "vllm_base_url": f"http://{ip}:{_VLLM_PORT}/v1",
        "hourly_price": price,
        "gpu_specs": gpu_specs,
        "concurrency": concurrency,
    }


def _get_together_key() -> Optional[str]:
    """Resolve Together AI key from env or config."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            return json.load(f).get("together_api_key")
    return None
