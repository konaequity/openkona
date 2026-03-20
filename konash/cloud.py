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

def _find_cheapest_gpu(gpu_type: str = "H100", api_key: Optional[str] = None) -> dict:
    """Find the cheapest available GPU instance."""
    data = _shadeform_request(
        "GET",
        f"/instances/types?gpu_type={gpu_type}&num_gpus=1&available=true&sort=price",
        api_key=api_key,
    )
    instances = data.get("instance_types", [])
    if not instances:
        raise RuntimeError(f"No {gpu_type} GPUs available on Shadeform right now.")

    best = instances[0]
    # Find an available region
    for avail in best.get("availability", []):
        if avail.get("available"):
            best["_region"] = avail["region"]
            break

    return best


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
    gpu: str, api_key: str, verbose: bool = True,
) -> tuple:
    """Find, launch, and wait for a GPU instance.

    Handles insufficient funds by prompting the user to add balance
    and retrying — without losing their place in the training flow.

    Returns (instance_id, ip, port, user, price, ssh_key).
    """
    _print = print if verbose else lambda *a, **k: None

    ssh_key = _ensure_ssh_key(api_key)

    _print(f"  Finding cheapest {gpu}...")
    best = _find_cheapest_gpu(gpu, api_key)
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
                best = _find_cheapest_gpu(gpu, api_key)
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
    _print(f"  [green]✓[/]  Instance active  [dim]{ip}[/]")

    return instance_id, ip, port, user, price, ssh_key


def train_oapl_from_rollouts(
    *,
    rollouts_path: str,
    base_model: str = "unsloth/GLM-4.5-Air",
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
    instance_id, ip, port, user, price, ssh_key = _provision_gpu(gpu, api_key, verbose)

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
    base_model: str = "unsloth/GLM-4.5-Air",
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
    instance_id, ip, port, user, price, ssh_key = _provision_gpu(gpu, api_key, verbose)

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

def _get_together_key() -> Optional[str]:
    """Resolve Together AI key from env or config."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            return json.load(f).get("together_api_key")
    return None
