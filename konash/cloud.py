"""Cloud GPU training via SkyPilot.

Provisions GPU instances on any supported cloud (RunPod, Lambda, AWS, GCP, Azure,
etc.), uploads the KONASH codebase, runs OAPL training, and retrieves checkpoints.

Requires: ``pip install konash[cloud]``
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Path to the SkyPilot task YAML shipped with the package
_SKY_YAML = Path(__file__).parent / "sky" / "train.yaml"


def _check_skypilot() -> None:
    """Verify SkyPilot is installed and at least one cloud is enabled."""
    try:
        import sky  # noqa: F401
    except ImportError:
        raise SystemExit(
            "SkyPilot is not installed.\n"
            "Install with: pip install konash[cloud]\n"
            "Then configure a provider: https://docs.skypilot.co/en/latest/getting-started/installation.html"
        )


def _get_together_key() -> str:
    """Resolve Together AI API key from env or config."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key
    config_path = os.path.expanduser("~/.konash/config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            key = json.load(f).get("together_api_key")
    if not key:
        raise SystemExit("No Together AI key. Run: konash setup")
    return key


def launch(
    *,
    corpus: str = "financebench",
    cluster_name: str = "konash",
    cloud: Optional[str] = None,
    gpu: str = "H100:1",
    iterations: int = 1,
    rollouts_per_example: int = 8,
    learning_rate: float = 1e-6,
    push_to_hub: Optional[str] = None,
    use_spot: bool = False,
    idle_minutes: int = 10,
) -> None:
    """Launch a training job on a cloud GPU.

    Parameters
    ----------
    corpus : str
        Corpus name (``"financebench"``) or path to documents folder.
    cluster_name : str
        SkyPilot cluster name (default: ``"konash"``).
    cloud : str | None
        Cloud provider (``"runpod"``, ``"lambda"``, ``"aws"``, ``"gcp"``, etc.).
        None lets SkyPilot pick the cheapest available.
    gpu : str
        GPU spec (default: ``"H100:1"``).
    iterations : int
        Training iterations.
    rollouts_per_example : int
        Rollouts per QA pair.
    learning_rate : float
        OAPL learning rate.
    push_to_hub : str | None
        HuggingFace repo to push the trained adapter.
    use_spot : bool
        Use spot/preemptible instances (cheaper but may be interrupted).
    idle_minutes : int
        Auto-stop after idle minutes (default: 10).
    """
    _check_skypilot()

    together_key = _get_together_key()
    hf_token = os.environ.get("HF_TOKEN", "")

    cmd = [
        "sky", "launch", "-c", cluster_name,
        str(_SKY_YAML),
        "--env", f"TOGETHER_API_KEY={together_key}",
        "--env", f"HF_TOKEN={hf_token}",
        "--env", f"CORPUS_PATH={corpus}",
        "--env", f"ITERATIONS={iterations}",
        "--env", f"ROLLOUTS_PER_EXAMPLE={rollouts_per_example}",
        "--env", f"LEARNING_RATE={learning_rate}",
        "--env", f"PUSH_TO_HUB={push_to_hub or ''}",
        "-i", str(idle_minutes),
        "-y",
    ]

    if cloud:
        cmd.extend(["--cloud", cloud])

    if use_spot:
        cmd.append("--use-spot")

    # Override GPU if different from YAML default
    if gpu != "H100:1":
        cmd.extend(["--gpus", gpu])

    subprocess.run(cmd, check=True)


def status() -> None:
    """Show status of all SkyPilot clusters."""
    _check_skypilot()
    subprocess.run(["sky", "status"], check=True)


def logs(cluster_name: str = "konash") -> None:
    """Stream logs from a training cluster."""
    _check_skypilot()
    subprocess.run(["sky", "logs", cluster_name], check=True)


def stop(cluster_name: str = "konash") -> None:
    """Stop a training cluster (can be restarted)."""
    _check_skypilot()
    subprocess.run(["sky", "stop", cluster_name, "-y"], check=True)


def down(cluster_name: str = "konash") -> None:
    """Tear down a training cluster permanently."""
    _check_skypilot()
    subprocess.run(["sky", "down", cluster_name, "-y"], check=True)


def download_checkpoint(
    cluster_name: str = "konash",
    remote_path: str = "~/sky_workdir/checkpoints",
    local_path: str = "./checkpoints",
) -> None:
    """Download training checkpoints from a cluster."""
    _check_skypilot()
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(
        ["rsync", "-avz", f"{cluster_name}:{remote_path}/", local_path],
        check=True,
    )


def show_gpus(cloud: Optional[str] = None) -> None:
    """Show available GPUs across clouds."""
    _check_skypilot()
    cmd = ["sky", "show-gpus"]
    if cloud:
        cmd.extend(["--cloud", cloud])
    subprocess.run(cmd, check=True)
