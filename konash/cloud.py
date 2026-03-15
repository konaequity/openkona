"""Cloud GPU training via SkyPilot.

Provisions a GPU, runs the full KARL pipeline (synthesis + rollouts + OAPL
training with iterative bootstrapping), and downloads the trained adapter.
Everything runs on the GPU so the trained model from iteration 1 becomes
the synthesizer for iteration 2 — matching the KARL paper exactly.

SkyPilot is a core dependency: ``pip install konash``
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

# SkyPilot task YAML shipped with the package
_SKY_YAML = Path(__file__).parent / "sky" / "train.yaml"


def train_remote(
    *,
    corpus: str = "financebench",
    base_model: str = "unsloth/GLM-4.5-Air",
    checkpoint_dir: str = ".konash/default/checkpoints",
    iterations: int = 1,
    rollouts_per_example: int = 8,
    learning_rate: float = 1e-6,
    cloud: Optional[str] = None,
    gpu: str = "H100:1",
    use_spot: bool = False,
    push_to_hub: Optional[str] = None,
    keep_alive: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full KARL training pipeline on a cloud GPU.

    The entire pipeline runs on the GPU — synthesis, rollouts, OAPL
    training, and iterative bootstrapping (the trained model from
    iteration N becomes the synthesizer for iteration N+1).

    1. Provisions a cloud GPU via SkyPilot (cheapest available)
    2. Uploads KONASH codebase
    3. Runs train_oapl_unsloth.py --corpus (full pipeline)
    4. Downloads trained LoRA adapter + value model
    5. Tears down the GPU (unless keep_alive=True)

    Parameters
    ----------
    corpus : str
        Corpus name (``"financebench"``) or path to documents.
    base_model : str
        Unsloth model ID.
    checkpoint_dir : str
        Where to save the downloaded adapter locally.
    iterations : int
        Training iterations (KARL paper uses 2).
    rollouts_per_example : int
        Rollouts per QA pair (KARL paper uses 8).
    learning_rate : float
        OAPL learning rate.
    cloud : str | None
        Force a specific cloud provider. None = cheapest available.
    gpu : str
        GPU spec (default: ``"H100:1"``).
    use_spot : bool
        Use spot instances (cheaper).
    push_to_hub : str | None
        HuggingFace repo to push the trained adapter.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Training stats from the remote run.
    """
    _print = print if verbose else lambda *a, **k: None

    try:
        import sky  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "SkyPilot not found. Reinstall konash:\n"
            "  pip install konash\n"
            "Then configure a GPU provider:\n"
            "  pip install runpod && runpod config\n"
            "  sky check"
        )

    # Resolve Together AI key
    together_key = os.environ.get("TOGETHER_API_KEY", "")
    if not together_key:
        config_path = os.path.expanduser("~/.konash/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                together_key = json.load(f).get("together_api_key", "")

    # Build SkyPilot launch command
    cluster_name = "konash-train"
    cmd = [
        "sky", "launch", "-c", cluster_name,
        str(_SKY_YAML),
        "--env", f"TOGETHER_API_KEY={together_key}",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--env", f"CORPUS_PATH={corpus}",
        "--env", "ROLLOUTS_PATH=",
        "--env", f"ITERATIONS={iterations}",
        "--env", f"ROLLOUTS_PER_EXAMPLE={rollouts_per_example}",
        "--env", f"LEARNING_RATE={learning_rate}",
        "--env", f"PUSH_TO_HUB={push_to_hub or ''}",
        "-i", "5",  # auto-stop after 5 min idle
        "-y",       # skip confirmation
    ]

    if cloud:
        cmd.extend(["--cloud", cloud])
    if use_spot:
        cmd.append("--use-spot")
    if gpu != "H100:1":
        cmd.extend(["--gpus", gpu])

    _print(f"  Provisioning {gpu} on {cloud or 'cheapest cloud'}...")
    _print(f"  Full pipeline: synthesis → rollouts → OAPL × {iterations} iterations")
    _print(f"  Iterative bootstrapping: trained model becomes next synthesizer")

    # Launch and wait for completion
    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        raise RuntimeError(
            f"Cloud training failed (exit {result.returncode}). "
            f"Check logs with: sky logs {cluster_name}"
        )

    # Download checkpoint from remote
    _print(f"  Downloading trained adapter...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    subprocess.run(
        ["rsync", "-avz", f"{cluster_name}:~/sky_workdir/checkpoints/", checkpoint_dir + "/"],
        capture_output=not verbose,
    )

    # Tear down unless caller wants to keep the cluster for more iterations
    if not keep_alive:
        _print(f"  Tearing down GPU cluster...")
        subprocess.run(["sky", "down", cluster_name, "-y"], capture_output=True)

    # Load and return training stats
    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    return {"status": "completed"}


def train_oapl_from_rollouts(
    *,
    rollouts_path: str,
    base_model: str = "unsloth/GLM-4.5-Air",
    checkpoint_dir: str = ".konash/default/checkpoints",
    learning_rate: float = 1e-6,
    cloud: Optional[str] = None,
    gpu: str = "H100:1",
    use_spot: bool = False,
    keep_alive: bool = False,
    verbose: bool = True,
) -> dict:
    """Run OAPL training only (no synthesis) on a cloud GPU.

    Used for iteration 1 where synthesis + rollouts ran locally.
    Uploads pre-generated rollout data, trains, downloads adapter.

    Parameters
    ----------
    rollouts_path : str
        Local path to the rollout checkpoint JSON.
    """
    _print = print if verbose else lambda *a, **k: None

    try:
        import sky  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "SkyPilot not found. Reinstall konash:\n"
            "  pip install konash\n"
            "Then configure a GPU provider:\n"
            "  pip install runpod && runpod config\n"
            "  sky check"
        )

    together_key = os.environ.get("TOGETHER_API_KEY", "")
    if not together_key:
        config_path = os.path.expanduser("~/.konash/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                together_key = json.load(f).get("together_api_key", "")

    cluster_name = "konash-train"
    cmd = [
        "sky", "launch", "-c", cluster_name,
        str(_SKY_YAML),
        "--env", f"TOGETHER_API_KEY={together_key}",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--env", f"ROLLOUTS_PATH={rollouts_path}",
        "--env", "CORPUS_PATH=none",
        "--env", f"LEARNING_RATE={learning_rate}",
        "--env", "ITERATIONS=1",
        "--env", "PUSH_TO_HUB=",
        "-i", "5",
        "-y",
    ]

    if cloud:
        cmd.extend(["--cloud", cloud])
    if use_spot:
        cmd.append("--use-spot")
    if gpu != "H100:1":
        cmd.extend(["--gpus", gpu])

    _print(f"  Provisioning {gpu} for OAPL training (rollouts already generated)...")

    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        raise RuntimeError(
            f"Cloud OAPL training failed (exit {result.returncode}). "
            f"Check logs with: sky logs {cluster_name}"
        )

    _print(f"  Downloading trained adapter...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    subprocess.run(
        ["rsync", "-avz", f"{cluster_name}:~/sky_workdir/checkpoints/", checkpoint_dir + "/"],
        capture_output=not verbose,
    )

    if not keep_alive:
        _print(f"  Tearing down GPU cluster...")
        subprocess.run(["sky", "down", cluster_name, "-y"], capture_output=True)

    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"status": "completed"}


def tear_down(cluster_name: str = "konash-train", verbose: bool = True) -> None:
    """Tear down the training cluster."""
    _print = print if verbose else lambda *a, **k: None
    _print(f"  Tearing down GPU cluster...")
    subprocess.run(["sky", "down", cluster_name, "-y"], capture_output=True)


def show_gpus(cloud: Optional[str] = None) -> None:
    """Show available GPUs across clouds."""
    try:
        import sky  # noqa: F401
    except ImportError:
        print("SkyPilot not installed. Run: pip install konash")
        return
    cmd = ["sky", "show-gpus"]
    if cloud:
        cmd.extend(["--cloud", cloud])
    subprocess.run(cmd)
