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


def _has_gpu_provider() -> bool:
    """Check if at least one GPU cloud provider is enabled in SkyPilot."""
    result = subprocess.run(
        ["sky", "check"],
        capture_output=True, text=True, timeout=30,
    )
    # Look for any "enabled" line in the output
    return "enabled" in result.stdout.lower()


def _ensure_gpu_provider(verbose: bool = True) -> None:
    """If no GPU provider is configured, walk the user through setup."""
    _print = print if verbose else lambda *a, **k: None

    if _has_gpu_provider():
        return

    _print()
    _print("  OAPL training needs a GPU. Let's set one up (takes 2 minutes).")
    _print()

    # Use Rich for the selector if available, otherwise fall back
    try:
        from rich.console import Console
        from rich.prompt import Prompt
        _con = Console()

        # Import arrow selector from CLI
        from konash.cli import _arrow_select
        idx = _arrow_select(_con, [
            {"label": "RunPod", "hint": "Easiest setup, good prices ($2.39-2.69/hr H100)"},
            {"label": "Lambda", "hint": "Often cheapest H100s ($2.76/hr)"},
            {"label": "AWS", "hint": "Requires existing AWS account"},
            {"label": "GCP", "hint": "Requires existing GCP account"},
        ])
        _con.print()

        providers = ["runpod", "lambda", "aws", "gcp"]
        provider = providers[idx]

        if provider == "runpod":
            _con.print("    1. Go to [bold]runpod.io[/] → Settings → API Keys")
            _con.print("    2. Create a key and copy it")
            _con.print()

            import webbrowser
            webbrowser.open("https://www.runpod.io/console/user/settings")

            _con.print()
            api_key = Prompt.ask("    Paste your RunPod API key", password=True)
            if api_key:
                os.makedirs(os.path.expanduser("~/.runpod"), exist_ok=True)
                with open(os.path.expanduser("~/.runpod/config.toml"), "w") as f:
                    f.write(f'[default]\napi_key = "{api_key}"\n')
                _con.print("    [green]✓[/]  RunPod configured")
            else:
                raise RuntimeError("No API key provided.")

        elif provider == "lambda":
            _con.print("    1. Go to [bold]lambdalabs.com[/] → API Keys")
            _con.print("    2. Create a key and copy it")
            _con.print()
            _con.print("    Then run: [cyan]sky check lambda[/]")
            _con.print()
            raise RuntimeError(
                "Lambda setup requires manual configuration.\n"
                "See: https://docs.skypilot.co/en/latest/getting-started/installation.html"
            )

        else:
            _con.print(f"    Run: [cyan]sky check {provider}[/]")
            _con.print(
                f"    See: https://docs.skypilot.co/en/latest/getting-started/installation.html"
            )
            raise RuntimeError(f"{provider.upper()} setup requires manual configuration.")

    except ImportError:
        raise RuntimeError(
            "No GPU provider configured. Set one up:\n"
            "  pip install runpod && runpod config\n"
            "  sky check"
        )

    # Verify it worked
    if not _has_gpu_provider():
        raise RuntimeError(
            "GPU provider setup failed. Try manually:\n"
            "  sky check"
        )

    _print("    [green]✓[/]  GPU provider ready")
    _print()


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
            "  pip install konash"
        )

    _ensure_gpu_provider(verbose)

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
            "  pip install konash"
        )

    # Check if any GPU provider is configured — if not, walk them through it
    _ensure_gpu_provider(verbose)

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

    _print(f"  Finding cheapest GPU...")

    # Launch SkyPilot (streams output so user sees provisioning progress)
    import time as _time
    gpu_start = _time.monotonic()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    # Parse output for KONASH phase markers and show clean status
    phase_times: dict = {}
    current_phase = "provisioning"
    phase_start = gpu_start
    oapl_loss = None
    vm_loss = None

    for line in proc.stdout:
        line = line.rstrip()

        if "##KONASH:" in line:
            elapsed = _time.monotonic() - phase_start
            marker = line.split("##KONASH:")[1].rstrip("#")

            if marker == "loading_data":
                phase_times["provisioning"] = elapsed
                _print(f"  [green]✓[/]  GPU provisioned  [dim]{elapsed:.0f}s[/]")
                current_phase = "loading_data"
                phase_start = _time.monotonic()

            elif marker == "loading_model":
                current_phase = "loading_model"
                _print(f"  Loading model...")
                phase_start = _time.monotonic()

            elif marker.startswith("model_loaded:"):
                load_time = marker.split(":")[1]
                _print(f"  [green]✓[/]  Model loaded  [dim]{load_time}[/]")

            elif marker == "oapl_start":
                current_phase = "oapl"
                _print(f"  Training OAPL...")
                phase_start = _time.monotonic()

            elif marker.startswith("oapl_done:"):
                elapsed = _time.monotonic() - phase_start
                oapl_loss = marker.split("loss=")[1] if "loss=" in marker else "?"
                _print(f"  [green]✓[/]  OAPL complete  [dim]loss {oapl_loss}  {elapsed:.0f}s[/]")

            elif marker == "value_model_start":
                current_phase = "value_model"
                _print(f"  Training value model...")
                phase_start = _time.monotonic()

            elif marker.startswith("value_model_done:"):
                elapsed = _time.monotonic() - phase_start
                vm_loss = marker.split("loss=")[1] if "loss=" in marker else "?"
                _print(f"  [green]✓[/]  Value model trained  [dim]loss {vm_loss}  {elapsed:.0f}s[/]")

            elif marker == "complete":
                total_gpu = _time.monotonic() - gpu_start
                _print(f"  [green]✓[/]  GPU training complete  [dim]{total_gpu:.0f}s total[/]")

        elif verbose and not line.startswith("##"):
            # Show SkyPilot output during provisioning only
            if current_phase == "provisioning" and line.strip():
                _print(f"  [dim]{line}[/]")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cloud OAPL training failed (exit {proc.returncode}). "
            f"Check logs with: sky logs {cluster_name}"
        )

    _print(f"  Downloading trained adapter...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    subprocess.run(
        ["rsync", "-avz", f"{cluster_name}:~/sky_workdir/checkpoints/", checkpoint_dir + "/"],
        capture_output=not verbose,
    )

    if not keep_alive:
        total_gpu = _time.monotonic() - gpu_start
        # Rough cost estimate: H100 ~$2.69/hr
        est_cost = (total_gpu / 3600) * 2.69
        _print(f"  [green]✓[/]  GPU released  [dim]~${est_cost:.2f} estimated[/]")
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
