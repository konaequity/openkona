#!/usr/bin/env python3
"""Launch a detached FinanceBench smoke test on Shadeform + vLLM.

This script provisions a single remote GPU, uploads the current repo snapshot,
starts a vLLM server for a Qwen2.5 model, and runs a detached
``konash eval financebench`` smoke test against that server.

The instance is intentionally left running after the eval starts so the GPU
stays alive even if the eval process fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from konash.cloud import _find_cheapest_gpu, _launch_instance, _poll_until_active


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path.home() / ".konash" / "config.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_GPU_CANDIDATES = ["RTX4090", "L4", "L40S", "A100", "H100"]
ARCHIVE_EXCLUDES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "build",
    "eval_results",
    ".venv",
    "venv",
}


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def _pick_ssh_key_id(config: dict, requested: str | None) -> str:
    if requested:
        return requested
    default_key = "66f14b43-e883-4ad4-b230-91199bc16429"
    return default_key


def _pick_ssh_key_path(requested: str | None) -> Path:
    if requested:
        return Path(requested).expanduser().resolve()
    return (Path.home() / ".ssh" / "id_ed25519").resolve()


def _pick_gpu(api_key: str, candidates: list[str]) -> tuple[str, dict]:
    offers: list[tuple[float, str, dict]] = []
    errors: list[str] = []
    for gpu in candidates:
        try:
            offer = _find_cheapest_gpu(gpu, api_key)
        except Exception as exc:  # pragma: no cover - network/runtime path
            errors.append(f"{gpu}: {exc}")
            continue
        offers.append((float(offer["hourly_price"]), gpu, offer))
    if not offers:
        joined = "; ".join(errors) if errors else "no GPU offers found"
        raise RuntimeError(f"No launchable GPU found. {joined}")
    offers.sort(key=lambda item: item[0])
    _, gpu, offer = offers[0]
    return gpu, offer


def _should_skip(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    return any(part in ARCHIVE_EXCLUDES for part in rel_parts)


def _make_archive() -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="konash-smoke-", suffix=".tar.gz")
    os.close(fd)
    archive_path = Path(temp_path)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in ROOT.rglob("*"):
            if _should_skip(path):
                continue
            tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    return archive_path


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def _ssh_base(key_path: Path, user: str, ip: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{ip}",
    ]


def _scp_base(key_path: Path) -> list[str]:
    return [
        "scp",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
    ]


def _quote(value: str) -> str:
    return shlex.quote(value)


def _remote_sh(ssh_base: list[str], script: str) -> None:
    _run([*ssh_base, script])


def _wait_for_vllm(ssh_base: list[str], timeout: int = 900) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        result = _run(
            [
                *ssh_base,
                "bash -lc " + _quote(
                    "curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1"
                ),
            ],
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(5)
    raise RuntimeError("vLLM did not become ready within the timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu", action="append", dest="gpus", default=None,
                        help="Preferred GPU type(s), can be passed multiple times.")
    parser.add_argument("--ssh-key-id", default=None,
                        help="Shadeform SSH key id to use for the new instance.")
    parser.add_argument("--ssh-key-path", default=None,
                        help="Local private key path that matches the registered Shadeform key.")
    parser.add_argument("--instance-name", default="financebench-qwen25-05b-smoke")
    parser.add_argument("--max-model-len", type=int, default=65536)
    args = parser.parse_args()

    config = _load_config()
    api_key = config.get("shadeform_api_key")
    if not api_key:
        raise RuntimeError("Missing shadeform_api_key in ~/.konash/config.json")

    openai_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("Missing HF token for FinanceBench index download")

    gpu_candidates = args.gpus or list(DEFAULT_GPU_CANDIDATES)
    chosen_gpu, offer = _pick_gpu(api_key, gpu_candidates)
    print(
        f"Launching {chosen_gpu} via {offer['cloud']} in {offer.get('_region', '')} "
        f"at ${float(offer['hourly_price']) / 100:.2f}/hr",
        flush=True,
    )

    ssh_key_id = _pick_ssh_key_id(config, args.ssh_key_id)
    instance_id = _launch_instance(
        offer["cloud"],
        offer.get("_region", ""),
        offer["shade_instance_type"],
        args.instance_name,
        ssh_key_id,
        offer.get("configuration", {}).get("os_options"),
        api_key,
    )
    info = _poll_until_active(instance_id, timeout=600, api_key=api_key)
    ip = info["ip"]
    user = info["ssh_user"]
    key_path = _pick_ssh_key_path(args.ssh_key_path)

    print(f"Instance: {instance_id}", flush=True)
    print(f"SSH: ssh -i {key_path} {user}@{ip}", flush=True)

    archive_path = _make_archive()
    try:
        remote_archive = "~/konash_repo.tar.gz"
        print(f"Uploading repo snapshot: {archive_path}", flush=True)
        _run([*_scp_base(key_path), str(archive_path), f"{user}@{ip}:{remote_archive}"])

        ssh_base = _ssh_base(key_path, user, ip)
        setup_script = f"""bash -lc {shlex.quote(f'''
set -euo pipefail
mkdir -p ~/openkona
tar xzf ~/konash_repo.tar.gz -C ~/openkona
mkdir -p /ephemeral/hf_cache ~/.cache
rm -rf ~/.cache/huggingface 2>/dev/null || true
ln -sfn /ephemeral/hf_cache ~/.cache/huggingface
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv pip install --system vllm datasets huggingface_hub rich numpy
cd ~/openkona
uv pip install --system .
''' )}"""
        print("Installing remote dependencies", flush=True)
        _remote_sh(ssh_base, setup_script)

        download_script = f"""bash -lc {shlex.quote(f'''
set -euo pipefail
cd ~/openkona
export PYTHONPATH=~/openkona
export HF_TOKEN={_quote(hf_token)}
python3 -c "from konash.download import download_financebench; from rich.console import Console; download_financebench(console=Console())"
''' )}"""
        print("Downloading FinanceBench corpus", flush=True)
        _remote_sh(ssh_base, download_script)

        vllm_script = f"""bash -lc {shlex.quote(f'''
set -euo pipefail
export PATH="$PATH:/home/shadeform/.local/bin"
cd ~/openkona
nohup vllm serve {args.model} \\
  --port 8000 \\
  --host 0.0.0.0 \\
  --max-model-len {args.max_model_len} \\
  --gpu-memory-utilization 0.90 \\
  --enable-auto-tool-choice \\
  --tool-call-parser hermes \\
  > ~/vllm.log 2>&1 &
echo $! > ~/vllm.pid
''' )}"""
        print("Starting vLLM", flush=True)
        _remote_sh(ssh_base, vllm_script)
        _wait_for_vllm(ssh_base)

        eval_cmd = [
            "cd ~/openkona",
            "export PATH=\"$PATH:/home/shadeform/.local/bin\"",
            "export PYTHONPATH=~/openkona",
            f"export HF_TOKEN={_quote(hf_token)}",
        ]
        if openai_key:
            eval_cmd.append(f"export OPENAI_API_KEY={_quote(openai_key)}")
        eval_cmd.extend([
            "nohup konash eval financebench "
            "--provider vllm "
            "--api-base http://localhost:8000/v1 "
            f"--model {shlex.quote(args.model)} "
            f"--workers {args.workers} "
            f"--limit {args.limit} "
            "> ~/smoke_eval.log 2>&1 &",
            "echo $! > ~/smoke_eval.pid",
        ])
        eval_script = f"bash -lc {shlex.quote(chr(10).join(eval_cmd))}"
        print("Starting detached FinanceBench smoke test", flush=True)
        _remote_sh(ssh_base, eval_script)

        print()
        print("Smoke test launched.")
        print(f"Instance id: {instance_id}")
        print(f"IP: {ip}")
        print("Monitor:")
        print(f"  ssh -i {key_path} {user}@{ip} 'tail -f ~/smoke_eval.log'")
        print(f"  ssh -i {key_path} {user}@{ip} 'tail -f ~/vllm.log'")
        print(f"  ssh -i {key_path} {user}@{ip} 'cat ~/smoke_eval.pid ~/vllm.pid'")
        print(f"  ssh -i {key_path} {user}@{ip} 'ps -fp $(cat ~/smoke_eval.pid) -p $(cat ~/vllm.pid)'")
        print("Teardown:")
        print(
            "  python - <<'PY'\n"
            "from konash.cloud import _delete_instance\n"
            f"_delete_instance('{instance_id}')\n"
            "PY"
        )
        return 0
    finally:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - operator workflow
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
