"""Synthesis runtime backends.

Owns the lifecycle of the OpenAI-compatible runtime used for Stage 1/2
(synthesis + rollouts).  Higher layers receive a resolved ``OpenAIConfig``
and never touch provisioning, vLLM startup, or health checks directly.

Design rules (from shadeform_synthesis_migration_plan.md):
- The synthesis runtime owns runtime lifecycle.
- Higher layers only consume resolved OpenAI connection info.
- Runtime preparation is explicit — no hiding launch behind generate calls.
- Runtime health is part of correctness — verify /v1/models.
- Reuse of warm runtimes is a core feature.
- The abstraction stays narrow — no registries or plugin systems.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenAI-compatible connection info
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenAIConfig:
    """Resolved connection details for an OpenAI-compatible runtime.

    Returned by ``SynthesisRuntimeBackend.prepare()`` and consumed by the
    rest of KONASH without knowledge of how the runtime was created.
    """

    api_base: str
    """Full base URL, e.g. ``http://1.2.3.4:8000/v1``."""

    api_key: str
    """Bearer token.  May be a placeholder for local runtimes."""

    model_name: str
    """Served model name as reported by ``/v1/models``."""

    runtime_id: str = ""
    """Opaque identifier for the runtime instance (e.g. Shadeform instance ID).
    Used for logging and reuse decisions, not for API calls."""

    model_revision: str = ""
    """Checkpoint or revision identity.  Allows callers to reason about
    whether the runtime is serving a stale model after an RL iteration."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary runtime metadata for observability (provider, region, price, etc.)."""


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class SynthesisRuntimeBackend(ABC):
    """Lifecycle manager for a Stage 1/2 synthesis runtime.

    Concrete implementations own provisioning, startup, health checks,
    reuse, and teardown.  The rest of KONASH only sees ``OpenAIConfig``.
    """

    @abstractmethod
    def prepare(
        self,
        *,
        model: str = "",
        model_revision: str = "",
        force_fresh: bool = False,
    ) -> OpenAIConfig:
        """Ensure a healthy runtime is available and return connection info.

        Parameters
        ----------
        model:
            Model to serve (e.g. ``zai-org/GLM-4.5-Air-FP8``).
        model_revision:
            Checkpoint / adapter identity.  When non-empty and different
            from the currently served revision, the backend should refresh
            the runtime rather than reuse it.
        force_fresh:
            Skip reuse checks and always provision a fresh runtime.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release runtime resources (delete instance, stop process, etc.)."""

    @abstractmethod
    def is_warm(self) -> bool:
        """Return True if a previously prepared runtime is still usable."""

    def refresh(
        self,
        *,
        model: str = "",
        model_revision: str = "",
    ) -> OpenAIConfig:
        """Restart the runtime against a new model or checkpoint.

        Default implementation tears down and re-prepares.  Backends may
        override with a faster hot-swap path (e.g. kill vLLM and restart
        without reprovisioning).

        Parameters
        ----------
        model:
            Model to serve after refresh.  Required for KARL iteration 2+
            where the merged/adapted model replaces the base model.
        model_revision:
            Checkpoint identity for the refreshed runtime.
        """
        self.teardown()
        return self.prepare(
            model=model, model_revision=model_revision, force_fresh=True,
        )

    def sleep(self) -> None:
        """Offload runtime weights to CPU to free GPU VRAM.

        Default implementation is a no-op.  Backends that support sleep mode
        (vLLM with ``--enable-sleep-mode``) should override this.
        """

    def wake(self) -> None:
        """Reload runtime weights from CPU to GPU after sleep.

        Default implementation is a no-op.
        """

    def load_lora(
        self, adapter_path: str, lora_name: str = "",
    ) -> str | None:
        """Hot-load a LoRA adapter into the running runtime.

        Returns the adapter name to use in subsequent requests, or ``None``
        if hot-loading is not supported.
        """
        return None


# ---------------------------------------------------------------------------
# Local vLLM lifecycle (on-GPU usage)
# ---------------------------------------------------------------------------


class VLLMLifecycle:
    """Manages a local vLLM server process with sleep/wake and LoRA support.

    Used by the on-GPU training script to alternate between vLLM inference
    and Unsloth training on a single GPU.

    Lifecycle::

        vllm = VLLMLifecycle(model="zai-org/GLM-4.5-Air-FP8")
        vllm.start()                       # launch + wait for health
        # ... synthesis + rollouts via vllm.api_url ...
        vllm.sleep()                       # offload weights to CPU (~2s)
        # ... Unsloth OAPL training ...
        vllm.wake()                        # reload weights to GPU (~10s)
        vllm.load_lora("iter1/adapter")    # hot-load trained LoRA
        # ... next iteration with updated policy ...
        vllm.stop()
    """

    def __init__(
        self,
        *,
        model: str,
        port: int = 8000,
        tensor_parallel: int = 1,
        max_model_len: int | None = None,
        max_lora_rank: int = 16,
        extra_args: list[str] | None = None,
        log_dir: str = ".",
    ) -> None:
        self._model = model
        self._port = port
        self._tensor_parallel = tensor_parallel
        self._max_model_len = max_model_len
        self._max_lora_rank = max_lora_rank
        self._extra_args = extra_args or []
        self._log_dir = log_dir
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._lora_counter = 0

    # -- Properties --------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    # -- Lifecycle ---------------------------------------------------------

    def start(self, timeout: int = 600) -> None:
        """Launch ``vllm serve`` with sleep mode + LoRA and wait for health."""
        import shutil

        vllm_bin = shutil.which("vllm") or "vllm"
        hf_token = os.environ.get("HF_TOKEN", "")
        env = {
            **os.environ,
            "HF_TOKEN": hf_token,
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
        }

        cmd = [
            vllm_bin, "serve", self._model,
            "--port", str(self._port),
            "--host", "0.0.0.0",
            "--tensor-parallel-size", str(self._tensor_parallel),
            "--enable-sleep-mode",
            "--enable-lora",
            "--max-lora-rank", str(self._max_lora_rank),
            "--enforce-eager",
        ]
        # Let vLLM auto-detect from model config when not specified
        if self._max_model_len is not None:
            cmd.extend(["--max-model-len", str(self._max_model_len)])
        cmd.extend(self._extra_args)

        log_path = os.path.join(self._log_dir, "vllm.log")
        self._log_fh = open(log_path, "w")
        _log.info("Starting vLLM: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd, stdout=self._log_fh, stderr=subprocess.STDOUT,
            env=env,
        )

        # Poll /v1/models until healthy
        self._wait_healthy(timeout)
        _log.info("vLLM ready on port %d (pid %d)", self._port, self._proc.pid)

    def stop(self) -> None:
        """Terminate the vLLM process."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            _log.info("vLLM stopped (pid %d)", self._proc.pid)
            self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def sleep(self) -> None:
        """Offload vLLM weights to CPU pinned memory (~2s)."""
        self._post("/sleep?level=1", timeout=30)
        _log.info("vLLM sleeping — GPU VRAM freed")

    def wake(self, timeout: int = 120) -> None:
        """Reload vLLM weights from CPU to GPU and wait for health."""
        self._post("/wake_up", timeout=timeout)
        self._wait_healthy(timeout)
        _log.info("vLLM awake — GPU VRAM reclaimed")

    def load_lora(self, adapter_path: str) -> str | None:
        """Hot-load a LoRA adapter.  Returns the adapter name or None."""
        self._lora_counter += 1
        lora_name = f"konash-iter-{self._lora_counter}"
        body = json.dumps({
            "lora_name": lora_name,
            "lora_path": os.path.abspath(adapter_path),
        }).encode()
        try:
            self._post(
                "/v1/load_lora_adapter", timeout=60,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            _log.info("Loaded LoRA adapter %s from %s", lora_name, adapter_path)
            return lora_name
        except Exception as exc:
            _log.warning("LoRA hot-load failed: %s", exc)
            return None

    def served_model(self) -> str:
        """Return the model name reported by ``/v1/models``."""
        import urllib.request
        req = urllib.request.Request(f"{self.api_url}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        return models[0]["id"] if models else self._model

    def is_healthy(self) -> bool:
        """Return True if vLLM is responding to health checks."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # -- Internals ---------------------------------------------------------

    def _post(
        self, path: str, *, timeout: int = 30,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        import urllib.request
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, data=data or b"", method="POST")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _wait_healthy(self, timeout: int) -> None:
        """Poll ``/v1/models`` until vLLM responds or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                import urllib.request
                req = urllib.request.Request(f"{self.api_url}/models")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data.get("data"):
                        return
            except Exception:
                pass
            # Check process is still alive
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM process exited with code {self._proc.returncode}"
                )
            time.sleep(10)
        raise RuntimeError(f"vLLM did not become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# Shadeform implementation
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.expanduser("~/.konash")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")
_SSH_KEY_PATH = os.path.join(_CONFIG_DIR, "shadeform_ssh_key")
_SYNTHESIS_RUNTIME_STATE = os.path.join(_CONFIG_DIR, "synthesis_runtime.json")

# vLLM defaults for GLM 4.5 Air
_DEFAULT_VLLM_MODEL = "zai-org/GLM-4.5-Air-FP8"
_DEFAULT_VLLM_PORT = 8000
_DEFAULT_VLLM_ARGS = [
    "--tensor-parallel-size", "2",
    "--max-model-len", "131072",
    "--trust-remote-code",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "glm45",
]

# SSH options (mirrors cloud.py)
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]


class ShadeformSynthesisBackend(SynthesisRuntimeBackend):
    """Manages a Shadeform GPU instance running vLLM for synthesis.

    Lifecycle:
    1. ``prepare()`` provisions (or reuses) a Shadeform instance, installs
       vLLM, starts the server, and verifies ``/v1/models``.
    2. The caller uses the returned ``OpenAIConfig`` for synthesis/rollouts.
    3. ``teardown()`` deletes the instance.

    Warm reuse: if a previous instance is still active and serving the
    right model/revision, ``prepare()`` skips provisioning and returns
    the cached config.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        gpu_type: str = "H100",
        vllm_port: int = _DEFAULT_VLLM_PORT,
        vllm_extra_args: list[str] | None = None,
        sleep_mode: bool = False,
        verbose: bool = True,
    ) -> None:
        self._api_key = api_key or _resolve_shadeform_key()
        if not self._api_key:
            raise RuntimeError(
                "No Shadeform API key.  Set SHADEFORM_API_KEY or run `konash setup`."
            )
        self._gpu_type = gpu_type
        self._vllm_port = vllm_port
        self._vllm_extra_args = vllm_extra_args or list(_DEFAULT_VLLM_ARGS)
        self._sleep_mode = sleep_mode
        self._verbose = verbose

        # Mutable state — populated by prepare()
        self._instance_id: str | None = None
        self._ip: str | None = None
        self._ssh_port: int = 22
        self._ssh_user: str = "shadeform"
        self._ssh_key: str = _SSH_KEY_PATH
        self._served_model: str = ""
        self._served_revision: str = ""
        self._hourly_price: float = 0.0
        self._config: OpenAIConfig | None = None

    # -- Public API --------------------------------------------------------

    def prepare(
        self,
        *,
        model: str = "",
        model_revision: str = "",
        force_fresh: bool = False,
    ) -> OpenAIConfig:
        model = model or _DEFAULT_VLLM_MODEL

        # Try reuse
        if not force_fresh and self._try_reuse(model, model_revision):
            assert self._config is not None
            _log.info(
                "Reusing warm synthesis runtime %s at %s (model=%s)",
                self._instance_id, self._ip, self._served_model,
            )
            self._print(f"  Reusing warm synthesis runtime  [dim]{self._ip}[/]")
            return self._config

        # If reuse failed but we have an active instance, reuse the box
        # for a fresh vLLM setup instead of provisioning a new one.
        import datetime as _dt
        self._provision_started_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

        if self._instance_id and self._ip and self._check_instance_active():
            self._print(f"  Reusing active instance {self._ip} for fresh setup...")
            _log.info(
                "Reuse failed but instance %s is active — doing fresh vLLM setup",
                self._instance_id,
            )
        else:
            # Provision fresh instance
            self._provision(model)

        # Install vLLM (idempotent — skips if already installed)
        self._install_vllm()

        # Start vLLM and verify health
        self._start_vllm(model)
        served_model = self._verify_health(model)

        # Build config
        self._served_model = served_model
        self._served_revision = model_revision
        self._config = OpenAIConfig(
            api_base=f"http://{self._ip}:{self._vllm_port}/v1",
            api_key="not-needed",
            model_name=served_model,
            runtime_id=self._instance_id or "",
            model_revision=model_revision,
            metadata={
                "provider": "shadeform",
                "gpu_type": self._gpu_type,
                "ip": self._ip,
                "hourly_price": self._hourly_price,
                "reused": False,
                "provision_started_at": self._provision_started_at,
            },
        )
        self._save_runtime_state()
        return self._config

    def teardown(self) -> None:
        if self._instance_id:
            self._print(f"  Shutting down synthesis runtime {self._instance_id}...")
            _delete_instance(self._instance_id, self._api_key)
            self._print(f"  Synthesis runtime released")
            _log.info("Deleted synthesis runtime %s", self._instance_id)
        self._clear_state()

    def is_warm(self) -> bool:
        """Check if a previously prepared runtime is still usable."""
        if not self._instance_id or not self._ip:
            # Try loading from persisted state
            if not self._load_runtime_state():
                return False
        return self._check_instance_active() and self._check_vllm_health()

    def refresh(
        self,
        *,
        model: str = "",
        model_revision: str = "",
    ) -> OpenAIConfig:
        """Restart vLLM on the existing instance with a new model/checkpoint.

        Avoids full reprovisioning when the Shadeform instance is still
        active — just kills the vLLM process and restarts it.  Falls back
        to the default teardown+prepare if the instance is gone.
        """
        model = model or _DEFAULT_VLLM_MODEL

        # If instance is still active, restart vLLM in-place
        if self._instance_id and self._ip and self._check_instance_active():
            self._print(f"  Refreshing synthesis runtime (new model: {model})...")
            self._ssh_run("pkill -f 'vllm serve' || true", timeout=15)
            time.sleep(2)  # Let process exit
            self._start_vllm(model)
            served_model = self._verify_health(model)
            self._served_model = served_model
            self._served_revision = model_revision
            self._config = OpenAIConfig(
                api_base=f"http://{self._ip}:{self._vllm_port}/v1",
                api_key="not-needed",
                model_name=served_model,
                runtime_id=self._instance_id or "",
                model_revision=model_revision,
                metadata={
                    "provider": "shadeform",
                    "gpu_type": self._gpu_type,
                    "ip": self._ip,
                    "hourly_price": self._hourly_price,
                    "reused": True,
                    "refreshed": True,
                },
            )
            self._save_runtime_state()
            _log.info(
                "Refreshed synthesis runtime %s: now serving %s (revision=%s)",
                self._instance_id, served_model, model_revision,
            )
            return self._config

        # Instance gone — full teardown + prepare
        _log.info("Instance not active for refresh, falling back to full prepare")
        self._clear_state()
        return self.prepare(
            model=model, model_revision=model_revision, force_fresh=True,
        )

    def sleep(self) -> None:
        """Sleep vLLM on the remote instance via SSH."""
        result = self._ssh_run(
            f"curl -s -X POST http://localhost:{self._vllm_port}/sleep?level=1",
            timeout=30,
        )
        if result.returncode != 0:
            _log.warning("vLLM sleep failed: %s", result.stderr)
        else:
            _log.info("vLLM sleeping on %s", self._ip)

    def wake(self) -> None:
        """Wake vLLM on the remote instance via SSH."""
        self._ssh_run(
            f"curl -s -X POST http://localhost:{self._vllm_port}/wake_up",
            timeout=120,
        )
        for _ in range(30):
            if self._check_vllm_health():
                _log.info("vLLM awake on %s", self._ip)
                return
            time.sleep(2)
        _log.warning("vLLM wake health check timed out on %s", self._ip)

    def load_lora(
        self, adapter_path: str, lora_name: str = "",
    ) -> str | None:
        """Hot-load a LoRA adapter on the remote instance via SSH."""
        if not lora_name:
            lora_name = f"konash-{int(time.time())}"
        body = json.dumps({"lora_name": lora_name, "lora_path": adapter_path})
        result = self._ssh_run(
            f"curl -s -X POST -H 'Content-Type: application/json' "
            f"-d '{body}' http://localhost:{self._vllm_port}/v1/load_lora_adapter",
            timeout=60,
        )
        if result.returncode == 0:
            _log.info("Loaded LoRA %s on %s", lora_name, self._ip)
            return lora_name
        _log.warning("LoRA hot-load failed on %s: %s", self._ip, result.stderr)
        return None

    # -- Provisioning ------------------------------------------------------

    def _provision(self, model: str) -> None:
        """Find cheapest GPU, launch, wait for SSH.

        Delegates to ``cloud._provision_gpu()`` which handles retry logic
        for insufficient funds, launch failures, and boot timeouts.
        """
        from konash.cloud import _provision_gpu

        instance_id, ip, port, user, price, ssh_key = _provision_gpu(
            self._gpu_type, self._api_key, verbose=self._verbose,
        )
        self._instance_id = instance_id
        self._ip = ip
        self._ssh_port = port
        self._ssh_user = user
        self._hourly_price = price
        self._ssh_key = ssh_key
        _log.info(
            "Provisioned synthesis instance %s at %s ($%.2f/hr)",
            self._instance_id, self._ip, self._hourly_price,
        )

    # -- vLLM setup --------------------------------------------------------

    def _install_vllm(self) -> None:
        """Install vLLM and set up HF cache on ephemeral disk.

        Idempotent — skips pip install if vLLM is already available.
        """
        # Check if already installed
        check = self._ssh_run("which vllm 2>/dev/null && echo INSTALLED || echo MISSING", timeout=15)
        if "INSTALLED" in (check.stdout or ""):
            self._print(f"  [green]✓[/]  vLLM already installed")
            return

        self._print(f"  Installing vLLM on synthesis instance...")

        # Symlink HF cache to ephemeral disk (root disk is only ~97GB,
        # model is ~65GB).  Install uv if needed, then install vLLM.
        setup_cmds = (
            "mkdir -p /ephemeral/hf_cache ~/.cache && "
            "rm -rf ~/.cache/huggingface 2>/dev/null; "
            "ln -sf /ephemeral/hf_cache ~/.cache/huggingface && "
            "command -v uv >/dev/null 2>&1 || "
            "(curl -LsSf https://astral.sh/uv/install.sh | sh) && "
            "export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH && "
            "sudo -E env PATH=$PATH uv pip install --system --link-mode=copy vllm 2>&1 | tail -3"
        )
        result = self._ssh_run(setup_cmds, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"vLLM installation failed: {result.stderr}"
            )
        self._print(f"  [green]✓[/]  vLLM installed")

    def _start_vllm(self, model: str) -> None:
        """Start ``vllm serve`` in background via nohup."""
        self._print(f"  Starting vLLM ({model})...")

        args_str = " ".join(self._vllm_extra_args)
        if self._sleep_mode:
            args_str += (
                " --enable-sleep-mode --enable-lora"
                " --max-lora-rank 16 --enforce-eager"
            )
        env_exports = "export PATH=$PATH:$HOME/.local/bin"
        if self._sleep_mode:
            env_exports += (
                " VLLM_SERVER_DEV_MODE=1"
                " VLLM_ALLOW_RUNTIME_LORA_UPDATING=1"
            )
        cmd = (
            f"{env_exports} && "
            f"nohup vllm serve {model} "
            f"{args_str} "
            f"--port {self._vllm_port} "
            f"> ~/vllm_synthesis.log 2>&1 </dev/null &"
        )
        self._ssh_run(cmd, timeout=60)
        _log.info("Started vLLM serve on %s for model %s", self._ip, model)

    def _verify_health(
        self, expected_model: str, timeout: int = 600, poll_interval: int = 10,
    ) -> str:
        """Poll ``/v1/models`` until the endpoint is healthy.

        Returns the model name as reported by the server.
        """
        self._print(f"  Waiting for vLLM to be ready (model download + load)...")

        deadline = time.monotonic() + timeout
        last_err = ""
        while time.monotonic() < deadline:
            try:
                result = self._ssh_run(
                    f"curl -s http://localhost:{self._vllm_port}/v1/models",
                    timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    data = json.loads(result.stdout)
                    models = data.get("data", [])
                    if models:
                        served = models[0].get("id", "")
                        self._print(f"  [green]✓[/]  vLLM serving {served}")
                        _log.info("vLLM healthy on %s, serving %s", self._ip, served)
                        return served
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                last_err = str(exc)
            time.sleep(poll_interval)

        raise RuntimeError(
            f"vLLM did not become healthy within {timeout}s on {self._ip}. "
            f"Last error: {last_err}"
        )

    # -- Reuse logic -------------------------------------------------------

    def _try_reuse(self, model: str, model_revision: str) -> bool:
        """Attempt to reuse an existing warm runtime.

        Returns True if the existing runtime is active, healthy, and
        serving the expected model/revision.
        """
        # First try in-memory state
        if not self._instance_id or not self._ip:
            if not self._load_runtime_state():
                return False

        # Check model compatibility
        if self._served_model and self._served_model != model:
            _log.info(
                "Cannot reuse runtime: model mismatch (have=%s, want=%s)",
                self._served_model, model,
            )
            return False

        # Revision safety: block reuse when revisions are inconsistent.
        # If the runtime was prepared with a revision (e.g. iteration 1
        # checkpoint) and the caller doesn't specify one, that's ambiguous
        # and likely a bug — block reuse to avoid serving a stale model.
        if self._served_revision and not model_revision:
            _log.warning(
                "Blocking reuse: runtime has revision %r but caller "
                "specified none — risk of stale model across RL iterations",
                self._served_revision,
            )
            return False

        if model_revision and self._served_revision != model_revision:
            _log.info(
                "Cannot reuse runtime: revision mismatch (have=%s, want=%s)",
                self._served_revision, model_revision,
            )
            return False

        # Check instance is still active
        if not self._check_instance_active():
            _log.info("Cannot reuse runtime: instance %s no longer active", self._instance_id)
            self._clear_state()
            return False

        # Check vLLM is still serving; if down, try installing (if needed)
        # and restarting on the existing instance before giving up (avoids
        # reprovisioning when only vLLM crashed or was never set up).
        if not self._check_vllm_health():
            _log.info("vLLM not healthy on %s, attempting recovery", self._ip)
            self._print(f"  vLLM down on warm instance, recovering...")
            self._ssh_run("pkill -f 'vllm serve' || true", timeout=15)
            time.sleep(2)
            # Check if vLLM is installed; if not, install it first
            result = self._ssh_run("which vllm || echo 'NOT_INSTALLED'", timeout=15)
            if "NOT_INSTALLED" in (result.stdout or ""):
                self._print(f"  vLLM not installed, installing...")
                self._install_vllm()
            self._start_vllm(model)
            try:
                self._verify_health(model, timeout=600)
            except RuntimeError:
                _log.info("vLLM recovery failed on %s, cannot reuse", self._ip)
                return False

        # Rebuild config
        self._config = OpenAIConfig(
            api_base=f"http://{self._ip}:{self._vllm_port}/v1",
            api_key="not-needed",
            model_name=self._served_model,
            runtime_id=self._instance_id or "",
            model_revision=self._served_revision,
            metadata={
                "provider": "shadeform",
                "gpu_type": self._gpu_type,
                "ip": self._ip,
                "hourly_price": self._hourly_price,
                "reused": True,
            },
        )
        return True

    def _check_instance_active(self) -> bool:
        """Query Shadeform to see if the instance is still active."""
        if not self._instance_id:
            return False
        try:
            from konash.cloud import _shadeform_request
            data = _shadeform_request(
                "GET",
                f"/instances/{self._instance_id}/info",
                api_key=self._api_key,
            )
            return data.get("status") == "active"
        except Exception as exc:
            _log.debug("Instance status check failed: %s", exc)
            return False

    def _check_vllm_health(self) -> bool:
        """Quick check that /v1/models responds."""
        if not self._ip:
            return False
        try:
            result = self._ssh_run(
                f"curl -s --max-time 5 http://localhost:{self._vllm_port}/v1/models",
                timeout=15,
            )
            if result.returncode != 0:
                return False
            data = json.loads(result.stdout)
            models = data.get("data", [])
            if models:
                self._served_model = models[0].get("id", "")
                return True
        except Exception:
            pass
        return False

    # -- Persistence -------------------------------------------------------

    def _save_runtime_state(self) -> None:
        """Persist runtime info so future prepare() calls can reuse it."""
        os.makedirs(_CONFIG_DIR, exist_ok=True)
        state = {
            "instance_id": self._instance_id,
            "ip": self._ip,
            "ssh_port": self._ssh_port,
            "ssh_user": self._ssh_user,
            "ssh_key": self._ssh_key,
            "served_model": self._served_model,
            "served_revision": self._served_revision,
            "vllm_port": self._vllm_port,
            "gpu_type": self._gpu_type,
            "hourly_price": self._hourly_price,
            "created_at": time.time(),
            "provision_started_at": getattr(self, "_provision_started_at", None),
        }
        with open(_SYNTHESIS_RUNTIME_STATE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_runtime_state(self) -> bool:
        """Load persisted runtime state.  Returns True if state was found."""
        if not os.path.exists(_SYNTHESIS_RUNTIME_STATE):
            return False
        try:
            with open(_SYNTHESIS_RUNTIME_STATE) as f:
                state = json.load(f)
            self._instance_id = state.get("instance_id")
            self._ip = state.get("ip")
            self._ssh_port = state.get("ssh_port", 22)
            self._ssh_user = state.get("ssh_user", "shadeform")
            self._ssh_key = state.get("ssh_key", _SSH_KEY_PATH)
            self._served_model = state.get("served_model", "")
            self._served_revision = state.get("served_revision", "")
            self._vllm_port = state.get("vllm_port", _DEFAULT_VLLM_PORT)
            self._hourly_price = state.get("hourly_price", 0.0)
            return bool(self._instance_id and self._ip)
        except (json.JSONDecodeError, KeyError):
            return False

    def _clear_state(self) -> None:
        """Clear in-memory and persisted runtime state."""
        self._instance_id = None
        self._ip = None
        self._served_model = ""
        self._served_revision = ""
        self._config = None
        if os.path.exists(_SYNTHESIS_RUNTIME_STATE):
            os.remove(_SYNTHESIS_RUNTIME_STATE)

    # -- SSH helpers -------------------------------------------------------

    def _ssh_run(
        self, command: str, timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        """Run a command on the remote instance via SSH."""
        cmd = [
            "ssh", *_SSH_OPTS,
            "-i", self._ssh_key,
            "-p", str(self._ssh_port),
            f"{self._ssh_user}@{self._ip}",
            command,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )

    # -- Output helper -----------------------------------------------------

    def _print(self, msg: str) -> None:
        if self._verbose:
            try:
                from rich.console import Console
                Console().print(msg)
            except ImportError:
                # Strip rich markup for plain output
                import re
                print(re.sub(r"\[/?[^\]]*\]", "", msg))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_shadeform_key() -> str | None:
    """Resolve Shadeform API key from env or config."""
    key = os.environ.get("SHADEFORM_API_KEY")
    if key:
        return key
    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE) as f:
            return json.load(f).get("shadeform_api_key")
    return None


def _delete_instance(instance_id: str, api_key: str | None = None) -> None:
    """Delete a Shadeform instance (best-effort)."""
    try:
        from konash.cloud import _shadeform_request
        _shadeform_request("POST", f"/instances/{instance_id}/delete", api_key=api_key)
    except Exception:
        pass
