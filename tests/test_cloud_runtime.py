from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from konash.cloud import (
    _provision_gpu,
    _run_remote_training,
    _setup_remote_minimal,
    _start_remote_runtime_install,
)
from konash.training.backends import ShadeformSynthesisBackend


def _load_train_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_oapl_unsloth.py"
    spec = importlib.util.spec_from_file_location("train_oapl_unsloth_for_tests", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_provision_gpu_retry_preserves_num_gpus():
    best = {
        "hourly_price": 100,
        "cloud": "provider-a",
        "_region": "region-1",
        "shade_instance_type": "type-a",
        "configuration": {"os_options": ["ubuntu22.04"]},
    }
    second_best = {
        "hourly_price": 125,
        "cloud": "provider-b",
        "_region": "region-2",
        "shade_instance_type": "type-b",
        "configuration": {"os_options": ["ubuntu22.04"]},
    }

    with patch("konash.cloud._ensure_ssh_key", return_value="/tmp/test-key"), \
         patch("konash.cloud._find_cheapest_gpu", side_effect=[best, second_best]) as mock_find, \
         patch("konash.cloud._launch_instance", side_effect=[RuntimeError("launch failed"), "inst-1"]), \
         patch("konash.cloud._save_instance_state"), \
         patch("konash.cloud._poll_until_active", return_value={"ip": "1.2.3.4", "ssh_port": 22, "ssh_user": "shadeform"}):
        _provision_gpu("H100", "api-key", verbose=False, num_gpus=2)

    assert mock_find.call_args_list[0].kwargs["num_gpus"] == 2
    assert mock_find.call_args_list[1].kwargs["num_gpus"] == 2


def test_shadeform_backend_provision_uses_tensor_parallel_gpu_count():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    with patch("konash.cloud._provision_gpu", return_value=("inst", "1.2.3.4", 22, "shadeform", 1.23, "/tmp/key")) as mock_provision:
        backend._provision("zai-org/GLM-4.5-Air-FP8")

    assert mock_provision.call_args.kwargs["num_gpus"] == 2


def test_verify_health_raises_early_when_vllm_exits():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._ip = "1.2.3.4"

    responses = [
        MagicMock(returncode=7, stdout="", stderr="connection refused"),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="RuntimeError: CUDA out of memory", stderr=""),
    ]

    with patch.object(backend, "_ssh_run", side_effect=responses), \
         patch("konash.training.backends.time.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            backend._verify_health("zai-org/GLM-4.5-Air-FP8", timeout=5, poll_interval=0)


def test_vllm_generate_fn_retries_on_context_length():
    module = _load_train_script_module()
    generate_fn = module._build_vllm_generate_fn("http://localhost:8000/v1", "test-model")

    class FakeErrorBody:
        def read(self):
            return b"maximum context length exceeded"

        def close(self):
            return None

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"SEARCH: retry works"}}]}'

    responses = [
        urllib.error.HTTPError(
            url="http://localhost:8000/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=FakeErrorBody(),
        ),
        FakeResponse(),
    ]
    captured_bodies: list[dict] = []

    def fake_urlopen(req, timeout=600):
        captured_bodies.append(module.json.loads(req.data.decode()))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = generate_fn([{"role": "user", "content": "test"}], max_tokens=1024)

    assert result["content"] == "SEARCH: retry works"
    assert captured_bodies[0]["max_tokens"] == 1024
    assert captured_bodies[1]["max_tokens"] == 512


def test_run_remote_training_quotes_display_name_and_touches_log():
    recorded_ssh_commands: list[str] = []

    def fake_ssh_cmd(ip, cmd, key_path, port, user):
        recorded_ssh_commands.append(cmd)
        return ["ssh", ip, cmd]

    tail_proc = SimpleNamespace(
        stdout=iter(["##KONASH:complete##\n"]),
        terminate=MagicMock(),
        wait=MagicMock(return_value=0),
    )

    with patch("konash.cloud._ssh_cmd", side_effect=fake_ssh_cmd), \
         patch("konash.cloud.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=b"", stderr=b"")), \
         patch("konash.cloud.subprocess.Popen", return_value=tail_proc), \
         patch("konash.cloud.time.sleep", return_value=None):
        _run_remote_training(
            "1.2.3.4",
            "python3 scripts/train_oapl_unsloth.py --project browsecomp --display-name 'BrowseComp-Plus on GLM-4.5-Air-FP8'",
            {},
            verbose=False,
        )

    launch_cmd = recorded_ssh_commands[0]
    tail_cmd = recorded_ssh_commands[1]

    assert "touch ~/konash/training.log" in launch_cmd
    assert "nohup bash -lc" in launch_cmd
    assert "'\"'\"'BrowseComp-Plus on GLM-4.5-Air-FP8'\"'\"'" in launch_cmd
    assert tail_cmd == "tail -n +1 -F ~/konash/training.log"


def test_run_remote_training_recovers_from_slow_ssh_launcher():
    recorded_ssh_commands: list[str] = []

    def fake_ssh_cmd(ip, cmd, key_path, port, user):
        recorded_ssh_commands.append(cmd)
        return ["ssh", ip, cmd]

    tail_proc = SimpleNamespace(
        stdout=iter(["##KONASH:complete##\n"]),
        terminate=MagicMock(),
        wait=MagicMock(return_value=0),
    )

    run_results = [
        subprocess.TimeoutExpired(cmd=["ssh"], timeout=30),
        SimpleNamespace(returncode=0, stdout="READY\n", stderr=""),
    ]

    def fake_run(*args, **kwargs):
        result = run_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("konash.cloud._ssh_cmd", side_effect=fake_ssh_cmd), \
         patch("konash.cloud.subprocess.run", side_effect=fake_run), \
         patch("konash.cloud.subprocess.Popen", return_value=tail_proc), \
         patch("konash.cloud.time.sleep", return_value=None):
        _run_remote_training(
            "1.2.3.4",
            "python3 scripts/train_oapl_unsloth.py --project browsecomp",
            {},
            verbose=False,
        )

    assert "test -f ~/konash/training.log && echo READY" in recorded_ssh_commands[1]
    assert recorded_ssh_commands[2] == "tail -n +1 -F ~/konash/training.log"


def test_setup_remote_minimal_only_installs_bootstrap_packages():
    captured = {}

    def fake_run(ip, remote_cmd, **kwargs):
        captured["cmd"] = remote_cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("konash.cloud._run_remote_blocking", side_effect=fake_run):
        _setup_remote_minimal("1.2.3.4")

    assert "uv pip install --system --link-mode=copy --upgrade numpy datasets huggingface_hub 'tqdm<4.67'" in captured["cmd"]
    assert "sentence-transformers" not in captured["cmd"]
    assert captured["kwargs"]["timeout"] == 600


def test_start_remote_runtime_install_installs_heavy_training_stack():
    captured = {}

    def fake_start(ip, **kwargs):
        captured["cmd"] = kwargs["remote_cmd"]
        return SimpleNamespace()

    with patch("konash.cloud._start_remote_async", side_effect=fake_start):
        _start_remote_runtime_install("1.2.3.4")

    assert "sentence-transformers torch unsloth peft accelerate" in captured["cmd"]
    assert "'vllm>=0.8' 'transformers>=5.2.0'" in captured["cmd"]
    assert "numpy datasets huggingface_hub" not in captured["cmd"]
