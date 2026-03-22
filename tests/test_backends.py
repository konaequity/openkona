"""Tests for konash.training.backends."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from konash.training.backends import (
    OpenAIConfig,
    ShadeformSynthesisBackend,
    SynthesisRuntimeBackend,
    VLLMLifecycle,
)


# ---------------------------------------------------------------------------
# OpenAIConfig
# ---------------------------------------------------------------------------


def test_openai_config_required_fields():
    cfg = OpenAIConfig(
        api_base="http://1.2.3.4:8000/v1",
        api_key="test-key",
        model_name="zai-org/GLM-4.5-Air-FP8",
    )
    assert cfg.api_base == "http://1.2.3.4:8000/v1"
    assert cfg.api_key == "test-key"
    assert cfg.model_name == "zai-org/GLM-4.5-Air-FP8"


def test_openai_config_optional_fields_default():
    cfg = OpenAIConfig(
        api_base="http://localhost:8000/v1",
        api_key="k",
        model_name="m",
    )
    assert cfg.runtime_id == ""
    assert cfg.model_revision == ""
    assert cfg.metadata == {}


def test_openai_config_all_fields():
    cfg = OpenAIConfig(
        api_base="http://1.2.3.4:8000/v1",
        api_key="test-key",
        model_name="zai-org/GLM-4.5-Air-FP8",
        runtime_id="inst-abc",
        model_revision="iter1-ckpt",
        metadata={"provider": "shadeform", "reused": True},
    )
    assert cfg.runtime_id == "inst-abc"
    assert cfg.model_revision == "iter1-ckpt"
    assert cfg.metadata["provider"] == "shadeform"


def test_openai_config_is_frozen():
    cfg = OpenAIConfig(api_base="x", api_key="k", model_name="m")
    with pytest.raises(Exception):
        cfg.api_base = "y"


# ---------------------------------------------------------------------------
# ShadeformSynthesisBackend — construction
# ---------------------------------------------------------------------------


def test_shadeform_backend_requires_api_key():
    """Missing API key should raise immediately at construction time."""
    with patch.dict(os.environ, {}, clear=True), \
         patch("konash.training.backends.os.path.exists", return_value=False):
        with pytest.raises(RuntimeError, match="No Shadeform API key"):
            ShadeformSynthesisBackend()


def test_shadeform_backend_accepts_explicit_key():
    backend = ShadeformSynthesisBackend(api_key="test-key", verbose=False)
    assert backend._api_key == "test-key"


# ---------------------------------------------------------------------------
# Reuse logic — revision safety
# ---------------------------------------------------------------------------


def test_try_reuse_blocks_when_served_revision_but_caller_empty():
    """Stale reuse prevention: runtime has revision, caller doesn't specify."""
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "zai-org/GLM-4.5-Air-FP8"
    backend._served_revision = "iter1-ckpt"

    result = backend._try_reuse("zai-org/GLM-4.5-Air-FP8", "")
    assert result is False


def test_try_reuse_blocks_on_revision_mismatch():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "zai-org/GLM-4.5-Air-FP8"
    backend._served_revision = "iter1-ckpt"

    result = backend._try_reuse("zai-org/GLM-4.5-Air-FP8", "iter2-ckpt")
    assert result is False


def test_try_reuse_blocks_on_model_mismatch():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "zai-org/GLM-4.5-Air-FP8"
    backend._served_revision = ""

    result = backend._try_reuse("some-other/model", "")
    assert result is False


def test_try_reuse_allows_when_both_revisions_empty():
    """Fresh base model, no checkpoints — reuse is safe."""
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "zai-org/GLM-4.5-Air-FP8"
    backend._served_revision = ""

    # Mock the instance + health checks
    with patch.object(backend, "_check_instance_active", return_value=True), \
         patch.object(backend, "_check_vllm_health", return_value=True):
        result = backend._try_reuse("zai-org/GLM-4.5-Air-FP8", "")
    assert result is True


def test_try_reuse_allows_when_revisions_match():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "zai-org/GLM-4.5-Air-FP8"
    backend._served_revision = "iter1-ckpt"

    with patch.object(backend, "_check_instance_active", return_value=True), \
         patch.object(backend, "_check_vllm_health", return_value=True):
        result = backend._try_reuse("zai-org/GLM-4.5-Air-FP8", "iter1-ckpt")
    assert result is True


# ---------------------------------------------------------------------------
# Teardown clears state
# ---------------------------------------------------------------------------


def test_teardown_clears_state():
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    backend._instance_id = "inst-1"
    backend._ip = "1.2.3.4"
    backend._served_model = "model"
    backend._config = OpenAIConfig(api_base="x", api_key="k", model_name="m")

    with patch("konash.training.backends._delete_instance"):
        backend.teardown()

    assert backend._instance_id is None
    assert backend._ip is None
    assert backend._served_model == ""
    assert backend._config is None


# ---------------------------------------------------------------------------
# OpenAIConfig smoke test — can be passed to Agent constructor
# ---------------------------------------------------------------------------


def test_openai_config_usable_as_agent_kwargs():
    """Verify OpenAIConfig fields can be destructured into Agent(api_base=, api_key=)."""
    cfg = OpenAIConfig(
        api_base="http://1.2.3.4:8000/v1",
        api_key="not-needed",
        model_name="zai-org/GLM-4.5-Air-FP8",
    )
    # These are the exact kwargs the CLI passes to Agent()
    kwargs = {"api_base": cfg.api_base, "api_key": cfg.api_key}
    assert kwargs["api_base"] == "http://1.2.3.4:8000/v1"
    assert kwargs["api_key"] == "not-needed"


# ---------------------------------------------------------------------------
# VLLMLifecycle
# ---------------------------------------------------------------------------


def test_vllm_lifecycle_urls():
    vllm = VLLMLifecycle(model="test-model", port=8000)
    assert vllm.base_url == "http://localhost:8000"
    assert vllm.api_url == "http://localhost:8000/v1"


def test_vllm_lifecycle_custom_port():
    vllm = VLLMLifecycle(model="test-model", port=9999)
    assert vllm.base_url == "http://localhost:9999"
    assert vllm.api_url == "http://localhost:9999/v1"


def test_vllm_lifecycle_stores_config():
    vllm = VLLMLifecycle(
        model="zai-org/GLM-4.5-Air-FP8",
        tensor_parallel=2,
        max_model_len=131072,
        max_lora_rank=32,
    )
    assert vllm._model == "zai-org/GLM-4.5-Air-FP8"
    assert vllm._tensor_parallel == 2
    assert vllm._max_model_len == 131072
    assert vllm._max_lora_rank == 32


def test_vllm_lifecycle_sleep_uses_longer_timeout():
    vllm = VLLMLifecycle(model="test-model", port=8000)
    with patch.object(vllm, "_post") as mock_post:
        vllm.sleep()
    mock_post.assert_called_once_with("/sleep?level=1", timeout=180)


# ---------------------------------------------------------------------------
# ABC sleep/wake/load_lora defaults
# ---------------------------------------------------------------------------


def test_abc_sleep_wake_defaults_are_noop():
    """Default ABC methods should not raise."""
    backend = ShadeformSynthesisBackend(api_key="k", verbose=False)
    # sleep/wake inherited from ABC are overridden, but load_lora default
    # on the ABC should return None
    assert SynthesisRuntimeBackend.load_lora(backend, "/path") is None


# ---------------------------------------------------------------------------
# ShadeformSynthesisBackend sleep/wake via SSH
# ---------------------------------------------------------------------------


def test_shadeform_sleep_calls_ssh():
    backend = ShadeformSynthesisBackend(
        api_key="k", verbose=False, sleep_mode=True,
    )
    backend._ip = "1.2.3.4"
    backend._instance_id = "inst-1"
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = MagicMock(returncode=0)
        backend.sleep()
    mock_ssh.assert_called_once()
    assert "sleep?level=1" in mock_ssh.call_args[0][0]


def test_shadeform_wake_calls_ssh_and_polls():
    backend = ShadeformSynthesisBackend(
        api_key="k", verbose=False, sleep_mode=True,
    )
    backend._ip = "1.2.3.4"
    backend._instance_id = "inst-1"
    with patch.object(backend, "_ssh_run") as mock_ssh, \
         patch.object(backend, "_check_vllm_health", return_value=True):
        mock_ssh.return_value = MagicMock(returncode=0)
        backend.wake()
    assert "wake_up" in mock_ssh.call_args[0][0]


def test_shadeform_load_lora_returns_name():
    backend = ShadeformSynthesisBackend(
        api_key="k", verbose=False, sleep_mode=True,
    )
    backend._ip = "1.2.3.4"
    backend._instance_id = "inst-1"
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = MagicMock(returncode=0)
        name = backend.load_lora("/path/to/adapter", lora_name="test-lora")
    assert name == "test-lora"
    assert "load_lora_adapter" in mock_ssh.call_args[0][0]


def test_shadeform_load_lora_returns_none_on_failure():
    backend = ShadeformSynthesisBackend(
        api_key="k", verbose=False, sleep_mode=True,
    )
    backend._ip = "1.2.3.4"
    backend._instance_id = "inst-1"
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = MagicMock(returncode=1, stderr="connection refused")
        name = backend.load_lora("/path/to/adapter")
    assert name is None


def test_shadeform_sleep_mode_flag():
    """sleep_mode=True should be stored for _start_vllm."""
    backend = ShadeformSynthesisBackend(
        api_key="k", verbose=False, sleep_mode=True,
    )
    assert backend._sleep_mode is True

    backend2 = ShadeformSynthesisBackend(
        api_key="k", verbose=False,
    )
    assert backend2._sleep_mode is False
