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
