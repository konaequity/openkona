from __future__ import annotations

import json
from argparse import Namespace

import pytest

from konash.eval.harness import HF_ROUTER_API_BASE, resolve_provider


def _args(**overrides):
    base = {
        "provider": None,
        "api_key": None,
        "api_base": None,
        "model": None,
        "judge_key": None,
        "judge_model": "gpt-4o-mini",
    }
    base.update(overrides)
    return Namespace(**base)


def test_resolve_provider_supports_huggingface_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    provider = resolve_provider(
        _args(provider="hf", model="meta-llama/Llama-3.1-8B-Instruct")
    )

    assert provider["provider"] == "hf"
    assert provider["solver_api_base"] == HF_ROUTER_API_BASE
    assert provider["solver_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert provider["api_key"] == "hf_test_token"
    assert provider["judge_api_base"] == HF_ROUTER_API_BASE
    assert provider["judge_key"] == "hf_test_token"


def test_resolve_provider_autodetects_huggingface_when_model_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_alt_token")

    provider = resolve_provider(
        _args(model="Qwen/Qwen2.5-7B-Instruct")
    )

    assert provider["provider"] == "hf"
    assert provider["solver_api_base"] == HF_ROUTER_API_BASE
    assert provider["api_key"] == "hf_alt_token"


def test_resolve_provider_reads_huggingface_token_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    config_dir = tmp_path / ".konash"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"hf_token": "hf_config_token"}))

    provider = resolve_provider(
        _args(provider="hf", model="google/gemma-2-9b-it")
    )

    assert provider["provider"] == "hf"
    assert provider["api_key"] == "hf_config_token"


def test_resolve_provider_reads_openai_judge_key_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "together_token")
    config_dir = tmp_path / ".konash"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({"openai_api_key": "openai_config_token"}))

    provider = resolve_provider(_args(provider="together"))

    assert provider["judge_api_base"] == "https://api.openai.com/v1"
    assert provider["judge_key"] == "openai_config_token"


def test_resolve_provider_requires_model_for_huggingface(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    with pytest.raises(SystemExit):
        resolve_provider(_args(provider="hf"))
