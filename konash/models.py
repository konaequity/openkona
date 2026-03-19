"""Shared model catalog for KONASH.

This module centralizes model metadata used by the API preset registry,
interactive CLI model pickers, and the arena UI. The goal is to keep model
definitions in one place and derive all view-specific representations from the
same source of truth.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


TOGETHER_API_BASE = "https://api.together.xyz/v1"
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"


_MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "key": "glm-5",
        "base_model": "zai-org/GLM-5",
        "name": "GLM 5",
        "description": "GLM 5",
        "hint": "Frontier  ·  latest generation  ·  strongest",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
    },
    {
        "key": "qwen3.5-397b",
        "base_model": "Qwen/Qwen3.5-397B-A17B",
        "name": "Qwen 3.5 397B",
        "description": "Qwen 3.5 397B MoE",
        "hint": "MoE  ·  largest open MoE  ·  frontier-scale",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
    },
    {
        "key": "minimax-m2.5",
        "base_model": "MiniMaxAI/MiniMax-M2.5",
        "name": "MiniMax M2.5",
        "description": "MiniMax M2.5 on Together AI",
        "hint": "Frontier MoE  ·  strong general reasoning  ·  Together AI",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "kimi-k2.5",
        "base_model": "moonshotai/Kimi-K2.5",
        "name": "Kimi K2.5",
        "description": "Kimi K2.5",
        "hint": "MoE  ·  reasoning-focused  ·  Together AI",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
    },
    {
        "key": "glm-4.7",
        "base_model": "zai-org/GLM-4.7",
        "name": "GLM 4.7",
        "description": "GLM 4.7",
        "hint": "Frontier  ·  prior generation  ·  Together AI",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
    },
    {
        "key": "qwen3.5-9b",
        "base_model": "Qwen/Qwen3.5-9B",
        "name": "Qwen 3.5 9B",
        "description": "Qwen 3.5 9B",
        "hint": "Dense 9B  ·  fast and cheap  ·  Together AI",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
    },
    {
        "key": "deepseek-r1",
        "base_model": "deepseek-ai/DeepSeek-R1",
        "name": "DeepSeek R1",
        "description": "DeepSeek R1",
        "hint": "Reasoning model  ·  671B MoE  ·  chain-of-thought",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "qwen3-80b-a3b",
        "base_model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "name": "Qwen3 80B-A3B",
        "description": "Qwen3 80B-A3B MoE",
        "hint": "MoE  ·  3B active params  ·  very cheap",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "llama-3.3-70b-turbo",
        "base_model": "meta-llama/Llama-3.3-70B-Turbo",
        "name": "Llama 3.3 70B Turbo",
        "description": "Llama 3.3 70B Turbo",
        "hint": "Dense 70B  ·  strong reasoning  ·  moderate cost",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "mixtral-8x22b",
        "base_model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "name": "Mixtral 8x22B",
        "description": "Mixtral 8x22B MoE",
        "hint": "MoE  ·  176B total / 39B active  ·  balanced",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "qwen-2.5-72b",
        "base_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "name": "Qwen 2.5 72B Turbo",
        "description": "Qwen 2.5 72B Turbo",
        "hint": "Dense 72B  ·  multilingual  ·  moderate cost",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "arena_visible": True,
        "cli_visible": True,
    },
    {
        "key": "glm-4.5-air-together",
        "base_model": "zai-org/GLM-4.5-Air-FP8",
        "name": "GLM 4.5 Air",
        "description": "GLM 4.5 Air (106B MoE, 12B active) on Together AI",
        "hint": "Frontier MoE  ·  best for KARL  ·  fast + cheap",
        "api_base": TOGETHER_API_BASE,
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "pricing": {"input_per_m": 0.20, "output_per_m": 1.10},
        "cli_visible": True,
    },
    {
        "key": "glm-4.5-air-unsloth",
        "base_model": "unsloth/GLM-4.5-Air",
        "name": "GLM 4.5 Air (Unsloth)",
        "description": "GLM 4.5 Air via Unsloth (local OAPL training, FP8)",
        "temperature": 0.7,
        "use_unsloth": True,
        "load_in_fp8": True,
    },
    {
        "key": "glm-4.5-air-zhipu",
        "base_model": "glm-4.5-air",
        "name": "GLM 4.5 Air (Zhipu)",
        "description": "GLM 4.5 Air on Zhipu (native provider)",
        "api_base": ZHIPU_API_BASE,
        "api_key_env": "ZHIPU_API_KEY",
        "temperature": 0.7,
    },
]


def get_model_catalog() -> List[Dict[str, Any]]:
    return deepcopy(_MODEL_CATALOG)


def get_model_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    for model in _MODEL_CATALOG:
        key = model["key"]
        preset = {
            "base_model": model["base_model"],
            "temperature": model.get("temperature", 0.7),
            "description": model["description"],
        }
        for field in ("api_base", "api_key_env", "pricing", "use_unsloth", "load_in_fp8"):
            if field in model:
                preset[field] = deepcopy(model[field])
        presets[key] = preset
    return presets


def get_cli_models() -> List[Dict[str, str]]:
    return [
        {
            "id": model["base_model"],
            "name": model["name"],
            "hint": model["hint"],
        }
        for model in _MODEL_CATALOG
        if model.get("cli_visible")
    ]


def get_arena_preset_order() -> List[str]:
    return [model["key"] for model in _MODEL_CATALOG if model.get("arena_visible")]
