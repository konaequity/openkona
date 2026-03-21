"""Shared typed model catalog for KONASH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TOGETHER_API_BASE = "https://api.together.xyz/v1"
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_m: float
    output_per_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "input_per_m": self.input_per_m,
            "output_per_m": self.output_per_m,
        }


@dataclass(frozen=True, slots=True)
class ModelPreset:
    base_model: str
    description: str
    temperature: float = 0.7
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    pricing: Optional[ModelPricing] = None
    use_unsloth: bool = False
    load_in_fp8: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "base_model": self.base_model,
            "description": self.description,
            "temperature": self.temperature,
        }
        if self.api_base:
            payload["api_base"] = self.api_base
        if self.api_key_env:
            payload["api_key_env"] = self.api_key_env
        if self.pricing:
            payload["pricing"] = self.pricing.to_dict()
        if self.use_unsloth:
            payload["use_unsloth"] = True
        if self.load_in_fp8:
            payload["load_in_fp8"] = True
        return payload


@dataclass(frozen=True, slots=True)
class CliModelOption:
    id: str
    name: str
    hint: str


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    key: str
    base_model: str
    name: str
    description: str
    hint: str = ""
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.7
    pricing: Optional[ModelPricing] = None
    use_unsloth: bool = False
    load_in_fp8: bool = False
    arena_visible: bool = False
    cli_visible: bool = False

    def to_preset(self) -> ModelPreset:
        return ModelPreset(
            base_model=self.base_model,
            description=self.description,
            temperature=self.temperature,
            api_base=self.api_base,
            api_key_env=self.api_key_env,
            pricing=self.pricing,
            use_unsloth=self.use_unsloth,
            load_in_fp8=self.load_in_fp8,
        )

    def to_cli_option(self) -> CliModelOption:
        return CliModelOption(
            id=self.base_model,
            name=self.name,
            hint=self.hint,
        )


_MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        key="glm-5",
        base_model="zai-org/GLM-5",
        name="GLM 5",
        description="GLM 5",
        hint="Frontier  ·  latest generation  ·  strongest",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
    ),
    ModelCatalogEntry(
        key="qwen3.5-397b",
        base_model="Qwen/Qwen3.5-397B-A17B",
        name="Qwen 3.5 397B",
        description="Qwen 3.5 397B MoE",
        hint="MoE  ·  largest open MoE  ·  frontier-scale",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
    ),
    ModelCatalogEntry(
        key="minimax-m2.5",
        base_model="MiniMaxAI/MiniMax-M2.5",
        name="MiniMax M2.5",
        description="MiniMax M2.5 on Together AI",
        hint="Frontier MoE  ·  strong general reasoning  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="kimi-k2.5",
        base_model="moonshotai/Kimi-K2.5",
        name="Kimi K2.5",
        description="Kimi K2.5",
        hint="MoE  ·  reasoning-focused  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
    ),
    ModelCatalogEntry(
        key="glm-4.7",
        base_model="zai-org/GLM-4.7",
        name="GLM 4.7",
        description="GLM 4.7",
        hint="Frontier  ·  prior generation  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
    ),
    ModelCatalogEntry(
        key="qwen3.5-9b",
        base_model="Qwen/Qwen3.5-9B",
        name="Qwen 3.5 9B",
        description="Qwen 3.5 9B",
        hint="Dense 9B  ·  fast and cheap  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
    ),
    ModelCatalogEntry(
        key="deepseek-r1",
        base_model="deepseek-ai/DeepSeek-R1",
        name="DeepSeek R1",
        description="DeepSeek R1",
        hint="Reasoning model  ·  671B MoE  ·  chain-of-thought",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="qwen3-80b-a3b",
        base_model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        name="Qwen3 80B-A3B",
        description="Qwen3 80B-A3B MoE",
        hint="MoE  ·  3B active params  ·  very cheap",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="llama-3.3-70b-turbo",
        base_model="meta-llama/Llama-3.3-70B-Turbo",
        name="Llama 3.3 70B Turbo",
        description="Llama 3.3 70B Turbo",
        hint="Dense 70B  ·  strong reasoning  ·  moderate cost",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="mixtral-8x22b",
        base_model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        name="Mixtral 8x22B",
        description="Mixtral 8x22B MoE",
        hint="MoE  ·  176B total / 39B active  ·  balanced",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="qwen-2.5-72b",
        base_model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        name="Qwen 2.5 72B Turbo",
        description="Qwen 2.5 72B Turbo",
        hint="Dense 72B  ·  multilingual  ·  moderate cost",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        arena_visible=True,
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="glm-4.5-air-together",
        base_model="zai-org/GLM-4.5-Air-FP8",
        name="GLM 4.5 Air",
        description="GLM 4.5 Air (106B MoE, 12B active) on Together AI",
        hint="Frontier MoE  ·  best for KARL  ·  fast + cheap",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
        pricing=ModelPricing(input_per_m=0.20, output_per_m=1.10),
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="glm-4.5-air-vllm",
        base_model="zai-org/GLM-4.5-Air",
        name="GLM 4.5 Air (vLLM)",
        description="GLM 4.5 Air served via vLLM on Shadeform GPU",
        hint="Self-hosted  ·  vLLM on Shadeform  ·  no API fees",
        api_base="http://localhost:8000/v1",
        pricing=ModelPricing(input_per_m=0.0, output_per_m=0.0),
        cli_visible=True,
    ),
    ModelCatalogEntry(
        key="glm-4.5-air-zhipu",
        base_model="glm-4.5-air",
        name="GLM 4.5 Air (Zhipu)",
        description="GLM 4.5 Air on Zhipu (native provider)",
        api_base=ZHIPU_API_BASE,
        api_key_env="ZHIPU_API_KEY",
    ),
)


def get_model_catalog() -> list[ModelCatalogEntry]:
    return list(_MODEL_CATALOG)


def get_model_presets() -> dict[str, ModelPreset]:
    return {model.key: model.to_preset() for model in _MODEL_CATALOG}


def get_cli_models() -> list[CliModelOption]:
    return [model.to_cli_option() for model in _MODEL_CATALOG if model.cli_visible]


def get_arena_preset_order() -> list[str]:
    return [model.key for model in _MODEL_CATALOG if model.arena_visible]
