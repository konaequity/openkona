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
    # Together AI models — arena uses live fetch from /v1/models instead.
    # These remain for CLI usage (cli_visible) and preset resolution.
    ModelCatalogEntry(
        key="glm-5",
        base_model="zai-org/GLM-5",
        name="GLM 5",
        description="GLM 5",
        hint="Frontier  ·  latest generation  ·  strongest",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
    ),
    ModelCatalogEntry(
        key="qwen3.5-397b",
        base_model="Qwen/Qwen3.5-397B-A17B",
        name="Qwen 3.5 397B",
        description="Qwen 3.5 397B MoE",
        hint="MoE  ·  largest open MoE  ·  frontier-scale",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
    ),
    ModelCatalogEntry(
        key="minimax-m2.5",
        base_model="MiniMaxAI/MiniMax-M2.5",
        name="MiniMax M2.5",
        description="MiniMax M2.5 on Together AI",
        hint="Frontier MoE  ·  strong general reasoning  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
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
    ),
    ModelCatalogEntry(
        key="glm-4.7",
        base_model="zai-org/GLM-4.7",
        name="GLM 4.7",
        description="GLM 4.7",
        hint="Frontier  ·  prior generation  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
    ),
    ModelCatalogEntry(
        key="qwen3.5-9b",
        base_model="Qwen/Qwen3.5-9B",
        name="Qwen 3.5 9B",
        description="Qwen 3.5 9B",
        hint="Dense 9B  ·  fast and cheap  ·  Together AI",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
    ),
    ModelCatalogEntry(
        key="deepseek-r1",
        base_model="deepseek-ai/DeepSeek-R1",
        name="DeepSeek R1",
        description="DeepSeek R1",
        hint="Reasoning model  ·  671B MoE  ·  chain-of-thought",
        api_base=TOGETHER_API_BASE,
        api_key_env="TOGETHER_API_KEY",
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
        key="glm-4.5-air-unsloth",
        base_model="zai-org/GLM-4.5-Air-FP8",
        name="GLM 4.5 Air (Unsloth)",
        description="GLM 4.5 Air via Unsloth (local OAPL training, FP8)",
        use_unsloth=True,
        load_in_fp8=True,
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


def fetch_together_models(api_key: str = "") -> list[ModelCatalogEntry]:
    """Fetch live chat models from Together AI's ``/v1/models`` endpoint.

    Returns ``ModelCatalogEntry`` objects for every chat model not already
    in the static catalog.  Falls back to an empty list on network errors.
    """
    import json
    import logging
    import urllib.request
    import urllib.error

    _log = logging.getLogger(__name__)

    if not api_key:
        import os
        api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        try:
            import os
            with open(os.path.expanduser("~/.konash/config.json")) as f:
                api_key = json.load(f).get("together_api_key", "")
        except Exception:
            pass
    if not api_key:
        return []

    try:
        req = urllib.request.Request(f"{TOGETHER_API_BASE}/models")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "konash/0.2")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        _log.debug("Failed to fetch Together AI models: %s", exc)
        return []

    raw = data if isinstance(data, list) else data.get("data", [])
    chat_models = [m for m in raw if m.get("type") == "chat"]

    # Sort by total price (input + output) descending — most expensive first
    def _total_price(m):
        p = m.get("pricing") or {}
        return (p.get("input", 0) or 0) + (p.get("output", 0) or 0)

    seen_ids: set[str] = set()
    entries = []
    for m in sorted(chat_models, key=_total_price, reverse=True):
        model_id = m.get("id", "")
        if not model_id or model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        # Skip vision and coder models — not useful for text-only search agents
        id_lower = model_id.lower()
        if any(tag in id_lower for tag in ("-vl-", "-vl", "vision", "coder", "instruct")):
            continue
        # Derive a short display name from the model ID
        name = model_id.split("/")[-1] if "/" in model_id else model_id
        key = model_id.lower().replace("/", "-").replace(".", "-")
        # Extract pricing
        p = m.get("pricing") or {}
        pricing = None
        if p.get("input") or p.get("output"):
            pricing = ModelPricing(
                input_per_m=float(p.get("input", 0)),
                output_per_m=float(p.get("output", 0)),
            )
        price_hint = ""
        if pricing:
            price_hint = f"  ·  ${pricing.input_per_m:.2f}/${pricing.output_per_m:.2f} per M tok"
        entries.append(ModelCatalogEntry(
            key=key,
            base_model=model_id,
            name=name,
            description=name,
            hint=f"Together AI{price_hint}",
            api_base=TOGETHER_API_BASE,
            api_key_env="TOGETHER_API_KEY",
            pricing=pricing,
            arena_visible=True,
        ))

    _log.info("Fetched %d live Together AI chat models (%d new)", len(chat_models), len(entries))
    return entries
