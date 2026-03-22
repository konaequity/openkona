"""Runtime compatibility shims for remote training environments.

Loaded automatically by Python when the project root is on ``PYTHONPATH``.
"""

from __future__ import annotations


def _patch_tokenizers_backend() -> None:
    """Backfill attributes vLLM expects from newer tokenizer wrappers."""
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except Exception:
        return

    if hasattr(TokenizersBackend, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):  # type: ignore[override]
        return list(getattr(self, "all_special_tokens", []))

    TokenizersBackend.all_special_tokens_extended = all_special_tokens_extended


_patch_tokenizers_backend()


def _patch_vllm_disabled_tqdm() -> None:
    """Avoid duplicate disable kwarg crashes in some vLLM+tqdm combos."""
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils
    except Exception:
        return

    base_tqdm = getattr(weight_utils, "tqdm", None)
    if base_tqdm is None:
        return

    class PatchedDisabledTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("disable", None)
            super().__init__(*args, **kwargs, disable=True)

    weight_utils.DisabledTqdm = PatchedDisabledTqdm


_patch_vllm_disabled_tqdm()
