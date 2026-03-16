"""Unsloth-based model engine for OAPL training on GLM 4.5 Air (MoE).

Drop-in replacement for ``LocalModelEngine`` that uses Unsloth's
``FastLanguageModel`` for memory-efficient LoRA on MoE architectures.

Key differences from ``LocalModelEngine``:

- Uses ``FastLanguageModel.from_pretrained`` (Unsloth's 2x faster loading,
  weight sharing with vLLM standby)
- MoE-specific LoRA targets: ``gate_up_proj``, ``down_proj`` instead of
  ``gate_proj``, ``up_proj``
- No 4-bit quantization for MoE (bitsandbytes doesn't support it)
- FP8 support for H100 native precision
- Unsloth gradient checkpointing (``"unsloth"`` mode, 30% less VRAM)

Usage on Together AI cluster (2× H100 SXM)::

    export UNSLOTH_VLLM_STANDBY=1

    engine = UnslothEngine(
        model_name="unsloth/GLM-4.5-Air",
        max_seq_length=2048,
        lora_r=16,
        load_in_fp8=True,   # H100 native FP8
    )

    # Works with existing OAPLTrainer:
    from konash.training.oapl import OAPLTrainer
    trainer = OAPLTrainer(beta_kl=0.001, beta_value=1.0)
    stats = trainer.train_epoch_torch(dataset, engine, learning_rate=1e-6)
    engine.save_adapter("./checkpoint/adapter")

Requirements: ``pip install unsloth``
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MoE models require different LoRA targets — the fused gate/up projection
# is a single module (gate_up_proj) rather than separate gate_proj + up_proj.
_MOE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_up_proj", "down_proj",
]

_DENSE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Known MoE model families (matched against model_name lowercase)
_MOE_PATTERNS = [
    "glm-4.5-air", "glm-4.7-flash", "mixtral", "dbrx",
    "qwen2-moe", "qwen3-moe", "deepseek-v2", "deepseek-v3",
]


class UnslothEngine:
    """Load a model with Unsloth for OAPL training.

    Implements the same interface as ``LocalModelEngine``
    (``generate``, ``compute_log_probs``, ``tokenize_rollout``,
    ``save_adapter``, ``trainable_params``, ``snapshot_reference``).

    Parameters
    ----------
    model_name : str
        Unsloth model ID (e.g. ``"unsloth/GLM-4.5-Air"``).
    max_seq_length : int
        Maximum sequence length for the model (default 2048).
    lora_r : int
        LoRA rank (default 16).
    lora_alpha : int
        LoRA alpha. Defaults to ``2 * lora_r``.
    lora_target_modules : list[str] | None
        LoRA target modules. Auto-detected for MoE vs dense if None.
    load_in_fp8 : bool
        Use FP8 quantization (recommended for H100).
    gpu_memory_utilization : float
        Fraction of GPU memory for vLLM standby (default 0.90).
    max_new_tokens : int
        Default max new tokens for generation.
    temperature : float
        Default sampling temperature.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        load_in_fp8: bool = False,
        gpu_memory_utilization: float = 0.90,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "Unsloth engine requires: pip install unsloth\n"
                "  See https://github.com/unslothai/unsloth for setup"
            ) from e

        import torch
        self._torch = torch

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._is_moe = self._detect_moe(model_name)
        self._ref_lora_state: Optional[Dict[str, Any]] = None

        # --- Load model via Unsloth ---
        load_kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": False,  # MoE: bitsandbytes doesn't support 4-bit
            "max_lora_rank": lora_r,
        }
        if load_in_fp8:
            load_kwargs["load_in_fp8"] = True
            logger.info("Loading in FP8 (H100 native precision)")

        # Enable vLLM standby if fast_inference is available
        try:
            load_kwargs["fast_inference"] = True
            load_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        except Exception:
            pass

        logger.info("Loading model via Unsloth: %s", model_name)
        print(f"[konash] Loading model via Unsloth: {model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            **load_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # --- Apply LoRA ---
        if lora_alpha is None:
            lora_alpha = lora_r * 2

        if lora_target_modules is None:
            lora_target_modules = (
                _MOE_LORA_TARGETS if self._is_moe else _DENSE_LORA_TARGETS
            )
            logger.info(
                "Auto-detected %s architecture, LoRA targets: %s",
                "MoE" if self._is_moe else "dense",
                lora_target_modules,
            )

        print(f"[konash] Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        print(f"[konash] Targets: {lora_target_modules}")

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        print(f"[konash] Unsloth engine ready ({model_name})")

    @staticmethod
    def _detect_moe(model_name: str) -> bool:
        """Check if the model is a Mixture-of-Experts architecture."""
        name_lower = model_name.lower()
        return any(pat in name_lower for pat in _MOE_PATTERNS)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a chat response.

        Returns ``{"role": "assistant", "content": "..."}``.
        """
        from unsloth import FastLanguageModel
        torch = self._torch

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except (ValueError, AttributeError):
            text = _format_messages(messages)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_new_tokens", self.max_new_tokens)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tok,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temp > 0:
            gen_kwargs.update(do_sample=True, temperature=temp, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False

        # Unsloth: switch to inference mode for fast generation
        FastLanguageModel.for_inference(self.model)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip thinking tags (Qwen3, GLM reasoning)
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        content = re.sub(r"<think>.*", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"</arg_value>\s*", "", content)

        # Switch back to training mode
        FastLanguageModel.for_training(self.model)

        return {"role": "assistant", "content": content}

    # ------------------------------------------------------------------
    # Per-token log-prob computation (for OAPL)
    # ------------------------------------------------------------------

    def snapshot_reference(self) -> None:
        """Snapshot current LoRA weights as reference policy π_vllm.

        The OAPL loss (Eq. 3 from Ritter et al. 2026) uses
        ``ln(π / π_vllm)`` where π_vllm is the policy that *generated
        the rollouts*. Call this BEFORE generating rollouts so the
        reference matches the data-generating policy.

        - For API-based rollouts (Together AI): call once at init.
          The base model (no LoRA) matches the API model that generates
          data. Do NOT re-snapshot after training.
        - For local rollouts: call before each rollout generation phase
          so the reference tracks the current policy.
        """
        torch = self._torch
        self._ref_lora_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        logger.info(
            "Snapshotted %d LoRA parameters as reference policy",
            len(self._ref_lora_state),
        )

    def compute_log_probs(
        self,
        input_ids: "torch.Tensor",
        labels: "torch.Tensor",
        use_reference: bool = False,
        return_entropy: bool = False,
    ) -> tuple:
        """Compute per-token log-probabilities.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(seq_len,)`` or ``(1, seq_len)``.
        labels : Tensor
            Same shape. Use ``-100`` for positions to ignore.
        use_reference : bool
            If True, use reference policy pi_ref for log-probs.
        return_entropy : bool
            If True, also return per-token entropy of the policy
            distribution (for monitoring distribution collapse).

        Returns
        -------
        (token_log_probs, mask) or (token_log_probs, mask, entropy)
            token_log_probs and mask have shape ``(seq_len - 1,)``.
            entropy (if requested) has shape ``(seq_len - 1,)`` with
            per-position entropy H = -sum(p * log(p)) over the vocab.
        """
        torch = self._torch

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        ref_state = self._ref_lora_state
        was_training = self.model.training

        if use_reference and ref_state is not None:
            # Swap in reference LoRA weights
            current_state = {}
            for name, param in self.model.named_parameters():
                if name in ref_state:
                    current_state[name] = param.detach().clone()
                    param.data.copy_(ref_state[name])
        elif use_reference:
            # No snapshot yet — disable LoRA (pi_ref = base model)
            self.model.disable_adapter_layers()

        if use_reference:
            self.model.eval()

        try:
            ctx = torch.no_grad() if use_reference else torch.enable_grad()
            with ctx:
                logits = self.model(input_ids=input_ids).logits

            # Next-token prediction shift
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            tok_lp = log_probs.gather(
                -1, shift_labels.clamp(min=0).unsqueeze(-1),
            ).squeeze(-1)

            mask = shift_labels != -100
            tok_lp = tok_lp * mask.float()

            if return_entropy:
                # H(t) = -Σ_v p(v) log p(v) per position
                probs = log_probs.exp()
                per_token_entropy = -(probs * log_probs).sum(dim=-1)
                return tok_lp.squeeze(0), mask.squeeze(0), per_token_entropy.squeeze(0)

            return tok_lp.squeeze(0), mask.squeeze(0)
        finally:
            if use_reference and ref_state is not None:
                for name, param in self.model.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name])
            elif use_reference:
                self.model.enable_adapter_layers()
            if use_reference and was_training:
                self.model.train()

    # ------------------------------------------------------------------
    # Rollout tokenization (for OAPL training)
    # ------------------------------------------------------------------

    def tokenize_rollout(
        self,
        prompt: str,
        rollout_steps: List[Dict[str, Any]],
    ) -> Dict[str, "torch.Tensor"]:
        """Tokenize a rollout into input_ids and labels.

        Includes the full trajectory: search queries (model-generated,
        trained on), retrieval results (tool output, masked via control
        tokens), reasoning thoughts, and answers.

        Tool output is wrapped in ``<|tool_start|>``/``<|tool_end|>``
        markers so OAPL token masking (Strategy 3) can exclude it from
        the policy gradient.  The model learns *what* to search for and
        *how* to reason about results, but isn't penalized for the
        content of search results it didn't generate.

        Prompt tokens are masked (labels = -100) so loss flows only
        through model-generated tokens.
        """
        torch = self._torch

        parts: List[str] = []
        for step in rollout_steps:
            stype = step.get("type")

            if stype == "retrieval":
                # Search query is model-generated → include, train on it
                query = step.get("query", "")
                if query:
                    parts.append(f'Search: "{query}"')

                # Results are tool output → wrap in markers, will be masked
                results_text = _format_tool_results(step)
                if results_text:
                    parts.append(
                        f"<|tool_start|>\n{results_text}\n<|tool_end|>"
                    )

            elif stype == "reasoning":
                thought = step.get("thought", "")
                if thought:
                    parts.append(thought)

                # Sub-retrieval results within reasoning are tool output
                sub = step.get("sub_retrieval", {})
                if sub:
                    sub_query = sub.get("query", "")
                    if sub_query:
                        parts.append(f'Search: "{sub_query}"')
                    sub_results = _format_tool_results(sub)
                    if sub_results:
                        parts.append(
                            f"<|tool_start|>\n{sub_results}\n<|tool_end|>"
                        )

            elif stype == "answer":
                answer = step.get("answer", "") or step.get("thought", "")
                if answer:
                    parts.append(answer)

            elif stype == "compression" and step.get("summary"):
                parts.append(
                    f"<|compression|>\n{step['summary']}\n<|/compression|>"
                )

        response = "\n".join(parts) if parts else "I don't know."
        user_msg = {"role": "user", "content": prompt}
        full_msgs = [user_msg, {"role": "assistant", "content": response}]

        try:
            full_text = self.tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True,
            )
        except (ValueError, AttributeError):
            full_text = _format_messages(full_msgs)
            prompt_text = _format_messages([user_msg])

        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").squeeze(0)
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").squeeze(0)

        labels = full_ids.clone()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = -100

        # Compute exact tool output token ranges from markers.
        # Primary: use tokenizer offset mapping (fast tokenizers).
        # Fallback: return decoded_tokens for Strategy 3.
        tool_output_ranges = _extract_marker_ranges(
            full_text, full_ids, self.tokenizer,
        )

        result = {
            "input_ids": full_ids.to(self.model.device),
            "labels": labels.to(self.model.device),
        }
        if tool_output_ranges is not None:
            result["tool_output_ranges"] = tool_output_ranges
        else:
            # Fallback: provide decoded tokens for marker-based detection
            result["decoded_tokens"] = self.tokenizer.convert_ids_to_tokens(
                full_ids.tolist()
            )

        return result

    # ------------------------------------------------------------------
    # Adapter I/O
    # ------------------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[konash] Adapter saved to {path}")

    @property
    def trainable_params(self) -> list:
        """Trainable (LoRA) parameters for the optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_marker_ranges(
    text: str,
    token_ids: "torch.Tensor",
    tokenizer: Any,
) -> Optional[List[tuple]]:
    """Find <|tool_start|>...<|tool_end|> spans and return token ranges.

    Uses the tokenizer's offset_mapping (fast tokenizers) for exact
    character-to-token alignment.  Returns None if offset mapping is
    unavailable (falls back to Strategy 3 in oapl.py).
    """
    import re

    # Find marker spans in the raw text
    marker_re = re.compile(
        r"<\|tool_start\|>(.*?)<\|tool_end\|>", re.DOTALL,
    )
    char_spans = [(m.start(), m.end()) for m in marker_re.finditer(text)]
    if not char_spans:
        return []  # no tool output — return empty list (not None)

    # Try offset mapping for precise alignment
    try:
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoding["offset_mapping"]
    except (TypeError, KeyError, Exception):
        return None  # signal fallback to Strategy 3

    if not offsets:
        return None

    ranges: List[tuple] = []
    for char_start, char_end in char_spans:
        tok_start = None
        tok_end = None
        for i, (cs, ce) in enumerate(offsets):
            if ce > char_start and tok_start is None:
                tok_start = i
            if cs >= char_end:
                tok_end = i
                break
        if tok_start is not None:
            ranges.append((tok_start, tok_end if tok_end else len(offsets)))

    return ranges if ranges else []


def _format_tool_results(step: Dict[str, Any]) -> str:
    """Format retrieval results from a rollout step into readable text.

    Handles both structured results (list of dicts with text/score) and
    pre-formatted results_text strings.
    """
    # Pre-formatted text (from some rollout formats)
    results_text = step.get("results_text", "")
    if results_text:
        return results_text

    # Structured results list
    results = step.get("results", [])
    if not results:
        return ""

    lines = []
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            text = r.get("text", str(r))
            score = r.get("score")
            source = r.get("source", "")
            header = f"[{i}]"
            if score is not None:
                header += f" (score: {score:.3f})"
            if source:
                header += f" {source}"
            lines.append(f"{header}\n{text}")
        else:
            lines.append(f"[{i}] {str(r)}")

    return "\n\n".join(lines)


def _format_messages(messages: List[Dict[str, str]]) -> str:
    """Fallback message formatting when no chat template is available."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
