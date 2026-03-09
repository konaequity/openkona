"""Local model engine: load a HuggingFace causal LM with LoRA for inference and OAPL training.

Supports single-GPU (default) and multi-GPU via HuggingFace Accelerate.
For distributed training, pass ``distributed=True`` to ``LocalModelEngine``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalModelEngine:
    """Loads a HuggingFace model with LoRA for both text generation and OAPL training.

    This replaces the OpenAI API client for local/single-GPU use. The same
    engine handles:

    - Chat-template-aware text generation (for QA synthesis and rollouts)
    - Per-token log-prob computation (for OAPL loss)
    - Reference policy log-probs (LoRA disabled)
    - LoRA adapter save/load (for checkpointing)

    Requirements: ``pip install openkona[train]`` (torch, transformers, peft)

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g. ``"THUDM/glm-4-9b-chat"``).
    device : str
        ``"auto"`` (default), ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.
    dtype : str
        ``"auto"`` (bf16 if supported, else fp16), ``"fp16"``, ``"bf16"``, ``"fp32"``.
    lora_r : int
        LoRA rank (default 16).
    lora_alpha : int
        LoRA alpha (default 32).
    load_in_4bit : bool
        Use 4-bit QLoRA quantization (requires ``bitsandbytes``).
    gradient_checkpointing : bool
        Enable gradient checkpointing to reduce VRAM at the cost of speed.
    adapter_path : str | None
        Load an existing LoRA adapter instead of creating a new one.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        gradient_checkpointing: bool = False,
        adapter_path: Optional[str] = None,
        distributed: bool = False,
        fsdp: bool = False,
        deepspeed_config: Optional[str] = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local model requires: pip install openkona[train]\n"
                "  (torch, transformers, peft, accelerate)\n"
                "For 4-bit quantization also: pip install bitsandbytes"
            ) from e

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._torch = torch

        # -- Device --
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # -- Dtype --
        if dtype == "auto":
            if self.device == "cuda":
                self.dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            elif self.device == "mps":
                self.dtype = torch.float16  # MPS doesn't support bf16
            else:
                self.dtype = torch.float32
        elif dtype in ("fp16", "float16"):
            self.dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # -- Tokenizer --
        print(f"[konash] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # -- Model --
        print(f"[konash] Loading model: {model_name} → {self.device} ({self.dtype})")
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
        }

        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            model_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        # MPS and CPU: load to CPU first, then .to(device)

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if "device_map" not in model_kwargs:
            base_model = base_model.to(self.device)

        # Enable input gradients for gradient checkpointing compatibility
        if gradient_checkpointing:
            base_model.enable_input_require_grads()

        # -- LoRA --
        from peft import LoraConfig, get_peft_model, PeftModel

        if adapter_path and os.path.exists(adapter_path):
            print(f"[konash] Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model, adapter_path, is_trainable=True,
            )
        else:
            if lora_target_modules:
                targets = lora_target_modules
            else:
                targets = self._detect_lora_targets(base_model)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, lora_config)

        self.model.print_trainable_parameters()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        # -- Distributed setup --
        self.distributed = distributed
        self._accelerator = None

        if distributed or fsdp or deepspeed_config:
            self._setup_distributed(
                fsdp=fsdp,
                deepspeed_config=deepspeed_config,
            )

        print(f"[konash] Model ready on {self.device}")

    @staticmethod
    def _detect_lora_targets(model) -> list:
        """Auto-detect LoRA target modules based on the model architecture.

        Checks for common linear layer naming conventions across architectures:
        - Llama/Qwen/Mistral: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        - GPT-2/GPT-J: c_attn, c_proj, c_fc
        - Falcon: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
        - BLOOM: query_key_value, dense, dense_h_to_4h, dense_4h_to_h

        Falls back to all Linear layers if no known pattern matches.
        """
        import torch.nn as nn

        named_modules = dict(model.named_modules())
        module_names = set()
        for name, mod in named_modules.items():
            if isinstance(mod, nn.Linear):
                # Get the short name (last component)
                short = name.split(".")[-1]
                module_names.add(short)

        # Try known architecture patterns in priority order
        llama_targets = {"q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"}
        gpt2_targets = {"c_attn", "c_proj", "c_fc"}
        falcon_targets = {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}

        for pattern in [llama_targets, gpt2_targets, falcon_targets]:
            found = pattern & module_names
            if len(found) >= 2:  # at least 2 matches = likely correct arch
                return sorted(found)

        # Fallback: target all Linear layers (PEFT supports this)
        if module_names:
            logger.info(
                "Could not detect model architecture; targeting all Linear layers: %s",
                sorted(module_names),
            )
            return sorted(module_names)

        # Last resort: use Llama-style defaults
        return ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

    def _setup_distributed(
        self,
        fsdp: bool = False,
        deepspeed_config: Optional[str] = None,
    ) -> None:
        """Configure multi-GPU training via HuggingFace Accelerate.

        Supports three modes:
        - **DataParallel** (default distributed): Simple multi-GPU with gradient sync.
        - **FSDP**: Fully Sharded Data Parallel for large models.
        - **DeepSpeed**: ZeRO Stage 2/3 for memory-efficient training.

        Usage::

            # Launch with: accelerate launch --multi_gpu script.py
            engine = LocalModelEngine("Qwen/Qwen2.5-7B", distributed=True)

            # Or with DeepSpeed:
            engine = LocalModelEngine(
                "Qwen/Qwen2.5-7B",
                deepspeed_config="ds_config.json",
            )
        """
        try:
            from accelerate import Accelerator
        except ImportError:
            logger.warning(
                "Distributed training requires: pip install accelerate\n"
                "Falling back to single-GPU mode."
            )
            self.distributed = False
            return

        kwargs: Dict[str, Any] = {}

        if deepspeed_config and os.path.exists(deepspeed_config):
            # DeepSpeed ZeRO integration
            from accelerate.utils import DeepSpeedPlugin
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=deepspeed_config,
            )
            kwargs["deepspeed_plugin"] = ds_plugin
            logger.info("Using DeepSpeed config: %s", deepspeed_config)
        elif fsdp:
            from accelerate.utils import FullyShardedDataParallelPlugin
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullStateDictConfig,
                StateDictType,
            )
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True,
                ),
                state_dict_type=StateDictType.FULL_STATE_DICT,
            )
            kwargs["fsdp_plugin"] = fsdp_plugin
            logger.info("Using FSDP for distributed training")

        self._accelerator = Accelerator(**kwargs)
        self.model = self._accelerator.prepare(self.model)
        self.device = str(self._accelerator.device)
        self.distributed = True
        logger.info(
            "Distributed training enabled: %d GPUs, device=%s",
            self._accelerator.num_processes,
            self.device,
        )

    @property
    def accelerator(self):
        """The Accelerate ``Accelerator`` instance, or None for single-GPU."""
        return self._accelerator

    # ------------------------------------------------------------------
    # Text generation (same interface as _OpenAILLMClient)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a chat response.

        Returns ``{"role": "assistant", "content": "..."}``, matching the
        interface expected by ``BaseAgent``.
        """
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

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip reasoning-model thinking tags (e.g. Qwen3 <think>...</think>)
        # Also handle unclosed <think> when model runs out of tokens mid-thought
        import re as _re
        content = _re.sub(r"<think>.*?</think>\s*", "", content, flags=_re.DOTALL)
        content = _re.sub(r"<think>.*", "", content, flags=_re.DOTALL).strip()

        return {"role": "assistant", "content": content}

    # ------------------------------------------------------------------
    # Per-token log-prob computation (for OAPL)
    # ------------------------------------------------------------------

    def snapshot_reference(self) -> None:
        """Snapshot current LoRA weights as the reference policy πref.

        After each training iteration, call this so that the next
        iteration's KL term ``ln(π/πref)`` is relative to the previous
        iteration's policy rather than the base model (KARL Section 4.2:
        iterative training replaces πref with the latest policy).

        If never called, ``compute_log_probs(use_reference=True)`` falls
        back to disabling LoRA (i.e., πref = base model), which is
        correct for iteration 1.
        """
        torch = self._torch
        self._ref_lora_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad  # only LoRA params
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
    ) -> tuple:
        """Compute per-token log-probabilities.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(seq_len,)`` or ``(1, seq_len)``.
        labels : Tensor
            Same shape.  Use ``-100`` for positions to ignore.
        use_reference : bool
            If *True*, use the reference policy πref for log-probs.
            If ``snapshot_reference()`` was called, swaps in the
            snapshotted LoRA weights. Otherwise disables LoRA entirely
            (πref = base model, correct for iteration 1).

        Returns
        -------
        (token_log_probs, mask)
            Both shape ``(seq_len - 1,)``.  *mask* is ``True`` for non-ignored
            positions.
        """
        torch = self._torch

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        ref_state = getattr(self, "_ref_lora_state", None)
        was_training = self.model.training

        if use_reference and ref_state is not None:
            # Swap in reference LoRA weights (πref = previous iteration)
            current_state = {}
            for name, param in self.model.named_parameters():
                if name in ref_state:
                    current_state[name] = param.detach().clone()
                    param.data.copy_(ref_state[name])
        elif use_reference:
            # No snapshot yet — disable LoRA (πref = base model)
            self.model.disable_adapter_layers()

        # Eval mode for reference: disables dropout so log-probs are
        # deterministic (LoRA dropout would corrupt the KL term).
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

            return tok_lp.squeeze(0), mask.squeeze(0)
        finally:
            if use_reference and ref_state is not None:
                # Restore current training weights
                for name, param in self.model.named_parameters():
                    if name in current_state:
                        param.data.copy_(current_state[name])
            elif use_reference:
                self.model.enable_adapter_layers()
            # Restore train/eval mode
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
        """Tokenize a rollout into ``input_ids`` and ``labels``.

        The prompt tokens are masked (``labels = -100``) so the OAPL loss is
        computed only on the model's generated reasoning and answer tokens.

        Compression steps are included in the token sequence so that the
        OAPL loss backpropagates through compression tokens.  This matches
        the KARL paper's end-to-end compression training (Section 3): the
        model learns *what* to compress through the task reward signal.
        """
        torch = self._torch

        # Reconstruct assistant response from rollout steps
        parts: List[str] = []
        for step in rollout_steps:
            stype = step.get("type")
            if stype == "reasoning" and step.get("thought"):
                parts.append(step["thought"])
            elif stype == "answer" and step.get("answer"):
                parts.append(step["answer"])
            elif stype == "compression" and step.get("summary"):
                # Include compression summary with markers so that:
                # 1. OAPL loss flows to compression tokens (RL training)
                # 2. RolloutSegmenter can split at compression boundaries
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

        # Mask prompt tokens
        labels = full_ids.clone()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = -100

        return {
            "input_ids": full_ids.to(self.model.device),
            "labels": labels.to(self.model.device),
        }

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

def _format_messages(messages: List[Dict[str, str]]) -> str:
    """Fallback message formatting when no chat template is available."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
