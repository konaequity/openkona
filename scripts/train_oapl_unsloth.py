#!/usr/bin/env python3
"""OAPL training with Unsloth on GLM 4.5 Air.

Designed for Shadeform GPU clusters (2× H100 SXM, 160 GB total VRAM).

Usage:
    # From pre-generated rollouts (Stage 3 output):
    python scripts/train_oapl_unsloth.py \
        --rollouts glm_test_results/stage3_results.json \
        --output ./checkpoints/iter1

    # Full pipeline (synthesis → rollouts → OAPL, iterative):
    python scripts/train_oapl_unsloth.py \
        --corpus ./my_docs \
        --iterations 2 \
        --output ./checkpoints

    # Resume from checkpoint:
    python scripts/train_oapl_unsloth.py \
        --rollouts glm_test_results/stage3_results.json \
        --adapter ./checkpoints/iter1/adapter \
        --output ./checkpoints/iter2

Environment:
    SHADEFORM_API_KEY      — For Shadeform provisioning (if not running on-box)
    UNSLOTH_VLLM_STANDBY=1 — Enable vLLM weight sharing (saves ~9 GB)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Enable vLLM standby before any imports
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


def parse_args():
    p = argparse.ArgumentParser(description="OAPL training with Unsloth")

    # Data source (pick one)
    p.add_argument("--rollouts", type=str, default=None,
                    help="Path to stage3_results.json (pre-generated rollouts)")
    p.add_argument("--corpus", type=str, default=None,
                    help="Path to corpus directory (for full pipeline)")

    # Model
    p.add_argument("--model", type=str, default="zai-org/GLM-4.5-Air-FP8",
                    help="Training model ID")
    p.add_argument("--vllm-model", type=str, default=None,
                    help="Raw model ID for vLLM serving (defaults to --model)")
    p.add_argument("--adapter", type=str, default=None,
                    help="Path to existing LoRA adapter (for resuming)")
    p.add_argument("--fp8", action="store_true", default=True,
                    help="Load model in FP8 (default: True for H100)")
    p.add_argument("--no-fp8", action="store_true",
                    help="Disable FP8, use BF16 instead")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=None,
                    help="LoRA alpha (default: 2 * lora_r)")

    # OAPL hyperparams
    p.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    p.add_argument("--beta-kl", type=float, default=0.001,
                    help="KL regularization coefficient")
    p.add_argument("--beta-value", type=float, default=1.0,
                    help="Temperature for V* estimation")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Max gradient norm for clipping")
    p.add_argument("--epochs", type=int, default=1,
                    help="Training epochs per iteration")

    # Pipeline (full mode only)
    p.add_argument("--iterations", type=int, default=2,
                    help="Number of synthesis→train iterations")
    p.add_argument("--synthesis-calls", type=int, default=1500,
                    help="Independent synthesis calls per iteration")
    p.add_argument("--rollouts-per-example", type=int, default=8,
                    help="Rollouts per QA pair")
    p.add_argument("--rollout-max-steps", type=int, default=50,
                    help="Max steps per rollout")
    p.add_argument("--max-examples", type=int, default=None,
                    help="Cap on training examples per iteration")
    p.add_argument("--max-seq-length", type=int, default=2048,
                    help="Max sequence length")

    # Parallelism
    p.add_argument("--rollout-workers", type=int, default=32,
                    help="Max concurrent QA pairs for rollout generation (default: 32). "
                         "High values work well with vLLM; lower for local models.")

    # Sleep/wake mode
    p.add_argument("--vllm-sleep-wake", action="store_true",
                    help="Use vLLM sleep/wake mode for single-GPU iterative training. "
                         "vLLM runs for inference, sleeps during OAPL training.")
    p.add_argument("--tensor-parallel", type=int, default=1,
                    help="Tensor parallel size for vLLM (default: 1)")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                    help="Override max model length for vLLM")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                    help="Override vLLM GPU memory utilization")

    # Output
    p.add_argument("--output", type=str, default="./checkpoints",
                    help="Output directory for checkpoints")

    # Export
    p.add_argument("--push-to-hub", type=str, default=None,
                    help="HuggingFace Hub repo to push adapter (e.g. your-name/konash-glm45-air-lora)")
    p.add_argument("--merge-and-export", type=str, default=None,
                    help="Path to save merged full model (base + LoRA)")
    p.add_argument("--export-gguf", type=str, default=None,
                    help="Export merged model as GGUF (e.g. q4_k_m, q8_0, f16)")
    p.add_argument("--deploy-together", type=str, default=None,
                    help="Deploy to Together AI endpoint (provide model name, e.g. konash-glm45-air-v1)")

    return p.parse_args()


def _target_synthesis_examples(
    *,
    current_count: int,
    max_examples: int | None,
    batch_size: int = 8,
) -> int:
    """Choose how many examples a synthesis call should request."""
    if max_examples is None:
        return batch_size
    remaining_examples = max_examples - current_count
    if remaining_examples <= 0:
        return 0
    return min(batch_size, remaining_examples)


def _should_sleep_vllm_for_training(
    *,
    iteration: int,
    total_iterations: int,
) -> bool:
    """Sleep only when a later iteration still needs the live vLLM server."""
    return iteration < total_iterations - 1


def _trim_messages_for_context(
    messages: list[dict],
    *,
    max_context_tokens: int,
    keep_last_messages: int = 6,
) -> list[dict]:
    """Trim oversized chat history to fit within a smaller context window."""
    if not messages:
        return []

    char_budget = max(int(max_context_tokens * 2), 4096)
    kept_messages: list[dict] = []
    tail_start = max(len(messages) - keep_last_messages, 0)

    for idx, message in enumerate(messages):
        cloned = dict(message)
        content = cloned.get("content")
        if not isinstance(content, str):
            kept_messages.append(cloned)
            continue

        if cloned.get("role") == "tool":
            max_chars = 800
        elif idx >= tail_start:
            max_chars = 1600
        else:
            max_chars = 400

        if len(content) > max_chars:
            cloned["content"] = content[:max_chars] + "\n...[truncated]"
        kept_messages.append(cloned)

    while len(kept_messages) > keep_last_messages + 1:
        total_chars = sum(len((m.get("content") or "")) for m in kept_messages)
        if total_chars <= char_budget:
            break
        del kept_messages[1]

    total_chars = sum(len((m.get("content") or "")) for m in kept_messages)
    if total_chars <= char_budget:
        return kept_messages

    trimmed_messages: list[dict] = []
    remaining_budget = char_budget
    for message in reversed(kept_messages):
        cloned = dict(message)
        content = cloned.get("content")
        if not isinstance(content, str):
            trimmed_messages.append(cloned)
            continue

        reserve = 256 if remaining_budget > 256 else 0
        max_chars = max(remaining_budget - reserve, 128)
        if len(content) > max_chars:
            cloned["content"] = content[:max_chars] + "\n...[truncated]"
        remaining_budget -= len(cloned.get("content") or "")
        trimmed_messages.append(cloned)

    return list(reversed(trimmed_messages))


def export_model(engine, args):
    """Export trained model: push to Hub, merge LoRA, convert to GGUF, deploy to Together AI."""
    hf_token = os.environ.get("HF_TOKEN")

    if args.push_to_hub:
        print(f"\n  Pushing adapter to HuggingFace Hub: {args.push_to_hub}")
        engine.model.push_to_hub(args.push_to_hub, token=hf_token)
        engine.tokenizer.push_to_hub(args.push_to_hub, token=hf_token)
        print(f"  Pushed: https://huggingface.co/{args.push_to_hub}")

    if args.merge_and_export:
        print(f"\n  Merging LoRA into base model → {args.merge_and_export}")
        engine.model.save_pretrained_merged(
            args.merge_and_export,
            engine.tokenizer,
            save_method="merged_16bit",
        )
        print(f"  Merged model saved to {args.merge_and_export}")

        # Optionally push merged model to Hub too
        if args.push_to_hub:
            merged_hub = args.push_to_hub + "-merged"
            print(f"  Pushing merged model to Hub: {merged_hub}")
            engine.model.push_to_hub_merged(
                merged_hub,
                engine.tokenizer,
                save_method="merged_16bit",
                token=hf_token,
            )

    if args.export_gguf:
        quant = args.export_gguf
        gguf_dir = os.path.join(args.output, f"gguf-{quant}")
        print(f"\n  Exporting GGUF ({quant}) → {gguf_dir}")
        engine.model.save_pretrained_gguf(
            gguf_dir,
            engine.tokenizer,
            quantization_method=quant,
        )
        print(f"  GGUF saved to {gguf_dir}")

        if args.push_to_hub:
            gguf_hub = args.push_to_hub + "-GGUF"
            print(f"  Pushing GGUF to Hub: {gguf_hub}")
            engine.model.push_to_hub_gguf(
                gguf_hub,
                engine.tokenizer,
                quantization_method=quant,
                token=hf_token,
            )

    if args.deploy_together:
        _deploy_to_together(engine, args)


def _deploy_to_together(engine, args):
    """Merge LoRA, push to HuggingFace, upload to Together AI, create endpoint."""
    import subprocess

    model_name = args.deploy_together
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = args.push_to_hub

    # Step 1: Merge LoRA into base model and push to HF (required for Together upload)
    if not hf_repo:
        hf_repo = f"konash/{model_name}"
        print(f"\n  No --push-to-hub specified, using: {hf_repo}")

    merged_path = args.merge_and_export or os.path.join(args.output, "merged")

    if not os.path.exists(merged_path):
        print(f"\n  Merging LoRA into base model → {merged_path}")
        engine.model.save_pretrained_merged(
            merged_path,
            engine.tokenizer,
            save_method="merged_16bit",
        )

    print(f"  Pushing merged model to HuggingFace: {hf_repo}")
    engine.model.push_to_hub_merged(
        hf_repo,
        engine.tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )

    # Step 2: Upload to Together AI from HuggingFace
    print(f"\n  Uploading to Together AI: {model_name}")
    upload_cmd = [
        "together", "models", "upload",
        "--model-name", model_name,
        "--model-source", hf_repo,
    ]
    if hf_token:
        upload_cmd.extend(["--hf-token", hf_token])

    result = subprocess.run(upload_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Upload failed: {result.stderr}")
        print("  You can retry manually:")
        print(f"    together models upload --model-name {model_name} --model-source {hf_repo}")
        return

    print(f"  Upload started: {result.stdout.strip()}")

    # Step 3: Create dedicated endpoint
    print(f"\n  Creating dedicated endpoint...")
    endpoint_cmd = [
        "together", "endpoints", "create",
        "--display-name", f"{model_name}-endpoint",
        "--model", model_name,
        "--gpu", "h100",
        "--gpu-count", "2",
        "--no-speculative-decoding",
    ]
    result = subprocess.run(endpoint_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Endpoint creation failed: {result.stderr}")
        print("  Model uploaded — create endpoint manually:")
        print(f"    together endpoints create --display-name {model_name} "
              f"--model {model_name} --gpu h100 --gpu-count 2")
        return

    print(f"  Endpoint created: {result.stdout.strip()}")
    print(f"\n  Your trained model is now available via Together AI API!")
    print(f"  Use model name: {model_name}")


def _train_value_model(output_dir: str, num_iterations: int) -> dict | None:
    """Train a NeuralValueModel (Qwen3-4B) on accumulated rollout data.

    Matches KARL paper Section 5.2: uses a small transformer LM as the value
    function, trained with binary cross-entropy at the token level.
    """
    print("##KONASH:value_model_start##")
    print("\n  Training value model (NeuralValueModel / Qwen3-4B)...")

    # Collect all rollouts across iterations
    all_rollouts = []
    all_rewards = []
    for i in range(1, num_iterations + 1):
        rollouts_path = os.path.join(output_dir, f"iter{i}", "rollouts.json")
        if not os.path.exists(rollouts_path):
            continue
        with open(rollouts_path) as f:
            rollout_dicts = json.load(f)
        for rd in rollout_dicts:
            all_rollouts.append(rd.get("rollout", rd.get("steps", [])))
            all_rewards.append(rd.get("reward", 0.0))

    if not all_rollouts:
        print("  No rollout data found — skipping value model.")
        return None

    print(f"  Collected {len(all_rollouts)} rollouts across {num_iterations} iterations")

    try:
        from konash.inference.value_model import NeuralValueModel
        vm = NeuralValueModel(model_name="Qwen/Qwen3-4B")
        vm_stats = vm.fit(
            all_rollouts,
            all_rewards,
            lr=1e-4,
            epochs=3,
            batch_size=4,
        )
        # Save value model checkpoint
        vm_path = os.path.join(output_dir, "value_model")
        vm.save(vm_path)
        print(f"  Value model trained — loss {vm_stats['final_loss']:.4f}")
        print(f"  Saved to {vm_path}")
        print(f"##KONASH:value_model_done:loss={vm_stats['final_loss']:.4f}##")
        return vm_stats
    except (ImportError, RuntimeError) as e:
        print(f"  NeuralValueModel unavailable ({e}), using lightweight ValueModel")
        from konash.inference.value_model import ValueModel
        vm = ValueModel(feature_dim=64)
        vm_stats = vm.fit(all_rollouts, all_rewards, lr=0.01, epochs=20)
        vm_path = os.path.join(output_dir, "value_model.json")
        weights = vm.weights
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        with open(vm_path, "w") as f:
            json.dump({
                "weights": weights,
                "bias": vm.bias,
                "feature_dim": vm.feature_dim,
            }, f)
        print(f"  Lightweight value model trained — loss {vm_stats['final_loss']:.4f}")
        print(f"  Saved to {vm_path}")
        print(f"##KONASH:value_model_done:loss={vm_stats['final_loss']:.4f}##")
        return vm_stats


def load_rollouts_from_stage3(path: str):
    """Load pre-generated rollouts from stage3_results.json."""
    from konash.training.dataset import OfflineRolloutDataset

    with open(path) as f:
        data = json.load(f)

    # Stage 3 results: groups can be at data["rollouts"]["groups"],
    # data["groups"], or data["rollout_groups"]
    groups = data.get("groups", data.get("rollout_groups", []))
    if not groups and isinstance(data.get("rollouts"), dict):
        groups = data["rollouts"].get("groups", [])
    if not groups:
        print(f"  ERROR: No rollout groups found in {path}")
        sys.exit(1)

    # Filter to surviving groups (pass rate in [0.1, 0.9])
    surviving = []
    for g in groups:
        rollouts = g.get("rollouts", [])
        if not rollouts:
            continue
        pass_rate = sum(1 for r in rollouts if r.get("passed")) / len(rollouts)
        if 0.1 <= pass_rate <= 0.9:
            surviving.append(g)

    if not surviving:
        print("  WARNING: No groups survived pass-rate filter [0.1, 0.9].")
        print("  Using all groups with mixed pass/fail instead.")
        surviving = [
            g for g in groups
            if g.get("rollouts")
            and 0 < sum(1 for r in g["rollouts"] if r.get("passed")) < len(g["rollouts"])
        ]

    print(f"  Loaded {len(groups)} groups, {len(surviving)} surviving")

    rollout_dicts = []
    for g in surviving:
        prompt = g.get("question", g.get("prompt", ""))
        for r in g.get("rollouts", []):
            # Full step dicts if available, otherwise construct minimal
            # steps from final_answer for tokenization
            steps = r.get("steps", [])
            if not steps and r.get("final_answer"):
                steps = [{"type": "answer", "answer": r["final_answer"]}]
            rollout_dicts.append({
                "prompt": prompt,
                "rollout": steps,
                "reward": 1.0 if r.get("passed") else 0.0,
            })

    dataset = OfflineRolloutDataset.from_rollouts(rollout_dicts)
    print(f"  Dataset: {len(dataset.prompts)} groups, {len(dataset)} rollouts")
    return dataset, rollout_dicts


def train_from_rollouts(args):
    """Train OAPL from pre-generated rollouts."""
    from konash.training.unsloth_engine import UnslothEngine
    from konash.training.oapl import OAPLTrainer
    from konash.training.logger import TrainingLogger

    print("=" * 60)
    print("  KONASH OAPL Training (Unsloth)")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  FP8:       {args.fp8 and not args.no_fp8}")
    print(f"  LoRA r:    {args.lora_r}")
    print(f"  LR:        {args.lr}")
    print(f"  Beta KL:   {args.beta_kl}")
    print(f"  Beta V:    {args.beta_value}")
    print(f"  Rollouts:  {args.rollouts}")
    print(f"  Output:    {args.output}")
    print()

    project_name = os.path.basename(os.path.abspath(args.output)) or "default"
    log = TrainingLogger(project_name)
    log.start(
        iterations=args.epochs,
        corpus=args.rollouts or "(rollouts)",
        model=args.model,
    )

    # Load data
    print("##KONASH:loading_data##")
    print("Loading rollout data...")
    dataset, rollout_dicts = load_rollouts_from_stage3(args.rollouts)

    # Load model
    print("##KONASH:loading_model##")
    print("Loading model via Unsloth...")
    t0 = time.time()
    engine = UnslothEngine(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_fp8=args.fp8 and not args.no_fp8,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # If resuming from adapter, load it
    if args.adapter and os.path.exists(args.adapter):
        print(f"\n  Loading adapter from {args.adapter}")
        from peft import PeftModel
        engine.model = PeftModel.from_pretrained(
            engine.model, args.adapter, is_trainable=True,
        )

    print(f"##KONASH:model_loaded:{time.time() - t0:.1f}s##")

    # Snapshot reference policy (pi_ref = current model before training)
    engine.snapshot_reference()

    # Create trainer
    trainer = OAPLTrainer(
        beta_kl=args.beta_kl,
        beta_value=args.beta_value,
    )

    # Train
    print("##KONASH:oapl_start##")
    all_stats = []
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        t0 = time.time()
        stats = trainer.train_epoch_torch(
            dataset=dataset,
            model_engine=engine,
            learning_rate=args.lr,
            max_grad_norm=args.max_grad_norm,
        )
        elapsed = time.time() - t0

        stats["epoch"] = epoch + 1
        stats["wall_time_s"] = elapsed
        all_stats.append(stats)

        print(f"  Loss:       {stats['mean_loss']:.4f}")
        print(f"  Groups:     {stats['num_groups']}")
        print(f"  Rollouts:   {stats['num_rollouts']}")
        print(f"  Segments:   {stats['num_segments']}")
        print(f"  Masked:     {stats['masked_token_pct']:.1f}% tokens")
        print(f"  Wall time:  {elapsed:.1f}s")
        log.oapl(
            iteration=1,
            epoch=epoch + 1,
            loss=stats["mean_loss"],
            kl=stats.get("mean_kl", 0),
            entropy=stats.get("mean_entropy", 0),
            num_groups=stats["num_groups"],
            num_rollouts=stats["num_rollouts"],
            learning_rate=args.lr,
            duration_seconds=elapsed,
        )

    # Save checkpoint
    os.makedirs(args.output, exist_ok=True)
    adapter_path = os.path.join(args.output, "adapter")
    engine.save_adapter(adapter_path)

    print(f"##KONASH:oapl_done:loss={all_stats[-1]['mean_loss']:.4f}##")

    # Save rollouts for value model training
    iter_dir = os.path.join(args.output, "iter1")
    os.makedirs(iter_dir, exist_ok=True)
    rollouts_save_path = os.path.join(iter_dir, "rollouts.json")
    with open(rollouts_save_path, "w") as f:
        json.dump(rollout_dicts, f, default=str)

    # Train value model (KARL Section 5.2)
    vm_stats = _train_value_model(args.output, 1)
    if vm_stats:
        log.value_model(
            loss=vm_stats.get("final_loss", 0),
            epochs=vm_stats.get("epochs", 0),
            duration_seconds=0,
        )

    meta = {
        "model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha or args.lora_r * 2,
        "beta_kl": args.beta_kl,
        "beta_value": args.beta_value,
        "learning_rate": args.lr,
        "fp8": args.fp8 and not args.no_fp8,
        "rollouts_source": args.rollouts,
        "stats": all_stats,
        "value_model": True,
    }
    meta_path = os.path.join(args.output, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.complete(
        iterations=args.epochs,
        total_seconds=sum(s.get("wall_time_s", 0) for s in all_stats),
        stats=all_stats,
    )

    print(f"##KONASH:complete##")
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Adapter: {adapter_path}")
    print(f"  Meta:    {meta_path}")
    print(f"{'='*60}")

    # Export (push to Hub, merge, GGUF)
    if args.push_to_hub or args.merge_and_export or args.export_gguf or args.deploy_together:
        export_model(engine, args)

    return all_stats


def _build_vllm_generate_fn(
    api_url: str,
    model_name: str,
    *,
    max_context_tokens: int | None = None,
):
    """Build an llm_fn that calls a vLLM OpenAI-compatible server."""
    import urllib.request
    import urllib.error

    def generate_fn(messages, **kwargs):
        max_tokens = kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 1024))
        temperature = kwargs.pop("temperature", 0.7)
        base_request_body = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        # Pass through tools/tool_choice for agentic synthesis
        if "tools" in kwargs:
            base_request_body["tools"] = kwargs.pop("tools")
        if "tool_choice" in kwargs:
            base_request_body["tool_choice"] = kwargs.pop("tool_choice")

        current_max_tokens = max_tokens
        current_messages = list(messages)
        for attempt_idx in range(4):
            request_body = dict(base_request_body)
            request_body["messages"] = current_messages
            request_body["max_tokens"] = current_max_tokens
            body = json.dumps(request_body).encode()
            req = urllib.request.Request(
                f"{api_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                # Don't strip <think> tags here — callers like qa.py's
                # _clean_thinking_tags handle it correctly. Stripping here
                # with a greedy regex destroys SEARCH:/PROPOSE: actions.
                result = {"role": "assistant", "content": content}
                # Return tool_calls if present
                if msg.get("tool_calls"):
                    result["tool_calls"] = msg["tool_calls"]
                return result
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400 and "maximum context length" in err_body.lower():
                    if max_context_tokens is not None:
                        trimmed_messages = _trim_messages_for_context(
                            current_messages,
                            max_context_tokens=max_context_tokens,
                        )
                        if trimmed_messages != current_messages:
                            current_messages = trimmed_messages
                            current_max_tokens = min(
                                current_max_tokens,
                                max(max_context_tokens // 8, 128),
                            )
                            continue
                    if attempt_idx < 3 and current_max_tokens > 128:
                        current_max_tokens = max(current_max_tokens // 2, 128)
                        continue
                raise

    return generate_fn


def train_full_pipeline(args):
    """Full iterative pipeline: synthesis → rollouts → OAPL → repeat."""
    if args.vllm_sleep_wake:
        return _train_sleep_wake_pipeline(args)

    from konash.training.unsloth_engine import UnslothEngine
    from konash.training.oapl import OAPLTrainer
    from konash.training.dataset import OfflineRolloutDataset
    from konash.corpus import Corpus
    from konash.synthesis.pipeline import SynthesisPipeline
    from konash.synthesis.qa import QuestionAnswerSynthesizer
    from konash.synthesis.rollouts import RolloutGenerator

    print("=" * 60)
    print("  KONASH Full Pipeline (Unsloth + OAPL)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Corpus:     {args.corpus}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Synthesis:  {args.synthesis_calls} calls/iter")
    print(f"  Output:     {args.output}")
    print()

    print("##KONASH:loading_data##", flush=True)

    # Load model
    print("##KONASH:loading_model##", flush=True)
    print("Loading model via Unsloth...")
    _model_t0 = time.time()
    engine = UnslothEngine(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_fp8=args.fp8 and not args.no_fp8,
    )
    print(f"##KONASH:model_loaded:{time.time() - _model_t0:.1f}s##", flush=True)

    # Ingest corpus
    print("Ingesting corpus...")
    corpus = Corpus(args.corpus, chunk_size=512)
    corpus.ingest()
    print(f"  Indexed {corpus.num_documents} chunks")

    # Build generate_fn from engine
    def generate_fn(messages, **kwargs):
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens", 512)
        return engine.generate(messages, **kwargs)

    # Trainer
    trainer = OAPLTrainer(beta_kl=args.beta_kl, beta_value=args.beta_value)

    # Snapshot initial reference
    engine.snapshot_reference()

    all_iteration_stats = []

    for iteration in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")

        # Stage 1: Synthesis
        print("  Synthesizing QA pairs...")
        synthesizer = QuestionAnswerSynthesizer(
            vector_search_tool=corpus,
            llm_fn=generate_fn,
        )
        rollout_gen = RolloutGenerator(
            search_tool=corpus,
            llm_fn=generate_fn,
            max_steps=args.rollout_max_steps,
        )
        pipeline = SynthesisPipeline(
            synthesizer=synthesizer,
            rollout_generator=rollout_gen,
        )

        raw_examples = []
        for call_idx in range(args.synthesis_calls):
            target_examples = _target_synthesis_examples(
                current_count=len(raw_examples),
                max_examples=args.max_examples,
            )
            if target_examples <= 0:
                break
            batch = synthesizer.synthesize(
                documents=None,
                num_examples=target_examples,
            )
            raw_examples.extend(batch)
            # Stream each QA pair to local monitor via ##KONASH## markers
            for ex in batch:
                q = ex.question.replace('\n', ' ').replace('#', '')
                a = ex.answer.replace('\n', ' ').replace('#', '')
                print(f"##KONASH:qa:{iteration + 1}:{q}\t{a}##", flush=True)
            print(f"  Call {call_idx + 1}/{args.synthesis_calls}: +{len(batch)} pairs ({len(raw_examples)} total)")
        examples = pipeline.deduplicate(raw_examples)
        if args.max_examples is not None:
            examples = examples[:args.max_examples]
        pipeline.synthetic_examples = examples
        print(f"  Generated {len(examples)} QA pairs")

        # Save synthesis checkpoints so the training monitor can display them
        iter_dir = os.path.join(args.output, f"iter{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)
        with open(os.path.join(iter_dir, "stage1_synthesis.json"), "w") as f:
            json.dump({
                "version": 1, "phase": "synthesis",
                "iteration": iteration + 1,
                "data": [{"question": e.question, "answer": e.answer} for e in raw_examples],
            }, f, default=str)
        with open(os.path.join(iter_dir, "stage1_deduped.json"), "w") as f:
            json.dump({
                "version": 1, "phase": "dedup",
                "iteration": iteration + 1,
                "data": [{"question": e.question, "answer": e.answer} for e in examples],
            }, f, default=str)

        # Stage 2: Rollouts + filtering
        # Cap workers for local Unsloth model (can't batch like vLLM)
        _rollout_workers = min(args.rollout_workers, 8)
        print(f"  Generating rollouts ({_rollout_workers} workers)...")
        final_examples = pipeline.run_stage_two(
            examples=examples,
            num_rollouts=args.rollouts_per_example,
            parallel_workers=_rollout_workers,
        )
        print(f"  {len(final_examples)} examples after filtering")

        if not final_examples or not pipeline.filtered_groups:
            print("  No training data — skipping.")
            continue

        # Build dataset
        rollout_dicts = []
        for group in pipeline.filtered_groups:
            for rollout in group.rollouts:
                rollout_dicts.append({
                    "prompt": group.prompt,
                    "rollout": rollout.steps,
                    "reward": 1.0 if rollout.passed else 0.0,
                })

        dataset = OfflineRolloutDataset.from_rollouts(rollout_dicts)
        print(f"  Training on {len(dataset.prompts)} groups, "
              f"{len(dataset)} rollouts")

        # Stage 3: OAPL training
        print("  Training with OAPL...")
        import time as _time
        _oapl_start = _time.monotonic()
        print("##KONASH:oapl_start##", flush=True)
        stats = trainer.train_epoch_torch(
            dataset=dataset,
            model_engine=engine,
            learning_rate=args.lr,
            max_grad_norm=args.max_grad_norm,
        )
        _oapl_dur = _time.monotonic() - _oapl_start
        print(f"  Loss: {stats['mean_loss']:.4f}  "
              f"Groups: {stats['num_groups']}  "
              f"Rollouts: {stats['num_rollouts']}")
        print(f"##KONASH:oapl_done:loss={stats['mean_loss']:.4f}##", flush=True)

        # Log OAPL stats
        try:
            from konash.training.logger import TrainingLogger
            _log = TrainingLogger(os.path.basename(args.output) or "default")
            _log.oapl(
                iteration=iteration + 1,
                loss=stats["mean_loss"],
                entropy=stats.get("mean_entropy", 0),
                num_groups=stats["num_groups"],
                num_rollouts=stats["num_rollouts"],
                learning_rate=args.lr,
                duration_seconds=_oapl_dur,
            )
        except Exception:
            pass

        all_iteration_stats.append({
            "iteration": iteration + 1,
            "examples": len(final_examples),
            **stats,
        })

        # Checkpoint
        iter_dir = os.path.join(args.output, f"iter{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)
        engine.save_adapter(os.path.join(iter_dir, "adapter"))

        # Save rollouts for reproducibility
        rollouts_path = os.path.join(iter_dir, "rollouts.json")
        with open(rollouts_path, "w") as f:
            json.dump(rollout_dicts, f, indent=2, default=str)

        # Rollouts in this mode are generated by the local on-cluster model,
        # so the next iteration naturally bootstraps from the trained policy.

    # Train value model (KARL Section 5.2: Qwen3-4B-Thinking)
    vm_stats = _train_value_model(args.output, args.iterations)
    if vm_stats:
        print("##KONASH:value_model_start##", flush=True)
        try:
            from konash.training.logger import TrainingLogger
            _log = TrainingLogger(os.path.basename(args.output) or "default")
            _log.value_model(
                loss=vm_stats.get("final_loss", 0),
                epochs=vm_stats.get("epochs", 0),
                duration_seconds=0,
            )
        except Exception:
            pass
        print(
            f"##KONASH:value_model_done:loss={vm_stats.get('final_loss', 0):.4f}##",
            flush=True,
        )

    # Save final meta
    meta_path = os.path.join(args.output, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model": args.model,
            "iterations": len(all_iteration_stats),
            "stats": all_iteration_stats,
            "value_model": True,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! {len(all_iteration_stats)} iterations")
    print(f"  Checkpoints: {args.output}")
    print(f"{'='*60}")
    print("##KONASH:complete##", flush=True)

    # Export (push to Hub, merge, GGUF)
    if args.push_to_hub or args.merge_and_export or args.export_gguf or args.deploy_together:
        export_model(engine, args)


def _train_sleep_wake_pipeline(args):
    """Full pipeline with vLLM sleep/wake for single-GPU iterative training.

    vLLM serves inference (synthesis + rollouts), then sleeps to free GPU.
    Unsloth loads for OAPL training, saves adapter, cleans up.
    vLLM wakes and hot-loads the trained LoRA for the next iteration.
    """
    from konash.training.backends import VLLMLifecycle
    from konash.training.unsloth_engine import UnslothEngine
    from konash.training.oapl import OAPLTrainer
    from konash.training.dataset import OfflineRolloutDataset
    from konash.corpus import Corpus
    from konash.synthesis.pipeline import SynthesisPipeline
    from konash.synthesis.qa import QuestionAnswerSynthesizer
    from konash.synthesis.rollouts import RolloutGenerator

    # Disable weight sharing — vLLM and Unsloth alternate, not share
    os.environ.pop("UNSLOTH_VLLM_STANDBY", None)

    print("=" * 60)
    print("  KONASH Sleep/Wake Pipeline (vLLM + Unsloth OAPL)")
    print("=" * 60)
    vllm_model = args.vllm_model or args.model
    print(f"  Train:      {args.model}")
    print(f"  vLLM:       {vllm_model}")
    print(f"  Corpus:     {args.corpus}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Synthesis:  {args.synthesis_calls} calls/iter")
    print(f"  TP:         {args.tensor_parallel}")
    if args.vllm_max_model_len is not None:
        print(f"  vLLM ctx:   {args.vllm_max_model_len}")
    if args.vllm_gpu_memory_utilization is not None:
        print(f"  vLLM mem:   {args.vllm_gpu_memory_utilization}")
    print(f"  Output:     {args.output}")
    print()

    vllm_extra_args: list[str] = []
    if args.vllm_gpu_memory_utilization is not None:
        vllm_extra_args.extend([
            "--gpu-memory-utilization",
            str(args.vllm_gpu_memory_utilization),
        ])

    # Start vLLM with sleep mode + LoRA.
    vllm = VLLMLifecycle(
        model=vllm_model,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.vllm_max_model_len,
        max_lora_rank=args.lora_r,
        extra_args=vllm_extra_args,
        log_dir=args.output,
    )

    os.makedirs(args.output, exist_ok=True)
    print("##KONASH:loading_model##", flush=True)
    print("Starting vLLM with sleep mode...")
    _model_t0 = time.time()
    vllm.start()
    served_model = vllm.served_model()
    print(f"##KONASH:model_loaded:{time.time() - _model_t0:.1f}s##", flush=True)
    print(f"  vLLM serving: {served_model}")

    # Build generate_fn from vLLM API
    generate_fn = _build_vllm_generate_fn(
        vllm.api_url,
        served_model,
        max_context_tokens=args.vllm_max_model_len,
    )

    # Ingest corpus
    print("##KONASH:loading_data##", flush=True)
    print("Ingesting corpus...")
    corpus = Corpus(args.corpus, chunk_size=512)
    corpus.ingest()
    print(f"  Indexed {corpus.num_documents} chunks")

    trainer = OAPLTrainer(beta_kl=args.beta_kl, beta_value=args.beta_value)
    all_iteration_stats = []
    prev_adapter_path = args.adapter  # None on first run

    try:
        for iteration in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration + 1}/{args.iterations}")
            print(f"{'='*60}")

            # ---- Stages 1+2: vLLM ACTIVE (synthesis + rollouts) ----

            synthesizer = QuestionAnswerSynthesizer(
                vector_search_tool=corpus,
                llm_fn=generate_fn,
            )
            rollout_gen = RolloutGenerator(
                search_tool=corpus,
                llm_fn=generate_fn,
                max_steps=args.rollout_max_steps,
            )
            pipeline = SynthesisPipeline(
                synthesizer=synthesizer,
                rollout_generator=rollout_gen,
            )

            # Stage 1: Synthesis
            print("  Synthesizing QA pairs...")
            raw_examples = []
            for call_idx in range(args.synthesis_calls):
                target_examples = _target_synthesis_examples(
                    current_count=len(raw_examples),
                    max_examples=args.max_examples,
                )
                if target_examples <= 0:
                    break
                batch = synthesizer.synthesize(
                    documents=None,
                    num_examples=target_examples,
                )
                raw_examples.extend(batch)
                for ex in batch:
                    q = ex.question.replace('\n', ' ').replace('#', '')
                    a = ex.answer.replace('\n', ' ').replace('#', '')
                    print(f"##KONASH:qa:{iteration + 1}:{q}\t{a}##", flush=True)
                print(f"  Call {call_idx + 1}/{args.synthesis_calls}: "
                      f"+{len(batch)} pairs ({len(raw_examples)} total)")
            examples = pipeline.deduplicate(raw_examples)
            if args.max_examples is not None:
                examples = examples[:args.max_examples]
            pipeline.synthetic_examples = examples
            print(f"  Generated {len(examples)} QA pairs")

            # Save synthesis checkpoints
            iter_dir = os.path.join(args.output, f"iter{iteration + 1}")
            os.makedirs(iter_dir, exist_ok=True)
            with open(os.path.join(iter_dir, "stage1_synthesis.json"), "w") as f:
                json.dump({
                    "version": 1, "phase": "synthesis",
                    "iteration": iteration + 1,
                    "data": [{"question": e.question, "answer": e.answer}
                             for e in raw_examples],
                }, f, default=str)
            with open(os.path.join(iter_dir, "stage1_deduped.json"), "w") as f:
                json.dump({
                    "version": 1, "phase": "dedup",
                    "iteration": iteration + 1,
                    "data": [{"question": e.question, "answer": e.answer}
                             for e in examples],
                }, f, default=str)

            # Stage 2: Rollouts + filtering
            # vLLM backend — use full parallelism, continuous batching handles it
            print(f"  Generating rollouts ({args.rollout_workers} workers)...")
            final_examples = pipeline.run_stage_two(
                examples=examples,
                num_rollouts=args.rollouts_per_example,
                parallel_workers=args.rollout_workers,
            )
            print(f"  {len(final_examples)} examples after filtering")

            if not final_examples or not pipeline.filtered_groups:
                print("  No training data — skipping.")
                continue

            # Build dataset
            rollout_dicts = []
            for group in pipeline.filtered_groups:
                for rollout in group.rollouts:
                    rollout_dicts.append({
                        "prompt": group.prompt,
                        "rollout": rollout.steps,
                        "reward": 1.0 if rollout.passed else 0.0,
                    })

            dataset = OfflineRolloutDataset.from_rollouts(rollout_dicts)
            print(f"  Training on {len(dataset.prompts)} groups, "
                  f"{len(dataset)} rollouts")

            # ---- Stage 3: OAPL Training (vLLM sleeps) ----
            should_sleep_vllm = _should_sleep_vllm_for_training(
                iteration=iteration,
                total_iterations=args.iterations,
            )
            if should_sleep_vllm:
                print("##KONASH:vllm_sleep##", flush=True)
                print("  Sleeping vLLM (freeing GPU for training)...")
                vllm.sleep()
            else:
                print("  Stopping vLLM for final training pass...")
                vllm.stop()

            # Load Unsloth engine (without fast_inference — vLLM is separate)
            print("  Loading Unsloth engine for OAPL...")
            engine = UnslothEngine(
                model_name=args.model,
                max_seq_length=args.max_seq_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                load_in_fp8=args.fp8 and not args.no_fp8,
                fast_inference=False,
            )

            # Load previous adapter for continued training (iteration 2+)
            if prev_adapter_path:
                print(f"  Loading previous adapter: {prev_adapter_path}")
                from peft import PeftModel
                engine.model = PeftModel.from_pretrained(
                    engine.model, prev_adapter_path, is_trainable=True,
                )

            engine.snapshot_reference()

            # OAPL training
            print("  Training with OAPL...")
            _oapl_start = time.monotonic()
            print("##KONASH:oapl_start##", flush=True)
            stats = trainer.train_epoch_torch(
                dataset=dataset,
                model_engine=engine,
                learning_rate=args.lr,
                max_grad_norm=args.max_grad_norm,
            )
            _oapl_dur = time.monotonic() - _oapl_start
            print(f"  Loss: {stats['mean_loss']:.4f}  "
                  f"Groups: {stats['num_groups']}  "
                  f"Rollouts: {stats['num_rollouts']}")
            print(f"##KONASH:oapl_done:loss={stats['mean_loss']:.4f}##", flush=True)

            # Log OAPL stats
            try:
                from konash.training.logger import TrainingLogger
                _log = TrainingLogger(os.path.basename(args.output) or "default")
                _log.oapl(
                    iteration=iteration + 1,
                    loss=stats["mean_loss"],
                    entropy=stats.get("mean_entropy", 0),
                    num_groups=stats["num_groups"],
                    num_rollouts=stats["num_rollouts"],
                    learning_rate=args.lr,
                    duration_seconds=_oapl_dur,
                )
            except Exception:
                pass

            all_iteration_stats.append({
                "iteration": iteration + 1,
                "examples": len(final_examples),
                **stats,
            })

            # Save adapter
            adapter_path = os.path.join(iter_dir, "adapter")
            engine.save_adapter(adapter_path)
            prev_adapter_path = adapter_path

            # Save rollouts
            with open(os.path.join(iter_dir, "rollouts.json"), "w") as f:
                json.dump(rollout_dicts, f, indent=2, default=str)

            # Free GPU VRAM before waking vLLM
            print("  Releasing Unsloth model...")
            engine.cleanup()
            del engine

            # Wake vLLM and hot-load trained LoRA for next iteration
            if should_sleep_vllm:
                print("##KONASH:vllm_wake##", flush=True)
                print("  Waking vLLM...")
                vllm.wake()

                lora_name = vllm.load_lora(adapter_path)
                if lora_name:
                    generate_fn = _build_vllm_generate_fn(
                        vllm.api_url,
                        lora_name,
                        max_context_tokens=args.vllm_max_model_len,
                    )
                    print(f"  Next iteration using LoRA: {lora_name}")
                else:
                    print("  WARNING: LoRA hot-load failed, using base model")

    finally:
        vllm.stop()

    # Value model training (KARL Section 5.2)
    vm_stats = _train_value_model(args.output, args.iterations)
    if vm_stats:
        print("##KONASH:value_model_start##", flush=True)
        try:
            from konash.training.logger import TrainingLogger
            _log = TrainingLogger(os.path.basename(args.output) or "default")
            _log.value_model(
                loss=vm_stats.get("final_loss", 0),
                epochs=vm_stats.get("epochs", 0),
                duration_seconds=0,
            )
        except Exception:
            pass
        print(
            f"##KONASH:value_model_done:loss={vm_stats.get('final_loss', 0):.4f}##",
            flush=True,
        )

    # Save final meta
    meta_path = os.path.join(args.output, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model": args.model,
            "iterations": len(all_iteration_stats),
            "stats": all_iteration_stats,
            "value_model": True,
            "sleep_wake": True,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete! {len(all_iteration_stats)} iterations")
    print(f"  Checkpoints: {args.output}")
    print(f"{'='*60}")
    print("##KONASH:complete##", flush=True)


def main():
    args = parse_args()

    if args.rollouts:
        train_from_rollouts(args)
    elif args.corpus:
        train_full_pipeline(args)
    else:
        print("ERROR: Provide either --rollouts (pre-generated) or --corpus (full pipeline)")
        sys.exit(1)


if __name__ == "__main__":
    main()
