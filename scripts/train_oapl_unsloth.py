#!/usr/bin/env python3
"""OAPL training with Unsloth on GLM 4.5 Air.

Designed for Together AI GPU clusters (2× H100 SXM, 160 GB total VRAM).

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
    TOGETHER_API_KEY       — Only needed for API-backed rollout generation
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
    p.add_argument("--model", type=str, default="zai-org/GLM-4.5-Air",
                    help="HuggingFace model ID")
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

    # Output
    p.add_argument("--output", type=str, default="./checkpoints",
                    help="Output directory for checkpoints")

    # vLLM (use vLLM for synthesis/rollouts instead of Unsloth generate)
    p.add_argument("--vllm", action="store_true",
                    help="Use vLLM for synthesis/rollouts (start server, generate, stop, then train)")
    p.add_argument("--vllm-url", type=str, default=None,
                    help="URL of an already-running vLLM server (skip auto-start)")
    p.add_argument("--tensor-parallel", type=int, default=2,
                    help="Tensor parallel size for vLLM (default: 2 for 2xH100)")

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
    from konash.training.logger import TrainingLogger, configure_file_logging

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
    debug_log_path = configure_file_logging(project_name)
    log.start(
        iterations=args.epochs,
        corpus=args.rollouts or "(rollouts)",
        model=args.model,
    )
    print(f"  Debug log: {debug_log_path}")

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


def _build_vllm_generate_fn(vllm_url: str, model: str):
    """Build an llm_fn that calls a vLLM OpenAI-compatible server."""
    import re as _re
    import urllib.request
    import urllib.error

    def generate_fn(messages, **kwargs):
        # Normalize token limit keys
        max_tokens = kwargs.pop("max_new_tokens", kwargs.pop("max_tokens", 1024))
        temperature = kwargs.pop("temperature", 0.7)

        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

        req = urllib.request.Request(
            f"{vllm_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            # Context-length errors: halve max_tokens and retry once
            if e.code == 400 and "maximum context length" in err_body:
                retry_tokens = max(max_tokens // 2, 128)
                retry_body = json.dumps({
                    "model": model,
                    "messages": messages,
                    "max_tokens": retry_tokens,
                    "temperature": temperature,
                }).encode()
                req2 = urllib.request.Request(
                    f"{vllm_url}/chat/completions",
                    data=retry_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req2, timeout=300) as resp:
                    data = json.loads(resp.read())
            else:
                raise RuntimeError(
                    f"vLLM HTTP {e.code}: {err_body[:500]}"
                ) from e

        content = data["choices"][0]["message"]["content"]
        # Strip thinking tags (Qwen3, GLM reasoning)
        content = _re.sub(r"<think>.*?</think>\s*", "", content, flags=_re.DOTALL)
        content = _re.sub(r"<think>.*", "", content, flags=_re.DOTALL).strip()

        return {"role": "assistant", "content": content}

    return generate_fn


def _build_unsloth_generate_fn(engine):
    """Build an llm_fn backed by a local Unsloth engine."""
    def generate_fn(messages, **kwargs):
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens", 512)
        return engine.generate(messages, **kwargs)

    return generate_fn


def train_full_pipeline(args):
    """Full iterative pipeline: synthesis → rollouts → OAPL → repeat."""
    from konash.training.unsloth_engine import UnslothEngine
    from konash.training.oapl import OAPLTrainer
    from konash.training.dataset import OfflineRolloutDataset
    from konash.corpus import Corpus
    from konash.synthesis.pipeline import SynthesisPipeline
    from konash.synthesis.qa import QuestionAnswerSynthesizer
    from konash.synthesis.rollouts import RolloutGenerator

    use_vllm = args.vllm or args.vllm_url
    backend_label = "vLLM" if use_vllm else "Unsloth"

    print("=" * 60)
    print(f"  KONASH Full Pipeline ({backend_label} + OAPL)")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Backend:    {backend_label}")
    print(f"  Corpus:     {args.corpus}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Synthesis:  {args.synthesis_calls} calls/iter")
    print(f"  Output:     {args.output}")
    print()

    # --- vLLM lifecycle helpers (local server mode only) ---
    import subprocess as _sp
    import urllib.request
    import urllib.error

    _vllm_proc = None  # Tracks the local vLLM process
    _vllm_log_fp = None
    _manages_vllm = use_vllm and not args.vllm_url  # True if we own the server
    _vllm_model_source = args.model

    if args.vllm_url and args.iterations > 1:
        raise RuntimeError(
            "--vllm-url cannot match KARL iterative training across multiple "
            "iterations because this script cannot advance the external "
            "server to the newly trained policy. Use --vllm for managed "
            "local serving, or run a single iteration."
        )

    _VLLM_BASE = "http://localhost:8000"
    _lora_counter = 0  # Monotonically increasing LoRA adapter ID

    def _start_local_vllm():
        """Start a local vLLM server with sleep mode + LoRA support."""
        import shutil
        nonlocal _vllm_proc, _vllm_log_fp, _vllm_model_source
        os.makedirs(args.output, exist_ok=True)
        # Resolve the vllm binary — it may not be on $PATH (e.g. ~/.local/bin)
        vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            # Common pip --user install location
            _candidate = os.path.expanduser("~/.local/bin/vllm")
            if os.path.isfile(_candidate):
                vllm_bin = _candidate
            else:
                # Fall back to running as a Python module
                vllm_bin = None
        if vllm_bin:
            vllm_cmd = [vllm_bin, "serve", _vllm_model_source]
        else:
            vllm_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                        "--model", _vllm_model_source]
        vllm_cmd += [
            "--tensor-parallel-size", str(args.tensor_parallel),
            "--port", "8000", "--host", "0.0.0.0",
            "--max-model-len", "65536",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
            # Eager mode avoids CUDA graph compilation issues across
            # PyTorch / vLLM version combinations.
            "--enforce-eager",
            # Sleep mode: offload weights to CPU instead of killing server
            "--enable-sleep-mode",
            # LoRA: enable dynamic adapter loading between iterations
            "--enable-lora",
            "--max-lora-rank", str(args.lora_r),
        ]
        env = os.environ.copy()
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"

        log_path = os.path.join(args.output, "vllm.log")
        _vllm_log_fp = open(log_path, "a")
        _vllm_proc = _sp.Popen(
            vllm_cmd,
            stdout=_vllm_log_fp,
            stderr=_sp.STDOUT,
            env=env,
        )
        print(f"  Waiting for model to load (PID {_vllm_proc.pid})...")
        _start = time.time()
        while time.time() - _start < 600:
            if _vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {_vllm_proc.returncode}. "
                    f"Check {log_path}"
                )
            try:
                req = urllib.request.Request(f"{_VLLM_BASE}/health")
                with urllib.request.urlopen(req, timeout=5):
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(10)
        else:
            _stop_local_vllm()
            raise RuntimeError("vLLM server did not start within 10 minutes")
        print(f"  vLLM ready ({time.time() - _start:.0f}s)")

    def _sleep_vllm():
        """Put vLLM into sleep mode — offload weights to CPU, free GPU memory."""
        try:
            req = urllib.request.Request(
                f"{_VLLM_BASE}/sleep?level=1", method="POST",
            )
            with urllib.request.urlopen(req, timeout=30):
                pass
            print("  vLLM sleeping (weights offloaded to CPU)")
        except (urllib.error.URLError, OSError) as e:
            print(f"  WARNING: vLLM sleep failed ({e}), falling back to stop")
            _stop_local_vllm()

    def _wake_vllm():
        """Wake vLLM from sleep — reload weights to GPU from CPU."""
        try:
            req = urllib.request.Request(
                f"{_VLLM_BASE}/wake_up", method="POST",
            )
            with urllib.request.urlopen(req, timeout=120):
                pass
            # Wait for health check after wake
            for _ in range(30):
                try:
                    hreq = urllib.request.Request(f"{_VLLM_BASE}/health")
                    with urllib.request.urlopen(hreq, timeout=5):
                        print("  vLLM awake")
                        return
                except (urllib.error.URLError, OSError):
                    time.sleep(2)
            print("  WARNING: vLLM wake health check timed out")
        except (urllib.error.URLError, OSError) as e:
            print(f"  WARNING: vLLM wake failed ({e}), restarting")
            _stop_local_vllm()
            _start_local_vllm()

    def _load_lora_into_vllm(adapter_path: str):
        """Hot-load a LoRA adapter into the running vLLM server.

        After OAPL training, this loads the trained adapter so subsequent
        rollouts come from the updated policy (KARL Section 4.2 iterative
        training: π_ref advances each iteration).
        """
        nonlocal _lora_counter
        _lora_counter += 1
        lora_name = f"konash-iter-{_lora_counter}"

        body = json.dumps({
            "lora_name": lora_name,
            "lora_path": adapter_path,
        }).encode()
        try:
            req = urllib.request.Request(
                f"{_VLLM_BASE}/v1/load_lora_adapter",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60):
                pass
            print(f"  LoRA adapter loaded into vLLM: {lora_name}")
            return lora_name
        except (urllib.error.URLError, OSError) as e:
            print(f"  WARNING: LoRA hot-load failed ({e})")
            return None

    def _stop_local_vllm():
        """Stop the local vLLM server."""
        nonlocal _vllm_proc, _vllm_log_fp
        if _vllm_proc is not None and _vllm_proc.poll() is None:
            print("  Stopping vLLM server...")
            _vllm_proc.terminate()
            try:
                _vllm_proc.wait(timeout=30)
            except _sp.TimeoutExpired:
                _vllm_proc.kill()
                _vllm_proc.wait()
            time.sleep(3)
        _vllm_proc = None
        if _vllm_log_fp is not None:
            _vllm_log_fp.close()
            _vllm_log_fp = None

    # The engine used for training/export. In managed vLLM mode this is
    # created lazily when we first enter Stage 3.
    engine = None

    try:
        # --- Build initial generate_fn ---
        vllm_url = None
        if use_vllm:
            if args.vllm_url:
                vllm_url = args.vllm_url
                print(f"Using existing vLLM server at {vllm_url}")
            else:
                print("Starting vLLM server...")
                _start_local_vllm()
                vllm_url = "http://localhost:8000/v1"

            generate_fn = _build_vllm_generate_fn(vllm_url, _vllm_model_source)
        else:
            # Load model via Unsloth for generation
            print("Loading model via Unsloth...")
            engine = UnslothEngine(
                model_name=args.model,
                max_seq_length=args.max_seq_length,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                load_in_fp8=args.fp8 and not args.no_fp8,
            )
            generate_fn = _build_unsloth_generate_fn(engine)

        # Ingest corpus
        print("Ingesting corpus...")
        corpus = Corpus(args.corpus, chunk_size=512)
        corpus.ingest()
        print(f"  Indexed {corpus.num_documents} chunks")

        # Trainer (Unsloth engine loaded later for vLLM path)
        trainer = OAPLTrainer(beta_kl=args.beta_kl, beta_value=args.beta_value)
        oapl_engine = None  # Lazily loaded for vLLM path

        # Initialize training logger so the dashboard can track this run
        from konash.training.logger import TrainingLogger, configure_file_logging
        project_name = os.path.basename(os.path.abspath(args.output)) or "default"
        log = TrainingLogger(project_name)
        debug_log_path = configure_file_logging(project_name)
        log.start(
            iterations=args.iterations,
            corpus=args.corpus,
            model=args.model,
        )
        print(f"  Debug log: {debug_log_path}")

        # For Unsloth path, snapshot reference now
        if not use_vllm:
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
            # Detect optimal concurrency for the backend.
            # gpu_specs comes from Shadeform provisioning (set via --gpu-specs)
            # and tells us actual VRAM so concurrency adapts to H100/H200/B200/etc.
            from konash.synthesis.rollouts import detect_concurrency
            _gpu_specs = getattr(args, 'gpu_specs', None)
            concurrency = detect_concurrency(
                vllm_url if use_vllm else None,
                gpu_specs=_gpu_specs,
            )
            if use_vllm:
                print(f"  Concurrency: {concurrency} (vLLM continuous batching)")

            rollout_gen = RolloutGenerator(
                search_tool=corpus,
                llm_fn=generate_fn,
                max_steps=args.rollout_max_steps,
                concurrency=concurrency,
            )
            pipeline = SynthesisPipeline(
                synthesizer=synthesizer,
                rollout_generator=rollout_gen,
            )

            # Parallelize synthesis calls — each call is an independent
            # multi-turn agent loop. With vLLM, running many concurrently
            # keeps the GPU saturated via continuous batching.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            _synth_start = time.time()

            synth_workers = min(args.synthesis_calls, concurrency)
            examples_per_call = 8
            min_searches_per_call = max(3, examples_per_call)
            min_llm_turns_per_call = min_searches_per_call + 1  # searches + first proposal
            print(f"  Running {args.synthesis_calls} synthesis calls "
                  f"({synth_workers} parallel workers)")
            print(
                f"  Each synthesis call needs at least {min_llm_turns_per_call} "
                f"LLM turns ({min_searches_per_call} searches + 1 proposal), "
                f"so expect at least "
                f"{args.synthesis_calls * min_llm_turns_per_call} chat completions"
            )

            raw_examples = []
            completed = 0

            # Prepare incremental save path for real-time dashboard updates
            iter_dir = os.path.join(args.output, f"iter{iteration + 1}")
            os.makedirs(iter_dir, exist_ok=True)
            _synth_file = os.path.join(iter_dir, "stage1_deduped.json")
            _synth_progress_file = os.path.join(iter_dir, "stage1_progress.json")
            _synth_lock = threading.Lock()

            def _save_synth_incremental():
                """Save current QA pairs so the dashboard updates in real time."""
                with _synth_lock:
                    data = {
                        "examples": [
                            {"question": ex.question, "answer": ex.answer,
                             "citations": getattr(ex, "citations", [])}
                            for ex in raw_examples
                        ]
                    }
                try:
                    with open(_synth_file, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                except OSError:
                    pass

            def _save_synth_progress(last_call=None):
                elapsed = time.time() - _synth_start
                calls_done = completed
                rate = (calls_done / elapsed) if elapsed > 0 else 0.0
                eta_s = (
                    (args.synthesis_calls - calls_done) / rate
                    if rate > 0 and calls_done < args.synthesis_calls
                    else 0.0
                )
                payload = {
                    "completed_calls": calls_done,
                    "total_calls": args.synthesis_calls,
                    "parallel_workers": synth_workers,
                    "raw_examples": len(raw_examples),
                    "elapsed_seconds": elapsed,
                    "calls_per_second": rate,
                    "eta_seconds": eta_s,
                    "min_llm_turns_per_call": min_llm_turns_per_call,
                }
                if last_call is not None:
                    payload["last_call"] = last_call
                try:
                    with open(_synth_progress_file, "w") as f:
                        json.dump(payload, f, indent=2, default=str)
                except OSError:
                    pass

            def _synth_one(_idx):
                t0 = time.time()
                result = synthesizer.synthesize(
                    documents=None,
                    num_examples=examples_per_call,
                )
                return {
                    "call_idx": _idx,
                    "examples": result,
                    "duration_s": time.time() - t0,
                }

            with ThreadPoolExecutor(max_workers=synth_workers) as pool:
                futures = [
                    pool.submit(_synth_one, i)
                    for i in range(args.synthesis_calls)
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with _synth_lock:
                            raw_examples.extend(result["examples"])
                        completed += 1
                        elapsed = time.time() - _synth_start
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        eta_s = (
                            (args.synthesis_calls - completed) / rate
                            if rate > 0 and completed < args.synthesis_calls
                            else 0.0
                        )
                        print(
                            f"    Synthesis: {completed}/{args.synthesis_calls} "
                            f"calls, {len(raw_examples)} raw QA pairs, "
                            f"last_call={result['call_idx'] + 1} "
                            f"({result['duration_s']:.1f}s), "
                            f"elapsed={elapsed:.1f}s, eta={eta_s:.1f}s"
                        )
                        _save_synth_incremental()
                        _save_synth_progress({
                            "call_idx": result["call_idx"] + 1,
                            "duration_s": result["duration_s"],
                            "examples": len(result["examples"]),
                        })
                    except Exception as e:
                        completed += 1
                        print(f"    Synthesis call failed: {e}")
                        _save_synth_progress({
                            "error": str(e),
                        })

            _synth_dur = time.time() - _synth_start if '_synth_start' in dir() else 0
            examples = pipeline.deduplicate(raw_examples)
            if args.max_examples is not None:
                examples = examples[:args.max_examples]
            pipeline.synthetic_examples = examples
            print(f"  Generated {len(examples)} QA pairs (from {len(raw_examples)} raw)")

            # Log synthesis completion + save data for dashboard
            log.synthesis(
                iteration=iteration + 1,
                calls_completed=completed,
                calls_total=args.synthesis_calls,
                raw_pairs=len(raw_examples),
                deduped=len(examples),
                duration_seconds=_synth_dur,
            )

            # Overwrite incremental file with final deduped examples
            synth_data = {
                "examples": [
                    {"question": ex.question, "answer": ex.answer,
                     "citations": getattr(ex, "citations", [])}
                    for ex in examples
                ]
            }
            with open(_synth_file, "w") as f:
                json.dump(synth_data, f, indent=2, default=str)

            # Stage 2: Rollouts + filtering
            print("  Generating rollouts...")
            _rollout_start = time.time()
            final_examples = pipeline.run_stage_two(
                examples=examples,
                num_rollouts=args.rollouts_per_example,
            )
            _rollout_dur = time.time() - _rollout_start
            print(f"  {len(final_examples)} examples after filtering")

            if not final_examples or not pipeline.filtered_groups:
                print("  No training data — skipping.")
                continue

            # Log rollouts completion
            total_rollouts = sum(len(g.rollouts) for g in pipeline.rollout_groups)
            avg_pass = (
                sum(g.pass_rate for g in pipeline.rollout_groups) / len(pipeline.rollout_groups)
                if pipeline.rollout_groups else 0
            )
            log.rollouts(
                iteration=iteration + 1,
                examples=len(examples),
                rollouts=total_rollouts,
                filtered=len(final_examples),
                pass_rate=round(avg_pass, 3),
                duration_seconds=_rollout_dur,
            )

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

            # Save rollouts
            iter_dir = os.path.join(args.output, f"iter{iteration + 1}")
            os.makedirs(iter_dir, exist_ok=True)
            rollouts_path = os.path.join(iter_dir, "rollouts.json")
            with open(rollouts_path, "w") as f:
                json.dump(rollout_dicts, f, indent=2, default=str)

            # Stage 3: OAPL training
            # Sleep vLLM to free GPU memory for Unsloth training.
            # Sleep offloads weights to CPU pinned memory (~2s) instead of
            # killing the process (~30-60s restart). After training, we wake
            # vLLM and hot-load the trained LoRA adapter.
            if use_vllm:
                if _manages_vllm:
                    _sleep_vllm()

                if oapl_engine is None:
                    print("  Loading model via Unsloth for OAPL training...")
                    oapl_engine = UnslothEngine(
                        model_name=args.model,
                        max_seq_length=args.max_seq_length,
                        lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        load_in_fp8=args.fp8 and not args.no_fp8,
                    )
                    oapl_engine.snapshot_reference()
                engine = oapl_engine

            print("  Training with OAPL...")
            import time as _time
            _oapl_start = _time.monotonic()
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

            # Log OAPL stats
            try:
                log.oapl(
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

            # Save adapter checkpoint
            engine.save_adapter(os.path.join(iter_dir, "adapter"))

            # KARL iterative training (Section 4.2): advance π_ref to the
            # trained policy and serve it for the next iteration's rollouts.
            if use_vllm and iteration < args.iterations - 1:
                engine.snapshot_reference()
                print(f"  Updated π_ref for iteration {iteration + 2}")

            # Wake vLLM and hot-load the trained LoRA adapter so subsequent
            # rollouts come from the updated policy. No server restart needed.
            if _manages_vllm and iteration < args.iterations - 1:
                adapter_path = os.path.join(iter_dir, "adapter")
                print("  Waking vLLM and loading trained adapter...")
                _wake_vllm()
                lora_name = _load_lora_into_vllm(adapter_path)
                if lora_name:
                    # Update generate_fn to request the LoRA adapter
                    generate_fn = _build_vllm_generate_fn(
                        vllm_url, lora_name,
                    )

        # Sleep vLLM before value-model work to free GPU memory.
        # Final stop happens in the finally block.
        if _manages_vllm:
            _sleep_vllm()

        # Train value model (KARL Section 5.2: Qwen3-4B-Thinking)
        vm_stats = _train_value_model(args.output, args.iterations)
        if vm_stats:
            try:
                log.value_model(
                    loss=vm_stats.get("final_loss", 0),
                    epochs=vm_stats.get("epochs", 0),
                    duration_seconds=0,
                )
            except Exception:
                pass

        log.complete(
            iterations=len(all_iteration_stats),
            total_seconds=sum(s.get("wall_time_s", 0) for s in all_iteration_stats if isinstance(s, dict)),
            stats=all_iteration_stats,
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

        # Export (push to Hub, merge, GGUF)
        if args.push_to_hub or args.merge_and_export or args.export_gguf or args.deploy_together:
            if engine is None:
                raise RuntimeError(
                    "Cannot export because no training step completed. "
                    "No model engine is available."
                )
            export_model(engine, args)
    finally:
        if _manages_vllm:
            _stop_local_vllm()


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
