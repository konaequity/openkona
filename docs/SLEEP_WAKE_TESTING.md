# Sleep/Wake Pipeline Testing Guide

## What's Been Built

The sleep/wake pipeline (`--vllm-sleep-wake` flag) enables single-GPU iterative training where vLLM and Unsloth alternate VRAM usage:

1. vLLM serves the model → synthesis + rollouts
2. vLLM sleeps (offloads weights to CPU) → GPU freed
3. Unsloth loads model → OAPL training → saves LoRA adapter
4. Unsloth cleanup → GPU freed
5. vLLM wakes → hot-loads trained LoRA → next iteration

### Key Files
- `konash/training/backends.py` — `VLLMLifecycle` class (start/stop/sleep/wake/load_lora)
- `scripts/train_oapl_unsloth.py` — `_train_sleep_wake_pipeline()` orchestration
- `konash/cloud.py` — `train_remote()` provisions GPU and streams progress
- `konash/synthesis/qa.py` — Agentic synthesis with `[synth]` trace logging

## Current Status

All code is implemented and unit-tested (211 tests pass). The pipeline has been tested on Shadeform 2x H100 PCIe with GLM 4.5 Air FP8. The infrastructure plumbing works but **synthesis is too slow on PCIe H100s** — each LLM call takes 5+ minutes due to slow inter-GPU communication.

## What Needs To Be Done

1. **Get synthesis working end-to-end on appropriate hardware**
2. **Verify the full loop**: synthesis → rollouts → vLLM sleep → OAPL training → wake → LoRA hot-load

## Hardware Requirements

| Model | Size (FP8) | Min Hardware | Recommended |
|-------|-----------|-------------|-------------|
| GLM 4.5 Air | ~53GB | 2x H100 SXM (NVLink) | 1x H200 (141GB) |
| Qwen2.5-7B-Instruct | ~14GB | 1x any H100 | 1x H100 PCIe |
| Qwen3.5-9B | ~18GB | 1x any H100 | 1x H100 PCIe (but it's a VL model) |

**Critical**: GLM 4.5 Air on **PCIe** H100s is ~5x slower than SXM due to TP=2 cross-GPU communication. Use SXM or a single H200.

**For pipeline validation**, use `Qwen2.5-7B-Instruct` on 1x H100 — it's fast and text-only.

## Pitfalls (Hard-Won Lessons)

### 1. Remote Setup Permissions
`_setup_remote()` installs packages into `/usr/local/lib/python3.10/dist-packages/` which is root-owned. The Shadeform user is `shadeform`, not root.

**Fix**: Already applied — uses `sudo -E env PATH=$PATH uv pip install --system --link-mode=copy`.

### 2. Disk Space
The root disk on Shadeform instances is ~97GB. GLM 4.5 Air FP8 is ~53GB. HuggingFace downloads to `~/.cache/huggingface` on the root disk → disk full.

**Fix**: Already applied — `_setup_remote()` symlinks `~/.cache/huggingface` to `/ephemeral/hf_cache` (1.5TB ephemeral disk).

### 3. vLLM Version
Without a version floor, `uv` resolves `vllm==0.2.5` (ancient, no `vllm serve` CLI).

**Fix**: Already applied — pinned `'vllm>=0.8'`.

### 4. Model ID Mismatch
vLLM needs the raw HuggingFace model ID (`zai-org/GLM-4.5-Air-FP8`), NOT the Unsloth wrapper (`unsloth/GLM-4.5-Air`). The Unsloth ID is only for the training engine.

### 5. Tool Call Parser
GLM 4.5 Air requires `--enable-auto-tool-choice --tool-call-parser glm45 --trust-remote-code` for vLLM to support function calling. Without this, the `/v1/chat/completions` endpoint returns 400 when `tools` are passed.

**Fix**: Already applied — `VLLMLifecycle` auto-detects GLM/Qwen models and adds the right flags.

### 6. `<think>` Tag Stripping
vLLM returns raw `<think>...</think>` tags in GLM/Qwen3 responses. Together AI strips them server-side. The `_build_vllm_generate_fn()` originally had a greedy regex `<think>.*` (with `re.DOTALL`) that ate the ENTIRE response after the opening `<think>` tag — including the `SEARCH:` and `PROPOSE:` actions the synthesizer needs.

**Fix**: Already applied — removed tag stripping from `generate_fn`. The synthesizer's `_clean_thinking_tags()` in `qa.py` handles it correctly (paired tags first, then unclosed).

### 7. Prebuilt Index Text Resolution
FinanceBench uses a prebuilt embedding index (`prebuilt_index.npz`) that stores vectors and doc_ids but NOT document text. Documents are stored as stubs with `text: ""`. The `Corpus.search()` method calls `_resolve_text()` to read files from disk on demand — but `corpus.vector_search.search()` skips this.

**Fix**: Already applied — synthesizer and rollout generator now receive `corpus` (the `Corpus` object) instead of `corpus.vector_search` (the raw `VectorSearchTool`).

**How to verify**: After `corpus.ingest()`, call `corpus.search("test query", top_k=1)` and check that `result[0]["text"]` is non-empty.

### 8. num_gpus=1 Hardcoded
`_find_cheapest_gpu()` was hardcoded to `num_gpus=1`. GLM 4.5 Air needs 2 GPUs.

**Fix**: Already applied — `train_remote()` auto-sets `num_gpus=2` for GLM 4.5 Air with sleep_wake mode.

### 9. SSH Dropping During pkill
SSH to Shadeform instances drops (exit 255) when running `pkill` commands that kill background processes. The SSH session itself can be affected.

**Workaround**: Don't combine `pkill` with other commands. Run `pkill` in a separate SSH call.

### 10. Multiple vLLM / Training Processes
When restarting, old vLLM processes may still hold GPU memory. New vLLM can't allocate VRAM → "Free memory on device cuda:1 (5.38/79.19 GiB)" error.

**Fix**: Before starting a new run:
```bash
sudo kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
sleep 5  # Wait for VRAM to free
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # Verify 0 MiB
```

## Testing Checklist

### Quick Validation (Qwen2.5-7B, 1x H100, ~$1.90/hr)
```bash
python3 -u -c "
from konash.cloud import train_remote
train_remote(
    corpus='financebench',
    base_model='Qwen/Qwen2.5-7B-Instruct',
    iterations=1, synthesis_calls=2,
    rollouts_per_example=2, rollout_max_steps=5,
    gpu='H100', keep_alive=True, sleep_wake=True,
)
"
```

### Full Test (GLM 4.5 Air, needs SXM H100 or H200)
```bash
python3 -u -c "
from konash.cloud import train_remote
train_remote(
    corpus='financebench',
    base_model='zai-org/GLM-4.5-Air-FP8',
    iterations=2, synthesis_calls=5,
    rollouts_per_example=4, rollout_max_steps=10,
    gpu='H100', keep_alive=True, sleep_wake=True,
)
"
```

### Manual SSH Debugging
```bash
# SSH into instance
ssh -i ~/.konash/shadeform_ssh_key shadeform@<IP>

# Check GPU
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Check processes
ps aux | grep -E 'train_oapl|vllm' | grep -v grep

# Test corpus search has text
cd ~/konash && PYTHONPATH=. python3 -c "
from konash.corpus import Corpus
c = Corpus('/home/shadeform/.konash/corpora/financebench/documents', chunk_size=512)
c.ingest()
r = c.search('3M capital expenditure', top_k=1)
print('Text:', repr(r[0].get('text', '')[:100]))
"

# Test vLLM tool calling
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-4.5-Air-FP8","messages":[{"role":"user","content":"Search for 3M revenue"}],"tools":[{"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"max_tokens":200}'

# Watch synthesis trace
tail -f ~/konash/training*.log | grep '\[synth\]'
```

## Synthesis Trace Output
With the `[synth]` trace prints, you should see:
```
[synth] step 0: calling LLM...
[synth] step 0: got 22 chars, preview: 'SEARCH: 3M financial results'
[synth] step 0: action=search query='3M financial results'
[synth] step 0: SEARCH #1 query='3M financial results' -> 20 results
[synth] step 1: calling LLM...
[synth] step 1: got 45 chars, preview: 'SEARCH: Nike revenue 2020'
[synth] step 1: action=search query='Nike revenue 2020'
...
[synth] step 8: action=propose examples=8
```

If `action=unknown` appears, the model isn't outputting `SEARCH:` or `PROPOSE:` — check if `<think>` tags are being stripped correctly.

If search returns 20 results but the model says "no content" — check that `Corpus.search()` (not `vector_search.search()`) is being used.
