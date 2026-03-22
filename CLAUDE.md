# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KONASH (Knowledge-grounded Off-policy Networks for Agentic System Harnesses) — an RL-based framework for training knowledge agents that search, retrieve, compress, and reason over document corpora. Inspired by the KARL paper (Databricks, March 2026). Reference paper at `karl.pdf`.

## Commands

```bash
# Install (editable, dev + search)
pip install -e ".[dev,search]"

# Install with training deps
pip install -e ".[dev,train]"

# Install everything
pip install -e ".[all,dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_corpus.py

# Run a single test
pytest tests/test_corpus.py::test_name -v

# CLI
konash setup    # configure API keys
konash train    # interactive training wizard
konash ask      # query trained agent
konash search   # search corpus
konash download # download benchmark corpora
konash status   # check configuration
```

## Architecture

### Two-stage training pipeline

**Stage 1 — Synthesis + Dedup** (`konash/synthesis/pipeline.py`):
- `qa.py` — Agentic multi-turn synthesizer that searches corpus, extracts facts, proposes multi-constraint QA pairs
- `dedup.py` — MiniLM embedding-based near-duplicate detection

**Stage 2 — Rollouts + Filtering** (`konash/synthesis/pipeline.py`):
- `rollouts.py` — Multi-step solver agent generates answer trajectories (parallelized: 4 threads within-group, 4 workers across QA pairs)
- `filters.py` — Pass-rate filter keeps questions at learning frontier (default range [0.1, 0.9])

### Training (`konash/training/`)
- `backends.py` — Synthesis runtime lifecycle: `OpenAIConfig` (resolved connection info), `SynthesisRuntimeBackend` (ABC), `ShadeformSynthesisBackend` (provisions GPU, installs/starts vLLM, verifies `/v1/models`, supports warm reuse and in-place refresh)
- `execution.py` — Stage-based execution planning: `local_prep` (synthesis via Shadeform vLLM, OAPL on separate GPU) or `remote_full` (all stages on one GPU, required for multi-iteration)
- `oapl.py` — OAPL trainer: squared-advantage loss with KL regularization, soft value estimation via logsumexp over grouped rollouts
- `unsloth_engine.py` — Unsloth engine for MoE models (GLM 4.5 Air). LoRA targets: `gate_up_proj`, `down_proj` (NOT `gate_proj`, `up_proj` — these are fused in MoE). No 4-bit for MoE; use FP8.
- `dataset.py` — `OfflineRolloutDataset` groups rollouts by prompt

### Inference (`konash/inference/`)
- Standard single rollout, parallel thinking (N independent rollouts + generative aggregation), and Value-Guided Search (learned value model scores partial rollouts, BFS-style exploration)

### High-level API (`konash/api.py`)
- `Agent` class with `train()`, `solve()`, `from_preset()`. Model presets: `glm-4.5-air-together`, `glm-4.5-air-unsloth`, `glm-4.5-air-zhipu`

### Corpus & Retrieval (`konash/corpus.py`, `konash/retrieval/`)
- Corpus ingests documents, chunks with overlap (512 words, 64-word overlap), embeds via character-trigram hash (default, zero ML deps) or real models (MiniLM, Qwen3)
- FAISS vector search with BM25 fallback

### Agent Harness (`konash/harness/`)
- Environment manages agent-tool interaction loop; Dispatcher spawns parallel environments; plugin system for compression and step budgets

## Key Design Patterns

- **Lazy imports**: Core package has minimal deps (numpy, rich, google-genai). Training, search, and data deps are optional and imported only when needed.
- **OpenAI-compatible API**: LLM calls use generic chat completions protocol, abstracting away provider differences.
- **Spec-first testing**: `tests/conftest.py` uses `SymbolSpec` + `load_symbol()` to verify KARL spec compliance by dynamically importing and checking required symbols.
- **Qwen3 quirk**: Strip `<think>...</think>` tags (and unclosed `<think>.*`) from all Qwen3 model responses — they consume token budget.

## CI

GitHub Actions on push/PR to main. Matrix: Python 3.11, 3.12. Runs import checks for core modules then `pytest -ra --tb=short`. No linting or type checking enforced.

## Config & Credentials

API keys stored in `~/.konash/config.json` (created by `konash setup`). Env vars: `SHADEFORM_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`, `ZHIPU_API_KEY`. `TOGETHER_API_KEY` is optional (eval/serving only). Checkpoints saved under `.konash/<project>/checkpoints/`.
