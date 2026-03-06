<div align="center">

# KONASH

**Knowledge-grounded Off-policy Networks for Agentic System Harnesses**

<p>
Train knowledge agents that search, retrieve, compress, and reason — on a single GPU.
</p>

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](#)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](#)

</div>

KONASH trains knowledge agents via reinforcement learning that match or exceed frontier models on grounded reasoning tasks — at a fraction of the cost. **Single-GPU training, open-source models, 1/100th the compute.**

---

## Key Benefits

- **2-3x faster development** — Skip reward function engineering entirely
- **General-purpose** — Works across any task without modification
- **Strong performance** — Matches or exceeds hand-crafted rewards in 3/4 benchmarks
- **Easy integration** — Drop-in replacement for manual reward functions

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with KONASH
judged_group = await konash_score_group(group, "konash/oapl")
```

[Learn more about KONASH](https://kona.sh)

---

## KONASH Overview

KONASH is an open-source RL framework that improves agent reliability by training knowledge agents to search, retrieve, compress, and reason over evidence — all on a single GPU using off-policy RL. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://kona.sh).

### Notebooks

| Agent Task | Example Notebook | Description | Comparative Performance |
|---|---|---|---|
| **Trivia Night** | [Train agent](#) | Qwen 3.5 7B learns to answer multi-constraint trivia by searching Wikipedia | [Link coming soon] |
| **20 Questions** | [Train agent](#) | Qwen 3.5 7B learns to identify a mystery entity in 20 yes/no searches | [Link coming soon] |
| **GeoGuessr** | [Train agent](#) | Qwen 3.5 7B learns to pinpoint locations from landmark and terrain descriptions | [Link coming soon] |

## KONASH News

Explore our latest research and updates on building SOTA knowledge agents.

- **[OAPL: Off-Policy RL That Actually Works on One GPU](#)** — Train knowledge agents without multi-node clusters using large-batch iterative off-policy reinforcement learning.
- **[Agentic Data Synthesis: Let Your Model Write Its Own Curriculum](#)** — Generate diverse, grounded training data from any corpus — no manual annotation required.
- **[Parallel Thinking: How a 7B Model Beats Frontier Single-Shot](#)** — Scale quality at inference time with N parallel rollouts and generative aggregation.
- **[Compression as an RL Skill: Teaching Models What to Remember](#)** — Train context compression end-to-end with task reward, not as a separate summarization step.

[See all blog posts](https://kona.sh/blog)

## Why KONASH?

- KONASH provides a complete pipeline for training knowledge agents on **existing corpora**. We abstract the training, synthesis, and serving into a modular system that your code doesn't need to interface with.
- **Train from anywhere.** Run the KONASH client on your laptop and let the server kick off training on a single GPU — local or cloud. No multi-node clusters required.
- Integrations with hosted platforms like W&B and Langfuse provide flexible observability and **simplify debugging** across the full synthesis-train-eval loop.
- KONASH is customizable with **intelligent defaults**. You can configure OAPL hyperparameters, compression thresholds, and inference engine settings to meet specific needs, or take advantage of defaults optimized for single-GPU training efficiency and stability.

---

## Training Loop

KONASH's training loop is built around **large-batch iterative off-policy RL**. All rollouts are generated first, then training happens in a single offline pass.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KONASH Training Loop                         │
│                                                                     │
│  ┌───────────────┐   ┌────────────────┐   ┌─────────────────────┐  │
│  │ 1. Synthesize │ → │ 2. Generate    │ → │ 3. Filter & Build   │  │
│  │    QA Pairs   │   │    Rollouts    │   │    Offline Dataset  │  │
│  └───────────────┘   └────────────────┘   └─────────────────────┘  │
│        │                                            │               │
│        │             ┌────────────────┐              │               │
│        └──────────── │ 5. Update π_ref│ ←───────────┘               │
│                      │    Repeat      │                             │
│                      └────────────────┘                             │
│                             ↑                                       │
│                      ┌────────────────┐                             │
│                      │ 4. Train OAPL  │                             │
│                      │    (QLoRA)     │                             │
│                      └────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Synthesize** — Agentic QA synthesis explores your corpus via vector search, producing diverse, grounded question-answer pairs. Bootstrap from public datasets or generate from your own documents.

2. **Generate Rollouts** — The model (or latest checkpoint) generates 4 rollouts per prompt, interacting with vector search and compression tools. Each rollout is a full multi-step agent trajectory.

3. **Filter & Build Dataset** — Pass-rate filtering keeps prompts at the learning frontier (not too easy, not too hard). Quality filtering removes ambiguous or incorrect data. This becomes the offline training batch.

4. **Train OAPL** — Single-GPU QLoRA training using the OAPL least-squares regression objective. Rollouts are segmented at compression boundaries with tool-call masking on log-prob computation. ~4 hours per iteration.

5. **Iterate** — Swap in the trained checkpoint as the new reference policy, regenerate rollouts with the improved model, and train again. 2–3 iterations yields the best results.

---

## Agent Harness

KONASH includes a lightweight open-source agent harness — the runtime that's identical across data synthesis, training, evaluation, and production serving.

```
┌─────────────────────────────────────────────────┐
│                KONASH Harness                    │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Prompts  │→ │ Dispatcher│→ │   Strategy   │ │
│  └──────────┘  └───────────┘  └──────────────┘ │
│                                      │          │
│                     ┌────────────────┘          │
│                     ▼                           │
│  ┌──────────────────────────────────────────┐  │
│  │             Environment                   │  │
│  │  ┌────────┐  ┌────────┐  ┌───────────┐  │  │
│  │  │ Vector │  │ Reward │  │ Compress  │  │  │
│  │  │ Search │  │   Fn   │  │  Plugin   │  │  │
│  │  └────────┘  └────────┘  └───────────┘  │  │
│  └──────────────────────────────────────────┘  │
│                     ▲                           │
│  ┌──────────────────────────────────────────┐  │
│  │             Agent (LLM)                   │  │
│  │  ┌──────────┐     ┌──────────────────┐   │  │
│  │  │  vLLM /  │     │  LoRA Adapter    │   │  │
│  │  │ Unsloth  │     │  Hot-Swap        │   │  │
│  │  └──────────┘     └──────────────────┘   │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

- **Dispatcher** — Feeds prompts to environments, collects rollouts
- **Strategy** — Standard (single rollout) or Parallel Thinking (N rollouts + generative aggregation)
- **Environment** — Manages tool calls, rewards, and context lifecycle
- **Compression Plugin** — Triggers when context exceeds threshold; model compresses its own history, trained end-to-end via OAPL reward signal
- **Vector Search** — FAISS index with CPU-based embedding model (110M params). No GPU contention
- **Agent** — vLLM serving with LoRA hot-swap between base and trained checkpoints

---

## Quickstart

```bash
pip install konash
```

### Synthesize training data over your corpus

```bash
konash synth --corpus /path/to/documents --output ./data/synthetic
```

### Generate rollouts

```bash
konash rollouts --data ./data/synthetic --model Qwen/Qwen3.5-7B --output ./data/rollouts
```

### Train with OAPL

```bash
konash train --data ./data/rollouts --model Qwen/Qwen3.5-7B --output ./checkpoints/iter1
```

### Evaluate on KONASHBench

```bash
konash eval --model ./checkpoints/iter1 --benchmark konashbench
```

### Run inference with Parallel Thinking

```bash
konash serve --model ./checkpoints/iter1 --parallel-thinking 10
```

---

## Supported Models

KONASH works with any model supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) for training and [vLLM](https://github.com/vllm-project/vllm) for inference. Recommended base models:

| Model | Params | Why |
|---|---|---|
| **Qwen 3.5** | 7B | Strong tool-calling, long context, sweet spot for single-GPU OAPL |
| **GLM-4.5-Air** | 12B active / 106B total (MoE) | MIT licensed MoE model. Needs H100 80GB+ |
| **Any Unsloth model** | Varies | If Unsloth supports it and it fits your GPU, it works with KONASH |

---

## KONASHBench

A lightweight evaluation suite covering six grounded reasoning capabilities:

| Capability | Dataset | Questions | Description |
|---|---|---|---|
| Constraint-driven entity search | HotpotQA-Hard | 500 | Multi-hop, verifiable entity search |
| Cross-document report synthesis | TREC-Biogen | 65 | Synthesize findings across biomedical sources |
| Tabular numerical reasoning | FinanceBench | 150 | Navigate financial reports, extract and compute |
| Exhaustive entity retrieval | QAMPARI | 1,000 | Find all entities matching a condition |
| Procedural technical reasoning | FreshStack | 203 | Step-by-step solutions from docs and source code |
| Domain-specific search | Custom | Varies | Build per deployment over your own corpus |

Four of six benchmarks are publicly available at zero dataset cost.

Evaluation uses **nugget-based completion scoring** — the same methodology as TREC-RAG and DeepScholar-Bench.

---

## Cost Comparison

| Component | Traditional Approach | KONASH | Reduction |
|---|---|---|---|
| Training infrastructure | Multi-node GPU cluster | 1x A100/H100 | ~50-100x |
| Data synthesis | Frontier model APIs | Self-hosted open-weight model | ~100x |
| Quality filtering | Frontier API judges | Rule-based + self-judge | ~200x |
| Rollout generation | Multi-GPU vLLM cluster | 1x GPU vLLM | ~20x |
| RL training | Multi-GPU DDP/FSDP | 1x GPU QLoRA | ~30x |
| Embedding / indexing | 8B model on GPU | 110M model on CPU | ~50x |
| **Total per iteration** | **~$10K-50K** | **~$100-500** | **~100x** |

---

## The Bigger Picture: KONASH as a Platform

KONASH v1 is an open-source training recipe. The endgame is a platform:

1. **Collect training data at scale** — Point the agentic synthesis pipeline at any corpus. It explores via vector search and produces diverse, hard, verifiable training data — no manual annotation.

2. **Select the highest-quality training data** — Pass-rate filtering, quality judges, and deduplication automatically surface the data where learning signal is richest.

3. **Fine-tune open-source LLMs** — OAPL is model-agnostic. Plug in Qwen 3.5, GLM-4.5-Air, or any Unsloth-supported model. QLoRA on a single GPU. Full fine-tuning if you have the compute.

4. **Evaluate fine-tuned models** — KONASHBench tracks in-distribution gains, OOD generalization, cost per query, and latency. Know exactly when your 7B model beats GPT-5 on your domain.

5. **Host inference with zero lock-in** — Deploy via vLLM with LoRA hot-swapping. Parallel thinking scales quality. Same harness in training serves in production. Your model, your weights, your infrastructure.

---

## Contributing

KONASH is in active development and contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

---

## Citation

```bibtex
@misc{konaequity2026konash,
  author = {Kona Equity},
  title = {KONASH: Knowledge-grounded Off-policy Networks for Agentic System Harnesses},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/konaequity/openkona}}
}
```

---

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

---

## Credits

KONASH builds directly on the research and open-source work of:

- [OAPL](https://arxiv.org/abs/2503.01735) — Ritter et al., 2026 (the RL algorithm)
- [Unsloth](https://github.com/unslothai/unsloth) — Parameter-efficient training backend
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput inference engine
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
