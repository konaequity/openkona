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

## Why KONASH?

- **Cost** — State-of-the-art search agents at ~$100–500 per training iteration instead of ~$10K–50K. A single A100/H100 replaces a multi-node cluster.
- **Quality** — RL-trained agents that search more efficiently, retrieve more diversely, and reason more accurately than their base models. The gains come from the algorithm (OAPL), not model scale.
- **Quality Consistency** — Parallel thinking (N=10–20 rollouts + aggregation) turns probabilistic search into near-deterministic accuracy. Cheap rollouts on a 7B model mean you can afford this on every query.

---

## KONASH Overview

KONASH is an open-source RL framework purpose-built for **knowledge agents** — models that iteratively search, retrieve, compress context, and reason over evidence to answer complex questions. It trains on a single GPU using parameter-efficient methods and off-policy RL.

Unlike general-purpose agent RL frameworks, KONASH ships with the full knowledge-agent stack: agentic data synthesis, vector search environment, compression-as-RL-skill, multi-task training, and domain-specific evaluation.

### Core Innovations

| Innovation | What It Does | KONASH Implementation |
|---|---|---|
| **OAPL** | Off-policy RL via least-squares regression — no online RL instability | QLoRA on single GPU; same algorithm, 1/100th compute |
| **Agentic Data Synthesis** | Two-stage pipeline generates grounded QA pairs + solution rollouts | Self-hosted on open-weight models; no frontier API calls |
| **Compression as RL Skill** | Model learns *what* to compress to maximize task reward | End-to-end in the OAPL training loop |
| **Multi-Task RL** | Combined loss across structurally different tasks for OOD generalization | Same — train on entity search + report synthesis simultaneously |
| **Parallel Thinking** | N independent rollouts + generative aggregation at inference | Especially powerful on small models — N=10 is cheap |
| **Iterative Self-Improvement** | Train, update reference, regenerate data, train again | 2–3 iterations, ~14 hours each on 1 GPU |

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
