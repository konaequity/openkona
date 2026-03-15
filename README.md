<div align="center">

<img src="assets/logo.png" alt="KONASH" width="200">

# KONASH

**Knowledge-grounded Off-policy Networks for Agentic System Harnesses**

<p>
Point it at any document corpus — it trains a model that learns <i>how to search</i>, not just facts.
</p>

[![PyPI](https://img.shields.io/pypi/v/konash)](https://pypi.org/project/konash/)
[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

KONASH trains knowledge agents via reinforcement learning that match or exceed frontier models on grounded reasoning tasks — at a fraction of the cost.

---

## Key Benefits

- 💰 **100x cheaper training** — Small training clusters. ~$100 per iteration instead of ~$100K–500K.
- 🎯 **Higher quality** — RL-trained agents search more efficiently, retrieve more diversely, and reason more accurately than frontier models. The gains are algorithmic, not scale-dependent.
- 🔁 **Consistent results** — Parallel thinking (N=10–20 rollouts + aggregation) turns probabilistic search into near-deterministic accuracy. Cheap rollouts on a small model mean you can afford this on every query.
- 🔓 **Zero lock-in** — Your model, your weights, your infrastructure. Deploy anywhere with vLLM and LoRA hot-swapping.

## 🚀 Quickstart

```bash
pip install konash
konash setup    # walks you through API keys
konash train    # pick a corpus, model, and scale — hit go
```

Setup takes 2 minutes. Training scales from ~1 hour (Quick) to several hours (Exhaustive).

### What happens under the hood

1. **Corpus ingestion** — Embeds and indexes your documents for vector search (pre-built indexes ship with supported datasets)
2. **QA synthesis** — An agentic loop explores the corpus via search and generates grounded, multi-constraint question-answer pairs
3. **Rollout generation** — The model attempts to answer each question via multi-step search, generating full agent trajectories
4. **Pass-rate filtering** — Keeps questions at the learning frontier (not too easy, not too hard)
5. **OAPL training** — Off-policy RL with squared advantage loss trains the model on successful search strategies
6. **Value-Guided Search** — A learned value model scores partial rollouts to guide test-time tree search

### Ask questions

After training, query your agent:

```bash
konash ask "Which Nobel physicist was born in the same city as the author of The Trial?"
```

Or in Python:

```python
import konash

agent = konash.Agent(
    base_model="zai-org/GLM-4.5-Air-FP8",
    corpus="./my_documents",
)
agent.train(iterations=1)
answer = agent.solve("Your question here", parallel_rollouts=3)
```

---

## Features

| Feature | Description |
|---|---|
| **Agentic QA Synthesis** | Multi-turn agent loop explores your corpus via search, generates grounded multi-constraint question-answer pairs |
| **OAPL Training** | Off-policy RL with squared advantage loss trains on successful search trajectories |
| **Value-Guided Search** | Learned value model scores partial rollouts, parallel BFS tree search at inference time |
| **Pass-Rate Filtering** | Keeps questions at the learning frontier — not too easy, not too hard (0.1–0.9 pass rate) |
| **Parallel Rollouts** | N=10–20 independent rollouts + aggregation for consistent answers |
| **Pre-built Indexes** | Ships with Qwen3-Embedding-8B indexes for supported datasets — no embedding step needed |
| **Any Corpus** | Point at a local folder of documents — KONASH builds the index on first run |

---

## 🔍 How It Works

Standard retrieval systems use a frozen model with a single retrieve-then-read pass. KONASH trains the model's **search policy** through reinforcement learning:

- The model learns **what to search for** (query generation)
- The model learns **when to search again** (multi-step retrieval)
- The model learns **how to reason** over retrieved evidence (cross-document synthesis)
- The trained model **generalizes to new corpora** it hasn't seen

<p align="center">
  <img src="assets/diagram.png" alt="KONASH training pipeline" width="700">
</p>

Each iteration: synthesize → rollout → filter → train → repeat with improved model.

---

## Evaluation

### FinanceBench (150 questions, SEC filings)

GLM 4.5 Air on FinanceBench — no training, base model only:

| Mode | Accuracy | Avg Score | Avg Latency |
|------|----------|-----------|-------------|
| Single rollout | 48% | 0.487 | 7.7s |
| Parallel thinking (N=3) | 51% | 0.520 | 30.8s |

Scored with LLM-based nugget evaluation (KARL paper, Appendix D.1). Parallel thinking runs 3 independent rollouts and aggregates answers, improving accuracy by +3%.

The KARL paper reports **76%** on FinanceBench after RL training (2 iterations, 12K synthesized QA pairs). KONASH implements this training pipeline — the gap between 48% (base) and 76% (trained) is what OAPL training closes.

<details>
<summary>Reproduce these results</summary>

```bash
pip install konash
konash setup
python scripts/eval_financebench.py              # single + parallel
python scripts/eval_financebench.py --parallel 5  # try N=5
python scripts/eval_financebench.py --train       # train + eval
```

Results are saved to `eval_results/financebench_eval.json`. Traces are written to `tools/trace_viewer/data/` for visualization.
</details>

---

## Requirements

### API Keys (set up via `konash setup`)

| Service | Purpose | Cost |
|---------|---------|------|
| **Together AI** | LLM inference (synthesis, rollouts, solving) | Pay-as-you-go (~$5 for a quick training run) |
| **HuggingFace** | Pre-built embedding indexes, model hosting | Free |
| **Google AI** *(optional)* | Corpus embeddings via Gemini (when no pre-built index) | Free tier available |

### Python

- Python >= 3.11
- Core dependencies: `numpy`, `rich`, `together`, `google-genai`
- Optional: `faiss-cpu` (fast vector search), `torch` + `transformers` + `peft` (local training)

---

## Cloud Training

KONASH automatically provisions cloud GPUs when training needs gradient updates. Synthesis and rollout generation run locally via API — only the OAPL step (which takes minutes, not hours) uses a GPU.

```bash
pip install konash

# Configure a GPU provider (RunPod, Lambda, AWS, GCP, Azure, etc.):
pip install runpod && runpod config  # or any SkyPilot-supported provider
sky check                            # verify it's enabled

# Then just train normally — GPU provisioning is automatic:
konash train
```

When `konash train` reaches the OAPL gradient step, it automatically finds the cheapest available H100 across all configured providers via [SkyPilot](https://skypilot.co), runs training, downloads the adapter, and tears down the GPU.

| GPU | Provider | OAPL step cost |
|-----|----------|----------------|
| H100 SXM | RunPod | ~$0.50 (minutes, not hours) |
| H100 PCIe | RunPod | ~$0.40 |
| H100 | Lambda | ~$0.46 |

---

## Supported Datasets

Datasets download automatically when selected in `konash train`:

| Dataset | Domain | Docs | Pre-built Index |
|---------|--------|------|-----------------|
| **BrowseComp-Plus** | Web documents (67K articles) | 67,707 | Qwen3-Embedding-8B via Tevatron |
| **FinanceBench** | SEC filings, financial reports | ~150 | — |
| **QAMPARI** | Encyclopedic entity search | 250K+ chunks | — |
| **FreshStack** | Technical documentation | Varies | — |
| **Local folder** | Your own documents | Any | Built on first run |

### Supported file formats (local folders)

`.txt` `.md` `.rst` `.csv` `.log` `.json` `.html` `.htm` `.py` `.js` `.ts` `.java` `.go` `.rs` `.c` `.cpp` `.h`

---

## Supported Models

Any model available on [Together AI](https://api.together.xyz/models):

| Model | Type | Notes |
|-------|------|-------|
| **GLM 4.5 Air** | Frontier MoE | Default — best for KARL, fast + cheap |
| **Qwen3 80B-A3B** | MoE | Good value |
| **Llama 3.3 70B Turbo** | Dense | Strong general-purpose |
| **DeepSeek R1** | MoE | Reasoning-focused |
| **Mixtral 8x22B** | MoE | Balanced |
| Custom | Any | Enter any Together AI model ID |

---

## 📓 Notebooks

| Agent Task | Notebook | Description |
|---|---|---|
| **Trivia Night** | *Coming soon* | Train a model to answer multi-constraint trivia by searching Wikipedia |
| **20 Questions** | *Coming soon* | Train an agent to identify entities through iterative yes/no questioning |
| **GeoGuessr** | *Coming soon* | Train an agent to geolocate images by searching geographic knowledge bases |

---

## Contributing

Contributions are welcome! Please open an issue or PR on [GitHub](https://github.com/konaequity/konash/issues).

---

## Citation

```bibtex
@misc{konaequity2026konash,
  author = {Kona Equity},
  title = {KONASH: Knowledge-grounded Off-policy Networks for Agentic System Harnesses},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/konaequity/konash}}
}
```

---

## 🙏 Credits

KONASH builds directly on:

- [KARL: Knowledge Agents via Reinforcement Learning](https://www.databricks.com/sites/default/files/2026-03/karl.pdf) — Databricks, 2026. The architecture, training pipeline, and evaluation methodology that KONASH implements.
- [OAPL](https://arxiv.org/abs/2602.19362) — Ritter et al., 2026 (the RL algorithm)
- [Tevatron](https://huggingface.co/Tevatron) — Pre-built BrowseComp-Plus embedding indexes
- [Unsloth](https://github.com/unslothai/unsloth) — Parameter-efficient training
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search

## License

[Apache 2.0](LICENSE)
