<div align="center">

# KONASH

**Knowledge-grounded Off-policy Networks for Agentic System Harnesses**

<p>
Point it at any document corpus — it trains a model that learns <i>how to search</i>, not just facts.
</p>

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](CONTRIBUTING.md)
[![PyPI](https://img.shields.io/pypi/v/konash)](https://pypi.org/project/konash/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

KONASH trains knowledge agents via reinforcement learning that match or exceed frontier models on grounded reasoning tasks — at a fraction of the cost. 

---

## Quickstart

```bash
pip install konash
konash setup    # walks you through API keys
konash train    # pick a corpus, model, and scale — hit go
```

That's it. Setup takes 2 minutes. Training takes 5 minutes (quick test) to several hours (KARL scale).

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

## Requirements

### API Keys (set up via `konash setup`)

| Service | Purpose | Cost |
|---------|---------|------|
| **Together AI** | LLM inference (synthesis, rollouts, solving) | Pay-as-you-go (~$5 for a small run) |
| **HuggingFace** | Pre-built embedding indexes, model hosting | Free |
| **Google AI** *(optional)* | Corpus embeddings via Gemini (when no pre-built index) | Free tier available |

### Python

- Python >= 3.11
- Core dependencies: `numpy`, `rich`, `together`, `google-genai`
- Optional: `faiss-cpu` (fast vector search), `torch` + `transformers` + `peft` (local training)

### Hardware

- **No GPU required** for API-based training (Together AI handles inference, OAPL runs on their cluster)
- **Single GPU** (T4+) for local QLoRA training via Unsloth

---

## Supported Datasets

Datasets download automatically when selected in `konash train`:

| Dataset | Domain | Docs | Size | Pre-built Index |
|---------|--------|------|------|-----------------|
| **BrowseComp-Plus** | Web documents (67K articles) | 67,707 | 2.5 GB | Qwen3-Embedding-8B via Tevatron |
| **FinanceBench** | SEC filings, financial reports | ~150 | Varies | — |
| **QAMPARI** | Encyclopedic entity search | 250K+ chunks | Varies | — |
| **FreshStack** | Technical documentation | Varies | Varies | — |
| **Local folder** | Your own documents | Any | Any | Built on first run |

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

## Notebooks

| Agent Task | Notebook | Description |
|---|---|---|
| **Trivia Night** | *Coming soon* | Train a model to answer multi-constraint trivia by searching Wikipedia |
| **20 Questions** | *Coming soon* | Train a model to identify a mystery entity in 20 yes/no searches |
| **GeoGuessr** | *Coming soon* | Train a model to pinpoint locations from landmark and terrain descriptions |

---

## Agent Harness + Reinforcement Learning

KONASH wraps each model in an **agent harness** — an environment where the model interacts with tools (vector search, context compression) across multi-step episodes. The harness records full trajectories: what the model searched for, what it retrieved, how it reasoned, and whether it got the right answer.

These trajectories become training data for **off-policy RL (OAPL)**. The model learns from both successes and failures:

- **What to search for** — query generation improves with reward signal
- **When to search again** — the model learns multi-step retrieval strategies
- **How to reason** over retrieved evidence — cross-document synthesis emerges from training
- **Generalization** — the trained search policy transfers to new corpora the model hasn't seen

Test-time compute scaling (parallel rollouts or Value-Guided Search) further amplifies the trained model's capabilities.

---

## Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Agentic    │     │   Rollout    │     │  Pass-rate   │     │     OAPL     │
│  QA Synthesis│────▶│  Generation  │────▶│  Filtering   │────▶│   Training   │
│  (parallel)  │     │  (parallel)  │     │  [0.1, 0.9]  │     │  (off-policy)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                              ┌───────▼───────┐
                                                              │ Value Model   │
                                                              │   Training    │
                                                              └───────┬───────┘
                                                                      │
                                                              ┌───────▼───────┐
                                                              │  Value-Guided │
                                                              │    Search     │
                                                              │   (VGS)       │
                                                              └───────────────┘
```

Each iteration: synthesize → rollout → filter → train → repeat with improved model.

---

## Contributing

Contributions are welcome! Please open an issue or PR on [GitHub](https://github.com/konaequity/konash/issues).

---

## Citation

```bibtex
@misc{konaequity2026konash,
  author = {Kona Equity},
  title = {KONASH: Knowledge Agents via Reinforcement Learning},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/konaequity/konash}}
}
```

---

## Credits

KONASH builds directly on:

- [KARL: Knowledge Agents via Reinforcement Learning](https://www.databricks.com/sites/default/files/2026-03/karl.pdf) — Databricks, 2026. The architecture, training pipeline, and evaluation methodology that KONASH implements.
- [OAPL](https://arxiv.org/abs/2602.19362) — Ritter et al., 2026 (the RL algorithm)
- [Tevatron](https://huggingface.co/Tevatron) — Pre-built BrowseComp-Plus embedding indexes
- [Unsloth](https://github.com/unslothai/unsloth) — Parameter-efficient training
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search

## License

[Apache 2.0](LICENSE)
