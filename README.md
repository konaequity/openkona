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

- **100x cheaper training** — Single GPU replaces multi-node clusters. ~$100–500 per iteration instead of ~$10K–50K.
- **Higher quality** — RL-trained agents search more efficiently, retrieve more diversely, and reason more accurately than their base models. The gains are algorithmic, not scale-dependent.
- **Consistent results** — Parallel thinking (N=10–20 rollouts + aggregation) turns probabilistic search into near-deterministic accuracy. Cheap rollouts on a small model mean you can afford this on every query.
- **Zero lock-in** — Your model, your weights, your infrastructure. Deploy anywhere with vLLM and LoRA hot-swapping.

```python
# Before: Static RAG — one query, hope for the best
docs = retriever.search(query, top_k=10)
answer = llm.generate(f"Answer based on: {docs}\n\n{query}")

# After: KONASH — RL-trained agent that searches iteratively
agent = konash.Agent("./checkpoints/iter2", corpus="./my_docs")
answer = agent.solve(query, parallel_rollouts=10)
```

[Learn more about KONASH](https://kona.sh)

---

## KONASH Overview

KONASH is an open-source RL framework that improves agent reliability by training knowledge agents to search, retrieve, compress, and reason over evidence — all on a single GPU using off-policy RL. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://kona.sh).

### Notebooks

| Agent Task | Example Notebook | Description | Comparative Performance |
|---|---|---|---|
| **Trivia Night** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konaequity/konash/blob/main/notebooks/trivia_night.ipynb) | Qwen 3 4B learns to answer multi-constraint trivia by searching Wikipedia | [Link coming soon] |
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

## Installation

KONASH agents can be trained from any client machine that runs Python. To add to an existing project, run this command:

```
pip install konash
```

---

## Training Loop Overview

KONASH uses **large-batch iterative off-policy RL** — unlike online RL frameworks, all data is generated upfront and training happens in a single offline pass. Each iteration improves the model, which then generates better data for the next iteration.

1. **Data Synthesis**

   1. KONASH generates training questions from your corpus using an agentic synthesis pipeline — the model explores documents via vector search and proposes grounded QA pairs.
   2. A deduplication step ensures no overlap with your evaluation set.
   3. On later iterations, the improved model synthesizes its own curriculum — harder, more diverse questions.

2. **Rollout Generation**

   1. The model (or latest checkpoint) generates multiple rollouts per question, interacting with vector search and compression tools.
   2. Each rollout is a full multi-step agent trajectory: search queries, retrieved documents, context compression, and a final answer.
   3. Rewards are computed automatically from answer correctness against ground truth.
   4. Pass-rate filtering keeps questions at the learning frontier — not too easy, not too hard.

3. **Training**

   1. The full set of trajectories becomes a large offline dataset. Training runs in a single batch — no interleaving with inference.
   2. The server trains your model using OAPL with QLoRA. Long trajectories are segmented at compression boundaries, and tool outputs are masked from log-prob computation.
   3. The newly trained LoRA is saved and becomes the starting point for the next iteration.

4. **Iterate**

   1. The trained checkpoint becomes the new reference policy.
   2. All rollouts are **regenerated from scratch** with the improved model — this is what makes each iteration progressively better.
   3. Training runs again on the fresh data. 2–3 iterations yields the best results.

## Supported Models

KONASH should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If any model isn't working for you, please let us know on [Discord](#) or open an issue on [GitHub](https://github.com/konaequity/konash/issues)!

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
  howpublished = {\url{https://github.com/konaequity/konash}}
}
```

---

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

---

## Credits

KONASH builds directly on the research and open-source work of:

- [KARL: Knowledge Agent trained via Reinforcement Learning](https://www.databricks.com/sites/default/files/2026-03/karl.pdf) — Databricks, 2026. The architecture, training pipeline, and evaluation methodology that KONASH implements. KARL introduced agentic data synthesis, off-policy RL for knowledge agents, end-to-end compression training, and the KARLBench evaluation suite.
- [OAPL](https://arxiv.org/abs/2602.19362) — Ritter et al., 2026 (the RL algorithm)
- [Unsloth](https://github.com/unslothai/unsloth) — Parameter-efficient training backend
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput inference engine
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
