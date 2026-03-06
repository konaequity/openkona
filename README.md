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

## Installation

KONASH agents can be trained from any client machine that runs Python. To add to an existing project, run this command:

```
pip install konash
```

---

## Training Loop Overview

KONASH's functionality is divided into a **client** and a **server**. The client is responsible for interfacing between KONASH and your codebase — you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU, abstracting away inference and training complexity. An outline of the training loop is shown below:

1. **Inference**

   1. Your code uses the KONASH client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
   2. Completion requests are routed to the KONASH server, which runs the model's latest LoRA in vLLM.
   3. As the agent executes, each `system`, `user`, and `assistant` message — along with tool calls and retrieved documents — is stored in a Trajectory.
   4. When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

2. **Training**
   1. When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
   2. The server trains your model using OAPL, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
   3. The server saves the newly trained LoRA to a local directory and loads it into vLLM.
   4. Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

## Supported Models

KONASH should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If any model isn't working for you, please let us know on [Discord](#) or open an issue on [GitHub](https://github.com/konaequity/openkona/issues)!

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
