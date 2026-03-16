# KONASH Roadmap

Bottlenecks and friction points standing between KONASH and open-source traction.

---

## P0 — Prove it works

- [ ] Run a full train → eval loop that shows measurable improvement over the base model
- [ ] Build a benchmark table: base model accuracy vs KONASH-trained accuracy on at least one dataset
- [ ] Fix the 100% pass-rate problem at small scale — synthesized questions are too easy, yielding 0 training data on quick test runs
- [ ] Validate OAPL loss is converging and the policy is actually improving during training
- [ ] Run at least one KARL-scale iteration (1,735 synthesis calls) to prove the pipeline holds at scale

## P1 — Working demo

- [ ] Ship the Trivia Night Colab notebook — must run end-to-end with one click
- [ ] Time-to-first-result under 10 minutes on Colab free tier (T4 GPU)
- [ ] Add a second inference provider (that has glm 4.5 air) so the demo doesn't depend on a single API
- [ ] Remove Together AI as a single point of failure — got Cloudflare-blocked during normal usage

## P2 — Testing and CI

- [x] Add a basic test suite (unit tests for synthesis, rollouts, OAPL loss, VGS)
- [x] Set up GitHub Actions for CI on every PR
- [ ] Test the parallelized synthesis (api.py) and VGS (value_search.py) changes that are currently unvalidated
- [ ] Integration test: full pipeline at quick-test scale runs without errors

## P3 — Embedding performance

- [ ] KARL uses in-process vector search at 500 QPS; KONASH uses HF Inference API at ~1.5s/query (750x slower)
- [ ] Evaluate smaller embedding models that can run locally (e5-small, MiniLM)
- [ ] Add local FAISS-only path that doesn't require any API for embeddings
- [ ] Pre-build indexes for all supported datasets so users skip the embedding step entirely

## P4 — Community and visibility

- [ ] Write a blog post or Twitter thread explaining the KARL approach and why KONASH exists
- [x] Enable GitHub Discussions
- [ ] Set up Discord or Slack
- [ ] Record a demo video / terminal GIF for the README
- [ ] Add a terminal screenshot to the README (placeholder exists)

## P5 — Local training path

- [ ] Validate the Unsloth engine end-to-end: train → save adapter → load → eval → show improvement
- [ ] Add vLLM local inference as an alternative to Together AI for rollouts and synthesis
- [ ] Test on common hardware: T4 (Colab), A100 (cloud), 4090 (consumer)

## P6 — Polish

- [ ] Add remaining notebooks: 20 Questions, GeoGuessr
- [ ] API reference documentation
- [x] Contributing guide with development setup
- [x] Add topics/tags to the GitHub repo for discoverability
- [x] Publish benchmark results in the README (KARLBench table added)

---

## Open Questions

Strategic decisions that need answers before we can commit to architecture.

### Training infrastructure
- [ ] **Synthesis: API calls or GPU?** Keep running synthesis via Together AI API calls (cheap, no GPU), or fire up a GPU for faster local synthesis? API is simpler for users but adds latency and provider dependency.
- [ ] **Keep using Shadeform?** It works for GPU provisioning but adds a dependency and API key. Evaluate alternatives (Modal, RunPod, Lambda) or provide a "bring your own GPU" path that's equally smooth.
- [ ] **LoRA vs full fine-tuning?** KARL does full parameter training. We do LoRA. LoRA is 10-20x cheaper but may only capture 50-70% of gains on MoE models (router weights frozen, expert specialization can't shift). Offer both? Full fine-tune only? Need to benchmark the actual gap.

### Deployment and serving
- [ ] **Where does the trained model live for fast inference?** Currently `konash ask` hits Together's base GLM 4.5 Air — the trained LoRA adapter isn't applied. Options: merge LoRA + upload to Together as a custom model, self-hosted vLLM with adapter hot-swapping, or a serverless endpoint (Modal/Replicate). Need a path where `konash train` → `konash ask` just works without the user managing serving infrastructure.
- [ ] **Cold start problem.** Users spin up a GPU, wait for 110GB model download, train for 10 minutes, tear it down. Next time they train, same download wait. Persistent storage across sessions? Pre-baked VM images with weights? Skip user GPUs entirely by pushing training to a hosted API (Together fine-tuning)?

### Does the core pipeline actually work?
- [ ] **Does OAPL produce real improvements?** We haven't validated that our OAPL implementation moves the needle. Need a controlled experiment: base model vs 1 iteration vs 2 iterations on a single benchmark.
- [ ] **Does VGS help?** Value-Guided Search is implemented but untested end-to-end. KARL reports 70.4 on BrowseComp-Plus with VGS — does our version improve over parallel thinking?
- [ ] **Does the value model actually work?** The value model predicts future success probability for partial rollouts. Is ours well-calibrated? Does it guide search toward better answers or just add latency?

### Embeddings
- [ ] **GTE vs Qwen embedding model?** KARL uses Qwen3-Embedding-8B for BrowseComp-Plus and GTE-large for PMBench. Qwen3-8B produces better retrieval but is slow to run locally on a user's own corpus. GTE-large is much faster. Is the retrieval quality difference worth the wait, or should we default to GTE (or even smaller models like MiniLM) for local corpora and reserve Qwen3-8B for pre-built indexes?

### Evaluation
- [ ] **What evals matter?** KARLBench has 6 tasks but we can't run all of them easily. Pick 1-2 that are (a) easy to set up, (b) show clear improvement, and (c) resonate with users. FinanceBench (SEC filings) and BrowseComp-Plus (web search) are the strongest candidates.

### Training methodology
- [ ] **Multi-task vs single-corpus?** KARL gets OOD generalization by training on BrowseComp-Plus + TREC-Biogen simultaneously — combining losses from structurally different search tasks. KONASH appears single-corpus. Is multi-corpus training supported or planned? Without it, trained agents may only improve on the specific corpus they trained on.
- [ ] **Compression training.** KARL trains compression end-to-end with RL — the model learns *what* to keep when context overflows, optimized by outcome rewards. Does KONASH's compression plugin do the same, or is it a simpler summarization heuristic? If it's just "summarize the history," we're leaving significant performance on the table (KARL's ablation shows removing compression drops BrowseComp-Plus from 0.570 → 0.389).

### Demos and notebooks
- [ ] **What are good notebooks?** Trivia Night is planned but may not showcase the value well enough. Consider: a FinanceBench notebook (business users), a "search your own docs" notebook (developers), or a head-to-head vs vanilla RAG notebook (convincing skeptics).
- [ ] **How do we show value in 10 seconds without payment?** The current flow requires API keys and training time. Ideas: pre-trained demo agent hosted somewhere, interactive web playground, terminal recording, or a read-only Colab that loads a pre-trained checkpoint and runs queries live.
