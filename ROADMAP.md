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
- [ ] Add a second inference provider (Groq free tier, local vLLM) so the demo doesn't depend on a paid API
- [ ] Remove Together AI as a single point of failure — got Cloudflare-blocked during normal usage

## P2 — Testing and CI

- [ ] Add a basic test suite (unit tests for synthesis, rollouts, OAPL loss, VGS)
- [ ] Set up GitHub Actions for CI on every PR
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
- [ ] Document the single-GPU training path (what hardware, how long, what to expect)
- [ ] Add vLLM local inference as an alternative to Together AI for rollouts and synthesis
- [ ] Test on common hardware: T4 (Colab), A100 (cloud), 4090 (consumer)

## P6 — Polish

- [ ] Add remaining notebooks: 20 Questions, GeoGuessr
- [ ] API reference documentation
- [x] Contributing guide with development setup
- [x] Add topics/tags to the GitHub repo for discoverability
- [ ] Publish benchmark results in the README once they exist
