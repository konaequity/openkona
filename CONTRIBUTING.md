# Contributing to KONASH

Thanks for your interest in contributing! This guide will help you get set up.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/konaequity/konash.git
cd konash

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,search]"

# For local training development
pip install -e ".[dev,train]"

# For everything
pip install -e ".[all,dev]"
```

## API Keys

Run setup to configure your keys:

```bash
konash setup
```

This stores keys in `~/.konash/config.json`. You'll need:
- **Together AI** — LLM inference ([api.together.xyz](https://api.together.xyz))
- **HuggingFace** — Embeddings and pre-built indexes ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
- **OpenAI** *(optional)* — Judge model for eval scoring ([platform.openai.com](https://platform.openai.com))

## Project Structure

```
konash/
  api.py              # High-level Agent class
  benchmarks.py       # Dataset and benchmark registry
  cli.py              # CLI (konash train, konash ask, konash eval, konash setup)
  corpus.py           # Corpus loading, embedding, vector search
  download.py         # Dataset and pre-built index downloads
  eval/
    harness.py        # Shared eval harness for all benchmarks
    nuggets.py        # Nugget-based scoring (KARL Appendix D.1)
  harness/
    environment.py    # Agent-tool interaction loop
  inference/
    parallel.py       # Parallel Thinking engine
    value_search.py   # Value-Guided Search (VGS) engine
  plugins/
    control.py        # Step budget, compression plugins
  retrieval/
    vector_search.py  # FAISS vector search, embedding models
  synthesis/
    qa.py             # Agentic QA synthesizer
    rollouts.py       # Rollout generator
    pipeline.py       # Orchestrates synthesis + rollouts + filtering
  training/
    oapl.py           # OAPL trainer (squared-advantage loss)
    unsloth_engine.py # Unsloth-based training engine

tools/
  server.py           # Unified dev server (all tools below)
  arena/              # Side-by-side model comparison
  eval/               # Eval trace viewer (karlbench)
  trace_viewer/       # Rollout trace viewer + training monitor
  shared/             # Shared components (topbar)

scripts/
  shadeform_eval_guide.md   # Guide for running evals on Shadeform GPUs
```

## Running Tests

```bash
pytest
```

## Running the Eval Tools

```bash
python tools/server.py
# Open http://localhost:5050/eval/ for eval traces
# Open http://localhost:5050/arena/ for model comparison
```

## Running Evals

```bash
konash eval financebench              # FinanceBench (150 questions)
konash eval qampari --limit 20        # QAMPARI (first 20 questions)
konash eval freshstack                # FreshStack (LangChain domain)
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests if applicable
4. Run `pytest` to make sure nothing is broken
5. Open a PR against `main`

## Style

- Python 3.11+
- Keep dependencies minimal — core package should stay lightweight
- Prefer simple, direct code over abstractions
- Match the style of surrounding code

## Questions?

Open a [GitHub Discussion](https://github.com/konaequity/konash/discussions) or file an [issue](https://github.com/konaequity/konash/issues).
