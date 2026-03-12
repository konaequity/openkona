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
- **Google AI** *(optional)* — Gemini embeddings ([aistudio.google.com](https://aistudio.google.com))

## Project Structure

```
konash/
  api.py              # High-level Agent class
  cli.py              # CLI (konash train, konash ask, konash setup)
  corpus.py           # Corpus loading, embedding, vector search
  download.py         # Dataset and pre-built index downloads
  inference/
    value_search.py   # Value-Guided Search (VGS) engine
  synthesis/
    qa.py             # Agentic QA synthesizer
    rollouts.py       # Rollout generator
    pipeline.py       # Orchestrates synthesis + rollouts + filtering
  training/
    oapl.py           # OAPL trainer (squared-advantage loss)
    unsloth_engine.py # Unsloth-based training engine
```

## Running Tests

```bash
pytest
```

## Making Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests if applicable
4. Run `pytest` to make sure nothing is broken
5. Open a PR against `main`

## What to Work On

Check [ROADMAP.md](ROADMAP.md) for prioritized tasks. Issues labeled `good first issue` are a good starting point.

## Style

- Python 3.11+
- Keep dependencies minimal — core package should stay lightweight
- Prefer simple, direct code over abstractions
- Match the style of surrounding code

## Questions?

Open a [GitHub Discussion](https://github.com/konaequity/konash/discussions) or file an [issue](https://github.com/konaequity/konash/issues).
