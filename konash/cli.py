"""KONASH command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

# Together AI defaults (matching the KARL paper with GLM 4.5 Air)
TOGETHER_API_BASE = "https://api.together.xyz/v1"
DEFAULT_MODEL = "zai-org/GLM-4.5-Air-FP8"
CONFIG_DIR = os.path.expanduser("~/.konash")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def _save_config(config: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def _get_key(name: str, env_vars: list[str], config_key: str) -> str | None:
    """Resolve a key from env vars or saved config."""
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val
    config = _load_config()
    return config.get(config_key)


def _get_together_key() -> str | None:
    return _get_key("Together AI", ["TOGETHER_API_KEY"], "together_api_key")


def _get_hf_token() -> str | None:
    return _get_key("HuggingFace", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"], "hf_token")


# ---------------------------------------------------------------------------
# konash setup
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> None:
    """Interactive setup wizard — get API keys, validate, save."""
    print()
    print("  Welcome to KONASH")
    print("  " + "=" * 40)
    print()
    print("  KONASH trains knowledge agents that search, retrieve,")
    print("  and reason over your documents using reinforcement learning.")
    print()
    print("  You'll need two free API keys:")
    print("    1. Together AI  — runs the AI model")
    print("    2. HuggingFace  — stores your trained model")
    print()

    config = _load_config()

    # --- Together AI ---
    together_key = _get_together_key()
    if together_key:
        masked = together_key[:8] + "..." + together_key[-4:]
        print(f"  Together AI key found: {masked}")
        resp = input("  Keep this key? [Y/n] ").strip().lower()
        if resp in ("n", "no"):
            together_key = None

    if not together_key:
        print()
        print("  Step 1: Get your Together AI API key")
        print("  " + "-" * 40)
        print("  1. Go to: https://api.together.xyz/settings/api-keys")
        print("  2. Sign up (free) or log in")
        print("  3. Click 'Create API Key'")
        print("  4. Copy the key")
        print()

        if input("  Open the page in your browser? [Y/n] ").strip().lower() != "n":
            webbrowser.open("https://api.together.xyz/settings/api-keys")

        together_key = input("\n  Paste your Together AI key: ").strip()

        if not together_key:
            print("  Skipped. You can set TOGETHER_API_KEY later.")
        else:
            # Validate
            print("  Validating...", end=" ", flush=True)
            if _validate_together_key(together_key):
                print("OK")
                config["together_api_key"] = together_key
            else:
                print("FAILED")
                print("  Key didn't work. Check it and try again.")
                return

    else:
        config["together_api_key"] = together_key

    # --- HuggingFace ---
    print()
    hf_token = _get_hf_token()
    if hf_token:
        masked = hf_token[:8] + "..." + hf_token[-4:]
        print(f"  HuggingFace token found: {masked}")
        resp = input("  Keep this token? [Y/n] ").strip().lower()
        if resp in ("n", "no"):
            hf_token = None

    if not hf_token:
        print()
        print("  Step 2: Get your HuggingFace token")
        print("  " + "-" * 40)
        print("  1. Go to: https://huggingface.co/settings/tokens")
        print("  2. Sign up (free) or log in")
        print("  3. Click 'Create new token' (select 'Write' access)")
        print("  4. Copy the token")
        print()

        if input("  Open the page in your browser? [Y/n] ").strip().lower() != "n":
            webbrowser.open("https://huggingface.co/settings/tokens")

        hf_token = input("\n  Paste your HuggingFace token: ").strip()

        if not hf_token:
            print("  Skipped. You can set HF_TOKEN later.")
            print("  (Only needed if you want to deploy your trained model.)")
        else:
            config["hf_token"] = hf_token
    else:
        config["hf_token"] = hf_token

    # Save
    _save_config(config)
    print()
    print("  " + "=" * 40)
    print(f"  Config saved to {CONFIG_FILE}")
    print()
    print("  You're ready! Next steps:")
    print()
    print("    # Train on your documents:")
    print("    konash train ./my_docs")
    print()
    print("    # Ask questions:")
    print('    konash ask --corpus ./my_docs "What is X?"')
    print()


def _validate_together_key(key: str) -> bool:
    """Make a test API call to validate the key."""
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            "https://api.together.xyz/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# konash train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    try:
        from konash.api import Agent
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    api_key = _get_together_key() or args.api_key
    if not api_key:
        print("  No API key found. Run 'konash setup' first.")
        sys.exit(1)

    print()
    print("  KONASH Training")
    print("  " + "=" * 40)
    print(f"  Corpus:     {args.corpus}")
    print(f"  Model:      {args.model}")
    print(f"  Iterations: {args.iterations}")
    print()

    agent = Agent(
        base_model=args.model,
        corpus=args.corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
        chunk_size=args.chunk_size,
    )
    agent.train(
        iterations=args.iterations,
        rollouts_per_example=args.rollouts,
        max_examples=args.max_examples,
        learning_rate=args.lr,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# konash ask
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> None:
    try:
        from konash.api import Agent
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    api_key = _get_together_key() or args.api_key
    if not api_key:
        print("  No API key found. Run 'konash setup' first.")
        sys.exit(1)

    agent = Agent(
        base_model=args.model,
        corpus=args.corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
    )
    answer = agent.solve(
        args.query,
        parallel_rollouts=args.parallel,
        top_k=args.top_k,
    )
    print()
    print(answer)
    print()


# ---------------------------------------------------------------------------
# konash search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    try:
        from konash.corpus import Corpus
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    corpus = Corpus(args.corpus, chunk_size=args.chunk_size)
    corpus.ingest()
    print(f"  Indexed {corpus.num_documents} chunks from {args.corpus}\n")

    results = corpus.search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("source", "?")
        text = r.get("text", "")[:200]
        print(f"  [{i}] (score: {score:.3f}) {source}")
        print(f"      {text}...")
        print()


# ---------------------------------------------------------------------------
# konash status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Show current configuration and key status."""
    config = _load_config()

    print()
    print("  KONASH Status")
    print("  " + "=" * 40)

    # Together key
    together_key = _get_together_key()
    if together_key:
        masked = together_key[:8] + "..." + together_key[-4:]
        print(f"  Together AI:  {masked}")
    else:
        print("  Together AI:  NOT SET")

    # HF token
    hf_token = _get_hf_token()
    if hf_token:
        masked = hf_token[:8] + "..." + hf_token[-4:]
        print(f"  HuggingFace:  {masked}")
    else:
        print("  HuggingFace:  NOT SET")

    # Config file
    print(f"  Config:       {CONFIG_FILE}")

    # Check for trained models
    project_dir = os.path.join(".konash", "default", "checkpoints")
    if os.path.exists(project_dir):
        meta_path = os.path.join(project_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  Trained:      {meta.get('base_model', '?')} "
                  f"({meta.get('iterations', 0)} iterations)")
        else:
            print("  Trained:      checkpoint dir exists, no meta")
    else:
        print("  Trained:      no")

    print()
    if not together_key:
        print("  Run 'konash setup' to configure.")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dependency_error(exc: ModuleNotFoundError) -> None:
    missing = exc.name or "a required package"
    print(f"  Error: Missing dependency '{missing}'.")
    print("  Install with: pip install konash")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="konash",
        description="KONASH — Train knowledge agents on your documents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- setup ---
    p_setup = subparsers.add_parser(
        "setup",
        help="Set up API keys (interactive wizard).",
    )
    p_setup.set_defaults(func=cmd_setup)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Train a knowledge agent on your documents.",
    )
    p_train.add_argument("corpus", help="Path to your documents folder.")
    p_train.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_train.add_argument("--project", default="default", help="Project name.")
    p_train.add_argument("--iterations", type=int, default=2, help="Training iterations.")
    p_train.add_argument("--rollouts", type=int, default=8, help="Rollouts per example.")
    p_train.add_argument("--max-examples", type=int, default=None, help="Max training examples.")
    p_train.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p_train.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words.")
    p_train.add_argument("--api-key", default=None, help="Together AI key (or run konash setup).")
    p_train.set_defaults(func=cmd_train)

    # --- ask ---
    p_ask = subparsers.add_parser(
        "ask",
        help="Ask a question using your knowledge agent.",
    )
    p_ask.add_argument("query", help="Your question.")
    p_ask.add_argument("--corpus", required=True, help="Path to documents folder.")
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_ask.add_argument("--project", default="default", help="Project name.")
    p_ask.add_argument("--parallel", type=int, default=1, help="Parallel rollouts.")
    p_ask.add_argument("--top-k", type=int, default=10, help="Documents per search.")
    p_ask.add_argument("--api-key", default=None, help="Together AI key.")
    p_ask.set_defaults(func=cmd_ask)

    # --- search ---
    p_search = subparsers.add_parser(
        "search",
        help="Search your documents directly (no AI).",
    )
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("--corpus", required=True, help="Path to documents folder.")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results.")
    p_search.add_argument("--chunk-size", type=int, default=512, help="Chunk size.")
    p_search.set_defaults(func=cmd_search)

    # --- status ---
    p_status = subparsers.add_parser(
        "status",
        help="Show current setup and key status.",
    )
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)

    if args.command is None:
        # No command — show help with a friendly message
        print()
        print("  KONASH — Train knowledge agents on your documents")
        print()
        print("  Get started:")
        print("    konash setup              Set up API keys")
        print("    konash train ./my_docs    Train on your documents")
        print('    konash ask --corpus ./my_docs "What is X?"')
        print("    konash status             Check your setup")
        print()
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
