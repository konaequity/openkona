"""KONASH command-line interface."""

from __future__ import annotations

import argparse
import os
import sys

# Zhipu AI defaults (matching the KARL paper)
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.5-air"


def _dependency_error(exc: ModuleNotFoundError) -> None:
    missing = exc.name or "a required package"
    print(f"Error: Missing runtime dependency '{missing}'.")
    print("Install the package first with:")
    print("  pip install konash")
    print()
    print("For local development from this checkout, use:")
    print("  pip install -e .")
    sys.exit(1)


def _resolve_api_config(args: argparse.Namespace) -> tuple:
    """Resolve API base URL and key from args, env vars, or defaults.

    Priority: explicit args > env vars > Zhipu defaults.
    """
    api_base = args.api_base or os.environ.get("KONASH_API_BASE") or os.environ.get("ZHIPU_API_BASE") or ZHIPU_API_BASE
    api_key = args.api_key or os.environ.get("KONASH_API_KEY") or os.environ.get("ZHIPU_API_KEY")
    return api_base, api_key


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from konash.api import Agent
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    api_base, api_key = _resolve_api_config(args)
    if api_key is None:
        print("Error: No API key found. Set one of:")
        print("  export ZHIPU_API_KEY=your_key_here")
        print("  export KONASH_API_KEY=your_key_here")
        print("  or pass --api-key")
        print()
        print("Get a free key at https://z.ai")
        sys.exit(1)

    agent = Agent(
        base_model=args.model,
        corpus=args.corpus,
        project=args.project,
        api_base=api_base,
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


def cmd_solve(args: argparse.Namespace) -> None:
    try:
        from konash.api import Agent
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    api_base, api_key = _resolve_api_config(args)
    if api_key is None:
        print("Error: No API key found. Set ZHIPU_API_KEY or KONASH_API_KEY.")
        sys.exit(1)

    agent = Agent(
        base_model=args.model,
        corpus=args.corpus,
        project=args.project,
        api_base=api_base,
        api_key=api_key,
    )
    answer = agent.solve(
        args.query,
        parallel_rollouts=args.parallel,
        top_k=args.top_k,
    )
    print(answer)


def cmd_search(args: argparse.Namespace) -> None:
    try:
        from konash.corpus import Corpus
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    corpus = Corpus(args.corpus, chunk_size=args.chunk_size)
    corpus.ingest()
    print(f"Indexed {corpus.num_documents} chunks from {args.corpus}\n")

    results = corpus.search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("source", "?")
        text = r.get("text", "")[:200]
        print(f"[{i}] (score: {score:.3f}) {source}")
        print(f"    {text}...")
        print()


def cmd_ingest(args: argparse.Namespace) -> None:
    try:
        from konash.corpus import Corpus
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    corpus = Corpus(args.corpus, chunk_size=args.chunk_size)
    corpus.ingest()
    print(f"Ingested {corpus.num_documents} chunks from {args.corpus}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="konash",
        description="KONASH — Train and run knowledge agents on a single GPU.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Train a knowledge agent on a corpus.",
    )
    p_train.add_argument("--corpus", required=True, help="Path to document directory.")
    p_train.add_argument("--model", default=DEFAULT_MODEL, help="Model ID (default: glm-4.5-air via Zhipu API).")
    p_train.add_argument("--project", default="default", help="Project name.")
    p_train.add_argument("--iterations", type=int, default=2, help="Training iterations.")
    p_train.add_argument("--rollouts", type=int, default=8, help="Rollouts per example.")
    p_train.add_argument("--max-examples", type=int, default=None, help="Cap on training examples.")
    p_train.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p_train.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words.")
    p_train.add_argument("--api-base", default=None, help="LLM API base URL (default: Zhipu AI).")
    p_train.add_argument("--api-key", default=None, help="API key (or set ZHIPU_API_KEY env var).")
    p_train.set_defaults(func=cmd_train)

    # --- solve ---
    p_solve = subparsers.add_parser(
        "solve",
        help="Ask a question using a trained knowledge agent.",
    )
    p_solve.add_argument("query", help="Question to answer.")
    p_solve.add_argument("--corpus", required=True, help="Path to document directory.")
    p_solve.add_argument("--model", default=DEFAULT_MODEL, help="Model ID (default: glm-4.5-air via Zhipu API).")
    p_solve.add_argument("--project", default="default", help="Project name.")
    p_solve.add_argument("--parallel", type=int, default=1, help="Parallel rollouts (1 = single, 10-20 = parallel thinking).")
    p_solve.add_argument("--top-k", type=int, default=10, help="Documents to retrieve per search.")
    p_solve.add_argument("--api-base", default=None, help="LLM API base URL (default: Zhipu AI).")
    p_solve.add_argument("--api-key", default=None, help="API key (or set ZHIPU_API_KEY env var).")
    p_solve.set_defaults(func=cmd_solve)

    # --- search ---
    p_search = subparsers.add_parser(
        "search",
        help="Search a corpus directly (no LLM, for debugging retrieval).",
    )
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("--corpus", required=True, help="Path to document directory.")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results.")
    p_search.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words.")
    p_search.set_defaults(func=cmd_search)

    # --- ingest ---
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest and index a corpus (validates documents are readable).",
    )
    p_ingest.add_argument("--corpus", required=True, help="Path to document directory.")
    p_ingest.add_argument("--chunk-size", type=int, default=512, help="Chunk size in words.")
    p_ingest.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
