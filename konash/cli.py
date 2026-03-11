"""KONASH command-line interface.

Polished with ``rich`` — panels, spinners, progress, styled output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from konash.auth import (
    TOGETHER_KEYS_PAGE,
    HF_TOKENS_PAGE,
    detect_hf_token,
    hf_device_flow,
    validate_hf_token,
    validate_together_key,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

console = Console()

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


def _get_key(env_vars: list[str], config_key: str) -> str | None:
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            return val
    return _load_config().get(config_key)


def _get_together_key() -> str | None:
    return _get_key(["TOGETHER_API_KEY"], "together_api_key")


def _get_hf_token() -> str | None:
    token = _get_key(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"], "hf_token")
    if token:
        return token
    return detect_hf_token()


def _mask(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:8] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# konash  (no args)
# ---------------------------------------------------------------------------

def cmd_default() -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]KONASH[/]  v" + _get_version() + "\n\n"
        "Train knowledge agents that search, retrieve, and reason\n"
        "over your documents using reinforcement learning.",
        border_style="cyan",
        padding=(1, 4),
    ))

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold cyan", min_width=36)
    grid.add_column(style="dim")
    grid.add_row("konash setup", "Set up API keys")
    grid.add_row("konash download browsecomp-plus", "Download benchmark corpus")
    grid.add_row("konash train", "Train (interactive wizard)")
    grid.add_row('konash ask --corpus ./docs "Q"', "Ask a question")
    grid.add_row("konash search --corpus ./docs Q", "Search documents")
    grid.add_row("konash status", "Check your setup")

    console.print(Panel(
        grid, title="[bold]Commands", border_style="dim", padding=(1, 2),
    ))
    console.print()


def _get_version() -> str:
    try:
        from konash import __version__
        return __version__
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# konash setup
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> None:
    console.print()
    console.print(Panel(
        "[bold cyan]KONASH Setup[/]\n\n"
        "We'll configure your API keys. Takes about 2 minutes.\n\n"
        "[dim]Together AI  — runs the AI model  (free tier)[/]\n"
        "[dim]HuggingFace  — stores your trained model  (free)[/]",
        border_style="cyan",
        padding=(1, 4),
    ))

    config = _load_config()

    # ── Step 1: Together AI ──────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Step 1 of 2  ·  Together AI")
    console.print()

    together_key = _get_together_key()

    if together_key:
        console.print(f"    Key found: [dim]{_mask(together_key)}[/]")
        with console.status("    [cyan]Validating...", spinner="dots"):
            valid = validate_together_key(together_key)
        if valid:
            console.print("    [bold green]✓[/] Valid")
            if not Confirm.ask("    Keep this key?", default=True):
                together_key = None
        else:
            console.print("    [bold red]✗[/] Key no longer works")
            together_key = None

    if not together_key:
        console.print("    [bold]To get your free API key:[/]")
        console.print("    1. Go to Together AI → Settings → API Keys")
        console.print("    2. Sign up or log in")
        console.print("    3. Copy your [bold]User API Key[/]")
        console.print("    [dim]   (Not the Legacy API Key — use the User key at the top)[/]")
        console.print()

        if Confirm.ask("    Open Together AI in your browser?", default=True):
            webbrowser.open(TOGETHER_KEYS_PAGE)

        console.print()
        together_key = Prompt.ask("    Paste your API key", password=True)

        if together_key:
            with console.status("    [cyan]Validating...", spinner="dots"):
                valid = validate_together_key(together_key)
            if valid:
                console.print("    [bold green]✓[/] Key is valid")
                config["together_api_key"] = together_key
            else:
                console.print("    [bold red]✗[/] Invalid key — check and try again")
                return
        else:
            console.print("    [dim]Skipped. Set TOGETHER_API_KEY env var later.[/]")
    else:
        config["together_api_key"] = together_key

    # ── Step 2: HuggingFace ──────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Step 2 of 2  ·  HuggingFace")
    console.print()

    hf_token = _get_hf_token()

    if hf_token:
        console.print(f"    Token found: [dim]{_mask(hf_token)}[/]")
        with console.status("    [cyan]Validating...", spinner="dots"):
            username = validate_hf_token(hf_token)
        if username:
            console.print(f"    [bold green]✓[/] Logged in as [bold]{username}[/]")
            if not Confirm.ask("    Keep this token?", default=True):
                hf_token = None
        else:
            console.print("    [bold red]✗[/] Token no longer works")
            hf_token = None

    if not hf_token:
        # Try OAuth device flow first (no manual copy needed)
        hf_token = hf_device_flow(console)

        if hf_token:
            console.print("    [bold green]✓[/] Authorized via OAuth")
        else:
            # Manual fallback
            console.print("    [bold]To get your free token:[/]")
            console.print("    1. Go to HuggingFace → Settings → Access Tokens")
            console.print("    2. Sign up or log in")
            console.print("    3. Create a token with [bold]Write[/] access")
            console.print("    4. Copy the token")
            console.print()

            if Confirm.ask("    Open HuggingFace in your browser?", default=True):
                webbrowser.open(HF_TOKENS_PAGE)

            console.print()
            hf_token = Prompt.ask("    Paste your token", password=True)

        if hf_token:
            config["hf_token"] = hf_token
        else:
            console.print(
                "    [dim]Skipped. Only needed to deploy trained models.[/]"
            )
    else:
        config["hf_token"] = hf_token

    # ── Save & Summary ───────────────────────────────────────────────
    _save_config(config)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", justify="right", min_width=14)
    summary.add_column()

    if config.get("together_api_key"):
        summary.add_row(
            "Together AI",
            f"[green]✓[/]  {_mask(config['together_api_key'])}",
        )
    else:
        summary.add_row("Together AI", "[red]✗  not set[/]")

    if config.get("hf_token"):
        summary.add_row(
            "HuggingFace",
            f"[green]✓[/]  {_mask(config['hf_token'])}",
        )
    else:
        summary.add_row("HuggingFace", "[yellow]–  skipped[/]")

    summary.add_row("Config", f"[dim]{CONFIG_FILE}[/]")

    console.print()
    console.print(Panel(
        summary,
        title="[bold green]Setup Complete",
        border_style="green",
        padding=(1, 2),
    ))

    console.print()
    console.print("    [bold]Next steps:[/]")
    console.print()
    console.print("    [cyan]konash train ./my_docs[/]")
    console.print('    [cyan]konash ask --corpus ./my_docs "What is X?"[/]')
    console.print()


# ---------------------------------------------------------------------------
# konash setup --check
# ---------------------------------------------------------------------------

def cmd_setup_check() -> None:
    """Non-interactive validation of all keys."""
    together_key = _get_together_key()
    hf_token = _get_hf_token()
    all_ok = True

    console.print()

    if together_key:
        with console.status("[cyan]Checking Together AI...", spinner="dots"):
            valid = validate_together_key(together_key)
        if valid:
            console.print("[green]✓[/] Together AI key valid")
        else:
            console.print("[red]✗[/] Together AI key invalid")
            all_ok = False
    else:
        console.print("[red]✗[/] Together AI key not found")
        all_ok = False

    if hf_token:
        with console.status("[cyan]Checking HuggingFace...", spinner="dots"):
            username = validate_hf_token(hf_token)
        if username:
            console.print(f"[green]✓[/] HuggingFace token valid ({username})")
        else:
            console.print("[red]✗[/] HuggingFace token invalid")
            all_ok = False
    else:
        console.print("[yellow]–[/] HuggingFace token not found (optional)")

    console.print()
    if not all_ok:
        console.print("Run [cyan]konash setup[/] to fix.")
        console.print()
        sys.exit(1)


# ---------------------------------------------------------------------------
# konash download
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    corpus_name = args.corpus_name.lower().replace("_", "-")

    if corpus_name == "browsecomp-plus":
        from konash.download import download_browsecomp_plus

        console.print()
        console.rule("[bold cyan]Downloading BrowseComp-Plus")
        console.print()

        with console.status(
            "    [cyan]Downloading and decrypting...", spinner="dots"
        ):
            output_dir = download_browsecomp_plus(console=console)

        console.print()
        console.print(Panel(
            f"Corpus saved to: [bold]{output_dir}[/]\n\n"
            f"Train with:\n"
            f"  [cyan]konash train {output_dir}/documents[/]",
            title="[bold green]Download Complete",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()
    else:
        console.print(f"\n[red]Unknown corpus:[/] {corpus_name}")
        console.print("Available: [cyan]browsecomp-plus[/]\n")
        sys.exit(1)


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
        console.print(
            "\n[red]No API key found.[/] Run [cyan]konash setup[/] first.\n"
        )
        sys.exit(1)

    # ── Interactive wizard ───────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold cyan]KONASH Training[/]\n\n"
        "Configure your training run.\n"
        "[dim]Press Enter to accept defaults.[/]",
        border_style="cyan",
        padding=(1, 4),
    ))

    # Corpus
    console.print()
    console.rule("[bold cyan]Corpus")
    console.print()

    corpus = args.corpus
    if not corpus:
        console.print('    Enter a path to your documents folder.')
        console.print(
            '    [dim]Type "browsecomp-plus" to download the benchmark.[/]'
        )
        console.print()
        corpus = Prompt.ask("    Path to documents")

        if corpus.lower().replace("_", "-") in ("browsecomp-plus", "bcp"):
            from konash.download import download_browsecomp_plus

            console.print()
            output_dir = download_browsecomp_plus(console=console)
            corpus = os.path.join(output_dir, "documents")
            console.print()

    if not os.path.exists(corpus):
        console.print(f"    [red]Path not found:[/] {corpus}")
        sys.exit(1)

    # Model
    console.print()
    console.rule("[bold cyan]Model")
    console.print()
    model = Prompt.ask(
        "    Model ID", default=args.model or DEFAULT_MODEL,
    )

    # Scale
    console.print()
    console.rule("[bold cyan]Scale")
    console.print()
    console.print(
        "    [dim]KARL paper: 1,735 synthesis calls, 8 rollouts, "
        "200 rollout steps[/]"
    )
    console.print()

    synthesis_calls = IntPrompt.ask(
        "    Synthesis calls / iteration",
        default=args.synthesis_calls,
    )
    rollouts = IntPrompt.ask(
        "    Rollouts per example",
        default=args.rollouts,
    )
    rollout_steps = IntPrompt.ask(
        "    Max steps per rollout",
        default=args.rollout_steps,
    )
    iterations = IntPrompt.ask(
        "    Training iterations",
        default=args.iterations,
    )

    # Advanced
    console.print()
    console.rule("[bold cyan]Advanced")
    console.print()

    lr = FloatPrompt.ask("    Learning rate", default=args.lr)
    chunk_size = IntPrompt.ask("    Chunk size (words)", default=args.chunk_size)

    # ── Summary + confirm ────────────────────────────────────────────
    est_synth = synthesis_calls * iterations
    est_roll = synthesis_calls * 8 * rollouts
    est_cost = est_synth * 0.05 + est_roll * 0.02

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right", min_width=22)
    grid.add_column()
    grid.add_row("Corpus", corpus)
    grid.add_row("Model", model)
    grid.add_row("Synthesis calls", f"{synthesis_calls:,} / iteration")
    grid.add_row("Rollouts / example", str(rollouts))
    grid.add_row("Rollout steps", str(rollout_steps))
    grid.add_row("Iterations", str(iterations))
    grid.add_row("Learning rate", f"{lr:.0e}")
    grid.add_row("Chunk size", f"{chunk_size} words")
    grid.add_row("Est. cost", f"[yellow]~${est_cost:,.0f}[/] on Together AI")

    console.print()
    console.print(Panel(
        grid,
        title="[bold]Training Summary",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    if not Confirm.ask("    Start training?", default=True):
        console.print("    [dim]Aborted.[/]\n")
        return

    # ── Train ────────────────────────────────────────────────────────
    console.print()

    agent = Agent(
        base_model=model,
        corpus=corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
        chunk_size=chunk_size,
    )
    agent.train(
        iterations=iterations,
        synthesis_calls=synthesis_calls,
        rollouts_per_example=rollouts,
        rollout_max_steps=rollout_steps,
        max_examples=args.max_examples,
        learning_rate=lr,
        verbose=True,
    )

    console.print()
    console.print(Panel(
        "[bold green]Training complete![/]",
        border_style="green",
        padding=(0, 2),
    ))
    console.print()


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
        console.print(
            "\n[red]No API key found.[/] Run [cyan]konash setup[/] first.\n"
        )
        sys.exit(1)

    agent = Agent(
        base_model=args.model,
        corpus=args.corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
    )

    with console.status("[cyan]Thinking...", spinner="dots"):
        answer = agent.solve(
            args.query,
            parallel_rollouts=args.parallel,
            top_k=args.top_k,
        )

    console.print()
    console.print(Panel(
        answer,
        title="[bold green]Answer",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# konash search
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    try:
        from konash.corpus import Corpus
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    with console.status("[cyan]Indexing corpus...", spinner="dots"):
        corpus = Corpus(args.corpus, chunk_size=args.chunk_size)
        corpus.ingest()

    console.print(f"[dim]Indexed {corpus.num_documents} chunks[/]\n")

    with console.status("[cyan]Searching...", spinner="dots"):
        results = corpus.search(args.query, top_k=args.top_k)

    table = Table(
        title=f'Results for "{args.query}"',
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Source", style="cyan", max_width=30)
    table.add_column("Text")

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        source = r.get("source", "?")
        text = r.get("text", "")[:150]
        table.add_row(str(i), f"{score:.3f}", source, text + "...")

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# konash status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Key", style="bold", min_width=14)
    table.add_column("Status")

    # Together key
    together_key = _get_together_key()
    if together_key:
        table.add_row("Together AI", f"[green]✓[/]  {_mask(together_key)}")
    else:
        table.add_row("Together AI", "[red]✗  NOT SET[/]")

    # HF token
    hf_token = _get_hf_token()
    if hf_token:
        table.add_row("HuggingFace", f"[green]✓[/]  {_mask(hf_token)}")
    else:
        table.add_row("HuggingFace", "[yellow]–  not set[/]")

    # Config
    table.add_row("Config", f"[dim]{CONFIG_FILE}[/]")

    # Training status
    project_dir = os.path.join(".konash", "default", "checkpoints")
    if os.path.exists(project_dir):
        meta_path = os.path.join(project_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            model = meta.get("base_model", "?")
            iters = meta.get("iterations", 0)
            table.add_row(
                "Trained",
                f"[green]✓[/]  {model} ({iters} iters)",
            )
        else:
            table.add_row("Trained", "[yellow]checkpoint exists, no meta[/]")
    else:
        table.add_row("Trained", "[dim]no[/]")

    console.print()
    console.print(Panel(
        table,
        title="[bold cyan]KONASH Status",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    if not together_key:
        console.print("    Run [cyan]konash setup[/] to configure.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dependency_error(exc: ModuleNotFoundError) -> None:
    missing = exc.name or "a required package"
    console.print(f"\n[red]Missing dependency:[/] [bold]{missing}[/]")
    console.print("Install with: [cyan]pip install konash[/]\n")
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
    p_setup = subparsers.add_parser("setup", help="Set up API keys.")
    p_setup.add_argument(
        "--check", action="store_true",
        help="Validate existing keys (non-interactive).",
    )
    p_setup.set_defaults(func=cmd_setup)

    # --- download ---
    p_dl = subparsers.add_parser(
        "download", help="Download a benchmark corpus.",
    )
    p_dl.add_argument(
        "corpus_name", help="Corpus to download (e.g. browsecomp-plus).",
    )
    p_dl.set_defaults(func=cmd_download)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a knowledge agent.")
    p_train.add_argument(
        "corpus", nargs="?", default=None,
        help="Path to your documents folder (interactive if omitted).",
    )
    p_train.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_train.add_argument("--project", default="default", help="Project name.")
    p_train.add_argument(
        "--iterations", type=int, default=2,
        help="Training iterations (default: 2).",
    )
    p_train.add_argument(
        "--synthesis-calls", type=int, default=1500,
        help="Synthesis calls per iteration (default: 1500, KARL: 1735).",
    )
    p_train.add_argument(
        "--rollouts", type=int, default=8,
        help="Rollouts per example (default: 8).",
    )
    p_train.add_argument(
        "--rollout-steps", type=int, default=50,
        help="Max steps per rollout (default: 50, KARL BCP: 200).",
    )
    p_train.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap on training examples per iteration.",
    )
    p_train.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate.",
    )
    p_train.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size in words.",
    )
    p_train.add_argument(
        "--api-key", default=None,
        help="Together AI key (or run konash setup).",
    )
    p_train.set_defaults(func=cmd_train)

    # --- ask ---
    p_ask = subparsers.add_parser("ask", help="Ask your knowledge agent.")
    p_ask.add_argument("query", help="Your question.")
    p_ask.add_argument("--corpus", required=True, help="Path to documents folder.")
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_ask.add_argument("--project", default="default", help="Project name.")
    p_ask.add_argument(
        "--parallel", type=int, default=1, help="Parallel rollouts."
    )
    p_ask.add_argument(
        "--top-k", type=int, default=10, help="Documents per search."
    )
    p_ask.add_argument("--api-key", default=None, help="Together AI key.")
    p_ask.set_defaults(func=cmd_ask)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search your documents.")
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument(
        "--corpus", required=True, help="Path to documents folder."
    )
    p_search.add_argument(
        "--top-k", type=int, default=5, help="Number of results."
    )
    p_search.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size."
    )
    p_search.set_defaults(func=cmd_search)

    # --- status ---
    p_status = subparsers.add_parser("status", help="Show setup status.")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)

    if args.command is None:
        cmd_default()
        return

    # Handle setup --check
    if args.command == "setup" and args.check:
        cmd_setup_check()
        return

    args.func(args)


if __name__ == "__main__":
    main()
