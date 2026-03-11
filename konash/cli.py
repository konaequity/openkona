"""KONASH command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

from rich import box
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table


def _arrow_select(console: Console, options: list[dict]) -> int:
    """Arrow-key selector. Returns the chosen index.

    Each option is ``{"label": str, "hint": str}``.
    """
    import tty
    import termios

    selected = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def _render() -> None:
        for i, opt in enumerate(options):
            marker = "[bold]>[/]" if i == selected else " "
            style = "bold" if i == selected else "dim"
            console.print(f"    {marker}  [{style}]{opt['label']}[/]")
            if i == selected:
                console.print(f"       [dim]{opt['hint']}[/]")

    def _clear(n: int) -> None:
        for _ in range(n):
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.flush()

    def _display_lines() -> int:
        """Count how many lines the current render uses."""
        total = 0
        for i, opt in enumerate(options):
            total += 1  # label line
            if i == selected:
                total += 1  # hint line
        return total

    try:
        tty.setraw(fd)
        # Initial render — switch to normal mode briefly
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _render()
        prev_lines = _display_lines()
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                break
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":  # up
                        selected = (selected - 1) % len(options)
                    elif ch3 == "B":  # down
                        selected = (selected + 1) % len(options)

            # Re-render
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            _clear(prev_lines)
            _render()
            prev_lines = _display_lines()
            tty.setraw(fd)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return selected

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

# Available datasets for the corpus picker
DATASETS = [
    {
        "name": "BrowseComp-Plus",
        "key": "browsecomp-plus",
        "desc": "Web retrieval benchmark  ·  40K docs  ·  encrypted",
        "source": "Tevatron/browsecomp-plus",
    },
    {
        "name": "FinanceBench",
        "key": "financebench",
        "desc": "Financial QA  ·  SEC filings  ·  150 questions",
        "source": "PatronusAI/financebench",
    },
    {
        "name": "QAMPARI",
        "key": "qampari",
        "desc": "Multi-answer QA  ·  Wikipedia  ·  entity-rich",
        "source": "samsam3232/qampari",
    },
    {
        "name": "FreshStack",
        "key": "freshstack",
        "desc": "Multi-domain retrieval  ·  5 domains  ·  recent docs",
        "source": "freshstack",
    },
]


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
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]")
    console.print(
        "[dim]Train knowledge agents that search, retrieve, and reason.[/]"
    )
    console.print()
    console.rule(style="dim")

    # Commands grid — clean, no panels
    console.print()
    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold", min_width=36)
    grid.add_column(style="dim")
    grid.add_row("konash setup", "Configure API keys")
    grid.add_row("konash train", "Train an agent")
    grid.add_row("konash download browsecomp-plus", "Download benchmark corpus")
    grid.add_row('konash ask --corpus ./docs "Q"', "Ask a question")
    grid.add_row("konash search --corpus ./docs Q", "Search documents")
    grid.add_row("konash status", "Check configuration")
    console.print(grid)
    console.print()


def _get_version() -> str:
    try:
        from konash import __version__
        return f"v{__version__}"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# konash setup
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> None:
    console.print()
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Setup")
    console.print()
    console.rule(style="dim")
    console.print()
    console.print("    [dim]Together AI[/]  — runs the model  (free tier)")
    console.print("    [dim]HuggingFace[/]  — stores trained models  (free)")
    console.print()

    config = _load_config()

    # ── 1 · Together AI ──────────────────────────────────────────────
    console.rule("[bold]1[/]  Together AI", style="dim")
    console.print()

    together_key = _get_together_key()

    if together_key:
        console.print(f"    Key found: [dim]{_mask(together_key)}[/]")
        with console.status("    Validating...", spinner="dots"):
            valid = validate_together_key(together_key)
        if valid:
            console.print("    [green]✓[/]  Valid")
            if not Confirm.ask("    Keep this key?", default=True):
                together_key = None
        else:
            console.print("    [red]✗[/]  Key no longer works")
            together_key = None

    if not together_key:
        console.print("    Get your free User API Key:")
        console.print("    [dim]Settings → API Keys → copy the User key (not Legacy)[/]")
        console.print()

        if Confirm.ask("    Open together.ai?", default=True):
            webbrowser.open(TOGETHER_KEYS_PAGE)

        console.print()
        together_key = Prompt.ask("    Paste your User API Key", password=True)

        if together_key:
            with console.status("    Validating...", spinner="dots"):
                valid = validate_together_key(together_key)
            if valid:
                console.print("    [green]✓[/]  Valid")
                config["together_api_key"] = together_key
            else:
                console.print("    [red]✗[/]  Invalid — check and try again")
                return
        else:
            console.print("    [dim]Skipped. Set TOGETHER_API_KEY later.[/]")
    else:
        config["together_api_key"] = together_key

    # ── 2 · HuggingFace ──────────────────────────────────────────────
    console.print()
    console.rule("[bold]2[/]  HuggingFace", style="dim")
    console.print()

    hf_token = _get_hf_token()

    if hf_token:
        console.print(f"    Token found: [dim]{_mask(hf_token)}[/]")
        with console.status("    Validating...", spinner="dots"):
            username = validate_hf_token(hf_token)
        if username:
            console.print(f"    [green]✓[/]  {username}")
            if not Confirm.ask("    Keep this token?", default=True):
                hf_token = None
        else:
            console.print("    [red]✗[/]  Token no longer works")
            hf_token = None

    if not hf_token:
        hf_token = hf_device_flow(console)

        if hf_token:
            console.print("    [green]✓[/]  Authorized via OAuth")
        else:
            console.print("    Create a token with Write access:")
            console.print("    [dim]Settings → Access Tokens → New token[/]")
            console.print()

            if Confirm.ask("    Open huggingface.co?", default=True):
                webbrowser.open(HF_TOKENS_PAGE)

            console.print()
            hf_token = Prompt.ask("    Paste your token", password=True)

        if hf_token:
            config["hf_token"] = hf_token
        else:
            console.print("    [dim]Skipped — only needed to deploy models.[/]")
    else:
        config["hf_token"] = hf_token

    # ── Summary ──────────────────────────────────────────────────────
    _save_config(config)

    console.print()
    console.rule(style="dim")
    console.print()

    if config.get("together_api_key"):
        console.print(
            f"    [green]✓[/]  Together AI    {_mask(config['together_api_key'])}"
        )
    else:
        console.print("    [red]✗[/]  Together AI    not set")

    if config.get("hf_token"):
        console.print(
            f"    [green]✓[/]  HuggingFace    {_mask(config['hf_token'])}"
        )
    else:
        console.print("    [dim]–[/]  HuggingFace    skipped")

    console.print(f"    [dim]Config saved to {CONFIG_FILE}[/]")

    # ── Flow directly into training ─────────────────────────────────
    if config.get("together_api_key"):
        console.print()
        train_args = argparse.Namespace(
            corpus=None,
            model=DEFAULT_MODEL,
            project="default",
            iterations=2,
            synthesis_calls=1500,
            rollouts=8,
            rollout_steps=50,
            max_examples=None,
            lr=1e-5,
            chunk_size=512,
            api_key=None,
        )
        cmd_train(train_args)
    else:
        console.print()
        console.print("    Run [bold]konash setup[/] again to add your API key.")
        console.print()


def _download_dataset(key: str) -> str:
    """Download a dataset by key and return the documents path."""
    from konash.download import (
        download_browsecomp_plus,
        download_financebench,
        download_freshstack,
        download_qampari,
    )

    downloaders = {
        "browsecomp-plus": download_browsecomp_plus,
        "financebench": download_financebench,
        "qampari": download_qampari,
        "freshstack": download_freshstack,
    }

    fn = downloaders.get(key)
    if not fn:
        console.print(f"    [red]Unknown dataset:[/] {key}")
        sys.exit(1)

    console.print()
    output_dir = fn(console=console)
    return os.path.join(output_dir, "documents")


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
    valid_keys = {ds["key"] for ds in DATASETS}

    if corpus_name not in valid_keys:
        console.print(f"\n[red]Unknown corpus:[/] {corpus_name}")
        console.print("Available: " + ", ".join(f"[bold]{k}[/]" for k in valid_keys))
        console.print()
        sys.exit(1)

    console.print()
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Download  [dim]{corpus_name}[/]")
    console.print()
    console.rule(style="dim")
    console.print()

    docs_path = _download_dataset(corpus_name)

    console.print()
    console.rule(style="dim")
    console.print()
    console.print(f"    [green]✓[/]  Saved to {os.path.dirname(docs_path)}")
    console.print(f"    [dim]Train with:[/]  konash train {docs_path}")
    console.print()


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
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Train")
    console.print("[dim]Press Enter to accept defaults.[/]")
    console.print()
    console.rule(style="dim")

    # Corpus
    console.print()
    console.rule("[bold]Corpus[/]", style="dim")
    console.print()

    corpus = args.corpus
    if not corpus:
        options = [
            {"label": ds["name"], "hint": ds["desc"]}
            for ds in DATASETS
        ]
        options.append(
            {"label": "Local folder", "hint": "Point to your own documents directory"}
        )

        console.print("    [dim]Use arrow keys, press Enter to select[/]")
        console.print()
        idx = _arrow_select(console, options)
        console.print()

        if idx < len(DATASETS):
            ds = DATASETS[idx]
            corpus = _download_dataset(ds["key"])
        else:
            corpus = Prompt.ask("    Path to documents")

    if not corpus or not os.path.exists(corpus):
        console.print(f"    [red]Path not found:[/] {corpus}")
        sys.exit(1)

    # Model
    console.print()
    console.rule("[bold]Model[/]", style="dim")
    console.print()
    model = Prompt.ask(
        "    Model ID", default=args.model or DEFAULT_MODEL,
    )

    # Scale
    console.print()
    console.rule("[bold]Scale[/]", style="dim")
    console.print()
    console.print(
        "    [dim]KARL: 1,735 synthesis calls · 8 rollouts · 200 steps[/]"
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
    console.rule("[bold]Advanced[/]", style="dim")
    console.print()

    lr = FloatPrompt.ask("    Learning rate", default=args.lr)
    chunk_size = IntPrompt.ask("    Chunk size (words)", default=args.chunk_size)

    # ── Summary + confirm ────────────────────────────────────────────
    est_synth = synthesis_calls * iterations
    est_roll = synthesis_calls * 8 * rollouts
    est_cost = est_synth * 0.05 + est_roll * 0.02

    console.print()
    console.rule(style="dim")
    console.print()

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", min_width=22)
    grid.add_column()
    grid.add_row("Corpus", corpus)
    grid.add_row("Model", model)
    grid.add_row("Synthesis calls", f"{synthesis_calls:,} / iteration")
    grid.add_row("Rollouts / example", str(rollouts))
    grid.add_row("Rollout steps", str(rollout_steps))
    grid.add_row("Iterations", str(iterations))
    grid.add_row("Learning rate", f"{lr:.0e}")
    grid.add_row("Chunk size", f"{chunk_size} words")
    grid.add_row("Est. cost", f"~${est_cost:,.0f}")
    console.print(grid)
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
    console.rule(style="dim")
    console.print()
    console.print("    [green]✓[/]  Training complete")
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
    console.rule(style="dim")
    console.print()
    console.print(answer)
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
    console.print()
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Status")
    console.print()
    console.rule(style="dim")
    console.print()

    # Together key
    together_key = _get_together_key()
    if together_key:
        console.print(f"    [green]✓[/]  Together AI    {_mask(together_key)}")
    else:
        console.print("    [red]✗[/]  Together AI    not set")

    # HF token
    hf_token = _get_hf_token()
    if hf_token:
        console.print(f"    [green]✓[/]  HuggingFace    {_mask(hf_token)}")
    else:
        console.print("    [dim]–[/]  HuggingFace    not set")

    # Config
    console.print(f"    [dim]Config  {CONFIG_FILE}[/]")

    # Training status
    project_dir = os.path.join(".konash", "default", "checkpoints")
    if os.path.exists(project_dir):
        meta_path = os.path.join(project_dir, "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            model = meta.get("base_model", "?")
            iters = meta.get("iterations", 0)
            console.print(f"    [green]✓[/]  Trained        {model} ({iters} iters)")
        else:
            console.print("    [dim]–[/]  Trained        checkpoint exists, no meta")
    else:
        console.print("    [dim]–[/]  Trained        no")

    console.print()

    if not together_key:
        console.print("    Run [bold]konash setup[/] to configure.")
        console.print()


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
