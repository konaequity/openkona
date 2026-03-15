"""KONASH command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    GOOGLE_AI_KEYS_PAGE,
    TOGETHER_KEYS_PAGE,
    HF_TOKENS_PAGE,
    detect_hf_token,
    hf_device_flow,
    validate_google_key,
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
PROJECTS_DIR = os.path.join(CONFIG_DIR, "projects")

# Models available on Together AI
MODELS = [
    {
        "id": "zai-org/GLM-4.5-Air-FP8",
        "name": "GLM 4.5 Air",
        "hint": "Frontier MoE  ·  best for KARL  ·  fast + cheap",
    },
    {
        "id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "name": "Qwen3 80B-A3B",
        "hint": "MoE  ·  3B active params  ·  very cheap",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "name": "Llama 3.3 70B Turbo",
        "hint": "Dense 70B  ·  strong reasoning  ·  moderate cost",
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "name": "Qwen 2.5 72B Turbo",
        "hint": "Dense 72B  ·  multilingual  ·  moderate cost",
    },
    {
        "id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "name": "Mixtral 8x22B",
        "hint": "MoE  ·  176B total / 39B active  ·  balanced",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "name": "DeepSeek R1",
        "hint": "Reasoning model  ·  671B MoE  ·  chain-of-thought",
    },
]

# Training scale presets
# qa_pairs is the user-facing number; internally divided by 8 to get API calls
SCALE_PRESETS = [
    {
        "name": "Quick",
        "qa_pairs": 400,
        "rollouts": 8,
        "rollout_steps": 30,
        "iterations": 1,
    },
    {
        "name": "Recommended",
        "qa_pairs": 12000,
        "rollouts": 8,
        "rollout_steps": 50,
        "iterations": 2,
    },
    {
        "name": "Exhaustive",
        "qa_pairs": 13880,
        "rollouts": 8,
        "rollout_steps": 200,
        "iterations": 2,
    },
]


def _estimate_training(qa_pairs, rollouts, rollout_steps, iterations):
    """Compute dynamic cost and time estimates."""
    synthesis_calls = max(1, qa_pairs // 8)

    # Cost: Together AI GLM 4.5 Air — $0.20/M in, $1.10/M out
    synth_in = synthesis_calls * iterations * 2000
    synth_out = synthesis_calls * iterations * 1000
    rollout_in = qa_pairs * rollouts * rollout_steps * 500
    rollout_out = qa_pairs * rollouts * rollout_steps * 200
    cost = (
        (synth_in + rollout_in) / 1_000_000 * 0.20
        + (synth_out + rollout_out) / 1_000_000 * 1.10
    )

    # Time: ~2s per synthesis call (parallelized ~20x → /20),
    # ~1s per rollout (parallelized ~4x → /4), ~15 min OAPL per iteration
    synth_secs = (synthesis_calls * iterations * 2) / 20
    rollout_secs = (qa_pairs * rollouts * iterations * 1) / 4
    oapl_secs = iterations * 15 * 60
    total_secs = synth_secs + rollout_secs + oapl_secs

    return cost, total_secs

# Available datasets for the corpus picker
DATASETS = [
    {
        "name": "FinanceBench",
        "key": "financebench",
        "desc": "Financial QA  ·  SEC filings  ·  150 questions",
        "source": "PatronusAI/financebench",
    },
    {
        "name": "BrowseComp-Plus",
        "key": "browsecomp-plus",
        "desc": "Web retrieval benchmark  ·  40K docs  ·  encrypted",
        "source": "Tevatron/browsecomp-plus",
    },
    {
        "name": "QAMPARI",
        "key": "qampari",
        "desc": "Multi-answer QA  ·  Wikipedia  ·  entity-rich",
        "source": "momo4382/QAMPARI",
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


def _get_google_key() -> str | None:
    return _get_key(["GOOGLE_API_KEY"], "google_api_key")


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
    grid.add_row('konash ask "Q"', "Ask a question")
    grid.add_row("konash search --corpus ./docs Q", "Search documents")
    grid.add_row("konash logs", "Stream GPU training logs")
    grid.add_row("konash stop", "Tear down GPU cluster")
    grid.add_row("konash projects", "List trained projects")
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

def _animate_logo(con: Console) -> None:
    """Animate the KONASH logo on startup."""
    import time as _t
    from rich.text import Text
    from rich.live import Live

    con.print()

    # Phase 1: Search beam animation (bypass Rich to avoid \r conflicts)
    import sys as _sys
    beam_width = 36
    for i in range(beam_width + 1):
        bar = "━" * i + ("◉" if i < beam_width else "")
        pad = " " * (beam_width - i)
        _sys.stdout.write(f"\r    \033[96m{bar}\033[0m{pad}")
        _sys.stdout.flush()
        _t.sleep(0.015)

    # Phase 2: Beam dissolves into the logo
    _t.sleep(0.15)
    _sys.stdout.write("\r" + " " * 50 + "\r")
    _sys.stdout.flush()

    # Phase 3: Big letters fade in line by line (gradient white → cyan)
    logo = [
        "[bold bright_white]█  █  ▄▀▀▄  █▄  █  ▄▀▀▄  ▄▀▀▀  █  █[/]",
        "[bold bright_white]█▄▀   █  █  █ █ █  █▀▀█  ▀▀▀█  █▀▀█[/]",
        "[bold bright_cyan]█  █  ▀▄▄▀  █  ▀█  █  █  ▄▄▄▀  █  █[/]",
    ]
    for line in logo:
        con.print(f"    {line}")
        _t.sleep(0.05)
    con.print()


def cmd_setup(args: argparse.Namespace) -> None:
    _start_web_ui()
    _animate_logo(console)
    console.print(f"    [bold]Welcome to KONASH[/]  [dim]{_get_version()}[/]")
    console.print(
        "    [dim]Train knowledge agents that search, retrieve, and reason.[/]"
    )
    console.print()
    console.print(
        "    KONASH uses Together AI to run large language models.\n"
        "    You'll need a free API key to get started."
    )
    console.print()
    console.rule(style="dim")
    console.print()

    config = _load_config()

    # ── Together AI (the only required key) ───────────────────────────
    together_key = _get_together_key()

    if together_key:
        console.print(f"    Key found: [dim]{_mask(together_key)}[/]")
        with console.status("    Validating...", spinner="dots"):
            valid = validate_together_key(together_key)
        if valid:
            console.print("    [green]✓[/]  Connected to Together AI")
            config["together_api_key"] = together_key
        else:
            console.print("    [red]✗[/]  Key no longer works")
            together_key = None

    if not together_key:
        idx = _arrow_select(console, [
            {"label": "Open together.ai", "hint": "Sign up and grab a free API key (takes 30 seconds)"},
            {"label": "I already have a key", "hint": ""},
        ])
        console.print()

        if idx == 0:
            webbrowser.open(TOGETHER_KEYS_PAGE)
            console.print("    [dim]Settings → API Keys → copy the User key (not Legacy)[/]")
            console.print()

        together_key = Prompt.ask("    Paste your API key", password=True)

        if together_key:
            with console.status("    Validating...", spinner="dots"):
                valid = validate_together_key(together_key)
            if valid:
                console.print("    [green]✓[/]  Connected to Together AI")
                config["together_api_key"] = together_key
            else:
                console.print("    [red]✗[/]  Invalid key — check and try again")
                return
        else:
            console.print()
            console.print("    Run [bold]konash setup[/] when you have a key.")
            console.print()
            return

    # ── Save and go ──────────────────────────────────────────────────
    _save_config(config)

    console.print()
    console.rule(style="dim")
    console.print()
    console.print("    [green]✓[/]  You're all set. Let's train your first agent.")
    console.print()

    # ── Flow directly into training ─────────────────────────────────
    train_args = argparse.Namespace(
        corpus=None,
        model=DEFAULT_MODEL,
        project="default",
        iterations=2,
        qa_pairs=12000,
        rollouts=8,
        rollout_steps=50,
        max_examples=None,
        lr=1e-5,
        chunk_size=512,
        api_key=None,
    )
    cmd_train(train_args)


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
    google_key = _get_google_key()
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

    if google_key:
        with console.status("[cyan]Checking Google AI...", spinner="dots"):
            valid = validate_google_key(google_key)
        if valid:
            console.print("[green]✓[/] Google AI key valid")
        else:
            console.print("[red]✗[/] Google AI key invalid")
            all_ok = False
    else:
        console.print("[red]✗[/] Google AI key not found")
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
# Web UI (auto-started during training)
# ---------------------------------------------------------------------------

_web_ui_process = None


def _start_web_ui() -> None:
    """Start the trace viewer / training monitor in the background."""
    global _web_ui_process
    import subprocess

    if _web_ui_process is not None:
        return  # Already running

    app_path = os.path.join(os.path.dirname(__file__), "..", "tools", "trace_viewer", "app.py")
    if not os.path.exists(app_path):
        return  # Not available (pip install, no tools/)

    try:
        _web_ui_process = subprocess.Popen(
            [sys.executable, app_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print("  [dim]Dashboard: http://localhost:5050/training/[/]")
    except Exception:
        pass  # Non-critical


def _stop_web_ui() -> None:
    """Stop the background web UI."""
    global _web_ui_process
    if _web_ui_process is not None:
        _web_ui_process.terminate()
        _web_ui_process = None


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
    console.print(
        "    Pick a document corpus to train on. KONASH will synthesize\n"
        "    questions from these documents, then train your model to\n"
        "    search and reason over them."
    )
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

    # ── Model + Scale (with ability to go back) ────────────────────
    model = args.model or DEFAULT_MODEL
    qa_pairs = args.qa_pairs if hasattr(args, "qa_pairs") else 12000
    rollouts = args.rollouts
    rollout_steps = args.rollout_steps
    iterations = args.iterations
    lr = args.lr
    chunk_size = args.chunk_size

    def _pick_model() -> str:
        console.print()
        console.rule("[bold]Model[/]", style="dim")
        console.print()
        opts = [{"label": m["name"], "hint": m["hint"]} for m in MODELS]
        opts.append({"label": "Custom", "hint": "Enter a model ID manually"})
        console.print("    [dim]Use arrow keys, press Enter to select[/]")
        console.print()
        idx = _arrow_select(console, opts)
        console.print()
        if idx < len(MODELS):
            return MODELS[idx]["id"]
        return Prompt.ask("    Model ID")

    def _pick_scale() -> tuple:
        console.print()
        console.rule("[bold]Scale[/]", style="dim")
        console.print()
        opts = []
        for p in SCALE_PRESETS:
            _, t = _estimate_training(p["qa_pairs"], p["rollouts"], p["rollout_steps"], p["iterations"])
            time_str = _format_duration(t).lstrip("~")
            opts.append({
                "label": p["name"],
                "hint": (
                    f"~{p['qa_pairs']:,} QA pairs  ·  "
                    f"{p['rollouts']} rollouts  ·  "
                    f"{p['iterations']} iter  ·  "
                    f"~{time_str}"
                ),
            })
        opts.append({"label": "Custom", "hint": "Set each parameter manually"})
        console.print("    [dim]Use arrow keys, press Enter to select[/]")
        console.print()
        idx = _arrow_select(console, opts)
        console.print()
        if idx < len(SCALE_PRESETS):
            p = SCALE_PRESETS[idx]
            return p["qa_pairs"], p["rollouts"], p["rollout_steps"], p["iterations"]
        qp = IntPrompt.ask("    QA pairs to synthesize", default=12000)
        ro = IntPrompt.ask("    Rollouts per example", default=8)
        rs = IntPrompt.ask("    Max steps per rollout", default=50)
        it = IntPrompt.ask("    Training iterations", default=2)
        return qp, ro, rs, it

    model = _pick_model()
    qa_pairs, rollouts, rollout_steps, iterations = _pick_scale()

    # ── Summary + confirm (with go-back loop) ────────────────────────
    while True:
        synthesis_calls = max(1, qa_pairs // 8)
        est_cost, est_total_secs = _estimate_training(
            qa_pairs, rollouts, rollout_steps, iterations,
        )

        console.print()
        console.rule(style="dim")
        console.print()

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", justify="right", min_width=22)
        grid.add_column()
        grid.add_row("Corpus", corpus)
        grid.add_row("Model", model)
        grid.add_row("QA pairs", f"~{qa_pairs:,} / iteration")
        grid.add_row("Rollouts / example", str(rollouts))
        grid.add_row("Rollout steps", str(rollout_steps))
        grid.add_row("Iterations", str(iterations))
        grid.add_row("Est. cost", f"~${est_cost:,.0f}" if est_cost >= 1 else f"~${est_cost:,.2f}")
        grid.add_row("Est. time", _format_duration(est_total_secs))
        console.print(grid)
        console.print()

        confirm_options = [
            {"label": "Start training", "hint": "Begin with these settings"},
            {"label": "Change model", "hint": f"Currently: {model}"},
            {"label": "Change scale", "hint": f"Currently: ~{qa_pairs:,} QA pairs"},
            {"label": "Cancel", "hint": "Exit without training"},
        ]
        confirm_idx = _arrow_select(console, confirm_options)
        console.print()

        if confirm_idx == 0:
            break
        elif confirm_idx == 1:
            model = _pick_model()
        elif confirm_idx == 2:
            qa_pairs, rollouts, rollout_steps, iterations = _pick_scale()
        else:
            console.print("    [dim]Cancelled.[/]\n")
            return

    # ── Train ────────────────────────────────────────────────────────
    console.print()

    agent = Agent(
        base_model=model,
        corpus=corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
        hf_token=_get_hf_token(),
        chunk_size=chunk_size,
    )
    _start_web_ui()  # no-op if already running from setup
    train_start = time.monotonic()
    agent.train(
        iterations=iterations,
        synthesis_calls=synthesis_calls,
        rollouts_per_example=rollouts,
        rollout_max_steps=rollout_steps,
        max_examples=args.max_examples,
        learning_rate=lr,
        verbose=True,
    )
    elapsed = time.monotonic() - train_start

    console.print()
    console.rule(style="dim")
    console.print()
    console.print(
        f"    [green]✓[/]  Training complete  [dim]{_format_duration(elapsed).lstrip('~')}[/]"
    )
    console.print()
    console.print("    [bold]What's next?[/]")
    console.print()

    next_options = [
        {"label": "Ask a question", "hint": "Test your trained agent on the corpus"},
        {"label": "Exit", "hint": "Come back later with `konash ask`"},
    ]
    next_idx = _arrow_select(console, next_options)
    console.print()

    if next_idx == 0:
        # Interactive ask loop — use VGS if value model was trained
        has_vgs = agent._value_model is not None
        if has_vgs:
            console.print("    [dim]Using Value-Guided Search. Type a question, Ctrl+C to exit.[/]")
        else:
            console.print("    [dim]Type a question and press Enter. Ctrl+C to exit.[/]")
        console.print()
        while True:
            try:
                question = Prompt.ask("    [bold]Q[/]")
            except (KeyboardInterrupt, EOFError):
                console.print()
                break
            if not question.strip():
                continue
            spinner_label = "[cyan]Searching (VGS)..." if has_vgs else "[cyan]Thinking..."
            with console.status(spinner_label, spinner="dots"):
                answer = agent.solve(
                    question,
                    parallel_rollouts=3 if has_vgs else 1,
                    use_vgs=True if has_vgs else None,
                )
            console.print(f"    [bold]A[/]  {answer}")
            console.print()
    else:
        project_name = args.project or "default"
        console.print(f"    [dim]Run [cyan]konash ask --project {project_name} "
                       f"--corpus {corpus} \"your question\"[/] to query later.[/]")
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

    # Resolve corpus: explicit flag > training metadata > error
    corpus = args.corpus
    if not corpus:
        meta_path = os.path.join(PROJECTS_DIR, args.project, "checkpoints", "training_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            corpus = meta.get("corpus")
            if corpus:
                console.print(f"[dim]Using corpus from project \"{args.project}\": {corpus}[/]")
        if not corpus:
            console.print(
                "\n[red]No corpus specified.[/] Use [cyan]--corpus[/] or train first "
                "with [cyan]konash train[/].\n"
            )
            sys.exit(1)

    agent = Agent(
        base_model=args.model,
        corpus=corpus,
        project=args.project,
        api_base=TOGETHER_API_BASE,
        api_key=api_key,
        hf_token=_get_hf_token(),
    )

    # Try loading value model from checkpoint
    _use_vgs = args.vgs
    if _use_vgs:
        import json as _json
        vm_path = os.path.join(PROJECTS_DIR, args.project, "checkpoints", "value_model.json")
        if os.path.exists(vm_path):
            from konash.inference.value_model import ValueModel
            with open(vm_path) as f:
                vm_data = _json.load(f)
            agent._value_model = ValueModel(
                weights=vm_data["weights"],
                bias=vm_data["bias"],
                feature_dim=vm_data["feature_dim"],
            )
        else:
            console.print("[yellow]No value model found — falling back to standard inference.[/]")
            _use_vgs = False

    spinner_label = "[cyan]Searching (VGS)..." if _use_vgs else "[cyan]Thinking..."
    with console.status(spinner_label, spinner="dots"):
        answer = agent.solve(
            args.query,
            parallel_rollouts=args.parallel if args.parallel > 1 else (3 if _use_vgs else 1),
            top_k=args.top_k,
            use_vgs=True if _use_vgs else None,
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

    _cache_dir = os.path.join(CONFIG_DIR, "search", "index_cache")
    corpus = Corpus(args.corpus, chunk_size=args.chunk_size, cache_dir=_cache_dir)

    _status_msg = console.status("[cyan]Indexing corpus...", spinner="dots")
    _status_msg.start()

    def _search_progress(phase: str, current: int, total: int) -> None:
        labels = {"reading": "Reading", "chunking": "Chunking", "embedding": "Embedding"}
        label = labels.get(phase, phase)
        if current < total:
            pct = current * 100 // total if total else 0
            _status_msg.update(f"[cyan]{label}  {current:,}/{total:,}  ({pct}%)")
        else:
            _status_msg.update(f"[cyan]{label}  {total:,} done")

    corpus.ingest(progress_callback=_search_progress)
    _status_msg.stop()

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

    # Together key (required)
    together_key = _get_together_key()
    if together_key:
        console.print(f"    [green]✓[/]  Together AI    {_mask(together_key)}")
    else:
        console.print("    [red]✗[/]  Together AI    not set  [dim](run konash setup)[/]")

    # Shadeform (GPU provider)
    config = _load_config()
    sf_key = config.get("shadeform_api_key")
    if sf_key:
        console.print(f"    [green]✓[/]  Shadeform      {_mask(sf_key)}")

    # Optional keys — only show if configured
    google_key = _get_google_key()
    if google_key:
        console.print(f"    [green]✓[/]  Google AI      {_mask(google_key)}")

    hf_token = _get_hf_token()
    if hf_token:
        console.print(f"    [green]✓[/]  HuggingFace    {_mask(hf_token)}")

    # Config
    console.print(f"    [dim]Config  {CONFIG_FILE}[/]")

    # Training status
    project_dir = os.path.join(PROJECTS_DIR, "default", "checkpoints")
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
# konash projects
# ---------------------------------------------------------------------------

def cmd_projects(args: argparse.Namespace) -> None:
    console.print()
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Projects")
    console.print()
    console.rule(style="dim")
    console.print()

    konash_dir = PROJECTS_DIR
    if not os.path.isdir(konash_dir):
        console.print("    [dim]No projects found. Run [cyan]konash train[/] to get started.[/]")
        console.print()
        return

    rows = []
    for name in sorted(os.listdir(konash_dir)):
        meta_path = os.path.join(konash_dir, name, "checkpoints", "training_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        model = meta.get("base_model", "?")
        # Shorten model ID: "zai-org/GLM-4.5-Air-FP8" → "GLM-4.5-Air-FP8"
        if "/" in model:
            model = model.split("/", 1)[1]
        corpus_path = meta.get("corpus", "?")
        iters = meta.get("iterations", 0)
        has_vgs = meta.get("value_model", False)
        rows.append((name, model, corpus_path, iters, has_vgs))

    if not rows:
        console.print("    [dim]No trained projects found. Run [cyan]konash train[/] to get started.[/]")
        console.print()
        return

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("Project", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Corpus")
    table.add_column("Iters", justify="right")
    table.add_column("VGS", justify="center")

    for name, model, corpus_path, iters, has_vgs in rows:
        vgs_marker = "[green]✓[/]" if has_vgs else "[dim]–[/]"
        table.add_row(name, model, corpus_path, str(iters), vgs_marker)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# konash logs
# ---------------------------------------------------------------------------

def cmd_logs(args: argparse.Namespace) -> None:
    console.print()
    console.print(f"[bold]KONASH[/]  Training Logs")
    console.print()
    from konash.cloud import stream_logs
    stream_logs()


# ---------------------------------------------------------------------------
# konash stop
# ---------------------------------------------------------------------------

def cmd_stop(args: argparse.Namespace) -> None:
    console.print()
    console.print(f"[bold]KONASH[/]  Stop Training")
    console.print()
    from konash.cloud import tear_down
    tear_down()
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"~{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"~{m}m {s}s" if s else f"~{m}m"
    h, remainder = divmod(seconds, 3600)
    m = remainder // 60
    return f"~{h}h {m}m" if m else f"~{h}h"


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
        "--qa-pairs", type=int, default=12000,
        help="QA pairs to synthesize per iteration (default: 12000).",
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
    p_ask.add_argument("--corpus", default=None, help="Path to documents folder (auto-detected from training).")
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_ask.add_argument("--project", default="default", help="Project name.")
    p_ask.add_argument(
        "--parallel", type=int, default=1, help="Parallel rollouts."
    )
    p_ask.add_argument(
        "--top-k", type=int, default=10, help="Documents per search."
    )
    p_ask.add_argument("--api-key", default=None, help="Together AI key.")
    p_ask.add_argument(
        "--vgs", action="store_true", default=False,
        help="Use Value-Guided Search (requires trained value model).",
    )
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

    # --- logs ---
    p_logs = subparsers.add_parser("logs", help="Stream GPU training logs.")
    p_logs.set_defaults(func=cmd_logs)

    # --- stop ---
    p_stop = subparsers.add_parser("stop", help="Tear down GPU cluster.")
    p_stop.set_defaults(func=cmd_stop)

    # --- projects ---
    p_projects = subparsers.add_parser("projects", help="List trained projects.")
    p_projects.set_defaults(func=cmd_projects)

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
