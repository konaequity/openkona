"""KONASH command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser

from konash.benchmarks import get_dataset, list_datasets
from konash.models import CliModelOption, get_cli_models
from konash.training.project_state import (
    LEGACY_DEFAULT_PROJECT,
    archive_legacy_default_project,
    archive_project_run_state,
    assess_project_reuse,
    begin_training_run,
    build_dataset_spec,
    mark_training_run_status,
    suggest_project_name,
    TrainingRunConfig,
)
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
    HF_TOKENS_PAGE,
    OPENAI_KEYS_PAGE,
    SHADEFORM_KEYS_PAGE,
    TOGETHER_KEYS_PAGE,
    detect_hf_token,
    hf_device_flow,
    validate_google_key,
    validate_hf_token,
    validate_openai_key,
    validate_shadeform_key,
    validate_together_key,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

console = Console()

DEFAULT_MODEL = "zai-org/GLM-4.5-Air-FP8"
CONFIG_DIR = os.path.expanduser("~/.konash")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
PROJECTS_DIR = os.path.join(CONFIG_DIR, "projects")

MODELS: list[CliModelOption] = get_cli_models()

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

DEFAULT_SCALE_PRESET = SCALE_PRESETS[0]


def _resolve_dataset_aliases() -> dict[str, tuple[str, str]]:
    aliases: dict[str, tuple[str, str]] = {}
    for ds in DATASETS:
        root = os.path.realpath(os.path.expanduser(ds.corpus_root()))
        aliases[root] = (ds.key, ds.name)
    return aliases


def _build_training_dataset_spec(corpus: str):
    return build_dataset_spec([corpus], aliases=_resolve_dataset_aliases())


def _short_model_name(model: str) -> str:
    return model.split("/", 1)[-1]


def _should_use_sleep_wake(model: str, gpu_type: str) -> bool:
    """Use the sleep/wake training path for GLM 4.5 Air on Shadeform GPUs."""
    _ = gpu_type  # kept for signature stability and future GPU-specific tuning
    return "GLM-4.5" in model.upper()


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


def _estimate_training_plan(
    qa_pairs: int,
    rollouts: int,
    rollout_steps: int,
    iterations: int,
    synthesis_backend: str,
) -> dict[str, float]:
    """Estimate training cost/time for the visible execution plan."""
    synthesis_calls = max(1, qa_pairs // 8)

    synth_in = synthesis_calls * iterations * 2000
    synth_out = synthesis_calls * iterations * 1000
    rollout_in = qa_pairs * rollouts * iterations * rollout_steps * 500
    rollout_out = qa_pairs * rollouts * iterations * rollout_steps * 200
    api_cost = (
        (synth_in + rollout_in) / 1_000_000 * 0.20
        + (synth_out + rollout_out) / 1_000_000 * 1.10
    )

    synth_secs = (synthesis_calls * iterations * 2) / 20
    rollout_secs = (qa_pairs * rollouts * iterations * 1) / 4
    oapl_secs = iterations * 15 * 60

    gpu_hourly = 2.29  # rough midpoint for a single H100 on Shadeform
    gpu_secs = synth_secs + rollout_secs + oapl_secs
    total_cost = gpu_secs / 3600 * gpu_hourly
    api_cost = 0.0

    return {
        "api_cost": api_cost,
        "gpu_cost": gpu_secs / 3600 * gpu_hourly,
        "total_cost": total_cost,
        "total_secs": synth_secs + rollout_secs + oapl_secs,
    }

DATASETS = list_datasets()


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


def _get_shadeform_key() -> str | None:
    return _get_key(["SHADEFORM_API_KEY"], "shadeform_api_key")


def _get_zhipu_key() -> str | None:
    return _get_key(["ZHIPU_API_KEY"], "zhipu_api_key")


def _get_google_key() -> str | None:
    return _get_key(["GOOGLE_API_KEY"], "google_api_key")


def _get_hf_token() -> str | None:
    token = _get_key(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"], "hf_token")
    if token:
        return token
    return detect_hf_token()


def _get_openai_key() -> str | None:
    return _get_key(["OPENAI_API_KEY"], "openai_api_key")


def _mask(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:8] + "..." + key[-4:]


def _render_capability_summary(config: dict) -> None:
    together_key = _get_together_key()
    shadeform_key = _get_shadeform_key()
    hf_token = _get_hf_token()
    openai_key = _get_openai_key()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("Stage", style="bold")
    table.add_column("Status", width=10)
    table.add_column("What It Enables")

    stages = [
        (
            "Synthesis + training",
            shadeform_key,
            "Shadeform GPU for synthesis, rollouts, and OAPL training",
            True,  # required
        ),
        (
            "Together AI",
            together_key,
            "Eval benchmarks and model serving (optional)",
            False,
        ),
        (
            "HF assets + embeddings",
            hf_token,
            "HF-hosted corpora/indexes and HF embedding/query paths",
            False,
        ),
        (
            "Eval judge",
            openai_key,
            "gpt-4o-mini benchmark judging instead of solver-as-judge fallback",
            False,
        ),
    ]

    for name, key, effect, required in stages:
        if key:
            status = "[green]Enabled[/]"
        elif required:
            status = "[red]Missing[/]"
        else:
            status = "[yellow]Optional[/]"
        table.add_row(name, status, effect)

    console.print(table)
    console.print()

    blocked = []
    if not shadeform_key:
        blocked.append("Training requires a Shadeform API key for synthesis and OAPL.")
    if not together_key:
        blocked.append("Together AI is not configured — eval benchmarks may use solver-as-judge instead.")
    if not hf_token:
        blocked.append("Some HF-hosted corpus/index/embedding flows may prompt later or fall back to slower local paths.")
    if not openai_key:
        blocked.append("Benchmark evals will still run, but judging will fall back to the solver model instead of gpt-4o-mini.")

    console.print("[bold]What is enabled[/]")
    if shadeform_key:
        console.print("    •  Training is ready on Shadeform.")
    if together_key:
        console.print("    •  Together AI available for eval and serving.")
    if hf_token:
        console.print("    •  HF-hosted assets and embedding paths are available.")
    if openai_key:
        console.print("    •  Benchmark evals can use the OpenAI judge.")
    console.print()

    console.print("[bold]What may still be limited[/]")
    if blocked:
        for item in blocked:
            console.print(f"    •  {item}")
    else:
        console.print("    •  Nothing obvious is blocked by missing credentials.")
    console.print()


def _prompt_optional_key(
    *,
    stage_name: str,
    existing_key: str | None,
    config_key: str,
    console_hint: str,
    open_url: str,
    validate_fn,
    prompt_label: str,
    success_label: str,
    config: dict,
) -> None:
    console.rule(f"[bold]{stage_name}[/]", style="dim")
    console.print()
    console.print(f"    [dim]{console_hint}[/]")
    console.print()

    if existing_key:
        with console.status("    Validating...", spinner="dots"):
            result = validate_fn(existing_key)
        valid = result[0] if isinstance(result, tuple) else bool(result)
        detail = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
        if valid:
            console.print(f"    [green]✓[/]  {success_label}")
            config[config_key] = existing_key
            console.print()
            return
        console.print(f"    [yellow]–[/]  Existing key not usable: {detail or 'validation failed'}")
        console.print()

    idx = _arrow_select(console, [
        {"label": "Set up now", "hint": "Open the provider page and paste a key"},
        {"label": "Skip for now", "hint": "You can add this later with konash setup"},
    ])
    console.print()

    if idx == 1:
        return

    webbrowser.open(open_url)
    console.print(f"    [dim]{prompt_label}[/]")
    console.print()

    while True:
        entered = Prompt.ask("    Paste your API key or token", default="").strip()
        if not entered:
            console.print("    [dim]Skipped.[/]")
            console.print()
            return
        with console.status("    Validating...", spinner="dots"):
            result = validate_fn(entered)
        valid = result[0] if isinstance(result, tuple) else bool(result)
        detail = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
        if valid:
            console.print(f"    [green]✓[/]  {success_label}")
            config[config_key] = entered
            console.print()
            return
        console.print(f"    [red]✗[/]  {detail or 'Validation failed'}")
        if not Confirm.ask("    Try again?", default=False):
            console.print()
            return
        console.print()


def _show_training_plan(
    *,
    project: str,
    corpus: str,
    model: str,
    qa_pairs: int,
    rollouts: int,
    rollout_steps: int,
    iterations: int,
    backend: str,
) -> None:
    estimate = _estimate_training_plan(
        qa_pairs, rollouts, rollout_steps, iterations, backend,
    )
    oapl_backend = "Shadeform"
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold", justify="right", min_width=22)
    summary.add_column()
    summary.add_row("Project", project)
    summary.add_row("Corpus", corpus)
    summary.add_row("Model", model)
    summary.add_row("QA pairs", f"~{qa_pairs:,} / iteration")
    summary.add_row("Rollouts / example", str(rollouts))
    summary.add_row("Rollout steps", str(rollout_steps))
    summary.add_row("Iterations", str(iterations))
    synth_label = "Remote GPU"
    rollout_label = "Remote GPU"
    summary.add_row("Synthesis backend", synth_label)
    summary.add_row("Rollout backend", rollout_label)
    summary.add_row("OAPL backend", oapl_backend)
    summary.add_row(
        "Expected cost",
        f"~${estimate['total_cost']:,.0f}" if estimate["total_cost"] >= 1 else f"~${estimate['total_cost']:.2f}",
    )
    summary.add_row("Expected time", _format_duration(estimate["total_secs"]))
    console.print(summary)
    if iterations > 1:
        console.print()
        console.print("    [cyan]Bootstrapped multi-iteration training enabled.[/]")
    console.print()
    console.print("    [dim]Estimate includes API and/or GPU cost depending on backend; actual cost depends on corpus difficulty and provider pricing.[/]")
    console.print()


def _pick_eval_model(provider: str) -> str | None:
    if provider == "together":
        console.print()
        console.rule("[bold]Model[/]", style="dim")
        console.print()
        opts = [{"label": m.name, "hint": m.hint} for m in MODELS]
        opts.append({"label": "Custom", "hint": "Enter a model ID manually"})
        console.print("    [dim]Use arrow keys, press Enter to select[/]")
        console.print()
        idx = _arrow_select(console, opts)
        console.print()
        if idx < len(MODELS):
            return MODELS[idx].id
        return Prompt.ask("    Model ID")

    defaults = {
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "zhipu": "glm-4.5-air",
        "vllm": DEFAULT_MODEL,
    }
    return Prompt.ask("    Model ID", default=defaults.get(provider, DEFAULT_MODEL))


def _run_guided_eval_from_setup() -> None:
    eval_choices = [ds for ds in DATASETS if ds.benchmark is not None]

    console.rule("[bold]Benchmark[/]", style="dim")
    console.print()
    eval_options = [
        {"label": ds.name, "hint": ds.description}
        for ds in eval_choices
    ]
    console.print("    [dim]Use arrow keys, press Enter to select[/]")
    console.print()
    eval_idx = _arrow_select(console, eval_options)
    console.print()
    chosen = eval_choices[eval_idx]

    console.rule("[bold]Scope[/]", style="dim")
    console.print()
    scope_options = [
        {"label": "Smoke test", "hint": "Run 1 question first (recommended)"},
        {"label": "Short run", "hint": "Run 5 questions for a quick signal"},
        {"label": "Full benchmark", "hint": "Run the full registered benchmark"},
    ]
    console.print("    [dim]Use arrow keys, press Enter to select[/]")
    console.print()
    scope_idx = _arrow_select(console, scope_options)
    console.print()
    limit = None
    workers = None
    if scope_idx == 0:
        limit = 1
        workers = 1
    elif scope_idx == 1:
        limit = 5
        workers = 2

    providers = []
    if _get_together_key():
        providers.append(("together", "Together", "Recommended default backend"))
    if _get_hf_token():
        providers.append(("hf", "Hugging Face", "Use the HF router with your chosen model"))
    if _get_zhipu_key():
        providers.append(("zhipu", "Zhipu", "Use the native Zhipu backend"))

    console.rule("[bold]Provider[/]", style="dim")
    console.print()
    provider_options = [
        {"label": label, "hint": hint}
        for _, label, hint in providers
    ]
    console.print("    [dim]Use arrow keys, press Enter to select[/]")
    console.print()
    provider_idx = _arrow_select(console, provider_options)
    console.print()
    provider = providers[provider_idx][0]

    model = _pick_eval_model(provider)

    eval_args = ["--provider", provider]
    if model:
        eval_args.extend(["--model", model])
    if limit is not None:
        eval_args.extend(["--limit", str(limit)])
    if workers is not None:
        eval_args.extend(["--workers", str(workers)])

    cmd_eval(argparse.Namespace(
        benchmark=chosen.key,
        eval_args=eval_args,
    ))


# ---------------------------------------------------------------------------
# konash  (no args)
# ---------------------------------------------------------------------------

def cmd_launcher() -> None:
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
    grid.add_row("konash eval", "Run benchmark evals")
    grid.add_row("konash logs", "Stream GPU training logs")
    grid.add_row("konash stop", "Tear down GPU cluster")
    grid.add_row("konash projects", "List trained projects")
    grid.add_row("konash status", "Check configuration")
    console.print(grid)
    console.print()


def cmd_default() -> None:
    """Smart entrypoint: onboard first-time users, otherwise show launcher."""
    if not _get_together_key():
        console.print()
        console.print(
            "[dim]No core inference key found. Starting setup so KONASH is usable.[/]"
        )
        console.print()
        cmd_setup(argparse.Namespace())
        return

    cmd_launcher()


def _get_version() -> str:
    try:
        from konash import __version__
        return f"v{__version__}"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# konash setup
# ---------------------------------------------------------------------------

def _interactive_path_picker(con: Console) -> str:
    """Interactive path input with live directory suggestions."""
    import tty
    import termios
    import glob as _glob

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    current = os.path.expanduser("~/")
    selected = -1  # -1 = no suggestion selected, Enter confirms path

    def _get_matches(text):
        expanded = os.path.expanduser(text)
        if not expanded:
            return []
        matches = _glob.glob(expanded + "*")
        # Only show directories
        matches = [m for m in matches if os.path.isdir(m)]
        matches.sort()
        return matches[:8]  # max 8 suggestions

    def _render():
        matches = _get_matches(current)
        # Display line
        con.print(f"    Path: [bold]{current}[/]█", end="")
        con.print()
        # Suggestions
        for i, m in enumerate(matches):
            name = os.path.basename(m) + "/"
            if i == selected and selected >= 0:
                con.print(f"      [bold cyan]> {name}[/]")
            else:
                con.print(f"      [dim]  {name}[/]")
        return len(matches)

    def _clear(num_lines):
        for _ in range(num_lines + 1):  # +1 for the input line
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.flush()

    try:
        # Initial render
        num_shown = _render()

        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)

            if ch == "\r" or ch == "\n":
                matches = _get_matches(current)
                if selected >= 0 and selected < len(matches):
                    # Enter on a suggestion — navigate into it, keep picking
                    current = matches[selected] + "/"
                    selected = -1
                else:
                    # No suggestion selected — confirm current path
                    result = os.path.expanduser(current)
                    break

            elif ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt

            elif ch == "\x7f" or ch == "\x08":  # Backspace
                if current:
                    current = current[:-1]
                    selected = 0

            elif ch == "\t":  # Tab — autocomplete selection
                matches = _get_matches(current)
                if matches:
                    idx = selected if selected >= 0 else 0
                    if idx < len(matches):
                        current = matches[idx] + "/"
                    selected = -1

            elif ch == "\x1b":  # Escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    matches = _get_matches(current)
                    if ch3 == "A":  # Up
                        selected = max(-1, selected - 1)
                    elif ch3 == "B":  # Down
                        selected = min(len(matches) - 1, selected + 1) if matches else -1

            elif ch >= " " and ch <= "~":  # Printable char
                current += ch
                selected = -1

            # Re-render
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            _clear(num_shown)
            num_shown = _render()
            tty.setraw(fd)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # Clear suggestions
    _clear(num_shown)
    con.print(f"    Path: [bold]{result}[/]")

    # Validate
    while not os.path.exists(result):
        con.print(f"    [red]Path not found:[/] {result}")
        con.print(f"    [dim]Try again or Ctrl+C to cancel.[/]")
        con.print()
        result = Prompt.ask("    Path to documents")
        if result:
            result = os.path.expanduser(result)

    return result


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
    _animate_logo(console)
    console.print(f"    [bold]Welcome to KONASH[/]  [dim]{_get_version()}[/]")
    console.print(
        "    [dim]Train knowledge agents that search, retrieve, and reason.[/]"
    )
    console.print()
    console.print(
        "    Setup is staged so you can see exactly what is enabled:\n"
        "    core inference, cloud training, HF assets, and the optional eval judge."
    )
    console.print()
    console.rule(style="dim")
    console.print()

    config = _load_config()

    # ── Stage 1: Synthesis + Training (Shadeform — required) ────────
    console.rule("[bold]Synthesis + Training[/]", style="dim")
    console.print()
    console.print("    [dim]Shadeform provisions GPUs for synthesis (vLLM) and OAPL training.[/]")
    console.print()

    shadeform_key_setup = _get_shadeform_key()

    if shadeform_key_setup:
        console.print(f"    Key found: [dim]{_mask(shadeform_key_setup)}[/]")
        with console.status("    Validating...", spinner="dots"):
            valid = validate_shadeform_key(shadeform_key_setup)
        if valid:
            console.print("    [green]✓[/]  Connected to Shadeform")
            config["shadeform_api_key"] = shadeform_key_setup
        else:
            console.print("    [red]✗[/]  Invalid key")
            shadeform_key_setup = None

    if not shadeform_key_setup:
        idx = _arrow_select(console, [
            {"label": "Open shadeform.ai", "hint": "Sign up and grab an API key (free, takes 30 seconds)"},
            {"label": "I already have a key", "hint": ""},
        ])
        console.print()

        if idx == 0:
            webbrowser.open(SHADEFORM_KEYS_PAGE)
            console.print("    [dim]Settings → API Keys → create a key[/]")
            console.print()

        while True:
            shadeform_key_setup = Prompt.ask("    Paste your Shadeform API key")

            if not shadeform_key_setup:
                console.print("    [dim]No key entered. Try again or Ctrl+C to cancel.[/]")
                console.print()
                continue

            with console.status("    Validating...", spinner="dots"):
                valid = validate_shadeform_key(shadeform_key_setup)
            if valid:
                console.print("    [green]✓[/]  Connected to Shadeform")
                config["shadeform_api_key"] = shadeform_key_setup
                break
            else:
                console.print("    [red]✗[/]  Invalid key — check and try again")
                console.print()

    # ── Stage 2: Together AI (optional — for eval/serving) ───────────
    _prompt_optional_key(
        stage_name="Together AI (Optional)",
        existing_key=_get_together_key(),
        config_key="together_api_key",
        console_hint="Together AI is optional — used for eval benchmarks and model serving, not required for training.",
        open_url=TOGETHER_KEYS_PAGE,
        validate_fn=validate_together_key,
        prompt_label="Open together.ai, create an API key, then paste it here.",
        success_label="Together AI enabled",
        config=config,
    )

    # ── Stage 3: HF assets and embeddings ────────────────────────────
    _prompt_optional_key(
        stage_name="HF Assets And Embeddings",
        existing_key=_get_hf_token(),
        config_key="hf_token",
        console_hint="Hugging Face improves access to hosted corpora, prebuilt indexes, and query embedding paths.",
        open_url=HF_TOKENS_PAGE,
        validate_fn=lambda token: (bool(validate_hf_token(token)), "Invalid token"),
        prompt_label="Open Hugging Face tokens, create a read token, then paste it here.",
        success_label="HF assets and embeddings enabled",
        config=config,
    )

    # ── Stage 4: Optional eval judge ─────────────────────────────────
    _prompt_optional_key(
        stage_name="Optional Eval Judge",
        existing_key=_get_openai_key(),
        config_key="openai_api_key",
        console_hint="OpenAI is only used for the benchmark judge; eval still works without it using solver-as-judge fallback.",
        open_url=OPENAI_KEYS_PAGE,
        validate_fn=validate_openai_key,
        prompt_label="Open OpenAI API keys, create a key, then paste it here.",
        success_label="OpenAI eval judge enabled",
        config=config,
    )

    # ── Save and summarize ───────────────────────────────────────────
    _save_config(config)

    console.print()
    console.rule(style="dim")
    console.print()
    console.print("    [green]✓[/]  Setup summary")
    console.print()
    _render_capability_summary(config)

    next_options = [
        {"label": "Train", "hint": "Start training on your own documents"},
        {"label": "Eval", "hint": "Run a benchmark first to verify the system"},
        {"label": "Exit", "hint": "Come back later"},
    ]
    next_idx = _arrow_select(console, next_options)
    console.print()

    if next_idx == 0:
        train_args = argparse.Namespace(
            corpus=None,
            model=DEFAULT_MODEL,
            project=None,
            iterations=DEFAULT_SCALE_PRESET["iterations"],
            qa_pairs=DEFAULT_SCALE_PRESET["qa_pairs"],
            rollouts=DEFAULT_SCALE_PRESET["rollouts"],
            rollout_steps=DEFAULT_SCALE_PRESET["rollout_steps"],
            max_examples=None,
            lr=1e-5,
            chunk_size=512,
            synthesis_backend="auto",
            api_key=None,
            resume=False,
            fresh=False,
        )
        cmd_train(train_args)
    elif next_idx == 1:
        _run_guided_eval_from_setup()


def _download_dataset(key: str) -> str:
    """Download a dataset by key and return the corpus root path."""
    try:
        dataset = get_dataset(key)
    except KeyError:
        console.print(f"    [red]Unknown dataset:[/] {key}")
        sys.exit(1)

    console.print()
    return dataset.download(console=console)


# ---------------------------------------------------------------------------
# konash setup --check
# ---------------------------------------------------------------------------

def cmd_setup_check() -> None:
    """Non-interactive validation of all keys."""
    together_key = _get_together_key()
    shadeform_key = _get_shadeform_key()
    hf_token = _get_hf_token()
    openai_key = _get_openai_key()
    all_ok = True

    console.print()

    if together_key:
        with console.status("[cyan]Checking Together AI...", spinner="dots"):
            valid, err = validate_together_key(together_key)
        if valid:
            console.print("[green]✓[/] Together AI key valid")
        else:
            console.print(f"[red]✗[/] Together AI: {err}")
            all_ok = False
    else:
        console.print("[red]✗[/] Together AI key not found")
        all_ok = False

    if shadeform_key:
        with console.status("[cyan]Checking Shadeform...", spinner="dots"):
            valid = validate_shadeform_key(shadeform_key)
        if valid:
            console.print("[green]✓[/] Shadeform key valid")
        else:
            console.print("[red]✗[/] Shadeform key invalid")
            all_ok = False
    else:
        console.print("[yellow]–[/] Shadeform key not found (cloud training unavailable)")

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

    if openai_key:
        with console.status("[cyan]Checking OpenAI judge...", spinner="dots"):
            valid, err = validate_openai_key(openai_key)
        if valid:
            console.print("[green]✓[/] OpenAI key valid")
        else:
            console.print(f"[red]✗[/] OpenAI: {err}")
            all_ok = False
    else:
        console.print("[yellow]–[/] OpenAI key not found (optional)")

    console.print()
    if not all_ok:
        console.print("Run [cyan]konash setup[/] to fix.")
        console.print()
        sys.exit(1)


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

    # Prefer unified server (arena + traces + training) over trace_viewer alone
    server_path = os.path.join(os.path.dirname(__file__), "..", "tools", "server.py")
    app_path = os.path.join(os.path.dirname(__file__), "..", "tools", "trace_viewer", "app.py")
    port = int(os.environ.get("KONASH_PORT", 5050))
    if os.path.exists(server_path):
        launch_path = server_path
    elif os.path.exists(app_path):
        launch_path = app_path
    else:
        return  # Not available (pip install, no tools/)

    try:
        _web_ui_process = subprocess.Popen(
            [sys.executable, launch_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print(f"  [dim]Dashboard: http://localhost:{port}/training/[/]")
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
    if getattr(args, "preview_ui", False):
        _preview_training_ui(args)
        return

    try:
        from konash.api import Agent
        from konash.training.execution import plan_training_execution
    except ModuleNotFoundError as exc:
        _dependency_error(exc)

    try:
        plan = plan_training_execution(
            iterations=args.iterations,
            synthesis_rollout_backend="remote_full",
        )
    except ValueError as exc:
        console.print(f"\n[red]{exc}[/]\n")
        sys.exit(1)

    shadeform_key = _get_shadeform_key()
    if not shadeform_key:
        console.print(
            "\n[red]No Shadeform API key found.[/] Run [cyan]konash setup[/] first.\n"
        )
        sys.exit(1)

    legacy_default_archive = archive_legacy_default_project(PROJECTS_DIR)
    if legacy_default_archive is not None:
        console.print()
        console.print(
            f"    [yellow]Archived legacy default project to[/] "
            f"[dim]{legacy_default_archive}[/]"
        )
        console.print()

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
    remote_corpus_name = None
    if not corpus:
        options = [
            {"label": ds.name, "hint": ds.description}
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
            remote_corpus_name = ds.key
            corpus = _download_dataset(ds.key)
        else:
            corpus = _interactive_path_picker(console)

    # Resolve named datasets (e.g. "financebench") to local paths
    if corpus and not os.path.exists(corpus):
        try:
            remote_corpus_name = corpus
            corpus = _download_dataset(corpus)
        except (KeyError, SystemExit):
            console.print(f"    [red]Path not found and not a known dataset:[/] {corpus}")
            sys.exit(1)

    if not corpus:
        console.print(f"    [red]No corpus specified[/]")
        sys.exit(1)

    # ── Model + Scale (with ability to go back) ────────────────────
    model = args.model or DEFAULT_MODEL
    qa_pairs = args.qa_pairs if hasattr(args, "qa_pairs") else DEFAULT_SCALE_PRESET["qa_pairs"]
    rollouts = args.rollouts
    rollout_steps = args.rollout_steps
    iterations = args.iterations
    lr = args.lr
    chunk_size = args.chunk_size
    def _pick_model() -> str:
        console.print()
        console.rule("[bold]Model[/]", style="dim")
        console.print()
        opts = [{"label": m.name, "hint": m.hint} for m in MODELS]
        opts.append({"label": "Custom", "hint": "Enter a model ID manually"})
        console.print("    [dim]Use arrow keys, press Enter to select[/]")
        console.print()
        idx = _arrow_select(console, opts)
        console.print()
        if idx < len(MODELS):
            return MODELS[idx].id
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
        qp = IntPrompt.ask("    QA pairs to synthesize", default=DEFAULT_SCALE_PRESET["qa_pairs"])
        ro = IntPrompt.ask("    Rollouts per example", default=DEFAULT_SCALE_PRESET["rollouts"])
        rs = IntPrompt.ask("    Max steps per rollout", default=DEFAULT_SCALE_PRESET["rollout_steps"])
        it = IntPrompt.ask("    Training iterations", default=DEFAULT_SCALE_PRESET["iterations"])
        return qp, ro, rs, it

    # Skip interactive wizard in non-TTY environments (pipes, Colab, CI).
    # CLI args already have sensible defaults from argparse.
    _interactive = sys.stdin.isatty() and not getattr(args, "yes", False)
    if _interactive:
        model = _pick_model()
        qa_pairs, rollouts, rollout_steps, iterations = _pick_scale()

    # ── Summary + confirm (with go-back loop) ────────────────────────
    requested_project = args.project
    while _interactive:
        try:
            plan = plan_training_execution(
                iterations=iterations,
                synthesis_rollout_backend="remote_full",
            )
        except ValueError as exc:
            console.print(f"    [red]{exc}[/]")
            console.print()
            continue

        console.print()
        console.rule("[bold]Training Plan[/]", style="dim")
        console.print()
        dataset_spec = _build_training_dataset_spec(corpus)
        if requested_project and requested_project != LEGACY_DEFAULT_PROJECT:
            project_name = requested_project
        else:
            project_name = suggest_project_name(model, dataset_spec)
        _show_training_plan(
            project=project_name,
            corpus=corpus,
            model=model,
            qa_pairs=qa_pairs,
            rollouts=rollouts,
            rollout_steps=rollout_steps,
            iterations=iterations,
            backend=plan.synthesis_rollout_backend,
        )

        confirm_options = [
            {"label": "Approve plan", "hint": "Begin training with this execution plan"},
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
    try:
        plan = plan_training_execution(
            iterations=iterations,
            synthesis_rollout_backend="remote_full",
        )
    except ValueError as exc:
        console.print(f"\n[red]{exc}[/]\n")
        return

    synthesis_calls = max(1, qa_pairs // 8)
    dataset_spec = _build_training_dataset_spec(corpus)
    if requested_project and requested_project != LEGACY_DEFAULT_PROJECT:
        project_name = requested_project
    else:
        project_name = suggest_project_name(model, dataset_spec)
    run_config = TrainingRunConfig(
        synthesis_backend=plan.synthesis_rollout_backend,
        iterations=iterations,
        synthesis_calls=synthesis_calls,
        rollouts_per_example=rollouts,
        rollout_max_steps=rollout_steps,
    )
    display_name = f"{dataset_spec.display_label()} on {_short_model_name(model)}"
    assessment = assess_project_reuse(
        project=project_name,
        base_model=model,
        dataset_spec=dataset_spec,
        config=run_config,
        projects_dir=PROJECTS_DIR,
    )

    if _interactive:
        if assessment.resume_available:
            console.rule("[bold]Resume Checkpoint[/]", style="dim")
            console.print()
            console.print(
                f"    [bold]{project_name}[/] matches the same dataset and model."
            )
            if assessment.checkpoint.latest_phase:
                console.print(
                    f"    Last checkpoint: [dim]iteration {assessment.checkpoint.latest_iteration} · "
                    f"{assessment.checkpoint.latest_phase}[/]"
                )
            console.print()
            action = _arrow_select(console, [
                {"label": "Resume", "hint": "Continue from the compatible unfinished checkpoint"},
                {"label": "Start fresh in same project", "hint": "Archive current logs/checkpoints and restart clean"},
                {"label": "Start new project", "hint": "Keep the old project and create a fresh project name"},
                {"label": "Cancel", "hint": "Exit without training"},
            ])
            console.print()
            if action == 1:
                archive_project_run_state(project_name, projects_dir=PROJECTS_DIR)
            elif action == 2:
                project_name = suggest_project_name(
                    model, dataset_spec, projects_dir=PROJECTS_DIR, ensure_unique=True,
                )
            elif action == 3:
                console.print("    [dim]Cancelled.[/]\n")
                return
        elif assessment.project_exists:
            console.rule("[bold]Project State[/]", style="dim")
            console.print()
            if assessment.compatible_project:
                console.print(
                    f"    [bold]{project_name}[/] already exists for the same dataset and model."
                )
                hint = (
                    "The previous run completed."
                    if assessment.has_completed_training else
                    "The previous run is not safe to resume with this plan."
                )
                console.print(f"    [dim]{hint}[/]")
                console.print()
                action = _arrow_select(console, [
                    {"label": "Start fresh in same project", "hint": "Archive current logs/checkpoints and restart clean"},
                    {"label": "Start new project", "hint": "Keep the old project and create a fresh project name"},
                    {"label": "Cancel", "hint": "Exit without training"},
                ])
                console.print()
                if action == 0:
                    archive_project_run_state(project_name, projects_dir=PROJECTS_DIR)
                elif action == 1:
                    project_name = suggest_project_name(
                        model, dataset_spec, projects_dir=PROJECTS_DIR, ensure_unique=True,
                    )
                else:
                    console.print("    [dim]Cancelled.[/]\n")
                    return
            else:
                console.print(
                    f"    [bold]{project_name}[/] already exists but does [red]not[/] match "
                    "the same dataset and model."
                )
                console.print(
                    f"    [dim]Reason: {assessment.reason.replace('_', ' ')}[/]"
                )
                console.print()
                action = _arrow_select(console, [
                    {"label": "Start new project", "hint": "Create a new compatible project name"},
                    {"label": "Cancel", "hint": "Exit without training"},
                ])
                console.print()
                if action == 0:
                    project_name = suggest_project_name(
                        model, dataset_spec, projects_dir=PROJECTS_DIR, ensure_unique=True,
                    )
                else:
                    console.print("    [dim]Cancelled.[/]\n")
                    return
    else:
        if assessment.resume_available:
            if args.resume:
                pass
            elif args.fresh:
                archive_project_run_state(project_name, projects_dir=PROJECTS_DIR)
            else:
                console.print(
                    f"\n[red]Compatible unfinished checkpoint found for project {project_name}.[/] "
                    "Use [cyan]--resume[/], [cyan]--fresh[/], or a different [cyan]--project[/].\n"
                )
                return
        elif assessment.project_exists:
            if not assessment.compatible_project:
                console.print(
                    f"\n[red]Project {project_name} already exists with a different dataset or model.[/] "
                    "Choose a different [cyan]--project[/].\n"
                )
                return
            if args.fresh:
                archive_project_run_state(project_name, projects_dir=PROJECTS_DIR)
            elif requested_project:
                console.print(
                    f"\n[red]Project {project_name} already exists.[/] "
                    "Use [cyan]--fresh[/] to restart it or choose a different [cyan]--project[/].\n"
                )
                return
            else:
                project_name = suggest_project_name(
                    model, dataset_spec, projects_dir=PROJECTS_DIR, ensure_unique=True,
                )

    begin_training_run(
        project=project_name,
        display_name=display_name,
        base_model=model,
        dataset_spec=dataset_spec,
        config=run_config,
        projects_dir=PROJECTS_DIR,
    )

    console.print()
    console.rule("[bold]Execution Plan[/]", style="dim")
    console.print()
    _show_training_plan(
        project=project_name,
        corpus=corpus,
        model=model,
        qa_pairs=qa_pairs,
        rollouts=rollouts,
        rollout_steps=rollout_steps,
        iterations=iterations,
        backend=plan.synthesis_rollout_backend,
    )

    # Start the dashboard before provisioning so the training page is
    # already available while Shadeform boots and vLLM loads.
    _start_web_ui()  # no-op if already running from setup

    agent = Agent(
        base_model=model,
        corpus=corpus,
        project=project_name,
        display_name=display_name,
        hf_token=_get_hf_token(),
        remote_corpus_name=remote_corpus_name,
        chunk_size=chunk_size,
    )
    train_start = time.monotonic()
    try:
        agent.train(
            iterations=iterations,
            synthesis_calls=synthesis_calls,
            rollouts_per_example=rollouts,
            rollout_max_steps=rollout_steps,
            max_examples=args.max_examples,
            learning_rate=lr,
            synthesis_rollout_backend=plan.synthesis_rollout_backend,
            gpu_type=args.gpu_type,
            sleep_wake=_should_use_sleep_wake(model, args.gpu_type),
            keep_alive=args.keep_alive,
            verbose=True,
        )
    except Exception:
        mark_training_run_status(project_name, status="failed", projects_dir=PROJECTS_DIR)
        if args.keep_alive:
            console.print(
                "    [yellow]GPU kept alive[/] — fix the issue and re-run, "
                "or release with [bold]konash teardown[/]"
            )
        raise
    mark_training_run_status(project_name, status="completed", projects_dir=PROJECTS_DIR)
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
        {"label": "Exit", "hint": "Come back later with `konash eval` or another `konash train` run"},
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
        console.print(
            "    [dim]Run [cyan]konash eval[/] to benchmark, or rerun "
            "[cyan]konash train[/] to continue tuning this corpus.[/]"
        )
        console.print()


# ---------------------------------------------------------------------------
# konash eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    from konash.eval.entrypoint import main_for_benchmark

    argv = [sys.argv[0], *(args.eval_args or [])]
    old_argv = sys.argv
    try:
        sys.argv = argv
        main_for_benchmark(args.benchmark)
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# konash status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    console.print()
    console.print(f"[bold]KONASH[/]  [dim]{_get_version()}[/]  Status")
    console.print()
    console.rule(style="dim")
    console.print()

    config = _load_config()
    _render_capability_summary(config)
    console.print(f"    [dim]Config  {CONFIG_FILE}[/]")

    # Training status
    trained_projects = []
    if os.path.isdir(PROJECTS_DIR):
        for name in sorted(os.listdir(PROJECTS_DIR)):
            meta_path = os.path.join(PROJECTS_DIR, name, "checkpoints", "training_meta.json")
            if os.path.exists(meta_path):
                trained_projects.append((name, meta_path))
    if trained_projects:
        name, meta_path = trained_projects[-1]
        with open(meta_path) as f:
            meta = json.load(f)
        model = meta.get("base_model", "?")
        iters = meta.get("iterations", 0)
        console.print(f"    [green]✓[/]  Latest trained  {name} · {model} ({iters} iters)")
    else:
        console.print("    [dim]–[/]  Trained        no")

    console.print()

    if not _get_together_key():
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


def _preview_training_ui(args: argparse.Namespace) -> None:
    """Render a local mock training UI without hitting any APIs."""
    from rich.live import Live
    from rich.table import Table as _Table
    from rich.text import Text

    model = args.model or DEFAULT_MODEL
    qa_pairs = max(int(args.qa_pairs or DEFAULT_SCALE_PRESET["qa_pairs"]), 1)
    synthesis_calls = max(1, qa_pairs // 8)
    preview_seconds = max(int(args.preview_seconds or 18), 3)
    project_name = args.project or suggest_project_name(
        model,
        build_dataset_spec([args.corpus or "financebench"], aliases=_resolve_dataset_aliases()),
    )

    console.print()
    console.rule("[bold]Execution Plan[/]", style="dim")
    console.print()
    _show_training_plan(
        project=project_name,
        corpus=args.corpus or os.path.expanduser("~/.konash/corpora/financebench"),
        model=model,
        qa_pairs=qa_pairs,
        rollouts=args.rollouts,
        rollout_steps=args.rollout_steps,
        iterations=args.iterations,
        backend="remote_full",
    )
    console.print("  [dim]Preview mode · no API calls · no corpus indexing · no checkpoints[/]")
    console.print()
    console.rule("[bold]Iteration 1/1[/]", style="dim")
    console.print()
    console.print("  [bold]Phase 1: Synthesis + rollouts (Preview)[/]")
    console.print("  [green]✓[/]  Indexed 53,803 chunks")

    started = time.monotonic()
    latest_q = ""
    latest_a = ""

    def _build() -> _Table:
        nonlocal latest_q, latest_a
        elapsed = max(time.monotonic() - started, 0.0)
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]
        spinner = spinner_frames[int(elapsed * 8) % len(spinner_frames)]
        completed = min(int((elapsed / preview_seconds) * synthesis_calls), synthesis_calls - 1)
        pct = completed * 100 // synthesis_calls if synthesis_calls else 0
        bar_w = 32
        filled = bar_w * completed // synthesis_calls if synthesis_calls else 0
        bar = f"[cyan]{'━' * filled}[/][dim]{'─' * (bar_w - filled)}[/]"
        response_window = float(preview_seconds)
        progress_window = min(elapsed / response_window, 1.0)
        pulse_width = 10
        pulse_start = min(
            int((bar_w - pulse_width) * progress_window),
            max(bar_w - pulse_width, 0),
        )
        pulse_bar = (
            f"[dim]{'─' * pulse_start}[/]"
            f"[cyan]{'━' * pulse_width}[/]"
            f"[dim]{'─' * max(bar_w - pulse_start - pulse_width, 0)}[/]"
        )

        if elapsed < preview_seconds * 0.2:
            latest_q = ""
            latest_a = ""
            status_hint = "warming up request"
        elif elapsed < preview_seconds * 0.55:
            latest_q = "What two criteria must be met for license revenue to be recognized at sale?"
            latest_a = ""
            status_hint = "reading retrieved passages"
        elif elapsed < preview_seconds * 0.85:
            latest_q = "What two criteria must be met for license revenue to be recognized at sale?"
            latest_a = "The arrangement must provide a fixed license right and persuasive evidence of delivery."
            status_hint = "drafting grounded questions"
        else:
            latest_q = "What two criteria must be met for license revenue to be recognized at sale?"
            latest_a = "The arrangement must provide a fixed license right and persuasive evidence of delivery."
            status_hint = "finalizing proposal batch"

        outer = _Table(box=None, show_header=False, pad_edge=False, expand=True, padding=(0, 0))
        outer.add_row(Text("  Synthesizing QA pairs", style="bold"))
        outer.add_row(Text(""))
        outer.add_row(Text.from_markup(
            f"    {bar}  [dim]{completed}/{synthesis_calls}[/]  "
            f"[bold]0[/] pairs  [dim]{pct}%[/]"
        ))
        outer.add_row(Text.from_markup(
            f"    [cyan]{spinner}[/]  [dim]Calling {model} for 1 active synthesis request · {elapsed:.1f}s elapsed[/]"
        ))
        outer.add_row(Text.from_markup(
            f"    [dim]Model activity · {status_hint} · usual response window 20-90s[/]"
        ))
        outer.add_row(Text.from_markup(
            f"    {pulse_bar}  [dim]in-flight response progress[/]"
        ))
        if latest_q:
            outer.add_row(Text(""))
            outer.add_row(Text.from_markup(
                f"    [dim]Q:[/]  {latest_q[:90]}" + ("[dim]...[/]" if len(latest_q) > 90 else "")
            ))
        if latest_a:
            outer.add_row(Text.from_markup(
                f"    [dim]A:[/]  {latest_a[:90]}" + ("[dim]...[/]" if len(latest_a) > 90 else "")
            ))
        return outer

    with Live(_build(), console=console, refresh_per_second=8, transient=False) as live:
        while (time.monotonic() - started) < preview_seconds:
            live.update(_build())
            time.sleep(0.125)

    console.print(f"  [green]✓[/]  {min(8, qa_pairs)} QA pairs synthesized")
    console.print()
    console.print("  [dim]Preview complete. Re-run [cyan]konash train --preview-ui[/] after each style change.[/]")
    console.print()


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

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a knowledge agent.")
    p_train.add_argument(
        "corpus", nargs="?", default=None,
        help="Path to your documents folder (interactive if omitted).",
    )
    p_train.add_argument("--model", default=DEFAULT_MODEL, help="Model ID.")
    p_train.add_argument("--project", default=None, help="Project name. If omitted, KONASH generates one from the dataset and model.")
    p_train.add_argument(
        "--iterations", type=int, default=DEFAULT_SCALE_PRESET["iterations"],
        help=f"Training iterations (default: {DEFAULT_SCALE_PRESET['iterations']}, Quick scale).",
    )
    p_train.add_argument(
        "--qa-pairs", type=int, default=DEFAULT_SCALE_PRESET["qa_pairs"],
        help=f"QA pairs to synthesize per iteration (default: {DEFAULT_SCALE_PRESET['qa_pairs']}, Quick scale).",
    )
    p_train.add_argument(
        "--rollouts", type=int, default=DEFAULT_SCALE_PRESET["rollouts"],
        help=f"Rollouts per example (default: {DEFAULT_SCALE_PRESET['rollouts']}, Quick scale).",
    )
    p_train.add_argument(
        "--rollout-steps", type=int, default=DEFAULT_SCALE_PRESET["rollout_steps"],
        help=f"Max steps per rollout (default: {DEFAULT_SCALE_PRESET['rollout_steps']}, Quick scale; KARL BCP: 200).",
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
        "--gpu-type", default="H100",
        help="Shadeform GPU type for remote_full training (default: H100).",
    )
    p_train.add_argument(
        "--keep-alive", action="store_true",
        help="Keep the Shadeform GPU alive after training (success or failure). Use konash teardown to release later.",
    )
    p_train.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip interactive confirmation and start training immediately.",
    )
    p_train.add_argument(
        "--api-key", default=None,
        help="Shadeform API key (or run konash setup).",
    )
    p_train.add_argument(
        "--resume", action="store_true",
        help="Resume a compatible unfinished checkpoint when running non-interactively.",
    )
    p_train.add_argument(
        "--fresh", action="store_true",
        help="Archive the current project state and start fresh.",
    )
    p_train.add_argument(
        "--preview-ui", action="store_true",
        help="Preview the training terminal UI locally without API calls.",
    )
    p_train.add_argument(
        "--preview-seconds", type=int, default=18,
        help="How long the local training UI preview should run.",
    )
    p_train.set_defaults(func=cmd_train)

    # --- eval ---
    benchmark_choices = sorted(ds.key for ds in DATASETS if ds.benchmark is not None)
    p_eval = subparsers.add_parser("eval", help="Evaluate a registered benchmark.")
    p_eval.add_argument("benchmark", choices=benchmark_choices, help="Benchmark key.")
    p_eval.add_argument(
        "eval_args", nargs=argparse.REMAINDER,
        help="Additional benchmark eval args, e.g. --parallel 3 --limit 5.",
    )
    p_eval.set_defaults(func=cmd_eval)

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
