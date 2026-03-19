"""Shared eval harness for KONASH benchmarks.

Handles provider resolution, question evaluation, threaded execution,
results display, and JSON output for registered benchmarks.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from konash.benchmarks import BenchmarkConfig, get_dataset
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOGETHER_API_BASE = "https://api.together.xyz/v1"
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
OPENAI_API_BASE = "https://api.openai.com/v1"
HF_ROUTER_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL_TOGETHER = "zai-org/GLM-4.5-Air-FP8"
DEFAULT_MODEL_ZHIPU = "glm-4.5-air"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def resolve_provider(args) -> dict:
    """Resolve solver and judge provider settings from CLI args + config.

    Returns a dict with keys: provider, solver_api_base, solver_model,
    api_key, judge_api_base, judge_model, judge_key.
    """
    console = Console()

    config_path = os.path.expanduser("~/.konash/config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    provider = args.provider
    api_key = args.api_key
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or config.get("hf_token")
    )

    if args.api_base:
        provider = provider or "vllm"
        solver_api_base = args.api_base
        solver_model = args.model or DEFAULT_MODEL_TOGETHER
        api_key = api_key or "none"
    elif provider == "vllm":
        solver_api_base = args.api_base or "http://localhost:8000/v1"
        solver_model = args.model or DEFAULT_MODEL_TOGETHER
        api_key = api_key or "none"
    else:
        if not provider:
            zhipu_key = os.environ.get("ZHIPU_API_KEY") or config.get("zhipu_api_key")
            together_key = os.environ.get("TOGETHER_API_KEY") or config.get("together_api_key")
            if zhipu_key:
                provider = "zhipu"
                api_key = api_key or zhipu_key
            elif together_key:
                provider = "together"
                api_key = api_key or together_key
            elif hf_token and args.model:
                provider = "hf"
                api_key = api_key or hf_token

        if provider == "zhipu":
            solver_api_base = ZHIPU_API_BASE
            solver_model = args.model or DEFAULT_MODEL_ZHIPU
            api_key = api_key or os.environ.get("ZHIPU_API_KEY") or config.get("zhipu_api_key")
        elif provider == "hf":
            if not args.model:
                console.print("[red]Hugging Face eval requires --model.[/]")
                console.print("Set [cyan]--model org/model[/] and provide [cyan]HF_TOKEN[/] or run [cyan]konash setup[/].")
                sys.exit(1)
            solver_api_base = HF_ROUTER_API_BASE
            solver_model = args.model
            api_key = api_key or hf_token
        else:
            solver_api_base = TOGETHER_API_BASE
            solver_model = args.model or DEFAULT_MODEL_TOGETHER
            api_key = api_key or os.environ.get("TOGETHER_API_KEY") or config.get("together_api_key")

    if not api_key:
        console.print(
            "[red]No API key found.[/] Run [cyan]konash setup[/] or set "
            "ZHIPU_API_KEY / TOGETHER_API_KEY / HF_TOKEN."
        )
        sys.exit(1)

    # Resolve judge
    judge_key = args.judge_key or os.environ.get("OPENAI_API_KEY")
    if judge_key:
        judge_api_base = OPENAI_API_BASE
        judge_model = args.judge_model
    else:
        console.print("  [yellow]No OpenAI key — using solver model as judge (less accurate)[/]")
        console.print("  [dim]Set OPENAI_API_KEY for gpt-4o-mini judge (matches KARL paper)[/]")
        judge_api_base = solver_api_base
        judge_model = solver_model
        judge_key = api_key

    return {
        "provider": provider,
        "solver_api_base": solver_api_base,
        "solver_model": solver_model,
        "api_key": api_key,
        "judge_api_base": judge_api_base,
        "judge_model": judge_model,
        "judge_key": judge_key,
    }


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def add_common_args(parser):
    """Add the standard eval CLI arguments to an argparse parser."""
    parser.add_argument("--limit", type=int, default=None, help="Limit eval questions")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N eval questions")
    parser.add_argument("--parallel", type=int, default=0, help="Run an additional parallel-thinking eval with N rollouts")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent eval threads (default: 4)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Hide full traces during eval")
    parser.set_defaults(verbose=True)
    parser.add_argument("--model", default=None, help="Model ID (required for Hugging Face)")
    parser.add_argument("--provider", default=None, choices=["hf", "zhipu", "together", "vllm"], help="LLM provider (default: auto-detect from available keys)")
    parser.add_argument("--api-base", default=None, help="Custom API base URL (e.g. http://localhost:8000/v1 for vLLM)")
    parser.add_argument("--api-key", default=None, help="API key for solver provider")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model (default: gpt-4o-mini)")
    parser.add_argument("--judge-key", default=None, help="OpenAI API key for judge (or set OPENAI_API_KEY)")
    return parser


# ---------------------------------------------------------------------------
# Core eval functions
# ---------------------------------------------------------------------------

def eval_one_question(
    agent, question: str, reference: str, scorer, policy,
    bench_config: BenchmarkConfig,
    *, parallel_rollouts: int = 1, max_steps: int = 50,
    verbose_trace: bool = False,
    nuggets: Optional[List[str]] = None,
    console: Optional[Console] = None,
) -> dict:
    """Evaluate a single question. Returns result dict with trace.

    Parameters
    ----------
    agent : Agent
        The KONASH agent.
    question : str
        The question text.
    reference : str
        The reference answer string.
    scorer : NuggetScorer
        Scorer with LLM judge.
    policy : NuggetEvaluationPolicy
        Scoring policy.
    bench_config : BenchmarkConfig
        Benchmark configuration.
    parallel_rollouts : int
        Number of parallel rollouts.
    max_steps : int
        Max agent steps.
    verbose_trace : bool
        Print full trace.
    nuggets : list[str] | None
        Pre-defined nuggets (for FreshStack). None = auto-extract.
    console : Console | None
        Rich console for output.
    """
    if console is None:
        console = Console()

    top_k = bench_config.top_k

    t0 = time.monotonic()
    result = agent.solve(
        question,
        parallel_rollouts=parallel_rollouts,
        max_steps=max_steps,
        top_k=top_k,
        return_trace=True,
    )
    latency = time.monotonic() - t0

    answer = result["answer"] if isinstance(result, dict) else result
    full_response = result.get("full_response", answer) if isinstance(result, dict) else answer
    trajectory = result.get("trajectory", []) if isinstance(result, dict) else []

    # Count searches and extract queries
    num_searches = 0
    search_queries = []
    for step in trajectory:
        resp = step.get("agent_response") or {}
        tool_calls = resp.get("tool_calls", [])
        if tool_calls:
            num_searches += 1
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                q = args.get("query", str(args)) if isinstance(args, dict) else str(args)
                search_queries.append(q)

    if verbose_trace:
        console.print(f"\n  [bold cyan]Q:[/] {question}")
        console.print(f"  [bold]Ref:[/] {reference}")
        for i, step in enumerate(trajectory):
            resp = step.get("agent_response") or {}
            content = resp.get("content", "") or ""
            reasoning = resp.get("reasoning_content", "") or resp.get("reasoning", "") or ""
            tool_calls = resp.get("tool_calls", [])
            tool_results = step.get("tool_results", [])
            done = step.get("done", False)

            console.print(f"\n  [dim]--- Step {i} {'[FINAL]' if done else ''} ---[/]")
            if reasoning:
                console.print(f"  [yellow]Thinking:[/] {reasoning}")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    console.print(f"  [blue]Search:[/] {fn.get('arguments', '')}")
            if tool_results:
                for tr in tool_results:
                    tr_content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                    console.print(f"  [dim]Results ({len(tr_content)} chars):[/] {tr_content}")
            if content and not tool_calls:
                console.print(f"  [green]Answer:[/] {content}")

        console.print(f"\n  [bold]Final answer:[/] {answer or '(empty)'}")
        console.print()

    # Send full response + extracted answer to the judge for maximum context.
    if full_response and full_response != answer:
        judge_text = f"{full_response}\n\nExtracted Answer: {answer}"
    else:
        judge_text = answer or ""

    # Set question context (benchmark-specific or plain question)
    if bench_config.get_judge_context:
        scorer.judge.question_context = bench_config.get_judge_context(question, bench_config)
    else:
        scorer.judge.question_context = question

    # Score — pass nuggets= if pre-defined (FreshStack)
    score_kwargs = {}
    if nuggets is not None:
        score_kwargs["nuggets"] = nuggets
    score_result = scorer.score(judge_text, reference, policy=policy, **score_kwargs)
    score = score_result["score"]

    if verbose_trace:
        console.print(f"  [bold]Judge score:[/] {score}  nuggets: {score_result.get('nuggets', [])}")
        if hasattr(scorer.judge, 'last_prompt'):
            console.print(f"  [dim]Judge input (answer):[/] {judge_text}")
        if hasattr(scorer.judge, 'last_raw_response') and scorer.judge.last_raw_response:
            console.print(f"  [dim]Judge reasoning:[/] {scorer.judge.last_raw_response}")
        console.print()

    result_dict = {
        "question": question,
        "reference": reference,
        "answer": answer,
        "score": score,
        "latency": latency,
        "trajectory": trajectory,
        "num_steps": len(trajectory),
        "num_searches": num_searches,
        "search_queries": search_queries,
    }

    # Include nugget details if available
    if score_result.get("nugget_scores"):
        result_dict["nugget_scores"] = score_result["nugget_scores"]
    if score_result.get("nuggets"):
        result_dict["nuggets"] = score_result["nuggets"]
        result_dict["num_nuggets"] = len(score_result["nuggets"])

    return result_dict


def run_eval(
    agent, questions: list[dict], scorer, policy, label: str,
    bench_config: BenchmarkConfig,
    *, parallel_rollouts: int = 1, workers: int = 4, verbose_trace: bool = False,
    console: Optional[Console] = None,
) -> dict:
    """Run eval on all questions with thread-level parallelism."""
    if console is None:
        console = Console()

    results = [None] * len(questions)
    completed = [0]
    lock = threading.Lock()
    total_start = time.monotonic()

    mode_label = f"N={parallel_rollouts}" if parallel_rollouts > 1 else "single"

    def _run(idx: int, q: dict) -> None:
        question_text = bench_config.get_question_text(q)
        reference = bench_config.get_reference(q)
        nuggets_list = bench_config.get_nuggets(q) if bench_config.get_nuggets else None

        r = eval_one_question(
            agent, question_text, reference, scorer, policy,
            bench_config,
            parallel_rollouts=parallel_rollouts,
            verbose_trace=verbose_trace,
            nuggets=nuggets_list,
            console=console,
        )
        results[idx] = r
        with lock:
            completed[0] += 1
            n = completed[0]
            elapsed = time.monotonic() - total_start
            avg_per_q = elapsed / n
            remaining = (len(questions) - n) * avg_per_q / max(workers, 1)
            eta_min = int(remaining // 60)
            eta_sec = int(remaining % 60)

            done_scores = [results[i]["score"] for i in range(len(results)) if results[i] is not None]
            running_avg = sum(done_scores) / len(done_scores) if done_scores else 0
            running_correct = sum(1 for s in done_scores if s >= 0.6)
            running_acc = running_correct / len(done_scores) if done_scores else 0

            marker = "[green]✓[/]" if r["score"] >= 0.6 else "[yellow]~[/]" if r["score"] > 0 else "[red]✗[/]"
            n_steps = r.get("num_steps", 0)
            n_searches = r.get("num_searches", 0)
            queries = r.get("search_queries", [])
            answer_preview = (r.get("answer") or "(empty)")[:120].replace("\n", " ")

            console.print(
                f"\n  [dim]{n}/{len(questions)}[/]  {marker}  {r['score']:.2f}  "
                f"[dim]{r['latency']:.1f}s[/]  "
                f"[bold]{running_avg:.1%}[/] avg  "
                f"[dim]({running_correct}/{n} ≥0.6)[/]  "
                f"[dim]ETA {eta_min}m{eta_sec:02d}s[/]"
            )

            # Question preview
            q_preview = question_text[:100]
            console.print(f"  [dim]Q:[/] {q_preview}")

            # Optional per-benchmark detail line
            if bench_config.get_progress_detail:
                detail = bench_config.get_progress_detail(r)
                if detail:
                    console.print(f"  {detail}")
            else:
                console.print(f"  [dim]Steps:[/] {n_steps} ({n_searches} searches)")

            if queries:
                for sq in queries[:3]:
                    console.print(f"    [blue]→[/] [dim]{sq[:80]}[/]")
                if len(queries) > 3:
                    console.print(f"    [dim]... +{len(queries)-3} more[/]")
            console.print(f"  [dim]Answer:[/] {answer_preview}")

            # Reference preview
            if bench_config.get_progress_ref_display:
                ref_preview = bench_config.get_progress_ref_display(q)
                console.print(f"  [dim]Ref:[/] {ref_preview}")
            else:
                console.print(f"  [dim]Ref:[/] {reference[:120]}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            fut.result()  # propagate exceptions

    total_time = time.monotonic() - total_start
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    correct = sum(1 for s in scores if s >= 0.6)

    eval_result = {
        "label": label,
        "mode": mode_label,
        "parallel_rollouts": parallel_rollouts,
        "avg_score": avg_score,
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results) if results else 0,
        "avg_latency": avg_latency,
        "total_time": total_time,
        "results": results,
    }

    # Compute benchmark-specific extra metrics
    if bench_config.get_extra_output:
        eval_result.update(bench_config.get_extra_output(eval_result))

    return eval_result


# ---------------------------------------------------------------------------
# Results display and saving
# ---------------------------------------------------------------------------

def display_results(
    baseline: dict, parallel: Optional[dict], bench_config: BenchmarkConfig,
    args, console: Console,
) -> None:
    """Display the summary results table."""
    console.print()
    console.rule("[bold]Results[/]", style="dim")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Score", justify="right")

    for col_name, _ in bench_config.extra_table_columns:
        table.add_column(col_name, justify="right")

    table.add_column("Avg Latency", justify="right")
    table.add_column("Total Time", justify="right")

    def _row_values(result: dict) -> list:
        vals = [
            f"{result['accuracy']:.0%}",
            f"{result['avg_score']:.3f}",
        ]
        for _, key in bench_config.extra_table_columns:
            val = result.get(key)
            if val is not None:
                vals.append(f"{val:.3f}")
            else:
                vals.append("")
        vals.extend([
            f"{result['avg_latency']:.1f}s",
            f"{result['total_time']:.0f}s",
        ])
        return vals

    table.add_row("Single rollout", *_row_values(baseline))

    if parallel:
        table.add_row(
            f"Parallel (N={args.parallel})",
            *_row_values(parallel),
        )
        delta = parallel["avg_score"] - baseline["avg_score"]
        sign = "+" if delta >= 0 else ""
        n_extra = len(bench_config.extra_table_columns)
        table.add_row(
            "Delta",
            "",
            f"{sign}{delta:.3f}",
            *[""] * (n_extra + 2),
            style="dim",
        )

    console.print(table)
    console.print()

    if bench_config.paper_target:
        console.print(f"  [dim]{bench_config.paper_target}[/]")
        console.print()


def save_results(
    baseline: dict, parallel: Optional[dict],
    bench_config: BenchmarkConfig,
    solver_model: str, provider: str,
    console: Console,
) -> str:
    """Save results to a timestamped JSON file. Returns the output path."""
    results_dir = "eval_results"
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "model": solver_model,
        "benchmark": bench_config.name,
        "num_questions": baseline["total"],
        "single": {k: v for k, v in baseline.items() if k != "results"},
        "single_details": baseline["results"],
    }
    if parallel:
        output["parallel"] = {k: v for k, v in parallel.items() if k != "results"}
        output["parallel_details"] = parallel["results"]

    # Add any extra static output fields
    output.update(bench_config.extra_output_fields)

    ts = datetime.now(timezone.utc)
    output["run_id"] = f"{bench_config.benchmark_key}_{solver_model.split('/')[-1]}_{ts.strftime('%Y%m%d_%H%M%S')}"
    output["timestamp"] = ts.isoformat()
    output["provider"] = provider

    safe_model = solver_model.split("/")[-1].lower().replace(" ", "_")
    timestamp = ts.strftime("%Y%m%d_%H%M%S")
    filename = f"{bench_config.benchmark_key}_{safe_model}_{timestamp}.json"
    out_path = os.path.join(results_dir, filename)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save/update a "latest" symlink for backward compat
    latest_path = os.path.join(results_dir, f"{bench_config.benchmark_key}_eval.json")
    try:
        if os.path.islink(latest_path):
            os.unlink(latest_path)
        elif os.path.exists(latest_path):
            pass  # don't overwrite, the new file is already saved
        os.symlink(os.path.basename(out_path), latest_path)
    except OSError:
        pass  # symlink might fail on some systems

    console.print(f"  [dim]Results saved to {out_path}[/]")
    return out_path


# ---------------------------------------------------------------------------
# EvalHarness — main orchestrator
# ---------------------------------------------------------------------------

class EvalHarness:
    """Orchestrates a full benchmark evaluation run.

    Usage from a benchmark entrypoint::

        harness = EvalHarness(bench_config)
        harness.run()
    """

    def __init__(self, bench_config: BenchmarkConfig):
        self.bench_config = bench_config
        self.console = Console()

    def run(self, extra_args_fn: Optional[Callable] = None):
        """Parse args, set up components, and run the evaluation.

        Parameters
        ----------
        extra_args_fn : callable | None
            Optional function that takes an argparse parser and adds
            benchmark-specific arguments (e.g. --train for FinanceBench).
        """
        import argparse

        parser = argparse.ArgumentParser(
            description=f"Evaluate KONASH on {self.bench_config.name}"
        )
        add_common_args(parser)
        if extra_args_fn:
            extra_args_fn(parser)
        args = parser.parse_args()

        bc = self.bench_config

        # Resolve provider
        prov = resolve_provider(args)
        provider = prov["provider"]
        solver_api_base = prov["solver_api_base"]
        solver_model = prov["solver_model"]
        api_key = prov["api_key"]
        judge_api_base = prov["judge_api_base"]
        judge_model = prov["judge_model"]
        judge_key = prov["judge_key"]

        # Download corpus
        self.console.print()
        self.console.print(f"[bold]KONASH {bc.name} Evaluation[/]")
        self.console.print()
        self.console.rule(style="dim")

        dataset = get_dataset(bc.benchmark_key)
        corpus_dir = dataset.download(console=self.console)

        # Load eval questions
        eval_path = dataset.eval_questions_path(corpus_dir)
        if not os.path.exists(eval_path):
            self.console.print(f"[red]No eval questions at {eval_path}[/]")
            sys.exit(1)
        with open(eval_path) as f:
            questions = json.load(f)

        if args.offset:
            questions = questions[args.offset:]
        if args.limit:
            questions = questions[:args.limit]
        self.console.print(f"  {len(questions)} eval questions loaded")
        self.console.print()

        # Set up scorer with LLM judge
        from konash.eval.nuggets import NuggetScorer, NuggetPolicyRegistry, LLMNuggetJudge
        policy = NuggetPolicyRegistry.get(bc.policy_name)

        from konash.api import _OpenAILLMClient
        judge_client = _OpenAILLMClient(
            api_base=judge_api_base,
            api_key=judge_key,
            model=judge_model,
            temperature=0.0,
        )
        judge = LLMNuggetJudge(llm_fn=judge_client.generate)
        scorer = NuggetScorer(judge=judge)
        self.console.print(f"  Solver: {solver_model} via {provider}")
        self.console.print(f"  Judge: {judge_model} via {judge_api_base.split('//')[1].split('/')[0]}")

        # Create agent
        from konash.api import Agent
        agent = Agent(
            base_model=solver_model,
            corpus=corpus_dir,
            project=bc.project_name,
            api_base=solver_api_base,
            api_key=api_key,
        )

        # Optional training (FinanceBench)
        if hasattr(args, "train") and args.train:
            self.console.rule("[bold]Training[/]  (Quick scale)", style="dim")
            self.console.print()
            train_start = time.monotonic()
            agent.train(
                iterations=1,
                synthesis_calls=50,
                rollouts_per_example=8,
                rollout_max_steps=30,
                verbose=True,
            )
            self.console.print(f"  Training took {time.monotonic() - train_start:.0f}s")
            self.console.print()

        # Determine worker count
        if provider == "zhipu" and args.workers > 4:
            eval_workers = 4
            self.console.print(f"  [dim]Zhipu: throttling to {eval_workers} workers (limit 5)[/]")
        else:
            eval_workers = args.workers

        # Single rollout eval
        self.console.rule("[bold]Single rollout[/]", style="dim")
        self.console.print()

        baseline = run_eval(
            agent, questions, scorer, policy, "Single rollout",
            bc,
            parallel_rollouts=1, workers=eval_workers,
            verbose_trace=args.verbose,
            console=self.console,
        )
        self.console.print()
        self.console.print(
            f"  [bold]Single:[/]  {baseline['correct']}/{baseline['total']} "
            f"({baseline['accuracy']:.0%})  score {baseline['avg_score']:.3f}  "
            f"{baseline['total_time']:.0f}s total"
        )
        self.console.print()

        # Parallel thinking eval
        parallel = None
        if not args.single_only and args.parallel > 0:
            self.console.rule(f"[bold]Parallel thinking[/]  (N={args.parallel})", style="dim")
            self.console.print()

            parallel = run_eval(
                agent, questions, scorer, policy, f"Parallel (N={args.parallel})",
                bc,
                parallel_rollouts=args.parallel, workers=eval_workers,
                verbose_trace=args.verbose,
                console=self.console,
            )
            self.console.print()
            self.console.print(
                f"  [bold]Parallel:[/]  {parallel['correct']}/{parallel['total']} "
                f"({parallel['accuracy']:.0%})  score {parallel['avg_score']:.3f}  "
                f"{parallel['total_time']:.0f}s total"
            )
            self.console.print()

        # Display and save
        display_results(baseline, parallel, bc, args, self.console)
        save_results(baseline, parallel, bc, solver_model, provider, self.console)
        self.console.print()

        return baseline, parallel
