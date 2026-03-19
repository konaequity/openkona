#!/usr/bin/env python3
"""Evaluate KONASH on FreshStack (LangChain domain) — procedural technical reasoning.

FreshStack tests procedural reasoning over technical documentation and source code.
Each question has pre-defined nuggets (fixed_nuggets mode) with ground truth
relevance labels. The KARL paper evaluates on the LangChain domain (203 questions,
49,514 docs).

Usage:
    # Full eval (203 questions, single rollout):
    python scripts/eval_freshstack.py --single-only

    # Quick dev test:
    python scripts/eval_freshstack.py --limit 5 --single-only --verbose

    # With parallel thinking:
    python scripts/eval_freshstack.py --parallel 3
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

TOGETHER_API_BASE = "https://api.together.xyz/v1"
ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL_TOGETHER = "zai-org/GLM-4.5-Air-FP8"
DEFAULT_MODEL_ZHIPU = "glm-4.5-air"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
FRESHSTACK_DOMAIN = "langchain"


def eval_one_question(
    agent, question: str, nuggets: list[dict], scorer, policy,
    *, parallel_rollouts: int = 1, max_steps: int = 50, top_k: int = 10,
    verbose_trace: bool = False,
) -> dict:
    """Evaluate a single FreshStack question."""
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

    # Count searches
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
        console.print(f"\n  [bold cyan]Q:[/] {question[:200]}")
        nugget_texts = [n["text"] for n in nuggets]
        console.print(f"  [bold]Nuggets ({len(nuggets)}):[/] {nugget_texts}")
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

    # Score using pre-defined nuggets
    # FreshStack has fixed nuggets — use the nugget text directly
    nugget_texts = [n["text"] for n in nuggets]

    # Send full response to judge
    if full_response and full_response != answer:
        judge_text = f"{full_response}\n\nExtracted Answer: {answer}"
    else:
        judge_text = answer or ""

    scorer.judge.question_context = question
    score_result = scorer.score(judge_text, "; ".join(nugget_texts), policy=policy,
                                nuggets=nugget_texts)
    score = score_result["score"]

    if verbose_trace:
        console.print(f"  [bold]Judge score:[/] {score:.3f}  nuggets: {nugget_texts}")
        console.print(f"  [bold]Nugget scores:[/] {score_result.get('nugget_scores', [])}")
        if hasattr(scorer.judge, 'last_prompt'):
            console.print(f"  [dim]Judge input (answer):[/] {judge_text}")
        if hasattr(scorer.judge, 'last_raw_response') and scorer.judge.last_raw_response:
            console.print(f"  [dim]Judge reasoning:[/] {scorer.judge.last_raw_response}")
        console.print()

    return {
        "question": question,
        "nuggets": nugget_texts,
        "answer": answer,
        "score": score,
        "nugget_scores": score_result.get("nugget_scores", []),
        "num_nuggets": len(nugget_texts),
        "latency": latency,
        "trajectory": trajectory,
        "num_steps": len(trajectory),
        "num_searches": num_searches,
        "search_queries": search_queries,
    }


def run_eval(
    agent, questions: list[dict], scorer, policy, label: str,
    *, parallel_rollouts: int = 1, workers: int = 4, verbose_trace: bool = False,
) -> dict:
    """Run eval on all questions with thread-level parallelism."""
    import threading

    results = [None] * len(questions)
    completed = [0]
    lock = threading.Lock()
    total_start = time.monotonic()

    mode_label = f"N={parallel_rollouts}" if parallel_rollouts > 1 else "single"

    def _run(idx: int, q: dict) -> None:
        r = eval_one_question(
            agent, q["question"], q["nuggets"], scorer, policy,
            parallel_rollouts=parallel_rollouts,
            verbose_trace=verbose_trace,
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

            marker = "[green]✓[/]" if r["score"] >= 0.6 else "[yellow]~[/]" if r["score"] > 0.3 else "[red]✗[/]"
            n_nuggets = r.get("num_nuggets", 0)
            found = sum(1 for s in r.get("nugget_scores", []) if s >= 0.6)

            console.print(
                f"\n  [dim]{n}/{len(questions)}[/]  {marker}  {r['score']:.2f}  "
                f"[dim]{r['latency']:.1f}s[/]  "
                f"[bold]{running_avg:.1%}[/] avg  "
                f"[dim]({running_correct}/{n} ≥0.6)[/]  "
                f"[dim]ETA {eta_min}m{eta_sec:02d}s[/]"
            )
            console.print(f"  [dim]Q:[/] {q['question_title'][:100]}")
            console.print(f"  [dim]Nuggets:[/] {found}/{n_nuggets} supported")
            if r.get("search_queries"):
                for sq in r["search_queries"][:3]:
                    console.print(f"    [blue]→[/] [dim]{sq[:80]}[/]")
            console.print(f"  [dim]Answer:[/] {(r.get('answer') or '(empty)')[:120]}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            fut.result()

    total_time = time.monotonic() - total_start
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    correct = sum(1 for s in scores if s >= 0.6)

    return {
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KONASH on FreshStack (LangChain)")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval questions")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N eval questions")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel rollouts (0 to skip)")
    parser.add_argument("--single-only", action="store_true", help="Only run single rollout")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent eval threads")
    parser.add_argument("--verbose", action="store_true", help="Print full trace")
    parser.add_argument("--model", default=None, help="Model ID")
    parser.add_argument("--provider", default=None, choices=["zhipu", "together", "vllm"])
    parser.add_argument("--api-base", default=None, help="Custom API base URL")
    parser.add_argument("--api-key", default=None, help="API key for solver")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-key", default=None, help="OpenAI API key for judge")
    args = parser.parse_args()

    # Resolve provider
    config_path = os.path.expanduser("~/.konash/config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    provider = args.provider
    api_key = args.api_key

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

        if provider == "zhipu":
            solver_api_base = ZHIPU_API_BASE
            solver_model = args.model or DEFAULT_MODEL_ZHIPU
            api_key = api_key or os.environ.get("ZHIPU_API_KEY") or config.get("zhipu_api_key")
        else:
            solver_api_base = TOGETHER_API_BASE
            solver_model = args.model or DEFAULT_MODEL_TOGETHER
            api_key = api_key or os.environ.get("TOGETHER_API_KEY") or config.get("together_api_key")

    if not api_key:
        console.print("[red]No API key found.[/]")
        sys.exit(1)

    judge_key = args.judge_key or os.environ.get("OPENAI_API_KEY")
    if judge_key:
        judge_api_base = OPENAI_API_BASE
        judge_model = args.judge_model
    else:
        console.print("  [yellow]No OpenAI key — using solver model as judge[/]")
        judge_api_base = solver_api_base
        judge_model = solver_model
        judge_key = api_key

    # Step 1: Download FreshStack
    console.print()
    console.print("[bold]KONASH FreshStack Evaluation (LangChain)[/]")
    console.print()
    console.rule(style="dim")

    from konash.download import download_freshstack as _download
    corpus_dir = _download(console=console)

    eval_path = os.path.join(corpus_dir, "eval_questions.json")
    with open(eval_path) as f:
        questions = json.load(f)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]
    console.print(f"  {len(questions)} eval questions loaded")

    avg_nuggets = sum(len(q["nuggets"]) for q in questions) / len(questions)
    console.print(f"  Scoring: fixed_nuggets (avg {avg_nuggets:.1f} nuggets/Q)")
    console.print()

    # Step 2: Set up scorer
    from konash.eval.nuggets import NuggetScorer, NuggetPolicyRegistry, LLMNuggetJudge
    policy = NuggetPolicyRegistry.get("FreshStack")

    from konash.api import _OpenAILLMClient
    judge_client = _OpenAILLMClient(
        api_base=judge_api_base,
        api_key=judge_key,
        model=judge_model,
        temperature=0.0,
    )
    judge = LLMNuggetJudge(llm_fn=judge_client.generate)
    scorer = NuggetScorer(judge=judge)
    console.print(f"  Solver: {solver_model} via {provider}")
    console.print(f"  Judge: {judge_model} via {judge_api_base.split('//')[1].split('/')[0]}")

    # Step 3: Create agent (k=10 for FreshStack, not k=20)
    from konash.api import Agent
    agent = Agent(
        base_model=solver_model,
        corpus=corpus_dir,
        project="eval-freshstack",
        api_base=solver_api_base,
        api_key=api_key,
    )

    # Step 4: Single rollout eval
    console.rule("[bold]Single rollout[/]", style="dim")
    console.print()

    if provider == "zhipu" and args.workers > 4:
        eval_workers = 4
    else:
        eval_workers = args.workers

    baseline = run_eval(
        agent, questions, scorer, policy, "Single rollout",
        parallel_rollouts=1, workers=eval_workers,
        verbose_trace=args.verbose,
    )
    console.print()
    console.print(
        f"  [bold]Single:[/]  {baseline['correct']}/{baseline['total']} "
        f"(≥0.6: {baseline['accuracy']:.0%})  "
        f"avg score {baseline['avg_score']:.3f}  "
        f"{baseline['total_time']:.0f}s total"
    )
    console.print()

    # Step 5: Parallel thinking
    parallel = None
    if not args.single_only and args.parallel > 0:
        console.rule(f"[bold]Parallel thinking[/]  (N={args.parallel})", style="dim")
        console.print()
        parallel = run_eval(
            agent, questions, scorer, policy, f"Parallel (N={args.parallel})",
            parallel_rollouts=args.parallel, workers=eval_workers,
            verbose_trace=args.verbose,
        )
        console.print()
        console.print(
            f"  [bold]Parallel:[/]  {parallel['correct']}/{parallel['total']} "
            f"(≥0.6: {parallel['accuracy']:.0%})  "
            f"avg score {parallel['avg_score']:.3f}  "
            f"{parallel['total_time']:.0f}s total"
        )
        console.print()

    # Step 6: Summary
    console.print()
    console.rule("[bold]Results[/]", style="dim")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("Accuracy (≥0.6)", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Total Time", justify="right")

    table.add_row(
        "Single rollout",
        f"{baseline['accuracy']:.0%}",
        f"{baseline['avg_score']:.3f}",
        f"{baseline['avg_latency']:.1f}s",
        f"{baseline['total_time']:.0f}s",
    )
    if parallel:
        table.add_row(
            f"Parallel (N={args.parallel})",
            f"{parallel['accuracy']:.0%}",
            f"{parallel['avg_score']:.3f}",
            f"{parallel['avg_latency']:.1f}s",
            f"{parallel['total_time']:.0f}s",
        )

    console.print(table)
    console.print()
    console.print(f"  [dim]KARL paper target (GLM 4.5 Air base): 52.9% avg score[/]")
    console.print()

    # Step 7: Save results
    from datetime import datetime, timezone

    results_dir = "eval_results"
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "model": solver_model,
        "benchmark": "FreshStack",
        "domain": FRESHSTACK_DOMAIN,
        "num_questions": len(questions),
        "single": {k: v for k, v in baseline.items() if k != "results"},
        "single_details": baseline["results"],
    }
    if parallel:
        output["parallel"] = {k: v for k, v in parallel.items() if k != "results"}
        output["parallel_details"] = parallel["results"]

    output["run_id"] = f"freshstack_{solver_model.split('/')[-1]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output["timestamp"] = datetime.now(timezone.utc).isoformat()
    output["provider"] = provider

    safe_model = solver_model.split("/")[-1].lower().replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"freshstack_{safe_model}_{timestamp}.json"
    out_path = os.path.join(results_dir, filename)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"  [dim]Results saved to {out_path}[/]")
    console.print()


if __name__ == "__main__":
    main()
