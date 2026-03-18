#!/opt/anaconda3/bin/python3
"""Evaluate KONASH on QAMPARI — exhaustive entity search over Wikipedia.

QAMPARI tests exhaustive entity retrieval: finding ALL entities satisfying
a condition across 250K+ Wikipedia sentence-level chunks. Each question has
~14.7 answer entities on average (entity-per-nugget scoring).

Usage:
    # Full eval (1,000 questions, single rollout):
    python scripts/eval_qampari.py --single-only

    # Quick dev test (20 questions):
    python scripts/eval_qampari.py --limit 20 --single-only --verbose

    # With parallel thinking:
    python scripts/eval_qampari.py --parallel 3

    # Specific question range:
    python scripts/eval_qampari.py --offset 100 --limit 10 --verbose
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


def load_eval_questions(corpus_dir: str) -> list[dict]:
    eval_path = os.path.join(corpus_dir, "eval_questions.json")
    if not os.path.exists(eval_path):
        console.print(f"[red]No eval questions at {eval_path}[/]")
        sys.exit(1)
    with open(eval_path) as f:
        questions = json.load(f)
    return questions


def eval_one_question(
    agent, question: str, reference_answers: list[str], scorer, policy,
    *, parallel_rollouts: int = 1, max_steps: int = 50, top_k: int = 20,
    verbose_trace: bool = False,
) -> dict:
    """Evaluate a single QAMPARI question. Returns result dict with trace."""
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
        console.print(f"\n  [bold cyan]Q:[/] {question}")
        console.print(f"  [bold]Ref:[/] {', '.join(reference_answers[:5])}{'...' if len(reference_answers) > 5 else ''}")
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

    # Join reference answers into a single string for nuggetization
    # The entity_per_nugget policy splits by , ; \n
    reference_str = "; ".join(reference_answers)

    # Send full response + extracted answer to the judge
    if full_response and full_response != answer:
        judge_text = f"{full_response}\n\nExtracted Answer: {answer}"
    else:
        judge_text = answer or ""

    scorer.judge.question_context = question
    score_result = scorer.score(judge_text, reference_str, policy=policy)
    score = score_result["score"]

    if verbose_trace:
        console.print(f"  [bold]Judge score:[/] {score:.3f}  nuggets: {score_result.get('nuggets', [])[:5]}...")
        console.print(f"  [bold]Nugget scores:[/] {score_result.get('nugget_scores', [])[:10]}...")
        if hasattr(scorer.judge, 'last_raw_response') and scorer.judge.last_raw_response:
            console.print(f"  [dim]Judge reasoning:[/] {scorer.judge.last_raw_response}")
        console.print()

    return {
        "question": question,
        "reference_answers": reference_answers,
        "reference_str": reference_str,
        "answer": answer,
        "score": score,
        "nugget_scores": score_result.get("nugget_scores", []),
        "nuggets": score_result.get("nuggets", []),
        "num_nuggets": len(score_result.get("nuggets", [])),
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
            agent, q["question"], q["answers"], scorer, policy,
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
            running_acc = running_correct / len(done_scores) if done_scores else 0

            marker = "[green]✓[/]" if r["score"] >= 0.6 else "[yellow]~[/]" if r["score"] > 0.3 else "[red]✗[/]"
            n_nuggets = r.get("num_nuggets", 0)
            found = sum(1 for s in r.get("nugget_scores", []) if s >= 0.6)
            answer_preview = (r.get("answer") or "(empty)")[:120].replace("\n", " ")

            console.print(
                f"\n  [dim]{n}/{len(questions)}[/]  {marker}  {r['score']:.2f}  "
                f"[dim]{r['latency']:.1f}s[/]  "
                f"[bold]{running_avg:.1%}[/] avg  "
                f"[dim]({running_correct}/{n} ≥0.6)[/]  "
                f"[dim]ETA {eta_min}m{eta_sec:02d}s[/]"
            )
            console.print(f"  [dim]Q:[/] {q['question'][:100]}")
            console.print(f"  [dim]Entities:[/] {found}/{n_nuggets} found")
            if search_queries := r.get("search_queries", []):
                for sq in search_queries[:3]:
                    console.print(f"    [blue]→[/] [dim]{sq[:80]}[/]")
                if len(search_queries) > 3:
                    console.print(f"    [dim]... +{len(search_queries)-3} more[/]")
            console.print(f"  [dim]Answer:[/] {answer_preview}")
            console.print(f"  [dim]Ref:[/] {', '.join(q['answers'][:5])}{'...' if len(q['answers']) > 5 else ''}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            fut.result()  # propagate exceptions

    total_time = time.monotonic() - total_start
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    correct = sum(1 for s in scores if s >= 0.6)

    # QAMPARI-specific: average nugget completion
    all_nugget_scores = []
    for r in results:
        all_nugget_scores.extend(r.get("nugget_scores", []))
    avg_nugget_completion = sum(all_nugget_scores) / len(all_nugget_scores) if all_nugget_scores else 0

    return {
        "label": label,
        "mode": mode_label,
        "parallel_rollouts": parallel_rollouts,
        "avg_score": avg_score,
        "avg_nugget_completion": avg_nugget_completion,
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results) if results else 0,
        "avg_latency": avg_latency,
        "total_time": total_time,
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KONASH on QAMPARI")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval questions")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N eval questions")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel rollouts for PT mode (default: 3, 0 to skip)")
    parser.add_argument("--single-only", action="store_true", help="Only run single rollout, skip parallel thinking")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent eval threads (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Print full trace for each question")
    parser.add_argument("--model", default=None, help="Model ID (auto-detected from provider)")
    parser.add_argument("--provider", default=None, choices=["zhipu", "together", "vllm"], help="LLM provider")
    parser.add_argument("--api-base", default=None, help="Custom API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None, help="API key for solver provider")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model (default: gpt-4o-mini)")
    parser.add_argument("--judge-key", default=None, help="OpenAI API key for judge")
    args = parser.parse_args()

    # Resolve solver provider and key
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
        console.print("[red]No API key found.[/] Run [cyan]konash setup[/] or set ZHIPU_API_KEY / TOGETHER_API_KEY.")
        sys.exit(1)

    judge_key = args.judge_key or os.environ.get("OPENAI_API_KEY")
    if judge_key:
        judge_api_base = OPENAI_API_BASE
        judge_model = args.judge_model
    else:
        console.print("  [yellow]No OpenAI key — using solver model as judge (less accurate)[/]")
        judge_api_base = solver_api_base
        judge_model = solver_model
        judge_key = api_key

    # Step 1: Download QAMPARI
    console.print()
    console.print("[bold]KONASH QAMPARI Evaluation[/]")
    console.print()
    console.rule(style="dim")

    from konash.download import download_qampari
    corpus_dir = download_qampari(console=console)

    questions = load_eval_questions(corpus_dir)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]
    console.print(f"  {len(questions)} eval questions loaded")
    console.print()

    # Step 2: Set up scorer with LLM judge (entity-per-nugget mode)
    from konash.eval.nuggets import NuggetScorer, NuggetPolicyRegistry, LLMNuggetJudge
    policy = NuggetPolicyRegistry.get("QAMPARI")

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
    console.print(f"  Scoring: entity-per-nugget (avg {sum(len(q['answers']) for q in questions)/len(questions):.1f} entities/Q)")

    # Step 3: Create agent
    from konash.api import Agent
    agent = Agent(
        base_model=solver_model,
        corpus=corpus_dir,
        project="eval-qampari",
        api_base=solver_api_base,
        api_key=api_key,
    )

    # Step 4: Baseline eval (single rollout)
    console.rule("[bold]Single rollout[/]", style="dim")
    console.print()

    if provider == "zhipu" and args.workers > 4:
        eval_workers = 4
        console.print(f"  [dim]Zhipu: throttling to {eval_workers} workers (limit 5)[/]")
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
        f"nugget completion {baseline['avg_nugget_completion']:.3f}  "
        f"{baseline['total_time']:.0f}s total"
    )
    console.print()

    # Step 5: Parallel thinking eval
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
            f"nugget completion {parallel['avg_nugget_completion']:.3f}  "
            f"{parallel['total_time']:.0f}s total"
        )
        console.print()

    # Step 6: Summary table
    console.print()
    console.rule("[bold]Results[/]", style="dim")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("Accuracy (≥0.6)", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Nugget Completion", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Total Time", justify="right")

    table.add_row(
        "Single rollout",
        f"{baseline['accuracy']:.0%}",
        f"{baseline['avg_score']:.3f}",
        f"{baseline['avg_nugget_completion']:.3f}",
        f"{baseline['avg_latency']:.1f}s",
        f"{baseline['total_time']:.0f}s",
    )
    if parallel:
        table.add_row(
            f"Parallel (N={args.parallel})",
            f"{parallel['accuracy']:.0%}",
            f"{parallel['avg_score']:.3f}",
            f"{parallel['avg_nugget_completion']:.3f}",
            f"{parallel['avg_latency']:.1f}s",
            f"{parallel['total_time']:.0f}s",
        )
        delta = parallel["avg_score"] - baseline["avg_score"]
        sign = "+" if delta >= 0 else ""
        table.add_row(
            "Delta",
            "",
            f"{sign}{delta:.3f}",
            "",
            "",
            "",
            style="dim",
        )

    console.print(table)
    console.print()
    console.print(f"  [dim]KARL paper target (GLM 4.5 Air base): 45.9% avg score[/]")
    console.print()

    # Step 7: Save results
    results_dir = "eval_results"
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "model": args.model,
        "benchmark": "QAMPARI",
        "num_questions": len(questions),
        "single": {k: v for k, v in baseline.items() if k != "results"},
        "single_details": baseline["results"],
    }
    if parallel:
        output["parallel"] = {k: v for k, v in parallel.items() if k != "results"}
        output["parallel_details"] = parallel["results"]

    out_path = os.path.join(results_dir, "qampari_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"  [dim]Results saved to {out_path}[/]")
    console.print()


if __name__ == "__main__":
    main()
