#!/opt/anaconda3/bin/python3
"""Evaluate KONASH on FinanceBench — single rollout + parallel thinking.

Usage:
    # Full eval (single + parallel N=3):
    python scripts/eval_financebench.py

    # Limit to N eval questions (for testing):
    python scripts/eval_financebench.py --limit 5

    # Custom parallel rollouts:
    python scripts/eval_financebench.py --parallel 5

    # Also train before eval:
    python scripts/eval_financebench.py --train
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


def load_eval_questions(corpus_dir: str, limit: int | None = None) -> list[dict]:
    eval_path = os.path.join(corpus_dir, "eval_questions.json")
    if not os.path.exists(eval_path):
        console.print(f"[red]No eval questions at {eval_path}[/]")
        sys.exit(1)
    with open(eval_path) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    return questions


def eval_one_question(
    agent, question: str, reference: str, scorer, policy,
    *, parallel_rollouts: int = 1, max_steps: int = 50, top_k: int = 20,
    verbose_trace: bool = False,
) -> dict:
    """Evaluate a single question. Returns result dict with trace."""
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
    trajectory = result.get("trajectory", []) if isinstance(result, dict) else []

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
                console.print(f"  [yellow]Thinking:[/] {reasoning[:300]}")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    console.print(f"  [blue]Search:[/] {fn.get('arguments', '')}")
            if tool_results:
                for tr in tool_results:
                    tr_content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                    # Show first 300 chars of search results
                    console.print(f"  [dim]Results ({len(tr_content)} chars):[/] {tr_content[:300]}")
            if content and not tool_calls:
                console.print(f"  [green]Answer:[/] {content[:400]}")

        console.print(f"\n  [bold]Final answer:[/] {(answer or '(empty)')[:300]}")
        console.print()

    # Pass question context to the judge (KARL Figure 31 includes it)
    scorer.judge.question_context = question
    score_result = scorer.score(answer, reference, policy=policy)
    score = score_result["score"]

    if verbose_trace:
        console.print(f"  [bold]Judge score:[/] {score}  nuggets: {score_result.get('nuggets', [])}")
        console.print()

    return {
        "question": question,
        "reference": reference,
        "answer": answer,
        "score": score,
        "latency": latency,
        "trajectory": trajectory,
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
            agent, q["question"], q["answer"], scorer, policy,
            parallel_rollouts=parallel_rollouts,
            verbose_trace=verbose_trace,
        )
        results[idx] = r
        with lock:
            completed[0] += 1
            n = completed[0]
            marker = "[green]✓[/]" if r["score"] >= 0.6 else "[yellow]~[/]" if r["score"] > 0 else "[red]✗[/]"
            console.print(
                f"  [dim]{n}/{len(questions)}[/]  {marker}  {r['score']:.2f}  "
                f"[dim]{r['latency']:.1f}s[/]  {q['question'][:70]}..."
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            fut.result()  # propagate exceptions

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


def write_traces(
    eval_results: dict, model_name: str, trace_dir: str,
) -> str:
    """Write eval traces in trace viewer format to a JSON file."""
    os.makedirs(trace_dir, exist_ok=True)

    sessions = []
    for i, r in enumerate(eval_results["results"]):
        trajectory = r.get("trajectory", [])

        # Convert environment step records to KONASH rollout step format
        steps = []
        for step_idx, step in enumerate(trajectory):
            agent_response = step.get("agent_response", {})
            tool_results = step.get("tool_results", [])

            # Extract search query from tool calls
            tool_calls = agent_response.get("tool_calls", [])
            query = ""
            if tool_calls:
                tc = tool_calls[0]
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    query = args.get("query", "") if isinstance(args, dict) else ""

            # Extract search results from tool observations
            results_list = []
            for tr in tool_results:
                content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                # Parse "[1] (score: 0.847) text..." format
                for line in content.split("\n\n"):
                    if line.strip().startswith("["):
                        results_list.append({"text": line.strip()[:200]})

            content = agent_response.get("content", "")
            step_type = "retrieval" if query else ("answer" if step.get("done") else "reasoning")

            steps.append({
                "step": step_idx,
                "type": step_type,
                "query": query,
                "results": results_list,
                "num_results": len(results_list),
                "thought": content[:300] if content else "",
                "has_answer": step.get("done", False),
                "answer": r["answer"] if step.get("done") else "",
            })

        session = {
            "query_id": i + 1,
            "question": r["question"],
            "reference_answer": r["reference"],
            "expected_documents": [],
            "models": [{
                "name": model_name,
                "traces": [{
                    "trace_id": 1,
                    "coverage": r["score"],
                    "total_steps": len(steps),
                    "found_count": int(r["score"] >= 0.6),
                    "total_expected": 1,
                    "steps": steps,
                    "final_answer": r["answer"],
                }],
            }],
            "source": "eval",
        }
        sessions.append(session)

    # Write individual session files for trace viewer auto-discovery
    for session in sessions:
        session_file = os.path.join(
            trace_dir, f"financebench_q{session['query_id']}.json"
        )
        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)

    return trace_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate KONASH on FinanceBench")
    parser.add_argument("--train", action="store_true", help="Train at Quick scale before eval")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval questions")
    parser.add_argument("--parallel", type=int, default=3, help="Parallel rollouts for PT mode (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent eval threads (default: 4)")
    parser.add_argument("--verbose", action="store_true", help="Print full trace for each question")
    parser.add_argument("--model", default=None, help="Model ID (auto-detected from provider)")
    parser.add_argument("--provider", default=None, choices=["zhipu", "together"], help="LLM provider (default: auto-detect from available keys)")
    parser.add_argument("--api-key", default=None, help="API key for solver provider")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model (default: gpt-4o-mini)")
    parser.add_argument("--judge-key", default=None, help="OpenAI API key for judge (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    # Resolve solver provider and key
    config_path = os.path.expanduser("~/.konash/config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Auto-detect provider: prefer Zhipu (free), fall back to Together
    provider = args.provider
    api_key = args.api_key

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

    # Resolve OpenAI key (for judge) — falls back to solver model if not set
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

    # Step 1: Download FinanceBench
    console.print()
    console.print("[bold]KONASH FinanceBench Evaluation[/]")
    console.print()
    console.rule(style="dim")

    from konash.download import download_financebench
    corpus_dir = download_financebench(console=console)
    docs_dir = os.path.join(corpus_dir, "documents")

    questions = load_eval_questions(corpus_dir, limit=args.limit)
    console.print(f"  {len(questions)} eval questions loaded")
    console.print()

    # Step 2: Set up scorer with LLM judge
    from konash.eval.nuggets import NuggetScorer, NuggetPolicyRegistry, LLMNuggetJudge
    policy = NuggetPolicyRegistry.get("FinanceBench")

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

    # Step 3: Create agent
    # Use corpus_dir (not docs_dir) so Corpus finds the prebuilt_index.npz
    # alongside the documents/ subdirectory — avoids re-embedding 168 files
    from konash.api import Agent
    agent = Agent(
        base_model=solver_model,
        corpus=corpus_dir,
        project="eval-financebench",
        api_base=solver_api_base,
        api_key=api_key,
    )

    # Optional: train first
    if args.train:
        console.rule("[bold]Training[/]  (Quick scale)", style="dim")
        console.print()
        train_start = time.monotonic()
        agent.train(
            iterations=1,
            synthesis_calls=50,
            rollouts_per_example=8,
            rollout_max_steps=30,
            verbose=True,
        )
        console.print(f"  Training took {time.monotonic() - train_start:.0f}s")
        console.print()

    # Step 4: Baseline eval (single rollout, parallelized across questions)
    console.rule("[bold]Single rollout[/]", style="dim")
    console.print()

    # Force single worker in verbose mode to keep output readable
    eval_workers = 1 if args.verbose else args.workers
    baseline = run_eval(
        agent, questions, scorer, policy, "Single rollout",
        parallel_rollouts=1, workers=eval_workers,
        verbose_trace=args.verbose,
    )
    console.print()
    console.print(
        f"  [bold]Single:[/]  {baseline['correct']}/{baseline['total']} "
        f"({baseline['accuracy']:.0%})  score {baseline['avg_score']:.3f}  "
        f"{baseline['total_time']:.0f}s total"
    )
    console.print()

    # Step 5: Parallel thinking eval (N=3 rollouts + aggregation)
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
        f"({parallel['accuracy']:.0%})  score {parallel['avg_score']:.3f}  "
        f"{parallel['total_time']:.0f}s total"
    )
    console.print()

    # Step 6: Summary table
    console.print()
    console.rule("[bold]Results[/]", style="dim")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("Accuracy", justify="right")
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
    table.add_row(
        f"Parallel (N={args.parallel})",
        f"{parallel['accuracy']:.0%}",
        f"{parallel['avg_score']:.3f}",
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
        style="dim",
    )

    console.print(table)
    console.print()

    # Step 7: Save results + traces
    results_dir = "eval_results"
    os.makedirs(results_dir, exist_ok=True)

    output = {
        "model": args.model,
        "benchmark": "FinanceBench",
        "num_questions": len(questions),
        "single": {k: v for k, v in baseline.items() if k != "results"},
        "single_details": baseline["results"],
        "parallel": {k: v for k, v in parallel.items() if k != "results"},
        "parallel_details": parallel["results"],
    }

    out_path = os.path.join(results_dir, "financebench_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"  [dim]Results saved to {out_path}[/]")

    # Write traces for trace viewer
    trace_dir = "tools/trace_viewer/data"
    write_traces(baseline, f"{args.model} (single)", trace_dir)
    write_traces(parallel, f"{args.model} (N={args.parallel})", trace_dir)
    console.print(f"  [dim]Traces saved to {trace_dir}/ — run the trace viewer to explore[/]")
    console.print()


if __name__ == "__main__":
    main()
