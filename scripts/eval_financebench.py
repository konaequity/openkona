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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich import box

from konash.download import download_financebench
from konash.eval.harness import (
    BenchmarkConfig, EvalHarness,
    add_common_args, resolve_provider, run_eval, save_results, display_results,
    eval_one_question,
)

console = Console()


def _judge_context(question: str, config: BenchmarkConfig) -> str:
    """Rounding tolerance guidance for financial data."""
    return (
        f"{question}\n\n"
        "Note: For numerical values, rounding differences should be ignored "
        "if they do not meaningfully change the answer. For example, $8.74 "
        "billion and $8.70 billion refer to the same figure with different "
        "rounding precision. Two numbers are considered equivalent if one "
        "can be rounded to the other."
    )


def _progress_detail(r: dict) -> str:
    n_steps = r.get("num_steps", 0)
    n_searches = r.get("num_searches", 0)
    return f"[dim]Steps:[/] {n_steps} ({n_searches} searches)"


BENCH_CONFIG = BenchmarkConfig(
    name="FinanceBench",
    policy_name="FinanceBench",
    project_name="eval-financebench",
    benchmark_key="financebench",
    top_k=20,
    get_reference=lambda q: q["answer"],
    get_nuggets=lambda q: None,
    get_question_text=lambda q: q["question"],
    get_judge_context=_judge_context,
    get_progress_ref_display=lambda q: q["answer"][:120],
    get_progress_detail=_progress_detail,
    download_fn=download_financebench,
)


# ---------------------------------------------------------------------------
# Pass@k (FinanceBench-only feature)
# ---------------------------------------------------------------------------

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def run_pass_at_k_eval(
    agent, questions, scorer, policy, label,
    *, n_rollouts=10, workers=4, k_values=None,
):
    """Run Pass@k evaluation: N independent rollouts per question."""
    import threading

    if k_values is None:
        k_values = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= n_rollouts]
        if n_rollouts not in k_values:
            k_values.append(n_rollouts)

    total_start = time.monotonic()
    completed = [0]
    lock = threading.Lock()
    question_scores = [None] * len(questions)

    def _eval_question(q_idx, q):
        rollout_scores = []
        for _ in range(n_rollouts):
            try:
                result = agent.solve(q["question"], parallel_rollouts=1, return_trace=False)
                answer = result["answer"] if isinstance(result, dict) else result
                scorer.judge.question_context = q["question"]
                score_result = scorer.score(answer, q["answer"], policy=policy)
                rollout_scores.append(score_result["score"])
            except Exception:
                rollout_scores.append(0.0)
        return rollout_scores

    def _run(idx, q):
        scores = _eval_question(idx, q)
        question_scores[idx] = scores
        with lock:
            completed[0] += 1
            n = completed[0]
            n_correct = sum(1 for s in scores if s >= 0.6)
            console.print(
                f"  [dim]{n}/{len(questions)}[/]  "
                f"{n_correct}/{n_rollouts} correct  "
                f"{q['question'][:60]}..."
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            fut.result()

    total_time = time.monotonic() - total_start
    pass_at_k = {}
    for k in k_values:
        per_q_pass = []
        for scores in question_scores:
            n = len(scores)
            c = sum(1 for s in scores if s >= 0.6)
            per_q_pass.append(compute_pass_at_k(n, c, k))
        pass_at_k[k] = sum(per_q_pass) / len(per_q_pass) if per_q_pass else 0.0

    return {
        "label": label, "n_rollouts": n_rollouts, "k_values": k_values,
        "pass_at_k": pass_at_k, "per_question_scores": question_scores,
        "total_time": total_time, "total_questions": len(questions),
    }


# ---------------------------------------------------------------------------
# Trace writing (FinanceBench-only feature)
# ---------------------------------------------------------------------------

def write_traces(eval_results, model_name, trace_dir):
    """Write eval traces in trace viewer format to a JSON file."""
    os.makedirs(trace_dir, exist_ok=True)

    for i, r in enumerate(eval_results["results"]):
        trajectory = r.get("trajectory", [])
        steps = []
        for step_idx, step in enumerate(trajectory):
            agent_response = step.get("agent_response", {})
            tool_results = step.get("tool_results", [])
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

            results_list = []
            for tr in tool_results:
                content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                for line in content.split("\n\n"):
                    if line.strip().startswith("["):
                        results_list.append({"text": line.strip()[:200]})

            content = agent_response.get("content", "")
            step_type = "retrieval" if query else ("answer" if step.get("done") else "reasoning")
            steps.append({
                "step": step_idx, "type": step_type, "query": query,
                "results": results_list, "num_results": len(results_list),
                "thought": content[:300] if content else "",
                "has_answer": step.get("done", False),
                "answer": r["answer"] if step.get("done") else "",
            })

        session = {
            "query_id": i + 1, "question": r["question"],
            "reference_answer": r["reference"],
            "expected_documents": [],
            "models": [{"name": model_name, "traces": [{
                "trace_id": 1, "coverage": r["score"],
                "total_steps": len(steps),
                "found_count": int(r["score"] >= 0.6),
                "total_expected": 1, "steps": steps,
                "final_answer": r["answer"],
            }]}],
            "source": "eval",
        }
        session_file = os.path.join(trace_dir, f"financebench_q{session['query_id']}.json")
        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)

    return trace_dir


# ---------------------------------------------------------------------------
# Main — uses harness but adds --train, --passk, and trace writing
# ---------------------------------------------------------------------------

def main():
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Evaluate KONASH on FinanceBench")
    add_common_args(parser)
    parser.add_argument("--train", action="store_true", help="Train at Quick scale before eval")
    parser.add_argument("--passk", type=int, default=None, metavar="N",
                        help="Run Pass@k eval with N rollouts per question (e.g. --passk 10)")
    args = parser.parse_args()

    harness = EvalHarness(BENCH_CONFIG)
    harness.console = console

    # Resolve provider
    prov = resolve_provider(args)
    provider = prov["provider"]
    solver_api_base = prov["solver_api_base"]
    solver_model = prov["solver_model"]
    api_key = prov["api_key"]

    # Download
    console.print()
    console.print("[bold]KONASH FinanceBench Evaluation[/]")
    console.print()
    console.rule(style="dim")

    corpus_dir = download_financebench(console=console)

    eval_path = os.path.join(corpus_dir, "eval_questions.json")
    with open(eval_path) as f:
        questions = json.load(f)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]
    console.print(f"  {len(questions)} eval questions loaded")
    console.print()

    # Set up scorer
    from konash.eval.nuggets import NuggetScorer, NuggetPolicyRegistry, LLMNuggetJudge
    from konash.api import _OpenAILLMClient, Agent
    policy = NuggetPolicyRegistry.get("FinanceBench")

    judge_client = _OpenAILLMClient(
        api_base=prov["judge_api_base"], api_key=prov["judge_key"],
        model=prov["judge_model"], temperature=0.0,
    )
    judge = LLMNuggetJudge(llm_fn=judge_client.generate)
    scorer = NuggetScorer(judge=judge)
    console.print(f"  Solver: {solver_model} via {provider}")
    console.print(f"  Judge: {prov['judge_model']} via {prov['judge_api_base'].split('//')[1].split('/')[0]}")

    # Create agent
    agent = Agent(
        base_model=solver_model, corpus=corpus_dir, project="eval-financebench",
        api_base=solver_api_base, api_key=api_key,
    )

    # Optional training
    if args.train:
        console.rule("[bold]Training[/]  (Quick scale)", style="dim")
        console.print()
        train_start = time.monotonic()
        agent.train(iterations=1, synthesis_calls=50, rollouts_per_example=8, rollout_max_steps=30, verbose=True)
        console.print(f"  Training took {time.monotonic() - train_start:.0f}s")
        console.print()

    # Workers
    if provider == "zhipu" and args.workers > 4:
        eval_workers = 4
        console.print(f"  [dim]Zhipu: throttling to {eval_workers} workers (limit 5)[/]")
    else:
        eval_workers = args.workers

    # Single rollout
    console.rule("[bold]Single rollout[/]", style="dim")
    console.print()
    baseline = run_eval(
        agent, questions, scorer, policy, "Single rollout", BENCH_CONFIG,
        parallel_rollouts=1, workers=eval_workers, verbose_trace=args.verbose, console=console,
    )
    console.print()
    console.print(
        f"  [bold]Single:[/]  {baseline['correct']}/{baseline['total']} "
        f"({baseline['accuracy']:.0%})  score {baseline['avg_score']:.3f}  "
        f"{baseline['total_time']:.0f}s total"
    )
    console.print()

    # Parallel thinking
    parallel = None
    if not args.single_only and args.parallel > 0:
        console.rule(f"[bold]Parallel thinking[/]  (N={args.parallel})", style="dim")
        console.print()
        parallel = run_eval(
            agent, questions, scorer, policy, f"Parallel (N={args.parallel})", BENCH_CONFIG,
            parallel_rollouts=args.parallel, workers=eval_workers, verbose_trace=args.verbose, console=console,
        )
    if parallel:
        console.print()
        console.print(
            f"  [bold]Parallel:[/]  {parallel['correct']}/{parallel['total']} "
            f"({parallel['accuracy']:.0%})  score {parallel['avg_score']:.3f}  "
            f"{parallel['total_time']:.0f}s total"
        )
        console.print()

    # Pass@k (optional)
    passk_result = None
    if args.passk:
        n = args.passk
        console.rule(f"[bold]Pass@k[/]  (N={n} rollouts per question)", style="dim")
        console.print()
        passk_result = run_pass_at_k_eval(
            agent, questions, scorer, policy, f"Pass@k (N={n})",
            n_rollouts=n, workers=eval_workers,
        )
        console.print()
        for k, score in sorted(passk_result["pass_at_k"].items()):
            console.print(f"  Pass@{k:<3d}  {score:.1%}")
        console.print(f"  [dim]{passk_result['total_time']:.0f}s total[/]")
        console.print()

    # Summary table
    console.print()
    console.rule("[bold]Results[/]", style="dim")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, pad_edge=False, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Total Time", justify="right")

    table.add_row("Single rollout",
        f"{baseline['accuracy']:.0%}", f"{baseline['avg_score']:.3f}",
        f"{baseline['avg_latency']:.1f}s", f"{baseline['total_time']:.0f}s")
    if parallel:
        table.add_row(f"Parallel (N={args.parallel})",
            f"{parallel['accuracy']:.0%}", f"{parallel['avg_score']:.3f}",
            f"{parallel['avg_latency']:.1f}s", f"{parallel['total_time']:.0f}s")
        delta = parallel["avg_score"] - baseline["avg_score"]
        sign = "+" if delta >= 0 else ""
        table.add_row("Delta", "", f"{sign}{delta:.3f}", "", "", style="dim")
    if passk_result:
        table.add_section()
        for k, score in sorted(passk_result["pass_at_k"].items()):
            table.add_row(f"Pass@{k}", f"{score:.1%}", "", "", "")

    console.print(table)
    console.print()

    # Save results
    out_path = save_results(baseline, parallel, BENCH_CONFIG, solver_model, provider, console)

    # Add pass@k to saved output if present
    if passk_result:
        with open(out_path) as f:
            output = json.load(f)
        output["pass_at_k"] = {
            "n_rollouts": passk_result["n_rollouts"],
            "pass_at_k": {str(k): v for k, v in passk_result["pass_at_k"].items()},
            "per_question_scores": passk_result["per_question_scores"],
            "total_time": passk_result["total_time"],
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    # Write traces
    trace_dir = "tools/trace_viewer/data"
    write_traces(baseline, f"{args.model} (single)", trace_dir)
    if parallel:
        write_traces(parallel, f"{args.model} (N={args.parallel})", trace_dir)
    console.print(f"  [dim]Traces saved to {trace_dir}/ — run the trace viewer to explore[/]")
    console.print()


if __name__ == "__main__":
    main()
