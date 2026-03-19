"""Shared benchmark evaluation entrypoints."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import box
from rich.console import Console
from rich.table import Table

from konash.benchmarks import get_dataset
from konash.eval.harness import (
    add_common_args,
    display_results,
    resolve_provider,
    run_eval,
    save_results,
)


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
    *, n_rollouts=10, workers=4, console: Console,
):
    """Run Pass@k evaluation: N independent rollouts per question."""
    import threading

    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= n_rollouts]
    if n_rollouts not in k_values:
        k_values.append(n_rollouts)

    total_start = time.monotonic()
    completed = [0]
    lock = threading.Lock()
    question_scores = [None] * len(questions)

    def _eval_question(q):
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
        scores = _eval_question(q)
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
        "label": label,
        "n_rollouts": n_rollouts,
        "k_values": k_values,
        "pass_at_k": pass_at_k,
        "per_question_scores": question_scores,
        "total_time": total_time,
        "total_questions": len(questions),
    }


def write_eval_traces(eval_results, model_name, trace_dir, benchmark_key):
    """Write eval traces in trace viewer format to JSON files."""
    os.makedirs(trace_dir, exist_ok=True)

    for i, result in enumerate(eval_results["results"]):
        trajectory = result.get("trajectory", [])
        steps = []
        for step_idx, step in enumerate(trajectory):
            agent_response = step.get("agent_response", {})
            tool_results = step.get("tool_results", [])
            tool_calls = agent_response.get("tool_calls", [])
            query = ""
            if tool_calls:
                tool_call = tool_calls[0]
                if isinstance(tool_call, dict):
                    fn = tool_call.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    query = args.get("query", "") if isinstance(args, dict) else ""

            results_list = []
            for tool_result in tool_results:
                content = tool_result.get("content", "") if isinstance(tool_result, dict) else str(tool_result)
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
                "answer": result["answer"] if step.get("done") else "",
            })

        session = {
            "query_id": i + 1,
            "question": result["question"],
            "reference_answer": result["reference"],
            "expected_documents": [],
            "models": [{"name": model_name, "traces": [{
                "trace_id": 1,
                "coverage": result["score"],
                "total_steps": len(steps),
                "found_count": int(result["score"] >= 0.6),
                "total_expected": 1,
                "steps": steps,
                "final_answer": result["answer"],
            }]}],
            "source": "eval",
        }
        session_file = os.path.join(trace_dir, f"{benchmark_key}_q{session['query_id']}.json")
        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)


def _add_benchmark_args(parser, hooks) -> None:
    if hooks.supports_train_quick:
        parser.add_argument("--train", action="store_true", help="Train at Quick scale before eval")
    if hooks.supports_passk:
        parser.add_argument(
            "--passk", type=int, default=None, metavar="N",
            help="Run Pass@k eval with N rollouts per question (e.g. --passk 10)",
        )


def _render_results_with_passk(baseline, parallel, passk_result, args, console: Console) -> None:
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
    if parallel:
        table.add_row(
            f"Parallel (N={args.parallel})",
            f"{parallel['accuracy']:.0%}",
            f"{parallel['avg_score']:.3f}",
            f"{parallel['avg_latency']:.1f}s",
            f"{parallel['total_time']:.0f}s",
        )
        delta = parallel["avg_score"] - baseline["avg_score"]
        sign = "+" if delta >= 0 else ""
        table.add_row("Delta", "", f"{sign}{delta:.3f}", "", "", style="dim")
    table.add_section()
    for k, score in sorted(passk_result["pass_at_k"].items()):
        table.add_row(f"Pass@{k}", f"{score:.1%}", "", "", "")

    console.print(table)
    console.print()


def main_for_benchmark(benchmark_key: str) -> None:
    import argparse

    console = Console()
    dataset = get_dataset(benchmark_key)
    bench_config = dataset.benchmark
    if bench_config is None:
        raise KeyError(f"Benchmark {benchmark_key!r} is not registered for evaluation")
    hooks = dataset.hooks

    parser = argparse.ArgumentParser(description=f"Evaluate KONASH on {bench_config.name}")
    add_common_args(parser)
    _add_benchmark_args(parser, hooks)
    args = parser.parse_args()

    prov = resolve_provider(args)
    provider = prov["provider"]
    solver_api_base = prov["solver_api_base"]
    solver_model = prov["solver_model"]
    api_key = prov["api_key"]

    console.print()
    console.print(f"[bold]KONASH {bench_config.name} Evaluation[/]")
    console.print()
    console.rule(style="dim")

    corpus_dir = dataset.download(console=console)
    eval_path = dataset.eval_questions_path(corpus_dir)
    with open(eval_path) as f:
        questions = json.load(f)
    if args.offset:
        questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]
    console.print(f"  {len(questions)} eval questions loaded")
    console.print()

    from konash.api import _OpenAILLMClient, Agent
    from konash.eval.nuggets import LLMNuggetJudge, NuggetPolicyRegistry, NuggetScorer

    policy = NuggetPolicyRegistry.get(bench_config.policy_name)
    judge_client = _OpenAILLMClient(
        api_base=prov["judge_api_base"],
        api_key=prov["judge_key"],
        model=prov["judge_model"],
        temperature=0.0,
    )
    judge = LLMNuggetJudge(llm_fn=judge_client.generate)
    scorer = NuggetScorer(judge=judge)

    console.print(f"  Solver: {solver_model} via {provider}")
    console.print(f"  Judge: {prov['judge_model']} via {prov['judge_api_base'].split('//')[1].split('/')[0]}")

    agent = Agent(
        base_model=solver_model,
        corpus=corpus_dir,
        project=bench_config.project_name,
        api_base=solver_api_base,
        api_key=api_key,
    )

    if hooks.supports_train_quick and getattr(args, "train", False):
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

    eval_workers = 4 if provider == "zhipu" and args.workers > 4 else args.workers
    if eval_workers != args.workers:
        console.print(f"  [dim]Zhipu: throttling to {eval_workers} workers (limit 5)[/]")

    console.rule("[bold]Single rollout[/]", style="dim")
    console.print()
    baseline = run_eval(
        agent, questions, scorer, policy, "Single rollout", bench_config,
        parallel_rollouts=1, workers=eval_workers, verbose_trace=args.verbose, console=console,
    )
    console.print()
    console.print(
        f"  [bold]Single:[/]  {baseline['correct']}/{baseline['total']} "
        f"({baseline['accuracy']:.0%})  score {baseline['avg_score']:.3f}  "
        f"{baseline['total_time']:.0f}s total"
    )
    console.print()

    parallel = None
    if args.parallel > 0:
        console.rule(f"[bold]Parallel thinking[/]  (N={args.parallel})", style="dim")
        console.print()
        parallel = run_eval(
            agent, questions, scorer, policy, f"Parallel (N={args.parallel})", bench_config,
            parallel_rollouts=args.parallel, workers=eval_workers, verbose_trace=args.verbose, console=console,
        )
        console.print()
        console.print(
            f"  [bold]Parallel:[/]  {parallel['correct']}/{parallel['total']} "
            f"({parallel['accuracy']:.0%})  score {parallel['avg_score']:.3f}  "
            f"{parallel['total_time']:.0f}s total"
        )
        console.print()

    passk_result = None
    if hooks.supports_passk and getattr(args, "passk", None):
        console.rule(f"[bold]Pass@k[/]  (N={args.passk} rollouts per question)", style="dim")
        console.print()
        passk_result = run_pass_at_k_eval(
            agent, questions, scorer, policy, f"Pass@k (N={args.passk})",
            n_rollouts=args.passk, workers=eval_workers, console=console,
        )
        console.print()
        for k, score in sorted(passk_result["pass_at_k"].items()):
            console.print(f"  Pass@{k:<3d}  {score:.1%}")
        console.print(f"  [dim]{passk_result['total_time']:.0f}s total[/]")
        console.print()

    if passk_result:
        _render_results_with_passk(baseline, parallel, passk_result, args, console)
    else:
        display_results(baseline, parallel, bench_config, args, console)

    out_path = save_results(baseline, parallel, bench_config, solver_model, provider, console)
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

    if hooks.writes_traces:
        trace_dir = hooks.trace_dir
        model_label = args.model or solver_model
        write_eval_traces(baseline, f"{model_label} (single)", trace_dir, benchmark_key)
        if parallel:
            write_eval_traces(parallel, f"{model_label} (N={args.parallel})", trace_dir, benchmark_key)
        console.print(f"  [dim]Traces saved to {trace_dir}/ — run the trace viewer to explore[/]")
        console.print()


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate KONASH on a registered benchmark")
    parser.add_argument("benchmark", choices=sorted(
        key for key in ("financebench", "qampari", "freshstack")
    ))
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    main_for_benchmark(args.benchmark)
