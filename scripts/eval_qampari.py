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

from konash.download import download_qampari
from konash.eval.harness import BenchmarkConfig, EvalHarness


def _compute_nugget_completion(eval_result: dict) -> dict:
    """Compute QAMPARI-specific average nugget completion metric."""
    all_nugget_scores = []
    for r in eval_result["results"]:
        all_nugget_scores.extend(r.get("nugget_scores", []))
    avg = sum(all_nugget_scores) / len(all_nugget_scores) if all_nugget_scores else 0
    return {"avg_nugget_completion": avg}


def _progress_detail(r: dict) -> str:
    n_nuggets = r.get("num_nuggets", 0)
    found = sum(1 for s in r.get("nugget_scores", []) if s >= 0.6)
    return f"[dim]Entities:[/] {found}/{n_nuggets} found"


BENCH_CONFIG = BenchmarkConfig(
    name="QAMPARI",
    policy_name="QAMPARI",
    project_name="eval-qampari",
    benchmark_key="qampari",
    top_k=20,
    get_reference=lambda q: "; ".join(q["answers"]),
    get_nuggets=lambda q: None,
    get_question_text=lambda q: q["question"],
    get_judge_context=None,  # plain question text
    get_progress_ref_display=lambda q: (
        ", ".join(q["answers"][:5]) + ("..." if len(q["answers"]) > 5 else "")
    ),
    get_progress_detail=_progress_detail,
    get_extra_output=_compute_nugget_completion,
    extra_table_columns=[("Nugget Completion", "avg_nugget_completion")],
    paper_target="KARL paper target (GLM 4.5 Air base): 45.9% avg score",
    download_fn=download_qampari,
)


def main():
    harness = EvalHarness(BENCH_CONFIG)
    harness.run()


if __name__ == "__main__":
    main()
