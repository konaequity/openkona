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

from konash.download import download_freshstack
from konash.eval.harness import BenchmarkConfig, EvalHarness


def _progress_detail(r: dict) -> str:
    n_nuggets = r.get("num_nuggets", 0)
    found = sum(1 for s in r.get("nugget_scores", []) if s >= 0.6)
    return f"[dim]Nuggets:[/] {found}/{n_nuggets} supported"


BENCH_CONFIG = BenchmarkConfig(
    name="FreshStack",
    policy_name="FreshStack",
    project_name="eval-freshstack",
    benchmark_key="freshstack",
    top_k=10,
    get_reference=lambda q: "; ".join(n["text"] for n in q["nuggets"]),
    get_nuggets=lambda q: [n["text"] for n in q["nuggets"]],
    get_question_text=lambda q: q["question"],
    get_judge_context=None,  # plain question text
    get_progress_ref_display=lambda q: q.get("question_title", q["question"])[:100],
    get_progress_detail=_progress_detail,
    paper_target="KARL paper target (GLM 4.5 Air base): 52.9% avg score",
    download_fn=download_freshstack,
    extra_output_fields={"domain": "langchain"},
)


def main():
    harness = EvalHarness(BENCH_CONFIG)
    harness.run()


if __name__ == "__main__":
    main()
