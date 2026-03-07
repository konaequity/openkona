from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


EXPECTED_CAPABILITIES = {
    "constraint_driven_entity_search",
    "cross_document_report_synthesis",
    "tabular_numerical_reasoning",
    "exhaustive_entity_search",
    "procedural_technical_reasoning",
    "enterprise_fact_aggregation",
}


def test_evaluation_modules_exist():
    for module_name in [
        "konash.eval.benchmarks",
        "konash.eval.nuggets",
        "konash.eval.metrics",
        "konash.eval.runner",
    ]:
        load_symbol(SymbolSpec(module_name))


def test_benchmark_registry_covers_all_karlbench_capabilities():
    registry_cls = load_symbol(SymbolSpec("konash.eval.benchmarks", "BenchmarkRegistry"))
    assert inspect.isclass(registry_cls)
    assert_has_attrs(registry_cls, ["list_benchmarks", "get", "capabilities"], "BenchmarkRegistry")
    capabilities = set(getattr(registry_cls, "capabilities", set()))
    missing = EXPECTED_CAPABILITIES - capabilities
    assert not missing, f"BenchmarkRegistry.capabilities is missing: {sorted(missing)}"


def test_nugget_scorer_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.nuggets", "NuggetScorer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["score", "judge_nugget", "aggregate_scores"],
        "NuggetScorer",
    )


def test_metrics_report_quality_cost_latency_and_generalization():
    cls = load_symbol(SymbolSpec("konash.eval.metrics", "EvaluationReport"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["quality", "cost_per_query", "latency_seconds", "in_distribution", "out_of_distribution"],
        "EvaluationReport",
    )


def test_eval_runner_can_compare_single_rollout_and_ttc_modes():
    cls = load_symbol(SymbolSpec("konash.eval.runner", "EvaluationRunner"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["run_single_rollout", "run_parallel_thinking", "run_value_guided_search", "summarize"],
        "EvaluationRunner",
    )
