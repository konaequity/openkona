from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


EXPECTED_BENCHMARKS = {
    "BrowseCompPlus": {
        "capability": "constraint_driven_entity_search",
        "indexing_policy": "first_512_tokens",
    },
    "TRECBiogen": {
        "capability": "cross_document_report_synthesis",
        "indexing_policy": "short_abstracts",
    },
    "FinanceBench": {
        "capability": "tabular_numerical_reasoning",
        "indexing_policy": "page_level",
    },
    "QAMPARI": {
        "capability": "exhaustive_entity_search",
        "indexing_policy": "sentence_level_gold_entity_corpus",
    },
    "FreshStack": {
        "capability": "procedural_technical_reasoning",
        "indexing_policy": "semantic_chunks_2048",
    },
    "PMBench": {
        "capability": "enterprise_fact_aggregation",
        "indexing_policy": "first_2048_tokens",
    },
}

EXPECTED_TRAINING_TASKS = {"BrowseCompPlus", "TRECBiogen"}
EXPECTED_HELD_OUT_TASKS = {"FinanceBench", "QAMPARI", "FreshStack", "PMBench"}


def test_benchmark_spec_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.benchmarks", "BenchmarkSpec"))
    assert inspect.isclass(cls), "BenchmarkSpec must be a class"
    assert_has_attrs(
        cls,
        ["name", "capability", "indexing_policy", "evaluation_mode"],
        "BenchmarkSpec",
    )


def test_benchmark_registry_declares_all_karlbench_tasks_with_metadata():
    registry_cls = load_symbol(SymbolSpec("konash.eval.benchmarks", "BenchmarkRegistry"))
    benchmarks = getattr(registry_cls, "benchmarks", None)
    assert isinstance(benchmarks, dict), "BenchmarkRegistry.benchmarks must be a dict"

    missing = EXPECTED_BENCHMARKS.keys() - benchmarks.keys()
    assert not missing, f"Missing benchmark specs: {sorted(missing)}"

    for name, expected in EXPECTED_BENCHMARKS.items():
        spec = benchmarks[name]
        assert_has_attrs(
            spec,
            ["name", "capability", "indexing_policy", "evaluation_mode"],
            f"BenchmarkRegistry.benchmarks[{name!r}]",
        )
        assert getattr(spec, "capability") == expected["capability"]
        assert getattr(spec, "indexing_policy") == expected["indexing_policy"]


def test_registry_encodes_paper_training_split_for_generalization_checks():
    registry_cls = load_symbol(SymbolSpec("konash.eval.benchmarks", "BenchmarkRegistry"))
    training = set(getattr(registry_cls, "training_tasks", set()))
    held_out = set(getattr(registry_cls, "held_out_tasks", set()))

    assert training == EXPECTED_TRAINING_TASKS
    assert held_out == EXPECTED_HELD_OUT_TASKS
    assert training.isdisjoint(held_out), "Training and held-out tasks must not overlap"


def test_each_benchmark_spec_declares_nugget_based_evaluation():
    registry_cls = load_symbol(SymbolSpec("konash.eval.benchmarks", "BenchmarkRegistry"))
    benchmarks = getattr(registry_cls, "benchmarks", {})
    for name in EXPECTED_BENCHMARKS:
        spec = benchmarks[name]
        assert getattr(spec, "evaluation_mode", None) == "nugget_based_completion", (
            f"{name} must use nugget-based evaluation"
        )
