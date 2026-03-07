from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_ttc_results_module_exists():
    load_symbol(SymbolSpec("konash.eval.ttc"))


def test_parallel_thinking_result_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.ttc", "ParallelThinkingResult"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["benchmark_name", "num_rollouts", "quality_score", "model_name"],
        "ParallelThinkingResult",
    )


def test_parallel_thinking_rollout_cost_contract_tracks_table_seven():
    cls = load_symbol(SymbolSpec("konash.eval.ttc", "ParallelThinkingCost"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["benchmark_name", "llm_turns", "rollout_token_length", "num_rollouts"],
        "ParallelThinkingCost",
    )

    registry_cls = load_symbol(SymbolSpec("konash.eval.ttc", "TTCCostRegistry"))
    costs = getattr(registry_cls, "parallel_thinking_n10", {})
    expected = {
        "BrowseCompPlus": (3.7, 32156),
        "TRECBiogen": (1.5, 9641),
        "FreshStack": (1.3, 14678),
        "FinanceBench": (1.6, 15105),
        "QAMPARI": (2.0, 8128),
        "PMBench": (2.1, 20444),
    }
    for benchmark_name, (llm_turns, rollout_tokens) in expected.items():
        assert benchmark_name in costs, f"Missing TTC cost entry for {benchmark_name}"
        entry = costs[benchmark_name]
        assert getattr(entry, "num_rollouts") == 10
        assert getattr(entry, "llm_turns") == llm_turns
        assert getattr(entry, "rollout_token_length") == rollout_tokens


def test_vgs_result_contract_tracks_aggregation_modes():
    cls = load_symbol(SymbolSpec("konash.eval.ttc", "ValueGuidedSearchResult"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["benchmark_name", "num_search_trees", "aggregation_mode", "quality_score", "recall"],
        "ValueGuidedSearchResult",
    )


def test_vgs_registry_encodes_reported_browsecomp_peak_result():
    registry_cls = load_symbol(SymbolSpec("konash.eval.ttc", "VGSResultRegistry"))
    results = getattr(registry_cls, "browsecomp_results", {})
    peak = results.get(("weighted_majority_vote", 17))
    assert peak is not None, "Missing BrowseComp VGS result for WMV at N=17"
    assert getattr(peak, "quality_score") == 70.4


def test_parallel_thinking_registry_encodes_reported_main_budgets():
    registry_cls = load_symbol(SymbolSpec("konash.eval.ttc", "ParallelThinkingResultRegistry"))
    results = getattr(registry_cls, "results", {})
    required = {
        ("KARL", 3),
        ("KARL", 10),
        ("KARL", 20),
    }
    missing = required - set(results.keys())
    assert not missing, f"Missing main TTC result keys: {sorted(missing)}"
