from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_ttc_config_module_exists():
    load_symbol(SymbolSpec("konash.inference.config"))


def test_parallel_thinking_config_tracks_reported_budgets():
    cls = load_symbol(SymbolSpec("konash.inference.config", "ParallelThinkingConfig"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["supported_budgets", "default_budget", "aggregation_mode"],
        "ParallelThinkingConfig",
    )

    budgets = set(getattr(cls, "supported_budgets", set()))
    if budgets:
        assert {3, 10, 20}.issubset(budgets)


def test_value_guided_search_config_tracks_paper_defaults():
    cls = load_symbol(SymbolSpec("konash.inference.config", "ValueGuidedSearchConfig"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "candidate_width",
            "parallel_searches",
            "value_model_name",
            "supported_aggregation_modes",
        ],
        "ValueGuidedSearchConfig",
    )

    candidate_width = getattr(cls, "candidate_width", None)
    if isinstance(candidate_width, int):
        assert candidate_width == 2

    value_model_name = getattr(cls, "value_model_name", None)
    if value_model_name is not None:
        assert value_model_name == "Qwen3-4B-Thinking-2507"

    modes = set(getattr(cls, "supported_aggregation_modes", set()))
    if modes:
        assert {"best_of_n", "weighted_majority_vote"}.issubset(modes)


def test_experiment_registry_tracks_multi_iteration_and_distillation_baselines():
    cls = load_symbol(SymbolSpec("konash.eval.experiments", "ExperimentRegistry"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["experiments", "get", "list_experiments"],
        "ExperimentRegistry",
    )

    experiments = set(getattr(cls, "experiments", {}).keys())
    expected = {
        "main_results",
        "cost_latency",
        "multi_expert_distillation_vs_multitask_rl",
        "multi_iteration_training",
    }
    assert expected.issubset(experiments)
