from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_main_results_table_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.experiments", "MainResultsRow"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "model_name",
            "browsecomp_plus",
            "trec_biogen",
            "freshstack",
            "financebench",
            "qampari",
            "pmbench",
            "in_distribution",
            "out_of_distribution",
            "total",
        ],
        "MainResultsRow",
    )


def test_table_four_tracks_key_karl_rows():
    registry_cls = load_symbol(SymbolSpec("konash.eval.experiments", "ExperimentRegistry"))
    table = getattr(registry_cls, "main_results_table", {})

    expected_total_scores = {
        "KARL": 58.9,
        "KARL (par. N = 3)": 64.1,
        "KARL (par. N = 10)": 67.5,
        "KARL (par. N = 20)": 68.1,
        "KARL-BCP (VGS N = 17)": 55.5,
        "KARL-TREC": 56.8,
    }
    for model_name, total in expected_total_scores.items():
        assert model_name in table, f"Missing main-results row for {model_name}"
        assert getattr(table[model_name], "total") == total


def test_distillation_comparison_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.experiments", "DistillationComparison"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "model_name",
            "in_distribution",
            "out_of_distribution",
            "total",
            "parallel_thinking_budget",
        ],
        "DistillationComparison",
    )


def test_distillation_vs_rl_registry_tracks_reported_scaling_gap():
    registry_cls = load_symbol(SymbolSpec("konash.eval.experiments", "ExperimentRegistry"))
    comparison = getattr(registry_cls, "distillation_vs_rl", {})

    expected = {
        ("SFT Distillation", 0): (69.1, 59.4),
        ("SFT Distillation", 15): (75.3, 59.6),
        ("KARL", 0): (67.9, 62.7),
        ("KARL", 15): (78.4, 64.8),
    }
    for key, (in_dist, ood) in expected.items():
        assert key in comparison, f"Missing distillation comparison for {key}"
        row = comparison[key]
        assert getattr(row, "in_distribution") == in_dist
        assert getattr(row, "out_of_distribution") == ood


def test_iteration_progression_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.experiments", "IterationProgression"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["task_name", "base", "iteration_1", "iteration_2", "iteration_3"],
        "IterationProgression",
    )


def test_multi_iteration_registry_tracks_reported_trec_case_study():
    registry_cls = load_symbol(SymbolSpec("konash.eval.experiments", "ExperimentRegistry"))
    progression = getattr(registry_cls, "multi_iteration_progression", {})
    expected = {
        "TRECBiogen": (66.0, 82.0, 85.0, None),
        "FreshStack": (52.9, 49.8, 56.7, None),
        "QAMPARI": (45.9, 48.2, 50.8, None),
    }
    for task_name, (base, iteration_1, iteration_2, iteration_3) in expected.items():
        assert task_name in progression, f"Missing progression for {task_name}"
        row = progression[task_name]
        assert getattr(row, "base") == base
        assert getattr(row, "iteration_1") == iteration_1
        assert getattr(row, "iteration_2") == iteration_2
