from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_ablation_registry_module_exists():
    load_symbol(SymbolSpec("konash.eval.ablations"))


def test_search_environment_ablation_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.ablations", "SearchEnvironmentAblation"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["component", "setting", "benchmark_score", "benchmark_recall"],
        "SearchEnvironmentAblation",
    )


def test_table_five_ablation_entries_are_tracked():
    registry_cls = load_symbol(SymbolSpec("konash.eval.ablations", "AblationRegistry"))
    ablations = getattr(registry_cls, "search_environment", {})

    expected = {
        ("compression", "with"): (0.570, 0.681),
        ("compression", "without"): (0.389, 0.503),
        ("retrieval", "Qwen3-Embedding-8B"): (0.570, 0.681),
        ("retrieval", "Vector Search (GTE-large + hybrid)"): (0.568, 0.698),
    }
    for key, (score, recall) in expected.items():
        assert key in ablations, f"Missing ablation entry for {key}"
        entry = ablations[key]
        assert getattr(entry, "benchmark_score") == score
        assert getattr(entry, "benchmark_recall") == recall


def test_compression_transfer_matrix_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.ablations", "CompressionTransferResult"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["search_model", "compression_model", "browsecomp_score"],
        "CompressionTransferResult",
    )


def test_table_six_search_and_compression_cross_eval_is_tracked():
    registry_cls = load_symbol(SymbolSpec("konash.eval.ablations", "AblationRegistry"))
    matrix = getattr(registry_cls, "compression_transfer", {})
    expected = {
        ("GLM 4.5 Air", "GLM 4.5 Air"): 0.44,
        ("GLM 4.5 Air", "KARL-BCP"): 0.54,
        ("KARL-BCP", "GLM 4.5 Air"): 0.46,
        ("KARL-BCP", "KARL-BCP"): 0.57,
    }
    for key, score in expected.items():
        assert key in matrix, f"Missing compression transfer result for {key}"
        assert getattr(matrix[key], "browsecomp_score") == score


def test_sharpening_analysis_contract_tracks_max_at_k_curves_and_pass_rate_flow():
    cls = load_symbol(SymbolSpec("konash.eval.ablations", "SharpeningAnalysis"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["max_at_k_curve", "pass_rate_flow", "transition_matrix", "supports_partial_solved_unsolved"],
        "SharpeningAnalysis",
    )


def test_search_scaling_contract_tracks_horizon_and_retrieval_sweeps():
    cls = load_symbol(SymbolSpec("konash.eval.ablations", "SearchScalingSweep"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["search_horizons", "retrieval_counts", "scores_by_horizon", "scores_by_retrieval_count"],
        "SearchScalingSweep",
    )
