from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_parallel_thinking_supports_tool_enabled_aggregation():
    cls = load_symbol(SymbolSpec("konash.inference.parallel", "ParallelThinkingEngine"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "num_rollouts",
            "generate_parallel_rollouts",
            "extract_answers",
            "run_aggregation_rollout",
            "aggregate",
        ],
        "ParallelThinkingEngine",
    )


def test_generative_aggregator_is_more_expressive_than_voting():
    cls = load_symbol(SymbolSpec("konash.inference.aggregation", "GenerativeAggregator"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["tool_access_enabled", "build_aggregation_prompt", "aggregate"],
        "GenerativeAggregator",
    )


def test_value_model_scores_partial_rollouts_on_policy_tokens_only():
    cls = load_symbol(SymbolSpec("konash.inference.value_model", "ValueModel"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["fit", "score_partial_rollout", "score_rollout", "mask_policy_tokens"],
        "ValueModel",
    )


def test_value_guided_search_uses_parallel_bfs_style_candidate_selection():
    cls = load_symbol(SymbolSpec("konash.inference.value_search", "ValueGuidedSearchEngine"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["candidate_width", "parallel_searches", "value_model", "run_parallel_bfs", "aggregate"],
        "ValueGuidedSearchEngine",
    )


def test_vector_search_tool_supports_embedded_cached_indexes_for_rollout_throughput():
    cls = load_symbol(SymbolSpec("konash.retrieval.vector_search", "VectorSearchTool"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["embedded_index", "load_cached_index", "search", "batch_search"],
        "VectorSearchTool",
    )
    target_qps = getattr(cls, "target_qps_per_host", None)
    if isinstance(target_qps, (int, float)):
        assert target_qps >= 500
