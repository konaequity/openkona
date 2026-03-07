from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_inference_modules_exist():
    for module_name in [
        "konash.inference.parallel",
        "konash.inference.aggregation",
        "konash.inference.value_search",
    ]:
        load_symbol(SymbolSpec(module_name))


def test_parallel_thinking_generates_multiple_rollouts_then_aggregates():
    cls = load_symbol(SymbolSpec("konash.inference.parallel", "ParallelThinkingEngine"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["run", "generate_parallel_rollouts", "extract_answers", "aggregate"],
        "ParallelThinkingEngine",
    )


def test_generative_aggregator_can_synthesize_beyond_majority_vote():
    cls = load_symbol(SymbolSpec("konash.inference.aggregation", "GenerativeAggregator"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["aggregate", "build_aggregation_prompt"],
        "GenerativeAggregator",
    )


def test_value_guided_search_is_available_as_optional_advanced_ttc():
    cls = load_symbol(SymbolSpec("konash.inference.value_search", "ValueGuidedSearchEngine"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["run", "expand", "score_candidates", "aggregate"],
        "ValueGuidedSearchEngine",
    )
