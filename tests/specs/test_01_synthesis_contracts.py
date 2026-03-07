from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_synthesis_modules_exist():
    for module_name in [
        "konash.synthesis.qa",
        "konash.synthesis.rollouts",
        "konash.synthesis.filters",
        "konash.synthesis.dedup",
    ]:
        load_symbol(SymbolSpec(module_name))


def test_question_answer_synthesizer_matches_agentic_search_recipe():
    cls = load_symbol(SymbolSpec("konash.synthesis.qa", "QuestionAnswerSynthesizer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["synthesize", "bootstrap_from_examples", "explore_corpus", "build_prompt"],
        "QuestionAnswerSynthesizer",
    )


def test_rollout_generator_supports_group_rollouts_and_step_limits():
    cls = load_symbol(SymbolSpec("konash.synthesis.rollouts", "RolloutGenerator"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["generate_group", "generate_single", "max_steps", "top_k"],
        "RolloutGenerator",
    )


def test_filtering_pipeline_exists_for_pass_rate_and_quality_checks():
    filters_module = load_symbol(SymbolSpec("konash.synthesis.filters"))
    for symbol in ["PassRateFilter", "QualityFilter", "GroundingFilter"]:
        cls = getattr(filters_module, symbol, None)
        assert cls is not None, f"Missing filter {symbol}"
        assert inspect.isclass(cls), f"{symbol} must be a class"
        assert_has_attrs(cls, ["apply"], symbol)


def test_deduplication_uses_embedding_similarity_contract():
    cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "EmbeddingDeduplicator"))
    assert inspect.isclass(cls)
    assert_has_attrs(cls, ["score_pairs", "deduplicate", "threshold"], "EmbeddingDeduplicator")
