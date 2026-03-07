from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_synthesis_pipeline_module_exists():
    load_symbol(SymbolSpec("konash.synthesis.pipeline"))


def test_question_answer_synthesizer_tracks_few_shot_examples_and_grounding_tool():
    cls = load_symbol(SymbolSpec("konash.synthesis.qa", "QuestionAnswerSynthesizer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["few_shot_examples", "task_prompt", "vector_search_tool", "explore_corpus", "synthesize"],
        "QuestionAnswerSynthesizer",
    )


def test_stage_two_pipeline_supports_pass_rate_estimation_and_quality_filtering():
    cls = load_symbol(SymbolSpec("konash.synthesis.pipeline", "SynthesisPipeline"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "run_stage_one",
            "run_stage_two",
            "deduplicate",
            "estimate_pass_rate",
            "apply_quality_filter",
        ],
        "SynthesisPipeline",
    )


def test_pass_rate_filter_rejects_groups_at_both_extremes():
    cls = load_symbol(SymbolSpec("konash.synthesis.filters", "PassRateFilter"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["min_pass_rate", "max_pass_rate", "apply"],
        "PassRateFilter",
    )


def test_quality_filter_models_both_ambiguity_and_reference_accuracy_checks():
    cls = load_symbol(SymbolSpec("konash.synthesis.filters", "QualityFilter"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["judge_ambiguity", "judge_reference_accuracy", "apply"],
        "QualityFilter",
    )


def test_deduplicator_supports_exact_and_near_duplicate_detection():
    cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "EmbeddingDeduplicator"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["find_exact_duplicates", "score_pairs", "deduplicate", "threshold"],
        "EmbeddingDeduplicator",
    )
