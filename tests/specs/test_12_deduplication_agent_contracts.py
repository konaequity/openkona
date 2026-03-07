from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_deduplication_agent_module_exists():
    load_symbol(SymbolSpec("konash.synthesis.dedup"))


def test_deduplication_agent_models_two_stage_pipeline_from_paper():
    cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "DeduplicationAgent"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "remove_exact_duplicates",
            "remove_near_duplicates",
            "deduplicate_against_eval_set",
            "deduplicate_within_synthetic_set",
            "judge_paraphrase",
            "run",
        ],
        "DeduplicationAgent",
    )


def test_deduplication_agent_tracks_eval_and_synthetic_contamination_separately():
    cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "DeduplicationAgent"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "evaluation_questions",
            "synthetic_questions",
            "removed_exact_matches",
            "removed_near_duplicates",
        ],
        "DeduplicationAgent",
    )


def test_near_duplicate_pipeline_uses_embedding_retrieval_plus_paraphrase_judge():
    cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "DeduplicationAgent"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "embedding_model",
            "similarity_top_k",
            "retrieve_similar_questions",
            "paraphrase_judge",
            "judge_paraphrase",
        ],
        "DeduplicationAgent",
    )


def test_trec_biogen_dedup_policy_matches_paper_recipe():
    policy_cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "TRECBiogenDedupPolicy"))
    assert inspect.isclass(policy_cls)
    assert_has_attrs(
        policy_cls,
        [
            "embedding_model",
            "similarity_top_k",
            "exact_match_scope",
            "paraphrase_judge_model",
        ],
        "TRECBiogenDedupPolicy",
    )

    embedding_model = getattr(policy_cls, "embedding_model", None)
    similarity_top_k = getattr(policy_cls, "similarity_top_k", None)
    if embedding_model is not None:
        assert embedding_model == "Qwen3-8B-Embedding"
    if similarity_top_k is not None:
        assert similarity_top_k == 20


def test_browsecomp_dedup_policy_matches_paper_recipe():
    policy_cls = load_symbol(SymbolSpec("konash.synthesis.dedup", "BrowseCompDedupPolicy"))
    assert inspect.isclass(policy_cls)
    assert_has_attrs(
        policy_cls,
        [
            "embedding_model",
            "similarity_top_k",
            "exact_answer_blocklist",
            "paraphrase_judge_model",
        ],
        "BrowseCompDedupPolicy",
    )

    embedding_model = getattr(policy_cls, "embedding_model", None)
    similarity_top_k = getattr(policy_cls, "similarity_top_k", None)
    if embedding_model is not None:
        assert embedding_model == "Qwen3-0.6B-Embedding"
    if similarity_top_k is not None:
        assert similarity_top_k == 10


def test_synthesized_examples_keep_citations_for_dedup_auditability():
    cls = load_symbol(SymbolSpec("konash.synthesis.qa", "SyntheticExample"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["question", "answer", "citations"],
        "SyntheticExample",
    )
