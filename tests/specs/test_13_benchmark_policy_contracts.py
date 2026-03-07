from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_corpus_policy_module_exists():
    load_symbol(SymbolSpec("konash.corpora.policies"))


def test_corpus_policy_registry_covers_all_benchmark_ingestion_rules():
    registry_cls = load_symbol(SymbolSpec("konash.corpora.policies", "CorpusPolicyRegistry"))
    assert inspect.isclass(registry_cls)
    assert_has_attrs(
        registry_cls,
        ["policies", "get", "list_policies"],
        "CorpusPolicyRegistry",
    )

    policies = getattr(registry_cls, "policies", {})
    for name in [
        "BrowseCompPlus",
        "TRECBiogen",
        "FinanceBench",
        "QAMPARI",
        "FreshStack",
        "PMBench",
    ]:
        assert name in policies, f"Missing corpus policy for {name}"


def test_benchmark_corpus_policies_expose_chunking_and_retrieval_defaults():
    policy_cls = load_symbol(SymbolSpec("konash.corpora.policies", "CorpusPolicy"))
    assert inspect.isclass(policy_cls)
    assert_has_attrs(
        policy_cls,
        ["name", "chunking_mode", "indexing_policy", "embedding_model", "top_k"],
        "CorpusPolicy",
    )


def test_specific_benchmark_retrieval_defaults_match_paper():
    registry_cls = load_symbol(SymbolSpec("konash.corpora.policies", "CorpusPolicyRegistry"))
    policies = getattr(registry_cls, "policies", {})

    expectations = {
        "BrowseCompPlus": ("first_512_tokens", "Qwen3-8B-Embedding", 20),
        "TRECBiogen": ("short_abstracts", "Qwen3-0.6B-Embedding", 20),
        "FinanceBench": ("page_level", "Qwen3-0.6B-Embedding", 20),
        "QAMPARI": ("sentence_level_gold_entity_corpus", "Qwen3-0.6B-Embedding", 20),
        "FreshStack": ("semantic_chunks_2048", "Qwen3-0.6B-Embedding", 10),
        "PMBench": ("first_2048_tokens", "GTE-large", 20),
    }

    for name, (indexing_policy, embedding_model, top_k) in expectations.items():
        policy = policies[name]
        assert getattr(policy, "indexing_policy") == indexing_policy
        assert getattr(policy, "embedding_model") == embedding_model
        assert getattr(policy, "top_k") == top_k


def test_retrieval_budget_policy_scales_inverse_to_chunk_length_and_caps_k():
    cls = load_symbol(SymbolSpec("konash.retrieval.vector_search", "RetrievalBudgetPolicy"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["compute_top_k", "max_top_k", "target_token_budget"],
        "RetrievalBudgetPolicy",
    )

    max_top_k = getattr(cls, "max_top_k", None)
    if isinstance(max_top_k, int):
        assert max_top_k == 20
