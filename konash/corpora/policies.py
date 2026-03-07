from __future__ import annotations


class CorpusPolicy:
    name = None
    chunking_mode = None
    indexing_policy = None
    embedding_model = None
    top_k = None

    def __init__(self, name=None, chunking_mode=None, indexing_policy=None, embedding_model=None, top_k=None):
        self.name = name
        self.chunking_mode = chunking_mode
        self.indexing_policy = indexing_policy
        self.embedding_model = embedding_model
        self.top_k = top_k


class CorpusPolicyRegistry:
    policies = {
        "BrowseCompPlus": CorpusPolicy(
            name="BrowseCompPlus",
            chunking_mode="first_512_tokens",
            indexing_policy="first_512_tokens",
            embedding_model="Qwen3-8B-Embedding",
            top_k=20,
        ),
        "TRECBiogen": CorpusPolicy(
            name="TRECBiogen",
            chunking_mode="short_abstracts",
            indexing_policy="short_abstracts",
            embedding_model="Qwen3-0.6B-Embedding",
            top_k=20,
        ),
        "FinanceBench": CorpusPolicy(
            name="FinanceBench",
            chunking_mode="page_level",
            indexing_policy="page_level",
            embedding_model="Qwen3-0.6B-Embedding",
            top_k=20,
        ),
        "QAMPARI": CorpusPolicy(
            name="QAMPARI",
            chunking_mode="sentence_level_gold_entity_corpus",
            indexing_policy="sentence_level_gold_entity_corpus",
            embedding_model="Qwen3-0.6B-Embedding",
            top_k=20,
        ),
        "FreshStack": CorpusPolicy(
            name="FreshStack",
            chunking_mode="semantic_chunks_2048",
            indexing_policy="semantic_chunks_2048",
            embedding_model="Qwen3-0.6B-Embedding",
            top_k=10,
        ),
        "PMBench": CorpusPolicy(
            name="PMBench",
            chunking_mode="first_2048_tokens",
            indexing_policy="first_2048_tokens",
            embedding_model="GTE-large",
            top_k=20,
        ),
    }

    @classmethod
    def get(cls, name):
        return cls.policies[name]

    @classmethod
    def list_policies(cls):
        return list(cls.policies.keys())
