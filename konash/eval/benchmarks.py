from __future__ import annotations


class BenchmarkSpec:
    name = None
    capability = None
    indexing_policy = None
    evaluation_mode = None

    def __init__(self, name=None, capability=None, indexing_policy=None, evaluation_mode=None):
        self.name = name
        self.capability = capability
        self.indexing_policy = indexing_policy
        self.evaluation_mode = evaluation_mode


class BenchmarkRegistry:
    capabilities = {
        "constraint_driven_entity_search",
        "cross_document_report_synthesis",
        "tabular_numerical_reasoning",
        "exhaustive_entity_search",
        "procedural_technical_reasoning",
        "enterprise_fact_aggregation",
    }

    benchmarks = {
        "BrowseCompPlus": BenchmarkSpec(
            name="BrowseCompPlus",
            capability="constraint_driven_entity_search",
            indexing_policy="first_512_tokens",
            evaluation_mode="nugget_based_completion",
        ),
        "TRECBiogen": BenchmarkSpec(
            name="TRECBiogen",
            capability="cross_document_report_synthesis",
            indexing_policy="short_abstracts",
            evaluation_mode="nugget_based_completion",
        ),
        "FinanceBench": BenchmarkSpec(
            name="FinanceBench",
            capability="tabular_numerical_reasoning",
            indexing_policy="page_level",
            evaluation_mode="nugget_based_completion",
        ),
        "QAMPARI": BenchmarkSpec(
            name="QAMPARI",
            capability="exhaustive_entity_search",
            indexing_policy="sentence_level_gold_entity_corpus",
            evaluation_mode="nugget_based_completion",
        ),
        "FreshStack": BenchmarkSpec(
            name="FreshStack",
            capability="procedural_technical_reasoning",
            indexing_policy="semantic_chunks_2048",
            evaluation_mode="nugget_based_completion",
        ),
        "PMBench": BenchmarkSpec(
            name="PMBench",
            capability="enterprise_fact_aggregation",
            indexing_policy="first_2048_tokens",
            evaluation_mode="nugget_based_completion",
        ),
    }

    training_tasks = {"BrowseCompPlus", "TRECBiogen"}
    held_out_tasks = {"FinanceBench", "QAMPARI", "FreshStack", "PMBench"}

    @classmethod
    def list_benchmarks(cls):
        return list(cls.benchmarks.keys())

    @classmethod
    def get(cls, name):
        return cls.benchmarks[name]
