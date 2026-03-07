from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


EXPECTED_STATS = {
    "BrowseCompPlus": {
        "num_questions": 830,
        "avg_question_tokens": 123.2,
        "avg_answer_nuggets": 1.0,
        "num_indexed_documents": 100_195,
        "avg_document_tokens": 480.9,
    },
    "TRECBiogen": {
        "num_questions": 65,
        "avg_question_tokens": 15.6,
        "avg_answer_nuggets": 7.1,
        "num_indexed_documents": 26_805_982,
        "avg_document_tokens": 309.4,
    },
    "FinanceBench": {
        "num_questions": 150,
        "avg_question_tokens": 35.3,
        "avg_answer_nuggets": 1.0,
        "num_indexed_documents": 53_399,
        "avg_document_tokens": 717.9,
    },
    "QAMPARI": {
        "num_questions": 1_000,
        "avg_question_tokens": 12.3,
        "avg_answer_nuggets": 14.7,
        "num_indexed_documents": 256_680,
        "avg_document_tokens": 129.8,
    },
    "FreshStack": {
        "num_questions": 203,
        "avg_question_tokens": 475.0,
        "avg_answer_nuggets": 3.1,
        "num_indexed_documents": 49_514,
        "avg_document_tokens": 1098.5,
    },
    "PMBench": {
        "num_questions": 57,
        "avg_question_tokens": 40.4,
        "avg_answer_nuggets": 10.5,
        "num_indexed_documents": 3_395,
        "avg_document_tokens": 1518.4,
    },
}


def test_benchmark_stats_module_exists():
    load_symbol(SymbolSpec("konash.eval.stats"))


def test_benchmark_stats_contract_exists():
    cls = load_symbol(SymbolSpec("konash.eval.stats", "BenchmarkStats"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "name",
            "num_questions",
            "avg_question_tokens",
            "avg_answer_nuggets",
            "num_indexed_documents",
            "avg_document_tokens",
        ],
        "BenchmarkStats",
    )


def test_karlbench_table_two_statistics_are_encoded():
    registry_cls = load_symbol(SymbolSpec("konash.eval.stats", "BenchmarkStatsRegistry"))
    stats = getattr(registry_cls, "stats", {})
    for name, expected in EXPECTED_STATS.items():
        assert name in stats, f"Missing stats for {name}"
        entry = stats[name]
        for field, expected_value in expected.items():
            assert getattr(entry, field) == expected_value, f"{name}.{field} mismatch"
