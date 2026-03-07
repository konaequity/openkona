from __future__ import annotations


class BenchmarkStats:
    name = None
    num_questions = None
    avg_question_tokens = None
    avg_answer_nuggets = None
    num_indexed_documents = None
    avg_document_tokens = None

    def __init__(
        self,
        name=None,
        num_questions=None,
        avg_question_tokens=None,
        avg_answer_nuggets=None,
        num_indexed_documents=None,
        avg_document_tokens=None,
    ):
        self.name = name
        self.num_questions = num_questions
        self.avg_question_tokens = avg_question_tokens
        self.avg_answer_nuggets = avg_answer_nuggets
        self.num_indexed_documents = num_indexed_documents
        self.avg_document_tokens = avg_document_tokens


class BenchmarkStatsRegistry:
    stats = {
        "BrowseCompPlus": BenchmarkStats(
            name="BrowseCompPlus",
            num_questions=830,
            avg_question_tokens=123.2,
            avg_answer_nuggets=1.0,
            num_indexed_documents=100_195,
            avg_document_tokens=480.9,
        ),
        "TRECBiogen": BenchmarkStats(
            name="TRECBiogen",
            num_questions=65,
            avg_question_tokens=15.6,
            avg_answer_nuggets=7.1,
            num_indexed_documents=26_805_982,
            avg_document_tokens=309.4,
        ),
        "FinanceBench": BenchmarkStats(
            name="FinanceBench",
            num_questions=150,
            avg_question_tokens=35.3,
            avg_answer_nuggets=1.0,
            num_indexed_documents=53_399,
            avg_document_tokens=717.9,
        ),
        "QAMPARI": BenchmarkStats(
            name="QAMPARI",
            num_questions=1_000,
            avg_question_tokens=12.3,
            avg_answer_nuggets=14.7,
            num_indexed_documents=256_680,
            avg_document_tokens=129.8,
        ),
        "FreshStack": BenchmarkStats(
            name="FreshStack",
            num_questions=203,
            avg_question_tokens=475.0,
            avg_answer_nuggets=3.1,
            num_indexed_documents=49_514,
            avg_document_tokens=1098.5,
        ),
        "PMBench": BenchmarkStats(
            name="PMBench",
            num_questions=57,
            avg_question_tokens=40.4,
            avg_answer_nuggets=10.5,
            num_indexed_documents=3_395,
            avg_document_tokens=1518.4,
        ),
    }
