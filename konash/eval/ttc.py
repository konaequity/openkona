from __future__ import annotations


class ParallelThinkingResult:
    benchmark_name = None
    num_rollouts = None
    quality_score = None
    model_name = None

    def __init__(self, benchmark_name=None, num_rollouts=None, quality_score=None, model_name=None):
        self.benchmark_name = benchmark_name
        self.num_rollouts = num_rollouts
        self.quality_score = quality_score
        self.model_name = model_name


class ParallelThinkingCost:
    benchmark_name = None
    llm_turns = None
    rollout_token_length = None
    num_rollouts = None

    def __init__(self, benchmark_name=None, llm_turns=None, rollout_token_length=None, num_rollouts=None):
        self.benchmark_name = benchmark_name
        self.llm_turns = llm_turns
        self.rollout_token_length = rollout_token_length
        self.num_rollouts = num_rollouts


class ValueGuidedSearchResult:
    benchmark_name = None
    num_search_trees = None
    aggregation_mode = None
    quality_score = None
    recall = None

    def __init__(self, benchmark_name=None, num_search_trees=None, aggregation_mode=None, quality_score=None, recall=None):
        self.benchmark_name = benchmark_name
        self.num_search_trees = num_search_trees
        self.aggregation_mode = aggregation_mode
        self.quality_score = quality_score
        self.recall = recall


class TTCCostRegistry:
    parallel_thinking_n10 = {
        "BrowseCompPlus": ParallelThinkingCost(
            benchmark_name="BrowseCompPlus",
            llm_turns=3.7,
            rollout_token_length=32156,
            num_rollouts=10,
        ),
        "TRECBiogen": ParallelThinkingCost(
            benchmark_name="TRECBiogen",
            llm_turns=1.5,
            rollout_token_length=9641,
            num_rollouts=10,
        ),
        "FreshStack": ParallelThinkingCost(
            benchmark_name="FreshStack",
            llm_turns=1.3,
            rollout_token_length=14678,
            num_rollouts=10,
        ),
        "FinanceBench": ParallelThinkingCost(
            benchmark_name="FinanceBench",
            llm_turns=1.6,
            rollout_token_length=15105,
            num_rollouts=10,
        ),
        "QAMPARI": ParallelThinkingCost(
            benchmark_name="QAMPARI",
            llm_turns=2.0,
            rollout_token_length=8128,
            num_rollouts=10,
        ),
        "PMBench": ParallelThinkingCost(
            benchmark_name="PMBench",
            llm_turns=2.1,
            rollout_token_length=20444,
            num_rollouts=10,
        ),
    }


class VGSResultRegistry:
    browsecomp_results = {
        ("weighted_majority_vote", 17): ValueGuidedSearchResult(
            benchmark_name="BrowseCompPlus",
            num_search_trees=17,
            aggregation_mode="weighted_majority_vote",
            quality_score=70.4,
        ),
    }


class ParallelThinkingResultRegistry:
    results = {
        ("KARL", 3): ParallelThinkingResult(
            benchmark_name="KARLBench",
            num_rollouts=3,
            quality_score=64.1,
            model_name="KARL",
        ),
        ("KARL", 10): ParallelThinkingResult(
            benchmark_name="KARLBench",
            num_rollouts=10,
            quality_score=67.5,
            model_name="KARL",
        ),
        ("KARL", 20): ParallelThinkingResult(
            benchmark_name="KARLBench",
            num_rollouts=20,
            quality_score=68.1,
            model_name="KARL",
        ),
    }
