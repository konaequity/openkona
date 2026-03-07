from __future__ import annotations


class ParallelThinkingConfig:
    supported_budgets = {3, 5, 10, 15, 20}
    default_budget = 10
    aggregation_mode = "generative"


class ValueGuidedSearchConfig:
    candidate_width = 2
    parallel_searches = None
    value_model_name = "Qwen3-4B-Thinking-2507"
    supported_aggregation_modes = {"best_of_n", "weighted_majority_vote", "generative"}
