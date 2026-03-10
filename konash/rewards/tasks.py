from __future__ import annotations

from typing import Dict

from konash.rewards.base import TaskRewardSpec


TASK_REWARD_SPECS: Dict[str, TaskRewardSpec] = {
    "BrowseCompPlus": TaskRewardSpec(
        task_name="BrowseCompPlus",
        policy_name="BrowseCompPlus",
        description="Single-nugget binary reward for constraint-driven entity search.",
    ),
    "TRECBiogen": TaskRewardSpec(
        task_name="TRECBiogen",
        policy_name="TRECBiogen",
        description="Report-style multi-reference nugget reward for cross-document synthesis.",
    ),
    "FinanceBench": TaskRewardSpec(
        task_name="FinanceBench",
        policy_name="FinanceBench",
        description="Single-nugget binary reward for financial question answering.",
    ),
    "QAMPARI": TaskRewardSpec(
        task_name="QAMPARI",
        policy_name="QAMPARI",
        description="Entity-per-nugget reward for exhaustive entity retrieval.",
    ),
    "FreshStack": TaskRewardSpec(
        task_name="FreshStack",
        policy_name="FreshStack",
        description="Fixed-nugget reward for procedural technical reasoning.",
    ),
    "PMBench": TaskRewardSpec(
        task_name="PMBench",
        policy_name="PMBench",
        description="Fixed-nugget reward for enterprise fact aggregation.",
    ),
}
