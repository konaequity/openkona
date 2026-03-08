from __future__ import annotations

from typing import Any, Optional, Sequence

from konash.eval.nuggets import NuggetPolicyRegistry, NuggetScorer

from konash.rewards.base import RewardFn


def make_nugget_reward(
    task_name: str,
    *,
    threshold: float = 0.5,
    binary: bool = True,
) -> RewardFn:
    """Build a nugget-based reward for a specific benchmark task.

    This mirrors the paper's reward structure:
    - nugget-based completion scoring
    - task-specific nuggetization policy
    - binary rollout reward by default
    """
    policy = NuggetPolicyRegistry.get(task_name)

    def _reward(
        prediction: str = "",
        reference: Any = "",
        **kwargs: Any,
    ) -> float:
        if not prediction:
            return 0.0

        judge = kwargs.get("nugget_judge")
        scorer = NuggetScorer(judge=judge, policy=policy)
        nuggets = kwargs.get("nuggets")
        references: Optional[Sequence[Any]] = kwargs.get("references")

        if (
            references
            and policy.reference_handling == "multi_reference_consolidation"
            and nuggets is None
        ):
            nuggets = scorer.consolidate_references(references, policy=policy)
            result = scorer.score(
                prediction,
                reference="",
                policy=policy,
                nuggets=list(nuggets),
            )
        else:
            result = scorer.score(
                prediction,
                reference,
                policy=policy,
                nuggets=nuggets,
            )

        score = float(result.get("score", 0.0))
        if binary:
            return 1.0 if score >= threshold else 0.0
        return score

    _reward.__name__ = f"{task_name}_reward"
    _reward.__qualname__ = f"{task_name}_reward"
    return _reward

