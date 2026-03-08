from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from konash.rewards.base import RewardFn, TaskRewardSpec
from konash.rewards.nugget import make_nugget_reward
from konash.rewards.tasks import TASK_REWARD_SPECS


class RewardRegistry:
    """Central registry for task-specific reward functions."""

    default_tasks: Set[str] = set(TASK_REWARD_SPECS.keys())

    def __init__(self) -> None:
        self._rewards: Dict[str, RewardFn] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        for task_name, spec in TASK_REWARD_SPECS.items():
            self.register_default(spec)

    def register(
        self,
        name: str,
        fn: RewardFn,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        if name in self._rewards and not overwrite:
            raise ValueError(
                f"Reward '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._rewards[name] = fn
        self._metadata[name] = metadata if metadata is not None else {}

    def register_default(self, spec: TaskRewardSpec) -> None:
        reward_fn = make_nugget_reward(
            spec.policy_name,
            threshold=spec.threshold,
            binary=spec.binary,
        )
        metadata = {
            "source": "default",
            "task": spec.task_name,
            "policy": spec.policy_name,
            "threshold": spec.threshold,
            "binary": spec.binary,
            "description": spec.description,
            **spec.metadata,
        }
        self.register(spec.task_name, reward_fn, metadata=metadata, overwrite=True)

    def get(
        self,
        name: str,
        **kwargs: Any,
    ) -> RewardFn:
        if name not in self._rewards:
            raise KeyError(f"No reward registered under '{name}'.")
        return self._rewards[name]

    def metadata(self, name: str) -> Dict[str, Any]:
        if name not in self._metadata:
            raise KeyError(f"No reward metadata registered under '{name}'.")
        return dict(self._metadata[name])

    def compose(
        self,
        names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> RewardFn:
        fns = [self.get(n) for n in names]

        if weights is None:
            w = [1.0 / len(fns)] * len(fns)
        else:
            if len(weights) != len(fns):
                raise ValueError(
                    f"Length mismatch: {len(weights)} weights for "
                    f"{len(fns)} rewards."
                )
            total = sum(weights)
            w = [wi / total for wi in weights] if total != 0 else [0.0] * len(weights)

        def _composite(*args: Any, **kw: Any) -> float:
            return sum(wi * fn(*args, **kw) for wi, fn in zip(w, fns))

        return _composite

    def list_rewards(self, **kwargs: Any) -> List[str]:
        return sorted(self._rewards.keys())


__all__ = [
    "RewardFn",
    "RewardRegistry",
    "TaskRewardSpec",
    "TASK_REWARD_SPECS",
    "make_nugget_reward",
]
