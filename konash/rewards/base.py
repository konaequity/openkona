from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict


RewardFn = Callable[..., float]


@dataclass(frozen=True)
class TaskRewardSpec:
    """Declarative description of a task reward.

    The KARL paper uses task-specific reward configuration while keeping the
    harness interface uniform. This object captures the minimum policy needed
    to build a concrete reward function for a task.
    """

    task_name: str
    policy_name: str
    threshold: float = 0.5
    binary: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

