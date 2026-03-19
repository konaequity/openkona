"""Training execution planning.

This module makes the synthesis/rollout backend explicit so training does not
silently switch execution modes mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExecutionPlan:
    """Resolved training execution strategy."""

    requested_backend: str
    synthesis_rollout_backend: str
    requires_remote_full_pipeline: bool


def plan_training_execution(
    *,
    iterations: int,
    synthesis_rollout_backend: str = "auto",
) -> TrainingExecutionPlan:
    """Resolve how synthesis and rollouts should be executed.

    Rules:
    - ``together`` means synthesis + rollouts stay on API-backed inference.
      This is currently only supported for a single iteration.
    - ``shadeform`` means synthesis + rollouts + OAPL all run on the
      provisioned GPU.
    - ``auto`` chooses ``together`` for single-iteration runs and
      ``shadeform`` for multi-iteration runs, since bootstrapping later
      iterations requires the trained model to stay next to the compute.
    """
    backend = synthesis_rollout_backend
    if backend not in {"auto", "together", "shadeform"}:
        raise ValueError(
            "Unknown synthesis/rollout backend: "
            f"{synthesis_rollout_backend!r}. Use auto, together, or shadeform."
        )

    if backend == "auto":
        backend = "together" if iterations <= 1 else "shadeform"

    if backend == "together" and iterations > 1:
        raise ValueError(
            "Multi-iteration training requires synthesis + rollouts on shadeform "
            "so later iterations can bootstrap from the trained model. "
            "Use --synthesis-backend shadeform or --synthesis-backend auto."
        )

    return TrainingExecutionPlan(
        requested_backend=synthesis_rollout_backend,
        synthesis_rollout_backend=backend,
        requires_remote_full_pipeline=(backend == "shadeform"),
    )
