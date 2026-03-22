"""Training execution planning.

This module makes stage-level execution decisions explicit so training does
not silently switch modes mid-run.

Stage terminology:
- **Stage 1/2 (synthesis + rollouts):** Runs on the provisioned remote GPU.
- **Stage 3 (OAPL training):** Runs on the same remote GPU with the model
  loaded locally (Unsloth engine).

Execution modes:
- ``remote_full`` — Stages 1–3 all run on the same remote GPU.  Required
  for multi-iteration runs where the trained model must stay next to
  the compute for bootstrapping later iterations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExecutionPlan:
    """Resolved training execution strategy.

    Attributes
    ----------
    requested_mode:
        What the caller originally asked for.
    synthesis_rollout_backend:
        Resolved execution mode: ``remote_full``.
    requires_remote_full_pipeline:
        True when the entire pipeline (synthesis + rollouts + OAPL) must
        run on the same remote GPU.
    """

    requested_mode: str
    synthesis_rollout_backend: str
    requires_remote_full_pipeline: bool


def plan_training_execution(
    *,
    iterations: int,
    synthesis_rollout_backend: str = "remote_full",
) -> TrainingExecutionPlan:
    """Resolve how synthesis, rollouts, and OAPL should be executed.

    Parameters
    ----------
    iterations:
        Number of RL training iterations.
    synthesis_rollout_backend:
        Execution mode. Training now only supports ``remote_full``.

    Rules
    -----
    - ``remote_full`` — All stages run on the provisioned GPU. This is
      the only supported training path.
    """
    mode = synthesis_rollout_backend

    if mode != "remote_full":
        raise ValueError(
            f"Unknown synthesis/rollout backend: {synthesis_rollout_backend!r}. "
            "Training only supports remote_full."
        )

    return TrainingExecutionPlan(
        requested_mode=synthesis_rollout_backend,
        synthesis_rollout_backend=mode,
        requires_remote_full_pipeline=True,
    )
