from __future__ import annotations

import pytest

from konash.training.execution import plan_training_execution


def test_remote_full_single_iteration():
    plan = plan_training_execution(iterations=1, synthesis_rollout_backend="remote_full")

    assert plan.synthesis_rollout_backend == "remote_full"
    assert plan.requires_remote_full_pipeline is True
    assert plan.requested_mode == "remote_full"


def test_remote_full_multi_iteration():
    plan = plan_training_execution(iterations=2, synthesis_rollout_backend="remote_full")

    assert plan.synthesis_rollout_backend == "remote_full"
    assert plan.requires_remote_full_pipeline is True
    assert plan.requested_mode == "remote_full"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Training only supports remote_full"):
        plan_training_execution(iterations=1, synthesis_rollout_backend="invalid")


def test_auto_is_rejected():
    with pytest.raises(ValueError, match="Training only supports remote_full"):
        plan_training_execution(iterations=1, synthesis_rollout_backend="auto")
