from __future__ import annotations

import pytest

from konash.training.execution import plan_training_execution


def test_auto_uses_together_for_single_iteration():
    plan = plan_training_execution(iterations=1, synthesis_rollout_backend="auto")

    assert plan.synthesis_rollout_backend == "together"
    assert plan.requires_remote_full_pipeline is False


def test_auto_uses_shadeform_for_multi_iteration():
    plan = plan_training_execution(iterations=2, synthesis_rollout_backend="auto")

    assert plan.synthesis_rollout_backend == "shadeform"
    assert plan.requires_remote_full_pipeline is True


def test_explicit_together_rejects_multi_iteration():
    with pytest.raises(ValueError, match="Multi-iteration training requires synthesis \\+ rollouts on shadeform"):
        plan_training_execution(iterations=2, synthesis_rollout_backend="together")
