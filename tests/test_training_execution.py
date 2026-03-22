from __future__ import annotations

import pytest

from scripts.train_oapl_unsloth import _target_synthesis_examples
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


def test_sleep_wake_enabled_for_multi_iteration():
    plan = plan_training_execution(
        iterations=3, synthesis_rollout_backend="remote_full", sleep_wake=True,
    )
    assert plan.supports_sleep_wake is True


def test_sleep_wake_disabled_for_single_iteration():
    plan = plan_training_execution(
        iterations=1, synthesis_rollout_backend="remote_full", sleep_wake=True,
    )
    assert plan.supports_sleep_wake is False


def test_sleep_wake_defaults_to_false():
    plan = plan_training_execution(iterations=2, synthesis_rollout_backend="remote_full")
    assert plan.supports_sleep_wake is False


def test_target_synthesis_examples_uses_full_batch_without_cap():
    assert _target_synthesis_examples(current_count=0, max_examples=None) == 8


def test_target_synthesis_examples_respects_remaining_cap():
    assert _target_synthesis_examples(current_count=0, max_examples=3) == 3
    assert _target_synthesis_examples(current_count=2, max_examples=3) == 1


def test_target_synthesis_examples_returns_zero_when_done():
    assert _target_synthesis_examples(current_count=3, max_examples=3) == 0
    assert _target_synthesis_examples(current_count=4, max_examples=3) == 0
