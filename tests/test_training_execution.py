from __future__ import annotations

import pytest

from scripts.train_oapl_unsloth import (
    _plan_synthesis_call_targets,
    _should_sleep_vllm_for_training,
    _target_synthesis_examples,
    _trim_messages_for_context,
)
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


def test_plan_synthesis_call_targets_respects_cap():
    assert _plan_synthesis_call_targets(synthesis_calls=2, max_examples=12) == [6, 6]
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=3) == [1, 1, 1]


def test_plan_synthesis_call_targets_spreads_work_across_workers():
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=12) == [3, 3, 3, 3]
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=10) == [3, 3, 2, 2]


def test_plan_synthesis_call_targets_without_cap_uses_all_calls():
    assert _plan_synthesis_call_targets(synthesis_calls=3, max_examples=None) == [8, 8, 8]


def test_should_sleep_vllm_for_intermediate_iteration():
    assert _should_sleep_vllm_for_training(iteration=0, total_iterations=2) is True
    assert _should_sleep_vllm_for_training(iteration=1, total_iterations=3) is True


def test_should_not_sleep_vllm_for_final_iteration():
    assert _should_sleep_vllm_for_training(iteration=0, total_iterations=1) is False
    assert _should_sleep_vllm_for_training(iteration=2, total_iterations=3) is False


def test_trim_messages_for_context_preserves_recent_history():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "u1 " + ("a" * 5000)},
        {"role": "assistant", "content": "a1 " + ("b" * 5000)},
        {"role": "tool", "content": "tool " + ("c" * 9000)},
        {"role": "user", "content": "latest question"},
    ]

    trimmed = _trim_messages_for_context(messages, max_context_tokens=2048)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "latest question"
    assert sum(len(m.get("content", "")) for m in trimmed) <= 4096
    assert any("[truncated]" in m.get("content", "") for m in trimmed)
