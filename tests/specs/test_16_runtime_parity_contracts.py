from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_runtime_mode_enum_covers_collection_training_eval_and_serving():
    runtime_module = load_symbol(SymbolSpec("konash.harness.runtime"))
    runtime_mode = getattr(runtime_module, "RuntimeMode", None)
    assert runtime_mode is not None, "Missing RuntimeMode"

    expected_names = {"COLLECT", "TRAIN_EVAL", "EVAL", "SERVE"}
    actual_names = {name for name in dir(runtime_mode) if name.isupper()}
    assert expected_names.issubset(actual_names)


def test_harness_factory_builds_identical_interfaces_for_each_runtime_mode():
    runtime_module = load_symbol(SymbolSpec("konash.harness.runtime"))
    build_runtime = getattr(runtime_module, "build_runtime", None)
    assert callable(build_runtime), "build_runtime must be callable"


def test_environment_plugin_hooks_can_rewrite_tool_calls_and_override_termination():
    base_cls = load_symbol(SymbolSpec("konash.plugins.base", "LifecyclePlugin"))
    assert inspect.isclass(base_cls)
    assert_has_attrs(
        base_cls,
        [
            "before_step",
            "after_step",
            "rewrite_tool_call",
            "override_termination",
            "reshape_history",
        ],
        "LifecyclePlugin",
    )


def test_parallel_thinking_routes_finished_rollouts_to_final_aggregation_rollout():
    strategy_cls = load_symbol(SymbolSpec("konash.harness.strategy", "ParallelThinkingStrategy"))
    assert inspect.isclass(strategy_cls)
    assert_has_attrs(
        strategy_cls,
        ["spawn_parallel_rollouts", "route_to_aggregation_rollout", "execute"],
        "ParallelThinkingStrategy",
    )


def test_value_guided_agent_is_a_drop_in_agent_variant():
    cls = load_symbol(SymbolSpec("konash.agent", "ValueGuidedAgent"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["generate_step", "candidate_width", "value_model"],
        "ValueGuidedAgent",
    )
