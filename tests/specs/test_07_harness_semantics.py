from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_agent_owns_llm_client_and_step_level_generation_surface():
    agent_cls = load_symbol(SymbolSpec("konash.agent", "Agent"))
    assert inspect.isclass(agent_cls)
    assert_has_attrs(
        agent_cls,
        ["llm_client", "generate_step", "extract_final_answer", "compress_history"],
        "konash.agent.Agent",
    )


def test_environment_exposes_execution_state_and_composable_rewards():
    env_cls = load_symbol(SymbolSpec("konash.harness.environment", "Environment"))
    assert inspect.isclass(env_cls)
    assert_has_attrs(
        env_cls,
        ["tool_executor", "reward_functions", "plugins", "conversation_history", "token_budget"],
        "konash.harness.environment.Environment",
    )


def test_dispatcher_and_strategies_share_batch_rollout_execution_contract():
    dispatcher_cls = load_symbol(SymbolSpec("konash.harness.dispatcher", "Dispatcher"))
    assert inspect.isclass(dispatcher_cls)
    assert_has_attrs(
        dispatcher_cls,
        ["dispatch", "run_batch", "stream_rollouts"],
        "konash.harness.dispatcher.Dispatcher",
    )

    standard_cls = load_symbol(SymbolSpec("konash.harness.strategy", "StandardStrategy"))
    parallel_cls = load_symbol(SymbolSpec("konash.harness.strategy", "ParallelThinkingStrategy"))
    vgs_cls = load_symbol(SymbolSpec("konash.harness.strategy", "ValueGuidedSearchStrategy"))

    for cls, label in [
        (standard_cls, "StandardStrategy"),
        (parallel_cls, "ParallelThinkingStrategy"),
        (vgs_cls, "ValueGuidedSearchStrategy"),
    ]:
        assert inspect.isclass(cls)
        assert_has_attrs(cls, ["execute", "execute_batch"], label)

    assert_has_attrs(parallel_cls, ["num_rollouts", "aggregator"], "ParallelThinkingStrategy")
    assert_has_attrs(vgs_cls, ["candidate_width", "value_model"], "ValueGuidedSearchStrategy")


def test_lifecycle_plugins_cover_compression_step_budgeting_and_tool_gating():
    compression_cls = load_symbol(SymbolSpec("konash.plugins.compression", "CompressionPlugin"))
    assert inspect.isclass(compression_cls)
    assert_has_attrs(
        compression_cls,
        ["threshold_tokens", "target_tokens", "should_compress", "compress"],
        "CompressionPlugin",
    )

    control_module = load_symbol(SymbolSpec("konash.plugins.control"))
    for symbol in ["StepBudgetPlugin", "ToolGatePlugin"]:
        cls = getattr(control_module, symbol, None)
        assert cls is not None, f"Missing plugin {symbol}"
        assert inspect.isclass(cls), f"{symbol} must be a class"
        assert_has_attrs(cls, ["before_step", "after_step"], symbol)


def test_runtime_mode_contract_requires_identical_interfaces_across_stages():
    runtime_module = load_symbol(SymbolSpec("konash.harness.runtime"))
    assert_has_attrs(runtime_module, ["RuntimeMode", "build_runtime"], "konash.harness.runtime")
