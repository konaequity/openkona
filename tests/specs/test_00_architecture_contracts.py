from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_callable, assert_has_attrs, load_symbol


def test_core_package_layout_exists():
    required_modules = [
        "konash",
        "konash.agent",
        "konash.harness.dispatcher",
        "konash.harness.environment",
        "konash.harness.strategy",
        "konash.plugins.compression",
        "konash.retrieval.vector_search",
        "konash.rewards",
    ]
    for module_name in required_modules:
        load_symbol(SymbolSpec(module_name))


def test_agent_contract_supports_generation_and_tool_use():
    agent_cls = load_symbol(SymbolSpec("konash.agent", "Agent"))
    assert inspect.isclass(agent_cls), "konash.agent.Agent must be a class"
    assert_has_attrs(
        agent_cls,
        ["generate", "generate_rollout", "compress_history", "load_adapter", "unload_adapter"],
        "konash.agent.Agent",
    )
    for method_name in ["generate", "generate_rollout", "compress_history", "load_adapter"]:
        assert_callable(getattr(agent_cls, method_name), f"Agent.{method_name}")


def test_dispatcher_coordinates_environment_and_strategy():
    dispatcher_cls = load_symbol(SymbolSpec("konash.harness.dispatcher", "Dispatcher"))
    assert inspect.isclass(dispatcher_cls), "Dispatcher must be a class"
    assert_has_attrs(
        dispatcher_cls,
        ["run", "run_batch", "collect_rollouts", "register_plugin"],
        "Dispatcher",
    )


def test_environment_owns_interaction_loop_and_rewards():
    env_cls = load_symbol(SymbolSpec("konash.harness.environment", "Environment"))
    assert inspect.isclass(env_cls), "Environment must be a class"
    assert_has_attrs(
        env_cls,
        ["reset", "step", "run_episode", "compute_reward", "available_tools"],
        "Environment",
    )


def test_strategy_surface_supports_standard_parallel_and_vgs():
    strategy_module = load_symbol(SymbolSpec("konash.harness.strategy"))
    for symbol in ["StandardStrategy", "ParallelThinkingStrategy", "ValueGuidedSearchStrategy"]:
        cls = getattr(strategy_module, symbol, None)
        assert cls is not None, f"Missing strategy class {symbol}"
        assert inspect.isclass(cls), f"{symbol} must be a class"
        assert_has_attrs(cls, ["execute"], symbol)


def test_compression_plugin_has_lifecycle_hooks():
    plugin_cls = load_symbol(SymbolSpec("konash.plugins.compression", "CompressionPlugin"))
    assert inspect.isclass(plugin_cls), "CompressionPlugin must be a class"
    assert_has_attrs(
        plugin_cls,
        ["should_compress", "compress", "before_step", "after_step"],
        "CompressionPlugin",
    )


def test_vector_search_tool_is_the_primary_grounding_tool():
    tool_cls = load_symbol(SymbolSpec("konash.retrieval.vector_search", "VectorSearchTool"))
    assert inspect.isclass(tool_cls), "VectorSearchTool must be a class"
    assert_has_attrs(tool_cls, ["index", "search", "batch_search"], "VectorSearchTool")
