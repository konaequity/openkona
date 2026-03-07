from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List, Optional


class RuntimeMode(enum.Enum):
    COLLECT = "collect"
    TRAIN_EVAL = "train_eval"
    EVAL = "eval"
    SERVE = "serve"


def build_runtime(
    mode: RuntimeMode,
    *,
    agent: Any = None,
    llm_client: Any = None,
    environment_factory: Callable[..., Any] | None = None,
    strategy: Any = None,
    plugins: List[Any] | None = None,
    value_model: Any = None,
    num_rollouts: int = 5,
    candidate_width: int = 2,
    max_steps: int = 20,
    **kwargs,
) -> Dict[str, Any]:
    """Factory that constructs a Dispatcher + Strategy + Environment for a
    given :class:`RuntimeMode`.

    Returns a dict with ``"dispatcher"``, ``"strategy"``, and
    ``"environment_factory"`` keys so that the caller can drive execution.

    The four modes map to the following defaults:

    * **COLLECT** -- ``StandardStrategy`` with a vanilla environment.  Designed
      for offline rollout collection (training data).
    * **TRAIN_EVAL** -- ``StandardStrategy``.  Identical interface to COLLECT
      so that the same evaluation harness works during training checkpoints.
    * **EVAL** -- ``ParallelThinkingStrategy`` for robust evaluation via
      multiple independent rollouts + aggregation.
    * **SERVE** -- ``ValueGuidedSearchStrategy`` for latency-aware serving
      with a value model selecting the best candidate at each step.
    """
    from konash.harness.dispatcher import Dispatcher
    from konash.harness.environment import Environment
    from konash.harness.strategy import (
        ParallelThinkingStrategy,
        StandardStrategy,
        ValueGuidedSearchStrategy,
    )

    plugins = list(plugins) if plugins else []

    # Default environment factory
    if environment_factory is None:
        def environment_factory(**_kw):
            return Environment(**{k: v for k, v in kwargs.items() if k in {
                "tool_executor", "reward_functions", "token_budget", "available_tools",
            }})

    # Build mode-appropriate strategy
    if strategy is not None:
        chosen_strategy = strategy
    elif mode == RuntimeMode.COLLECT:
        chosen_strategy = StandardStrategy(max_steps=max_steps)
    elif mode == RuntimeMode.TRAIN_EVAL:
        chosen_strategy = StandardStrategy(max_steps=max_steps)
    elif mode == RuntimeMode.EVAL:
        chosen_strategy = ParallelThinkingStrategy(
            num_rollouts=num_rollouts,
            max_steps=max_steps,
        )
    elif mode == RuntimeMode.SERVE:
        chosen_strategy = ValueGuidedSearchStrategy(
            candidate_width=candidate_width,
            value_model=value_model,
            max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unsupported runtime mode: {mode}")

    dispatcher = Dispatcher(
        strategy=chosen_strategy,
        environment_factory=environment_factory,
        plugins=plugins,
    )

    return {
        "dispatcher": dispatcher,
        "strategy": chosen_strategy,
        "environment_factory": environment_factory,
        "mode": mode,
    }
