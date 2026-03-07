from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional


class Dispatcher:
    """Coordinates strategy execution, environment creation, and plugin wiring.

    The dispatcher is the main entry-point used by training loops, evaluation
    harnesses, and serving runtimes to produce rollouts.  It accepts:

    * A *strategy* (``StandardStrategy``, ``ParallelThinkingStrategy``, etc.)
      that decides how many rollouts to run and how to aggregate them.
    * An *environment_factory* callable that returns a fresh ``Environment``
      for each rollout.
    * An optional list of *plugins* that are registered on every environment.
    """

    def __init__(
        self,
        *,
        strategy: Any = None,
        environment_factory: Callable[..., Any] | None = None,
        plugins: List[Any] | None = None,
    ) -> None:
        self.strategy = strategy
        self.environment_factory = environment_factory
        self._plugins: List[Any] = list(plugins) if plugins else []

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: Any) -> None:
        """Add a lifecycle plugin that will be injected into every new environment."""
        self._plugins.append(plugin)

    # ------------------------------------------------------------------
    # Single prompt
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        agent: Any,
        *,
        environment: Any | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute a single prompt through the strategy and return the result."""
        env = environment or self._make_environment()
        return self.dispatch(prompt=prompt, agent=agent, environment=env, **kwargs)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def run_batch(
        self,
        prompts: List[str],
        agent: Any,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of prompts, one environment per prompt.

        Returns a list of result dicts (one per prompt, in order).
        """
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            env = self._make_environment()
            result = self.dispatch(prompt=prompt, agent=agent, environment=env, **kwargs)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Rollout collection (training-oriented)
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        prompts: List[str],
        agent: Any,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run a batch and return the full rollout data suitable for training.

        This is a thin wrapper over ``run_batch`` that ensures each result
        includes trajectory and reward information.
        """
        rollouts = self.run_batch(prompts, agent, **kwargs)
        # Ensure every rollout has the expected keys
        for rollout in rollouts:
            rollout.setdefault("trajectory", [])
            rollout.setdefault("reward", None)
            rollout.setdefault("history", [])
        return rollouts

    # ------------------------------------------------------------------
    # Dispatch (routing)
    # ------------------------------------------------------------------

    def dispatch(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Route the prompt to the configured strategy for execution.

        If no strategy is set, falls back to a direct environment episode.
        """
        if self.strategy is not None:
            return self.strategy.execute(
                prompt=prompt,
                agent=agent,
                environment=environment,
                **kwargs,
            )

        # Fallback: run the environment directly
        environment.reset(prompt=prompt)
        result = environment.run_episode(agent=agent, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Streaming generator
    # ------------------------------------------------------------------

    def stream_rollouts(
        self,
        prompts: List[str],
        agent: Any,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield rollout results one at a time as they complete.

        This is useful for large-scale collection where the caller wants to
        process or persist results incrementally.
        """
        for prompt in prompts:
            env = self._make_environment()
            result = self.dispatch(prompt=prompt, agent=agent, environment=env, **kwargs)
            yield result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_environment(self) -> Any:
        """Create a fresh environment, injecting any registered plugins."""
        if self.environment_factory is not None:
            env = self.environment_factory()
        else:
            # Import locally to avoid circular imports at module level
            from konash.harness.environment import Environment
            env = Environment()

        # Inject dispatcher-level plugins
        if hasattr(env, "plugins"):
            if env.plugins is None:
                env.plugins = []
            env.plugins.extend(self._plugins)

        return env
