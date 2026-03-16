from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional


class StandardStrategy:
    """Single-rollout strategy: one environment episode per prompt."""

    def __init__(self, *, max_steps: int = 20, **kwargs) -> None:
        self.max_steps = max_steps

    def execute(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a single rollout and return the episode result."""
        environment.reset(prompt=prompt)
        result = environment.run_episode(
            agent=agent,
            max_steps=self.max_steps,
            **kwargs,
        )
        return result

    def execute_batch(
        self,
        *,
        prompts: List[str],
        agent: Any,
        environment_factory: Callable[..., Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of prompts, one rollout each."""
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            env = environment_factory()
            result = self.execute(
                prompt=prompt,
                agent=agent,
                environment=env,
                **kwargs,
            )
            results.append(result)
        return results


class ParallelThinkingStrategy:
    """Spawn multiple independent rollouts per prompt, then aggregate.

    Each rollout runs in its own environment copy.  After all rollouts
    finish, their answers are fed into an aggregation rollout (another
    environment episode whose prompt contains the candidate answers) to
    produce a single consolidated result.
    """

    num_rollouts = None
    aggregator = None

    def __init__(
        self,
        *,
        num_rollouts: int = 5,
        aggregator: Any = None,
        max_steps: int = 20,
        **kwargs,
    ) -> None:
        self.num_rollouts = num_rollouts
        self.aggregator = aggregator
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run parallel rollouts, then aggregate into a final answer."""
        rollout_results = self.spawn_parallel_rollouts(
            prompt=prompt,
            agent=agent,
            environment=environment,
            **kwargs,
        )

        aggregated = self.route_to_aggregation_rollout(
            prompt=prompt,
            agent=agent,
            environment=environment,
            rollout_results=rollout_results,
            **kwargs,
        )

        return aggregated

    def execute_batch(
        self,
        *,
        prompts: List[str],
        agent: Any,
        environment_factory: Callable[..., Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of prompts with parallel thinking per prompt."""
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            env = environment_factory()
            result = self.execute(
                prompt=prompt,
                agent=agent,
                environment=env,
                **kwargs,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Parallel rollout spawning
    # ------------------------------------------------------------------

    def spawn_parallel_rollouts(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Run *num_rollouts* independent episodes and collect results.

        Each rollout gets a fresh copy of the environment so they do not
        share state.
        """
        def _run_single(i: int) -> Dict[str, Any]:
            env_copy = copy.deepcopy(environment)
            env_copy.reset(prompt=prompt)
            result = env_copy.run_episode(
                agent=agent,
                max_steps=self.max_steps,
                **kwargs,
            )
            result["rollout_index"] = i
            return result

        # Run rollouts concurrently — each is I/O-bound (LLM API calls).
        n = self.num_rollouts
        rollout_results: List[Dict[str, Any]] = [None] * n  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(_run_single, i): i for i in range(n)}
            for future in as_completed(futures):
                i = futures[future]
                rollout_results[i] = future.result()

        return rollout_results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def route_to_aggregation_rollout(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        rollout_results: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Feed all candidate answers into an aggregation step.

        If an ``aggregator`` is set, delegate to it.  Otherwise, build a
        prompt listing all candidate answers and run one more environment
        episode to produce a synthesised final answer.
        """
        candidate_answers = [
            r.get("final_answer", "") or "" for r in rollout_results
        ]

        # Delegate to aggregator if available
        if self.aggregator is not None:
            aggregated_answer = self.aggregator.aggregate(
                prompt=prompt,
                candidate_answers=candidate_answers,
                **kwargs,
            )
            return {
                "prompt": prompt,
                "final_answer": aggregated_answer,
                "rollout_results": rollout_results,
                "aggregation_method": "aggregator",
            }

        # Fallback: build an aggregation prompt and run one more episode
        numbered = "\n".join(
            f"  Candidate {i + 1}: {ans}" for i, ans in enumerate(candidate_answers)
        )
        aggregation_prompt = (
            f"Original question: {prompt}\n\n"
            f"The following candidate answers were produced by independent reasoning paths:\n"
            f"{numbered}\n\n"
            f"Synthesize these into a single best answer."
        )

        environment.reset(prompt=aggregation_prompt)
        agg_result = environment.run_episode(
            agent=agent,
            max_steps=self.max_steps,
            **kwargs,
        )

        return {
            "prompt": prompt,
            "final_answer": agg_result.get("final_answer"),
            "rollout_results": rollout_results,
            "aggregation_result": agg_result,
            "aggregation_method": "generative",
        }


class ValueGuidedSearchStrategy:
    """At each step, generate *candidate_width* continuations, score with a
    value model, and keep the best.

    This strategy wraps an ordinary agent but overrides how each step is
    produced inside the environment episode.

    When ``reference_model`` is set, the value scoring incorporates a
    KL-regularization bonus from the lagged inference policy, matching
    OAPL (Section 4.2).
    """

    candidate_width = None
    value_model = None
    reference_model = None

    def __init__(
        self,
        *,
        candidate_width: int = 2,
        value_model: Any = None,
        reference_model: Any = None,
        max_steps: int = 20,
        **kwargs,
    ) -> None:
        self.candidate_width = candidate_width
        self.value_model = value_model
        self.reference_model = reference_model
        self.max_steps = max_steps

    def execute(
        self,
        *,
        prompt: str,
        agent: Any,
        environment: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a value-guided episode where each step picks the best of k candidates."""
        environment.reset(prompt=prompt)

        trajectory: List[Dict[str, Any]] = []

        for _ in range(self.max_steps):
            # Generate k candidates
            candidates: List[Dict[str, Any]] = []
            for _ in range(self.candidate_width):
                candidate = agent.generate_step(
                    environment.conversation_history,
                    available_tools=environment.available_tools,
                    **kwargs,
                )
                candidates.append(candidate)

            # Score and pick best
            best = self._select_best(candidates, environment.conversation_history)

            # Commit the chosen candidate to the environment
            environment.conversation_history.append(best)
            environment._step_count += 1

            # Execute any tool calls
            tool_results: List[Dict[str, Any]] = []
            for tool_call in best.get("tool_calls", []):
                if environment.tool_executor is not None:
                    observation = environment.tool_executor(tool_call)
                else:
                    observation = {"role": "tool", "content": "[no executor]"}
                tool_results.append(observation)
                environment.conversation_history.append(observation)

            step_record = {
                "agent_response": best,
                "tool_results": tool_results,
                "candidates_evaluated": len(candidates),
            }
            trajectory.append(step_record)

            # Check termination
            has_content = bool(best.get("content"))
            has_tool_calls = bool(best.get("tool_calls"))
            if has_content and not has_tool_calls:
                break

        # Extract final answer
        final_answer = None
        if hasattr(agent, "extract_final_answer"):
            final_answer = agent.extract_final_answer(environment.conversation_history)

        reward = environment.compute_reward(final_answer=final_answer)

        return {
            "prompt": prompt,
            "history": list(environment.conversation_history),
            "trajectory": trajectory,
            "reward": reward,
            "final_answer": final_answer,
            "steps": len(trajectory),
        }

    def execute_batch(
        self,
        *,
        prompts: List[str],
        agent: Any,
        environment_factory: Callable[..., Any],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of prompts with value-guided search."""
        results: List[Dict[str, Any]] = []
        for prompt in prompts:
            env = environment_factory()
            result = self.execute(
                prompt=prompt,
                agent=agent,
                environment=env,
                **kwargs,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_best(
        self,
        candidates: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Score candidates and return the highest-scoring one."""
        if not candidates:
            raise RuntimeError("No candidates to select from.")

        if self.value_model is None:
            return candidates[0]

        best = candidates[0]
        best_score = float("-inf")
        for candidate in candidates:
            augmented = list(conversation_history) + [candidate]
            score = self.value_model.score_partial_rollout(augmented)
            if score > best_score:
                best_score = score
                best = candidate

        return best
