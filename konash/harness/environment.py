from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional


class Environment:
    """Manages the interaction loop between an agent and its tools/plugins.

    The environment owns:
    * ``tool_executor`` -- callable that executes a tool call and returns an
      observation dict ``{"role": "tool", "content": ...}``.
    * ``reward_functions`` -- list of callables that score the agent's
      ``final_answer`` against a reference answer.
    * ``plugins`` -- list of ``LifecyclePlugin`` instances whose hooks fire
      around every step.
    * ``conversation_history`` -- the running message list for the current
      episode.
    * ``token_budget`` -- optional cap on total tokens consumed in an episode.
    * ``available_tools`` -- list of tool-schema dicts exposed to the agent.
    """

    tool_executor = None
    reward_functions = None
    plugins = None
    conversation_history = None
    token_budget = None
    available_tools = None

    def __init__(
        self,
        *,
        tool_executor: Callable[..., Dict[str, Any]] | None = None,
        reward_functions: List[Callable[..., float]] | None = None,
        plugins: List[Any] | None = None,
        token_budget: int | None = None,
        available_tools: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.reward_functions = reward_functions or []
        self.plugins = plugins or []
        self.conversation_history: List[Dict[str, str]] = []
        self.token_budget = token_budget
        self.available_tools = available_tools or []
        self._tokens_used: int = 0
        self._step_count: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, *, prompt: str | None = None, **kwargs) -> None:
        """Clear all episode state and optionally seed the history with a user prompt."""
        self.conversation_history = []
        self._tokens_used = 0
        self._step_count = 0
        self._done = False

        if prompt is not None:
            self.conversation_history.append({"role": "user", "content": prompt})

        # Notify plugins
        for plugin in self.plugins:
            if hasattr(plugin, "reset"):
                plugin.reset()
            if hasattr(plugin, "on_reset"):
                plugin.on_reset(self)

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(
        self,
        agent: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute one agent step: generate -> execute tools -> run plugin hooks.

        Returns a step record::

            {
                "agent_response": {...},
                "tool_results": [...],
                "done": bool,
                "step_index": int,
            }
        """
        # --- before_step plugin hooks ---
        compression_events: List[Dict[str, Any]] = []
        for plugin in self.plugins:
            if hasattr(plugin, "before_step"):
                override = plugin.before_step(
                    step_index=self._step_count,
                    history=self.conversation_history,
                )
                # A plugin may signal early termination
                if isinstance(override, dict) and override.get("terminate"):
                    self._done = True
                    return {
                        "agent_response": None,
                        "tool_results": [],
                        "done": True,
                        "step_index": self._step_count,
                    }
                if isinstance(override, dict) and override.get("history") is not None:
                    self.conversation_history = list(override["history"])
                    # Record compression event for the trajectory
                    if isinstance(override, dict) and override.get("compression_event"):
                        compression_events.append(override["compression_event"])

        # --- Agent generates a step ---
        response = agent.generate_step(
            self.conversation_history,
            available_tools=self.available_tools,
            **kwargs,
        )
        self.conversation_history.append(response)
        self._step_count += 1

        # Estimate token usage (rough heuristic: 4 chars ~ 1 token)
        content = response.get("content", "") or ""
        self._tokens_used += max(len(content) // 4, 1)

        # --- Tool execution ---
        tool_results: List[Dict[str, Any]] = []
        tool_calls = self._filter_tool_calls(response.get("tool_calls", []), tool_results)
        for tool_call in tool_calls:
            # Let plugins rewrite the tool call
            rewritten = tool_call
            for plugin in self.plugins:
                if hasattr(plugin, "rewrite_tool_call"):
                    t_name = (
                        rewritten.get("function", {}).get("name", "")
                        if isinstance(rewritten, dict) else ""
                    )
                    result = plugin.rewrite_tool_call(
                        tool_name=t_name,
                        tool_input=rewritten,
                        environment=self,
                    )
                    if result is not None:
                        rewritten = result

            # Execute the tool
            if self.tool_executor is not None:
                observation = self.tool_executor(rewritten)
            else:
                observation = {
                    "role": "tool",
                    "content": f"[No tool executor configured for {rewritten}]",
                }
            if (
                isinstance(observation, dict)
                and isinstance(rewritten, dict)
                and rewritten.get("id")
                and "tool_call_id" not in observation
            ):
                observation["tool_call_id"] = rewritten["id"]
            if (
                isinstance(observation, dict)
                and isinstance(rewritten, dict)
                and isinstance(rewritten.get("function"), dict)
                and "name" not in observation
            ):
                function_name = rewritten["function"].get("name")
                if function_name:
                    observation["name"] = function_name
            tool_results.append(observation)
            self.conversation_history.append(observation)
            self._tokens_used += max(len(observation.get("content", "")) // 4, 1)

        # --- Determine termination ---
        done = self._check_termination(response)

        # --- after_step plugin hooks ---
        for plugin in self.plugins:
            if hasattr(plugin, "after_step"):
                plugin.after_step(
                    step_index=self._step_count,
                    history=self.conversation_history,
                    step_result={"agent_response": response, "tool_results": tool_results},
                )

        # Let plugins override termination
        for plugin in self.plugins:
            if hasattr(plugin, "override_termination"):
                override = plugin.override_termination(
                    should_terminate=done,
                    history=self.conversation_history,
                    environment=self,
                )
                if override is not None:
                    done = bool(override)

        # Let plugins reshape history (e.g. compression)
        for plugin in self.plugins:
            if hasattr(plugin, "reshape_history"):
                reshaped = plugin.reshape_history(
                    history=self.conversation_history,
                    environment=self,
                )
                if reshaped is not None:
                    self.conversation_history = reshaped

        self._done = done

        step_record = {
            "agent_response": response,
            "tool_results": tool_results,
            "done": done,
            "step_index": self._step_count,
        }
        # Attach compression events that fired before this step
        if compression_events:
            step_record["compression_events"] = compression_events
        return step_record

    # ------------------------------------------------------------------
    # Full episode
    # ------------------------------------------------------------------

    def run_episode(
        self,
        agent: Any,
        *,
        max_steps: int = 20,
        reference_answer: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the agent until termination or *max_steps*, then compute reward.

        Returns::

            {
                "history": [...],
                "trajectory": [...],
                "reward": float | None,
                "final_answer": str | None,
                "steps": int,
            }
        """
        trajectory: List[Dict[str, Any]] = []

        for _ in range(max_steps):
            step_record = self.step(agent, **kwargs)
            trajectory.append(step_record)
            if step_record["done"]:
                break

        # If the agent exhausted max_steps without terminating (common with
        # GLM via Zhipu which always makes tool calls), force one final
        # generation with tool_choice="none" to get an answer.
        if not self._done and self.conversation_history:
            try:
                final_kwargs = dict(kwargs)
                final_kwargs["tool_choice"] = "none"
                response = agent.generate_step(
                    self.conversation_history,
                    available_tools=self.available_tools,
                    **final_kwargs,
                )
                self.conversation_history.append(response)
                trajectory.append({
                    "agent_response": response,
                    "tool_results": [],
                    "done": True,
                    "step_index": self._step_count + 1,
                })
            except Exception:
                pass  # Best-effort; extract_final_answer will still try

        # Extract final answer
        final_answer = None
        if hasattr(agent, "extract_final_answer"):
            final_answer = agent.extract_final_answer(self.conversation_history)

        # Compute reward
        reward = self.compute_reward(
            reference_answer=reference_answer,
            final_answer=final_answer,
        )

        return {
            "history": list(self.conversation_history),
            "trajectory": trajectory,
            "reward": reward,
            "final_answer": final_answer,
            "steps": self._step_count,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        *,
        reference_answer: str | None = None,
        final_answer: str | None = None,
        **kwargs,
    ) -> float | None:
        """Evaluate the episode outcome by composing all registered reward functions.

        Each reward function is called with the extracted final answer plus
        any reference/answer information. Their outputs are summed.
        """
        if not self.reward_functions:
            return None

        total_reward = 0.0
        for reward_fn in self.reward_functions:
            score = reward_fn(
                final_answer or "",
                reference=reference_answer,
                final_answer=final_answer,
                **kwargs,
            )
            total_reward += float(score)

        return total_reward

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_termination(self, response: Dict[str, Any]) -> bool:
        """Decide whether the episode should end after this response."""
        # Budget exceeded
        if self.token_budget is not None and self._tokens_used >= self.token_budget:
            return True

        # No tool calls and content present -> agent is giving a final answer
        # GLM 4.5 Air puts answers in reasoning/reasoning_content instead of content
        has_tool_calls = bool(response.get("tool_calls"))
        has_content = bool(
            response.get("content")
            or response.get("reasoning_content")
            or response.get("reasoning")
        )
        if has_content and not has_tool_calls:
            return True

        return False

    def _filter_tool_calls(
        self,
        tool_calls: Any,
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply any plugin-level tool gating before executing tool calls."""
        if not isinstance(tool_calls, list):
            return []

        filtered = list(tool_calls)
        for plugin in self.plugins:
            if not hasattr(plugin, "is_tool_allowed"):
                continue

            allowed: List[Dict[str, Any]] = []
            denied_names: List[str] = []
            for call in filtered:
                name = self._tool_name(call)
                if plugin.is_tool_allowed(name):
                    allowed.append(call)
                else:
                    denied_names.append(name or "<unknown>")

            filtered = allowed
            if denied_names and getattr(plugin, "on_deny", "error") == "error":
                tool_results.append({
                    "role": "tool",
                    "name": "tool_gate",
                    "content": (
                        "Tool(s) not permitted by policy: "
                        + ", ".join(denied_names)
                    ),
                })

        return filtered

    @staticmethod
    def _tool_name(tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            return ""
        function = tool_call.get("function")
        if isinstance(function, dict):
            return str(function.get("name", ""))
        return str(tool_call.get("tool_name", ""))
