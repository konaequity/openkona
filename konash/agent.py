from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional


class Agent:
    """Core agent that wraps an injected LLM client to drive multi-step rollouts.

    The LLM client is any object that exposes a ``generate(messages, **kwargs)``
    method returning an assistant message dict (at minimum ``{"role": "assistant",
    "content": "..."}``).  Tool calls, adapters, and history compression are
    handled at this layer so that the harness environment stays LLM-agnostic.
    """

    llm_client = None

    def __init__(
        self,
        llm_client: Any = None,
        *,
        system_prompt: str | None = None,
        max_steps: int = 20,
        stop_sequences: List[str] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.stop_sequences = stop_sequences or []
        self._active_adapters: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send *messages* to the LLM client and return the raw response.

        The caller is responsible for building the message list (including any
        system prompt).  Extra ``kwargs`` are forwarded to the client.
        """
        if self.llm_client is None:
            raise RuntimeError("No llm_client configured on this Agent.")
        return self.llm_client.generate(messages, **kwargs)

    # ------------------------------------------------------------------
    # Step-level generation
    # ------------------------------------------------------------------

    def generate_step(
        self,
        conversation_history: List[Dict[str, str]],
        available_tools: List[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Produce a single agent step (one LLM turn).

        Returns a dict with at least ``{"role": "assistant", "content": ...}``.
        If the model emits a tool-call the dict may also contain a
        ``"tool_calls"`` key.
        """
        messages = self._build_messages(conversation_history)
        gen_kwargs: Dict[str, Any] = dict(kwargs)
        if available_tools:
            gen_kwargs["tools"] = available_tools
        if self.stop_sequences:
            gen_kwargs.setdefault("stop", self.stop_sequences)

        response = self.generate(messages, **gen_kwargs)
        return response

    # ------------------------------------------------------------------
    # Full rollout
    # ------------------------------------------------------------------

    def generate_rollout(
        self,
        prompt: str,
        environment: Any = None,
        *,
        max_steps: int | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run a complete episode loop and return the trajectory.

        If an *environment* is provided its ``step`` / ``run_episode`` contract
        is used; otherwise we fall back to a simple generate-until-done loop.
        """
        steps = max_steps or self.max_steps

        if environment is not None:
            environment.reset(prompt=prompt)
            result = environment.run_episode(agent=self, max_steps=steps, **kwargs)
            return result

        # Standalone loop (no environment)
        history: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        trajectory: List[Dict[str, Any]] = []

        for _step_idx in range(steps):
            response = self.generate_step(history)
            trajectory.append(response)
            history.append(response)

            # Check for natural termination
            if self._is_terminal(response):
                break

        return {
            "prompt": prompt,
            "history": history,
            "trajectory": trajectory,
            "final_answer": self.extract_final_answer(history),
        }

    # ------------------------------------------------------------------
    # History compression
    # ------------------------------------------------------------------

    def compress_history(
        self,
        conversation_history: List[Dict[str, str]],
        *,
        target_tokens: int | None = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Ask the LLM to produce a shorter version of *conversation_history*.

        Returns a new message list whose semantic content is preserved but
        whose token footprint is reduced.
        """
        if not conversation_history:
            return []

        # Build a compression prompt
        serialized = "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
            for m in conversation_history
        )
        compression_prompt = (
            "Compress the following conversation into a concise summary that "
            "preserves all critical facts, tool results, and reasoning steps. "
            "Return ONLY the summary.\n\n" + serialized
        )
        if target_tokens is not None:
            compression_prompt += f"\n\nTarget length: roughly {target_tokens} tokens."

        messages = [{"role": "user", "content": compression_prompt}]
        response = self.generate(messages, **kwargs)
        summary_content = response.get("content", "") if isinstance(response, dict) else str(response)

        compressed: List[Dict[str, str]] = [
            {"role": "system", "content": f"[Compressed history] {summary_content}"},
        ]
        return compressed

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_final_answer(
        self,
        conversation_history: List[Dict[str, str]],
        **kwargs,
    ) -> str | None:
        """Extract the final answer from the conversation history.

        Walks the history backwards looking for the last assistant message
        that does not appear to be a tool call.
        """
        for message in reversed(conversation_history):
            if message.get("role") != "assistant":
                continue
            # Skip messages that are purely tool calls with no textual content
            if message.get("tool_calls") and not message.get("content"):
                continue
            content = message.get("content", "")
            if content:
                return content
        return None

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def load_adapter(
        self,
        adapter_path: str,
        *,
        adapter_name: str | None = None,
        weight: float = 1.0,
        **kwargs,
    ) -> None:
        """Load a LoRA adapter and merge it into the active set."""
        name = adapter_name or adapter_path.rstrip("/").split("/")[-1]
        adapter_record = {
            "path": adapter_path,
            "name": name,
            "weight": weight,
            **kwargs,
        }
        self._active_adapters.append(adapter_record)

        # Delegate to the LLM client if it supports adapter loading
        if hasattr(self.llm_client, "load_adapter"):
            self.llm_client.load_adapter(adapter_path, name=name, weight=weight, **kwargs)

    def unload_adapter(
        self,
        adapter_name: str | None = None,
        **kwargs,
    ) -> None:
        """Unload a LoRA adapter (or all adapters if *adapter_name* is ``None``)."""
        if adapter_name is None:
            removed = list(self._active_adapters)
            self._active_adapters.clear()
        else:
            removed = [a for a in self._active_adapters if a["name"] == adapter_name]
            self._active_adapters = [
                a for a in self._active_adapters if a["name"] != adapter_name
            ]

        # Delegate to the LLM client if it supports adapter unloading
        if hasattr(self.llm_client, "unload_adapter"):
            for adapter in removed:
                self.llm_client.unload_adapter(adapter["name"], **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepend the system prompt (if any) to the conversation history."""
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(conversation_history)
        return messages

    @staticmethod
    def _is_terminal(response: Dict[str, Any]) -> bool:
        """Heuristic: a response is terminal if it has content and no tool calls."""
        has_tool_calls = bool(response.get("tool_calls"))
        has_content = bool(response.get("content"))
        return has_content and not has_tool_calls


class ValueGuidedAgent:
    """Agent variant that generates multiple candidate continuations per step,
    scores each with a value model, and selects the best one.

    Drop-in replacement for ``Agent.generate_step`` when plugged into an
    environment or strategy that expects the same interface.
    """

    candidate_width = 2
    value_model = None

    def __init__(
        self,
        llm_client: Any = None,
        *,
        candidate_width: int = 2,
        value_model: Any = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        stop_sequences: List[str] | None = None,
    ) -> None:
        # Compose an inner Agent for actual generation
        self._agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            max_steps=max_steps,
            stop_sequences=stop_sequences,
        )
        self.candidate_width = candidate_width
        self.value_model = value_model
        self.llm_client = llm_client

    def generate_step(
        self,
        conversation_history: List[Dict[str, str]],
        available_tools: List[Dict[str, Any]] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate *candidate_width* candidates, score them, pick the best."""
        candidates: List[Dict[str, Any]] = []
        for _ in range(self.candidate_width):
            candidate = self._agent.generate_step(
                conversation_history,
                available_tools=available_tools,
                **kwargs,
            )
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError("No candidates generated.")

        # If no value model is available, fall back to first candidate
        if self.value_model is None:
            return candidates[0]

        # Score each candidate
        best_candidate = candidates[0]
        best_score = float("-inf")
        for candidate in candidates:
            augmented_history = list(conversation_history) + [candidate]
            score = self.value_model.score_partial_rollout(augmented_history)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate
