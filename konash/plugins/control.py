from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class StepBudgetPlugin:
    """Lifecycle plugin that enforces a maximum number of agent steps.

    When the step budget is exhausted the plugin signals termination so
    the environment can stop the episode gracefully.

    Parameters
    ----------
    max_steps:
        Maximum number of steps the agent is allowed to take.
        Defaults to 50.
    warn_at:
        Optionally emit a warning when this many steps remain.
        Defaults to 5.
    """

    def __init__(
        self,
        max_steps: int = 50,
        warn_at: int = 5,
    ) -> None:
        self.max_steps = max_steps
        self.warn_at = warn_at
        self._steps_taken: int = 0
        self._exhausted: bool = False

    # -- lifecycle hooks ------------------------------------------------------

    def before_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Block the step if the budget has been exhausted.

        Returns
        -------
        A dict with ``"terminate": True`` when the budget is used up,
        ``None`` otherwise.
        """
        if self._steps_taken >= self.max_steps:
            self._exhausted = True
            return {
                "terminate": True,
                "reason": f"Step budget exhausted ({self.max_steps} steps).",
            }

        remaining = self.max_steps - self._steps_taken
        result: Optional[Dict[str, Any]] = None
        if remaining <= self.warn_at:
            result = {
                "warning": f"Only {remaining} step(s) remaining in budget.",
            }
        return result

    def after_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        step_result: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Increment the internal step counter."""
        self._steps_taken += 1

    # -- query helpers --------------------------------------------------------

    @property
    def remaining(self) -> int:
        """Number of steps remaining in the budget."""
        return max(0, self.max_steps - self._steps_taken)

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def reset(self) -> None:
        """Reset the step counter (e.g. for a new episode)."""
        self._steps_taken = 0
        self._exhausted = False


class ToolGatePlugin:
    """Lifecycle plugin that restricts which tools the agent is allowed to use.

    Tools can be allow-listed or deny-listed.  If an ``allowed_tools`` set is
    provided, only those tools may be called.  If a ``denied_tools`` set is
    provided, those tools are blocked regardless of the allow list.

    Parameters
    ----------
    allowed_tools:
        If non-empty, only these tool names are permitted.
    denied_tools:
        These tool names are always blocked (takes precedence over
        ``allowed_tools``).
    on_deny:
        Action to take when a tool is denied.  One of ``"block"`` (silently
        skip the call) or ``"error"`` (inject an error message into the
        conversation).  Defaults to ``"error"``.
    """

    def __init__(
        self,
        allowed_tools: Optional[Set[str]] = None,
        denied_tools: Optional[Set[str]] = None,
        on_deny: str = "error",
    ) -> None:
        self.allowed_tools: Set[str] = allowed_tools if allowed_tools is not None else set()
        self.denied_tools: Set[str] = denied_tools if denied_tools is not None else set()
        self.on_deny = on_deny
        self._denied_calls: List[Dict[str, Any]] = []

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is permitted by the current policy."""
        if tool_name in self.denied_tools:
            return False
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False
        return True

    # -- lifecycle hooks ------------------------------------------------------

    def before_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Inspect pending tool calls and gate disallowed ones.

        Parameters
        ----------
        tool_calls:
            A list of ``{"tool_name": ..., "tool_input": ...}`` dicts
            representing the calls the model wants to make this step.

        Returns
        -------
        A dict with ``"tool_calls"`` filtered to only allowed tools and
        ``"denied"`` listing what was blocked, or ``None`` if nothing was
        blocked.
        """
        if not tool_calls:
            return None

        allowed: List[Dict[str, Any]] = []
        denied: List[Dict[str, Any]] = []

        for call in tool_calls:
            name = call.get("tool_name", "")
            if self.is_tool_allowed(name):
                allowed.append(call)
            else:
                denied.append(call)
                self._denied_calls.append(call)

        if not denied:
            return None

        result: Dict[str, Any] = {"tool_calls": allowed, "denied": denied}
        if self.on_deny == "error":
            error_names = [c.get("tool_name", "") for c in denied]
            result["error_message"] = (
                f"Tool(s) not permitted by policy: {', '.join(error_names)}"
            )
        return result

    def after_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        step_result: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """No-op for ToolGatePlugin; gating happens in ``before_step``."""
        return None
