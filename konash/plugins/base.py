from __future__ import annotations

from typing import Any, Dict, List, Optional


class LifecyclePlugin:
    """Base class for plugins that hook into the agent execution lifecycle.

    Subclasses can override any hook to inject behaviour at specific points
    in the step loop.  Default implementations are intentionally no-ops so
    that concrete plugins only need to override the hooks they care about.
    """

    # -- pre / post step hooks ------------------------------------------------

    def before_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called before each agent step.

        Parameters
        ----------
        step_index:
            Zero-based index of the upcoming step.
        history:
            The current conversation history (list of message dicts).

        Returns
        -------
        An optional dict of overrides that the environment may merge into
        the step context (e.g. ``{"skip": True}`` to skip the step).
        ``None`` means "no overrides".
        """
        return None

    def after_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        step_result: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called after each agent step completes.

        Parameters
        ----------
        step_index:
            Zero-based index of the step that just finished.
        history:
            The conversation history *after* the step has been appended.
        step_result:
            Dict describing what happened during the step (model output,
            tool calls, token counts, etc.).
        """
        return None

    # -- tool-call rewriting --------------------------------------------------

    def rewrite_tool_call(
        self,
        tool_name: str = "",
        tool_input: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Optionally rewrite a tool call before it is executed.

        Parameters
        ----------
        tool_name:
            Name of the tool the model wants to invoke.
        tool_input:
            The arguments dict the model provided.

        Returns
        -------
        A dict ``{"tool_name": ..., "tool_input": ...}`` if the call should
        be rewritten, or ``None`` to leave it unchanged.
        """
        return None

    # -- termination override -------------------------------------------------

    def override_termination(
        self,
        should_terminate: bool = False,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Override the default termination decision.

        Parameters
        ----------
        should_terminate:
            The environment's current decision about whether to stop.
        history:
            The current conversation history.

        Returns
        -------
        ``True`` to force termination, ``False`` to force continuation,
        or ``None`` to accept the environment's decision.
        """
        return None

    # -- history reshaping ----------------------------------------------------

    def reshape_history(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        """Optionally reshape the conversation history before the next LLM call.

        This hook is useful for compression, summarisation, or context-window
        management plugins.

        Parameters
        ----------
        history:
            The full conversation history.

        Returns
        -------
        A new history list to use instead, or ``None`` to keep the original.
        """
        return None
