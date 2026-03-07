from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


def _message_tokens(message: Dict[str, Any]) -> int:
    """Estimate the token count of a single conversation message."""
    total = 0
    content = message.get("content", "")
    if isinstance(content, str):
        total += _estimate_tokens(content)
    elif isinstance(content, list):
        # Multimodal messages may contain text blocks
        for block in content:
            if isinstance(block, dict):
                total += _estimate_tokens(block.get("text", ""))
            elif isinstance(block, str):
                total += _estimate_tokens(block)
    # Role / name / tool metadata overhead
    total += _estimate_tokens(message.get("role", ""))
    total += _estimate_tokens(message.get("name", ""))
    return total


def _history_tokens(history: List[Dict[str, Any]]) -> int:
    """Sum estimated tokens across all messages in a history."""
    return sum(_message_tokens(m) for m in history)


class CompressionPlugin:
    """Plugin that monitors conversation token usage and compresses history
    when a configurable threshold is exceeded.

    After compression, the history is reduced to approximately
    ``target_tokens`` by summarising or truncating older messages while
    preserving the system prompt and the most recent turns.

    Parameters
    ----------
    threshold_tokens:
        When estimated history tokens exceed this value,
        ``should_compress`` returns ``True``.  Defaults to 100_000.
    target_tokens:
        The desired token count after compression.  Defaults to 50_000.
    summarizer:
        Optional callable ``(List[Dict]) -> str`` that produces a summary
        of a batch of messages.  When not provided, a simple truncation
        strategy is used.
    """

    threshold_tokens: int = 100_000
    target_tokens: int = 50_000

    def __init__(
        self,
        threshold_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
        summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> None:
        if threshold_tokens is not None:
            self.threshold_tokens = threshold_tokens
        if target_tokens is not None:
            self.target_tokens = target_tokens
        self.summarizer = summarizer

        # Running accounting
        self._current_tokens: int = 0
        self._compression_count: int = 0

    # -- public query ---------------------------------------------------------

    def should_compress(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> bool:
        """Return ``True`` when the estimated token count exceeds the threshold."""
        if history is not None:
            self._current_tokens = _history_tokens(history)
        return self._current_tokens >= self.threshold_tokens

    # -- compression logic ----------------------------------------------------

    def compress(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Compress *history* down to approximately ``target_tokens``.

        Strategy
        --------
        1. The first message (typically the system prompt) is always kept.
        2. The most recent messages are kept as-is until the budget is met.
        3. Middle messages are either summarised (if a ``summarizer`` was
           provided) or dropped, replaced by a single assistant note
           indicating that compression occurred.

        Returns
        -------
        A new (shorter) history list.
        """
        if history is None or len(history) == 0:
            return history if history is not None else []

        self._compression_count += 1

        # Always preserve the first message (system prompt) and the tail.
        head = [history[0]] if history else []
        head_tokens = _history_tokens(head)

        remaining_budget = max(0, self.target_tokens - head_tokens)

        # Walk backwards to collect recent messages that fit in the budget.
        tail: List[Dict[str, Any]] = []
        tail_tokens = 0
        for msg in reversed(history[1:]):
            msg_tok = _message_tokens(msg)
            if tail_tokens + msg_tok > remaining_budget:
                break
            tail.insert(0, msg)
            tail_tokens += msg_tok

        # The middle section is everything between head and tail.
        tail_start = len(history) - len(tail) if tail else len(history)
        middle = history[1:tail_start]

        # Build a summary for the middle section.
        if middle:
            if self.summarizer is not None:
                summary_text = self.summarizer(middle)
            else:
                # Default: produce a brief note about what was dropped.
                n_dropped = len(middle)
                dropped_tokens = _history_tokens(middle)
                summary_text = (
                    f"[Compressed: {n_dropped} messages (~{dropped_tokens} tokens) "
                    f"were summarised to fit the context window.]"
                )
            summary_message: Dict[str, Any] = {
                "role": "assistant",
                "content": summary_text,
            }
            compressed = head + [summary_message] + tail
        else:
            compressed = head + tail

        self._current_tokens = _history_tokens(compressed)
        return compressed

    # -- lifecycle hooks ------------------------------------------------------

    def before_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Check whether compression is needed before the next step.

        If compression is triggered, returns a dict with a ``"history"``
        key containing the compressed conversation so the environment can
        swap it in.
        """
        if history is not None and self.should_compress(history):
            compressed = self.compress(history)
            return {"history": compressed}
        return None

    def after_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        step_result: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Update internal token accounting after a step."""
        if history is not None:
            self._current_tokens = _history_tokens(history)
