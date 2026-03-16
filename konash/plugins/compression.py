from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        self._last_summary: str = ""

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
            self._last_summary = summary_text
            compressed = head + [summary_message] + tail
        else:
            self._last_summary = ""
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
            pre_compression_history = list(history)
            compressed = self.compress(history)
            return {
                "history": compressed,
                "compression_event": {
                    "type": "compression",
                    "summary": self._last_summary,
                    "pre_tokens": _history_tokens(pre_compression_history),
                    "post_tokens": _history_tokens(compressed),
                    "messages_dropped": len(pre_compression_history) - len(compressed),
                },
            }
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

    @property
    def compression_count(self) -> int:
        """Number of compressions performed so far."""
        return self._compression_count


class RLTrainableCompressionPlugin:
    """Compression plugin that delegates summarization to the agent model itself,
    matching the KARL paper's approach (Section 3, Section 9 / Appendix G).

    Key difference from ``CompressionPlugin``: the compression step is
    included as part of the RL training rollout.  The agent learns *what*
    to compress and *how* to compress it through the task reward signal,
    rather than using a heuristic or separate summarization model.

    During rollout generation, when the token budget is exceeded:
    1. The full history is serialized into a compression prompt.
    2. The *same agent model* generates a summary.
    3. The summary replaces the middle portion of the history.
    4. A ``<|compression|>`` marker is inserted for rollout segmentation.

    During RL training (via ``konash.training.segmentation``), rollouts are
    split at compression boundaries into (x, y) pairs where x is the
    pre-compression context and y is the post-compression continuation.
    The compression quality is trained end-to-end via the task reward.

    Parameters
    ----------
    threshold_tokens : int
        Token count threshold to trigger compression.
    target_tokens : int
        Target token count after compression.
    agent_fn : callable or None
        ``(messages: list[dict]) -> dict`` — the agent's LLM function.
        When ``None``, falls back to mechanical truncation.
    preserve_recent_turns : int
        Number of recent turns to always preserve verbatim.
    """

    threshold_tokens: int = 100_000
    target_tokens: int = 50_000

    def __init__(
        self,
        threshold_tokens: Optional[int] = None,
        target_tokens: Optional[int] = None,
        agent_fn: Optional[Callable] = None,
        preserve_recent_turns: int = 4,
    ) -> None:
        if threshold_tokens is not None:
            self.threshold_tokens = threshold_tokens
        if target_tokens is not None:
            self.target_tokens = target_tokens
        self.agent_fn = agent_fn
        self.preserve_recent_turns = preserve_recent_turns

        self._current_tokens: int = 0
        self._compression_count: int = 0
        self._compression_markers: List[int] = []  # step indices where compression occurred

    def should_compress(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> bool:
        """Return True when estimated token count exceeds threshold."""
        if history is not None:
            self._current_tokens = _history_tokens(history)
        return self._current_tokens >= self.threshold_tokens

    def compress(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        step_index: int = 0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Compress history using the agent model itself.

        Strategy (matching KARL paper Section 3):
        1. Preserve system prompt (first message).
        2. Preserve the most recent ``preserve_recent_turns`` messages.
        3. Serialize the middle section into a compression prompt.
        4. Ask the agent to summarize, preserving key facts and evidence.
        5. Insert the summary with a ``<|compression|>`` marker.

        The marker enables downstream rollout segmentation for RL training.
        """
        if history is None or len(history) == 0:
            return history if history is not None else []

        self._compression_count += 1
        self._compression_markers.append(step_index)

        # Split into head / middle / tail
        head = [history[0]] if history else []
        head_tokens = _history_tokens(head)

        remaining_budget = max(0, self.target_tokens - head_tokens)

        # Preserve recent turns
        tail_count = min(self.preserve_recent_turns, len(history) - 1)
        tail = history[-tail_count:] if tail_count > 0 else []
        middle = history[1 : len(history) - tail_count] if tail_count > 0 else history[1:]

        if not middle:
            return history  # nothing to compress

        # Build compression prompt for the agent
        if self.agent_fn is not None:
            summary_text = self._agent_compress(middle)
        else:
            # Mechanical fallback: keep first + last messages of middle
            if len(middle) <= 2:
                summary_text = "\n".join(
                    m.get("content", "") for m in middle if isinstance(m, dict)
                )
            else:
                first_content = middle[0].get("content", "") if isinstance(middle[0], dict) else ""
                last_content = middle[-1].get("content", "") if isinstance(middle[-1], dict) else ""
                dropped = len(middle) - 2
                summary_text = (
                    f"{first_content}\n\n"
                    f"[{dropped} intermediate messages compressed]\n\n"
                    f"{last_content}"
                )

        # Insert compression marker for rollout segmentation
        summary_message: Dict[str, Any] = {
            "role": "assistant",
            "content": f"<|compression|>\n{summary_text}\n<|/compression|>",
            "type": "compression",
            "compression_index": self._compression_count,
        }

        compressed = head + [summary_message] + tail
        self._current_tokens = _history_tokens(compressed)
        return compressed

    def _agent_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Use the agent model to generate a compression summary.

        The prompt instructs the model to:
        - Preserve all key facts, entities, and evidence
        - Maintain search queries and their results
        - Discard verbose reasoning and redundant retrieval results
        - Keep the summary concise but information-dense
        """
        # Serialize the messages
        serialized = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                serialized.append(f"[{role}]: {content}")
            else:
                serialized.append(str(msg))
        full_text = "\n\n".join(serialized)

        # Truncate if extremely long
        if len(full_text) > 50000:
            full_text = full_text[:25000] + "\n...[truncated]...\n" + full_text[-25000:]

        compression_prompt = [
            {
                "role": "system",
                "content": (
                    "You are compressing a search agent's conversation history. "
                    "Your goal is to preserve ALL key information needed to continue "
                    "the search task while drastically reducing length.\n\n"
                    "PRESERVE:\n"
                    "- All specific facts, numbers, entities, and evidence found\n"
                    "- Search queries that were tried and their key results\n"
                    "- The current reasoning state and what remains to be found\n"
                    "- Any partial answers or hypotheses\n\n"
                    "DISCARD:\n"
                    "- Verbose document text (keep only relevant excerpts)\n"
                    "- Redundant retrieval results\n"
                    "- Detailed reasoning that led to dead ends\n\n"
                    "Output a concise summary that another agent could use to "
                    "continue the task effectively."
                ),
            },
            {
                "role": "user",
                "content": f"Compress this conversation history:\n\n{full_text}",
            },
        ]

        try:
            response = self.agent_fn(compression_prompt)
            if isinstance(response, dict):
                return response.get("content", "")
            return str(response)
        except Exception as exc:
            logger.warning("Agent compression failed: %s; using mechanical fallback", exc)
            # Mechanical fallback
            parts = []
            for msg in messages[:2] + messages[-2:]:
                if isinstance(msg, dict):
                    parts.append(msg.get("content", "")[:500])
            return "\n\n".join(parts)

    # -- lifecycle hooks --

    def before_step(
        self,
        step_index: int = 0,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Check whether compression is needed before the next step."""
        if history is not None and self.should_compress(history):
            compressed = self.compress(history, step_index=step_index)
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

    @property
    def compression_count(self) -> int:
        return self._compression_count

    @property
    def compression_step_indices(self) -> List[int]:
        """Step indices where compressions occurred (for segmentation)."""
        return list(self._compression_markers)
