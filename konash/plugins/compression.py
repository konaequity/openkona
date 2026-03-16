from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _message_chars(message: Dict[str, Any]) -> int:
    """Count the characters in a single conversation message."""
    total = 0
    content = message.get("content", "")
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                total += len(block.get("text", ""))
            elif isinstance(block, str):
                total += len(block)
    total += len(message.get("role", ""))
    total += len(message.get("name", ""))
    return total


def _history_chars(history: List[Dict[str, Any]]) -> int:
    """Sum characters across all messages in a history."""
    return sum(_message_chars(m) for m in history)


class RLTrainableCompressionPlugin:
    """Compression plugin that delegates summarization to the agent model itself,
    matching the KARL paper's approach (Section 3, Section 9 / Appendix G).

    The compression step is included as part of the RL training rollout.
    The agent learns *what* to compress and *how* to compress it through
    the task reward signal.

    During rollout generation, when the character budget is exceeded:
    1. The full history is serialized into a compression prompt.
    2. The *same agent model* generates a summary.
    3. The summary replaces the middle portion of the history.
    4. A ``<|compression|>`` marker is inserted for rollout segmentation.

    During RL training (via ``konash.training.oapl``), rollouts are split
    at compression boundaries into (x, y) pairs where x is the
    pre-compression context and y is the post-compression continuation.
    The compression quality is trained end-to-end via the task reward.

    Parameters
    ----------
    threshold_chars : int
        Character count threshold to trigger compression.
        KARL BrowseCompPlus uses 150,000 (matching ``compression_trigger_chars``
        in ``SynthesisConfigRegistry``).
    target_chars : int
        Target character count after compression.
    agent_fn : callable or None
        ``(messages: list[dict]) -> dict`` — the agent's LLM function.
        When ``None``, falls back to mechanical truncation.
    preserve_recent_turns : int
        Number of recent turns to always preserve verbatim.
    """

    threshold_chars: int = 150_000
    target_chars: int = 2_000

    def __init__(
        self,
        threshold_chars: Optional[int] = None,
        target_chars: Optional[int] = None,
        agent_fn: Optional[Callable] = None,
        preserve_recent_turns: int = 4,
    ) -> None:
        if threshold_chars is not None:
            self.threshold_chars = threshold_chars
        if target_chars is not None:
            self.target_chars = target_chars
        self.agent_fn = agent_fn
        self.preserve_recent_turns = preserve_recent_turns

        self._current_chars: int = 0
        self._compression_count: int = 0
        self._compression_markers: List[int] = []

    def should_compress(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> bool:
        """Return True when character count exceeds threshold."""
        if history is not None:
            self._current_chars = _history_chars(history)
        return self._current_chars >= self.threshold_chars

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
        head_chars = _history_chars(head)

        remaining_budget = max(0, self.target_chars - head_chars)

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
            # Mechanical fallback when no LLM available
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
        self._current_chars = _history_chars(compressed)
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
            pre_chars = self._current_chars
            compressed = self.compress(history, step_index=step_index)
            post_chars = _history_chars(compressed)

            # Find the compression summary from the inserted message
            summary = ""
            for msg in compressed:
                if isinstance(msg, dict) and msg.get("type") == "compression":
                    content = msg.get("content", "")
                    content = content.replace("<|compression|>", "").replace("<|/compression|>", "").strip()
                    summary = content
                    break

            return {
                "history": compressed,
                "compression_event": {
                    "summary": summary,
                    "pre_chars": pre_chars,
                    "post_chars": post_chars,
                    "messages_dropped": len(history) - len(compressed),
                    "step_index": step_index,
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
        """Update internal character accounting after a step."""
        if history is not None:
            self._current_chars = _history_chars(history)

    @property
    def compression_count(self) -> int:
        return self._compression_count

    @property
    def compression_step_indices(self) -> List[int]:
        """Step indices where compressions occurred (for segmentation)."""
        return list(self._compression_markers)
