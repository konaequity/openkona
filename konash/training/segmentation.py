from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RolloutSegmenter:
    """Segment rollouts at compression boundaries and manage token masking.

    In the KARL pipeline, rollouts may contain compression events where the
    context window is summarised.  This class splits rollouts into (x, y)
    input-output pairs at those boundaries and produces masks to exclude
    tool-output tokens from the training loss.

    Attributes
    ----------
    include_compression_segments : bool
        If True (default), segments that straddle a compression boundary are
        included in training data.  If False, only the final (post-compression)
        segment is kept.
    """

    include_compression_segments = True

    def __init__(self, include_compression_segments: bool = True):
        self.include_compression_segments = include_compression_segments

    def split_on_compression(
        self,
        rollout: List[Dict[str, Any]],
        compression_marker: str = "<|compression|>",
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Split a rollout at compression boundaries into (x, y) pairs.

        Each compression marker divides the rollout into a prefix (context up
        to and including the marker, treated as input *x*) and a suffix
        (everything after, treated as continuation *y*).  When
        ``include_compression_segments`` is True every boundary produces a
        pair; otherwise only the last segment is returned.

        Parameters
        ----------
        rollout:
            A list of message dicts (or token dicts) representing the full
            rollout.  Compression boundaries are identified by entries whose
            ``"content"`` field (if present) contains *compression_marker*, or
            whose ``"type"`` field equals ``"compression"``.
        compression_marker:
            The string marker identifying a compression event in the content.

        Returns
        -------
        list of (x, y) tuples
            Each tuple contains the input context *x* and the continuation *y*.
            If no compression boundaries are found the entire rollout is
            returned as a single pair with an empty *x*.
        """
        # Find indices of compression boundaries
        boundary_indices: List[int] = []
        for i, entry in enumerate(rollout):
            is_compression = False
            if isinstance(entry, dict):
                content = entry.get("content", "")
                if isinstance(content, str) and compression_marker in content:
                    is_compression = True
                if entry.get("type") == "compression":
                    is_compression = True
            if is_compression:
                boundary_indices.append(i)

        if not boundary_indices:
            # No compression boundaries: entire rollout is a single segment
            return [([], rollout)]

        pairs: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for boundary_idx in boundary_indices:
            x = rollout[: boundary_idx + 1]
            y = rollout[boundary_idx + 1 :]
            pairs.append((x, y))

        if not self.include_compression_segments:
            # Only keep the last segment (after the final compression)
            return [pairs[-1]]

        return pairs

    def mask_tool_outputs(
        self,
        tokens: List[Any],
        tool_start_marker: str = "<|tool_start|>",
        tool_end_marker: str = "<|tool_end|>",
    ) -> np.ndarray:
        """Create a boolean mask marking non-model (tool-output) tokens as False.

        Tokens between *tool_start_marker* and *tool_end_marker* (inclusive of
        the markers themselves) are considered tool outputs and are masked
        out so they do not contribute to the training loss.

        Parameters
        ----------
        tokens:
            Sequence of tokens (strings or dicts with a ``"content"`` key).
        tool_start_marker:
            String indicating the start of a tool-output span.
        tool_end_marker:
            String indicating the end of a tool-output span.

        Returns
        -------
        np.ndarray
            Boolean array of length ``len(tokens)``.  True = model-generated
            (include in loss), False = tool output (mask out).
        """
        mask = np.ones(len(tokens), dtype=bool)
        in_tool_output = False

        for i, token in enumerate(tokens):
            token_str = token
            if isinstance(token, dict):
                token_str = token.get("content", "")
            if not isinstance(token_str, str):
                token_str = str(token_str)

            if tool_start_marker in token_str:
                in_tool_output = True

            if in_tool_output:
                mask[i] = False

            if tool_end_marker in token_str:
                in_tool_output = False

        return mask

    def assign_rollout_reward(
        self,
        segments: List[Tuple[List[Any], List[Any]]],
        rollout_reward: float,
    ) -> List[Dict[str, Any]]:
        """Assign the full rollout reward to each segment.

        In the OAPL framework the reward is a property of the entire rollout,
        not of individual segments.  Each segment therefore receives the same
        reward value.

        Parameters
        ----------
        segments:
            List of (x, y) pairs as returned by ``split_on_compression``.
        rollout_reward:
            The scalar reward for the full rollout.

        Returns
        -------
        list of dict
            Each dict has keys ``"x"`` (input context), ``"y"`` (continuation),
            and ``"reward"`` (the rollout-level reward).
        """
        return [
            {"x": x, "y": y, "reward": rollout_reward}
            for x, y in segments
        ]
