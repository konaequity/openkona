from __future__ import annotations

from typing import Any, Dict, List, Optional


class EvaluationReport:
    """Container for evaluation results across benchmarks.

    Attributes:
        quality: Overall quality score (average nugget completion rate
            across all evaluated queries), float in [0, 1].
        cost_per_query: Average inference cost measured in rollout-equivalents
            (1 for single rollout, N for parallel thinking with N rollouts).
        latency_seconds: Average wall-clock latency per query in seconds.
        in_distribution: Average quality score on in-distribution
            (training) benchmarks.  ``None`` if no in-distribution results
            are available.
        out_of_distribution: Average quality score on out-of-distribution
            (held-out) benchmarks.  ``None`` if no OOD results are available.
    """

    quality = None
    cost_per_query = None
    latency_seconds = None
    in_distribution = None
    out_of_distribution = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def compute(self) -> Dict[str, Any]:
        """Return a summary dict of all evaluation metrics.

        This is a convenience method that packages the report's attributes
        into a plain dictionary suitable for serialisation or logging.
        """
        summary: Dict[str, Any] = {
            "quality": self.quality,
            "cost_per_query": self.cost_per_query,
            "latency_seconds": self.latency_seconds,
            "in_distribution": self.in_distribution,
            "out_of_distribution": self.out_of_distribution,
        }

        # Compute generalisation gap when both splits are available.
        if (
            isinstance(self.in_distribution, (int, float))
            and isinstance(self.out_of_distribution, (int, float))
        ):
            summary["generalisation_gap"] = (
                self.in_distribution - self.out_of_distribution
            )
        else:
            summary["generalisation_gap"] = None

        return summary

    def summary(self) -> Dict[str, Any]:
        """Alias for :meth:`compute` for discoverability."""
        return self.compute()

    def __repr__(self) -> str:
        parts = []
        for attr in (
            "quality",
            "cost_per_query",
            "latency_seconds",
            "in_distribution",
            "out_of_distribution",
        ):
            val = getattr(self, attr, None)
            if val is not None:
                if isinstance(val, float):
                    parts.append(f"{attr}={val:.4f}")
                else:
                    parts.append(f"{attr}={val!r}")
        return f"EvaluationReport({', '.join(parts)})"
