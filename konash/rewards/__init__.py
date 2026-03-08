from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Sequence


# Type alias for reward functions:
#   (prediction: str, reference: str, **kwargs) -> float
RewardFn = Callable[..., float]


class RewardRegistry:
    """Central registry for task-specific reward functions.

    Reward functions are callables that score an agent's output against a
    reference answer, returning a float (typically 0.0 -- 1.0).  The
    registry supports per-task registration, retrieval, and composition
    (weighted combination of multiple reward signals).

    Class Attributes
    ----------------
    default_tasks : set of task names that ship with built-in reward
        definitions.
    """

    default_tasks: Set[str] = {"BrowseCompPlus", "TRECBiogen"}

    def __init__(self) -> None:
        self._rewards: Dict[str, RewardFn] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Seed default reward stubs so ``list_rewards`` always surfaces them.
        for task in self.default_tasks:
            if task not in self._rewards:
                self._rewards[task] = self._make_default_reward(task)
                self._metadata[task] = {"source": "default", "task": task}

    # -- registration ---------------------------------------------------------

    def register(
        self,
        name: str,
        fn: RewardFn,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Register a reward function under *name*.

        Parameters
        ----------
        name:
            Unique identifier for this reward (often a task name).
        fn:
            The reward callable.
        metadata:
            Optional dict of extra information (description, weight, etc.).
        overwrite:
            If ``False`` (default) and *name* is already registered, raise
            ``ValueError``.
        """
        if name in self._rewards and not overwrite:
            raise ValueError(
                f"Reward '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._rewards[name] = fn
        self._metadata[name] = metadata if metadata is not None else {}

    # -- retrieval ------------------------------------------------------------

    def get(
        self,
        name: str,
        **kwargs: Any,
    ) -> RewardFn:
        """Return the reward function registered under *name*.

        Raises ``KeyError`` if *name* is not found.
        """
        if name not in self._rewards:
            raise KeyError(f"No reward registered under '{name}'.")
        return self._rewards[name]

    # -- composition ----------------------------------------------------------

    def compose(
        self,
        names: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> RewardFn:
        """Create a composite reward that is a weighted sum of named rewards.

        Parameters
        ----------
        names:
            Reward names to combine.
        weights:
            Per-reward weights.  If ``None``, all rewards are weighted
            equally (uniform average).

        Returns
        -------
        A new reward function that, when called, evaluates every
        constituent reward and returns the weighted sum.
        """
        fns = [self.get(n) for n in names]

        if weights is None:
            w = [1.0 / len(fns)] * len(fns)
        else:
            if len(weights) != len(fns):
                raise ValueError(
                    f"Length mismatch: {len(weights)} weights for "
                    f"{len(fns)} rewards."
                )
            total = sum(weights)
            w = [wi / total for wi in weights] if total != 0 else [0.0] * len(weights)

        def _composite(*args: Any, **kw: Any) -> float:
            return sum(wi * fn(*args, **kw) for wi, fn in zip(w, fns))

        return _composite

    # -- listing --------------------------------------------------------------

    def list_rewards(self, **kwargs: Any) -> List[str]:
        """Return a sorted list of all registered reward names."""
        return sorted(self._rewards.keys())

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _make_default_reward(task: str) -> RewardFn:
        """Create a nugget-based binary reward matching the KARL paper.

        The paper (Section 2.3, 5) uses nugget-based completion scoring
        as the reward signal for RL training:

        1. Decompose the reference answer into nuggets (atomic facts).
        2. Judge each nugget against the candidate answer.
        3. Compute the mean nugget score (nugget completion rate).
        4. Binary reward: 1.0 if completion >= 0.5, else 0.0.

        This is strictly better than exact-match because it handles:
        - Partial matches and paraphrases
        - Entity-level matching (QAMPARI)
        - Report-style multi-nugget evaluation (TREC-Biogen)
        - Single-nugget containment (BrowseComp, FinanceBench)

        When an ``LLMNuggetJudge`` is passed via the ``nugget_judge``
        kwarg, the full paper-faithful LLM evaluation is used.
        Otherwise, the heuristic substring/word-overlap judge provides
        a reasonable approximation.
        """
        from konash.eval.nuggets import NuggetPolicyRegistry, NuggetScorer

        # Get task-specific nugget policy if available
        try:
            policy = NuggetPolicyRegistry.get(task)
        except KeyError:
            policy = None

        def _nugget_reward(
            prediction: str = "",
            reference: str = "",
            **kw: Any,
        ) -> float:
            if not reference or not prediction:
                return 0.0

            # Fast path: exact match always passes
            if prediction.strip().lower() == reference.strip().lower():
                return 1.0

            # Build scorer — use caller-supplied judge if available
            judge = kw.get("nugget_judge", None)
            scorer = NuggetScorer(judge=judge, policy=policy)

            result = scorer.score(prediction, reference)
            score = result.get("score", 0.0)

            # Binary reward: pass if nugget completion >= 0.5
            return 1.0 if score >= 0.5 else 0.0

        _nugget_reward.__qualname__ = f"nugget_reward[{task}]"
        return _nugget_reward
