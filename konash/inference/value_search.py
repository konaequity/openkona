from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ValueGuidedSearchEngine:
    """Value-Guided Search (VGS) for test-time compute (KARL Section 6).

    Implements a parallel breadth-first search over reasoning trajectories,
    where a learned value model scores partial rollouts and the top-*k*
    candidates are expanded at each step.  Multiple independent search trees
    are run in parallel and their results are aggregated.

    Key paper details:
    - The value model uses a *lagged inference policy* (frozen reference model)
      so value scoring is decoupled from the current training policy.
    - Expansion builds proper conversation history from the trajectory so each
      step has full context of prior retrieval and reasoning.
    - The agent generates steps in the retrieval → reasoning → answer flow.

    Attributes:
        candidate_width: Number of candidate continuations to generate at
            each expansion step (called *k* in the paper).
        parallel_searches: Number of independent BFS trees to run.
            ``None`` means determined at call time.
        value_model: A :class:`ValueModel` instance used to score partial
            rollouts.  ``None`` means scores default to 0.
        reference_model: Frozen reference model retained for potential use
            in weighted aggregation (e.g., Best-of-N with reward weighting).
            KL-regularization is handled by the OAPL training loss (Eq. 1),
            not by VGS inference scoring.
    """

    candidate_width = 2
    parallel_searches = None
    value_model = None
    reference_model = None

    def __init__(
        self,
        agent=None,
        value_model=None,
        aggregator=None,
        *,
        reference_model=None,
        candidate_width: int = 2,
        parallel_searches: Optional[int] = None,
        max_depth: int = 10,
    ):
        """
        Args:
            agent: Object with ``generate_step(state, **kw)`` that produces
                one reasoning step.
            value_model: A :class:`ValueModel` (or compatible) for scoring
                partial rollouts.
            aggregator: A :class:`GenerativeAggregator` (or compatible) for
                final answer aggregation.
            reference_model: Frozen reference model for potential use in
                weighted aggregation.  KL-regularization is in the OAPL
                training loss, not VGS scoring.
            candidate_width: *k* -- how many candidates per expansion.
            parallel_searches: *N* -- how many independent search trees.
            max_depth: Maximum number of BFS expansion steps per tree.
        """
        self.agent = agent
        if value_model is not None:
            self.value_model = value_model
        if reference_model is not None:
            self.reference_model = reference_model
        if parallel_searches is not None:
            self.parallel_searches = parallel_searches
        self.candidate_width = candidate_width
        self.aggregator = aggregator
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        *,
        parallel_searches: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full VGS pipeline for a single query.

        1. Run *N* independent BFS trees via :meth:`run_parallel_bfs`.
        2. Aggregate the best trajectories via :meth:`aggregate`.

        Returns a dict with ``answer``, ``search_trees``, ``candidates``,
        and metadata.
        """
        n = parallel_searches or self.parallel_searches
        if n is None:
            n = 1

        search_trees = self.run_parallel_bfs(
            query, num_trees=n, context=context
        )

        # Collect the best trajectory from each tree.
        tree_answers: List[str] = []
        tree_scores: List[float] = []
        for tree in search_trees:
            best = tree.get("best_trajectory", {})
            answer = best.get("answer", "")
            score = best.get("score", 0.0)
            tree_answers.append(answer)
            tree_scores.append(score)

        final_answer = self.aggregate(
            tree_answers, query=query, scores=tree_scores, context=context
        )

        return {
            "answer": final_answer,
            "search_trees": search_trees,
            "candidates": tree_answers,
            "scores": tree_scores,
            "num_trees": n,
        }

    def expand(
        self,
        state: Dict[str, Any],
        *,
        k: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate *k* candidate continuations from *state*.

        Each candidate is a new state dict that extends the trajectory by one
        step.  Builds conversation history from the state's prior steps so
        the agent has full context of retrieval results and reasoning so far.

        If no agent is configured, returns *k* copies of *state* with
        a placeholder step appended.
        """
        k = k or self.candidate_width
        candidates: List[Dict[str, Any]] = []

        # Build conversation history from prior steps for agent context
        history = self._build_conversation_history(state)

        # Merge history into context so the agent sees the full trajectory
        expanded_context = dict(context or {})
        if history:
            expanded_context["conversation_history"] = history

        if self.agent is not None and k > 1:
            # Generate k candidates in parallel (KARL Section 5.2)
            def _gen(i):
                step = self.agent.generate_step(
                    history, candidate_index=i, context=expanded_context,
                )
                return self._extend_state(state, step)

            with ThreadPoolExecutor(max_workers=k) as pool:
                candidates = list(pool.map(_gen, range(k)))
        else:
            for i in range(k):
                if self.agent is not None:
                    step = self.agent.generate_step(
                        history, candidate_index=i, context=expanded_context,
                    )
                    new_state = self._extend_state(state, step)
                else:
                    new_state = copy.deepcopy(state)
                    steps = new_state.setdefault("steps", [])
                    steps.append({"candidate_index": i, "content": ""})
                candidates.append(new_state)
        return candidates

    def score_candidates(
        self, candidates: Sequence[Dict[str, Any]]
    ) -> List[float]:
        """Score each candidate trajectory using the value model.

        Per Section 5.2 of the KARL paper, the value model predicts
        sigma(V(x, y<=t)) — the probability that the rollout will succeed.
        At each step, we select the candidate with the highest value score.

        Note: KL-regularization (log π/π_ref) belongs in the OAPL training
        loss (Eq. 1), not in VGS inference scoring.  The ``reference_model``
        attribute is retained for potential use in weighted aggregation
        (e.g., Best-of-N with σ(V(x,y)) weighting) but does not affect
        per-step candidate selection.

        Returns a list of float scores, one per candidate.  When no value
        model is available, all candidates receive a score of 0.0.
        """
        if self.value_model is None:
            return [0.0] * len(candidates)

        scores: List[float] = []
        for cand in candidates:
            steps = cand.get("steps", [])
            rollout_tokens = cand.get("tokens", steps)
            score = self.value_model.score_partial_rollout(rollout_tokens)
            scores.append(float(score))
        return scores

    def aggregate(
        self,
        candidates: Sequence[str],
        *,
        query: Optional[str] = None,
        scores: Optional[Sequence[float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Aggregate the best answers from *N* search trees.

        If an aggregator is available, delegates to it.  Otherwise picks the
        candidate with the highest score, or falls back to the first
        candidate.
        """
        if not candidates:
            return ""

        if self.aggregator is not None:
            return self.aggregator.aggregate(
                candidates,
                query=query,
                context=context,
                weights=scores,
            )

        # Fallback: pick highest-scored candidate.
        if scores is not None and any(s != 0.0 for s in scores):
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx]

        # No scores -- return first candidate.
        return candidates[0] if candidates else ""

    def run_parallel_bfs(
        self,
        query: str,
        *,
        num_trees: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run *N* independent BFS search trees.

        Each tree performs iterative expand-and-prune:
        1. Start from the initial state (the query).
        2. Expand each surviving state into *k* candidates.
        3. Score all candidates and keep the top-*k*.
        4. Repeat up to ``max_depth`` times or until all candidates are
           terminal.

        Returns a list of tree result dicts, each containing the best
        trajectory found.
        """
        n = num_trees or self.parallel_searches or 1

        if n <= 1:
            return [self._run_single_bfs(query, tree_index=0, context=context)]

        # Run N independent BFS trees in parallel (KARL Section 5.2)
        def _run_tree(tree_idx):
            return self._run_single_bfs(
                query, tree_index=tree_idx, context=context
            )

        with ThreadPoolExecutor(max_workers=n) as pool:
            trees = list(pool.map(_run_tree, range(n)))

        return trees

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_bfs(
        self,
        query: str,
        *,
        tree_index: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one BFS search tree."""
        initial_state: Dict[str, Any] = {
            "query": query,
            "steps": [],
            "tree_index": tree_index,
            "terminal": False,
        }
        beam = [initial_state]

        for depth in range(self.max_depth):
            all_candidates: List[Dict[str, Any]] = []
            for state in beam:
                if state.get("terminal", False):
                    all_candidates.append(state)
                    continue
                expanded = self.expand(state, context=context)
                all_candidates.extend(expanded)

            if not all_candidates:
                break

            scores = self.score_candidates(all_candidates)

            # Attach scores to candidates.
            for cand, score in zip(all_candidates, scores):
                cand["score"] = score

            # Keep top-k by score.
            k = self.candidate_width
            ranked = sorted(
                zip(scores, range(len(all_candidates)), all_candidates),
                key=lambda t: (-t[0], t[1]),
            )
            beam = [cand for _, _, cand in ranked[:k]]

            # Check if all beams are terminal.
            if all(s.get("terminal", False) for s in beam):
                break

        # Select the best trajectory from the final beam.
        if beam:
            best = max(beam, key=lambda s: s.get("score", 0.0))
        else:
            best = initial_state

        answer = self._extract_answer_from_state(best)

        return {
            "tree_index": tree_index,
            "best_trajectory": {
                "state": best,
                "answer": answer,
                "score": best.get("score", 0.0),
            },
            "beam_size": len(beam),
            "depth": min(depth + 1, self.max_depth) if beam else 0,
        }

    @staticmethod
    def _build_conversation_history(
        state: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Build a conversation history from the state's step trajectory.

        Converts each step into a message so the agent has full context of
        prior retrieval results and reasoning when generating the next step.
        This matches the KARL paper's approach where the agent sees its full
        trajectory during search expansion.
        """
        messages: List[Dict[str, str]] = []
        query = state.get("query", "")
        if query:
            messages.append({"role": "user", "content": query})

        for step in state.get("steps", []):
            if isinstance(step, dict):
                step_type = step.get("type", "reasoning")
                content = step.get("content", "")

                if step_type == "tool_call" or step_type == "retrieval":
                    # Agent's tool call / search action
                    messages.append({"role": "assistant", "content": content})
                    # Tool result (search results, retrieval output)
                    tool_result = step.get("result", step.get("tool_output", ""))
                    if tool_result:
                        messages.append({"role": "tool", "content": str(tool_result)})
                elif step_type == "compression":
                    # Compressed context replaces prior messages
                    messages.append({"role": "assistant", "content": content})
                else:
                    # Reasoning / answer step
                    if content:
                        messages.append({"role": "assistant", "content": content})
            elif isinstance(step, str) and step:
                messages.append({"role": "assistant", "content": step})

        return messages

    @staticmethod
    def _extend_state(
        state: Dict[str, Any], step: Any
    ) -> Dict[str, Any]:
        """Create a new state by appending *step* to the trajectory."""
        new_state = copy.deepcopy(state)
        steps = new_state.setdefault("steps", [])
        steps.append(step)
        # Mark terminal if the step indicates completion.
        if isinstance(step, dict) and step.get("terminal", False):
            new_state["terminal"] = True
        return new_state

    @staticmethod
    def _extract_answer_from_state(state: Dict[str, Any]) -> str:
        """Best-effort extraction of the final answer from a search state."""
        if state.get("final_answer"):
            return str(state["final_answer"])
        steps = state.get("steps", [])
        if steps:
            last = steps[-1]
            if isinstance(last, dict):
                return str(last.get("content", last.get("answer", "")))
            return str(last)
        return ""
