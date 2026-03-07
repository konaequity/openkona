from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


class Rollout:
    """A single reasoning rollout: a sequence of steps with a final answer."""

    def __init__(
        self,
        steps: Optional[List[Dict[str, Any]]] = None,
        final_answer: Optional[str] = None,
        passed: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.steps = steps or []
        self.final_answer = final_answer
        self.passed = passed
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        status = "pass" if self.passed else ("fail" if self.passed is False else "?")
        return f"Rollout(steps={len(self.steps)}, answer={self.final_answer!r}, {status})"

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class RolloutGroup:
    """A collection of rollouts for a single prompt, with pass-rate tracking."""

    def __init__(
        self,
        prompt: str,
        reference_answer: Optional[str] = None,
        rollouts: Optional[List[Rollout]] = None,
    ):
        self.prompt = prompt
        self.reference_answer = reference_answer
        self.rollouts = rollouts or []

    @property
    def pass_rate(self) -> float:
        """Fraction of rollouts that passed (0.0 if none evaluated)."""
        evaluated = [r for r in self.rollouts if r.passed is not None]
        if not evaluated:
            return 0.0
        return sum(1 for r in evaluated if r.passed) / len(evaluated)

    @property
    def size(self) -> int:
        return len(self.rollouts)

    def __repr__(self) -> str:
        return f"RolloutGroup(n={self.size}, pass_rate={self.pass_rate:.2f})"


class RolloutGenerator:
    """Generates solver rollouts for synthetic QA pairs.

    Each rollout is a multi-step reasoning trace that attempts to answer
    a question using retrieval (top-k search) and chain-of-thought reasoning.

    Attributes
    ----------
    max_steps : int or None
        Maximum number of reasoning / retrieval steps per rollout.
    top_k : int or None
        Number of documents to retrieve at each search step.
    """

    max_steps = None
    top_k = None

    def __init__(
        self,
        max_steps: Optional[int] = None,
        top_k: Optional[int] = None,
        search_tool: Any = None,
        solver_model: Optional[str] = None,
    ):
        self.max_steps = max_steps
        self.top_k = top_k
        self.search_tool = search_tool
        self.solver_model = solver_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_group(
        self,
        prompt: str,
        reference_answer: Optional[str] = None,
        num_rollouts: int = 8,
        *,
        seed: Optional[int] = None,
    ) -> RolloutGroup:
        """Generate multiple independent rollouts for a single prompt.

        Parameters
        ----------
        prompt : str
            The question or problem statement.
        reference_answer : str | None
            Gold answer for pass/fail evaluation.
        num_rollouts : int
            How many rollouts to produce (default 8).
        seed : int | None
            Optional random seed for reproducibility.

        Returns
        -------
        RolloutGroup
        """
        if seed is not None:
            random.seed(seed)

        rollouts: List[Rollout] = []
        for i in range(num_rollouts):
            rollout = self.generate_single(
                prompt,
                reference_answer=reference_answer,
                rollout_id=i,
            )
            rollouts.append(rollout)

        return RolloutGroup(
            prompt=prompt,
            reference_answer=reference_answer,
            rollouts=rollouts,
        )

    def generate_single(
        self,
        prompt: str,
        reference_answer: Optional[str] = None,
        rollout_id: int = 0,
    ) -> Rollout:
        """Generate a single reasoning rollout for a prompt.

        A rollout consists of up to ``max_steps`` reasoning steps, where each
        step may involve retrieval, reflection, or answer formulation.

        Parameters
        ----------
        prompt : str
            The question or problem statement.
        reference_answer : str | None
            Gold answer for automatic pass/fail scoring.
        rollout_id : int
            Index of this rollout within a group (for logging / seeding).

        Returns
        -------
        Rollout
        """
        steps: List[Dict[str, Any]] = []
        effective_max_steps = self.max_steps or 50
        effective_top_k = self.top_k or 20

        final_answer: Optional[str] = None

        for step_idx in range(effective_max_steps):
            step_record: Dict[str, Any] = {"step": step_idx, "type": None}

            if step_idx == 0:
                # Step 0: initial retrieval
                retrieved = self._retrieve(prompt, effective_top_k)
                step_record["type"] = "retrieval"
                step_record["query"] = prompt
                step_record["num_results"] = len(retrieved)
                step_record["results"] = retrieved
                steps.append(step_record)
                continue

            # Intermediate reasoning steps
            if step_idx < effective_max_steps - 1:
                reasoning = self._reason(prompt, steps)
                step_record["type"] = "reasoning"
                step_record["thought"] = reasoning.get("thought", "")

                if reasoning.get("needs_retrieval"):
                    sub_query = reasoning.get("sub_query", prompt)
                    retrieved = self._retrieve(sub_query, effective_top_k)
                    step_record["sub_retrieval"] = {
                        "query": sub_query,
                        "num_results": len(retrieved),
                    }

                if reasoning.get("has_answer"):
                    final_answer = reasoning["answer"]
                    step_record["type"] = "answer"
                    step_record["answer"] = final_answer
                    steps.append(step_record)
                    break

                steps.append(step_record)
            else:
                # Last step: force an answer
                final_answer = self._force_answer(prompt, steps)
                step_record["type"] = "answer"
                step_record["answer"] = final_answer
                steps.append(step_record)

        # Evaluate pass/fail
        passed = None
        if reference_answer is not None and final_answer is not None:
            passed = self._evaluate(final_answer, reference_answer)

        return Rollout(
            steps=steps,
            final_answer=final_answer,
            passed=passed,
            metadata={"rollout_id": rollout_id, "max_steps": effective_max_steps, "top_k": effective_top_k},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, top_k: int) -> List[str]:
        """Retrieve documents using the configured search tool."""
        if self.search_tool is None:
            return []
        try:
            results = self.search_tool.search(query, top_k=top_k)
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def _reason(self, prompt: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform a single reasoning step.

        In a full deployment this calls an LLM. The base implementation
        provides a deterministic heuristic: after gathering retrieval
        results, it formulates an answer based on available evidence.
        """
        # Check if we have retrieval results to work with
        retrieval_results: List[str] = []
        for s in steps:
            if s.get("type") == "retrieval" and s.get("results"):
                retrieval_results.extend(s["results"])

        if retrieval_results:
            # We have evidence -- formulate an answer
            combined = " ".join(retrieval_results[:3])
            return {
                "thought": f"Based on retrieved evidence, I can answer the question about: {prompt[:100]}",
                "has_answer": True,
                "answer": combined[:500] if combined else "No answer found",
                "needs_retrieval": False,
            }

        # No evidence yet -- request retrieval
        return {
            "thought": f"I need more information to answer: {prompt[:100]}",
            "has_answer": False,
            "needs_retrieval": True,
            "sub_query": prompt,
        }

    def _force_answer(self, prompt: str, steps: List[Dict[str, Any]]) -> str:
        """Force an answer when max_steps is reached without a natural conclusion."""
        # Gather any evidence collected so far
        evidence_pieces: List[str] = []
        for s in steps:
            if s.get("results"):
                evidence_pieces.extend(s["results"])
            if s.get("thought"):
                evidence_pieces.append(s["thought"])

        if evidence_pieces:
            return " ".join(evidence_pieces[:3])[:500]
        return f"Unable to determine answer for: {prompt[:200]}"

    @staticmethod
    def _evaluate(predicted: str, reference: str) -> bool:
        """Simple exact-match evaluation (case-insensitive, whitespace-normalized)."""
        def normalize(text: str) -> str:
            return " ".join(text.lower().split())

        pred_norm = normalize(predicted)
        ref_norm = normalize(reference)

        # Exact match
        if pred_norm == ref_norm:
            return True

        # Containment check (reference answer appears in prediction)
        if ref_norm in pred_norm:
            return True

        # Token overlap (F1-style soft match with threshold)
        pred_tokens = set(pred_norm.split())
        ref_tokens = set(ref_norm.split())
        if not ref_tokens:
            return False

        overlap = pred_tokens & ref_tokens
        if not overlap:
            return False

        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(ref_tokens)
        if precision + recall == 0:
            return False
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5
