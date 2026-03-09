from __future__ import annotations

import json as _json
import random
import re as _re
from typing import Any, Callable, Dict, List, Optional


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

    Each rollout is a multi-step agentic search trace where the solver:
    1. Retrieves documents via vector search (the sole tool)
    2. Reasons about the evidence (chain-of-thought)
    3. Optionally issues follow-up searches with refined queries
    4. Formulates a final answer

    This matches the KARL paper's solver agent (Section 4.1, Stage II):
    - Multi-step reasoning with retrieval as the only tool
    - Context compression for long trajectories
    - Binary reward based on answer correctness

    Parameters
    ----------
    max_steps : int
        Maximum reasoning/retrieval steps per rollout (paper: 50 for TREC,
        200 for BrowseComp-Plus).
    top_k : int
        Documents to retrieve per search (paper: 20).
    search_tool : object
        Must expose ``.search(query, top_k=...) -> list``.
    llm_fn : callable
        ``(messages) -> {"role": "assistant", "content": "..."}``.
    compression_trigger_chars : int | None
        Compress context when total chars exceed this (paper: 150K for
        BrowseComp-Plus). None = no compression.
    on_step : callable | None
        Progress callback ``(qa_idx, rollout_idx, step_idx, step_record)``.
    """

    max_steps = None
    top_k = None

    def __init__(
        self,
        max_steps: Optional[int] = None,
        top_k: Optional[int] = None,
        search_tool: Any = None,
        solver_model: Optional[str] = None,
        llm_fn: Any = None,
        compression_trigger_chars: Optional[int] = None,
        on_step: Optional[Callable] = None,
        nugget_scorer: Any = None,
    ):
        self.max_steps = max_steps or 50
        self.top_k = top_k or 20
        self.search_tool = search_tool
        self.solver_model = solver_model
        self.llm_fn = llm_fn
        self.compression_trigger_chars = compression_trigger_chars
        self.on_step = on_step
        self.nugget_scorer = nugget_scorer

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
        qa_idx: int = 0,
    ) -> RolloutGroup:
        """Generate multiple independent rollouts for a single prompt.

        Parameters
        ----------
        prompt : str
            The question or problem statement.
        reference_answer : str | None
            Gold answer for pass/fail evaluation.
        num_rollouts : int
            How many rollouts to produce (paper default: 8).
        seed : int | None
            Optional random seed for reproducibility.
        qa_idx : int
            Index of this QA pair (for progress callbacks).

        Returns
        -------
        RolloutGroup
        """
        if seed is not None:
            random.seed(seed)

        # Decompose reference answer into nuggets once, reuse for all rollouts
        # (KARL paper Section 2.3, Appendix D.1 Figure 31)
        nuggets = None
        if reference_answer and self.llm_fn is not None:
            nuggets = self._decompose_nuggets(prompt, reference_answer)

        rollouts: List[Rollout] = []
        for i in range(num_rollouts):
            rollout = self.generate_single(
                prompt,
                reference_answer=reference_answer,
                rollout_id=i,
                qa_idx=qa_idx,
                nuggets=nuggets,
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
        qa_idx: int = 0,
        nuggets: Optional[List[str]] = None,
    ) -> Rollout:
        """Generate a single reasoning rollout for a prompt.

        Trajectory structure (matching KARL paper Section 4.1):
        - Step 0: Initial retrieval (vector search with the question)
        - Steps 1..N-1: Reasoning — the LLM examines evidence and either:
            (a) issues a follow-up search with a refined sub-query, or
            (b) formulates a final answer
        - Step N (if no earlier answer): Forced answer from all evidence

        Returns
        -------
        Rollout
        """
        steps: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None

        # Track total context size for compression
        total_chars = 0

        for step_idx in range(self.max_steps):
            step_record: Dict[str, Any] = {"step": step_idx, "type": None}

            if step_idx == 0:
                # Step 0: initial retrieval
                retrieved = self._retrieve(prompt, self.top_k)
                step_record["type"] = "retrieval"
                step_record["query"] = prompt
                step_record["num_results"] = len(retrieved)
                step_record["results"] = retrieved
                total_chars += sum(
                    len(r.get("text", str(r))) if isinstance(r, dict) else len(str(r))
                    for r in retrieved
                )
                steps.append(step_record)
                self._notify(qa_idx, rollout_id, step_idx, step_record)
                continue

            # Check if we need context compression (KARL paper Section 4.2)
            if (self.compression_trigger_chars
                    and total_chars > self.compression_trigger_chars):
                steps = self._compress_context(steps, prompt)
                total_chars = sum(
                    len(str(s.get("thought", ""))) + len(str(s.get("results", "")))
                    + len(str(s.get("summary", "")))
                    for s in steps
                )

            # Intermediate reasoning steps
            if step_idx < self.max_steps - 1:
                reasoning = self._reason(prompt, steps, step_idx)
                step_record["type"] = "reasoning"
                step_record["thought"] = reasoning.get("thought", "")
                total_chars += len(step_record["thought"])

                if reasoning.get("needs_retrieval"):
                    sub_query = reasoning.get("sub_query", prompt)
                    retrieved = self._retrieve(sub_query, self.top_k)
                    step_record["sub_retrieval"] = {
                        "query": sub_query,
                        "num_results": len(retrieved),
                        "results": retrieved,
                    }
                    total_chars += sum(
                        len(r.get("text", str(r))) if isinstance(r, dict) else len(str(r))
                        for r in retrieved
                    )

                if reasoning.get("has_answer"):
                    final_answer = reasoning["answer"]
                    step_record["type"] = "answer"
                    step_record["answer"] = final_answer
                    steps.append(step_record)
                    self._notify(qa_idx, rollout_id, step_idx, step_record)
                    break

                steps.append(step_record)
                self._notify(qa_idx, rollout_id, step_idx, step_record)
            else:
                # Last step: force an answer
                final_answer = self._force_answer(prompt, steps)
                step_record["type"] = "answer"
                step_record["answer"] = final_answer
                steps.append(step_record)
                self._notify(qa_idx, rollout_id, step_idx, step_record)

        # Evaluate pass/fail
        passed = None
        if reference_answer is not None and final_answer is not None:
            passed = self._evaluate(
                final_answer, reference_answer,
                question=prompt, nuggets=nuggets,
            )

        return Rollout(
            steps=steps,
            final_answer=final_answer,
            passed=passed,
            metadata={
                "rollout_id": rollout_id,
                "max_steps": self.max_steps,
                "top_k": self.top_k,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notify(
        self, qa_idx: int, rollout_idx: int, step_idx: int,
        step_record: Dict[str, Any],
    ) -> None:
        """Fire the on_step callback if configured."""
        if self.on_step is not None:
            try:
                self.on_step(qa_idx, rollout_idx, step_idx, step_record)
            except Exception:
                pass

    def _retrieve(self, query: str, top_k: int) -> List[Any]:
        """Retrieve documents using the configured search tool."""
        if self.search_tool is None:
            return []
        try:
            results = self.search_tool.search(query, top_k=top_k)
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def _collect_evidence(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Gather all retrieved text from the trajectory so far."""
        evidence: List[str] = []
        for s in steps:
            if s.get("type") == "retrieval" and s.get("results"):
                for r in s["results"]:
                    if isinstance(r, dict):
                        evidence.append(r.get("text", str(r)))
                    else:
                        evidence.append(str(r))
            # Also gather sub-retrieval results
            sub = s.get("sub_retrieval")
            if sub and sub.get("results"):
                for r in sub["results"]:
                    if isinstance(r, dict):
                        evidence.append(r.get("text", str(r)))
                    else:
                        evidence.append(str(r))
        return evidence

    def _reason(
        self, prompt: str, steps: List[Dict[str, Any]], step_idx: int,
    ) -> Dict[str, Any]:
        """Perform a single reasoning step.

        When ``llm_fn`` is configured, calls the LLM to decide whether to
        search again or formulate an answer.  Otherwise uses a deterministic
        heuristic.
        """
        evidence = self._collect_evidence(steps)

        if self.llm_fn is not None:
            return self._reason_with_llm(prompt, steps, evidence, step_idx)

        # Heuristic fallback
        if evidence:
            combined = " ".join(evidence[:3])
            return {
                "thought": f"Based on retrieved evidence, I can answer: {prompt[:100]}",
                "has_answer": True,
                "answer": combined[:500] if combined else "No answer found",
                "needs_retrieval": False,
            }
        return {
            "thought": f"I need more information to answer: {prompt[:100]}",
            "has_answer": False,
            "needs_retrieval": True,
            "sub_query": prompt,
        }

    def _reason_with_llm(
        self,
        prompt: str,
        steps: List[Dict[str, Any]],
        evidence: List[str],
        step_idx: int,
    ) -> Dict[str, Any]:
        """Use the LLM to decide the next reasoning action.

        The system prompt instructs the agent to act as a knowledge-retrieval
        solver that can search or answer.  It includes the step budget so the
        LLM knows when to commit to an answer.
        """
        evidence_text = "\n".join(f"- {e[:1000]}" for e in evidence[:10])
        steps_remaining = self.max_steps - step_idx - 1

        # Detect cycling: if we've done 2+ reasoning steps with evidence,
        # the LLM should answer instead of searching again
        reasoning_count = sum(1 for s in steps if s.get("type") == "reasoning")
        should_answer = (
            steps_remaining <= 1
            or (evidence and reasoning_count >= 2)
        )

        if should_answer:
            # Force answer mode — skip LLM reasoning and go straight to
            # answer generation to avoid cycling
            answer = self._force_answer(prompt, steps)
            return {
                "thought": f"Sufficient evidence gathered after {reasoning_count} reasoning steps.",
                "has_answer": True,
                "answer": answer,
                "needs_retrieval": False,
            }

        messages = [
            {"role": "system", "content": (
                "You are a knowledge agent solving a question using retrieval. "
                "You have ONE tool: vector search over a document corpus.\n\n"
                "Output JSON with ONE of these two forms:\n\n"
                "If you can answer from the evidence:\n"
                '{"thought": "...", "has_answer": true, "answer": "your concise answer"}\n\n'
                "If you need MORE information (only if evidence is clearly missing):\n"
                '{"thought": "...", "needs_retrieval": true, "sub_query": "specific new query"}\n\n'
                "IMPORTANT: If the evidence contains the answer, you MUST answer. "
                "Do NOT search for information you already have."
            )},
            {"role": "user", "content": (
                f"Question: {prompt}\n\n"
                f"Step: {step_idx}/{self.max_steps} "
                f"({steps_remaining} steps remaining)\n\n"
                f"Evidence gathered ({len(evidence)} passages):\n"
                f"{evidence_text or '(none yet)'}\n\n"
                "Respond with JSON only."
            )},
        ]
        response = self.llm_fn(messages)
        content = response.get("content", "") if isinstance(response, dict) else str(response)

        # Parse JSON response
        match = _re.search(r"\{.*\}", content, _re.DOTALL)
        if match:
            try:
                parsed = _json.loads(match.group())
                return {
                    "thought": parsed.get("thought", content[:200]),
                    "has_answer": parsed.get("has_answer", False),
                    "answer": parsed.get("answer", ""),
                    "needs_retrieval": parsed.get("needs_retrieval", False),
                    "sub_query": parsed.get("sub_query", prompt),
                }
            except _json.JSONDecodeError:
                pass

        # If we can't parse JSON, force a clean natural-language answer
        # instead of leaking raw LLM content (which may be malformed JSON).
        if evidence:
            answer = self._force_answer(prompt, steps)
            return {
                "thought": content[:200],
                "has_answer": True,
                "answer": answer,
                "needs_retrieval": False,
            }
        return {
            "thought": content[:200],
            "has_answer": False,
            "needs_retrieval": True,
            "sub_query": prompt,
        }

    def _force_answer(self, prompt: str, steps: List[Dict[str, Any]]) -> str:
        """Force an answer when max_steps is reached.

        Uses a larger token budget (512) than reasoning steps because
        thinking models (e.g. Qwen3) spend most tokens inside
        ``<think>...</think>`` tags which get stripped, leaving little
        room for the actual answer at the default 256 budget.
        """
        evidence = self._collect_evidence(steps)
        thoughts = [s["thought"] for s in steps if s.get("thought")]

        if self.llm_fn is not None and (evidence or thoughts):
            evidence_text = "\n".join(f"- {e[:1000]}" for e in evidence[:10])
            thoughts_text = "\n".join(f"- {t[:200]}" for t in thoughts[:5])
            messages = [
                {"role": "system", "content": (
                    "Answer the question concisely based on the evidence. "
                    "If the evidence is insufficient, give your best answer. "
                    "Do NOT output JSON — answer in plain text only."
                )},
                {"role": "user", "content": (
                    f"Question: {prompt}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    f"Reasoning so far:\n{thoughts_text}\n\n"
                    "Final answer:"
                )},
            ]
            # Use a larger token budget so thinking models have room
            # for <think> tags AND a complete answer.
            response = self.llm_fn(messages, max_new_tokens=512)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            if content.strip():
                return content.strip()

        if evidence:
            return " ".join(evidence[:3])[:500]
        return f"Unable to determine answer for: {prompt[:200]}"

    def _compress_context(
        self,
        steps: List[Dict[str, Any]],
        prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """Compress the trajectory context using LLM-generated summarization.

        Matches the KARL paper's compression mechanism (Section 4.2):

        - **The agent compresses its own history** — the same LLM that runs
          the search trajectory produces the summary.
        - Achieves ~100x reduction (e.g. 112K chars -> ~1,100 chars) while
          preserving conclusions, key evidence, and reasoning state.
        - Compression quality improves through RL training because bad
          compressions lead to wrong answers and low reward.

        When no ``llm_fn`` is available, falls back to a mechanical
        keep-first-and-last truncation strategy.

        Parameters
        ----------
        steps : list[dict]
            The trajectory steps accumulated so far.
        prompt : str
            The question being answered — included so the LLM knows what
            information is most important to retain.

        Returns
        -------
        list[dict]
            Compressed trajectory with a ``compression`` step marker.
        """
        if self.llm_fn is not None:
            # LLM compression can handle any number of steps
            if len(steps) <= 1:
                return steps
            return self._llm_compress(steps, prompt)

        # Mechanical fallback needs at least 4 steps (first + middle + last 2)
        if len(steps) <= 3:
            return steps
        return self._mechanical_compress(steps)

    def _llm_compress(
        self,
        steps: List[Dict[str, Any]],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Use the LLM to compress the trajectory (paper-faithful).

        The full trajectory is serialized and sent to the LLM with a
        compression prompt.  The LLM returns a concise summary preserving
        key findings, conclusions, and evidence.
        """
        # Serialize the trajectory into a readable format
        trajectory_parts: List[str] = []
        for s in steps:
            stype = s.get("type", "unknown")
            if stype == "retrieval":
                results = s.get("results", [])
                docs_preview = []
                for r in results[:5]:
                    text = r.get("text", str(r)) if isinstance(r, dict) else str(r)
                    docs_preview.append(text[:300])
                trajectory_parts.append(
                    f"[Step {s.get('step', '?')}] RETRIEVAL for: "
                    f"{s.get('query', '?')}\n"
                    f"  Retrieved {s.get('num_results', 0)} documents. "
                    f"Top results:\n" +
                    "\n".join(f"  - {d}" for d in docs_preview)
                )
            elif stype == "reasoning":
                thought = s.get("thought", "")
                part = f"[Step {s.get('step', '?')}] REASONING: {thought}"
                sub = s.get("sub_retrieval")
                if sub:
                    sub_results = sub.get("results", [])
                    sub_preview = []
                    for r in sub_results[:3]:
                        text = r.get("text", str(r)) if isinstance(r, dict) else str(r)
                        sub_preview.append(text[:200])
                    part += (
                        f"\n  Sub-retrieval for: {sub.get('query', '?')}\n" +
                        "\n".join(f"  - {d}" for d in sub_preview)
                    )
                trajectory_parts.append(part)
            elif stype == "answer":
                trajectory_parts.append(
                    f"[Step {s.get('step', '?')}] ANSWER: {s.get('answer', '')}"
                )
            elif stype == "compression":
                trajectory_parts.append(
                    f"[Step {s.get('step', '?')}] PREVIOUS COMPRESSION: "
                    f"{s.get('summary', '')}"
                )

        trajectory_text = "\n\n".join(trajectory_parts)

        # Truncate if the serialized trajectory itself is enormous
        if len(trajectory_text) > 200_000:
            trajectory_text = trajectory_text[:200_000] + "\n\n[... truncated ...]"

        messages = [
            {"role": "system", "content": (
                "You are compressing your own search trajectory into a concise "
                "summary. Your goal is to preserve ALL information needed to "
                "answer the question correctly.\n\n"
                "What to preserve:\n"
                "- Any conclusions or candidate answers you have reached\n"
                "- Key evidence and facts discovered from retrieved documents\n"
                "- Important entity names, dates, numbers, and relationships\n"
                "- Which search queries have already been tried\n\n"
                "What to drop:\n"
                "- Redundant or irrelevant retrieved passages\n"
                "- Verbose reasoning that can be stated more concisely\n"
                "- Document text that doesn't relate to the question\n\n"
                "Return ONLY the compressed summary. Be concise but complete."
            )},
            {"role": "user", "content": (
                f"Question being answered: {prompt}\n\n"
                f"Full search trajectory ({len(steps)} steps, "
                f"{len(trajectory_text)} chars):\n\n"
                f"{trajectory_text}\n\n"
                "Compress this trajectory into a summary of roughly "
                "1000-2000 characters, preserving all critical information "
                "needed to answer the question."
            )},
        ]

        try:
            response = self.llm_fn(messages)
            summary = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            ).strip()

            if summary:
                compressed_step: Dict[str, Any] = {
                    "step": "compressed",
                    "type": "compression",
                    "original_steps": len(steps),
                    "original_chars": len(trajectory_text),
                    "summary_chars": len(summary),
                    "summary": summary,
                }
                # Keep the most recent step alongside the compression
                # so the agent has immediate context
                recent = steps[-1:] if len(steps) > 1 else []
                return [compressed_step] + recent
        except Exception:
            pass

        # LLM call failed — fall back to mechanical compression
        return self._mechanical_compress(steps)

    @staticmethod
    def _mechanical_compress(
        steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Mechanical fallback: keep first + last 2 steps, drop the middle.

        Used only when no LLM is available for proper compression.
        """
        if len(steps) <= 3:
            return steps

        first = steps[0]
        recent = steps[-2:]
        middle = steps[1:-2]

        summary_parts = []
        for s in middle:
            stype = s.get("type", "?")
            if stype == "retrieval":
                summary_parts.append(
                    f"[retrieval: {s.get('num_results', 0)} docs for "
                    f"'{s.get('query', '?')[:50]}']"
                )
            elif stype == "reasoning":
                thought = s.get("thought", "")[:100]
                summary_parts.append(f"[reasoning: {thought}]")
            elif stype == "answer":
                summary_parts.append(
                    f"[answer attempt: {s.get('answer', '')[:100]}]"
                )

        compressed: Dict[str, Any] = {
            "step": "compressed",
            "type": "compression",
            "original_steps": len(middle),
            "summary": " | ".join(summary_parts),
        }

        return [first, compressed] + recent

    def _evaluate(
        self,
        predicted: str,
        reference: str,
        question: str = "",
        nuggets: Optional[List[str]] = None,
    ) -> bool:
        """Evaluate answer correctness.

        Uses LLM-based nugget evaluation when ``llm_fn`` is available
        (KARL paper Section 2.3, Appendix D.1 Figure 31).  Each nugget
        is judged as support (1.0) / partial_support (0.5) /
        not_support (0.0), and the mean score determines pass/fail.

        Falls back to heuristic token matching when no LLM is available.
        """
        # --- LLM-based nugget evaluation ---
        if self.llm_fn is not None:
            return self._llm_evaluate(predicted, reference, question, nuggets)

        # --- Legacy nugget scorer ---
        if self.nugget_scorer is not None:
            result = self.nugget_scorer.score(predicted, reference)
            return result.get("score", 0.0) >= 0.5

        # --- Heuristic fallback ---
        return _heuristic_evaluate(predicted, reference)

    def _decompose_nuggets(
        self, question: str, reference: str,
    ) -> List[str]:
        """Decompose a reference answer into atomic decompositional facts.

        Called once per question in ``generate_group()`` and reused across
        all rollouts for that question.
        """
        messages = [
            {"role": "system", "content": (
                "Decompose the reference answer into a list of atomic, "
                "independently verifiable facts (nuggets). Each nugget should "
                "be a single specific claim that can be judged as supported or "
                "not supported by a candidate answer.\n\n"
                "Return ONLY a Python list of strings, one per fact. "
                "Do not include any explanation.\n\n"
                "Example:\n"
                'Question: "When and where was the first heart transplant?"\n'
                'Answer: "The first successful human heart transplant was '
                'performed by Christiaan Barnard on December 3, 1967, at '
                'Groote Schuur Hospital in Cape Town, South Africa."\n'
                "Nuggets:\n"
                '["Christiaan Barnard performed the first heart transplant", '
                '"The transplant occurred on December 3, 1967", '
                '"It took place at Groote Schuur Hospital", '
                '"The hospital is in Cape Town, South Africa"]'
            )},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Answer: {reference}\n"
                f"Nuggets:"
            )},
        ]

        try:
            response = self.llm_fn(messages, max_new_tokens=512)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            # Strip thinking tags
            content = _re.sub(r'<think>.*?</think>\s*', '', content, flags=_re.DOTALL)
            content = _re.sub(r'<think>.*', '', content, flags=_re.DOTALL).strip()

            # Parse as Python list
            match = _re.search(r'\[.*\]', content, _re.DOTALL)
            if match:
                nuggets = _json.loads(match.group())
                if isinstance(nuggets, list) and all(isinstance(n, str) for n in nuggets):
                    return [n.strip() for n in nuggets if n.strip()]

            # Fallback: split by newlines / bullets
            lines = []
            for line in content.split("\n"):
                line = _re.sub(r'^[\s\-\*\d\.\)]+', '', line).strip().strip('"')
                if line and len(line) > 5:
                    lines.append(line)
            if lines:
                return lines

        except Exception:
            pass

        # If decomposition fails, treat the whole answer as one nugget
        return [reference]

    def _llm_evaluate(
        self,
        predicted: str,
        reference: str,
        question: str = "",
        nuggets: Optional[List[str]] = None,
    ) -> bool:
        """LLM-based nugget-completeness evaluation (KARL Figure 31).

        Each nugget is scored:
        - support (1.0): answer fully captures the fact
        - partial_support (0.5): answer partially captures it
        - not_support (0.0): answer does not capture it

        Pass if mean score >= 0.5.
        """
        if not nuggets:
            nuggets = [reference]

        nugget_text = "\n".join(f"  {i+1}. {n}" for i, n in enumerate(nuggets))

        messages = [
            {"role": "system", "content": (
                "You will evaluate whether an answer sufficiently supports "
                "each decompositional fact.\n\n"
                "Process:\n"
                f"1. Read the question and the answer.\n"
                f"2. Read each of the {len(nuggets)} decompositional facts "
                "carefully one by one.\n"
                "3. Based on the question and answer, judge whether the answer "
                "supports, partially supports, or does not support each fact.\n\n"
                "IMPORTANT: Judge based on MEANING, not exact wording. "
                "Paraphrases, synonyms, and semantic equivalents count. "
                'For example: "near Naples" supports "from Pozzuoli" '
                '(Pozzuoli is near Naples); "1700s" supports "18th century"; '
                '"volcanic material" partially supports "pozzolana" '
                "(same substance, different name).\n\n"
                'Label Definitions:\n'
                '- "support": The answer captures the essential meaning of '
                "the fact, even if using different words.\n"
                '- "partial_support": The answer partially captures the fact '
                "(e.g., refers to the concept without full specificity).\n"
                '- "not_support": The answer does not capture or does not '
                "provide information entailing the fact.\n\n"
                "Output Format: Return ONLY a Python list of label strings, "
                "in the same order as the facts. No explanation.\n"
                'Example: ["support", "not_support", "partial_support"]'
            )},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Answer: {predicted}\n"
                f"Decompositional Facts:\n{nugget_text}\n"
                f"Labels:"
            )},
        ]

        try:
            response = self.llm_fn(messages, max_new_tokens=1024)
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            content = _re.sub(r'<think>.*?</think>\s*', '', content, flags=_re.DOTALL)
            content = _re.sub(r'<think>.*', '', content, flags=_re.DOTALL).strip()

            # Parse the label list
            match = _re.search(r'\[.*\]', content, _re.DOTALL)
            if match:
                labels = _json.loads(match.group())
                if isinstance(labels, list):
                    score_map = {
                        "support": 1.0,
                        "partial_support": 0.5,
                        "not_support": 0.0,
                    }
                    scores = []
                    for label in labels:
                        label_str = str(label).lower().strip().replace(" ", "_")
                        scores.append(score_map.get(label_str, 0.0))
                    if scores:
                        return (sum(scores) / len(scores)) >= 0.5

        except Exception:
            pass

        # If LLM judge fails, fall back to heuristic
        return _heuristic_evaluate(predicted, reference)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heuristic_evaluate(predicted: str, reference: str) -> bool:
    """Heuristic answer-correctness evaluation (fallback when no LLM judge).

    Multi-tier approach:
    1. Exact match (normalized)
    2. Containment (reference in prediction)
    3. Key-phrase / nugget matching
    4. Token-level F1 (threshold 0.5)
    """
    def normalize(text: str) -> str:
        text = _re.sub(r'\[Document \d+\]', '', text)
        text = _re.sub(r'\(Document \d+\)', '', text)
        text = _re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.lower().split())

    pred_norm = normalize(predicted)
    ref_norm = normalize(reference)

    if pred_norm == ref_norm:
        return True
    if ref_norm in pred_norm:
        return True

    # Key-phrase matching
    nuggets = _extract_key_phrases(reference)
    if nuggets:
        matched = sum(1 for n in nuggets if n.lower() in pred_norm)
        if matched >= len(nuggets) * 0.6:
            return True

    # Token F1
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


def _extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from a reference answer for heuristic matching."""
    phrases: List[str] = []
    for m in _re.finditer(r"\(([^)]+)\)", text):
        phrases.append(m.group(1).strip())
    for m in _re.finditer(r'"([^"]+)"', text):
        phrases.append(m.group(1).strip())
    common = {"The", "This", "That", "These", "Those", "Each", "Some", "Any"}
    for m in _re.finditer(r"\b([A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+)*)\b", text):
        phrase = m.group(1)
        if phrase not in common and len(phrase) > 2:
            phrases.append(phrase)
    return phrases
