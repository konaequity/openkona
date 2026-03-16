from __future__ import annotations

import json as _json
import random
import re as _re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from konash.agent import Agent as BaseAgent
from konash.harness.environment import Environment
from konash.plugins.compression import CompressionPlugin
from konash.plugins.control import StepBudgetPlugin


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
        self._tls = threading.local()  # thread-local storage for per-rollout state

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

        # Vary temperature across rollouts for diversity (KARL paper uses
        # independent rollouts which naturally diverge at scale; at small
        # scale we need explicit temperature variation).
        base_temp = 0.7
        temp_offsets = [0.0, 0.15, -0.1, 0.25, 0.05, -0.05, 0.3, 0.1]
        rollout_args = []
        for i in range(num_rollouts):
            temp = base_temp + temp_offsets[i % len(temp_offsets)]
            temp = max(0.1, min(1.2, temp))  # clamp
            rollout_args.append((i, temp))

        # Run rollouts in parallel — they are independent I/O-bound LLM chains
        max_workers = min(num_rollouts, 4)
        rollouts: List[Optional[Rollout]] = [None] * num_rollouts

        def _run_rollout(idx_temp):
            idx, temp = idx_temp
            return idx, self.generate_single(
                prompt,
                reference_answer=reference_answer,
                rollout_id=idx,
                qa_idx=qa_idx,
                nuggets=nuggets,
                temperature=temp,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, rollout in pool.map(_run_rollout, rollout_args):
                rollouts[idx] = rollout

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
        temperature: Optional[float] = None,
    ) -> Rollout:
        """Generate a single reasoning rollout for a prompt.

        Uses the same ``Environment`` / ``Agent`` harness as inference
        (``solve()``), ensuring identical agent behavior during training
        data collection and serving (KARL paper Section 6.2).

        Returns
        -------
        Rollout
        """
        # -- Build tool executor wrapping our search tool --
        def _tool_executor(tool_call: Any) -> Dict[str, Any]:
            query_text = _extract_tool_query(tool_call)
            results = self._retrieve(query_text, self.top_k)
            result_text = "\n\n".join(
                f"[{i+1}] (score: {r.get('score', 0):.3f}) {r.get('text', '')}"
                if isinstance(r, dict) else f"[{i+1}] {r}"
                for i, r in enumerate(results)
            )
            observation: Dict[str, Any] = {"role": "tool", "content": result_text}
            if isinstance(tool_call, dict) and tool_call.get("id"):
                observation["tool_call_id"] = tool_call["id"]
            return observation

        # -- Plugins matching the inference path --
        plugins: List[Any] = [StepBudgetPlugin(max_steps=self.max_steps)]
        if self.compression_trigger_chars:
            threshold_tokens = self.compression_trigger_chars // 4
            plugins.append(CompressionPlugin(
                threshold_tokens=threshold_tokens,
                target_tokens=threshold_tokens // 2,
            ))

        search_tool_schema = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the knowledge base for relevant documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }]

        env = Environment(
            tool_executor=_tool_executor,
            plugins=plugins,
            available_tools=search_tool_schema,
        )

        # -- Build LLM client with temperature support --
        class _LLMClient:
            def __init__(self, llm_fn, temp):
                self._fn = llm_fn
                self._temp = temp

            def generate(self, messages, **kwargs):
                if self._temp is not None:
                    kwargs.setdefault("temperature", self._temp)
                return self._fn(messages, **kwargs)

        # Use the same system prompt as api.py:solve()
        agent = BaseAgent(
            llm_client=_LLMClient(self.llm_fn, temperature) if self.llm_fn else None,
            system_prompt=(
                "You are a knowledge agent. You have access to a search tool that "
                "retrieves relevant documents from a knowledge base. Use it to find "
                "evidence before answering. Search iteratively — refine your queries "
                "based on what you find. When you have enough evidence, provide a "
                "clear, well-supported answer."
            ),
            max_steps=self.max_steps,
        )

        # -- Run through the harness --
        env.reset(prompt=prompt)
        result = env.run_episode(agent, max_steps=self.max_steps)

        # -- Convert episode result to Rollout format --
        steps = _episode_to_steps(result)
        final_answer = result.get("final_answer")

        # Fire progress callbacks
        for step in steps:
            self._notify(qa_idx, rollout_id, step.get("step", 0), step)

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
                "history": result.get("history", []),
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
        # Short answers (< 10 words) are already atomic — skip LLM call
        if len(reference.split()) < 10:
            return [reference]

        messages = [
            {"role": "system", "content": (
                "Decompose the reference answer into a list of atomic, "
                "independently verifiable facts (nuggets). Each nugget should "
                "be a single specific claim that can be judged as supported or "
                "not supported by a candidate answer.\n\n"
                "CRITICAL RULES:\n"
                "- ONLY decompose facts stated IN THE ANSWER TEXT itself\n"
                "- Do NOT include facts from the question — the question is "
                "provided only for context\n"
                "- Each nugget must be something the ANSWER explicitly states\n"
                "- If the answer is a short entity name or phrase, return it "
                "as a single nugget\n\n"
                "Return ONLY a Python list of strings, one per fact. "
                "Do not include any explanation.\n\n"
                "Example 1 (long answer):\n"
                'Question: "When and where was the first heart transplant?"\n'
                'Answer: "The first successful human heart transplant was '
                'performed by Christiaan Barnard on December 3, 1967, at '
                'Groote Schuur Hospital in Cape Town, South Africa."\n'
                "Nuggets:\n"
                '["Christiaan Barnard performed the first heart transplant", '
                '"The transplant occurred on December 3, 1967", '
                '"It took place at Groote Schuur Hospital", '
                '"The hospital is in Cape Town, South Africa"]\n\n'
                "Example 2 (short answer):\n"
                'Question: "Who wrote Romeo and Juliet?"\n'
                'Answer: "William Shakespeare"\n'
                "Nuggets:\n"
                '["William Shakespeare"]'
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
            # Strip thinking / reasoning tags from various models
            content = _re.sub(r'<think>.*?</think>\s*', '', content, flags=_re.DOTALL)
            content = _re.sub(r'<think>.*', '', content, flags=_re.DOTALL)
            content = _re.sub(r'</?(arg_value|think)>', '', content).strip()

            # If content is empty (e.g. GLM reasoning_content not captured),
            # fall back to heuristic immediately
            if not content.strip():
                return _heuristic_evaluate(predicted, reference)

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


def _extract_tool_query(tool_call: Any) -> str:
    """Extract the search query string from a tool call dict."""
    if not isinstance(tool_call, dict):
        return str(tool_call)

    query_text = tool_call.get("query", "") or tool_call.get("input", "")
    if query_text:
        return str(query_text)

    function_call = tool_call.get("function")
    if isinstance(function_call, dict):
        arguments = function_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = _json.loads(arguments)
            except _json.JSONDecodeError:
                return arguments
        if isinstance(arguments, dict):
            nested_query = arguments.get("query", "") or arguments.get("input", "")
            if nested_query:
                return str(nested_query)

    return str(tool_call)


def _episode_to_steps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an Environment episode result into Rollout step records.

    Preserves the step structure that downstream consumers (OAPL trainer,
    pass-rate filter, serialization) expect.
    """
    steps: List[Dict[str, Any]] = []
    trajectory = result.get("trajectory", [])

    for idx, step_record in enumerate(trajectory):
        agent_response = step_record.get("agent_response")
        tool_results = step_record.get("tool_results", [])

        if agent_response is None:
            # Plugin terminated early (e.g. step budget exhausted)
            continue

        step: Dict[str, Any] = {"step": idx}

        if tool_results:
            # Agent made a tool call — this is a retrieval step
            content = agent_response.get("content", "")
            tool_calls = agent_response.get("tool_calls", [])
            query = ""
            if tool_calls:
                query = _extract_tool_query(tool_calls[0])

            tool_content = ""
            for tr in tool_results:
                if isinstance(tr, dict):
                    tool_content += tr.get("content", "")

            step["type"] = "retrieval"
            step["query"] = query
            step["thought"] = content
            step["results_text"] = tool_content
            step["num_results"] = tool_content.count("[") if tool_content else 0
        else:
            # No tool call — this is an answer step
            content = agent_response.get("content", "")
            step["type"] = "answer"
            step["answer"] = content
            step["thought"] = content

        steps.append(step)

    return steps


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
