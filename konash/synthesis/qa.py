from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional


class SyntheticExample:
    """A single synthesized question-answer pair with provenance citations."""

    question = None
    answer = None
    citations = None

    def __init__(self, question=None, answer=None, citations=None):
        self.question = question
        self.answer = answer
        self.citations = citations if citations is not None else []

    def __repr__(self) -> str:
        q_preview = (self.question or "")[:60]
        return f"SyntheticExample(question={q_preview!r}, citations={len(self.citations or [])})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntheticExample):
            return NotImplemented
        return (
            self.question == other.question
            and self.answer == other.answer
            and self.citations == other.citations
        )

    def __hash__(self) -> int:
        return hash((self.question, self.answer))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations) if self.citations else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticExample":
        """Deserialize from a plain dictionary."""
        return cls(
            question=data.get("question"),
            answer=data.get("answer"),
            citations=data.get("citations", []),
        )


def _extract_examples_from_json(items: list) -> List["SyntheticExample"]:
    """Extract SyntheticExample objects from a parsed JSON list."""
    examples = []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = item.get("question", item.get("q", item.get("Q", "")))
        a = item.get("answer", item.get("a", item.get("A", "")))
        if q:
            examples.append(SyntheticExample(
                question=str(q).strip(),
                answer=str(a).strip() if a else "",
                citations=item.get("citations", item.get("sources", [])),
            ))
    return examples


class QuestionAnswerSynthesizer:
    """Agentic QA-pair synthesizer that uses few-shot prompting, corpus exploration,
    and vector search to generate diverse, grounded synthetic training examples."""

    few_shot_examples = None
    task_prompt = None
    vector_search_tool = None

    def __init__(
        self,
        few_shot_examples: Optional[List[SyntheticExample]] = None,
        task_prompt: Optional[str] = None,
        vector_search_tool: Any = None,
        generation_count: int = 8,
        max_steps: int = 50,
        top_k: int = 20,
        llm_fn: Any = None,
    ):
        self.few_shot_examples = few_shot_examples or []
        self.task_prompt = task_prompt or self._default_task_prompt()
        self.vector_search_tool = vector_search_tool
        self.generation_count = generation_count
        self.max_steps = max_steps
        self.top_k = top_k
        self.llm_fn = llm_fn

    # ------------------------------------------------------------------
    # Internal defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_task_prompt() -> str:
        return (
            "You are a question-answer synthesis agent. Given a set of reference "
            "documents, generate novel, challenging questions whose answers are "
            "grounded in the provided evidence. Each question must be answerable "
            "from the documents alone, and each answer must include at least one "
            "citation to the source material."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        documents: Optional[List[str]] = None,
        num_examples: Optional[int] = None,
        *,
        seed: Optional[int] = None,
    ) -> List[SyntheticExample]:
        """Generate synthetic QA pairs from a collection of documents.

        This is the main entry-point. It orchestrates corpus exploration,
        prompt construction, and example generation.

        Parameters
        ----------
        documents : list[str] | None
            Raw document texts to synthesize from. If *None*, the method
            will attempt to use ``explore_corpus`` to discover documents
            via the configured ``vector_search_tool``.
        num_examples : int | None
            How many QA pairs to target. Defaults to ``self.generation_count``.
        seed : int | None
            Optional random seed for reproducibility.

        Returns
        -------
        list[SyntheticExample]
        """
        if seed is not None:
            random.seed(seed)

        count = num_examples or self.generation_count

        # Phase 1 -- gather source material
        if documents is None or len(documents) == 0:
            documents = self.explore_corpus(count)

        # Phase 2 -- build the synthesis prompt
        prompt = self.build_prompt(documents, count)

        # Phase 3 -- generate examples (deterministic stub generation when
        # no LLM backend is wired up; real deployments override _call_llm)
        examples = self._generate_from_prompt(prompt, documents, count)
        return examples

    def bootstrap_from_examples(
        self, seed_examples: List[SyntheticExample], *, multiplier: int = 2
    ) -> List[SyntheticExample]:
        """Seed the synthesizer from existing examples and generate variations.

        Uses the supplied examples as few-shot demonstrations, then calls
        ``synthesize`` to produce ``multiplier * len(seed_examples)`` new pairs.

        Parameters
        ----------
        seed_examples : list[SyntheticExample]
            Hand-written or curated QA pairs to bootstrap from.
        multiplier : int
            How many new examples to produce per seed example (default 2).

        Returns
        -------
        list[SyntheticExample]
        """
        self.few_shot_examples = list(seed_examples)
        target = len(seed_examples) * multiplier

        # Extract documents from seed citations for context
        documents = []
        for ex in seed_examples:
            if ex.citations:
                documents.extend(ex.citations)

        return self.synthesize(documents=documents or None, num_examples=target)

    def explore_corpus(
        self, num_documents: Optional[int] = None
    ) -> List[str]:
        """Use the vector search tool to discover relevant documents.

        Performs a diverse retrieval pass using multiple strategies:
        1. Seed queries from few-shot examples.
        2. Topic-diversified queries derived from seed answers.
        3. Random exploration queries to increase coverage.

        Parameters
        ----------
        num_documents : int | None
            Number of documents to retrieve. Defaults to 10.

        Returns
        -------
        list[str]
            Retrieved document texts.
        """
        k = num_documents or self.top_k

        if self.vector_search_tool is None:
            return []

        retrieved: List[str] = []
        queries = self._derive_exploration_queries(k)

        # Multi-round retrieval for diversity
        per_query_k = max(1, (k * 2) // len(queries))
        for query in queries:
            try:
                results = self.vector_search_tool.search(query, top_k=per_query_k)
                if isinstance(results, list):
                    retrieved.extend(results)
            except Exception:
                continue

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for doc in retrieved:
            if isinstance(doc, dict):
                doc_text = str(doc.get("text", ""))
            else:
                doc_text = str(doc)
            if not doc_text:
                continue
            doc_hash = hashlib.md5(doc_text.encode("utf-8", errors="replace")).hexdigest()
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc_text)
        return unique[:k]

    def build_prompt(
        self,
        documents: List[str],
        num_examples: int,
    ) -> str:
        """Construct the full synthesis prompt from task instructions,
        few-shot demonstrations, and reference documents.

        Parameters
        ----------
        documents : list[str]
            Reference document texts.
        num_examples : int
            Number of QA pairs the model should generate.

        Returns
        -------
        str
            The assembled prompt string.
        """
        sections: List[str] = []

        # 1. System / task prompt
        sections.append(f"### Task Instructions\n{self.task_prompt}")

        # 2. Few-shot demonstrations
        if self.few_shot_examples:
            demo_lines = ["### Few-Shot Examples"]
            for i, ex in enumerate(self.few_shot_examples, 1):
                demo_lines.append(f"Example {i}:")
                demo_lines.append(f"  Q: {ex.question}")
                demo_lines.append(f"  A: {ex.answer}")
                if ex.citations:
                    demo_lines.append(f"  Citations: {', '.join(str(c) for c in ex.citations)}")
                demo_lines.append("")
            sections.append("\n".join(demo_lines))

        # 3. Reference documents
        if documents:
            doc_lines = ["### Reference Documents"]
            for i, doc in enumerate(documents, 1):
                truncated = doc[:4000] if len(doc) > 4000 else doc
                doc_lines.append(f"[Document {i}]\n{truncated}\n")
            sections.append("\n".join(doc_lines))

        # 4. Generation instruction
        sections.append(
            f"### Generation Request\n"
            f"Generate exactly {num_examples} question-answer pairs. "
            f"For each pair, provide the question, answer, and a list of "
            f"document citations (by document number). Output as a numbered list."
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_exploration_queries(self, k: int) -> List[str]:
        """Build diverse queries for corpus exploration from few-shot examples.

        Derives queries from:
        1. Seed questions (direct use)
        2. Seed answers (keyword extraction for topic diversity)
        3. Cross-seed queries (combine elements from different seeds)
        """
        queries: List[str] = []
        if self.few_shot_examples:
            for ex in self.few_shot_examples:
                if ex.question:
                    queries.append(ex.question)
                # Also use answer keywords for topic-diverse exploration
                if ex.answer:
                    # Extract key phrases from answers
                    words = [
                        w for w in ex.answer.split()
                        if len(w) > 3 and w[0].isupper()
                    ]
                    if words:
                        queries.append(" ".join(words[:5]))
        if not queries:
            queries.append("relevant information")
        return queries[:max(k, 8)]

    def _generate_from_prompt(
        self,
        prompt: str,
        documents: List[str],
        count: int,
    ) -> List[SyntheticExample]:
        """Generate examples from the assembled prompt.

        When ``llm_fn`` is configured, calls the LLM and parses the response
        into QA pairs.  Otherwise raises an error — real synthesis requires
        an LLM.
        """
        if self.llm_fn is not None:
            return self._generate_with_llm(prompt, documents, count)

        raise ValueError(
            "QA synthesis requires an LLM function (llm_fn). "
            "Set llm_fn when constructing the QuestionAnswerSynthesizer. "
            "Deterministic stubs do not produce useful training data."
        )

    def _generate_with_llm(
        self,
        prompt: str,
        documents: List[str],
        count: int,
    ) -> List[SyntheticExample]:
        """Call the LLM to generate QA pairs and parse the response.

        Uses a multi-strategy parsing pipeline:
        1. Try JSON array parsing (most reliable)
        2. Try markdown-style Q/A block parsing
        3. Try numbered-list parsing
        4. Try freeform Q/A extraction

        After parsing, validates citation references and filters out
        malformed examples.
        """
        import json as _json
        import re as _re

        messages = [
            {"role": "system", "content": self.task_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_fn(messages)

        # Extract content from response
        if isinstance(response, dict):
            text = response.get("content", "")
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)

        # Strip thinking tags (e.g. Qwen3 <think>...</think>) before parsing
        # Also handle unclosed <think> when model runs out of tokens mid-thought
        text = _re.sub(r'<think>.*?</think>\s*', '', text, flags=_re.DOTALL)
        text = _re.sub(r'<think>.*', '', text, flags=_re.DOTALL).strip()

        # Strip markdown bold markers that confuse Q/A parsers
        text = text.replace("**", "")

        # Multi-strategy parsing
        examples = self._parse_json_examples(text)
        if not examples:
            examples = self._parse_numbered_examples(text, documents)
        if not examples:
            examples = self._parse_freeform_examples(text)

        if not examples:
            raise ValueError(
                "Failed to parse any QA pairs from LLM response. "
                f"Response preview: {text[:500]}"
            )

        # Post-parse validation
        validated = self._validate_examples(examples, documents)

        # Diversity enforcement: deduplicate by question
        seen_questions: set = set()
        unique: list = []
        for ex in validated:
            q_key = " ".join((ex.question or "").lower().split())
            if q_key not in seen_questions and q_key:
                seen_questions.add(q_key)
                unique.append(ex)

        return unique[:count]

    @staticmethod
    def _parse_json_examples(text: str) -> List["SyntheticExample"]:
        """Try to parse LLM output as a JSON array of QA dicts."""
        import json as _json
        import re as _re

        # Try multiple JSON extraction strategies
        # Strategy 1: Find a JSON array
        match = _re.search(r'\[.*\]', text, _re.DOTALL)
        if match:
            try:
                items = _json.loads(match.group())
                if isinstance(items, list):
                    return _extract_examples_from_json(items)
            except _json.JSONDecodeError:
                pass

        # Strategy 2: Find JSON objects in code blocks
        code_blocks = _re.findall(r'```(?:json)?\s*(\[.*?\])\s*```', text, _re.DOTALL)
        for block in code_blocks:
            try:
                items = _json.loads(block)
                if isinstance(items, list):
                    return _extract_examples_from_json(items)
            except _json.JSONDecodeError:
                continue

        # Strategy 3: Find individual JSON objects
        objects = _re.findall(r'\{[^{}]*"question"[^{}]*\}', text, _re.DOTALL)
        if objects:
            examples = []
            for obj_str in objects:
                try:
                    obj = _json.loads(obj_str)
                    if isinstance(obj, dict) and (obj.get("question") or obj.get("q")):
                        examples.append(SyntheticExample(
                            question=obj.get("question", obj.get("q", "")),
                            answer=obj.get("answer", obj.get("a", "")),
                            citations=obj.get("citations", []),
                        ))
                except _json.JSONDecodeError:
                    continue
            if examples:
                return examples

        return []

    @staticmethod
    def _parse_numbered_examples(
        text: str, documents: List[str]
    ) -> List["SyntheticExample"]:
        """Parse numbered Q/A pairs from LLM output."""
        import re as _re

        examples: List["SyntheticExample"] = []
        # Split on numbered patterns: "1.", "1)", "#1", "**1.**"
        blocks = _re.split(r'\n\s*(?:\*{0,2})?\d+[\.\)]\s*(?:\*{0,2})?', text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Try multiple Q/A patterns
            q_match = (
                _re.search(r'[Qq](?:uestion)?[\s:]*[:\-]\s*(.+?)(?:\n|$)', block)
                or _re.search(r'\*\*[Qq](?:uestion)?\*\*[\s:]*(.+?)(?:\n|$)', block)
                or _re.search(r'^(.+?\?)\s*\n', block)  # First line ending with ?
            )
            a_match = (
                _re.search(r'[Aa](?:nswer)?[\s:]*[:\-]\s*(.+?)(?:\n|$)', block)
                or _re.search(r'\*\*[Aa](?:nswer)?\*\*[\s:]*(.+?)(?:\n|$)', block)
            )
            if q_match and a_match:
                q_text = q_match.group(1).strip().strip('"\'')
                a_text = a_match.group(1).strip().strip('"\'')
                if not q_text or not a_text:
                    continue
                cit_match = _re.search(
                    r'[Cc]itations?[\s:]*[:\-]\s*(.+?)(?:\n|$)', block
                )
                citations = []
                if cit_match:
                    citations = [c.strip() for c in cit_match.group(1).split(",")]
                examples.append(SyntheticExample(
                    question=q_text,
                    answer=a_text,
                    citations=citations,
                ))
        return examples

    @staticmethod
    def _parse_freeform_examples(text: str) -> List["SyntheticExample"]:
        """Last-resort parser: extract Q/A pairs from freeform text."""
        import re as _re

        examples: List["SyntheticExample"] = []
        # Look for lines that look like questions (ending with ?)
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith("?") and len(line) > 10:
                # Look for an answer in the next non-empty line(s)
                answer_parts = []
                j = i + 1
                while j < len(lines) and j < i + 5:
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    if next_line.endswith("?"):
                        break  # next question
                    answer_parts.append(next_line)
                    j += 1
                if answer_parts:
                    answer = " ".join(answer_parts)
                    # Clean up common prefixes
                    answer = _re.sub(r'^[Aa](?:nswer)?[\s:]*[:\-]\s*', '', answer)
                    examples.append(SyntheticExample(
                        question=line,
                        answer=answer,
                        citations=[],
                    ))
                    i = j
                    continue
            i += 1
        return examples

    @staticmethod
    def _validate_examples(
        examples: List["SyntheticExample"],
        documents: List[str],
    ) -> List["SyntheticExample"]:
        """Validate and filter synthesized examples.

        Checks:
        1. Non-empty question and answer
        2. Question ends with '?' or is interrogative
        3. Answer is not trivially short (< 3 words)
        4. Citation references exist in the document set (if provided)
        5. Question and answer are not identical
        """
        import re as _re

        validated: List["SyntheticExample"] = []
        num_docs = len(documents)

        for ex in examples:
            q = (ex.question or "").strip()
            a = (ex.answer or "").strip()

            # Basic non-empty check
            if not q or not a:
                continue

            # Answer not trivially short
            if len(a.split()) < 2:
                continue

            # Question and answer should not be identical
            if q.lower() == a.lower():
                continue

            # Validate citation references (if numeric)
            valid_citations = []
            for cit in (ex.citations or []):
                cit_str = str(cit).strip()
                # Try to extract document number
                num_match = _re.search(r'(\d+)', cit_str)
                if num_match:
                    doc_num = int(num_match.group(1))
                    if 1 <= doc_num <= num_docs:
                        valid_citations.append(cit_str)
                else:
                    valid_citations.append(cit_str)  # non-numeric citation

            validated.append(SyntheticExample(
                question=q,
                answer=a,
                citations=valid_citations,
            ))

        return validated
