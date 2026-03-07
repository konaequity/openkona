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
    ):
        self.few_shot_examples = few_shot_examples or []
        self.task_prompt = task_prompt or self._default_task_prompt()
        self.vector_search_tool = vector_search_tool
        self.generation_count = generation_count
        self.max_steps = max_steps

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

        If ``vector_search_tool`` is set, performs a diverse retrieval pass
        using the few-shot examples as seed queries. Otherwise returns an
        empty list.

        Parameters
        ----------
        num_documents : int | None
            Number of documents to retrieve. Defaults to 10.

        Returns
        -------
        list[str]
            Retrieved document texts.
        """
        k = num_documents or 10

        if self.vector_search_tool is None:
            return []

        retrieved: List[str] = []
        queries = self._derive_exploration_queries(k)

        for query in queries:
            try:
                results = self.vector_search_tool.search(query, top_k=max(1, k // len(queries)))
                if isinstance(results, list):
                    retrieved.extend(results)
            except Exception:
                continue

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for doc in retrieved:
            doc_hash = hashlib.md5(doc.encode("utf-8", errors="replace")).hexdigest()
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)
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
        """Build diverse queries for corpus exploration from few-shot examples."""
        queries: List[str] = []
        if self.few_shot_examples:
            for ex in self.few_shot_examples:
                if ex.question:
                    queries.append(ex.question)
        if not queries:
            queries.append("relevant information")
        return queries[:max(k, 5)]

    def _generate_from_prompt(
        self,
        prompt: str,
        documents: List[str],
        count: int,
    ) -> List[SyntheticExample]:
        """Generate examples from the assembled prompt.

        In a full deployment this calls an LLM API. The base implementation
        produces deterministic placeholder examples derived from the input
        documents so that downstream pipeline stages can operate without
        an LLM backend.
        """
        examples: List[SyntheticExample] = []
        for i in range(count):
            doc_idx = i % max(len(documents), 1)
            doc_text = documents[doc_idx] if documents else ""
            snippet = doc_text[:200].strip() if doc_text else f"topic_{i}"

            question = f"What is described in the following passage: '{snippet}'?"
            answer = f"The passage discusses: {snippet}"
            citations = [f"Document {doc_idx + 1}"] if doc_text else []

            examples.append(SyntheticExample(
                question=question,
                answer=answer,
                citations=citations,
            ))
        return examples
