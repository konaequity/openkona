from __future__ import annotations

import hashlib
import random
import re as _re
import json as _json
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
    """Agentic QA-pair synthesizer that iteratively explores a corpus via
    vector search to generate diverse, grounded, multi-constraint questions.

    Matches the KARL paper Section 4.1 Stage I: the synthesis agent uses
    vector search as a tool, taking multiple exploration steps before
    proposing question-answer pairs grounded in retrieved evidence.

    The agent loop:
    1. SEARCH the corpus with a query → reads results
    2. Reason about what's interesting, find cross-document connections
    3. SEARCH again with a refined/different query
    4. PROPOSE multi-constraint QA pairs grounded in discovered evidence
    5. Repeat until enough examples are generated or max_steps is reached

    Parameters
    ----------
    few_shot_examples : list[SyntheticExample] | None
        Seed examples showing what good questions look like.
    task_prompt : str | None
        Additional task-specific instructions appended to the system prompt.
    vector_search_tool : object | None
        Must expose ``.search(query, top_k=...) -> list``.
    generation_count : int
        Default number of QA pairs to generate (default 8).
    max_steps : int
        Maximum agent steps (searches + proposals) per synthesis run
        (KARL paper: 50 for TREC-Biogen, 60 for BrowseComp-Plus).
    top_k : int
        Documents to retrieve per search (KARL paper: 20 for TREC, 5 for BCP).
    llm_fn : callable
        ``(messages, **kwargs) -> {"role": "assistant", "content": "..."}``.
    """

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
        self.task_prompt = task_prompt
        self.vector_search_tool = vector_search_tool
        self.generation_count = generation_count
        self.max_steps = max_steps
        self.top_k = top_k
        self.llm_fn = llm_fn

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
        """Generate synthetic QA pairs by agentically exploring the corpus.

        The agent iteratively searches the corpus, discovers facts, and
        proposes multi-constraint questions grounded in retrieved evidence.

        Parameters
        ----------
        documents : list[str] | None
            Optional starting documents for context. The agent will still
            search the corpus for additional information.
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

        if self.llm_fn is None:
            raise ValueError(
                "QA synthesis requires an LLM function (llm_fn). "
                "Set llm_fn when constructing the QuestionAnswerSynthesizer."
            )

        examples = self._agentic_synthesize(count, documents)

        # Deduplicate by normalized question text
        seen: set = set()
        unique: list = []
        for ex in examples:
            key = " ".join((ex.question or "").lower().split())
            if key and key not in seen:
                seen.add(key)
                unique.append(ex)

        return unique[:count]

    def bootstrap_from_examples(
        self, seed_examples: List[SyntheticExample], *, multiplier: int = 2
    ) -> List[SyntheticExample]:
        """Seed the synthesizer from existing examples and generate variations.

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
        return self.synthesize(num_examples=target)

    def explore_corpus(
        self, num_documents: Optional[int] = None
    ) -> List[str]:
        """Use the vector search tool to discover relevant documents.

        Kept for backward compatibility — the agentic synthesizer does its
        own exploration during ``synthesize()``.
        """
        k = num_documents or self.top_k
        if self.vector_search_tool is None:
            return []

        retrieved: List[str] = []
        queries = self._seed_queries()
        per_query_k = max(1, (k * 2) // max(len(queries), 1))
        for query in queries:
            try:
                results = self.vector_search_tool.search(query, top_k=per_query_k)
                if isinstance(results, list):
                    retrieved.extend(
                        r.get("text", str(r)) if isinstance(r, dict) else str(r)
                        for r in results
                    )
            except Exception:
                continue

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
        """Construct a synthesis prompt (kept for backward compatibility).

        The agentic synthesizer uses multi-turn conversation instead of
        a single prompt, but this method is preserved for callers that
        may still use it.
        """
        sections: List[str] = []
        sections.append(f"### Task Instructions\n{self.task_prompt or ''}")

        if self.few_shot_examples:
            demo_lines = ["### Few-Shot Examples"]
            for i, ex in enumerate(self.few_shot_examples, 1):
                demo_lines.append(f"Example {i}:")
                demo_lines.append(f"  Q: {ex.question}")
                demo_lines.append(f"  A: {ex.answer}")
                demo_lines.append("")
            sections.append("\n".join(demo_lines))

        if documents:
            doc_lines = ["### Reference Documents"]
            for i, doc in enumerate(documents, 1):
                doc_lines.append(f"[Document {i}]\n{doc[:4000]}\n")
            sections.append("\n".join(doc_lines))

        sections.append(
            f"### Generation Request\n"
            f"Generate exactly {num_examples} challenging, multi-constraint "
            f"question-answer pairs."
        )
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Agentic synthesis loop
    # ------------------------------------------------------------------

    def _agentic_synthesize(
        self,
        count: int,
        documents: Optional[List[str]] = None,
    ) -> List[SyntheticExample]:
        """Run a multi-step synthesis agent that explores the corpus.

        The agent maintains a conversation with the LLM where it can:
        - ``SEARCH: <query>`` — search the corpus via vector search
        - ``PROPOSE:`` — propose one or more QA pairs

        The agent searches at least a few times before proposing, building
        up a rich picture of the corpus before generating questions.
        """
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._task_message(count, documents)},
        ]

        proposed: List[SyntheticExample] = []
        search_count = 0
        min_searches = 3
        empty_count = 0

        for step_idx in range(self.max_steps):
            if len(proposed) >= count:
                break

            response = self.llm_fn(messages)
            content = _clean_thinking_tags(response)

            if not content:
                empty_count += 1
                if empty_count >= 2:
                    break
                messages.append({"role": "assistant", "content": "(no response)"})
                messages.append({"role": "user", "content": "SEARCH: "})
                continue
            empty_count = 0

            action = self._parse_action(content, search_count=search_count)

            # Enforce minimum searches before allowing proposals
            if (action["type"] == "propose"
                    and search_count < min_searches
                    and self.vector_search_tool is not None):
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    f"You have only searched {search_count} time(s). "
                    f"You MUST search at least {min_searches} times across "
                    f"different topics before proposing. SEARCH for a new topic:"
                )})
                continue

            if action["type"] == "search" and self.vector_search_tool is not None:
                query = action["query"]
                results = self._search(query)
                search_count += 1

                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": self._format_results(results, query),
                })

            elif action["type"] == "propose":
                new_examples = action["examples"]
                proposed.extend(new_examples)

                messages.append({"role": "assistant", "content": content})

                remaining = count - len(proposed)
                if remaining > 0:
                    messages.append({"role": "user", "content": (
                        f"Recorded {len(new_examples)} question(s). "
                        f"{remaining} more needed. "
                        f"SEARCH for DIFFERENT topics in the corpus — "
                        f"explore new areas you haven't searched yet."
                    )})

            else:
                # Couldn't parse — nudge the agent to search
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "Please respond with exactly ONE action:\n"
                    "SEARCH: <your query>\n"
                    "or\n"
                    "PROPOSE:\nQ: <question>\nA: <answer>"
                )})

        # If we still need more, force the agent to propose from what it found
        if len(proposed) < count and search_count > 0:
            remaining = count - len(proposed)
            messages.append({"role": "user", "content": (
                f"You have explored the corpus with {search_count} searches. "
                f"Now PROPOSE exactly {remaining} question-answer pair(s).\n\n"
                f"RULES for these questions:\n"
                f"- ONLY use facts that appeared in the search results above\n"
                f"- Combine facts from DIFFERENT search results\n"
                f"- Each answer must be verifiable from the search results\n\n"
                f"PROPOSE:"
            )})
            response = self.llm_fn(messages)
            content = _clean_thinking_tags(response)
            if content:
                action = self._parse_action(content)
                if action["type"] == "propose":
                    proposed.extend(action["examples"])

        return proposed

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Build the system prompt for the agentic synthesizer."""
        few_shot = ""
        if self.few_shot_examples:
            lines = []
            for i, ex in enumerate(self.few_shot_examples, 1):
                lines.append(f"  {i}. Q: {ex.question}\n     A: {ex.answer}")
            few_shot = (
                "\n\nExamples of the kind of questions to generate:\n"
                + "\n".join(lines)
            )

        custom = ""
        if self.task_prompt:
            custom = f"\n\nAdditional instructions: {self.task_prompt}"

        return (
            "You are a question-answer synthesis agent. Your job is to explore "
            "a document corpus and create challenging questions whose answers "
            "are ONLY found in the corpus documents.\n\n"
            "You have ONE tool: vector search over the corpus.\n\n"
            "CRITICAL RULES:\n"
            "- Output exactly ONE action per turn: either SEARCH or PROPOSE\n"
            "- You MUST search at least 3 times before proposing\n"
            "- Every question MUST be answerable from the search results you received\n"
            "- Do NOT use your own knowledge — ONLY facts from search results\n"
            "- Do NOT combine SEARCH and PROPOSE in the same response\n\n"
            "WORKFLOW:\n"
            "Turn 1: SEARCH: <query about a topic>\n"
            "  (you will receive search results)\n"
            "Turn 2: SEARCH: <different topic>\n"
            "  (you will receive more results)\n"
            "Turn 3: SEARCH: <yet another topic>\n"
            "  (you will receive more results)\n"
            "Turn 4+: PROPOSE questions that combine facts from your searches\n\n"
            "WHAT MAKES A GOOD QUESTION (retrieval-hard, verification-easy):\n"
            "- Answer is a specific fact: a name, date, number, place, or short list of these\n"
            "- Multiple constraints in the question NARROW to a single answer — each constraint\n"
            "  eliminates candidates until only one entity/fact remains\n"
            "- The facts are buried deep in the documents, not in the first paragraph\n"
            "- A human could verify the answer by pointing to exact sentences in the documents\n"
            "- Example: 'What is the name of the military officer who found the Rosetta Stone,\n"
            "  and under which treaty was it surrendered to Britain?'\n"
            "  → Answer: 'Pierre-François Bouchard discovered it; surrendered under the\n"
            "  Treaty of Alexandria (1801)'\n\n"
            "WHAT MAKES A BAD QUESTION (never generate these):\n"
            "- Answer is an explanation, comparison, or opinion ('How does X relate to Y?')\n"
            "- Answer requires reasoning beyond what the text states ('What principle connects...')\n"
            "- Question asks about similarities, differences, or analogies between topics\n"
            "- Question staples two UNRELATED facts together ('How heavy were theodolites,\n"
            "  and which Roman leader annexed Egypt?' — these have nothing to do with each other)\n"
            "- Answerable from a single sentence ('When was X built?')\n"
            "- Uses facts NOT found in the search results\n\n"
            "NOVELTY: The examples below show the FORMAT and DIFFICULTY level — do NOT\n"
            "reuse the same facts or answers from the examples. Find NEW facts buried\n"
            "deeper in the corpus that you discovered through your own searches.\n\n"
            "ANSWER FORMAT: Every answer MUST be 1-2 sentences using only names, dates,\n"
            "numbers, and places extracted verbatim from the corpus. Do NOT generate\n"
            "questions whose answers require subjective interpretation or explanation.\n\n"
            "FORMAT — one per turn:\n"
            "To search:  SEARCH: <your specific query>\n"
            "To propose: PROPOSE:\n"
            "Q: <question grounded in search results>\n"
            "A: <answer using only facts from search results>\n"
            "(repeat Q:/A: for multiple questions)"
            + custom
            + few_shot
        )

    def _task_message(
        self,
        count: int,
        documents: Optional[List[str]] = None,
    ) -> str:
        """Build the initial user message."""
        doc_context = ""
        if documents:
            doc_lines = []
            for i, doc in enumerate(documents[:3], 1):
                doc_lines.append(f"[Starting Document {i}] {doc[:400]}")
            doc_context = (
                "\n\nHere are some starting documents for context:\n"
                + "\n\n".join(doc_lines)
                + "\n\nUse SEARCH to discover more."
            )

        return (
            f"Generate {count} challenging, multi-constraint questions from "
            f"the corpus. Each question should require combining at least 2 "
            f"facts to answer correctly.\n\n"
            f"Start by searching the corpus to discover what topics it covers."
            + doc_context
        )

    # ------------------------------------------------------------------
    # Search and result formatting
    # ------------------------------------------------------------------

    def _search(self, query: str) -> List[Any]:
        """Execute vector search."""
        if self.vector_search_tool is None:
            return []
        try:
            results = self.vector_search_tool.search(query, top_k=self.top_k)
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def _format_results(self, results: List[Any], query: str) -> str:
        """Format search results as a user message."""
        if not results:
            return (
                f"No results found for '{query}'. "
                f"Try a different search query."
            )

        lines = [f"Results for '{query}' ({len(results)} passages):"]
        for i, r in enumerate(results[:10], 1):
            text = r.get("text", str(r)) if isinstance(r, dict) else str(r)
            lines.append(f"\n[{i}] {text[:600]}")

        lines.append(
            "\n\nYou can SEARCH again for different topics, or PROPOSE "
            "question-answer pairs based on what you've found."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_action(
        self, content: str, *, search_count: int = 0,
    ) -> Dict[str, Any]:
        """Parse the agent's response into a search or propose action.

        Prioritizes SEARCH over PROPOSE when the agent hasn't searched
        enough yet (< 3 searches).  If the agent outputs both in one
        response, the search is executed and the proposals are discarded
        (it will re-propose after seeing real results).
        """
        search_match = _re.search(
            r'SEARCH:\s*(.+?)(?:\n|$)', content, _re.IGNORECASE
        )
        has_propose = bool(_re.search(r'PROPOSE', content, _re.IGNORECASE))

        # If the agent hasn't searched enough, always prioritize search
        if search_match and search_count < 3:
            query = search_match.group(1).strip().strip('"\'')
            if query:
                return {"type": "search", "query": query}

        # If there's a PROPOSE, try to extract QA pairs
        if has_propose:
            examples = self._extract_qa_pairs(content)
            if examples:
                return {"type": "propose", "examples": examples}

        # If there's a search (even after 3 searches), take it
        if search_match:
            query = search_match.group(1).strip().strip('"\'')
            if query:
                return {"type": "search", "query": query}

        # Try to extract QA pairs even without explicit PROPOSE marker
        examples = self._extract_qa_pairs(content)
        if examples:
            return {"type": "propose", "examples": examples}

        # Try JSON format
        examples = _parse_json_qa(content)
        if examples:
            return {"type": "propose", "examples": examples}

        return {"type": "unknown"}

    def _extract_qa_pairs(self, content: str) -> List[SyntheticExample]:
        """Extract Q/A pairs from the agent's response."""
        examples: List[SyntheticExample] = []

        # Find content after PROPOSE marker (if present)
        propose_match = _re.search(
            r'PROPOSE[:\s]*\n?(.*)', content, _re.DOTALL | _re.IGNORECASE
        )
        text = propose_match.group(1) if propose_match else content

        # Strategy 1: Q: ... \n A: ... pattern (handles Q1:, Q2: etc.)
        qa_pairs = _re.findall(
            r'Q\d*:\s*(.+?)\s*\nA\d*:\s*(.+?)(?=\nQ\d*:|\n\n\S|\Z)',
            text, _re.DOTALL
        )
        for q, a in qa_pairs:
            q = q.strip().strip('"').replace("**", "")
            a = a.strip().strip('"').replace("**", "")
            if q and a and len(q) > 15 and len(a) > 2:
                examples.append(SyntheticExample(question=q, answer=a))

        if examples:
            return examples

        # Strategy 2: Numbered list with Question/Answer labels
        blocks = _re.split(r'\n\s*\d+[\.\)]\s*', text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            q_match = (
                _re.search(r'[Qq](?:uestion)?[:\s]+(.+?)(?:\n|$)', block)
                or _re.search(r'^(.+?\?)\s*\n', block)
            )
            a_match = _re.search(r'[Aa](?:nswer)?[:\s]+(.+?)(?:\n|$)', block)
            if q_match and a_match:
                q = q_match.group(1).strip().strip('"').replace("**", "")
                a = a_match.group(1).strip().strip('"').replace("**", "")
                if q and a and len(q) > 15 and len(a) > 2:
                    examples.append(SyntheticExample(question=q, answer=a))

        return examples

    # ------------------------------------------------------------------
    # Seed query helpers
    # ------------------------------------------------------------------

    def _seed_queries(self) -> List[str]:
        """Build seed queries from few-shot examples or corpus sampling."""
        queries: List[str] = []

        if self.few_shot_examples:
            for ex in self.few_shot_examples:
                if ex.question:
                    queries.append(ex.question)
                if ex.answer:
                    words = [
                        w for w in ex.answer.split()
                        if len(w) > 3 and w[0].isupper()
                    ]
                    if words:
                        queries.append(" ".join(words[:5]))

        if not queries and self.vector_search_tool is not None:
            docs = getattr(self.vector_search_tool, "_documents", [])
            if docs:
                step = max(1, len(docs) // 8)
                sampled = docs[::step][:8]
                for doc in sampled:
                    text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
                    first_sentence = text.split(".")[0].strip()
                    if len(first_sentence) > 15:
                        queries.append(first_sentence[:100])

        if not queries:
            queries.append("relevant information")

        return queries[:8]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _clean_thinking_tags(response: Any) -> str:
    """Extract text from an LLM response and strip thinking tags."""
    if isinstance(response, dict):
        content = response.get("content", "")
    elif isinstance(response, str):
        content = response
    else:
        content = str(response)

    content = _re.sub(r'<think>.*?</think>\s*', '', content, flags=_re.DOTALL)
    content = _re.sub(r'<think>.*', '', content, flags=_re.DOTALL)
    return content.strip()


def _parse_json_qa(text: str) -> List[SyntheticExample]:
    """Try to parse QA pairs from JSON in text."""
    # Try JSON array
    match = _re.search(r'\[.*\]', text, _re.DOTALL)
    if match:
        try:
            items = _json.loads(match.group())
            if isinstance(items, list):
                return _extract_examples_from_json(items)
        except _json.JSONDecodeError:
            pass

    # Try individual JSON objects
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
