from __future__ import annotations


class PromptTemplate:
    name = None
    category = None
    template = None
    version = None

    def __init__(self, name=None, category=None, template=None, version=None):
        self.name = name
        self.category = category
        self.template = template
        self.version = version


# ---------------------------------------------------------------------------
# Exact prompts from the KARL paper (Appendix D, Figures 31-36)
# ---------------------------------------------------------------------------

_NUGGET_COMPLETENESS_PROMPT = """\
Nugget-Completeness Prompt

Your Role: You will evaluate whether an answer to a question (which can \
include a code snippet or documentation) sufficiently supports each \
decompositional fact.

Process:
1. Read the question and the answer.
2. Read each of the {length} decompositional facts carefully one by one.
3. Based on the question and answer, judge whether the answer supports, \
partially supports, or does not support each decompositional fact. Read \
every fact and document pair carefully as you would when proofreading.

It may be helpful to ask yourself: "Does the answer provide sufficient \
evidence required to support the decompositional fact?" Be sure to check \
all of the information in the answer.

Label Definitions:
- support: The answer fully captures and entails all necessary parts of \
the decompositional fact.
- partial_support: The answer partially captures the decompositional fact, \
but does not fully capture all necessary parts.
- not_support: The answer does not capture or does not provide information \
entailing the decompositional fact.

Output Format: Return the labels as a Python list of strings (List[str]), \
in the same order as the decompositional facts. Provide a label for each \
fact. Do not provide any explanation or reasoning.
["support", "not_support", "partial_support", ...]

Input:
Question: {question}
Answer: {answer}
Decompositional Facts: {nugget}
Labels:"""

_TREC_DEDUP_PROMPT = """\
Question Deduplication Judge Prompt for TREC-Biogen

Your Role: You are judging whether two questions are semantically \
equivalent or duplicate.

Question 1: {generated_question}
Question 2: {validation_question}

Your Task: Determine if Question 1 and Question 2 are asking for the SAME \
information, even if phrased differently.

Guidelines:
- "What is the capital of France?" and "Which city is the capital of \
France?" are duplicates (same question).
- "What is the capital of France?" and "What is the population of France?" \
are NOT duplicates (different questions).
- "Who invented the telephone?" and "Who created the telephone?" are \
duplicates (same question).
- Minor differences in wording are acceptable if the core question is the same.
- Consider paraphrasing -- different words can ask the same question.

Output Format:
<reasoning>[Brief explanation of judgment]</reasoning>
<duplicate>[yes or no]</duplicate>"""

_BROWSECOMP_DEDUP_PROMPT = """\
Deduplication Judge Prompt for BrowseComp-Plus

You are judging whether two question-answer pairs are duplicates.

Question-Answer Pair 1 (Generated):
Question 1: {generated_question}
Answer 1: {generated_answer}

Question-Answer Pair 2 (Validation Set):
Question 2: {validation_question}
Answer 2: {validation_answer}

Your Task: Determine if these question-answer pairs are about the same \
underlying fact or relationship. Two pairs are duplicates if:
1. They are about the same underlying fact, relationship, or piece of knowledge.
2. This includes "inverse" questions where Q1's answer appears in Q2's \
question and vice versa.

Examples:
- Q1: "Who is the CEO of Apple?" A1: "Tim Cook" vs Q2: "Who leads Apple \
Inc?" A2: "Tim Cook" -> DUPLICATE (same fact)
- Q1: "Who is the CEO of Apple?" A1: "Tim Cook" vs Q2: "Who is Tim Cook?" \
A2: "CEO of Apple" -> DUPLICATE (same fact, inverse framing)
- Q1: "What year was Obama born?" A1: "1961" vs Q2: "When did Obama become \
president?" A2: "2009" -> NOT DUPLICATE (different facts about the same person)
- Q1: "Capital of France?" A1: "Paris" vs Q2: "Largest city in France?" \
A2: "Paris" -> NOT DUPLICATE (different facts, answer happens to be the same)
- Q1: "Who directed Inception?" A1: "Christopher Nolan" vs Q2: "Who \
directed The Dark Knight?" A2: "Christopher Nolan" -> NOT DUPLICATE \
(different facts, same answer)

Output Format:
<reasoning>Analyze whether both pairs encode the same underlying fact or \
relationship</reasoning>
<duplicate>yes or no</duplicate>"""

_SOLVER_PROMPT = """\
Task Solver Prompt

You are a deep research agent. You need to answer the given question by \
interacting with a search engine, using the search tool provided. Please \
perform reasoning and use the tool step by step, in an interleaved manner. \
You may use the search tool multiple times.

Question: {question}

Your response should be in the following format:
Explanation: {your explanation for your final answer. For this explanation \
section only, you should cite your evidence documents inline by enclosing \
their docids in square brackets [] at the end of sentences. For example, [20].}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

_BROWSECOMP_QUALITY_FILTER_PROMPT = """\
Quality Filter Prompt for BrowseComp-Plus

Your Role: You are evaluating synthetic question-answer pairs for training \
data quality.

Evaluation Context:
- The "ground truth" is a single answer generated by the task creator (P1).
- Each task solver attempt is scored as correct or incorrect (binary match).
- Only questions with mixed success (some correct, some incorrect) are evaluated.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Task Solver Agent Attempts (mixed success):
{attempts}

Your Task: Determine if the question and ground truth answer are VALID and \
UNAMBIGUOUS. Consider:
1. Answer is wrong: The synthesized ground truth is factually incorrect.
2. Question is ambiguous: The question has multiple valid interpretations, \
allowing for different correct answers.
3. Question-answer pair is correct: The synthesized answer is correct, and \
the task solver's failures are due to its limitations (question is hard but \
unambiguous).

Guidelines:
- Mark as INVALID if: Ground truth answer is clearly wrong OR question is \
genuinely ambiguous (multiple valid answers exist).
- Mark as VALID if: Ground truth answer is correct and question has one \
clear answer (task solver failures are acceptable).

Output Format:
<reasoning>[Your detailed analysis]</reasoning>
<valid>[yes or no]</valid>"""

_TREC_QUALITY_FILTER_PROMPT = """\
Quality Filter Prompt for TREC-Biogen

Your Role: You are evaluating question-answer pairs for a TREC-style \
information retrieval task.

Evaluation Context:
- The "ground truth" is a set of nuggets (key facts that a good answer \
should cover).
- Each answer is scored by nugget completion percentage (0-100%).
- A score of 70% means 70% of nuggets were mentioned, NOT that the answer \
is "wrong."

Question:
{question}

Required Nuggets (Ground Truth):
{nuggets}

Task Solver Agents Attempts:
{attempts}

Score Statistics:
Average nugget coverage: {avg}%  Best attempt: {max}%  Worst attempt: {min}%

Your Task: Determine if the question and nuggets are VALID for training. \
Consider:
1. Nuggets are problematic: Are the nuggets unclear, overlapping, or \
inconsistent? Do different valid approaches to answering lead to different \
nugget coverage?
2. Question is ambiguous: Does the question have multiple valid \
interpretations that would lead to covering different nuggets?
3. Question and nuggets are valid: The nuggets represent clear, distinct \
facts. Score variation is due to answer quality/completeness, not ambiguity.

Guidelines:
- Mark as INVALID if: nuggets are poorly defined OR question allows \
multiple valid interpretations with different nugget coverage.
- Mark as VALID if: question is clear, nuggets are well-defined, and score \
variation reflects answer quality.

Output Format:
<reasoning>[Your detailed analysis]</reasoning>
<valid>[yes or no]</valid>"""


class PromptRegistry:
    prompts = {
        "figure_31_nugget_completeness": PromptTemplate(
            name="figure_31_nugget_completeness",
            category="nugget_evaluation",
            template=_NUGGET_COMPLETENESS_PROMPT,
            version="1.0",
        ),
        "figure_32_trec_dedup_paraphrase_judge": PromptTemplate(
            name="figure_32_trec_dedup_paraphrase_judge",
            category="dedup_paraphrase_judge",
            template=_TREC_DEDUP_PROMPT,
            version="1.0",
        ),
        "figure_33_browsecomp_dedup_paraphrase_judge": PromptTemplate(
            name="figure_33_browsecomp_dedup_paraphrase_judge",
            category="dedup_paraphrase_judge",
            template=_BROWSECOMP_DEDUP_PROMPT,
            version="1.0",
        ),
        "figure_34_solver_rollout": PromptTemplate(
            name="figure_34_solver_rollout",
            category="rollout_solver",
            template=_SOLVER_PROMPT,
            version="1.0",
        ),
        "figure_35_browsecomp_quality_filter": PromptTemplate(
            name="figure_35_browsecomp_quality_filter",
            category="quality_filter",
            template=_BROWSECOMP_QUALITY_FILTER_PROMPT,
            version="1.0",
        ),
        "figure_36_trec_quality_filter": PromptTemplate(
            name="figure_36_trec_quality_filter",
            category="quality_filter",
            template=_TREC_QUALITY_FILTER_PROMPT,
            version="1.0",
        ),
        "appendix_d1_task_evaluation_prompts": PromptTemplate(
            name="appendix_d1_task_evaluation_prompts",
            category="nugget_evaluation",
            template=_NUGGET_COMPLETENESS_PROMPT,
            version="1.0",
        ),
        "synthesis_qa_generator": PromptTemplate(
            name="synthesis_qa_generator",
            category="synthesis",
            template="Generate a question-answer pair grounded in the retrieved evidence.",
            version="1.0",
        ),
        "nugget_consolidation": PromptTemplate(
            name="nugget_consolidation",
            category="nugget_consolidation",
            template="Consolidate nuggets from multiple references into a unified set.",
            version="1.0",
        ),
        "parallel_thinking_aggregation": PromptTemplate(
            name="parallel_thinking_aggregation",
            category="aggregation",
            template="Given the following answers from parallel rollouts, synthesize a final answer.",
            version="1.0",
        ),
    }

    @classmethod
    def get(cls, name):
        return cls.prompts[name]

    @classmethod
    def list_by_category(cls, category):
        return {k: v for k, v in cls.prompts.items() if v.category == category}
