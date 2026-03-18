# FinanceBench Evaluation — KARL Paper Reference

Everything from the KARL paper (Databricks AI Research, March 5 2026) relevant to the FinanceBench evaluation: task definition, corpus construction, evaluation methodology, agent setup, scoring, results, test-time compute scaling, and latency.

---

## 1. Task Definition

**FinanceBench** (Islam et al., 2023) — *Long-document traversal with tabular numerical reasoning.*

The task focuses on navigating lengthy financial reports (SEC filings), often exceeding 100 pages, to locate specific sections or tables. Answering requires extracting dispersed numerical values and calculating the final result.

**Example (Table 1, Figure 28):**
- **Query:** "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
- **Generated Answer:** KARL traverses a lengthy 10-K filing to locate capital expenditure in the cash flow statement and cross-validates the figure against a second table.
- **Exact Answer:** $1,577 million

**Capability tested:** Long-document traversal with tabular numerical reasoning.

---

## 2. Corpus Construction

### Dataset Statistics (Table 2)

| Stat | Value |
|---|---|
| Number of questions (#Q) | 150 |
| Average question token length | 35.3 |
| Relevant chunks per question (mean +/- std) | 1.2 +/- 0.5 |
| Answer nuggets per question | 1.0 +/- 0.0 |
| Number of indexed document chunks (#D) | 53,399 |
| Average document chunk token length | 717.9 |

### Indexing Strategy

- **FinanceBench is indexed at the page level.**
- No dataset-specific re-chunking, semantic augmentation, metadata enrichment, or chunk size tuning.
- Closed-corpus benchmark (no web search) — enables controlled comparison across methods.

### Embedding & Retrieval

- **Embedding model:** Qwen3-0.6B
- **Retrieval:** Vector search (FAISS) with k = 20 documents returned per search call
- The same vector search index is used across all models to ensure tool execution time is not a differentiating factor.

---

## 3. Agent Harness

### Single External Tool: Vector Search

The agent has access to a single tool: **vector search**. The agent generates a sequence of search queries via tool calls and produces a final answer once sufficient information has been gathered.

At each step, the model's context consists of:
- A system prompt
- A trajectory view containing prior tool calls and their outputs

### Context Management via Compression

Compression is triggered automatically when the interaction history exceeds a pre-defined token threshold. The model itself is instructed to compress the history into a shorter summary within a pre-defined token count. The compression step is included in RL training and trained end-to-end using outcome rewards, allowing the model to learn how and what to compress for the purpose of maximizing rewards.

### Task Solver Prompt (Figure 34)

```
You are a deep research agent. You need to answer the given question by
interacting with a search engine, using the search tool provided. Please
perform reasoning and use the tool step by step, in an interleaved manner.
You may use the search tool multiple times.

Question: {question}

Your response should be in the following format:
Explanation:  {your explanation for your final answer. For this explanation
section only, you should cite your evidence documents inline by enclosing
their docids in square brackets []. For example, [20].}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

---

## 4. Evaluation Methodology

### Nugget-Based Completion Scoring

All KARLBench tasks use **nugget-based evaluation**, consistent with the framework spearheaded by Voorhees (2003) and used in TREC-RAG and DeepScholar-Bench.

**FinanceBench is a special case:** BrowseComp-Plus and FinanceBench have only a **single nugget** per question that must be predicted correctly (answer nuggets/Q = 1.0 +/- 0.0). This makes the score effectively binary — the answer either matches or it doesn't.

### Judge Prompt: Nugget-Completeness (Figure 31)

The judge evaluates whether an answer sufficiently supports each decompositional fact:

```
Your Role: You will evaluate whether an answer to a question (which can
include a code snippet or documentation) sufficiently supports each
decompositional fact.

Process:
1. Read the question and the answer.
2. Read each of the {length} decompositional facts carefully one by one.
3. Based on the question and answer, judge whether the answer supports,
   partially supports, or does not support each decompositional fact.

Label Definitions:
- support: The answer fully captures and entails all necessary parts of
  the decompositional fact.
- partial_support: The answer partially captures the decompositional fact,
  but does not fully capture all necessary parts.
- not_support: The answer does not capture or does not provide information
  entailing the decompositional fact.

Output Format: Return the labels as a Python list of strings (List[str]),
in the same order as the decompositional facts.

Input:
Question:  {question}
Answer:  {answer}
Decompositional Facts:  {nugget}
Labels:
```

**Important:** The judge prompt includes the question context (not just answer vs. reference). The KARL paper (Figure 31) explicitly includes the question in the input.

### Score Threshold

A score >= 0.6 counts as "correct" for accuracy computation (this is the threshold used across KARLBench).

---

## 5. Results

### Main Results (Table 4)

FinanceBench is an **out-of-distribution** task — KARL is never trained on FinanceBench data. Training uses only BrowseComp-Plus and TREC-Biogen.

| Model | FinanceBench Score |
|---|---|
| **Base models** | |
| GLM 4.5 Air (base) | 72.7 |
| Qwen 3.5 397B A17B | 79.3 |
| Minimax m2.5 | 78.0 |
| GPT 5 | **86.7** |
| GPT 5.2 | 80.3 |
| Claude 4.5 Haiku | 73.7 |
| Claude 4.5 Sonnet | 79.3 |
| Claude 4.5 Opus | 80.7 |
| Claude 4.6 Sonnet | 81.3 |
| Claude 4.6 Opus | 83.0 |
| **Single-Task RL** | |
| KARL-TREC | 68.3 |
| KARL-BCP | 77.0 |
| **Multi-Task RL** | |
| KARL (single rollout) | 76.0 |
| KARL (par. N=3) | 80.8 |
| KARL (par. N=10) | 84.5 |
| KARL (par. N=20) | 84.2 |

### Key Observations

1. **OOD generalization:** KARL achieves 76.0 on FinanceBench despite never training on financial data — a +3.3 improvement over its base model (GLM 4.5 Air at 72.7).
2. **Parallel thinking scales well:** N=3 boosts to 80.8 (+4.8 from single), N=10 reaches 84.5 (+8.5 from single).
3. **Diminishing returns at high N:** N=20 (84.2) slightly below N=10 (84.5), suggesting saturation around N=10-15 for FinanceBench.
4. **Competitive with frontier models:** KARL (par. N=10) at 84.5 approaches Claude 4.6 Opus (83.0) and Claude 4.5 Opus (80.7), despite starting from a much cheaper base model.
5. **GPT 5 leads on FinanceBench:** GPT 5 scores 86.7, the highest single score on FinanceBench across all models.

---

## 6. Test-Time Compute: Parallel Thinking

### How It Works (Section 5.1, Figure 4)

1. Given prompt x, generate N independent rollouts y_1, ..., y_N in parallel
2. Extract the final answer from each rollout
3. Feed all N answers back to the model (the same model pi) to produce an aggregated final answer
4. The aggregator has access to tools — it can perform additional searches to resolve conflicts

The solver agent and aggregator agent are the **same model**.

### FinanceBench Parallel Thinking Results (Figure 13)

Scaling N from 5 to 20 for FinanceBench:

| N | GLM 4.5 Air (base) | KARL |
|---|---|---|
| 5 | ~77 | ~80 |
| 10 | ~80 | ~83 |
| 15 | ~82 | ~84 |
| 20 | ~82 | ~84 |

The gain from RL training (shaded region in Figure 13) is approximately **+1.9 points** at N=20 for FinanceBench, which is the smallest delta across all KARLBench tasks (BrowseComp-Plus: +4.7, TREC-Biogen: +5.9, FreshStack: +2.8, QAMPARI: +4.8, PMBench: +4.7).

### Aggregation Cost (Table 7)

For FinanceBench with N=10 parallel thinking:
- **LLM turns for aggregation:** 1.6 (very few extra steps needed)
- **Rollout token length:** 15,105 tokens

---

## 7. Cost and Latency

### FinanceBench-Specific Latency (Appendix B)

From the per-eval latency breakdowns:

| Model | FinanceBench Mean Latency (ms) | Between-Split Var (ms) | Within-Split Var (ms) |
|---|---|---|---|
| GLM 4.5 Air | 9,368 | 1,073 | 3,222 |
| KARL | 10,627 | 901 | 3,601 |
| Sonnet 4.6 | 7,406 | 222 | 1,885 |
| Opus 4.6 | 9,999 | 1,108 | 2,493 |
| GPT 5.2 | 13,081 | 4,388 | 10,895 |

FinanceBench is one of the **faster benchmarks** — mean latency ~7-13s, compared to BrowseComp-Plus at 44-372s.

### Latency Measurement Protocol

- 8 GPU H200 node with vLLM, tensor parallel 8 for GLM 4.5 Air and KARL
- Same vector search index across all models
- 5 prompts per benchmark, 30 trajectories per prompt at concurrency 1
- 3 trajectory warm-up discarded per prompt
- 3 different splits, average latency across splits
- Primary metric: wall-clock time to start the first answer token (time-to-first-actionable-token)

### Cost

- KARL achieves competitive scores at under $0.10 per query (the lowest cost of any model above 55 points on KARLBench overall)
- With parallel sampling (N=10), KARL matches Claude Opus 4.6 quality at roughly 33% lower cost per query

---

## 8. Training Details (Not FinanceBench-Specific, but Context)

FinanceBench is **never used for training** — it is purely an out-of-distribution evaluation task.

### What KARL Is Trained On

- **In-distribution:** BrowseComp-Plus + TREC-Biogen (multi-task RL)
- **Base model:** GLM 4.5 Air
- **Training algorithm:** OAPL (Optimal Advantage-based Policy Optimization with Lagged Inference) — iterative large-batch off-policy RL
- **Iterations:** 2 iterations of OAPL training (unless otherwise noted)
- **Rollouts per training example:** 8
- **Training data per iteration (Table 3):**
  - Iter 1: 1,218 BrowseComp-Plus + 6,270 TREC-Biogen prompts
  - Iter 2: 1,336 BrowseComp-Plus + 11,371 TREC-Biogen prompts

### Why FinanceBench Improves Without Training On It

The KARL paper demonstrates that multi-task RL develops **general search capabilities** rather than task-specific heuristics:
- RL training increases search diversity (+37% unique docs for BrowseComp-Plus, +8% for TREC-Biogen)
- RL training increases search efficiency (fewer wasteful searches after finding all evidence)
- These general improvements transfer to OOD tasks like FinanceBench
- The gain is complementary to test-time compute — both single-rollout and parallel thinking improve

---

## 9. Behavioral Notes Relevant to FinanceBench

### FinanceBench-Specific Challenges

- Documents are **long** (financial reports, 100+ pages) — requires the agent to navigate and find specific sections/tables
- Answers require **numerical reasoning** — extracting values and computing (e.g., percent change)
- Single nugget per question — evaluation is effectively binary (correct or not)
- Relatively few relevant chunks per question (1.2 +/- 0.5) — the challenge is finding the right page, not aggregating across many sources

### Early Stopping Under Complex Reasoning (Section 8.2.2)

The KARL paper notes a failure mode particularly relevant to FinanceBench: when answering correctly requires arithmetic reasoning over retrieved data, KARL sometimes stops early rather than performing the calculation. The model improves retrieval strategy but has not fully improved its capacity for post-retrieval numerical computation. This is identified as a natural next step — extending multi-task RL to include explicit arithmetic and tabular reasoning rewards.

---

## 10. Reproduction Checklist

To faithfully reproduce the KARL FinanceBench eval:

1. **Dataset:** 150 questions from FinanceBench (Islam et al., 2023), sourced from PatronusAI/financebench on HuggingFace
2. **Corpus:** 53,399 indexed document chunks, indexed at page level, avg 717.9 tokens per chunk
3. **Embedding:** Qwen3-0.6B embeddings
4. **Retrieval:** k=20 documents per search call
5. **Agent:** Single tool (vector search), task solver prompt from Figure 34
6. **Scoring:** Nugget-completeness judge (Figure 31) with question context included, single nugget per question, score >= 0.6 = correct
7. **Judge model:** The paper uses an LLM judge (likely gpt-4o-mini based on other task descriptions, though not explicitly specified for FinanceBench eval)
8. **Modes:** Single rollout + parallel thinking with N=3, 10, 20
9. **Aggregation:** Generative aggregator (same model), not voting (open-ended answers don't form discrete equivalence classes)
10. **Latency measurement:** vLLM on 8x H200, tensor parallel 8, concurrency 1, 3 splits of 30 trajectories each
