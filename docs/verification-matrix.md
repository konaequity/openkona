# OpenKona Verification Matrix

This suite defines the spec-first verification contract for implementing KARL-inspired OpenKona before product code exists. The tests are intended to fail until each paper-derived subsystem is present.

## Source Traceability

Primary source:
- `Desktop/konash/public/karl.pdf`

Supporting sources:
- `README.md` in this repo
- `Desktop/konash/KONASH.md`

## Coverage Areas

Top-level categories retained for the spec-first suite:
- Architecture
- Data synthesis
- Training
- Inference
- Evaluation

### KARLBench

The evaluation layer must model all six KARLBench capabilities and preserve the paper's benchmark split:
- BrowseComp-Plus: constraint-driven entity search
- TREC-Biogen: cross-document report synthesis
- FinanceBench: tabular numerical reasoning over long financial reports
- QAMPARI: exhaustive entity retrieval
- FreshStack: procedural technical reasoning
- PMBench: enterprise fact aggregation over internal notes

The registry must also encode:
- in-distribution training tasks: BrowseComp-Plus and TREC-Biogen
- held-out evaluation tasks: FinanceBench, QAMPARI, FreshStack, PMBench
- nugget-based evaluation across all benchmarks
- cost, latency, in-distribution, and out-of-distribution reporting

### Corpus And Retrieval

The retrieval layer must preserve benchmark-specific ingestion policies described in the paper:
- BrowseComp-Plus: first 512 tokens
- FinanceBench: page-level indexing
- FreshStack: provided semantic chunks up to 2048 tokens
- TREC-Biogen: short abstracts without extra segmentation
- QAMPARI: sentence-level chunks with exhaustive retrieval behavior
- PMBench: first 2048 tokens

Vector search is the primary and sole external grounding tool in the harness. The suite also expects:
- embedded or cached index loading
- batch search support
- throughput-oriented design for offline rollout generation
- benchmark-specific embedding defaults:
  BrowseComp-Plus uses Qwen3-8B embeddings for eval retrieval and Qwen3-0.6B embeddings for dedup search
  PMBench uses GTE-large
  TREC-Biogen, QAMPARI, and FinanceBench use Qwen3-0.6B
  FreshStack uses Qwen3-0.6B with `k = 10`
- retrieved chunk budget scaled inversely with average chunk length and capped at `k = 20`

### Agent Harness

The harness contract follows the paper's `aroll` layering:
- Dispatcher
- Exploration Strategy
- Environment
- Agent
- Lifecycle Plugins
- Rewards
- Tools

Behavioral requirements:
- identical harness interfaces across data collection, training, evaluation, and inference
- environment-owned interaction loop
- task-specific reward composition
- plugin-based context compression, step budgeting, and tool gating
- agent-owned LLM client and per-step generation
- strategy-driven spawning of one or more environment-agent pairs
- final aggregation rollouts for Parallel Thinking
- value-guided candidate selection as an agent replacement rather than an environment rewrite
- termination override hooks in lifecycle plugins

### Data Synthesis

The synthesis layer must cover both stages from the paper:
- Stage I: question-answer synthesis grounded in retrieved evidence with few-shot task-format examples
- Stage II: multiple solver rollouts, pass-rate filtering, and quality filtering

The suite verifies:
- agentic corpus exploration via vector search
- exact and near-duplicate filtering
- pass-rate frontier filtering that excludes all-correct and all-wrong groups
- quality checks for ambiguity and incorrect reference answers
- task-specific synthesis budgets and rollout counts for TREC-Biogen and BrowseComp-Plus
- two-stage deduplication:
  exact-match removal against evaluation data and within synthesized candidates
  near-duplicate detection using embedding retrieval plus a paraphrase judge
- synthesized outputs that include question, nuggetized answer, and supporting citations

### Training

The training layer must expose the OAPL-style offline RL pipeline:
- offline grouped rollout dataset
- KL-regularized off-policy objective
- separate `beta_value` and `beta_kl` controls
- optimal value estimation from grouped rollouts
- masking of non-model tokens during loss computation
- compression-aware segmentation
- rollout-level reward assignment to segments
- iterative training with rollout regeneration
- multi-task loss composition balanced by training tokens
- shared runtime behavior between offline collection, training evaluation, and inference serving
- separate reward functions registered for BrowseComp-Plus and TREC-Biogen
- value-model training from binary rollout rewards over policy-generated tokens

### Inference

The inference layer must cover both test-time-compute paths from the paper:
- single-rollout baseline
- Parallel Thinking
- optional Value-Guided Search

The suite verifies:
- N parallel rollout generation
- final-answer extraction per rollout
- generative aggregation that can use tools and synthesize answers beyond voting
- value-model-guided candidate selection over partial rollouts
- repeated parallel BFS-style search followed by aggregation
- support for Best-of-N and weighted-majority aggregation over value-scored rollouts

### Evaluation

The evaluation layer must encode the paper's nugget semantics:
- QAMPARI treats each entity as a separate nugget
- FreshStack and PMBench convert answers into fixed nuggets using task-specific prompts
- TREC-Biogen nuggetizes multiple references independently and then consolidates them
- BrowseComp-Plus and FinanceBench are single-nugget special cases

### Experiments And Defaults

The verification suite also tracks paper-level defaults and benchmark metadata:
- dataset statistics from KARLBench Table 2
- training prompt counts from Table 3
- TTC budgets from Table 4 and Section 5
- multi-iteration training defaults and limits
- prompt registries for synthesis, rollouts, quality filtering, dedup paraphrase judging, and evaluation nugget prompts
- aggregation modes mentioned in the paper, including Best-of-N and weighted-majority vote
- experiment surfaces for:
  RL beyond sharpening via max@k
  search-environment ablations
  compression-role transfer
  parallel-thinking rollout cost
  value-guided-search aggregation comparisons

## Expected Test Layout

The verification suite should keep explicit contracts for:
- architecture and package layout
- synthesis contracts
- training contracts
- inference contracts
- evaluation contracts
- benchmark metadata and corpus policies
- harness semantics and lifecycle plugins
- paper traceability in docs
