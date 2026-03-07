#!/usr/bin/env python3
"""Run Stage II of the KARL training pipeline: Off-Policy RL Data Pipeline.

From the KARL paper (Section 4.1, Stage II / Section 7.2):
  1. Rollout Synthesis: For each synthetic QA pair from Stage I, generate
     8 independent solver rollouts using GLM 4.5 Air with vector search (k=20).
  2. Pass-rate filtering: Remove QA pairs where the solver gets all correct
     (trivially easy) or all incorrect (too hard / broken). Range [0.1, 0.9].
  3. Quality filtering: gpt-4o-mini judges ambiguity + reference accuracy.

Set both API keys before running:
    export ZHIPU_API_KEY=your_key_here
    export OPENAI_API_KEY=your_key_here
"""

import json
import sys
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from konash.synthesis.pipeline import SynthesisPipeline
from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from konash.synthesis.rollouts import RolloutGenerator, RolloutGroup
from konash.synthesis.filters import PassRateFilter, QualityFilter
from konash.synthesis.dedup import DeduplicationAgent
from konash.synthesis.config import SynthesisTaskConfig, QualityFilterConfig
from konash.retrieval.vector_search import VectorSearchTool


# ── LLM Clients ──────────────────────────────────────────────────────

ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
ZHIPU_MODEL = "glm-4.5-air"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"


def make_zhipu_llm_fn(api_key, model=ZHIPU_MODEL):
    """GLM 4.5 Air via Zhipu AI — used for synthesis + solver rollouts."""
    import urllib.request

    def llm_fn(messages, **kwargs):
        url = f"{ZHIPU_API_BASE}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return {
            "role": "assistant",
            "content": result["choices"][0]["message"].get("content", ""),
        }

    return llm_fn


def make_openai_judge_fn(api_key, model=OPENAI_JUDGE_MODEL):
    """gpt-4o-mini via OpenAI — used for quality filter judge (matching paper)."""
    import urllib.request

    def judge_fn(messages, **kwargs):
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 256),
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return {
            "role": "assistant",
            "content": result["choices"][0]["message"].get("content", ""),
        }

    return judge_fn


# ── Corpus ───────────────────────────────────────────────────────────

CORPUS_DOCUMENTS = [
    {"text": "CRISPR-Cas9 gene editing has emerged as a transformative tool in molecular biology. By directing a guide RNA to a specific genomic locus, the Cas9 nuclease introduces a double-strand break that can be repaired via non-homologous end joining (NHEJ) or homology-directed repair (HDR). Recent clinical trials have demonstrated efficacy in treating sickle cell disease and beta-thalassemia through ex-vivo editing of hematopoietic stem cells.", "doc_id": "doc_001"},
    {"text": "Transformer-based language models have achieved state-of-the-art performance across natural language processing benchmarks. The self-attention mechanism enables the model to capture long-range dependencies between tokens. Scaling laws suggest that model performance improves predictably with increases in parameters, data, and compute, following a power-law relationship.", "doc_id": "doc_002"},
    {"text": "Reinforcement learning from human feedback (RLHF) has become a standard technique for aligning language model outputs with human preferences. The process involves training a reward model on pairwise human comparisons, then using proximal policy optimization (PPO) to fine-tune the language model against the learned reward signal. Alternatives such as direct preference optimization (DPO) bypass the explicit reward model.", "doc_id": "doc_003"},
    {"text": "mRNA vaccine technology, pioneered during the COVID-19 pandemic, encodes viral spike protein instructions in lipid nanoparticles. The immune system recognizes the translated protein and mounts both humoral and cellular responses. Ongoing research applies mRNA platforms to cancer immunotherapy, where personalized neoantigen vaccines are designed from individual tumor sequencing data.", "doc_id": "doc_004"},
    {"text": "Retrieval-augmented generation (RAG) combines dense retrieval with generative language models to ground outputs in external knowledge. The retriever encodes queries and documents into a shared embedding space and returns the top-k most similar chunks. The generator then conditions on these chunks to produce factually grounded responses, reducing hallucination rates.", "doc_id": "doc_005"},
    {"text": "AlphaFold2 leverages attention-based neural architectures to predict protein 3D structures from amino acid sequences with near-experimental accuracy. The model uses multiple sequence alignments and structural templates as inputs. Its predictions have accelerated drug discovery pipelines by providing rapid structural hypotheses for previously uncharacterized proteins.", "doc_id": "doc_006"},
    {"text": "Federated learning enables collaborative model training across distributed data silos without centralizing sensitive data. Each participating node trains a local model on its private dataset and shares only gradient updates or model weights with a central aggregator. Differential privacy mechanisms can be layered on top to provide formal privacy guarantees.", "doc_id": "doc_007"},
    {"text": "Quantum error correction codes such as the surface code protect logical qubits from decoherence and gate errors. By encoding a single logical qubit across many physical qubits arranged on a 2D lattice, syndrome measurements detect and correct errors without disturbing the encoded information. Achieving a break-even point where error correction extends qubit lifetime remains an active area of experimental research.", "doc_id": "doc_008"},
    {"text": "Single-cell RNA sequencing (scRNA-seq) enables transcriptomic profiling at individual cell resolution. Droplet-based platforms like 10x Genomics Chromium capture thousands of cells per run. Computational analysis pipelines perform quality control, normalization, dimensionality reduction (PCA, UMAP), and clustering to identify cell types and states within heterogeneous tissues.", "doc_id": "doc_009"},
    {"text": "Graph neural networks (GNNs) generalize deep learning to non-Euclidean domains by passing messages along edges of a graph. Applications include molecular property prediction, social network analysis, and recommendation systems. Over-smoothing, where node representations converge as layers deepen, remains a key challenge addressed by techniques such as residual connections and jumping knowledge.", "doc_id": "doc_010"},
]

EVAL_QUESTIONS = [
    "How does CRISPR-Cas9 introduce double-strand breaks in DNA?",
    "What are the scaling laws for transformer language models?",
    "How does retrieval-augmented generation reduce hallucinations?",
]

SEED_EXAMPLES = [
    SyntheticExample(question="What repair pathways are activated after Cas9 creates a double-strand break?", answer="Non-homologous end joining (NHEJ) and homology-directed repair (HDR).", citations=["doc_001"]),
    SyntheticExample(question="What technique bypasses the explicit reward model in RLHF?", answer="Direct preference optimization (DPO) bypasses the explicit reward model.", citations=["doc_003"]),
    SyntheticExample(question="What architecture does AlphaFold2 use for protein structure prediction?", answer="AlphaFold2 uses attention-based neural architectures with multiple sequence alignments and structural templates.", citations=["doc_006"]),
    SyntheticExample(question="How does federated learning protect data privacy during training?", answer="Each node trains locally on private data and shares only gradient updates or model weights; differential privacy can be added for formal guarantees.", citations=["doc_007"]),
]


# ── Vector search helpers ────────────────────────────────────────────

class StringSearchWrapper:
    """Wraps VectorSearchTool to return strings for QA synthesis."""
    def __init__(self, tool):
        self._tool = tool
    def search(self, query, top_k=10, **kwargs):
        results = self._tool.search(query, top_k=top_k, **kwargs)
        return [r["text"] if isinstance(r, dict) else str(r) for r in results]


class DictSearchWrapper:
    """Wraps VectorSearchTool to return dicts for rollout retrieval."""
    def __init__(self, tool):
        self._tool = tool
    def search(self, query, top_k=10, **kwargs):
        return self._tool.search(query, top_k=top_k, **kwargs)


def _trigram_embed(texts, dim=256):
    import numpy as np
    import hashlib
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        for i in range(len(normalized) - 2):
            h = int(hashlib.md5(normalized[i:i+3].encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def build_vector_search(documents):
    import numpy as np
    tool = VectorSearchTool(embed_fn=lambda texts: _trigram_embed(texts))
    tool.index(documents, embeddings=_trigram_embed([d["text"] for d in documents]))
    return tool


# ── Output helpers ───────────────────────────────────────────────────

def p(msg=""):
    print(msg, flush=True)


def section(title):
    p(f"\n{'=' * 70}")
    p(f"  {title}")
    p(f"{'=' * 70}")


def subsection(title):
    p(f"\n  {'─' * 60}")
    p(f"  {title}")
    p(f"  {'─' * 60}")


# ── Progress callback ───────────────────────────────────────────────

def make_step_callback(num_qa, num_rollouts):
    """Create a progress callback that logs rollout steps in real time."""
    def on_step(qa_idx, rollout_idx, step_idx, step_record):
        stype = step_record.get("type", "?")
        prefix = f"    [Q{qa_idx+1}/{num_qa} R{rollout_idx+1}/{num_rollouts} S{step_idx}]"
        if stype == "retrieval":
            n = step_record.get("num_results", 0)
            p(f"{prefix} retrieval: {n} docs")
        elif stype == "reasoning":
            thought = step_record.get("thought", "")[:80]
            sub = step_record.get("sub_retrieval")
            extra = ""
            if sub:
                extra = f" +search({sub.get('num_results', 0)} docs)"
            p(f"{prefix} reasoning: {thought}{extra}")
        elif stype == "answer":
            ans = step_record.get("answer", "")[:80]
            p(f"{prefix} ANSWER: {ans}")
    return on_step


# ── Main ─────────────────────────────────────────────────────────────

def main():
    zhipu_key = os.environ.get("ZHIPU_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not zhipu_key:
        p("ERROR: ZHIPU_API_KEY not set.")
        sys.exit(1)
    if not openai_key:
        p("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # ── Build LLM functions ──────────────────────────────────────────
    llm_fn = make_zhipu_llm_fn(zhipu_key)
    judge_fn = make_openai_judge_fn(openai_key)

    p(f"Solver LLM:    {ZHIPU_MODEL} (GLM 4.5 Air via Zhipu)")
    p(f"Quality judge: {OPENAI_JUDGE_MODEL} (via OpenAI)")

    # Connectivity checks
    p("Testing Zhipu API... ", )
    try:
        llm_fn([{"role": "user", "content": "Say ok"}], temperature=0.0)
        p("  Zhipu OK")
    except Exception as e:
        p(f"  FAILED: {e}")
        sys.exit(1)

    p("Testing OpenAI API... ")
    try:
        judge_fn([{"role": "user", "content": "Say ok"}])
        p("  OpenAI OK")
    except Exception as e:
        p(f"  FAILED: {e}")
        sys.exit(1)

    # ── Build vector search index ────────────────────────────────────
    raw_tool = build_vector_search(CORPUS_DOCUMENTS)
    doc_texts = [d["text"] for d in CORPUS_DOCUMENTS]

    # ══════════════════════════════════════════════════════════════════
    # STAGE I: QA Synthesis (recap from run_stage_one.py)
    # ══════════════════════════════════════════════════════════════════
    section("Stage I: QA Synthesis (GLM 4.5 Air)")

    synthesizer = QuestionAnswerSynthesizer(
        few_shot_examples=SEED_EXAMPLES,
        vector_search_tool=StringSearchWrapper(raw_tool),
        generation_count=8,
        max_steps=50,
        llm_fn=llm_fn,
    )

    t0 = time.time()
    qa_examples = synthesizer.synthesize(documents=doc_texts, num_examples=8, seed=42)
    stage1_time = time.time() - t0

    # Deduplicate against eval set
    dedup = DeduplicationAgent(evaluation_questions=EVAL_QUESTIONS)
    clean_qs = set(dedup.run(
        synthetic_questions=[ex.question for ex in qa_examples],
        evaluation_questions=EVAL_QUESTIONS,
    ))
    qa_examples = [ex for ex in qa_examples if ex.question in clean_qs]

    p(f"Stage I: {len(qa_examples)} QA pairs in {stage1_time:.1f}s")
    for i, ex in enumerate(qa_examples, 1):
        p(f"  [{i}] Q: {ex.question[:85]}")
        p(f"     A: {ex.answer[:85]}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE II: Off-Policy RL Data Pipeline
    # ══════════════════════════════════════════════════════════════════
    section("Stage II: Off-Policy RL Data Pipeline")

    # Paper parameters (Section 7.2.1):
    #   8 rollouts per QA, 50 max steps, k=20, pass-rate [0.1, 0.9]
    #   gpt-4o-mini as quality judge
    #
    # For this demo corpus (10 small docs), we use 8 rollouts but cap
    # max_steps at 8 since evidence is found quickly.
    NUM_ROLLOUTS = 8
    MAX_STEPS = 8
    TOP_K = 20

    p(f"  Config (KARL paper §7.2.1, adapted for demo corpus):")
    p(f"    Solver model:   {ZHIPU_MODEL}")
    p(f"    Rollouts/QA:    {NUM_ROLLOUTS}")
    p(f"    Max steps:      {MAX_STEPS}")
    p(f"    Top-k:          {TOP_K}")
    p(f"    Pass-rate:      [0.1, 0.9]")
    p(f"    Quality judge:  {OPENAI_JUDGE_MODEL}")

    # Build the pipeline with all components wired together
    rollout_gen = RolloutGenerator(
        max_steps=MAX_STEPS,
        top_k=TOP_K,
        search_tool=DictSearchWrapper(raw_tool),
        llm_fn=llm_fn,
        on_step=make_step_callback(len(qa_examples), NUM_ROLLOUTS),
    )

    pipeline = SynthesisPipeline(
        config=SynthesisTaskConfig(
            solver_rollout_count=NUM_ROLLOUTS,
            solver_max_steps=MAX_STEPS,
            solver_top_k=TOP_K,
            quality_filter=QualityFilterConfig(
                judge_model=OPENAI_JUDGE_MODEL,
                checks_ambiguity=True,
                checks_reference_accuracy=True,
            ),
        ),
        rollout_generator=rollout_gen,
        pass_rate_filter=PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9),
        quality_filter=QualityFilter(
            judge_fn=judge_fn,
            judge_model=OPENAI_JUDGE_MODEL,
        ),
        evaluation_questions=EVAL_QUESTIONS,
        judge_fn=judge_fn,
    )

    # ── Phase 1: Solver Rollout Generation ───────────────────────────
    subsection(f"Phase 1: Solver Rollouts ({NUM_ROLLOUTS} per QA, max {MAX_STEPS} steps)")

    t_rollout_start = time.time()
    rollout_groups = []
    for i, ex in enumerate(qa_examples):
        p(f"\n  QA [{i+1}/{len(qa_examples)}]: {ex.question[:65]}...")
        p(f"  Reference: {ex.answer[:65]}...")
        t0 = time.time()

        group = rollout_gen.generate_group(
            prompt=ex.question,
            reference_answer=ex.answer,
            num_rollouts=NUM_ROLLOUTS,
            seed=42 + i,
            qa_idx=i,
        )
        elapsed = time.time() - t0
        rollout_groups.append(group)

        passes = sum(1 for r in group.rollouts if r.passed)
        fails = sum(1 for r in group.rollouts if r.passed is False)
        p(f"  --> pass_rate={group.pass_rate:.0%} "
          f"({passes}P/{fails}F) in {elapsed:.1f}s")

    total_rollout_time = time.time() - t_rollout_start
    total_rollouts = sum(g.size for g in rollout_groups)
    p(f"\n  Phase 1 complete: {total_rollouts} rollouts in {total_rollout_time:.1f}s")

    # Store groups on the pipeline for downstream access
    pipeline.synthetic_examples = qa_examples
    pipeline.rollout_groups = rollout_groups

    # ── Phase 2: Pass-Rate Filtering ─────────────────────────────────
    subsection("Phase 2: Pass-Rate Filtering [0.1, 0.9]")

    for i, group in enumerate(rollout_groups, 1):
        keep = 0.1 <= group.pass_rate <= 0.9
        reason = ""
        if group.pass_rate < 0.1:
            reason = " (all fail — too hard/broken)"
        elif group.pass_rate > 0.9:
            reason = " (all pass — trivially easy)"
        p(f"  [{i}] rate={group.pass_rate:.2f} → {'KEEP' if keep else 'DROP'}{reason}")

    filtered_groups = pipeline.estimate_pass_rate(rollout_groups)
    dropped = len(rollout_groups) - len(filtered_groups)
    pipeline.filtered_groups = filtered_groups
    p(f"\n  Phase 2 result: {len(filtered_groups)} kept, {dropped} dropped")

    # ── Phase 3: Quality Filtering (gpt-4o-mini) ────────────────────
    subsection(f"Phase 3: Quality Filtering ({OPENAI_JUDGE_MODEL})")

    surviving_prompts = {g.prompt for g in filtered_groups}
    surviving_examples = [ex for ex in qa_examples if ex.question in surviving_prompts]

    p(f"  Evaluating {len(surviving_examples)} QA pairs with {OPENAI_JUDGE_MODEL}...")
    final_examples = []
    for ex in surviving_examples:
        p(f"\n  Checking: {ex.question[:60]}...")

        # Ambiguity check
        amb = pipeline.quality_filter.judge_ambiguity(ex.question, ex.answer)
        if amb.get("is_ambiguous"):
            p(f"    DROPPED (ambiguous): {amb.get('reason', '')}")
            continue
        p(f"    Ambiguity:  clear")

        # Reference accuracy check
        acc = pipeline.quality_filter.judge_reference_accuracy(
            ex.question, ex.answer, doc_texts
        )
        if not acc.get("is_accurate", True):
            p(f"    DROPPED (inaccurate): {acc.get('reason', '')}")
            continue
        p(f"    Accuracy:   grounded")
        p(f"    --> PASSED")
        final_examples.append(ex)

    quality_dropped = len(surviving_examples) - len(final_examples)
    pipeline.final_examples = final_examples
    p(f"\n  Phase 3 result: {len(final_examples)} passed, {quality_dropped} dropped")

    # ── Final RL Training Dataset ────────────────────────────────────
    section("Stage II Output: RL Training Data")

    final_prompts = {ex.question for ex in final_examples}
    final_groups = [g for g in rollout_groups if g.prompt in final_prompts]

    training_data = []
    for ex, group in zip(final_examples, final_groups):
        for rollout in group.rollouts:
            training_data.append({
                "prompt": ex.question,
                "reference_answer": ex.answer,
                "rollout_steps": [
                    {k: v for k, v in step.items()
                     if k not in ("results",)}  # exclude raw docs for brevity
                    for step in rollout.steps
                ],
                "final_answer": rollout.final_answer,
                "reward": 1.0 if rollout.passed else 0.0,
                "num_steps": len(rollout.steps),
            })

    positive = sum(1 for d in training_data if d["reward"] == 1.0)
    negative = sum(1 for d in training_data if d["reward"] == 0.0)
    step_counts = [d["num_steps"] for d in training_data]

    p(f"\n  Training QA pairs:   {len(final_examples)}")
    p(f"  Total rollouts:      {len(training_data)}")
    p(f"  Positive (reward=1): {positive}")
    p(f"  Negative (reward=0): {negative}")
    if step_counts:
        p(f"  Trajectory lengths:  avg={sum(step_counts)/len(step_counts):.1f}, "
          f"min={min(step_counts)}, max={max(step_counts)}")

    if final_examples:
        p(f"\n  Final training examples:")
        for i, ex in enumerate(final_examples, 1):
            group = final_groups[i - 1]
            p(f"    [{i}] Q: {ex.question}")
            p(f"        A: {ex.answer}")
            p(f"        Pass rate: {group.pass_rate:.0%}")
            for j, rollout in enumerate(group.rollouts, 1):
                status = "PASS" if rollout.passed else "FAIL"
                p(f"        Rollout {j}: {status} | {rollout.num_steps} steps | "
                  f"{(rollout.final_answer or '')[:60]}")
            p()

    # ── Save training data to JSON ───────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "stage2_training_data.json")
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2, default=str)
    p(f"  Training data saved to: {output_path}")

    # ── Pipeline Summary ─────────────────────────────────────────────
    section("Full Pipeline Summary")
    p(f"  Stage I  (GLM 4.5 Air)            → {len(qa_examples)} QA pairs ({stage1_time:.1f}s)")
    p(f"  Phase 1  ({NUM_ROLLOUTS} rollouts/QA)        → {total_rollouts} rollouts ({total_rollout_time:.1f}s)")
    p(f"  Phase 2  (pass-rate [0.1, 0.9])   → {len(filtered_groups)}/{len(rollout_groups)} QA pairs kept")
    p(f"  Phase 3  ({OPENAI_JUDGE_MODEL} judge)     → {len(final_examples)}/{len(surviving_examples)} QA pairs kept")
    p(f"  RL training rows:                 {len(training_data)} (prompt, rollout, reward)")
    p()


if __name__ == "__main__":
    main()
