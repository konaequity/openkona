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

# Force unbuffered output so background runs stream properly
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from konash.synthesis.pipeline import SynthesisPipeline
from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from konash.synthesis.rollouts import RolloutGenerator, RolloutGroup, Rollout
from konash.synthesis.dedup import DeduplicationAgent
from konash.synthesis.filters import PassRateFilter, QualityFilter, GroundingFilter
from konash.synthesis.config import SynthesisTaskConfig, QualityFilterConfig
from konash.retrieval.vector_search import VectorSearchTool, RetrievalBudgetPolicy


# ── LLM Clients ──────────────────────────────────────────────────────

ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
ZHIPU_MODEL = "glm-4.5-air"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"


def make_zhipu_llm_fn(api_key, model=ZHIPU_MODEL):
    """GLM 4.5 Air via Zhipu AI — used for synthesis + solver rollouts."""
    import urllib.request

    def llm_fn(messages, **kwargs):
        url = f"{ZHIPU_API_BASE}/chat/completions"
        body = {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.7)}
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return {"role": "assistant", "content": result["choices"][0]["message"].get("content", "")}

    return llm_fn


def make_openai_judge_fn(api_key, model=OPENAI_JUDGE_MODEL):
    """gpt-4o-mini via OpenAI — used for quality filter judge (matching paper)."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    def judge_fn(messages, **kwargs):
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return {"role": "assistant", "content": r.choices[0].message.content or ""}

    return judge_fn


# ── LLM-backed Quality Filter ───────────────────────────────────────

class LLMQualityFilter(QualityFilter):
    """Quality filter that uses gpt-4o-mini as the judge, matching the paper.

    Paper Section 7.2.1: "we apply gpt-4o-mini as the judge"
    Paper Section 7.2.2: "we use gpt-4o-mini as the judge"
    """

    def __init__(self, judge_fn, **kwargs):
        super().__init__(**kwargs)
        self._judge_fn = judge_fn

    def judge_ambiguity(self, question, answer):
        prompt = (
            "You are evaluating a synthetic question-answer pair for ambiguity.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Is this question ambiguous? An ambiguous question has multiple valid "
            "interpretations that would lead to different answers.\n\n"
            "Respond with JSON: {\"is_ambiguous\": true/false, \"reason\": \"...\"}"
        )
        try:
            resp = self._judge_fn([{"role": "user", "content": prompt}])
            text = resp.get("content", "")
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "is_ambiguous": parsed.get("is_ambiguous", False),
                    "confidence": 0.9,
                    "reason": parsed.get("reason", ""),
                }
        except Exception:
            pass
        # Fallback to heuristic
        return super().judge_ambiguity(question, answer)

    def judge_reference_accuracy(self, question, answer, reference_documents):
        docs_text = "\n\n".join(f"[Doc {i+1}]: {d[:500]}" for i, d in enumerate(reference_documents[:5]))
        prompt = (
            "You are evaluating whether a synthetic answer is factually accurate "
            "based on the reference documents.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Reference Documents:\n{docs_text}\n\n"
            "Is the answer factually accurate and grounded in the documents?\n\n"
            "Respond with JSON: {\"is_accurate\": true/false, \"reason\": \"...\"}"
        )
        try:
            resp = self._judge_fn([{"role": "user", "content": prompt}])
            text = resp.get("content", "")
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "is_accurate": parsed.get("is_accurate", True),
                    "confidence": 0.9,
                    "grounding_score": 1.0 if parsed.get("is_accurate") else 0.0,
                    "reason": parsed.get("reason", ""),
                }
        except Exception:
            pass
        return super().judge_reference_accuracy(question, answer, reference_documents)


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
    def __init__(self, tool):
        self._tool = tool
    def search(self, query, top_k=10, **kwargs):
        results = self._tool.search(query, top_k=top_k, **kwargs)
        return [r["text"] if isinstance(r, dict) else str(r) for r in results]


class DictSearchWrapper:
    def __init__(self, tool):
        self._tool = tool
    def search(self, query, top_k=10, **kwargs):
        return self._tool.search(query, top_k=top_k, **kwargs)


def _trigram_embed(texts, dim=256):
    import numpy as np, hashlib
    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        for i in range(len(normalized) - 2):
            h = int(hashlib.md5(normalized[i:i+3].encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def build_vector_search(documents):
    import numpy as np
    tool = VectorSearchTool(embed_fn=lambda texts: _trigram_embed(texts))
    tool.index(documents, embeddings=_trigram_embed([d["text"] for d in documents]))
    return tool


def p(msg=""):
    """Print with immediate flush."""
    print(msg, flush=True)


def section(title):
    p(f"\n{'=' * 70}")
    p(f"  {title}")
    p(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    zhipu_key = os.environ.get("ZHIPU_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not zhipu_key:
        p("ERROR: ZHIPU_API_KEY not set."); sys.exit(1)
    if not openai_key:
        p("ERROR: OPENAI_API_KEY not set."); sys.exit(1)

    llm_fn = make_zhipu_llm_fn(zhipu_key)
    judge_fn = make_openai_judge_fn(openai_key)

    p(f"Solver LLM:   {ZHIPU_MODEL} (GLM 4.5 Air via Zhipu)")
    p(f"Quality judge: {OPENAI_JUDGE_MODEL} (via OpenAI)")

    p("Testing Zhipu API... ", )
    try:
        llm_fn([{"role": "user", "content": "Say ok"}], temperature=0.0)
        p("  Zhipu OK")
    except Exception as e:
        p(f"  FAILED: {e}"); sys.exit(1)

    p("Testing OpenAI API... ")
    try:
        judge_fn([{"role": "user", "content": "Say ok"}])
        p("  OpenAI OK")
    except Exception as e:
        p(f"  FAILED: {e}"); sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # STAGE I (recap): Generate QA pairs with GLM 4.5 Air
    # ═══════════════════════════════════════════════════════════════════
    section("Stage I: QA Synthesis (GLM 4.5 Air)")

    raw_tool = build_vector_search(CORPUS_DOCUMENTS)
    doc_texts = [d["text"] for d in CORPUS_DOCUMENTS]

    synthesizer = QuestionAnswerSynthesizer(
        few_shot_examples=SEED_EXAMPLES,
        vector_search_tool=StringSearchWrapper(raw_tool),
        generation_count=8, max_steps=50, llm_fn=llm_fn,
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
    p()

    # ═══════════════════════════════════════════════════════════════════
    # STAGE II: Off-Policy RL Data Pipeline
    # ═══════════════════════════════════════════════════════════════════
    section("Stage II: Off-Policy RL Data Pipeline")
    p(f"  Solver: GLM 4.5 Air, 4 rollouts/QA, max 5 steps, k=20")
    p(f"  Pass-rate filter: [0.1, 0.9]")
    p(f"  Quality judge: gpt-4o-mini (ambiguity + reference accuracy)")

    # ── Phase 1: Solver Rollout Generation ───────────────────────────
    section("Phase 1: Solver Rollout Generation (GLM 4.5 Air)")

    # Reduced from paper defaults (50 steps, 8 rollouts) for practical demo runtime.
    # Full-scale runs would use max_steps=50, num_rollouts=8.
    rollout_gen = RolloutGenerator(
        max_steps=5, top_k=20,
        search_tool=DictSearchWrapper(raw_tool),
        llm_fn=llm_fn,
    )

    NUM_ROLLOUTS = 4
    rollout_groups = []
    total_rollout_time = 0.0

    for i, ex in enumerate(qa_examples, 1):
        p(f"\n  QA [{i}/{len(qa_examples)}]: {ex.question[:65]}...")
        p(f"  Reference: {ex.answer[:65]}...")
        t0 = time.time()

        group = rollout_gen.generate_group(
            prompt=ex.question,
            reference_answer=ex.answer,
            num_rollouts=NUM_ROLLOUTS,
            seed=42 + i,
        )
        elapsed = time.time() - t0
        total_rollout_time += elapsed
        rollout_groups.append(group)

        passes = sum(1 for r in group.rollouts if r.passed)
        fails = sum(1 for r in group.rollouts if r.passed is False)
        p(f"  Result: pass_rate={group.pass_rate:.0%} "
          f"({passes}P/{fails}F) — {elapsed:.1f}s")

        for j, rollout in enumerate(group.rollouts, 1):
            status = "PASS" if rollout.passed else ("FAIL" if rollout.passed is False else "N/A")
            n_steps = len(rollout.steps)
            ans = (rollout.final_answer or "")[:80]
            p(f"    [{j}] {status} | {n_steps} steps | {ans}")

    p(f"\n  Total: {sum(g.size for g in rollout_groups)} rollouts in {total_rollout_time:.1f}s")

    # ── Phase 2: Pass-Rate Filtering ─────────────────────────────────
    section("Phase 2: Pass-Rate Filtering [0.1, 0.9]")

    pass_filter = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)

    for i, group in enumerate(rollout_groups, 1):
        keep = 0.1 <= group.pass_rate <= 0.9
        reason = ""
        if group.pass_rate < 0.1: reason = " (all fail — too hard)"
        elif group.pass_rate > 0.9: reason = " (all pass — trivial)"
        p(f"  [{i}] rate={group.pass_rate:.2f} → {'KEEP' if keep else 'DROP'}{reason}")

    filtered_groups = pass_filter.apply(rollout_groups)
    dropped = len(rollout_groups) - len(filtered_groups)
    p(f"\n  Result: {len(filtered_groups)} kept, {dropped} dropped")

    # ── Phase 3: Quality Filtering (gpt-4o-mini) ────────────────────
    section("Phase 3: Quality Filtering (gpt-4o-mini)")

    surviving_prompts = {g.prompt for g in filtered_groups}
    surviving_examples = [ex for ex in qa_examples if ex.question in surviving_prompts]

    quality_filter = LLMQualityFilter(
        judge_fn=judge_fn,
        checks_ambiguity=True,
        checks_reference_accuracy=True,
    )

    p(f"  Evaluating {len(surviving_examples)} QA pairs with gpt-4o-mini...")
    final_examples = []
    for ex in surviving_examples:
        amb = quality_filter.judge_ambiguity(ex.question, ex.answer)
        if amb.get("is_ambiguous"):
            p(f"  DROPPED (ambiguous): {ex.question[:60]}...")
            p(f"    Reason: {amb.get('reason', '')}")
            continue

        acc = quality_filter.judge_reference_accuracy(ex.question, ex.answer, doc_texts)
        if not acc.get("is_accurate", True):
            p(f"  DROPPED (inaccurate): {ex.question[:60]}...")
            p(f"    Reason: {acc.get('reason', '')}")
            continue

        p(f"  PASSED: {ex.question[:60]}...")
        final_examples.append(ex)

    quality_dropped = len(surviving_examples) - len(final_examples)
    p(f"\n  Result: {len(final_examples)} passed, {quality_dropped} dropped by quality filter")

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
                "rollout_steps": rollout.steps,
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

    p(f"\n  Final training examples:")
    for i, ex in enumerate(final_examples, 1):
        group = final_groups[i - 1]
        p(f"    [{i}] Q: {ex.question}")
        p(f"        A: {ex.answer}")
        p(f"        Pass rate: {group.pass_rate:.0%}")
        p()

    # ── Pipeline Summary ─────────────────────────────────────────────
    section("Full Pipeline Summary")
    p(f"  Stage I  (GLM 4.5 Air)  → {len(qa_examples)} QA pairs")
    p(f"  Phase 1  (4 rollouts)   → {sum(g.size for g in rollout_groups)} rollouts")
    p(f"  Phase 2  (pass-rate)    → {len(filtered_groups)}/{len(rollout_groups)} QA pairs kept")
    p(f"  Phase 3  (gpt-4o-mini)  → {len(final_examples)}/{len(surviving_examples)} QA pairs kept")
    p(f"  RL training rows:       {len(training_data)} (prompt, rollout, reward)")
    p(f"  Total time: Stage I={stage1_time:.1f}s, Rollouts={total_rollout_time:.1f}s")
    p()


if __name__ == "__main__":
    main()
