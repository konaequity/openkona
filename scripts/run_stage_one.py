#!/usr/bin/env python3
"""Run Stage I of the QA Synthesis Pipeline on a sample corpus.

Matches the KARL paper setup:
  - Synthesis LLM: GLM 4.5 Air via Zhipu AI OpenAI-compatible API
  - Vector search: pseudo-embeddings (trigram, dim=256)
  - Dedup: exact + near-duplicate removal (within synthetic + against eval set)
  - Few-shot seed examples (4, matching paper Section 7.2.1)
  - QA generation count: 8 per prompt (matching paper)
  - Max steps: 50 (matching paper)

Set ZHIPU_API_KEY before running:
    export ZHIPU_API_KEY=your_key_here
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from konash.synthesis.pipeline import SynthesisPipeline
from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from konash.synthesis.dedup import DeduplicationAgent, EmbeddingDeduplicator
from konash.synthesis.config import SynthesisTaskConfig, QualityFilterConfig
from konash.retrieval.vector_search import VectorSearchTool, RetrievalBudgetPolicy


# ── Zhipu AI (GLM 4.5 Air) LLM Client ───────────────────────────────

ZHIPU_API_BASE = "https://api.z.ai/api/paas/v4"
ZHIPU_MODEL = "glm-4.5-air"  # GLM 4.5 Air model ID on Zhipu platform


def make_zhipu_llm_fn(api_key: str, model: str = ZHIPU_MODEL):
    """Create an llm_fn callable backed by Zhipu AI's OpenAI-compatible API.

    Returns a function: (messages: list[dict]) -> dict with 'content' key.
    """
    import urllib.request

    def llm_fn(messages, **kwargs):
        url = f"{ZHIPU_API_BASE}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        choice = result["choices"][0]
        message = choice["message"]
        return {
            "role": "assistant",
            "content": message.get("content", ""),
        }

    return llm_fn


# ── 1. Sample corpus ─────────────────────────────────────────────────
# Simulating a small biomedical / technical corpus (loosely TREC-Biogen style)
CORPUS_DOCUMENTS = [
    {
        "text": (
            "CRISPR-Cas9 gene editing has emerged as a transformative tool in molecular "
            "biology. By directing a guide RNA to a specific genomic locus, the Cas9 "
            "nuclease introduces a double-strand break that can be repaired via non-homologous "
            "end joining (NHEJ) or homology-directed repair (HDR). Recent clinical trials "
            "have demonstrated efficacy in treating sickle cell disease and beta-thalassemia "
            "through ex-vivo editing of hematopoietic stem cells."
        ),
        "doc_id": "doc_001",
    },
    {
        "text": (
            "Transformer-based language models have achieved state-of-the-art performance "
            "across natural language processing benchmarks. The self-attention mechanism "
            "enables the model to capture long-range dependencies between tokens. Scaling "
            "laws suggest that model performance improves predictably with increases in "
            "parameters, data, and compute, following a power-law relationship."
        ),
        "doc_id": "doc_002",
    },
    {
        "text": (
            "Reinforcement learning from human feedback (RLHF) has become a standard "
            "technique for aligning language model outputs with human preferences. The "
            "process involves training a reward model on pairwise human comparisons, then "
            "using proximal policy optimization (PPO) to fine-tune the language model "
            "against the learned reward signal. Alternatives such as direct preference "
            "optimization (DPO) bypass the explicit reward model."
        ),
        "doc_id": "doc_003",
    },
    {
        "text": (
            "mRNA vaccine technology, pioneered during the COVID-19 pandemic, encodes "
            "viral spike protein instructions in lipid nanoparticles. The immune system "
            "recognizes the translated protein and mounts both humoral and cellular "
            "responses. Ongoing research applies mRNA platforms to cancer immunotherapy, "
            "where personalized neoantigen vaccines are designed from individual tumor "
            "sequencing data."
        ),
        "doc_id": "doc_004",
    },
    {
        "text": (
            "Retrieval-augmented generation (RAG) combines dense retrieval with generative "
            "language models to ground outputs in external knowledge. The retriever encodes "
            "queries and documents into a shared embedding space and returns the top-k most "
            "similar chunks. The generator then conditions on these chunks to produce "
            "factually grounded responses, reducing hallucination rates."
        ),
        "doc_id": "doc_005",
    },
    {
        "text": (
            "AlphaFold2 leverages attention-based neural architectures to predict protein "
            "3D structures from amino acid sequences with near-experimental accuracy. The "
            "model uses multiple sequence alignments and structural templates as inputs. "
            "Its predictions have accelerated drug discovery pipelines by providing rapid "
            "structural hypotheses for previously uncharacterized proteins."
        ),
        "doc_id": "doc_006",
    },
    {
        "text": (
            "Federated learning enables collaborative model training across distributed "
            "data silos without centralizing sensitive data. Each participating node trains "
            "a local model on its private dataset and shares only gradient updates or model "
            "weights with a central aggregator. Differential privacy mechanisms can be "
            "layered on top to provide formal privacy guarantees."
        ),
        "doc_id": "doc_007",
    },
    {
        "text": (
            "Quantum error correction codes such as the surface code protect logical qubits "
            "from decoherence and gate errors. By encoding a single logical qubit across "
            "many physical qubits arranged on a 2D lattice, syndrome measurements detect "
            "and correct errors without disturbing the encoded information. Achieving a "
            "break-even point where error correction extends qubit lifetime remains an "
            "active area of experimental research."
        ),
        "doc_id": "doc_008",
    },
    {
        "text": (
            "Single-cell RNA sequencing (scRNA-seq) enables transcriptomic profiling at "
            "individual cell resolution. Droplet-based platforms like 10x Genomics Chromium "
            "capture thousands of cells per run. Computational analysis pipelines perform "
            "quality control, normalization, dimensionality reduction (PCA, UMAP), and "
            "clustering to identify cell types and states within heterogeneous tissues."
        ),
        "doc_id": "doc_009",
    },
    {
        "text": (
            "Graph neural networks (GNNs) generalize deep learning to non-Euclidean "
            "domains by passing messages along edges of a graph. Applications include "
            "molecular property prediction, social network analysis, and recommendation "
            "systems. Over-smoothing, where node representations converge as layers deepen, "
            "remains a key challenge addressed by techniques such as residual connections "
            "and jumping knowledge."
        ),
        "doc_id": "doc_010",
    },
]

# ── 2. Held-out evaluation questions (for contamination checking) ────
EVAL_QUESTIONS = [
    "How does CRISPR-Cas9 introduce double-strand breaks in DNA?",
    "What are the scaling laws for transformer language models?",
    "How does retrieval-augmented generation reduce hallucinations?",
]

# ── 3. Few-shot seed examples (4 seeds, matching paper Section 7.2.1) ─
SEED_EXAMPLES = [
    SyntheticExample(
        question="What repair pathways are activated after Cas9 creates a double-strand break?",
        answer="Non-homologous end joining (NHEJ) and homology-directed repair (HDR).",
        citations=["doc_001"],
    ),
    SyntheticExample(
        question="What technique bypasses the explicit reward model in RLHF?",
        answer="Direct preference optimization (DPO) bypasses the explicit reward model.",
        citations=["doc_003"],
    ),
    SyntheticExample(
        question="What architecture does AlphaFold2 use for protein structure prediction?",
        answer="AlphaFold2 uses attention-based neural architectures with multiple sequence alignments and structural templates.",
        citations=["doc_006"],
    ),
    SyntheticExample(
        question="How does federated learning protect data privacy during training?",
        answer="Each node trains locally on private data and shares only gradient updates or model weights; differential privacy can be added for formal guarantees.",
        citations=["doc_007"],
    ),
]


# ── Vector search helpers ────────────────────────────────────────────

class StringSearchWrapper:
    """Wraps VectorSearchTool to return plain strings (text field) instead of dicts.

    The QuestionAnswerSynthesizer.explore_corpus() expects search results to be
    strings, but VectorSearchTool.search() returns dicts with 'text', 'score', etc.
    """

    def __init__(self, tool):
        self._tool = tool

    def search(self, query, top_k=10, **kwargs):
        results = self._tool.search(query, top_k=top_k, **kwargs)
        return [r["text"] if isinstance(r, dict) else str(r) for r in results]


def _trigram_embed(texts, dim=256):
    """Character-trigram pseudo-embeddings (same approach as EmbeddingDeduplicator)."""
    import numpy as np
    import hashlib

    vectors = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        for i in range(len(normalized) - 2):
            trigram = normalized[i : i + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def build_vector_search(documents):
    """Build a VectorSearchTool indexed on the corpus using pseudo-embeddings."""
    import numpy as np

    tool = VectorSearchTool(embed_fn=lambda texts: _trigram_embed(texts))
    embeddings = _trigram_embed([d["text"] for d in documents])
    tool.index(documents, embeddings=embeddings)
    return tool, StringSearchWrapper(tool)


# ── Display helpers ──────────────────────────────────────────────────

def print_section(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # ── Resolve API key ──────────────────────────────────────────────
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print("ERROR: ZHIPU_API_KEY not set.")
        print("  export ZHIPU_API_KEY=your_key_here")
        print("  Get one at: https://open.bigmodel.cn")
        sys.exit(1)

    llm_fn = make_zhipu_llm_fn(api_key)
    print(f"LLM backend: Zhipu AI  model={ZHIPU_MODEL}")

    # ── Quick connectivity check ─────────────────────────────────────
    print("Testing API connectivity... ", end="", flush=True)
    try:
        test_resp = llm_fn([{"role": "user", "content": "Say 'ok' and nothing else."}], temperature=0.0)
        print(f"OK (response: {test_resp['content'][:30]!r})")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print_section("Stage I: Question-Answer Synthesis Pipeline (GLM 4.5 Air)")
    print(f"\nCorpus size: {len(CORPUS_DOCUMENTS)} documents")
    print(f"Seed examples: {len(SEED_EXAMPLES)} (matching paper: 4)")
    print(f"Eval questions (contamination check): {len(EVAL_QUESTIONS)}")
    print(f"Synthesis model: {ZHIPU_MODEL} (paper: GLM 4.5 Air)")

    # ── Build vector search tool ─────────────────────────────────────
    print_section("Building Vector Search Index")
    t0 = time.time()
    raw_tool, search_tool = build_vector_search(CORPUS_DOCUMENTS)
    print(f"Indexed {len(CORPUS_DOCUMENTS)} documents in {time.time() - t0:.3f}s")

    # Quick sanity check
    test_results = raw_tool.search("CRISPR gene editing", top_k=3)
    print(f"Sanity check — top-3 for 'CRISPR gene editing':")
    for r in test_results:
        print(f"  [{r['doc_id']}] score={r['score']:.4f}  {r['text'][:70]}...")

    # ── Retrieval budget policy (paper: k scales inversely w/ chunk length, max 20) ─
    policy = RetrievalBudgetPolicy(target_token_budget=2000, max_top_k=20)
    avg_chunk_len = sum(len(d["text"].split()) for d in CORPUS_DOCUMENTS) / len(CORPUS_DOCUMENTS)
    computed_k = policy.compute_top_k(avg_chunk_length=avg_chunk_len)
    print(f"\nRetrieval budget: target=2000 tokens, avg_chunk={avg_chunk_len:.0f}, top_k={computed_k}")

    # ── Configure synthesis (matching paper Section 7.2.1) ───────────
    config = SynthesisTaskConfig(
        task_name="DemoCorpus_GLM",
        seed_examples=len(SEED_EXAMPLES),
        qa_max_steps=50,             # paper: "up to fifty steps"
        qa_generation_count=8,       # paper: "generates eight candidate synthetic QA pairs"
        solver_rollout_count=8,      # paper: "eight rollouts"
        solver_max_steps=50,         # paper: "fifty steps"
        solver_top_k=computed_k,
        quality_filter=QualityFilterConfig(judge_model=None),
    )

    synthesizer = QuestionAnswerSynthesizer(
        few_shot_examples=SEED_EXAMPLES,
        vector_search_tool=search_tool,
        generation_count=config.qa_generation_count,
        max_steps=config.qa_max_steps,
        llm_fn=llm_fn,  # <── GLM 4.5 Air wired in
    )

    dedup_agent = DeduplicationAgent(
        evaluation_questions=EVAL_QUESTIONS,
    )

    pipeline = SynthesisPipeline(
        config=config,
        synthesizer=synthesizer,
        deduplication_agent=dedup_agent,
        evaluation_questions=EVAL_QUESTIONS,
    )

    # ── Phase 1: Corpus Exploration via Vector Search ────────────────
    print_section("Phase 1: Corpus Exploration via Vector Search")
    t0 = time.time()
    explored_docs = synthesizer.explore_corpus(num_documents=6)
    explore_time = time.time() - t0
    print(f"Explored {len(explored_docs)} documents in {explore_time:.3f}s")
    print(f"Exploration queries derived from {len(SEED_EXAMPLES)} seed examples")
    for i, doc in enumerate(explored_docs):
        preview = str(doc)[:90]
        print(f"  [{i+1}] {preview}...")

    # ── Phase 2: QA Generation (LLM-backed) ──────────────────────────
    print_section("Phase 2: QA Pair Generation (GLM 4.5 Air)")
    t0 = time.time()
    doc_texts = [d["text"] for d in CORPUS_DOCUMENTS]
    raw_examples = synthesizer.synthesize(
        documents=doc_texts,
        num_examples=config.qa_generation_count,
        seed=42,
    )
    gen_time = time.time() - t0
    print(f"Generated {len(raw_examples)} QA pairs in {gen_time:.1f}s")
    print()
    for i, ex in enumerate(raw_examples, 1):
        q_text = (ex.question or "")
        a_text = (ex.answer or "")
        cites = ex.citations or []
        print(f"  [{i}] Q: {q_text}")
        print(f"     A: {a_text}")
        if cites:
            print(f"     Citations: {cites}")
        print()

    # ── Phase 3: Deduplication ───────────────────────────────────────
    print_section("Phase 3: Deduplication")

    # Add intentional duplicates/near-duplicates to exercise the pipeline
    duplicate_examples = [
        SyntheticExample(
            question=raw_examples[0].question,  # exact dup
            answer=raw_examples[0].answer,
            citations=raw_examples[0].citations,
        ),
        # Near-duplicate of eval question (contamination)
        SyntheticExample(
            question="How does CRISPR-Cas9 introduce double-strand breaks in DNA?",
            answer="Via Cas9 nuclease guided by gRNA.",
            citations=["doc_001"],
        ),
    ]
    augmented_examples = raw_examples + duplicate_examples
    print(f"Before dedup: {len(augmented_examples)} examples "
          f"({len(raw_examples)} generated + {len(duplicate_examples)} injected duplicates)")

    t0 = time.time()
    clean_examples = pipeline.deduplicate(augmented_examples)
    dedup_time = time.time() - t0

    removed_count = len(augmented_examples) - len(clean_examples)
    print(f"After dedup:  {len(clean_examples)} examples ({removed_count} removed) in {dedup_time:.3f}s")

    if dedup_agent.removed_exact_matches:
        print(f"\n  Exact matches removed: {len(dedup_agent.removed_exact_matches)}")
        for record in dedup_agent.removed_exact_matches:
            reason = record.get("reason", "intra-set duplicate")
            q = record.get("question", "")[:80]
            print(f"    - [{reason}] {q}")

    if dedup_agent.removed_near_duplicates:
        print(f"\n  Near-duplicates removed: {len(dedup_agent.removed_near_duplicates)}")
        for record in dedup_agent.removed_near_duplicates:
            cand = record.get("candidate", "")[:60]
            ref = record.get("reference", "")[:60]
            sim = record.get("similarity", 0.0)
            print(f"    - sim={sim:.3f}: '{cand}' ~ '{ref}'")

    # ── Full Stage I end-to-end ──────────────────────────────────────
    print_section("Full Stage I via SynthesisPipeline.run_stage_one()")

    dedup_agent2 = DeduplicationAgent(evaluation_questions=EVAL_QUESTIONS)
    pipeline2 = SynthesisPipeline(
        config=config,
        synthesizer=synthesizer,
        deduplication_agent=dedup_agent2,
        evaluation_questions=EVAL_QUESTIONS,
    )

    t0 = time.time()
    stage_one_output = pipeline2.run_stage_one(
        documents=doc_texts,
        num_examples=config.qa_generation_count,
    )
    stage_one_time = time.time() - t0

    print(f"Stage I complete in {stage_one_time:.1f}s")
    print(f"  Input documents:        {len(CORPUS_DOCUMENTS)}")
    print(f"  Target QA pairs:        {config.qa_generation_count}")
    print(f"  Output QA pairs:        {len(stage_one_output)}")
    print(f"  Exact matches removed:  {len(dedup_agent2.removed_exact_matches)}")
    print(f"  Near-duplicates removed: {len(dedup_agent2.removed_near_duplicates)}")

    print("\n  Final synthetic QA pairs:")
    for i, ex in enumerate(stage_one_output, 1):
        print(f"    [{i}] Q: {ex.question}")
        print(f"        A: {ex.answer}")
        if ex.citations:
            print(f"        Citations: {ex.citations}")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print_section("Summary")
    print(f"  Synthesis model:      {ZHIPU_MODEL} (GLM 4.5 Air)")
    print(f"  Corpus:               {len(CORPUS_DOCUMENTS)} documents")
    print(f"  Seed examples:        {len(SEED_EXAMPLES)} few-shot")
    print(f"  Eval set:             {len(EVAL_QUESTIONS)} held-out questions")
    print(f"  Generated (Phase 2):  {len(raw_examples)} QA pairs")
    print(f"  After dedup:          {len(stage_one_output)} clean pairs")
    eval_contam = sum(
        1 for r in dedup_agent.removed_exact_matches
        if r.get("reason") == "exact_match_eval_set"
    )
    print(f"  Contamination blocked: {eval_contam}")
    print(f"  Retrieval top_k:      {computed_k}")
    print(f"  Max synthesis steps:  {config.qa_max_steps}")
    print(f"  Vector search:        pseudo-embeddings (trigram, dim=256)")
    print(f"  Generation time:      {gen_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
