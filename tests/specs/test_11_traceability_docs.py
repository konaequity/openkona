from __future__ import annotations

from pathlib import Path


def test_verification_matrix_traces_core_karl_subsystems():
    matrix = Path("/Users/joeyroth/Desktop/openkona/docs/verification-matrix.md").read_text()
    for phrase in [
        "KARLBench",
        "BrowseComp-Plus",
        "TREC-Biogen",
        "FinanceBench",
        "QAMPARI",
        "FreshStack",
        "PMBench",
        "Agent Harness",
        "Data Synthesis",
        "Training",
        "Inference",
        "Corpus And Retrieval",
        "Parallel Thinking",
        "Value-Guided Search",
        "compression-aware segmentation",
        "nugget-based evaluation",
        "dataset statistics from KARLBench Table 2",
        "training prompt counts from Table 3",
        "prompt registries",
        "RL beyond sharpening via max@k",
        "search-environment ablations",
        "parallel-thinking rollout cost",
    ]:
        assert phrase in matrix, f"verification-matrix.md must mention {phrase}"


def test_verification_matrix_records_in_distribution_and_held_out_split():
    matrix = Path("/Users/joeyroth/Desktop/openkona/docs/verification-matrix.md").read_text()
    for phrase in [
        "in-distribution training tasks: BrowseComp-Plus and TREC-Biogen",
        "held-out evaluation tasks: FinanceBench, QAMPARI, FreshStack, PMBench",
    ]:
        assert phrase in matrix
