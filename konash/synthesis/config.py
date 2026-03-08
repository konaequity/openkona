from __future__ import annotations


class QualityFilterConfig:
    judge_model = None
    checks_ambiguity = True
    checks_reference_accuracy = True

    def __init__(self, judge_model=None, checks_ambiguity=True, checks_reference_accuracy=True):
        self.judge_model = judge_model
        self.checks_ambiguity = checks_ambiguity
        self.checks_reference_accuracy = checks_reference_accuracy


class SynthesisTaskConfig:
    task_name = None
    seed_examples = None
    seed_documents = None
    qa_max_steps = None
    qa_top_k = None
    qa_generation_count = None
    solver_rollout_count = None
    solver_max_steps = None
    solver_top_k = None
    compression_trigger_chars = None
    quality_filter = None

    def __init__(
        self,
        task_name=None,
        seed_examples=None,
        seed_documents=None,
        qa_max_steps=None,
        qa_top_k=None,
        qa_generation_count=None,
        solver_rollout_count=None,
        solver_max_steps=None,
        solver_top_k=None,
        compression_trigger_chars=None,
        quality_filter=None,
    ):
        self.task_name = task_name
        self.seed_examples = seed_examples
        self.seed_documents = seed_documents
        self.qa_max_steps = qa_max_steps
        self.qa_top_k = qa_top_k
        self.qa_generation_count = qa_generation_count
        self.solver_rollout_count = solver_rollout_count
        self.solver_max_steps = solver_max_steps
        self.solver_top_k = solver_top_k
        self.compression_trigger_chars = compression_trigger_chars
        self.quality_filter = quality_filter


class SynthesisConfigRegistry:
    configs = {
        "TriviaNight": SynthesisTaskConfig(
            task_name="TriviaNight",
            seed_examples=6,
            seed_documents=None,
            qa_max_steps=30,
            qa_top_k=10,
            qa_generation_count=16,
            solver_rollout_count=8,
            solver_max_steps=20,
            solver_top_k=10,
            quality_filter=QualityFilterConfig(judge_model=None),
        ),
        "TRECBiogen": SynthesisTaskConfig(
            task_name="TRECBiogen",
            seed_examples=4,
            seed_documents=None,
            qa_max_steps=50,
            qa_top_k=20,
            qa_generation_count=8,
            solver_rollout_count=8,
            solver_max_steps=50,
            solver_top_k=20,
            quality_filter=QualityFilterConfig(judge_model="gpt-5-mini"),
        ),
        "BrowseCompPlus": SynthesisTaskConfig(
            task_name="BrowseCompPlus",
            seed_examples=4,
            seed_documents=10,
            qa_max_steps=60,
            qa_top_k=5,
            qa_generation_count=8,
            solver_rollout_count=8,
            solver_max_steps=200,
            solver_top_k=20,
            compression_trigger_chars=150_000,
            quality_filter=QualityFilterConfig(judge_model="gpt-4o-mini"),
        ),
    }
