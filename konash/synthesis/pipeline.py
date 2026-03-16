from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from konash.synthesis.rollouts import RolloutGenerator, RolloutGroup
from konash.synthesis.filters import PassRateFilter, QualityFilter, GroundingFilter
from konash.synthesis.dedup import EmbeddingDeduplicator, DeduplicationAgent
from konash.synthesis.config import SynthesisTaskConfig, QualityFilterConfig


class SynthesisPipeline:
    """End-to-end synthesis pipeline orchestrating QA generation, deduplication,
    rollout generation, pass-rate filtering, and quality filtering.

    Stage One: QA synthesis + deduplication
        - Generate synthetic QA pairs via ``QuestionAnswerSynthesizer``
        - Deduplicate against evaluation set and within synthetic set

    Stage Two: Rollout generation + filtering
        - Generate solver rollouts for each QA pair
        - Filter by pass rate (remove trivially easy / impossible)
        - Apply quality filters (ambiguity + reference accuracy)
    """

    def __init__(
        self,
        config: Optional[SynthesisTaskConfig] = None,
        synthesizer: Optional[QuestionAnswerSynthesizer] = None,
        rollout_generator: Optional[RolloutGenerator] = None,
        deduplication_agent: Optional[DeduplicationAgent] = None,
        pass_rate_filter: Optional[PassRateFilter] = None,
        quality_filter: Optional[QualityFilter] = None,
        grounding_filter: Optional[GroundingFilter] = None,
        evaluation_questions: Optional[List[str]] = None,
        judge_fn: Any = None,
    ):
        self.config = config
        self.evaluation_questions = evaluation_questions or []
        self.judge_fn = judge_fn

        # Build sub-components from config if not provided
        self.synthesizer = synthesizer or self._build_synthesizer(config)
        self.rollout_generator = rollout_generator or self._build_rollout_generator(config)
        self.deduplication_agent = deduplication_agent or DeduplicationAgent(
            evaluation_questions=self.evaluation_questions,
        )
        self.pass_rate_filter = pass_rate_filter or self._build_pass_rate_filter()
        self.quality_filter = quality_filter or self._build_quality_filter(config, judge_fn)
        self.grounding_filter = grounding_filter or GroundingFilter()

        # Pipeline state
        self.synthetic_examples: List[SyntheticExample] = []
        self.rollout_groups: List[RolloutGroup] = []
        self.filtered_groups: List[RolloutGroup] = []
        self.final_examples: List[SyntheticExample] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_stage_one(
        self,
        documents: Optional[List[str]] = None,
        seed_examples: Optional[List[SyntheticExample]] = None,
        num_examples: Optional[int] = None,
    ) -> List[SyntheticExample]:
        """Stage One: QA synthesis + deduplication.

        1. Generate synthetic QA pairs (optionally bootstrapping from seeds).
        2. Deduplicate within the synthetic set.
        3. Deduplicate against the evaluation set (contamination prevention).

        Parameters
        ----------
        documents : list[str] | None
            Source documents for synthesis.
        seed_examples : list[SyntheticExample] | None
            Seed examples for bootstrapping.
        num_examples : int | None
            Target number of QA pairs. Defaults to config's qa_generation_count.

        Returns
        -------
        list[SyntheticExample]
            Deduplicated synthetic examples.
        """
        count = num_examples
        if count is None and self.config is not None:
            count = self.config.qa_generation_count

        # Phase 1: Generate
        if seed_examples:
            self.synthetic_examples = self.synthesizer.bootstrap_from_examples(seed_examples)
        else:
            self.synthetic_examples = self.synthesizer.synthesize(
                documents=documents, num_examples=count
            )

        # Phase 2: Deduplicate
        self.synthetic_examples = self.deduplicate(self.synthetic_examples)

        return self.synthetic_examples

    def run_stage_two(
        self,
        examples: Optional[List[SyntheticExample]] = None,
        num_rollouts: Optional[int] = None,
        reference_documents: Optional[List[str]] = None,
        parallel_workers: int = 4,
    ) -> List[SyntheticExample]:
        """Stage Two: Rollout generation + pass-rate filter + quality filter.

        1. Generate solver rollouts for each QA pair (parallelized across pairs).
        2. Estimate pass rates and filter by range.
        3. Apply quality filters (ambiguity, reference accuracy).

        Parameters
        ----------
        examples : list[SyntheticExample] | None
            Examples to process. Defaults to output of stage one.
        num_rollouts : int | None
            Rollouts per example. Defaults to config's solver_rollout_count.
        reference_documents : list[str] | None
            Documents for quality filter reference-accuracy checking.
        parallel_workers : int
            Number of QA pairs to process in parallel (default 4).

        Returns
        -------
        list[SyntheticExample]
            Final filtered examples.
        """
        if examples is not None:
            self.synthetic_examples = list(examples)

        rollout_count = num_rollouts
        if rollout_count is None and self.config is not None:
            rollout_count = self.config.solver_rollout_count
        rollout_count = rollout_count or 8

        # Phase 1: Generate rollouts (parallel across QA pairs)
        num_qa = len(self.synthetic_examples)
        results: List[Optional[RolloutGroup]] = [None] * num_qa

        def _generate_for_qa(qa_idx_ex):
            qa_idx, ex = qa_idx_ex
            return qa_idx, self.rollout_generator.generate_group(
                prompt=ex.question,
                reference_answer=ex.answer,
                num_rollouts=rollout_count,
                qa_idx=qa_idx,
            )

        max_workers = min(parallel_workers, num_qa) if num_qa > 0 else 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for qa_idx, group in pool.map(
                _generate_for_qa, enumerate(self.synthetic_examples)
            ):
                results[qa_idx] = group

        self.rollout_groups = [g for g in results if g is not None]

        # Phase 2: Pass-rate filtering
        self.filtered_groups = self.estimate_pass_rate(self.rollout_groups)

        # Phase 3: Quality filtering
        # Map surviving groups back to examples, preserving rollout attempts
        prompt_to_group = {g.prompt: g for g in self.filtered_groups}
        surviving_examples = [
            ex for ex in self.synthetic_examples if ex.question in prompt_to_group
        ]

        # Build per-example rollout attempts for the quality filter
        # (KARL paper Sections 7.2.1-7.2.2: the quality judge receives the
        # synthesized question, reference answer, AND solver rollout attempts)
        rollout_attempts: List[List[Dict[str, Any]]] = []
        for ex in surviving_examples:
            group = prompt_to_group.get(ex.question)
            if group is not None:
                attempts = [
                    {
                        "answer": r.final_answer or "",
                        "score": 1.0 if r.passed else 0.0,
                        "passed": r.passed,
                    }
                    for r in group.rollouts
                    if r is not None
                ]
                rollout_attempts.append(attempts)
            else:
                rollout_attempts.append([])

        self.final_examples = self.apply_quality_filter(
            surviving_examples,
            reference_documents=reference_documents,
            rollout_attempts=rollout_attempts,
        )

        return self.final_examples

    def deduplicate(
        self,
        examples: List[SyntheticExample],
    ) -> List[SyntheticExample]:
        """Deduplicate synthetic examples using the deduplication agent.

        Removes exact duplicates and near-duplicates both within the
        synthetic set and against the evaluation set.

        Parameters
        ----------
        examples : list[SyntheticExample]
            Examples to deduplicate.

        Returns
        -------
        list[SyntheticExample]
            Deduplicated examples.
        """
        if not examples:
            return []

        questions = [ex.question for ex in examples]
        question_to_example: Dict[str, SyntheticExample] = {}
        for ex in examples:
            if ex.question not in question_to_example:
                question_to_example[ex.question] = ex

        # Run the deduplication agent
        clean_questions = self.deduplication_agent.run(
            synthetic_questions=questions,
            evaluation_questions=self.evaluation_questions or None,
        )

        # Map back to examples
        return [question_to_example[q] for q in clean_questions if q in question_to_example]

    def estimate_pass_rate(
        self,
        rollout_groups: List[RolloutGroup],
    ) -> List[RolloutGroup]:
        """Estimate pass rates for rollout groups and filter by range.

        Parameters
        ----------
        rollout_groups : list[RolloutGroup]
            Groups with rollout results.

        Returns
        -------
        list[RolloutGroup]
            Groups that pass the pass-rate filter.
        """
        return self.pass_rate_filter.apply(rollout_groups)

    def apply_quality_filter(
        self,
        examples: List[SyntheticExample],
        reference_documents: Optional[List[str]] = None,
        rollout_attempts: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[SyntheticExample]:
        """Apply the quality filter to a list of examples.

        Parameters
        ----------
        examples : list[SyntheticExample]
            Examples to filter.
        reference_documents : list[str] | None
            Source documents for reference-accuracy checking.
        rollout_attempts : list[list[dict]] | None
            Per-example solver rollout attempts.  Each attempt dict has
            ``answer`` (str), ``score`` (float), and ``passed`` (bool).
            When provided alongside a ``judge_fn``, the paper's structured
            quality prompts (Figures 35-36) are used.

        Returns
        -------
        list[SyntheticExample]
            Quality-filtered examples.
        """
        return self.quality_filter.apply(
            examples,
            reference_documents=reference_documents,
            rollout_attempts=rollout_attempts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_synthesizer(config: Optional[SynthesisTaskConfig]) -> QuestionAnswerSynthesizer:
        """Build a synthesizer from config."""
        kwargs: Dict[str, Any] = {}
        if config is not None:
            if config.qa_generation_count is not None:
                kwargs["generation_count"] = config.qa_generation_count
            if config.qa_max_steps is not None:
                kwargs["max_steps"] = config.qa_max_steps
            if config.qa_top_k is not None:
                kwargs["top_k"] = config.qa_top_k
        return QuestionAnswerSynthesizer(**kwargs)

    @staticmethod
    def _build_rollout_generator(config: Optional[SynthesisTaskConfig]) -> RolloutGenerator:
        """Build a rollout generator from config."""
        kwargs: Dict[str, Any] = {}
        if config is not None:
            if config.solver_max_steps is not None:
                kwargs["max_steps"] = config.solver_max_steps
            if config.solver_top_k is not None:
                kwargs["top_k"] = config.solver_top_k
        return RolloutGenerator(**kwargs)

    @staticmethod
    def _build_pass_rate_filter() -> PassRateFilter:
        """Build a default pass-rate filter.

        Default range [0.1, 0.9] rejects groups where the solver always
        fails or always succeeds.
        """
        return PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)

    @staticmethod
    def _build_quality_filter(
        config: Optional[SynthesisTaskConfig],
        judge_fn: Any = None,
    ) -> QualityFilter:
        """Build a quality filter from config, wiring in the LLM judge."""
        kwargs: Dict[str, Any] = {}
        if judge_fn is not None:
            kwargs["judge_fn"] = judge_fn
        if config is not None and config.quality_filter is not None:
            qf = config.quality_filter
            if hasattr(qf, "judge_model"):
                kwargs["judge_model"] = qf.judge_model
            if hasattr(qf, "checks_ambiguity"):
                kwargs["checks_ambiguity"] = qf.checks_ambiguity
            if hasattr(qf, "checks_reference_accuracy"):
                kwargs["checks_reference_accuracy"] = qf.checks_reference_accuracy
        return QualityFilter(**kwargs)
