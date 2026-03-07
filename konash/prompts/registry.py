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


class PromptRegistry:
    prompts = {
        "figure_32_trec_dedup_paraphrase_judge": PromptTemplate(
            name="figure_32_trec_dedup_paraphrase_judge",
            category="dedup_paraphrase_judge",
            template="Determine if the following two TREC-Biogen questions are paraphrases of each other.",
            version="1.0",
        ),
        "figure_33_browsecomp_dedup_paraphrase_judge": PromptTemplate(
            name="figure_33_browsecomp_dedup_paraphrase_judge",
            category="dedup_paraphrase_judge",
            template="Determine if the following two BrowseComp-Plus questions are paraphrases of each other.",
            version="1.0",
        ),
        "figure_34_solver_rollout": PromptTemplate(
            name="figure_34_solver_rollout",
            category="rollout_solver",
            template="You are a knowledge agent with access to vector search. Answer the question using the available tools.",
            version="1.0",
        ),
        "figure_35_browsecomp_quality_filter": PromptTemplate(
            name="figure_35_browsecomp_quality_filter",
            category="quality_filter",
            template="Evaluate whether the following BrowseComp-Plus QA pair is ambiguous or has an incorrect reference answer.",
            version="1.0",
        ),
        "figure_36_trec_quality_filter": PromptTemplate(
            name="figure_36_trec_quality_filter",
            category="quality_filter",
            template="Evaluate whether the following TREC-Biogen QA pair is ambiguous or has an incorrect reference answer.",
            version="1.0",
        ),
        "appendix_d1_task_evaluation_prompts": PromptTemplate(
            name="appendix_d1_task_evaluation_prompts",
            category="nugget_evaluation",
            template="Evaluate the completeness of the answer by judging each nugget independently.",
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
