from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MainResultsRow:
    model_name: Optional[str] = None
    browsecomp_plus: Optional[float] = None
    trec_biogen: Optional[float] = None
    freshstack: Optional[float] = None
    financebench: Optional[float] = None
    qampari: Optional[float] = None
    pmbench: Optional[float] = None
    in_distribution: Optional[float] = None
    out_of_distribution: Optional[float] = None
    total: Optional[float] = None


@dataclass
class DistillationComparison:
    model_name: Optional[str] = None
    in_distribution: Optional[float] = None
    out_of_distribution: Optional[float] = None
    total: Optional[float] = None
    parallel_thinking_budget: Optional[int] = None


@dataclass
class IterationProgression:
    task_name: Optional[str] = None
    base: Optional[float] = None
    iteration_1: Optional[float] = None
    iteration_2: Optional[float] = None
    iteration_3: Optional[float] = None


class ExperimentRegistry:
    experiments = {
        "main_results": "Main results table (Table 4)",
        "cost_latency": "Cost and latency Pareto analysis",
        "multi_expert_distillation_vs_multitask_rl": "SFT distillation vs multi-task RL comparison",
        "multi_iteration_training": "Multi-iteration training progression",
    }

    main_results_table = {
        "KARL": MainResultsRow(
            model_name="KARL",
            browsecomp_plus=58.5,
            trec_biogen=80.2,
            freshstack=55.2,
            financebench=76.0,
            qampari=47.8,
            pmbench=35.7,
            in_distribution=69.4,
            out_of_distribution=53.7,
            total=58.9,
        ),
        "KARL (par. N = 3)": MainResultsRow(
            model_name="KARL (par. N = 3)",
            browsecomp_plus=62.2,
            trec_biogen=83.7,
            freshstack=57.7,
            financebench=80.8,
            qampari=55.1,
            pmbench=44.8,
            in_distribution=73.0,
            out_of_distribution=59.6,
            total=64.1,
        ),
        "KARL (par. N = 10)": MainResultsRow(
            model_name="KARL (par. N = 10)",
            browsecomp_plus=67.5,
            trec_biogen=86.7,
            freshstack=58.6,
            financebench=84.5,
            qampari=59.7,
            pmbench=47.8,
            in_distribution=77.1,
            out_of_distribution=62.7,
            total=67.5,
        ),
        "KARL (par. N = 20)": MainResultsRow(
            model_name="KARL (par. N = 20)",
            browsecomp_plus=69.5,
            trec_biogen=86.7,
            freshstack=58.1,
            financebench=84.2,
            qampari=60.8,
            pmbench=49.0,
            in_distribution=78.1,
            out_of_distribution=63.0,
            total=68.1,
        ),
        "KARL-BCP (VGS N = 17)": MainResultsRow(
            model_name="KARL-BCP (VGS N = 17)",
            browsecomp_plus=70.4,
            trec_biogen=None,
            freshstack=None,
            financebench=None,
            qampari=None,
            pmbench=None,
            in_distribution=None,
            out_of_distribution=None,
            total=55.5,
        ),
        "KARL-TREC": MainResultsRow(
            model_name="KARL-TREC",
            browsecomp_plus=42.2,
            trec_biogen=85.0,
            freshstack=56.7,
            financebench=68.3,
            qampari=50.8,
            pmbench=37.5,
            in_distribution=63.6,
            out_of_distribution=53.3,
            total=56.8,
        ),
    }

    distillation_vs_rl = {
        ("SFT Distillation", 0): DistillationComparison(
            model_name="SFT Distillation",
            in_distribution=69.1,
            out_of_distribution=59.4,
            total=None,
            parallel_thinking_budget=0,
        ),
        ("SFT Distillation", 15): DistillationComparison(
            model_name="SFT Distillation",
            in_distribution=75.3,
            out_of_distribution=59.6,
            total=None,
            parallel_thinking_budget=15,
        ),
        ("KARL", 0): DistillationComparison(
            model_name="KARL",
            in_distribution=67.9,
            out_of_distribution=62.7,
            total=None,
            parallel_thinking_budget=0,
        ),
        ("KARL", 15): DistillationComparison(
            model_name="KARL",
            in_distribution=78.4,
            out_of_distribution=64.8,
            total=None,
            parallel_thinking_budget=15,
        ),
    }

    multi_iteration_progression = {
        "TRECBiogen": IterationProgression(
            task_name="TRECBiogen",
            base=66.0,
            iteration_1=82.0,
            iteration_2=85.0,
            iteration_3=None,
        ),
        "FreshStack": IterationProgression(
            task_name="FreshStack",
            base=52.9,
            iteration_1=49.8,
            iteration_2=56.7,
            iteration_3=None,
        ),
        "QAMPARI": IterationProgression(
            task_name="QAMPARI",
            base=45.9,
            iteration_1=48.2,
            iteration_2=50.8,
            iteration_3=None,
        ),
    }

    @classmethod
    def get(cls, name):
        return cls.experiments.get(name)

    @classmethod
    def list_experiments(cls):
        return list(cls.experiments.keys())
