"""Authoritative benchmark and corpus registry for KONASH."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


DEFAULT_CORPUS_DIR = os.path.expanduser("~/.konash/corpora")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Per-benchmark evaluation configuration."""

    name: str
    policy_name: str
    project_name: str
    benchmark_key: str
    top_k: int = 20
    get_reference: Callable[[dict], str] | None = None
    get_nuggets: Callable[[dict], Optional[List[str]]] | None = None
    get_question_text: Callable[[dict], str] | None = None
    get_judge_context: Callable[[str, "BenchmarkConfig"], str] | None = None
    get_progress_ref_display: Callable[[dict], str] | None = None
    get_progress_detail: Callable[[dict], Optional[str]] | None = None
    paper_target: Optional[str] = None
    extra_table_columns: List[tuple[str, str]] = field(default_factory=list)
    get_extra_output: Callable[[dict], dict] | None = None
    extra_output_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkHooks:
    """Optional benchmark-specific runner capabilities."""

    supports_train_quick: bool = False
    supports_passk: bool = False
    writes_traces: bool = False
    trace_dir: str = "tools/trace_viewer/data"


@dataclass(frozen=True)
class DatasetSpec:
    """Canonical corpus layout + benchmark metadata."""

    key: str
    name: str
    description: str
    source: str
    root_dirname: str
    content_subdir: str
    downloader_name: str
    eval_filename: str = "eval_questions.json"
    benchmark: BenchmarkConfig | None = None
    hooks: BenchmarkHooks = field(default_factory=BenchmarkHooks)

    def corpus_root(self, base_dir: Optional[str] = None) -> str:
        return os.path.join(base_dir or DEFAULT_CORPUS_DIR, self.root_dirname)

    def content_path(self, corpus_root: Optional[str] = None) -> str:
        root = corpus_root or self.corpus_root()
        return os.path.join(root, self.content_subdir)

    def eval_questions_path(self, corpus_root: Optional[str] = None) -> str:
        root = corpus_root or self.corpus_root()
        return os.path.join(root, self.eval_filename)

    def download_fn(self) -> Callable:
        from konash import download as _download

        return getattr(_download, self.downloader_name)

    def download(self, **kwargs: Any) -> str:
        return self.download_fn()(**kwargs)


def _financebench_judge_context(question: str, config: BenchmarkConfig) -> str:
    return (
        f"{question}\n\n"
        "Note: For numerical values, rounding differences should be ignored "
        "if they do not meaningfully change the answer. For example, $8.74 "
        "billion and $8.70 billion refer to the same figure with different "
        "rounding precision. Two numbers are considered equivalent if one "
        "can be rounded to the other."
    )


def _financebench_progress_detail(result: dict) -> str:
    return f"[dim]Steps:[/] {result.get('num_steps', 0)} ({result.get('num_searches', 0)} searches)"


def _qampari_progress_detail(result: dict) -> str:
    n_nuggets = result.get("num_nuggets", 0)
    found = sum(1 for s in result.get("nugget_scores", []) if s >= 0.6)
    return f"[dim]Entities:[/] {found}/{n_nuggets} found"


def _qampari_extra_output(eval_result: dict) -> dict:
    all_nugget_scores = []
    for result in eval_result["results"]:
        all_nugget_scores.extend(result.get("nugget_scores", []))
    avg = sum(all_nugget_scores) / len(all_nugget_scores) if all_nugget_scores else 0
    return {"avg_nugget_completion": avg}


def _freshstack_progress_detail(result: dict) -> str:
    n_nuggets = result.get("num_nuggets", 0)
    found = sum(1 for s in result.get("nugget_scores", []) if s >= 0.6)
    return f"[dim]Nuggets:[/] {found}/{n_nuggets} supported"


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "financebench": DatasetSpec(
        key="financebench",
        name="FinanceBench",
        description="Financial QA  ·  SEC filings  ·  150 questions",
        source="PatronusAI/financebench",
        root_dirname="financebench",
        content_subdir="documents",
        downloader_name="download_financebench",
        benchmark=BenchmarkConfig(
            name="FinanceBench",
            policy_name="FinanceBench",
            project_name="eval-financebench",
            benchmark_key="financebench",
            top_k=20,
            get_reference=lambda q: q["answer"],
            get_nuggets=lambda q: None,
            get_question_text=lambda q: q["question"],
            get_judge_context=_financebench_judge_context,
            get_progress_ref_display=lambda q: q["answer"][:120],
            get_progress_detail=_financebench_progress_detail,
        ),
        hooks=BenchmarkHooks(
            supports_train_quick=True,
            supports_passk=True,
            writes_traces=True,
        ),
    ),
    "browsecomp-plus": DatasetSpec(
        key="browsecomp-plus",
        name="BrowseComp-Plus",
        description="Web retrieval benchmark  ·  40K docs  ·  encrypted",
        source="Tevatron/browsecomp-plus",
        root_dirname="browsecomp-plus",
        content_subdir="documents",
        downloader_name="download_browsecomp_plus",
    ),
    "qampari": DatasetSpec(
        key="qampari",
        name="QAMPARI",
        description="Multi-answer QA  ·  Wikipedia  ·  entity-rich",
        source="momo4382/QAMPARI",
        root_dirname="qampari",
        content_subdir="documents",
        downloader_name="download_qampari",
        benchmark=BenchmarkConfig(
            name="QAMPARI",
            policy_name="QAMPARI",
            project_name="eval-qampari",
            benchmark_key="qampari",
            top_k=20,
            get_reference=lambda q: "; ".join(q["answers"]),
            get_nuggets=lambda q: None,
            get_question_text=lambda q: q["question"],
            get_progress_ref_display=lambda q: (
                ", ".join(q["answers"][:5]) + ("..." if len(q["answers"]) > 5 else "")
            ),
            get_progress_detail=_qampari_progress_detail,
            get_extra_output=_qampari_extra_output,
            extra_table_columns=[("Nugget Completion", "avg_nugget_completion")],
            paper_target="KARL paper target (GLM 4.5 Air base): 45.9% avg score",
        ),
        hooks=BenchmarkHooks(),
    ),
    "freshstack": DatasetSpec(
        key="freshstack",
        name="FreshStack",
        description="Multi-domain retrieval  ·  5 domains  ·  recent docs",
        source="freshstack",
        root_dirname="freshstack",
        content_subdir="documents",
        downloader_name="download_freshstack",
        benchmark=BenchmarkConfig(
            name="FreshStack",
            policy_name="FreshStack",
            project_name="eval-freshstack",
            benchmark_key="freshstack",
            top_k=10,
            get_reference=lambda q: "; ".join(n["text"] for n in q["nuggets"]),
            get_nuggets=lambda q: [n["text"] for n in q["nuggets"]],
            get_question_text=lambda q: q["question"],
            get_progress_ref_display=lambda q: q.get("question_title", q["question"])[:100],
            get_progress_detail=_freshstack_progress_detail,
            paper_target="KARL paper target (GLM 4.5 Air base): 52.9% avg score",
            extra_output_fields={"domain": "langchain"},
        ),
        hooks=BenchmarkHooks(),
    ),
}


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {key}") from exc


def get_benchmark_config(key: str) -> BenchmarkConfig:
    dataset = get_dataset(key)
    if dataset.benchmark is None:
        raise KeyError(f"Dataset {key!r} does not define benchmark config")
    return dataset.benchmark


def list_datasets() -> List[DatasetSpec]:
    return list(DATASET_REGISTRY.values())
