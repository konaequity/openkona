from __future__ import annotations

from konash.benchmarks import DEFAULT_CORPUS_DIR, get_benchmark_config, get_dataset, list_datasets


def test_dataset_registry_captures_corpus_layouts():
    financebench = get_dataset("financebench")
    qampari = get_dataset("qampari")
    freshstack = get_dataset("freshstack")
    browsecomp = get_dataset("browsecomp-plus")

    assert financebench.corpus_root() == f"{DEFAULT_CORPUS_DIR}/financebench"
    assert financebench.content_path() == f"{DEFAULT_CORPUS_DIR}/financebench/documents"
    assert qampari.content_path() == f"{DEFAULT_CORPUS_DIR}/qampari/documents"
    assert freshstack.content_path() == f"{DEFAULT_CORPUS_DIR}/freshstack/documents"
    assert browsecomp.content_path() == f"{DEFAULT_CORPUS_DIR}/browsecomp-plus/documents"
    assert financebench.content_path("/tmp/financebench") == "/tmp/financebench/documents"
    assert financebench.eval_questions_path("/tmp/financebench") == "/tmp/financebench/eval_questions.json"


def test_dataset_registry_resolves_downloaders_lazily():
    assert get_dataset("financebench").download_fn().__name__ == "download_financebench"
    assert get_dataset("qampari").download_fn().__name__ == "download_qampari"
    assert get_dataset("freshstack").download_fn().__name__ == "download_freshstack"
    assert get_dataset("browsecomp-plus").download_fn().__name__ == "download_browsecomp_plus"


def test_benchmark_registry_provides_eval_config():
    financebench = get_benchmark_config("financebench")
    qampari = get_benchmark_config("qampari")
    freshstack = get_benchmark_config("freshstack")

    assert financebench.policy_name == "FinanceBench"
    assert qampari.extra_table_columns == [("Nugget Completion", "avg_nugget_completion")]
    assert freshstack.extra_output_fields == {"domain": "langchain"}
    assert get_dataset("financebench").hooks.supports_passk is True
    assert get_dataset("financebench").hooks.writes_traces is True
    assert get_dataset("qampari").hooks.supports_passk is False
    assert get_dataset("freshstack").hooks.supports_train_quick is False


def test_list_datasets_includes_all_supported_benchmarks():
    assert {dataset.key for dataset in list_datasets()} == {
        "browsecomp-plus",
        "financebench",
        "freshstack",
        "qampari",
    }
