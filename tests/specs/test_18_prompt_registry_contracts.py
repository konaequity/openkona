from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_prompt_registry_module_exists():
    load_symbol(SymbolSpec("konash.prompts.registry"))


def test_prompt_template_contract_exists():
    cls = load_symbol(SymbolSpec("konash.prompts.registry", "PromptTemplate"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["name", "category", "template", "version"],
        "PromptTemplate",
    )


def test_prompt_registry_covers_paper_prompt_families():
    registry_cls = load_symbol(SymbolSpec("konash.prompts.registry", "PromptRegistry"))
    assert inspect.isclass(registry_cls)
    assert_has_attrs(
        cls := registry_cls,
        ["prompts", "get", "list_by_category"],
        "PromptRegistry",
    )

    prompts = getattr(cls, "prompts", {})
    required_categories = {
        "synthesis",
        "rollout_solver",
        "quality_filter",
        "dedup_paraphrase_judge",
        "nugget_evaluation",
        "nugget_consolidation",
        "aggregation",
    }
    categories = {getattr(prompt, "category", None) for prompt in prompts.values()}
    missing = required_categories - categories
    assert not missing, f"Missing prompt categories: {sorted(missing)}"


def test_prompt_registry_tracks_appendix_referenced_prompt_artifacts():
    registry_cls = load_symbol(SymbolSpec("konash.prompts.registry", "PromptRegistry"))
    prompts = getattr(registry_cls, "prompts", {})

    required_prompt_names = {
        "figure_32_trec_dedup_paraphrase_judge",
        "figure_33_browsecomp_dedup_paraphrase_judge",
        "figure_34_solver_rollout",
        "figure_35_browsecomp_quality_filter",
        "figure_36_trec_quality_filter",
        "appendix_d1_task_evaluation_prompts",
    }
    missing = required_prompt_names - prompts.keys()
    assert not missing, f"Missing prompt artifacts: {sorted(missing)}"
