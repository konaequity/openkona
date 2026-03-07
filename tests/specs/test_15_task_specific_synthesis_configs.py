from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_task_specific_synthesis_config_module_exists():
    load_symbol(SymbolSpec("konash.synthesis.config"))


def test_synthesis_task_config_contract_exists():
    cls = load_symbol(SymbolSpec("konash.synthesis.config", "SynthesisTaskConfig"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "task_name",
            "seed_examples",
            "seed_documents",
            "qa_max_steps",
            "qa_generation_count",
            "solver_rollout_count",
            "solver_max_steps",
            "solver_top_k",
        ],
        "SynthesisTaskConfig",
    )


def test_trec_biogen_synthesis_config_matches_paper_defaults():
    registry_cls = load_symbol(SymbolSpec("konash.synthesis.config", "SynthesisConfigRegistry"))
    configs = getattr(registry_cls, "configs", {})
    config = configs["TRECBiogen"]
    assert getattr(config, "seed_examples") == 4
    assert getattr(config, "qa_max_steps") == 50
    assert getattr(config, "qa_generation_count") == 8
    assert getattr(config, "solver_rollout_count") == 8
    assert getattr(config, "solver_max_steps") == 50
    assert getattr(config, "solver_top_k") == 20


def test_browsecomp_synthesis_config_matches_paper_defaults():
    registry_cls = load_symbol(SymbolSpec("konash.synthesis.config", "SynthesisConfigRegistry"))
    configs = getattr(registry_cls, "configs", {})
    config = configs["BrowseCompPlus"]
    assert getattr(config, "seed_examples") == 4
    assert getattr(config, "seed_documents") == 10
    assert getattr(config, "qa_max_steps") == 60
    assert getattr(config, "qa_generation_count") == 8
    assert getattr(config, "solver_rollout_count") == 8
    assert getattr(config, "solver_max_steps") == 200
    assert getattr(config, "solver_top_k") == 20
    assert getattr(config, "compression_trigger_chars") == 150_000


def test_quality_filter_configs_capture_task_specific_judge_models():
    cls = load_symbol(SymbolSpec("konash.synthesis.config", "QualityFilterConfig"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["judge_model", "checks_ambiguity", "checks_reference_accuracy"],
        "QualityFilterConfig",
    )

    registry_cls = load_symbol(SymbolSpec("konash.synthesis.config", "SynthesisConfigRegistry"))
    configs = getattr(registry_cls, "configs", {})

    trec = configs["TRECBiogen"]
    browse = configs["BrowseCompPlus"]

    trec_quality = getattr(trec, "quality_filter", None)
    browse_quality = getattr(browse, "quality_filter", None)
    if trec_quality is not None:
        assert getattr(trec_quality, "judge_model") == "gpt-4o-mini"
    if browse_quality is not None:
        assert getattr(browse_quality, "judge_model") == "gpt-4o-mini"
