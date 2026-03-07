from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_training_data_stats_module_exists():
    load_symbol(SymbolSpec("konash.training.stats"))


def test_training_prompt_counts_capture_table_three_values():
    cls = load_symbol(SymbolSpec("konash.training.stats", "TrainingPromptStats"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["iteration_name", "task_name", "num_training_prompts"],
        "TrainingPromptStats",
    )

    registry_cls = load_symbol(SymbolSpec("konash.training.stats", "TrainingPromptStatsRegistry"))
    stats = getattr(registry_cls, "stats", {})
    expected = {
        ("KARL Iter. 1", "BrowseCompPlus"): 1218,
        ("KARL Iter. 1", "TRECBiogen"): 6270,
        ("KARL Iter. 2", "BrowseCompPlus"): 1336,
        ("KARL Iter. 2", "TRECBiogen"): 11371,
    }
    for key, value in expected.items():
        assert key in stats, f"Missing training prompt count for {key}"
        assert getattr(stats[key], "num_training_prompts") == value


def test_iterative_training_defaults_capture_paper_representative_case():
    cls = load_symbol(SymbolSpec("konash.training.iteration", "IterationDefaults"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["max_iterations", "representative_case_task", "supports_large_batch_offline_updates"],
        "IterationDefaults",
    )

    task_name = getattr(cls, "representative_case_task", None)
    if task_name is not None:
        assert task_name == "TRECBiogen"
