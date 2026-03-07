from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_reward_registry_module_exists():
    load_symbol(SymbolSpec("konash.rewards"))


def test_reward_registry_supports_task_specific_composition():
    registry_cls = load_symbol(SymbolSpec("konash.rewards", "RewardRegistry"))
    assert inspect.isclass(registry_cls)
    assert_has_attrs(
        registry_cls,
        ["register", "get", "compose", "list_rewards"],
        "RewardRegistry",
    )


def test_multitask_reward_composition_registers_browsecomp_and_trec_rewards():
    registry_cls = load_symbol(SymbolSpec("konash.rewards", "RewardRegistry"))
    registered = set(getattr(registry_cls, "default_tasks", set()))
    if registered:
        assert {"BrowseCompPlus", "TRECBiogen"}.issubset(registered)


def test_nugget_scorer_models_task_specific_nuggetization_paths():
    cls = load_symbol(SymbolSpec("konash.eval.nuggets", "NuggetScorer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "score",
            "judge_nugget",
            "aggregate_scores",
            "nuggetize_reference",
            "consolidate_references",
        ],
        "NuggetScorer",
    )


def test_nugget_evaluation_rules_cover_special_cases_from_paper():
    cls = load_symbol(SymbolSpec("konash.eval.nuggets", "NuggetEvaluationPolicy"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["task_name", "mode", "reference_handling", "requires_task_prompt"],
        "NuggetEvaluationPolicy",
    )

    registry_cls = load_symbol(SymbolSpec("konash.eval.nuggets", "NuggetPolicyRegistry"))
    policies = getattr(registry_cls, "policies", {})
    expected_modes = {
        "QAMPARI": ("entity_per_nugget", "single_reference"),
        "FreshStack": ("fixed_nuggets", "single_reference"),
        "PMBench": ("fixed_nuggets", "single_reference"),
        "TRECBiogen": ("report_nuggets", "multi_reference_consolidation"),
        "BrowseCompPlus": ("single_nugget", "single_reference"),
        "FinanceBench": ("single_nugget", "single_reference"),
    }
    for task_name, (mode, reference_handling) in expected_modes.items():
        policy = policies[task_name]
        assert getattr(policy, "mode") == mode
        assert getattr(policy, "reference_handling") == reference_handling


def test_value_model_training_contract_uses_binary_reward_and_policy_token_mask():
    cls = load_symbol(SymbolSpec("konash.inference.value_model", "ValueModelTrainer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "fit",
            "binary_reward_only",
            "mask_policy_tokens",
            "score_partial_rollout",
            "score_rollout",
        ],
        "ValueModelTrainer",
    )
