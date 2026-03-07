from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_offline_rollout_dataset_preserves_grouped_rollouts_and_rewards():
    cls = load_symbol(SymbolSpec("konash.training.dataset", "OfflineRolloutDataset"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["prompts", "group_rollouts", "rewards", "from_rollouts", "group_by_prompt"],
        "OfflineRolloutDataset",
    )


def test_oapl_trainer_exposes_group_value_estimation_and_masking_controls():
    cls = load_symbol(SymbolSpec("konash.training.oapl", "OAPLTrainer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "reference_policy",
            "beta_value",
            "beta_kl",
            "compute_group_value_estimate",
            "compute_squared_advantage_loss",
            "mask_non_model_tokens",
        ],
        "OAPLTrainer",
    )


def test_segmenter_supports_compression_segments_and_rollout_level_reward_assignment():
    cls = load_symbol(SymbolSpec("konash.training.segmentation", "RolloutSegmenter"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "split_on_compression",
            "include_compression_segments",
            "mask_tool_outputs",
            "assign_rollout_reward",
        ],
        "RolloutSegmenter",
    )


def test_iterative_pipeline_limits_iterations_and_regenerates_data():
    cls = load_symbol(SymbolSpec("konash.training.iteration", "IterativeTrainingPipeline"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["max_iterations", "run_iteration", "regenerate_rollouts", "promote_checkpoint"],
        "IterativeTrainingPipeline",
    )
    max_iterations = getattr(cls, "max_iterations", None)
    if isinstance(max_iterations, int):
        assert max_iterations <= 3, "Paper-derived default should not exceed 3 iterations"


def test_multi_task_trainer_balances_training_tokens_for_two_in_distribution_tasks():
    cls = load_symbol(SymbolSpec("konash.training.multitask", "MultiTaskTrainer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "in_distribution_tasks",
            "balance_by_training_tokens",
            "register_task",
            "compute_combined_loss",
        ],
        "MultiTaskTrainer",
    )

    tasks = set(getattr(cls, "in_distribution_tasks", set()))
    if tasks:
        assert tasks == {"BrowseCompPlus", "TRECBiogen"}
