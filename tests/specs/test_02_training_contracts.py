from __future__ import annotations

import inspect

from tests.conftest import SymbolSpec, assert_has_attrs, load_symbol


def test_training_modules_exist():
    for module_name in [
        "konash.training.dataset",
        "konash.training.oapl",
        "konash.training.segmentation",
        "konash.training.multitask",
        "konash.training.iteration",
    ]:
        load_symbol(SymbolSpec(module_name))


def test_offline_rollout_dataset_contract_exists():
    cls = load_symbol(SymbolSpec("konash.training.dataset", "OfflineRolloutDataset"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["from_rollouts", "__len__", "__getitem__", "group_by_prompt"],
        "OfflineRolloutDataset",
    )


def test_oapl_trainer_exposes_reference_policy_and_kl_controls():
    cls = load_symbol(SymbolSpec("konash.training.oapl", "OAPLTrainer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        [
            "compute_loss",
            "estimate_optimal_value",
            "train_epoch",
            "reference_policy",
            "beta_value",
            "beta_kl",
        ],
        "OAPLTrainer",
    )


def test_segmentation_supports_compression_boundary_splitting_and_masking():
    cls = load_symbol(SymbolSpec("konash.training.segmentation", "RolloutSegmenter"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["split_on_compression", "mask_tool_outputs", "assign_rollout_reward"],
        "RolloutSegmenter",
    )


def test_multi_task_training_contract_combines_losses_without_task_specific_trainers():
    cls = load_symbol(SymbolSpec("konash.training.multitask", "MultiTaskTrainer"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["register_task", "compute_combined_loss", "balance_by_training_tokens"],
        "MultiTaskTrainer",
    )


def test_iterative_training_pipeline_regenerates_rollouts_between_iterations():
    cls = load_symbol(SymbolSpec("konash.training.iteration", "IterativeTrainingPipeline"))
    assert inspect.isclass(cls)
    assert_has_attrs(
        cls,
        ["run_iteration", "regenerate_rollouts", "promote_checkpoint", "evaluate_checkpoint"],
        "IterativeTrainingPipeline",
    )
