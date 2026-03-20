from __future__ import annotations

from pathlib import Path

from konash.training.project_state import (
    LEGACY_DEFAULT_PROJECT,
    archive_legacy_default_project,
    assess_project_reuse,
    begin_training_run,
    build_dataset_spec,
    save_project_manifest,
    suggest_project_name,
    TrainingProjectManifest,
    TrainingRunConfig,
)


def test_suggest_project_name_uses_dataset_and_model(tmp_path: Path):
    dataset_spec = build_dataset_spec([tmp_path / "financebench"])

    name = suggest_project_name("MiniMaxAI/MiniMax-M2.5", dataset_spec)

    assert name == "financebench-minimax-m2-5"


def test_archive_legacy_default_project_renames_existing_state(tmp_path: Path):
    projects_dir = tmp_path / "projects"
    legacy = projects_dir / LEGACY_DEFAULT_PROJECT
    legacy.mkdir(parents=True)
    (legacy / "training.jsonl").write_text("{}\n")

    archived = archive_legacy_default_project(str(projects_dir))

    assert archived is not None
    assert archived.exists()
    assert not legacy.exists()


def test_assess_project_reuse_requires_same_dataset_and_model(tmp_path: Path):
    projects_dir = tmp_path / "projects"
    corpus_dir = tmp_path / "financebench"
    corpus_dir.mkdir()
    dataset_spec = build_dataset_spec([corpus_dir])
    config = TrainingRunConfig(
        synthesis_backend="together",
        iterations=1,
        synthesis_calls=1,
        rollouts_per_example=8,
        rollout_max_steps=10,
    )
    save_project_manifest(
        TrainingProjectManifest(
            project="financebench-minimax-m2-5",
            display_name="FinanceBench on MiniMax M2.5",
            created_at="2026-03-20T00:00:00Z",
            base_model="MiniMaxAI/MiniMax-M2.5",
            dataset_spec=dataset_spec,
        ),
        projects_dir=str(projects_dir),
    )

    mismatch = assess_project_reuse(
        project="financebench-minimax-m2-5",
        base_model="zai-org/GLM-5",
        dataset_spec=dataset_spec,
        config=config,
        projects_dir=str(projects_dir),
    )

    assert mismatch.project_exists is True
    assert mismatch.compatible_project is False
    assert mismatch.reason == "project_identity_mismatch"


def test_assess_project_reuse_offers_resume_for_compatible_unfinished_run(tmp_path: Path):
    projects_dir = tmp_path / "projects"
    corpus_dir = tmp_path / "financebench"
    corpus_dir.mkdir()
    dataset_spec = build_dataset_spec([corpus_dir])
    config = TrainingRunConfig(
        synthesis_backend="together",
        iterations=1,
        synthesis_calls=1,
        rollouts_per_example=8,
        rollout_max_steps=10,
    )
    project = "financebench-minimax-m2-5"
    begin_training_run(
        project=project,
        display_name="FinanceBench on MiniMax M2.5",
        base_model="MiniMaxAI/MiniMax-M2.5",
        dataset_spec=dataset_spec,
        config=config,
        projects_dir=str(projects_dir),
    )
    checkpoint = projects_dir / project / "checkpoints" / "pipeline_state" / "iter1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "stage1_synthesis.json").write_text("{}")

    assessment = assess_project_reuse(
        project=project,
        base_model="MiniMaxAI/MiniMax-M2.5",
        dataset_spec=dataset_spec,
        config=config,
        projects_dir=str(projects_dir),
    )

    assert assessment.resume_available is True
    assert assessment.checkpoint.latest_phase == "synthesis"
