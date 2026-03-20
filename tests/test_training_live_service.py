from __future__ import annotations

from pathlib import Path

from tools.trace_viewer.training_live_service import (
    build_live_training_summary,
    list_live_training_projects,
)
from konash.training.logger import TrainingLogger


def test_build_live_training_summary_uses_latest_run_segment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    log = TrainingLogger("demo-project")
    log.start(iterations=1, corpus="old-corpus", model="old-model")
    log.oapl(iteration=1, epoch=1, loss=0.9, entropy=0.4, num_groups=4, num_rollouts=16)
    log.complete(iterations=1, total_seconds=8.0)

    fresh = TrainingLogger("demo-project")
    fresh.start(iterations=2, corpus="financebench", model="glm-5")
    fresh.oapl(
        iteration=2,
        epoch=3,
        loss=0.12,
        entropy=0.31,
        num_groups=8,
        num_rollouts=64,
        learning_rate=1e-6,
        duration_seconds=9.2,
    )

    summary = build_live_training_summary("demo-project")

    assert summary.summary.model == "glm-5"
    assert summary.summary.corpus == "financebench"
    assert summary.summary.latest_loss == 0.12
    assert summary.status == "running"
    assert summary.start is not None
    assert summary.start.model == "glm-5"
    assert summary.summary.iterations == 2


def test_list_live_training_projects_includes_demo_project(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    log = TrainingLogger("real-project")
    log.start(iterations=1, corpus="financebench", model="glm-5")
    log.oapl(iteration=1, epoch=1, loss=0.5, entropy=0.2, num_groups=4, num_rollouts=32)

    projects = list_live_training_projects("trainer-live-demo")
    names = {project.project for project in projects}

    assert "real-project" in names
    assert "trainer-live-demo" in names
