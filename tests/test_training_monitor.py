from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from konash.training.logger import TrainingLogger
from konash.training.project_state import (
    ActiveTrainingRun,
    TrainingDatasetSpec,
    TrainingRunConfig,
    save_active_run,
)
from tools.trace_viewer.app import app


def _write_project(
    *,
    projects_root: Path,
    project: str,
    model: str,
    status: str,
    updated_at: datetime,
    recent: bool,
) -> None:
    corpus_dir = projects_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    project_dir = projects_root / project
    corpus_spec = TrainingDatasetSpec.from_dict({
        "sources": [{
            "source_id": "corpus",
            "display_name": "Corpus",
            "path": str(corpus_dir),
            "fingerprint": "abc123",
        }],
        "mixing_strategy": "single",
        "weights": None,
    })
    config = TrainingRunConfig(
        synthesis_backend="together",
        iterations=1,
        synthesis_calls=1,
        rollouts_per_example=1,
        rollout_max_steps=1,
    )
    if recent:
        log = TrainingLogger(project)
        log.start(iterations=1, corpus="financebench", model=model)
        log.oapl(iteration=1, epoch=1, loss=0.5, entropy=0.2, num_groups=4, num_rollouts=8)
    else:
        log_path = project_dir / "training.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project": project,
            "elapsed_seconds": 0.0,
            "timestamp": updated_at.isoformat(),
            "iterations": 1,
            "corpus": "financebench",
            "model": model,
            "event": "start",
        }
        log_path.write_text(json.dumps(payload) + "\n")

    training_jsonl = project_dir / "training.jsonl"
    os.utime(training_jsonl, (updated_at.timestamp(), updated_at.timestamp()))

    save_active_run(
        project,
        ActiveTrainingRun(
            run_id=f"run-{project}",
            status=status,
            started_at=updated_at.isoformat(),
            updated_at=updated_at.isoformat(),
            base_model=model,
            dataset_spec=corpus_spec,
            config=config,
        ),
    )


def test_training_projects_prefers_recent_live_run(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    projects_root = tmp_path / ".konash" / "projects"
    old_time = datetime.now(timezone.utc) - timedelta(days=2)
    fresh_time = datetime.now(timezone.utc) - timedelta(minutes=5)

    _write_project(
        projects_root=projects_root,
        project="financebench-minimax-m2-5",
        model="MiniMaxAI/MiniMax-M2.5",
        status="running",
        updated_at=old_time,
        recent=False,
    )
    _write_project(
        projects_root=projects_root,
        project="test-checkpoints",
        model="zai-org/GLM-4.5-Air",
        status="running",
        updated_at=fresh_time,
        recent=True,
    )

    with app.test_client() as client:
        response = client.get("/training/api/projects")

    payload = response.get_json()
    assert payload["default_project"] == "test-checkpoints"

    projects = {project["name"]: project for project in payload["projects"]}
    assert projects["financebench-minimax-m2-5"]["is_running"] is False
    assert projects["test-checkpoints"]["is_running"] is True


def test_training_debug_trace_endpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    project = "trace-project"
    projects_root = tmp_path / ".konash" / "projects"
    _write_project(
        projects_root=projects_root,
        project=project,
        model="zai-org/GLM-4.5-Air",
        status="running",
        updated_at=datetime.now(timezone.utc),
        recent=True,
    )

    debug_dir = projects_root / project
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "training_debug.log").write_text(
        "2026-03-21T12:00:00 DEBUG    konash.synthesis.qa synthesis_search step=1 query='finance bench revenue' results=20\n"
        "2026-03-21T12:00:01 DEBUG    konash.synthesis.qa synthesis_search_results step=1 query='finance bench revenue' formatted_len=1234 formatted_preview='[1] revenue details'\n"
    )

    with app.test_client() as client:
        response = client.get(f"/training/api/debug/{project}?limit=5")

    payload = response.get_json()
    assert payload["project"] == project
    assert len(payload["events"]) == 2
    assert payload["events"][0]["kind"] == "search"
    assert "finance bench revenue" in payload["events"][0]["detail"]
