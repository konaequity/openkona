from __future__ import annotations

from pathlib import Path

from konash.training.events import OaplEpochLogged, PhaseTimingLogged, TrainingStarted
from konash.training.logger import TrainingLogger


def test_training_logger_round_trips_typed_events(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    log = TrainingLogger("demo-project")
    log.start(iterations=2, corpus="financebench", model="glm")
    log.oapl(
        iteration=1,
        epoch=3,
        loss=0.123456,
        entropy=0.4321,
        kl=0.0042,
        num_groups=8,
        num_rollouts=64,
        learning_rate=1e-6,
        duration_seconds=9.8,
    )
    log.phase(iteration=1, phase="upload", duration_seconds=3.1, remote_job_id="job-123")

    events = TrainingLogger.load("demo-project")

    assert isinstance(events[0], TrainingStarted)
    assert events[0].iterations == 2
    assert events[0].corpus == "financebench"

    assert isinstance(events[1], OaplEpochLogged)
    assert events[1].epoch == 3
    assert events[1].learning_rate == 1e-6

    assert isinstance(events[2], PhaseTimingLogged)
    assert events[2].phase == "upload"
    assert events[2].extra["remote_job_id"] == "job-123"


def test_training_logger_load_records_returns_json_safe_dicts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))

    log = TrainingLogger("demo-project")
    log.start(iterations=1, corpus="corpus", model="model")

    records = TrainingLogger.load_records("demo-project")

    assert records == [
        {
            "event": "start",
            "project": "demo-project",
            "iterations": 1,
            "corpus": "corpus",
            "model": "model",
            "billing_started_at": "",
            "elapsed_seconds": records[0]["elapsed_seconds"],
            "timestamp": records[0]["timestamp"],
        }
    ]


def test_training_logger_ignores_unknown_fields_on_known_event(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    log = TrainingLogger("demo-project")
    Path(log.path).write_text(
        '{"event":"start","project":"demo-project","elapsed_seconds":0.1,"timestamp":"2026-03-20T00:00:00+00:00","iterations":1,"corpus":"c","model":"m","unexpected":"field"}\n'
    )

    events = TrainingLogger.load("demo-project")

    assert len(events) == 1
    assert isinstance(events[0], TrainingStarted)
    assert events[0].model == "m"
