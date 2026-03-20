"""Service helpers for the training live monitor."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from konash.training.events import (
    OaplEpochLogged,
    TrainingCompleted,
    TrainingErrorLogged,
    TrainingStarted,
    TrainingEvent,
    ValueModelCompleted,
)
from konash.training.logger import TrainingLogger


def load_training_events_for_latest_run(project: str) -> list[TrainingEvent]:
    """Return only the events from the latest run segment for a project."""
    try:
        events = TrainingLogger.load(project)
    except Exception:
        return []
    start_indexes = [idx for idx, event in enumerate(events) if isinstance(event, TrainingStarted)]
    if start_indexes:
        return events[start_indexes[-1]:]
    return events


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    event: str
    title: str
    detail: str
    timestamp: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "title": self.title,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class LiveTrainingMetrics:
    model: str
    corpus: str
    iterations: int | None
    total_seconds: float
    latest_loss: float | None
    latest_entropy: float | None
    learning_rate: float | None
    num_groups: int | None
    num_rollouts: int | None
    value_model_loss: float | None
    avg_epoch_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "corpus": self.corpus,
            "iterations": self.iterations,
            "total_seconds": self.total_seconds,
            "latest_loss": self.latest_loss,
            "latest_entropy": self.latest_entropy,
            "learning_rate": self.learning_rate,
            "num_groups": self.num_groups,
            "num_rollouts": self.num_rollouts,
            "value_model_loss": self.value_model_loss,
            "avg_epoch_seconds": self.avg_epoch_seconds,
        }


@dataclass(frozen=True, slots=True)
class LiveTrainingCharts:
    loss: list[dict[str, Any]]
    entropy: list[dict[str, Any]]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"loss": self.loss, "entropy": self.entropy}


@dataclass(frozen=True, slots=True)
class LiveTrainingSummary:
    project: str
    status: str
    stage: str
    start: TrainingStarted | None
    latest_oapl: OaplEpochLogged | None
    latest_value_model: ValueModelCompleted | None
    summary: LiveTrainingMetrics
    charts: LiveTrainingCharts
    timeline: list[TimelineEntry]
    has_trainer_events: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "status": self.status,
            "stage": self.stage,
            "start": self.start.to_record() if self.start else {},
            "latest_oapl": self.latest_oapl.to_record() if self.latest_oapl else {},
            "latest_value_model": self.latest_value_model.to_record() if self.latest_value_model else {},
            "summary": self.summary.to_dict(),
            "charts": self.charts.to_dict(),
            "timeline": [entry.to_dict() for entry in self.timeline],
            "has_trainer_events": self.has_trainer_events,
        }


@dataclass(frozen=True, slots=True)
class LiveTrainingProjectOption:
    project: str
    status: str
    stage: str
    model: str
    latest_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "status": self.status,
            "stage": self.stage,
            "model": self.model,
            "latest_loss": self.latest_loss,
        }


def build_live_training_summary(project: str) -> LiveTrainingSummary:
    """Build the typed summary used by /training/live."""
    events = load_training_events_for_latest_run(project)
    trainer_events = [
        event for event in events
        if isinstance(event, (TrainingStarted, OaplEpochLogged, ValueModelCompleted, TrainingErrorLogged, TrainingCompleted))
    ]
    start = next((event for event in trainer_events if isinstance(event, TrainingStarted)), None)
    complete = next((event for event in reversed(trainer_events) if isinstance(event, TrainingCompleted)), None)
    error = next((event for event in reversed(trainer_events) if isinstance(event, TrainingErrorLogged)), None)
    oapl_events = [event for event in trainer_events if isinstance(event, OaplEpochLogged)]
    value_events = [event for event in trainer_events if isinstance(event, ValueModelCompleted)]
    latest_oapl = oapl_events[-1] if oapl_events else None
    latest_value = value_events[-1] if value_events else None
    last_event = trainer_events[-1] if trainer_events else None

    if error:
        status = "error"
    elif complete:
        status = "complete"
    elif latest_value or latest_oapl:
        status = "running"
    else:
        status = "idle"

    stage = "Awaiting trainer"
    if latest_value and (not latest_oapl or latest_value.elapsed_seconds >= latest_oapl.elapsed_seconds):
        stage = "Value Model"
    elif latest_oapl:
        stage = "OAPL Policy Training"

    if complete:
        total_seconds = float(complete.total_seconds or 0)
    elif last_event:
        total_seconds = float(last_event.elapsed_seconds or 0)
    else:
        total_seconds = 0.0

    loss_series = [
        {"x": idx + 1, "y": float(event.loss), "label": f"Epoch {event.epoch}"}
        for idx, event in enumerate(oapl_events)
        if event.loss is not None
    ]
    entropy_series = [
        {"x": idx + 1, "y": float(event.entropy), "label": f"Epoch {event.epoch}"}
        for idx, event in enumerate(oapl_events)
        if event.entropy is not None
    ]

    timeline: list[TimelineEntry] = []
    for event in trainer_events:
        if isinstance(event, TrainingStarted):
            timeline.append(TimelineEntry(
                event=event.event_name,
                title="Training started",
                detail=event.model or event.corpus,
                timestamp=event.timestamp,
                elapsed_seconds=float(event.elapsed_seconds or 0),
            ))
        elif isinstance(event, OaplEpochLogged):
            timeline.append(TimelineEntry(
                event=event.event_name,
                title=f"OAPL epoch {event.epoch}",
                detail=f"Loss {float(event.loss):.4f} · {event.num_rollouts} rollouts",
                timestamp=event.timestamp,
                elapsed_seconds=float(event.elapsed_seconds or 0),
            ))
        elif isinstance(event, ValueModelCompleted):
            timeline.append(TimelineEntry(
                event=event.event_name,
                title="Value model complete",
                detail=f"Final loss {float(event.final_loss):.4f} · {event.epochs} epochs",
                timestamp=event.timestamp,
                elapsed_seconds=float(event.elapsed_seconds or 0),
            ))
        elif isinstance(event, TrainingCompleted):
            timeline.append(TimelineEntry(
                event=event.event_name,
                title="Training complete",
                detail="",
                timestamp=event.timestamp,
                elapsed_seconds=float(event.elapsed_seconds or 0),
            ))
        elif isinstance(event, TrainingErrorLogged):
            timeline.append(TimelineEntry(
                event=event.event_name,
                title="Training error",
                detail=event.message,
                timestamp=event.timestamp,
                elapsed_seconds=float(event.elapsed_seconds or 0),
            ))

    return LiveTrainingSummary(
        project=project,
        status=status,
        stage=stage,
        start=start,
        latest_oapl=latest_oapl,
        latest_value_model=latest_value,
        summary=LiveTrainingMetrics(
            model=start.model if start else "—",
            corpus=start.corpus if start else "—",
            iterations=complete.iterations if complete else (start.iterations if start else None),
            total_seconds=total_seconds,
            latest_loss=latest_oapl.loss if latest_oapl else None,
            latest_entropy=latest_oapl.entropy if latest_oapl else None,
            learning_rate=latest_oapl.learning_rate if latest_oapl else None,
            num_groups=latest_oapl.num_groups if latest_oapl else None,
            num_rollouts=latest_oapl.num_rollouts if latest_oapl else None,
            value_model_loss=latest_value.final_loss if latest_value else None,
            avg_epoch_seconds=mean([float(event.duration_seconds or 0) for event in oapl_events]) if oapl_events else None,
        ),
        charts=LiveTrainingCharts(loss=loss_series, entropy=entropy_series),
        timeline=timeline,
        has_trainer_events=bool(oapl_events or value_events),
    )


def list_live_training_projects(demo_project: str) -> list[LiveTrainingProjectOption]:
    """List projects relevant to the live monitor."""
    projects: list[LiveTrainingProjectOption] = []
    all_projects = list(TrainingLogger.list_projects())
    if demo_project not in all_projects:
        all_projects.append(demo_project)
    for project in all_projects:
        summary = build_live_training_summary(project)
        if project == demo_project or summary.has_trainer_events or summary.status != "idle":
            projects.append(LiveTrainingProjectOption(
                project=project,
                status=summary.status,
                stage="Interactive demo" if project == demo_project and summary.status == "idle" else summary.stage,
                model=summary.summary.model if summary.summary.model != "—" else "Demo trainer feed",
                latest_loss=summary.summary.latest_loss,
            ))
    return projects
