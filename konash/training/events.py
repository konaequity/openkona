"""Typed training event models for the JSONL training log."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional, TypeAlias


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BaseTrainingEvent:
    """Base event envelope shared by all training log entries."""

    project: str
    elapsed_seconds: float
    timestamp: str = field(default_factory=now_utc_iso)
    event_name: ClassVar[str] = "unknown"

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event"] = self.event_name
        return payload


@dataclass(slots=True)
class UnknownTrainingEvent(BaseTrainingEvent):
    event_name: ClassVar[str] = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(slots=True)
class TrainingStarted(BaseTrainingEvent):
    event_name: ClassVar[str] = "start"
    iterations: int = 0
    corpus: str = ""
    model: str = ""


@dataclass(slots=True)
class SynthesisCompleted(BaseTrainingEvent):
    event_name: ClassVar[str] = "synthesis"
    iteration: int = 0
    calls_completed: int = 0
    calls_total: int = 0
    raw_pairs: int = 0
    deduped: int = 0
    dedup_rate: float = 0.0
    duration_seconds: float = 0.0


@dataclass(slots=True)
class RolloutsCompleted(BaseTrainingEvent):
    event_name: ClassVar[str] = "rollouts"
    iteration: int = 0
    examples_in: int = 0
    rollouts_total: int = 0
    examples_after_filter: int = 0
    filter_rate: float = 0.0
    avg_pass_rate: float = 0.0
    duration_seconds: float = 0.0


@dataclass(slots=True)
class OaplEpochLogged(BaseTrainingEvent):
    event_name: ClassVar[str] = "oapl"
    iteration: int = 0
    epoch: int = 1
    loss: float = 0.0
    kl: float = 0.0
    entropy: float = 0.0
    num_groups: int = 0
    num_rollouts: int = 0
    learning_rate: Optional[float] = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class ValueModelCompleted(BaseTrainingEvent):
    event_name: ClassVar[str] = "value_model"
    final_loss: float = 0.0
    epochs: int = 0
    duration_seconds: float = 0.0


@dataclass(slots=True)
class PhaseTimingLogged(BaseTrainingEvent):
    event_name: ClassVar[str] = "phase"
    iteration: int = 0
    phase: str = ""
    duration_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = BaseTrainingEvent.to_record(self)
        extra = record.pop("extra", {})
        record.update(extra)
        return record


@dataclass(slots=True)
class TrainingCompleted(BaseTrainingEvent):
    event_name: ClassVar[str] = "complete"
    iterations: int = 0
    total_seconds: float = 0.0
    stats: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class RolloutProgressLogged(BaseTrainingEvent):
    event_name: ClassVar[str] = "rollout_progress"
    iteration: int = 0
    completed: int = 0
    total: int = 0
    pct: float = 0.0


@dataclass(slots=True)
class FilterSummaryLogged(BaseTrainingEvent):
    event_name: ClassVar[str] = "filter_summary"
    iteration: int = 0
    phase: str = ""
    input_count: int = 0
    output_count: int = 0
    reject_count: int = 0


@dataclass(slots=True)
class TrainingErrorLogged(BaseTrainingEvent):
    event_name: ClassVar[str] = "error"
    message: str = ""
    phase: str = ""
    iteration: int = 0


TrainingEvent: TypeAlias = (
    TrainingStarted
    | SynthesisCompleted
    | RolloutsCompleted
    | OaplEpochLogged
    | ValueModelCompleted
    | PhaseTimingLogged
    | TrainingCompleted
    | RolloutProgressLogged
    | FilterSummaryLogged
    | TrainingErrorLogged
    | UnknownTrainingEvent
)


_EVENT_TYPES: dict[str, type[BaseTrainingEvent]] = {
    TrainingStarted.event_name: TrainingStarted,
    SynthesisCompleted.event_name: SynthesisCompleted,
    RolloutsCompleted.event_name: RolloutsCompleted,
    OaplEpochLogged.event_name: OaplEpochLogged,
    ValueModelCompleted.event_name: ValueModelCompleted,
    PhaseTimingLogged.event_name: PhaseTimingLogged,
    TrainingCompleted.event_name: TrainingCompleted,
    RolloutProgressLogged.event_name: RolloutProgressLogged,
    FilterSummaryLogged.event_name: FilterSummaryLogged,
    TrainingErrorLogged.event_name: TrainingErrorLogged,
}


def _filter_event_payload(
    event_type: type[BaseTrainingEvent],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    field_names = {f.name for f in fields(event_type)}
    known = {key: value for key, value in payload.items() if key in field_names}
    extra = {key: value for key, value in payload.items() if key not in field_names}
    return known, extra


def training_event_from_record(record: dict[str, Any]) -> TrainingEvent:
    event_name = str(record.get("event", "unknown"))
    event_type = _EVENT_TYPES.get(event_name)
    if event_type is None:
        return UnknownTrainingEvent(
            project=str(record.get("project", "default")),
            elapsed_seconds=float(record.get("elapsed_seconds", 0.0) or 0.0),
            timestamp=str(record.get("timestamp", now_utc_iso())),
            raw=dict(record),
        )

    payload = dict(record)
    payload.pop("event", None)
    known_payload, extra_payload = _filter_event_payload(event_type, payload)

    if event_type is PhaseTimingLogged:
        known_payload["extra"] = extra_payload

    try:
        return event_type(**known_payload)
    except TypeError:
        return UnknownTrainingEvent(
            project=str(record.get("project", "default")),
            elapsed_seconds=float(record.get("elapsed_seconds", 0.0) or 0.0),
            timestamp=str(record.get("timestamp", now_utc_iso())),
            raw=dict(record),
        )
