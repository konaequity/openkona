"""Training project identity, resume assessment, and project state helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


LEGACY_DEFAULT_PROJECT = "default"


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    cleaned = []
    previous_dash = False
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    result = "".join(cleaned).strip("-")
    return result or "run"


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _projects_root(projects_dir: str | None = None) -> Path:
    return Path(projects_dir or os.path.expanduser("~/.konash/projects"))


def project_dir(project: str, projects_dir: str | None = None) -> Path:
    return _projects_root(projects_dir) / project


def _project_manifest_path(project: str, projects_dir: str | None = None) -> Path:
    return project_dir(project, projects_dir) / "project.json"


def _active_run_path(project: str, projects_dir: str | None = None) -> Path:
    return project_dir(project, projects_dir) / "active_run.json"


def _training_meta_path(project: str, projects_dir: str | None = None) -> Path:
    return project_dir(project, projects_dir) / "checkpoints" / "training_meta.json"


@dataclass(frozen=True, slots=True)
class CorpusSourceSpec:
    source_id: str
    display_name: str
    path: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorpusSourceSpec":
        return cls(
            source_id=str(payload.get("source_id", "")),
            display_name=str(payload.get("display_name", "")),
            path=str(payload.get("path", "")),
            fingerprint=str(payload.get("fingerprint", "")),
        )


@dataclass(frozen=True, slots=True)
class TrainingDatasetSpec:
    sources: tuple[CorpusSourceSpec, ...]
    mixing_strategy: str = "single"
    weights: tuple[float, ...] | None = None

    def normalized_identity(self) -> dict[str, Any]:
        source_records = [source.to_dict() for source in self.sources]
        if self.weights is None:
            source_records = sorted(
                source_records,
                key=lambda item: (item["source_id"], item["path"], item["fingerprint"]),
            )
        return {
            "sources": source_records,
            "mixing_strategy": self.mixing_strategy,
            "weights": list(self.weights) if self.weights is not None else None,
        }

    def identity_key(self) -> str:
        return _hash_text(json.dumps(self.normalized_identity(), sort_keys=True))

    def display_label(self) -> str:
        labels = [source.display_name for source in self.sources]
        return " + ".join(labels)

    def slug_label(self) -> str:
        labels = [source.source_id for source in self.sources]
        return "-".join(_slugify(label) for label in labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "mixing_strategy": self.mixing_strategy,
            "weights": list(self.weights) if self.weights is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingDatasetSpec":
        weights = payload.get("weights")
        return cls(
            sources=tuple(CorpusSourceSpec.from_dict(item) for item in payload.get("sources", [])),
            mixing_strategy=str(payload.get("mixing_strategy", "single")),
            weights=tuple(float(weight) for weight in weights) if weights is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    synthesis_backend: str
    iterations: int
    synthesis_calls: int
    rollouts_per_example: int
    rollout_max_steps: int

    def resume_key(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRunConfig":
        return cls(
            synthesis_backend=str(payload.get("synthesis_backend", "auto")),
            iterations=int(payload.get("iterations", 1)),
            synthesis_calls=int(payload.get("synthesis_calls", 1)),
            rollouts_per_example=int(payload.get("rollouts_per_example", 1)),
            rollout_max_steps=int(payload.get("rollout_max_steps", 1)),
        )


@dataclass(frozen=True, slots=True)
class TrainingProjectManifest:
    project: str
    display_name: str
    created_at: str
    base_model: str
    dataset_spec: TrainingDatasetSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "project": self.project,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "dataset_spec": self.dataset_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingProjectManifest":
        return cls(
            project=str(payload.get("project", "")),
            display_name=str(payload.get("display_name", "")),
            created_at=str(payload.get("created_at", _utc_timestamp())),
            base_model=str(payload.get("base_model", "")),
            dataset_spec=TrainingDatasetSpec.from_dict(payload.get("dataset_spec", {})),
        )


@dataclass(frozen=True, slots=True)
class ActiveTrainingRun:
    run_id: str
    status: str
    started_at: str
    updated_at: str
    base_model: str
    dataset_spec: TrainingDatasetSpec
    config: TrainingRunConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "base_model": self.base_model,
            "dataset_spec": self.dataset_spec.to_dict(),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActiveTrainingRun":
        return cls(
            run_id=str(payload.get("run_id", "")),
            status=str(payload.get("status", "unknown")),
            started_at=str(payload.get("started_at", _utc_timestamp())),
            updated_at=str(payload.get("updated_at", _utc_timestamp())),
            base_model=str(payload.get("base_model", "")),
            dataset_spec=TrainingDatasetSpec.from_dict(payload.get("dataset_spec", {})),
            config=TrainingRunConfig.from_dict(payload.get("config", {})),
        )


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    has_checkpoint_state: bool
    latest_iteration: int | None
    latest_phase: str | None


@dataclass(frozen=True, slots=True)
class ProjectReuseAssessment:
    project: str
    project_exists: bool
    compatible_project: bool
    resume_available: bool
    has_completed_training: bool
    reason: str
    active_run: ActiveTrainingRun | None
    checkpoint: CheckpointSummary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fingerprint_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    prebuilt = resolved / "prebuilt_index.npz"
    target = prebuilt if prebuilt.exists() else resolved
    try:
        stat = target.stat()
        summary = {
            "path": str(resolved),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "is_dir": target.is_dir(),
        }
    except OSError:
        summary = {"path": str(resolved), "missing": True}
    return _hash_text(json.dumps(summary, sort_keys=True))


def build_dataset_spec(
    corpora: Sequence[str | os.PathLike[str]],
    *,
    aliases: dict[str, tuple[str, str]] | None = None,
    mixing_strategy: str = "single",
    weights: Sequence[float] | None = None,
) -> TrainingDatasetSpec:
    alias_map = aliases or {}
    sources: list[CorpusSourceSpec] = []
    for corpus in corpora:
        resolved = str(Path(corpus).expanduser().resolve())
        source_id, display_name = alias_map.get(
            resolved,
            (_slugify(Path(resolved).name), Path(resolved).name or resolved),
        )
        sources.append(CorpusSourceSpec(
            source_id=source_id,
            display_name=display_name,
            path=resolved,
            fingerprint=_fingerprint_path(Path(resolved)),
        ))
    return TrainingDatasetSpec(
        sources=tuple(sources),
        mixing_strategy=mixing_strategy,
        weights=tuple(weights) if weights is not None else None,
    )


def suggest_project_name(
    base_model: str,
    dataset_spec: TrainingDatasetSpec,
    *,
    projects_dir: str | None = None,
    ensure_unique: bool = False,
) -> str:
    model_slug = _slugify(base_model.split("/", 1)[-1])
    base = f"{dataset_spec.slug_label()}-{model_slug}"
    if not ensure_unique:
        return base

    root = _projects_root(projects_dir)
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def archive_legacy_default_project(projects_dir: str | None = None) -> Path | None:
    root = _projects_root(projects_dir)
    legacy = root / LEGACY_DEFAULT_PROJECT
    if not legacy.exists():
        return None
    target = root / f"_legacy_default_{time.strftime('%Y%m%d-%H%M%S')}"
    suffix = 2
    while target.exists():
        target = root / f"_legacy_default_{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
        suffix += 1
    os.replace(legacy, target)
    return target


def load_project_manifest(
    project: str,
    *,
    projects_dir: str | None = None,
) -> TrainingProjectManifest | None:
    payload = _read_json(_project_manifest_path(project, projects_dir))
    if payload is None:
        return None
    return TrainingProjectManifest.from_dict(payload)


def save_project_manifest(
    manifest: TrainingProjectManifest,
    *,
    projects_dir: str | None = None,
) -> None:
    _write_json(_project_manifest_path(manifest.project, projects_dir), manifest.to_dict())


def load_active_run(
    project: str,
    *,
    projects_dir: str | None = None,
) -> ActiveTrainingRun | None:
    payload = _read_json(_active_run_path(project, projects_dir))
    if payload is None:
        return None
    return ActiveTrainingRun.from_dict(payload)


def save_active_run(
    project: str,
    run: ActiveTrainingRun,
    *,
    projects_dir: str | None = None,
) -> None:
    _write_json(_active_run_path(project, projects_dir), run.to_dict())


def latest_checkpoint_summary(
    project: str,
    *,
    projects_dir: str | None = None,
) -> CheckpointSummary:
    pipeline_dir = project_dir(project, projects_dir) / "checkpoints" / "pipeline_state"
    if not pipeline_dir.exists():
        return CheckpointSummary(False, None, None)

    latest_iteration: int | None = None
    latest_phase: str | None = None
    phase_order = [
        ("stage1_synthesis.json", "synthesis"),
        ("stage1_deduped.json", "dedup"),
        ("stage2_rollouts.json", "rollouts"),
        ("stage3_oapl.json", "oapl"),
    ]
    for iter_dir in sorted(pipeline_dir.glob("iter*")):
        try:
            iteration = int(iter_dir.name.replace("iter", ""))
        except ValueError:
            continue
        phase_name = None
        for filename, phase in phase_order:
            if (iter_dir / filename).exists():
                phase_name = phase
        if phase_name is not None and (latest_iteration is None or iteration >= latest_iteration):
            latest_iteration = iteration
            latest_phase = phase_name

    return CheckpointSummary(latest_iteration is not None, latest_iteration, latest_phase)


def assess_project_reuse(
    *,
    project: str,
    base_model: str,
    dataset_spec: TrainingDatasetSpec,
    config: TrainingRunConfig,
    projects_dir: str | None = None,
) -> ProjectReuseAssessment:
    root = project_dir(project, projects_dir)
    if not root.exists():
        return ProjectReuseAssessment(
            project=project,
            project_exists=False,
            compatible_project=False,
            resume_available=False,
            has_completed_training=False,
            reason="new_project",
            active_run=None,
            checkpoint=CheckpointSummary(False, None, None),
        )

    manifest = load_project_manifest(project, projects_dir=projects_dir)
    checkpoint = latest_checkpoint_summary(project, projects_dir=projects_dir)
    active_run = load_active_run(project, projects_dir=projects_dir)
    has_completed_training = _training_meta_path(project, projects_dir).exists()

    if manifest is None:
        return ProjectReuseAssessment(
            project=project,
            project_exists=True,
            compatible_project=False,
            resume_available=False,
            has_completed_training=has_completed_training,
            reason="legacy_project_state",
            active_run=active_run,
            checkpoint=checkpoint,
        )

    compatible_project = (
        manifest.base_model == base_model
        and manifest.dataset_spec.identity_key() == dataset_spec.identity_key()
    )
    if not compatible_project:
        return ProjectReuseAssessment(
            project=project,
            project_exists=True,
            compatible_project=False,
            resume_available=False,
            has_completed_training=has_completed_training,
            reason="project_identity_mismatch",
            active_run=active_run,
            checkpoint=checkpoint,
        )

    if (
        active_run is not None
        and active_run.status in {"running", "failed", "interrupted"}
        and checkpoint.has_checkpoint_state
        and active_run.config.resume_key() == config.resume_key()
    ):
        return ProjectReuseAssessment(
            project=project,
            project_exists=True,
            compatible_project=True,
            resume_available=True,
            has_completed_training=has_completed_training,
            reason="resume_available",
            active_run=active_run,
            checkpoint=checkpoint,
        )

    reason = "completed_project" if has_completed_training else "fresh_start_required"
    if active_run is not None and active_run.config.resume_key() != config.resume_key():
        reason = "run_config_mismatch"
    return ProjectReuseAssessment(
        project=project,
        project_exists=True,
        compatible_project=True,
        resume_available=False,
        has_completed_training=has_completed_training,
        reason=reason,
        active_run=active_run,
        checkpoint=checkpoint,
    )


def begin_training_run(
    *,
    project: str,
    display_name: str,
    base_model: str,
    dataset_spec: TrainingDatasetSpec,
    config: TrainingRunConfig,
    projects_dir: str | None = None,
) -> ActiveTrainingRun:
    manifest = load_project_manifest(project, projects_dir=projects_dir)
    if manifest is None:
        manifest = TrainingProjectManifest(
            project=project,
            display_name=display_name,
            created_at=_utc_timestamp(),
            base_model=base_model,
            dataset_spec=dataset_spec,
        )
        save_project_manifest(manifest, projects_dir=projects_dir)

    run = ActiveTrainingRun(
        run_id=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        status="running",
        started_at=_utc_timestamp(),
        updated_at=_utc_timestamp(),
        base_model=base_model,
        dataset_spec=dataset_spec,
        config=config,
    )
    save_active_run(project, run, projects_dir=projects_dir)
    return run


def mark_training_run_status(
    project: str,
    *,
    status: str,
    projects_dir: str | None = None,
) -> None:
    existing = load_active_run(project, projects_dir=projects_dir)
    if existing is None:
        return
    save_active_run(
        project,
        ActiveTrainingRun(
            run_id=existing.run_id,
            status=status,
            started_at=existing.started_at,
            updated_at=_utc_timestamp(),
            base_model=existing.base_model,
            dataset_spec=existing.dataset_spec,
            config=existing.config,
        ),
        projects_dir=projects_dir,
    )


def archive_project_run_state(
    project: str,
    *,
    projects_dir: str | None = None,
) -> Path | None:
    root = project_dir(project, projects_dir)
    if not root.exists():
        return None

    archive_root = root / "archives"
    active = load_active_run(project, projects_dir=projects_dir)
    label = active.run_id if active is not None else time.strftime("%Y%m%d-%H%M%S")
    target = archive_root / label
    suffix = 2
    while target.exists():
        target = archive_root / f"{label}-{suffix}"
        suffix += 1
    target.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for name in ("checkpoints", "training.jsonl", "training_debug.log", "active_run.json"):
        source = root / name
        if source.exists():
            shutil.move(str(source), str(target / name))
            moved_any = True
    return target if moved_any else None
