"""Lightweight training logger — JSONL event stream.

Writes one JSON line per event to ``~/.konash/projects/<project>/training.jsonl``.
Append-only, crash-safe (each line is a complete JSON object).

Events are auto-discovered by the trace viewer at ``/training``.

Usage::

    from konash.training.logger import TrainingLogger

    log = TrainingLogger("my-project")
    log.synthesis(iteration=1, calls=50, raw_pairs=400, deduped=312)
    log.rollouts(iteration=1, examples=312, rollouts=2496, filtered=128, pass_rate=0.41)
    log.oapl(iteration=1, epoch=1, loss=0.234, kl=0.145, num_groups=128, num_rollouts=512)
    log.value_model(loss=0.102, epochs=3)
    log.phase(iteration=1, phase="synthesis", duration_seconds=3600)
    log.complete(iterations=2, total_seconds=7200)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional


def configure_file_logging(project: str, level: str = "DEBUG") -> str:
    """Set up file-based logging for all ``konash.*`` modules.

    Creates a ``RotatingFileHandler`` at
    ``~/.konash/projects/<project>/training_debug.log`` and attaches it
    to the ``"konash"`` logger.  All modules that call
    ``logging.getLogger(__name__)`` inherit this handler automatically.

    Safe to call multiple times — skips if a file handler is already
    attached.

    Returns the path to the log file.
    """
    log_dir = os.path.expanduser(f"~/.konash/projects/{project}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_debug.log")

    konash_logger = logging.getLogger("konash")

    # Don't add duplicate handlers on repeated calls
    if any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(log_path)
        for h in konash_logger.handlers
    ):
        return log_path

    handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    handler.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    konash_logger.addHandler(handler)
    konash_logger.setLevel(logging.DEBUG)

    return log_path


class TrainingLogger:
    """Append-only JSONL logger for training runs."""

    def __init__(self, project: str = "default"):
        self.project = project
        self._dir = os.path.expanduser(f"~/.konash/projects/{project}")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, "training.jsonl")
        self._start_time = time.monotonic()

    def _write(self, event: str, data: dict[str, Any]) -> None:
        """Append one JSON line to the log file."""
        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
            "project": self.project,
            **data,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def start(self, *, iterations: int, corpus: str, model: str) -> None:
        """Log training start."""
        self._write("start", {
            "iterations": iterations,
            "corpus": corpus,
            "model": model,
        })

    def synthesis(
        self,
        *,
        iteration: int,
        calls_completed: int,
        calls_total: int,
        raw_pairs: int,
        deduped: int,
        duration_seconds: float = 0,
    ) -> None:
        """Log synthesis phase completion."""
        self._write("synthesis", {
            "iteration": iteration,
            "calls_completed": calls_completed,
            "calls_total": calls_total,
            "raw_pairs": raw_pairs,
            "deduped": deduped,
            "dedup_rate": round(1 - deduped / raw_pairs, 3) if raw_pairs else 0,
            "duration_seconds": round(duration_seconds, 1),
        })

    def rollouts(
        self,
        *,
        iteration: int,
        examples: int,
        rollouts: int,
        filtered: int,
        pass_rate: float = 0,
        duration_seconds: float = 0,
    ) -> None:
        """Log rollout phase completion."""
        self._write("rollouts", {
            "iteration": iteration,
            "examples_in": examples,
            "rollouts_total": rollouts,
            "examples_after_filter": filtered,
            "filter_rate": round(1 - filtered / examples, 3) if examples else 0,
            "avg_pass_rate": round(pass_rate, 3),
            "duration_seconds": round(duration_seconds, 1),
        })

    def oapl(
        self,
        *,
        iteration: int,
        epoch: int = 1,
        loss: float,
        kl: float = 0,
        num_groups: int = 0,
        num_rollouts: int = 0,
        duration_seconds: float = 0,
    ) -> None:
        """Log one OAPL training epoch."""
        self._write("oapl", {
            "iteration": iteration,
            "epoch": epoch,
            "loss": round(loss, 6),
            "kl": round(kl, 6),
            "num_groups": num_groups,
            "num_rollouts": num_rollouts,
            "duration_seconds": round(duration_seconds, 1),
        })

    def value_model(
        self,
        *,
        loss: float,
        epochs: int = 0,
        duration_seconds: float = 0,
    ) -> None:
        """Log value model training completion."""
        self._write("value_model", {
            "final_loss": round(loss, 6),
            "epochs": epochs,
            "duration_seconds": round(duration_seconds, 1),
        })

    def phase(
        self,
        *,
        iteration: int,
        phase: str,
        duration_seconds: float,
        **extra: Any,
    ) -> None:
        """Log a generic phase timing."""
        self._write("phase", {
            "iteration": iteration,
            "phase": phase,
            "duration_seconds": round(duration_seconds, 1),
            **extra,
        })

    def complete(
        self,
        *,
        iterations: int,
        total_seconds: float,
        stats: Optional[list] = None,
    ) -> None:
        """Log training completion."""
        self._write("complete", {
            "iterations": iterations,
            "total_seconds": round(total_seconds, 1),
            "stats": stats or [],
        })

    def rollout_progress(
        self,
        *,
        iteration: int,
        completed: int,
        total: int,
        elapsed_seconds: float = 0,
    ) -> None:
        """Log incremental rollout progress (called every N completions)."""
        self._write("rollout_progress", {
            "iteration": iteration,
            "completed": completed,
            "total": total,
            "pct": round(completed / total * 100, 1) if total else 0,
            "elapsed_seconds": round(elapsed_seconds, 1),
        })

    def filter_summary(
        self,
        *,
        iteration: int,
        phase: str,
        input_count: int,
        output_count: int,
    ) -> None:
        """Log a filter phase summary (pass-rate or quality filter)."""
        self._write("filter_summary", {
            "iteration": iteration,
            "phase": phase,
            "input_count": input_count,
            "output_count": output_count,
            "reject_count": input_count - output_count,
        })

    def error(self, *, message: str, phase: str = "", iteration: int = 0) -> None:
        """Log an error."""
        self._write("error", {
            "message": message,
            "phase": phase,
            "iteration": iteration,
        })

    @property
    def path(self) -> str:
        return self._path

    @staticmethod
    def load(project: str = "default") -> list[dict]:
        """Load all log entries for a project."""
        path = os.path.expanduser(
            f"~/.konash/projects/{project}/training.jsonl"
        )
        if not os.path.exists(path):
            return []
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    @staticmethod
    def list_projects() -> list[str]:
        """List all projects that have training logs."""
        projects_dir = os.path.expanduser("~/.konash/projects")
        if not os.path.isdir(projects_dir):
            return []
        projects = []
        for name in sorted(os.listdir(projects_dir)):
            log_path = os.path.join(projects_dir, name, "training.jsonl")
            if os.path.exists(log_path):
                projects.append(name)
        return projects
