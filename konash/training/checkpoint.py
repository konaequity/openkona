"""Training pipeline checkpoints for crash recovery and resume.

Saves intermediate results after each phase (synthesis, dedup, rollouts)
so interrupted training can resume without re-doing completed work.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Optional


class Phase(IntEnum):
    """Training pipeline phases, in execution order."""
    SYNTHESIS = 1
    DEDUP = 2
    ROLLOUTS = 3
    OAPL = 4


_FILENAMES = {
    Phase.SYNTHESIS: "stage1_synthesis.json",
    Phase.DEDUP: "stage1_deduped.json",
    Phase.ROLLOUTS: "stage2_rollouts.json",
    Phase.OAPL: "stage3_oapl.json",
}


def checkpoint_dir(project_checkpoint_dir: str, iteration: int) -> str:
    """Return the checkpoint directory for a specific iteration."""
    d = os.path.join(project_checkpoint_dir, "pipeline_state", f"iter{iteration}")
    os.makedirs(d, exist_ok=True)
    return d


def save(
    project_checkpoint_dir: str,
    iteration: int,
    phase: Phase,
    data: Any,
) -> str:
    """Atomically save checkpoint data for a phase.

    Returns the path to the saved file.
    """
    d = checkpoint_dir(project_checkpoint_dir, iteration)
    filename = _FILENAMES[phase]
    target = os.path.join(d, filename)

    payload = {
        "version": 1,
        "phase": phase.name.lower(),
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    # Atomic write: write to temp file then rename
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, default=str)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return target


def load(
    project_checkpoint_dir: str,
    iteration: int,
    phase: Phase,
) -> Optional[Any]:
    """Load checkpoint data for a phase, or None if not found."""
    d = checkpoint_dir(project_checkpoint_dir, iteration)
    filename = _FILENAMES[phase]
    target = os.path.join(d, filename)

    if not os.path.exists(target):
        return None

    with open(target) as f:
        payload = json.load(f)

    return payload.get("data")


def find_latest_phase(
    project_checkpoint_dir: str,
    iteration: int,
) -> Optional[Phase]:
    """Find the most advanced completed phase for an iteration."""
    d = checkpoint_dir(project_checkpoint_dir, iteration)
    latest = None
    for phase in Phase:
        filename = _FILENAMES[phase]
        if os.path.exists(os.path.join(d, filename)):
            latest = phase
    return latest


def save_synthesis_incremental(
    project_checkpoint_dir: str,
    iteration: int,
    examples: list[dict],
    calls_completed: int,
    total_calls: int,
) -> str:
    """Save incremental synthesis progress (every N calls)."""
    return save(
        project_checkpoint_dir, iteration, Phase.SYNTHESIS,
        {
            "examples": examples,
            "calls_completed": calls_completed,
            "total_calls": total_calls,
        },
    )


def load_synthesis_incremental(
    project_checkpoint_dir: str,
    iteration: int,
) -> Optional[dict]:
    """Load incremental synthesis checkpoint."""
    data = load(project_checkpoint_dir, iteration, Phase.SYNTHESIS)
    if data and isinstance(data, dict) and "examples" in data:
        return data
    return None
