"""Resume helpers for staged training artifacts."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Any


def _json_examples_to_synthetic(items) -> list:
    """Convert JSON question/answer records into SyntheticExample objects."""
    from konash.synthesis.qa import SyntheticExample

    examples = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer", "")
        if isinstance(question, str) and question.strip():
            examples.append(
                SyntheticExample(question=question.strip(), answer=str(answer).strip())
            )
    return examples


def resolve_stage1_resume_path(path: str) -> str:
    """Resolve a stage-one resume path to a concrete JSON file."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "stage1_deduped.json"),
            os.path.join(path, "stage1_synthesis.json"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"Could not find stage-1 checkpoint under {path}")


def load_stage1_resume_examples(path: str):
    """Load saved stage-1 examples from synthesis or dedup checkpoints."""
    resolved = resolve_stage1_resume_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)

    phase = ""
    data = payload
    if isinstance(payload, dict):
        phase = str(payload.get("phase", "")).lower()
        data = payload.get("data", [])
    examples = _json_examples_to_synthetic(data)
    if not phase:
        if "dedup" in os.path.basename(resolved):
            phase = "dedup"
        else:
            phase = "synthesis"
    return resolved, phase, examples


def _serialize_rollout_groups(groups) -> list[dict]:
    """Serialize filtered rollout groups for stage-two checkpointing."""
    data = []
    for group in groups or []:
        data.append({
            "question": getattr(group, "prompt", ""),
            "prompt": getattr(group, "prompt", ""),
            "reference_answer": getattr(group, "reference_answer", ""),
            "rollouts": [
                {
                    "steps": getattr(rollout, "steps", []),
                    "final_answer": getattr(rollout, "final_answer", ""),
                    "passed": getattr(rollout, "passed", False),
                }
                for rollout in getattr(group, "rollouts", [])
            ],
        })
    return data


def save_stage2_artifacts(output_dir: str, iteration: int, groups, rollout_dicts) -> str:
    """Persist filtered stage-two groups early so training can resume later."""
    from konash.training import checkpoint as ckpt
    from konash.training.checkpoint import Phase

    iter_dir = os.path.join(output_dir, f"iter{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    groups_payload = {
        "version": 1,
        "phase": "rollouts",
        "iteration": iteration,
        "groups": _serialize_rollout_groups(groups),
    }
    stage2_path = os.path.join(iter_dir, "stage2_rollouts.json")
    with open(stage2_path, "w", encoding="utf-8") as f:
        json.dump(groups_payload, f, indent=2, default=str)

    rollouts_path = os.path.join(iter_dir, "rollouts.json")
    with open(rollouts_path, "w", encoding="utf-8") as f:
        json.dump(rollout_dicts, f, indent=2, default=str)

    ckpt.save(
        output_dir,
        iteration,
        Phase.ROLLOUTS,
        {"groups": groups_payload["groups"], "rollout_dicts": rollout_dicts},
    )
    return rollouts_path


def resolve_rollouts_artifact_path(path: str) -> str:
    """Resolve a stage-two resume path to a saved rollout artifact."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        direct_candidates = [
            os.path.join(path, "rollouts.json"),
            os.path.join(path, "stage2_rollouts.json"),
            os.path.join(path, "stage3_results.json"),
        ]
        for candidate in direct_candidates:
            if os.path.exists(candidate):
                return candidate

        iter_dirs = []
        for name in os.listdir(path):
            if not name.startswith("iter"):
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                iter_dirs.append(full)
        for iter_dir in sorted(iter_dirs, reverse=True):
            for filename in ("rollouts.json", "stage2_rollouts.json"):
                candidate = os.path.join(iter_dir, filename)
                if os.path.exists(candidate):
                    return candidate

        pipeline_dirs = []
        ps_root = os.path.join(path, "pipeline_state")
        if os.path.isdir(ps_root):
            for name in os.listdir(ps_root):
                full = os.path.join(ps_root, name)
                if os.path.isdir(full):
                    pipeline_dirs.append(full)
        for iter_dir in sorted(pipeline_dirs, reverse=True):
            candidate = os.path.join(iter_dir, "stage2_rollouts.json")
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(f"Could not find rollout artifact under {path}")


def normalize_stage_resume_args(args: Namespace) -> Namespace:
    """Normalize resume/skip aliases onto the existing training entrypoints."""
    if args.train_from_rollouts and not args.rollouts:
        args.rollouts = args.train_from_rollouts
    if args.resume_stage2_from and not args.rollouts:
        args.rollouts = resolve_rollouts_artifact_path(args.resume_stage2_from)
    if args.skip_rollouts and not args.rollouts:
        raise SystemExit(
            "ERROR: --skip-rollouts requires --resume-stage2-from or --train-from-rollouts."
        )
    if args.skip_synthesis and not args.resume_stage1_from:
        raise SystemExit(
            "ERROR: --skip-synthesis requires --resume-stage1-from."
        )
    return args

