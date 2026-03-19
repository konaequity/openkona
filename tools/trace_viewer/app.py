#!/opt/anaconda3/bin/python3
"""KONASH Trace Viewer — qualitative analysis tool for comparing model search behavior.

Run:
    python tools/trace_viewer/app.py

Then open http://localhost:5050 in a browser.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Allow importing from the KONASH package when running standalone
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from jinja2 import ChoiceLoader, FileSystemLoader

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

_SHARED_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "templates")
app.jinja_loader = ChoiceLoader([
    app.jinja_loader,
    FileSystemLoader(_SHARED_TEMPLATES),
])

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Paths for auto-discovery of real rollout data
STAGE2_PATH = Path(_PROJECT_ROOT) / "stage2_training_data.json"
ARENA_DATA_DIR = Path(_PROJECT_ROOT) / "tools" / "arena" / "data"
CHECKPOINTS_DIR = Path(_PROJECT_ROOT) / "checkpoints"


# ---------------------------------------------------------------------------
# KONASH rollout -> trace viewer conversion
# ---------------------------------------------------------------------------

def classify_step(
    step: Dict[str, Any],
    step_index: int,
    total_steps: int,
    prev_queries: List[str],
) -> str:
    """Heuristic classification of a rollout step into a viewer category.

    Categories:
        grounding    — initial broad searches establishing context (first ~20%)
        discovery    — targeted searches finding new information
        exploration  — following leads, broadening search
        verification — confirming/validating found information (last ~15%)

    The heuristic uses step position, query text similarity to previous
    queries, and step metadata to decide.
    """
    # Position-based thresholds
    position_ratio = step_index / max(total_steps, 1)

    query = step.get("query", "") or ""
    thought = (step.get("thought", "") or "").lower()

    # Verification signals
    verification_keywords = [
        "confirm", "verify", "validate", "check", "double-check",
        "make sure", "ensure", "revisit", "re-check",
    ]
    if any(kw in thought for kw in verification_keywords):
        return "verification"
    if position_ratio > 0.85:
        return "verification"

    # Grounding: first steps
    if step_index == 0 or position_ratio < 0.2:
        return "grounding"

    # Discovery vs exploration: check query novelty
    if prev_queries and query:
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        max_overlap = 0.0
        for pq in prev_queries:
            pq_tokens = set(pq.lower().split())
            if pq_tokens:
                overlap = len(query_tokens & pq_tokens) / max(len(query_tokens | pq_tokens), 1)
                max_overlap = max(max_overlap, overlap)

        # High overlap with previous query -> exploration (refining)
        if max_overlap > 0.6:
            return "exploration"
        # Low overlap -> discovery (new angle)
        return "discovery"

    # Default mid-trajectory: discovery
    return "discovery"


def _compute_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def convert_konash_rollout(
    rollout_steps: List[Dict[str, Any]],
    expected_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert KONASH rollout step dicts into trace viewer step format.

    KONASH rollout steps have:
        step (int), type ("retrieval"/"reasoning"/"answer"),
        query (str), results (list of {text, docid}), thought (str)

    Returns list of trace viewer steps with category classification.
    """
    expected_doc_ids = expected_doc_ids or []
    expected_set = set(expected_doc_ids)
    viewer_steps = []
    found_so_far: set = set()
    prev_queries: List[str] = []
    total = len(rollout_steps)

    for idx, raw_step in enumerate(rollout_steps):
        step_type = raw_step.get("type", "unknown")
        queries_in_step: List[str] = []
        docs_retrieved = 0

        # Gather queries
        main_query = raw_step.get("query", "")
        if main_query:
            queries_in_step.append(main_query)

        sub = raw_step.get("sub_retrieval", {})
        if sub and sub.get("query"):
            queries_in_step.append(sub["query"])

        # Count retrieved docs
        results = raw_step.get("results", [])
        docs_retrieved += len(results) if isinstance(results, list) else 0
        if sub and sub.get("results"):
            docs_retrieved += len(sub["results"])

        # Check which expected docs were found
        all_results = list(results) if isinstance(results, list) else []
        if sub and sub.get("results"):
            all_results.extend(sub["results"])

        for r in all_results:
            if isinstance(r, dict):
                docid = r.get("docid", r.get("doc_id", ""))
                if docid and docid in expected_set:
                    found_so_far.add(docid)

        # Use step type for classification when available
        if step_type == "answer":
            category = "verification"
        elif step_type == "retrieval":
            category = classify_step(raw_step, idx, total, prev_queries)
        elif step_type == "reasoning":
            category = classify_step(raw_step, idx, total, prev_queries)
        else:
            category = classify_step(raw_step, idx, total, prev_queries)

        thought = raw_step.get("thought", "") or raw_step.get("answer", "")

        viewer_step = {
            "step_number": raw_step.get("step", idx),
            "type": step_type,
            "category": category,
            "queries": queries_in_step if queries_in_step else [],
            "num_queries": len(queries_in_step),
            "docs_retrieved": docs_retrieved,
            "expected_found": len(found_so_far),
            "total_expected": len(expected_doc_ids),
            "thought": thought,
        }
        viewer_steps.append(viewer_step)
        prev_queries.extend(queries_in_step)

    return viewer_steps


# ---------------------------------------------------------------------------
# Auto-discovery: load real rollout data from the KONASH project
# ---------------------------------------------------------------------------

def _discover_stage2_data() -> List[Dict[str, Any]]:
    """Load stage2_training_data.json and convert to trace viewer sessions.

    Groups rollouts by prompt so each unique question becomes one session
    with a single "model" containing multiple traces (one per rollout).
    """
    if not STAGE2_PATH.exists():
        return []

    try:
        with open(STAGE2_PATH) as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    # Group by prompt
    by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_prompt[entry.get("prompt", "")].append(entry)

    sessions = []
    for q_idx, (prompt, rollouts) in enumerate(by_prompt.items(), start=1):
        ref = rollouts[0].get("reference_answer", "")

        # Build traces from each rollout
        traces = []
        for r_idx, rollout in enumerate(rollouts):
            raw_steps = rollout.get("rollout_steps", [])
            viewer_steps = convert_konash_rollout(raw_steps)
            found = viewer_steps[-1]["expected_found"] if viewer_steps else 0
            traces.append({
                "trace_id": r_idx + 1,
                "coverage": rollout.get("reward", 0.0),
                "total_steps": len(viewer_steps),
                "found_count": found,
                "total_expected": 0,
                "steps": viewer_steps,
                "final_answer": rollout.get("final_answer", ""),
                "reward": rollout.get("reward", 0.0),
            })

        sessions.append({
            "query_id": q_idx,
            "question": prompt,
            "reference_answer": ref,
            "expected_documents": [],
            "source": "stage2_training_data",
            "models": [{
                "name": "Training Rollouts",
                "traces": traces,
            }],
        })

    return sessions


def _discover_arena_results() -> List[Dict[str, Any]]:
    """Load completed arena runs and convert to trace viewer sessions.

    Reads from runs.jsonl which contains full run records with steps.
    """
    if not ARENA_DATA_DIR.exists():
        return []

    runs_file = ARENA_DATA_DIR / "runs.jsonl"
    if not runs_file.exists():
        return []

    sessions = []
    try:
        with open(runs_file) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = record.get("question", "")
                if not question:
                    continue

                results = record.get("results", {})
                models = []
                for side_key in ("a", "b"):
                    side = results.get(side_key, {})
                    steps_raw = side.get("steps", [])
                    viewer_steps = _convert_arena_steps(steps_raw)
                    preset = side.get("preset", record.get(f"model_{side_key}", f"Model {side_key.upper()}"))
                    models.append({
                        "name": preset,
                        "traces": [{
                            "trace_id": 1,
                            "coverage": 0.0,
                            "total_steps": len(viewer_steps),
                            "found_count": 0,
                            "total_expected": 0,
                            "steps": viewer_steps,
                            "final_answer": side.get("final_answer", ""),
                            "total_time": side.get("total_time", 0),
                            "ttft": side.get("ttft", 0),
                        }],
                    })

                vote = record.get("vote", {})
                sessions.append({
                    "query_id": line_num,
                    "question": question,
                    "expected_documents": [],
                    "source": "arena",
                    "run_id": record.get("run_id", ""),
                    "vote_winner": vote.get("winner") if vote else None,
                    "models": models,
                })
    except OSError:
        pass

    return sessions


def _convert_arena_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert arena serialized steps to trace viewer format."""
    viewer_steps = []
    prev_queries: List[str] = []
    total = len(steps)

    for idx, s in enumerate(steps):
        step_type = s.get("type", "unknown")
        queries: List[str] = []

        if s.get("query"):
            queries.append(s["query"])
        sub = s.get("sub_retrieval", {})
        if sub and sub.get("query"):
            queries.append(sub["query"])

        docs_retrieved = s.get("num_results", 0)
        if sub:
            docs_retrieved += sub.get("num_results", 0)

        category = classify_step(s, idx, total, prev_queries)

        viewer_steps.append({
            "step_number": s.get("step", idx),
            "category": category,
            "queries": queries if queries else ["(no query)"],
            "num_queries": len(queries),
            "docs_retrieved": docs_retrieved,
            "expected_found": 0,
            "total_expected": 0,
            "thought": s.get("thought", ""),
        })
        prev_queries.extend(queries)

    return viewer_steps


def _discover_checkpoint_rollouts() -> List[Dict[str, Any]]:
    """Scan checkpoints for rollout files.

    Checks two locations:
    1. {PROJECT_ROOT}/checkpoints/ — legacy path for rollouts_cache.json
    2. ~/.konash/projects/*/checkpoints/pipeline_state/ — training pipeline outputs
    """
    sessions = []

    # 1. Legacy: rollouts_cache.json in project-root checkpoints/
    if CHECKPOINTS_DIR.exists():
        for cache_file in sorted(CHECKPOINTS_DIR.rglob("rollouts_cache.json")):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            checkpoint_name = cache_file.parent.name
            if isinstance(data, list):
                for q_idx, group in enumerate(data):
                    if isinstance(group, dict) and "prompt" in group:
                        rollouts = group.get("rollouts", [])
                        traces = []
                        for r_idx, r in enumerate(rollouts):
                            raw_steps = r.get("steps", [])
                            viewer_steps = convert_konash_rollout(raw_steps)
                            traces.append({
                                "trace_id": r_idx + 1,
                                "coverage": 1.0 if r.get("passed") else 0.0,
                                "total_steps": len(viewer_steps),
                                "found_count": 0,
                                "total_expected": 0,
                                "steps": viewer_steps,
                                "final_answer": r.get("final_answer", ""),
                            })
                        if traces:
                            sessions.append({
                                "query_id": q_idx + 1,
                                "question": group.get("prompt", ""),
                                "reference_answer": group.get("reference_answer", ""),
                                "expected_documents": [],
                                "source": f"checkpoint:{checkpoint_name}",
                                "models": [{
                                    "name": checkpoint_name,
                                    "traces": traces,
                                }],
                            })

    # 2. Training pipeline: ~/.konash/projects/*/checkpoints/pipeline_state/
    konash_projects = Path(os.path.expanduser("~/.konash/projects"))
    if konash_projects.exists():
        for project_dir in sorted(konash_projects.iterdir()):
            if not project_dir.is_dir():
                continue
            project_name = project_dir.name
            pipeline_dir = project_dir / "checkpoints" / "pipeline_state"
            if not pipeline_dir.exists():
                continue

            for iter_dir in sorted(pipeline_dir.iterdir()):
                if not iter_dir.is_dir() or not iter_dir.name.startswith("iter"):
                    continue

                # Try stage2_rollouts.json first, then rollouts_incremental.json
                for fname in ["stage2_rollouts.json", "rollouts_incremental.json"]:
                    fpath = iter_dir / fname
                    if not fpath.exists():
                        continue
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        raw_groups = data.get("data", data).get("groups", [])
                    except (json.JSONDecodeError, OSError, AttributeError):
                        continue

                    for q_idx, group in enumerate(raw_groups):
                        rollouts = group.get("rollouts", [])
                        traces = []
                        for r_idx, r in enumerate(rollouts):
                            raw_steps = r.get("steps", [])
                            viewer_steps = convert_konash_rollout(raw_steps)
                            traces.append({
                                "trace_id": r_idx + 1,
                                "coverage": 1.0 if r.get("passed") else 0.0,
                                "total_steps": len(viewer_steps),
                                "found_count": 0,
                                "total_expected": 0,
                                "steps": viewer_steps,
                                "final_answer": r.get("final_answer", ""),
                            })
                        if traces:
                            question = group.get("prompt", group.get("question", ""))
                            sessions.append({
                                "query_id": len(sessions) + 1,
                                "question": question,
                                "reference_answer": group.get("reference_answer", ""),
                                "expected_documents": [],
                                "source": f"training:{project_name}/{iter_dir.name}",
                                "models": [{
                                    "name": f"{project_name} {iter_dir.name}",
                                    "traces": traces,
                                }],
                            })
                    break  # only load one file per iteration

    return sessions



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/traces")
@app.route("/traces/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Training monitor
# ---------------------------------------------------------------------------

@app.route("/training")
@app.route("/training/")
def training_index():
    return render_template("training.html")


@app.route("/training/api/projects")
def training_projects():
    """List all projects with training logs."""
    from konash.training.logger import TrainingLogger
    return jsonify({"projects": TrainingLogger.list_projects()})


@app.route("/training/api/logs/<project>")
def training_logs(project: str):
    """Get training log events for a project."""
    from konash.training.logger import TrainingLogger
    events = TrainingLogger.load(project)
    return jsonify({"project": project, "events": events})


@app.route("/training/api/rollouts/<project>")
def training_rollouts(project: str):
    """Get rollout checkpoint data (questions, answers, steps) for a project.

    Looks for pipeline_state/iter*/stage2_rollouts.json or
    rollouts_incremental.json in the project's checkpoint directory.
    """
    import glob

    ckpt_base = os.path.expanduser(f"~/.konash/projects/{project}/checkpoints/pipeline_state")
    groups = []

    def _parse_groups(raw_groups, iter_dir, fname, kept_prompts=None):
        """Convert raw checkpoint groups to API response format."""
        parsed = []
        for g in raw_groups:
            rollouts = g.get("rollouts", [])
            passed_count = sum(1 for r in rollouts if r.get("passed"))
            prompt = g.get("prompt", g.get("question", ""))
            parsed.append({
                "question": prompt,
                "reference_answer": g.get("reference_answer", ""),
                "num_rollouts": len(rollouts),
                "passed": passed_count,
                "pass_rate": round(passed_count / len(rollouts), 2) if rollouts else 0,
                "kept": prompt in kept_prompts if kept_prompts is not None else True,
                "rollouts": [
                    {
                        "final_answer": r.get("final_answer", ""),
                        "passed": r.get("passed"),
                        "num_steps": len(r.get("steps", [])),
                        "steps": [
                            {
                                "type": s.get("type", ""),
                                "query": s.get("query", ""),
                                "thought": s.get("thought") or "",
                                "num_results": len(s.get("results", [])) if isinstance(s.get("results"), list) else 0,
                            }
                            for s in r.get("steps", [])
                        ],
                    }
                    for r in rollouts
                ],
                "iteration": os.path.basename(iter_dir),
                "source": fname,
            })
        return parsed

    for iter_dir in sorted(glob.glob(os.path.join(ckpt_base, "iter*"))):
        # First, find which prompts survived filtering (from final rollouts)
        kept_prompts = None
        final_path = os.path.join(iter_dir, "stage2_rollouts.json")
        if os.path.exists(final_path):
            try:
                with open(final_path) as f:
                    final_data = json.load(f)
                final_groups = final_data.get("data", final_data).get("groups", [])
                kept_prompts = {
                    g.get("prompt", g.get("question", ""))
                    for g in final_groups
                }
            except (json.JSONDecodeError, OSError):
                pass

        # Load all groups from incremental checkpoint (has everything)
        # Fall back to final rollouts if no incremental exists
        for fname in ["rollouts_incremental.json", "stage2_rollouts.json"]:
            fpath = os.path.join(iter_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                raw_groups = data.get("data", data).get("groups", [])
                groups.extend(_parse_groups(raw_groups, iter_dir, fname, kept_prompts))
                break
            except (json.JSONDecodeError, OSError, KeyError):
                continue

    # Also load synthesis data for QA pairs without rollouts
    synthesis_groups = []
    for iter_dir in sorted(glob.glob(os.path.join(ckpt_base, "iter*"))):
        for fname in ["stage1_deduped.json", "stage1_synthesis.json"]:
            fpath = os.path.join(iter_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                raw = data.get("data", data)
                examples = raw if isinstance(raw, list) else raw.get("examples", [])
                synthesis_groups = [
                    {"question": e.get("question", ""), "answer": e.get("answer", "")}
                    for e in examples
                ]
                break
            except (json.JSONDecodeError, OSError):
                continue

    return jsonify({
        "project": project,
        "rollout_groups": groups,
        "synthesis_examples": synthesis_groups,
        "total_groups": len(groups),
        "total_synthesis": len(synthesis_groups),
    })


@app.route("/traces/api/traces", methods=["GET"])
def list_traces():
    """List all available trace sessions.

    Merges three sources:
    1. Static JSON files in data/
    2. Auto-discovered rollout data (stage2, arena, checkpoints)
    3. Marks each with its source for the UI
    """
    sessions = []

    # 1. Static files in data/
    for fp in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
            sessions.append({
                "session_id": fp.stem,
                "filename": fp.name,
                "query_id": data.get("query_id"),
                "question_preview": (data.get("question", "")[:120] + "..."
                                     if len(data.get("question", "")) > 120
                                     else data.get("question", "")),
                "num_models": len(data.get("models", [])),
                "source": data.get("source", "uploaded"),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    # 2. Auto-discovered data from the KONASH project
    discovered: List[Dict[str, Any]] = []
    discovered.extend(_discover_stage2_data())
    discovered.extend(_discover_arena_results())
    discovered.extend(_discover_checkpoint_rollouts())

    for d in discovered:
        sid = f"live_{d['source'].replace('/', '_').replace(':', '_')}_{d['query_id']}"
        # Avoid duplicates if already saved to data/
        if any(s["session_id"] == sid for s in sessions):
            continue
        sessions.append({
            "session_id": sid,
            "filename": None,
            "query_id": d.get("query_id"),
            "question_preview": (d.get("question", "")[:120] + "..."
                                 if len(d.get("question", "")) > 120
                                 else d.get("question", "")),
            "num_models": len(d.get("models", [])),
            "source": d.get("source", "discovered"),
        })

    # Assign sequential query_ids across all sources
    for i, s in enumerate(sessions, start=1):
        s["query_id"] = i

    return jsonify(sessions)


def _ensure_viewer_steps(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw rollout steps to viewer format if needed.

    Static JSON files may store steps in the raw KONASH format
    (step/type/query/thought) rather than the viewer format
    (step_number/category/queries/num_queries/docs_retrieved).
    Detect and convert in-place.
    """
    expected_docs = data.get("expected_documents", [])
    for model in data.get("models", []):
        for trace in model.get("traces", []):
            steps = trace.get("steps", [])
            if not steps:
                continue
            # Check if already in viewer format
            if "step_number" in steps[0] and "category" in steps[0]:
                continue
            # Convert from raw rollout format
            trace["steps"] = convert_konash_rollout(steps, expected_docs)
            trace["total_steps"] = len(trace["steps"])
    return data


@app.route("/traces/api/trace/<session_id>", methods=["GET"])
def get_trace(session_id: str):
    """Get full trace data for a session.

    Checks static files first, then auto-discovered sources.
    """
    # 1. Check static data/
    fp = DATA_DIR / f"{session_id}.json"
    if fp.exists():
        try:
            with open(fp) as f:
                data = json.load(f)
            return jsonify(_ensure_viewer_steps(data))
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON data"}), 500

    # 2. Check live/discovered sessions
    if session_id.startswith("live_"):
        all_discovered: List[Dict[str, Any]] = []
        all_discovered.extend(_discover_stage2_data())
        all_discovered.extend(_discover_arena_results())
        all_discovered.extend(_discover_checkpoint_rollouts())

        for d in all_discovered:
            sid = f"live_{d['source'].replace('/', '_').replace(':', '_')}_{d['query_id']}"
            if sid == session_id:
                return jsonify(d)

    return jsonify({"error": "Session not found"}), 404


@app.route("/traces/api/trace/upload", methods=["POST"])
def upload_trace():
    """Upload new trace data (JSON).

    Accepts either:
      - Raw trace viewer format (with models/steps)
      - KONASH rollout format (auto-converts)
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Auto-detect KONASH rollout format and convert
    if "rollouts" in data and "models" not in data:
        data = _convert_uploaded_rollouts(data)

    # Validate required fields
    required = ["question", "models"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Assign query_id if not present
    if "query_id" not in data:
        data["query_id"] = len(list(DATA_DIR.glob("*.json"))) + 1

    # Generate a filename
    session_id = f"session_{data['query_id']}_{uuid.uuid4().hex[:8]}"
    fp = DATA_DIR / f"{session_id}.json"
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"session_id": session_id, "status": "ok"})


def _convert_uploaded_rollouts(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert KONASH-native rollout upload format to trace viewer format.

    Expected input format:
    {
        "question": str,
        "expected_documents": [str],
        "rollouts": [
            {
                "model_name": str,
                "rollout_id": int,
                "steps": [KONASH step dicts],
                "passed": bool
            }
        ]
    }
    """
    question = data.get("question", "")
    expected_docs = data.get("expected_documents", [])
    raw_rollouts = data.get("rollouts", [])

    # Group rollouts by model
    models_map: Dict[str, List] = {}
    for rr in raw_rollouts:
        model_name = rr.get("model_name", "Unknown")
        if model_name not in models_map:
            models_map[model_name] = []
        models_map[model_name].append(rr)

    models = []
    for model_name, rollouts in models_map.items():
        traces = []
        for idx, rr in enumerate(rollouts):
            raw_steps = rr.get("steps", [])
            viewer_steps = convert_konash_rollout(raw_steps, expected_docs)
            found = viewer_steps[-1]["expected_found"] if viewer_steps else 0
            total_exp = len(expected_docs)
            traces.append({
                "trace_id": idx + 1,
                "coverage": found / total_exp if total_exp > 0 else 0.0,
                "total_steps": len(viewer_steps),
                "found_count": found,
                "total_expected": total_exp,
                "steps": viewer_steps,
            })
        models.append({
            "name": model_name,
            "traces": traces,
        })

    return {
        "query_id": data.get("query_id", 1),
        "question": question,
        "expected_documents": expected_docs,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Count discovered sources
    n_static = len(list(DATA_DIR.glob("*.json")))
    n_stage2 = len(_discover_stage2_data())
    n_arena = len(_discover_arena_results())
    n_ckpt = len(_discover_checkpoint_rollouts())

    print("KONASH Trace Viewer")
    print("=" * 40)
    print(f"Static sessions:     {n_static}  ({DATA_DIR})")
    print(f"Stage 2 rollouts:    {n_stage2}  ({STAGE2_PATH})")
    print(f"Arena results:       {n_arena}  ({ARENA_DATA_DIR})")
    print(f"Checkpoint rollouts: {n_ckpt}  ({CHECKPOINTS_DIR})")
    print(f"Total:               {n_static + n_stage2 + n_arena + n_ckpt}")
    print()
    print("Open http://localhost:5050 in your browser")
    print()
    app.run(host="0.0.0.0", port=5050, debug=True)
