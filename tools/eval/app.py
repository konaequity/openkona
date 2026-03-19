"""KONASH Eval Trace Viewer — browse and inspect evaluation run traces.

A Flask app for viewing eval results from FinanceBench, QAMPARI, and other
benchmark runs. Supports lazy-loading of large tool results and file upload.

Run:
    python tools/eval/app.py

Then open http://localhost:5118 in your browser.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
from flask import Flask, jsonify, render_template, request

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_EVAL_DIR, "templates"),
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Scan locations for eval JSON files
_SCAN_DIRS: List[Path] = [
    Path(PROJECT_ROOT) / "eval_results",
    DATA_DIR,
]

# Also scan ~/.konash/projects/*/eval_results/
_KONASH_PROJECTS = Path.home() / ".konash" / "projects"
if _KONASH_PROJECTS.exists():
    for project_dir in _KONASH_PROJECTS.iterdir():
        if project_dir.is_dir():
            eval_dir = project_dir / "eval_results"
            if eval_dir.exists():
                _SCAN_DIRS.append(eval_dir)

# ---------------------------------------------------------------------------
# File cache with mtime-based invalidation
# ---------------------------------------------------------------------------
# Maps absolute file path -> {"mtime": float, "data": parsed dict}
_file_cache: Dict[str, Dict[str, Any]] = {}


def _load_eval_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load and validate an eval JSON file, using mtime-based cache."""
    abs_path = str(filepath.resolve())
    try:
        current_mtime = filepath.stat().st_mtime
    except OSError:
        _file_cache.pop(abs_path, None)
        return None

    cached = _file_cache.get(abs_path)
    if cached and cached["mtime"] == current_mtime:
        return cached["data"]

    try:
        with open(filepath) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Validate: must have benchmark and single_details keys
    if not isinstance(data, dict):
        return None
    if "benchmark" not in data or "single_details" not in data:
        return None

    _file_cache[abs_path] = {"mtime": current_mtime, "data": data}
    return data


def _discover_benchmarks() -> Dict[str, Dict[str, Any]]:
    """Scan all configured directories for eval JSON files.

    Returns a dict mapping benchmark_id -> {"file": Path, "data": dict}.
    """
    benchmarks: Dict[str, Dict[str, Any]] = {}

    for scan_dir in _SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for json_file in sorted(scan_dir.glob("*.json")):
            data = _load_eval_file(json_file)
            if data is None:
                continue
            benchmark_id = json_file.stem
            # If we already have this id from an earlier scan dir, skip
            if benchmark_id not in benchmarks:
                benchmarks[benchmark_id] = {
                    "file": json_file,
                    "data": data,
                }

    return benchmarks


# ---------------------------------------------------------------------------
# Step normalization
# ---------------------------------------------------------------------------

def _normalize_step(raw_step: Dict[str, Any], pos_index: int = 0, lightweight: bool = True) -> Dict[str, Any]:
    """Transform a raw trajectory step into the normalized viewer format.

    Args:
        raw_step: Raw step dict from eval JSON (agent_response, tool_results, done, step_index).
        pos_index: 0-based positional index in the trajectory array.
        lightweight: If True, set tool_result to None (for list views).
    """
    agent_resp = raw_step.get("agent_response", {})
    tool_calls = agent_resp.get("tool_calls")
    done = raw_step.get("done", False)
    step_index = pos_index

    reasoning_content = agent_resp.get("reasoning_content") or agent_resp.get("content", "")
    content = agent_resp.get("content", "")

    # Determine step type
    if tool_calls:
        step_type = "search"
    elif done:
        step_type = "answer"
    else:
        step_type = "reasoning"

    # Parse tool_call
    tool_call = None
    if tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        fn_name = fn.get("name", "")
        fn_args = fn.get("arguments", {})
        # If arguments is a string, try to parse as JSON
        if isinstance(fn_args, str):
            try:
                fn_args = json.loads(fn_args)
            except (json.JSONDecodeError, ValueError):
                fn_args = {"query": fn_args}
        tool_call = {"name": fn_name, "arguments": fn_args}

    # Extract tool result and its length
    tool_results_list = raw_step.get("tool_results") or []
    tool_result_text: Optional[str] = None
    tool_result_length = 0
    if tool_results_list and len(tool_results_list) > 0:
        tool_result_text = tool_results_list[0].get("content", "")
        tool_result_length = len(tool_result_text) if tool_result_text else 0

    return {
        "step_index": step_index,
        "type": step_type,
        "reasoning": reasoning_content,
        "content": content,
        "tool_call": tool_call,
        "tool_result": None if lightweight else tool_result_text,
        "tool_result_length": tool_result_length,
        "done": done,
    }


def _derive_name(data: Dict[str, Any], filename_stem: str) -> str:
    """Derive a display name from the benchmark field or filename."""
    benchmark = data.get("benchmark")
    if benchmark and isinstance(benchmark, str):
        return benchmark
    return filename_stem.replace("_", " ").replace("-", " ").title()


def _build_question(index: int, detail: Dict[str, Any], lightweight: bool = True) -> Dict[str, Any]:
    """Build a normalized question dict from a single_details entry."""
    trajectory = detail.get("trajectory", [])
    steps = [_normalize_step(step, pos_index=i, lightweight=lightweight) for i, step in enumerate(trajectory)]

    return {
        "index": index,
        "question": detail.get("question", ""),
        "reference": detail.get("reference") or detail.get("reference_str"),
        "reference_answers": detail.get("reference_answers"),
        "answer": detail.get("answer", ""),
        "score": detail.get("score", 0.0),
        "passed": detail.get("score", 0.0) >= 0.6,
        "latency": detail.get("latency", 0.0),
        "num_steps": detail.get("num_steps", len(trajectory)),
        "num_searches": detail.get("num_searches", 0),
        "search_queries": detail.get("search_queries", []),
        "nuggets": detail.get("nuggets"),
        "nugget_scores": detail.get("nugget_scores"),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/eval")
@app.route("/eval/")
def index():
    return render_template("index.html")


@app.route("/eval/api/benchmarks")
def api_benchmarks():
    """List all discovered benchmarks with summary stats."""
    benchmarks_map = _discover_benchmarks()
    result = []

    for benchmark_id, entry in benchmarks_map.items():
        data = entry["data"]
        source_file = str(entry["file"])
        details = data.get("single_details", [])
        single_summary = data.get("single", {})

        num_questions = data.get("num_questions", len(details))
        accuracy = single_summary.get("accuracy", 0.0)
        avg_score = single_summary.get("avg_score", accuracy)
        avg_latency = single_summary.get("avg_latency", 0.0)
        total_time = single_summary.get("total_time", 0.0)

        # Compute pass/fail counts
        pass_count = 0
        fail_count = 0
        for d in details:
            if d.get("score", 0.0) >= 0.6:
                pass_count += 1
            else:
                fail_count += 1

        result.append({
            "id": benchmark_id,
            "name": _derive_name(data, benchmark_id),
            "benchmark_type": (data.get("benchmark") or benchmark_id).lower().replace(" ", ""),
            "source_file": source_file,
            "num_questions": num_questions,
            "accuracy": round(accuracy, 4),
            "avg_score": round(avg_score, 4),
            "avg_latency": round(avg_latency, 2),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total_time": round(total_time, 2),
        })

    return jsonify({"benchmarks": result})


@app.route("/eval/api/benchmark/<benchmark_id>")
def api_benchmark(benchmark_id: str):
    """Return all questions with reasoning but NO tool_results (hybrid loading)."""
    benchmarks_map = _discover_benchmarks()
    entry = benchmarks_map.get(benchmark_id)
    if not entry:
        return jsonify({"error": f"Benchmark '{benchmark_id}' not found"}), 404

    data = entry["data"]
    details = data.get("single_details", [])
    single_summary = data.get("single", {})

    questions = [
        _build_question(i, detail, lightweight=True)
        for i, detail in enumerate(details)
    ]

    return jsonify({
        "benchmark_id": benchmark_id,
        "name": _derive_name(data, benchmark_id),
        "accuracy": round(single_summary.get("accuracy", 0.0), 4),
        "questions": questions,
    })


@app.route("/eval/api/benchmark/<benchmark_id>/question/<int:q_index>/step/<int:step_index>/tool_result")
def api_tool_result(benchmark_id: str, q_index: int, step_index: int):
    """Lazy-load a single step's full tool result."""
    benchmarks_map = _discover_benchmarks()
    entry = benchmarks_map.get(benchmark_id)
    if not entry:
        return jsonify({"error": f"Benchmark '{benchmark_id}' not found"}), 404

    data = entry["data"]
    details = data.get("single_details", [])

    if q_index < 0 or q_index >= len(details):
        return jsonify({"error": f"Question index {q_index} out of range"}), 404

    trajectory = details[q_index].get("trajectory", [])

    # Find the step: try positional index first, then by step_index field
    target_step = None
    if 0 <= step_index < len(trajectory):
        target_step = trajectory[step_index]
    else:
        for step in trajectory:
            if step.get("step_index") == step_index:
                target_step = step
                break

    if target_step is None:
        return jsonify({"error": f"Step index {step_index} not found"}), 404

    # Extract the full tool result
    tool_results_list = target_step.get("tool_results") or []
    tool_result_text = ""
    if tool_results_list and len(tool_results_list) > 0:
        tool_result_text = tool_results_list[0].get("content", "")

    return jsonify({"tool_result": tool_result_text})


@app.route("/eval/api/upload", methods=["POST"])
def api_upload():
    """Accept eval JSON upload, save to tools/eval/data/, return benchmark_id."""
    if request.is_json:
        data = request.get_json()
    else:
        # Try to read from file upload
        uploaded = request.files.get("file")
        if not uploaded:
            return jsonify({"error": "No JSON data or file provided"}), 400
        try:
            data = json.load(uploaded)
        except (json.JSONDecodeError, ValueError) as e:
            return jsonify({"error": f"Invalid JSON: {e}"}), 400

    # Validate
    if not isinstance(data, dict):
        return jsonify({"error": "JSON must be an object"}), 400
    if "benchmark" not in data or "single_details" not in data:
        return jsonify({"error": "JSON must have 'benchmark' and 'single_details' keys"}), 400

    # Determine filename
    benchmark_name = data.get("benchmark", "unknown")
    safe_name = benchmark_name.lower().replace(" ", "_").replace("-", "_")
    filename = f"{safe_name}_eval.json"
    filepath = DATA_DIR / filename

    # Avoid overwriting: add suffix if needed
    counter = 1
    while filepath.exists():
        filename = f"{safe_name}_eval_{counter}.json"
        filepath = DATA_DIR / filename
        counter += 1

    with open(filepath, "w") as f:
        json.dump(data, f)

    benchmark_id = filepath.stem
    return jsonify({"status": "ok", "benchmark_id": benchmark_id})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("EVAL_PORT", 5118))
    print(f"\n  KONASH Eval Trace Viewer")
    print(f"  http://localhost:{port}\n")
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Data dir:      {DATA_DIR}")

    # Show discovered benchmarks
    benchmarks = _discover_benchmarks()
    if benchmarks:
        print(f"  Benchmarks:    {len(benchmarks)} found")
        for bid, entry in benchmarks.items():
            n = len(entry["data"].get("single_details", []))
            print(f"    - {bid} ({n} questions) from {entry['file']}")
    else:
        print(f"  Benchmarks:    none found (add eval JSON files)")

    print()
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
