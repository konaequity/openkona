"""KONASH Arena — side-by-side search agent comparison tool.

A Flask app inspired by LMArena / Chatbot Arena, purpose-built for comparing
knowledge agents that use retrieval-augmented reasoning (search + think + answer).

Run:
    python tools/arena/app.py

Then open http://localhost:5117 in your browser.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# KONASH imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from konash.api import Agent, MODEL_PRESETS
from konash.corpus import Corpus
from konash.synthesis.rollouts import RolloutGenerator

# ---------------------------------------------------------------------------
# Extend MODEL_PRESETS with additional Together AI models for arena testing
# ---------------------------------------------------------------------------
def _together_preset(model_id: str, desc: str, **extra) -> Dict[str, Any]:
    return {
        "base_model": model_id,
        "api_base": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "temperature": 0.7,
        "description": desc,
        **extra,
    }

# Ordered list — UI will display in this order
_ARENA_EXTRA_PRESETS_ORDERED = [
    ("glm-5",              _together_preset("zai-org/GLM-5", "GLM 5")),
    ("qwen3.5-397b",       _together_preset("Qwen/Qwen3.5-397B-A17B", "Qwen 3.5 397B MoE")),
    ("minimax-m2.5",       _together_preset("MiniMaxAI/MiniMax-M2.5", "MiniMax M2.5")),
    ("kimi-k2.5",          _together_preset("moonshotai/Kimi-K2.5", "Kimi K2.5")),
    ("glm-4.7",            _together_preset("zai-org/GLM-4.7", "GLM 4.7")),
    ("qwen3.5-9b",         _together_preset("Qwen/Qwen3.5-9B", "Qwen 3.5 9B")),
    ("deepseek-r1",        _together_preset("deepseek-ai/DeepSeek-R1", "DeepSeek R1")),
    ("qwen3-80b-a3b",      _together_preset("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3 80B-A3B MoE")),
    ("llama-3.3-70b-turbo", _together_preset("meta-llama/Llama-3.3-70B-Turbo", "Llama 3.3 70B Turbo")),
    ("mixtral-8x22b",      _together_preset("mistralai/Mixtral-8x22B-Instruct-v0.1", "Mixtral 8x22B MoE")),
    ("qwen-2.5-72b",       _together_preset("Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen 2.5 72B Turbo")),
]

_ARENA_EXTRA_PRESETS = dict(_ARENA_EXTRA_PRESETS_ORDERED)

# Merge without overwriting existing presets
for k, v in _ARENA_EXTRA_PRESETS.items():
    if k not in MODEL_PRESETS:
        MODEL_PRESETS[k] = v

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------
from flask import Flask, Response, jsonify, render_template, request

from jinja2 import ChoiceLoader, FileSystemLoader

_ARENA_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_ARENA_DIR, "templates"),
)

_SHARED_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared", "templates")
app.jinja_loader = ChoiceLoader([
    app.jinja_loader,
    FileSystemLoader(_SHARED_TEMPLATES),
])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = DATA_DIR / "results.jsonl"
RUNS_FILE = DATA_DIR / "runs.jsonl"

# In-memory stores
_active_runs: Dict[str, Dict[str, Any]] = {}
_event_queues: Dict[str, queue.Queue] = {}

# Default corpus path (can be overridden via env or request)
DEFAULT_CORPUS = os.environ.get(
    "KONASH_ARENA_CORPUS",
    os.path.join(PROJECT_ROOT, "data", "corpus"),
)

# Cache loaded corpora
_corpus_cache: Dict[str, Corpus] = {}


def _get_corpus(corpus_path: str) -> Corpus:
    """Get or create a Corpus instance, with caching."""
    if corpus_path not in _corpus_cache:
        corpus = Corpus(corpus_path)
        corpus.ingest()
        _corpus_cache[corpus_path] = corpus
    return _corpus_cache[corpus_path]


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

_KONASH_CONFIG_PATH = os.path.expanduser("~/.konash/config.json")
_ENV_TO_CONFIG_KEY = {
    "TOGETHER_API_KEY": "together_api_key",
    "ZHIPU_API_KEY": "zhipu_api_key",
    "HF_TOKEN": "hf_token",
    "GOOGLE_API_KEY": "google_api_key",
}


def _load_konash_api_key(env_var: str) -> str:
    """Load an API key from ~/.konash/config.json."""
    config_key = _ENV_TO_CONFIG_KEY.get(env_var)
    if not config_key:
        return ""
    try:
        with open(_KONASH_CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg.get(config_key, "")
    except (OSError, json.JSONDecodeError):
        return ""


def _make_llm_fn(preset_cfg: Dict[str, Any]) -> Any:
    """Create a callable llm_fn from a preset configuration."""
    api_base = preset_cfg.get("api_base")
    if not api_base:
        return None

    api_key_env = preset_cfg.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "")

    # Fall back to ~/.konash/config.json if env var not set
    if not api_key:
        api_key = _load_konash_api_key(api_key_env)
    model = preset_cfg["base_model"]
    temperature = preset_cfg.get("temperature", 0.7)

    # Use the same OpenAI-compatible client approach as konash.api
    import urllib.request
    import urllib.error

    def llm_fn(messages, **kwargs):
        url = f"{api_base.rstrip('/')}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", temperature),
            "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 2048)),
        }
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "KONASH-Arena/0.1",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

        choice = result["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return {"role": "assistant", "content": content}

    return llm_fn


def _run_agent_direct(
    run_id: str,
    side: str,
    question: str,
    preset_name: str,
    event_queue: queue.Queue,
):
    """Run a direct LLM call (no corpus/retrieval). Used when no corpus is available."""
    t_start = time.time()
    steps_out: List[Dict[str, Any]] = []
    final_answer = ""
    error_msg = None
    ttft = None

    try:
        preset_cfg = MODEL_PRESETS.get(preset_name)
        if not preset_cfg:
            raise ValueError(f"Unknown preset: {preset_name}")

        llm_fn = _make_llm_fn(preset_cfg)
        if not llm_fn:
            raise ValueError(f"Preset '{preset_name}' has no API endpoint (local-only model)")

        # Emit a reasoning step to show we're calling the LLM
        reasoning_step = {
            "step": 0, "type": "reasoning",
            "thought": f"Calling {preset_cfg.get('base_model', preset_name)} directly (no corpus)...",
        }
        ttft_event = round(time.time() - t_start, 2)
        event_queue.put({
            "run_id": run_id, "side": side, "event": "step",
            "step": _serialize_step(reasoning_step),
            "elapsed": ttft_event, "ttft": ttft_event,
        })
        steps_out.append(_serialize_step(reasoning_step))

        # Call LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question directly and concisely."},
            {"role": "user", "content": question},
        ]
        result = llm_fn(messages, max_tokens=2048)
        final_answer = result.get("content", "")
        ttft = time.time() - t_start

        # Emit answer step
        answer_step = {"step": 1, "type": "answer", "answer": final_answer}
        event_queue.put({
            "run_id": run_id, "side": side, "event": "step",
            "step": _serialize_step(answer_step),
            "elapsed": round(time.time() - t_start, 2),
            "ttft": round(ttft, 2),
        })
        steps_out.append(_serialize_step(answer_step))

    except Exception as e:
        error_msg = str(e)
        final_answer = f"Error: {error_msg}"

    t_total = time.time() - t_start

    done_event = {
        "run_id": run_id, "side": side, "event": "done",
        "final_answer": final_answer,
        "ttft": round(ttft, 2) if ttft else 0,
        "total_time": round(t_total, 2),
        "num_steps": len(steps_out),
        "error": error_msg,
    }
    event_queue.put(done_event)

    return {
        "side": side, "preset": preset_name, "final_answer": final_answer,
        "steps": steps_out, "ttft": round(ttft, 2) if ttft else 0,
        "total_time": round(t_total, 2), "error": error_msg,
    }


def _run_agent(
    run_id: str,
    side: str,
    question: str,
    preset_name: str,
    corpus_path: Optional[str],
    event_queue: queue.Queue,
    max_steps: int = 10,
    top_k: int = 10,
):
    """Run a single agent using Agent.solve() — same path as eval scripts.

    If corpus_path is None or points to a missing/empty directory,
    falls back to direct LLM mode (no retrieval).
    """
    # Check if corpus exists and has documents
    use_corpus = False
    if corpus_path:
        cp = Path(corpus_path)
        if cp.exists() and cp.is_dir() and any(cp.iterdir()):
            use_corpus = True

    if not use_corpus:
        return _run_agent_direct(run_id, side, question, preset_name, event_queue)

    t_start = time.time()
    ttft = None
    steps_out: List[Dict[str, Any]] = []
    final_answer = ""
    error_msg = None

    def _emit_status(message: str):
        event_queue.put({
            "run_id": run_id, "side": side, "event": "status",
            "message": message, "elapsed": round(time.time() - t_start, 2),
        })

    try:
        preset_cfg = MODEL_PRESETS.get(preset_name)
        if not preset_cfg:
            raise ValueError(f"Unknown preset: {preset_name}")

        model_id = preset_cfg["base_model"]
        api_base = preset_cfg.get("api_base")
        api_key_env = preset_cfg.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") or _load_konash_api_key(api_key_env)

        corpus_name = Path(corpus_path).name if corpus_path else "corpus"
        _emit_status(f"Loading {corpus_name} corpus...")

        # Use Agent — same as eval scripts. Handles embedding alignment, etc.
        agent = Agent(
            base_model=model_id,
            corpus=corpus_path,
            project=f"arena-{side}",
            api_base=api_base,
            api_key=api_key,
        )

        if not agent.corpus.indexed:
            agent.corpus.ingest()
        doc_count = len(agent.corpus.documents) if hasattr(agent.corpus, "documents") else 0
        _emit_status(f"Corpus ready ({doc_count:,} docs). Searching...")

        # Plugin that streams steps to SSE as they happen
        class _SSEStepPlugin:
            def after_step(self, step_index, history, step_result):
                nonlocal ttft
                if ttft is None:
                    ttft = time.time() - t_start
                step_data = _serialize_step(step_result)
                steps_out.append(step_data)
                event_queue.put({
                    "run_id": run_id, "side": side, "event": "step",
                    "step": step_data,
                    "elapsed": round(time.time() - t_start, 2),
                    "ttft": round(ttft, 2),
                })

        # Inject the SSE plugin into the agent's solve() call
        from konash.harness.environment import Environment
        from konash.plugins.control import StepBudgetPlugin

        base_agent = agent._make_agent(max_steps=max_steps)

        def _extract_tool_query(tool_call):
            if not isinstance(tool_call, dict):
                return str(tool_call)
            q = tool_call.get("query", "") or tool_call.get("input", "")
            if q:
                return str(q)
            fn = tool_call.get("function")
            if isinstance(fn, dict):
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        return args
                if isinstance(args, dict):
                    return str(args.get("query", "") or args.get("input", "") or args)
            return str(tool_call)

        def tool_executor(tool_call):
            query_text = _extract_tool_query(tool_call)
            results = agent.corpus.search(query_text, top_k=top_k)
            result_text = "\n\n".join(
                f"[{i+1}] (score: {r.get('score', 0):.3f}) [{os.path.basename(r.get('source', ''))}] {r.get('text', '')}"
                for i, r in enumerate(results)
            )
            obs = {"role": "tool", "content": result_text}
            if isinstance(tool_call, dict) and tool_call.get("id"):
                obs["tool_call_id"] = tool_call["id"]
            return obs

        env = Environment(
            tool_executor=tool_executor,
            plugins=[
                _SSEStepPlugin(),
                StepBudgetPlugin(max_steps=max_steps),
            ],
            available_tools=[{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the knowledge base for relevant documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                },
            }],
        )

        formatted_query = agent._format_solver_prompt(question)
        env.reset(prompt=formatted_query)
        episode = env.run_episode(agent=base_agent, max_steps=max_steps)

        final_answer = episode.get("final_answer", "") or ""
        if ttft is None:
            ttft = time.time() - t_start

    except Exception as e:
        error_msg = str(e)
        final_answer = f"Error: {error_msg}"

    t_total = time.time() - t_start

    done_event = {
        "run_id": run_id,
        "side": side,
        "event": "done",
        "final_answer": final_answer,
        "ttft": round(ttft, 2) if ttft else 0,
        "total_time": round(t_total, 2),
        "num_steps": len(steps_out),
        "error": error_msg,
    }
    event_queue.put(done_event)

    return {
        "side": side,
        "preset": preset_name,
        "final_answer": final_answer,
        "steps": steps_out,
        "ttft": round(ttft, 2) if ttft else 0,
        "total_time": round(t_total, 2),
        "error": error_msg,
    }


def _serialize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a rollout step dict into a JSON-safe representation for the UI.

    Handles two formats:
    - RolloutGenerator: {type, query, thought, results, ...}
    - Agent.solve() trajectory: {agent_response, tool_results, done, step_index}
    """
    # Detect Agent.solve() trajectory format
    if "agent_response" in step:
        ar = step.get("agent_response", {})
        tool_calls = ar.get("tool_calls", [])
        tool_results = step.get("tool_results", [])
        done = step.get("done", False)
        reasoning = ar.get("reasoning_content") or ""
        content = ar.get("content", "") or ""

        if tool_calls:
            # Search step
            tc = tool_calls[0]
            fn = tc.get("function", {})
            fn_args = fn.get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except (json.JSONDecodeError, ValueError):
                    fn_args = {"query": fn_args}
            query = fn_args.get("query", str(fn_args)) if isinstance(fn_args, dict) else str(fn_args)

            # Count results from tool_results content
            result_text = tool_results[0].get("content", "") if tool_results else ""
            num_results = result_text.count("\n[") + (1 if result_text.startswith("[") else 0)

            return {
                "step": step.get("step_index", 0),
                "type": "retrieval",
                "query": query,
                "num_results": num_results,
                "thought": reasoning or content,
                "results": [],  # Full results available via tool_result endpoint
            }
        elif done:
            return {
                "step": step.get("step_index", 0),
                "type": "answer",
                "answer": content,
                "thought": reasoning,
            }
        else:
            return {
                "step": step.get("step_index", 0),
                "type": "reasoning",
                "thought": reasoning or content,
            }

    # Original RolloutGenerator format
    out: Dict[str, Any] = {
        "step": step.get("step", 0),
        "type": step.get("type", "unknown"),
    }

    if step.get("type") == "retrieval":
        out["query"] = step.get("query", "")
        out["num_results"] = step.get("num_results", 0)
        results = step.get("results", [])
        out["results"] = [
            {
                "text": (r.get("text", str(r)) if isinstance(r, dict) else str(r))[:500],
                "score": r.get("score", 0) if isinstance(r, dict) else 0,
                "source": r.get("source", "") if isinstance(r, dict) else "",
            }
            for r in results[:5]
        ]

    elif step.get("type") == "reasoning":
        out["thought"] = step.get("thought", "")
        sub = step.get("sub_retrieval")
        if sub:
            out["sub_retrieval"] = {
                "query": sub.get("query", ""),
                "num_results": sub.get("num_results", 0),
            }

    elif step.get("type") == "answer":
        out["answer"] = step.get("answer", "")
        out["thought"] = step.get("thought", "")

    return out


def _save_run(run_record: Dict[str, Any]) -> None:
    """Persist a completed run (with full steps) for trace viewer discovery."""
    try:
        with open(RUNS_FILE, "a") as f:
            f.write(json.dumps(run_record, default=str) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/arena")
@app.route("/arena/")
def index():
    return render_template("index.html")


@app.route("/arena/api/presets")
def api_presets():
    """Return available model presets as an ordered list."""
    # Build ordered list: arena extras first (in defined order), then any from konash
    seen = set()
    presets = []

    # Arena models first, in the order defined above
    for name, _ in _ARENA_EXTRA_PRESETS_ORDERED:
        cfg = MODEL_PRESETS.get(name, {})
        if not cfg.get("api_base"):
            continue
        presets.append({
            "name": name,
            "description": cfg.get("description", name),
            "base_model": cfg.get("base_model", ""),
            "has_api": True,
        })
        seen.add(name)

    # Then any konash presets not already listed
    for name, cfg in MODEL_PRESETS.items():
        if name in seen or not cfg.get("api_base"):
            continue
        presets.append({
            "name": name,
            "description": cfg.get("description", name),
            "base_model": cfg.get("base_model", ""),
            "has_api": True,
        })

    return jsonify(presets)


@app.route("/arena/api/corpora")
def api_corpora():
    """Return available corpora (downloaded + any local folders)."""
    from konash.download import DEFAULT_CORPUS_DIR

    corpora = []
    corpus_root = Path(DEFAULT_CORPUS_DIR)
    if corpus_root.exists():
        for d in sorted(corpus_root.iterdir()):
            if d.is_dir():
                # Quick doc count: check for prebuilt index first, else count top-level files only
                index_file = d / "prebuilt_index.npz"
                if index_file.exists():
                    # Use metadata from index if available
                    try:
                        import numpy as np
                        idx = np.load(index_file, allow_pickle=True)
                        doc_count = len(idx.get("doc_ids", idx.get("sources", [])))
                    except Exception:
                        doc_count = sum(1 for f in d.iterdir() if f.is_file())
                else:
                    doc_count = sum(1 for f in d.iterdir() if f.is_file())
                corpora.append({
                    "name": d.name,
                    "path": str(d),
                    "doc_count": doc_count,
                })

    return jsonify(corpora)


def _ensure_preset(model_key: str) -> str:
    """Ensure a model key exists in MODEL_PRESETS.

    If model_key is not a known preset, treat it as a raw Together AI model ID
    and create an ad-hoc preset for it (e.g. 'meta-llama/Llama-4-Scout-17B-16E-Instruct').
    Returns the preset key to use.
    """
    if model_key in MODEL_PRESETS:
        return model_key

    # Treat as a raw Together AI model ID
    slug = model_key.replace("/", "--").lower()
    preset_key = f"custom:{slug}"
    if preset_key not in MODEL_PRESETS:
        MODEL_PRESETS[preset_key] = {
            "base_model": model_key,
            "api_base": "https://api.together.xyz/v1",
            "api_key_env": "TOGETHER_API_KEY",
            "temperature": 0.7,
            "description": f"Custom: {model_key}",
        }
    return preset_key


@app.route("/arena/api/run", methods=["POST"])
def api_run():
    """Start a comparison run with two agents."""
    data = request.json or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400

    model_a = _ensure_preset(data.get("model_a", "glm-4.5-air-together"))
    model_b = _ensure_preset(data.get("model_b", "glm-4.5-air-together"))
    corpus_path = data.get("corpus_path", DEFAULT_CORPUS)
    blind_mode = data.get("blind_mode", False)
    max_steps = data.get("max_steps", 10)
    top_k = data.get("top_k", 10)

    run_id = str(uuid.uuid4())[:12]

    # In blind mode, randomize A/B assignment
    import random
    if blind_mode:
        if random.random() > 0.5:
            model_a, model_b = model_b, model_a

    eq: queue.Queue = queue.Queue()
    _event_queues[run_id] = eq

    run_record = {
        "run_id": run_id,
        "question": question,
        "model_a": model_a,
        "model_b": model_b,
        "blind_mode": blind_mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "results": {},
    }
    _active_runs[run_id] = run_record

    def _run_both():
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                _run_agent, run_id, "a", question, model_a,
                corpus_path, eq, max_steps, top_k,
            )
            future_b = pool.submit(
                _run_agent, run_id, "b", question, model_b,
                corpus_path, eq, max_steps, top_k,
            )

            result_a = future_a.result()
            result_b = future_b.result()

        run_record["results"]["a"] = result_a
        run_record["results"]["b"] = result_b
        run_record["status"] = "complete"

        # Persist full run (with steps) for trace viewer
        _save_run(run_record)

        # Signal stream end
        eq.put({"run_id": run_id, "event": "end"})

    thread = threading.Thread(target=_run_both, daemon=True)
    thread.start()

    return jsonify({"run_id": run_id, "status": "started"})


@app.route("/arena/api/stream/<run_id>")
def api_stream(run_id):
    """SSE endpoint to stream agent steps in real time."""
    eq = _event_queues.get(run_id)
    if not eq:
        return jsonify({"error": "Run not found"}), 404

    def generate():
        while True:
            try:
                event = eq.get(timeout=120)
            except queue.Empty:
                yield "data: {\"event\": \"timeout\"}\n\n"
                break

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("event") == "end":
                break

        # Clean up
        _event_queues.pop(run_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/arena/api/run/<run_id>")
def api_run_status(run_id):
    """Get the status and results of a run."""
    run = _active_runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(run)


@app.route("/arena/api/vote", methods=["POST"])
def api_vote():
    """Record a preference vote."""
    data = request.json or {}
    run_id = data.get("run_id", "")
    winner = data.get("winner", "")  # "a", "b", or "tie"
    comments = data.get("comments", "")[:500]

    run = _active_runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404

    vote_record = {
        "run_id": run_id,
        "question": run.get("question", ""),
        "model_a": run.get("model_a", ""),
        "model_b": run.get("model_b", ""),
        "blind_mode": run.get("blind_mode", False),
        "winner": winner,
        "comments": comments,
        "voted_at": datetime.now(timezone.utc).isoformat(),
        "results_a": _summarize_result(run.get("results", {}).get("a", {})),
        "results_b": _summarize_result(run.get("results", {}).get("b", {})),
    }

    # Append to JSONL file
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(vote_record) + "\n")

    run["vote"] = vote_record
    return jsonify({"status": "recorded", "vote": vote_record})


def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact summary of a run result for storage."""
    return {
        "preset": result.get("preset", ""),
        "final_answer": result.get("final_answer", "")[:1000],
        "num_steps": result.get("num_steps", len(result.get("steps", []))),
        "ttft": result.get("ttft", 0),
        "total_time": result.get("total_time", 0),
        "error": result.get("error"),
    }


@app.route("/arena/api/leaderboard")
def api_leaderboard():
    """Compute and return leaderboard data from stored votes."""
    if not RESULTS_FILE.exists():
        return jsonify({"models": [], "total_votes": 0})

    model_stats: Dict[str, Dict[str, Any]] = {}

    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vote = json.loads(line)
            except json.JSONDecodeError:
                continue

            model_a = vote.get("model_a", "")
            model_b = vote.get("model_b", "")
            winner = vote.get("winner", "")

            for model in (model_a, model_b):
                if model not in model_stats:
                    model_stats[model] = {
                        "model": model,
                        "wins": 0,
                        "losses": 0,
                        "ties": 0,
                        "total": 0,
                        "avg_ttft": 0,
                        "avg_total_time": 0,
                        "ttft_sum": 0,
                        "time_sum": 0,
                    }

            if winner == "a":
                model_stats[model_a]["wins"] += 1
                model_stats[model_b]["losses"] += 1
            elif winner == "b":
                model_stats[model_b]["wins"] += 1
                model_stats[model_a]["losses"] += 1
            elif winner == "tie":
                model_stats[model_a]["ties"] += 1
                model_stats[model_b]["ties"] += 1

            model_stats[model_a]["total"] += 1
            model_stats[model_b]["total"] += 1

            # Accumulate timing stats
            res_a = vote.get("results_a", {})
            res_b = vote.get("results_b", {})
            model_stats[model_a]["ttft_sum"] += res_a.get("ttft", 0)
            model_stats[model_a]["time_sum"] += res_a.get("total_time", 0)
            model_stats[model_b]["ttft_sum"] += res_b.get("ttft", 0)
            model_stats[model_b]["time_sum"] += res_b.get("total_time", 0)

    # Compute averages and win rate
    models = []
    for stats in model_stats.values():
        total = stats["total"]
        if total > 0:
            stats["win_rate"] = round(
                (stats["wins"] + 0.5 * stats["ties"]) / total * 100, 1
            )
            stats["avg_ttft"] = round(stats["ttft_sum"] / total, 2)
            stats["avg_total_time"] = round(stats["time_sum"] / total, 2)
        else:
            stats["win_rate"] = 0
            stats["avg_ttft"] = 0
            stats["avg_total_time"] = 0
        del stats["ttft_sum"]
        del stats["time_sum"]
        models.append(stats)

    # Sort by win rate descending
    models.sort(key=lambda m: m["win_rate"], reverse=True)

    total_votes = sum(m["total"] for m in models) // 2 if models else 0
    return jsonify({"models": models, "total_votes": total_votes})


@app.route("/arena/api/share/<run_id>")
def api_share(run_id):
    """Return shareable run data."""
    run = _active_runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404

    share_data = {
        "run_id": run_id,
        "question": run.get("question", ""),
        "model_a": run.get("model_a", ""),
        "model_b": run.get("model_b", ""),
        "status": run.get("status", ""),
        "vote": run.get("vote"),
    }

    if run.get("status") == "complete":
        results = run.get("results", {})
        for side in ("a", "b"):
            r = results.get(side, {})
            share_data[f"result_{side}"] = {
                "final_answer": r.get("final_answer", ""),
                "num_steps": len(r.get("steps", [])),
                "ttft": r.get("ttft", 0),
                "total_time": r.get("total_time", 0),
                "steps": r.get("steps", []),
            }

    return jsonify(share_data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("ARENA_PORT", 5117))
    print(f"\n  KONASH Arena")
    print(f"  http://localhost:{port}\n")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Corpus path:  {DEFAULT_CORPUS}")
    print(f"  Results file: {RESULTS_FILE}\n")
    print(f"  Available presets: {', '.join(sorted(MODEL_PRESETS.keys()))}\n")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
