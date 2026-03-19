from __future__ import annotations

import json
import importlib.util
import time
from pathlib import Path

import konash.benchmarks as benchmarks
import pytest


_ARENA_APP_PATH = Path(__file__).resolve().parents[1] / "tools" / "arena" / "app.py"
_ARENA_SPEC = importlib.util.spec_from_file_location("arena_app_module", _ARENA_APP_PATH)
arena_app = importlib.util.module_from_spec(_ARENA_SPEC)
assert _ARENA_SPEC is not None and _ARENA_SPEC.loader is not None
pytest.importorskip("flask")
_ARENA_SPEC.loader.exec_module(arena_app)


def _make_financebench_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "financebench"
    docs_dir = corpus_root / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "doc1.txt").write_text("Revenue was 10 billion dollars.", encoding="utf-8")
    (corpus_root / "eval_questions.json").write_text(
        json.dumps([
            {
                "question": "What was revenue?",
                "answer": "$10 billion",
            }
        ]),
        encoding="utf-8",
    )
    return corpus_root


def test_api_questions_returns_registered_eval_questions(tmp_path, monkeypatch):
    corpus_root = _make_financebench_corpus(tmp_path)
    monkeypatch.setattr(benchmarks, "DEFAULT_CORPUS_DIR", str(tmp_path))

    client = arena_app.app.test_client()
    response = client.get("/arena/api/questions", query_string={"corpus_path": str(corpus_root)})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["benchmark_key"] == "financebench"
    assert payload["benchmark_name"] == "FinanceBench"
    assert len(payload["questions"]) == 1
    assert payload["questions"][0]["question"] == "What was revenue?"


def test_api_run_persists_eval_review_for_selected_question(tmp_path, monkeypatch):
    corpus_root = _make_financebench_corpus(tmp_path)
    monkeypatch.setattr(benchmarks, "DEFAULT_CORPUS_DIR", str(tmp_path))

    arena_app._active_runs.clear()
    arena_app._event_queues.clear()

    def fake_run_agent(run_id, side, question, preset_name, corpus_path, event_queue, max_steps=10, top_k=10):
        return {
            "side": side,
            "preset": preset_name,
            "final_answer": f"answer-{side}",
            "steps": [],
            "ttft": 0.1,
            "total_time": 0.2,
            "error": None,
        }

    def fake_judge(answer_a, answer_b, eval_case):
        return {
            "benchmark_key": eval_case["benchmark_key"],
            "benchmark_name": eval_case["benchmark_name"],
            "question_index": eval_case["question_index"],
            "question_text": eval_case["question_text"],
            "reference": eval_case["reference"],
            "winner": "a",
            "score_delta": 0.5,
            "judge_model": "gpt-4o-mini",
            "judge_provider": "openai",
            "a": {"score": 1.0, "supported": 1, "partial": 0, "total_nuggets": 1, "nuggets": ["$10 billion"], "labels": ["support"]},
            "b": {"score": 0.5, "supported": 0, "partial": 1, "total_nuggets": 1, "nuggets": ["$10 billion"], "labels": ["partial_support"]},
        }

    monkeypatch.setattr(arena_app, "_run_agent", fake_run_agent)
    monkeypatch.setattr(arena_app, "_judge_arena_answers", fake_judge)

    client = arena_app.app.test_client()
    response = client.post(
        "/arena/api/run",
        json={
            "question": "What was revenue?",
            "model_a": "glm-5",
            "model_b": "minimax-m2.5",
            "corpus_path": str(corpus_root),
            "benchmark_key": "financebench",
            "question_index": 0,
        },
    )

    assert response.status_code == 200
    run_id = response.get_json()["run_id"]

    deadline = time.time() + 2.0
    while time.time() < deadline:
        run = arena_app._active_runs.get(run_id)
        if run and run.get("status") == "complete":
            break
        time.sleep(0.02)

    run = arena_app._active_runs[run_id]
    assert run["status"] == "complete"
    assert run["eval_review"]["winner"] == "a"
    assert run["eval_review"]["benchmark_key"] == "financebench"
