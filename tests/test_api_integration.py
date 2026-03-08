from __future__ import annotations

import json
from pathlib import Path

from konash.api import Agent
from konash.corpus import Corpus


class StubLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("No stub responses left.")
        return self._responses.pop(0)


class StubLocalEngine:
    def generate(self, messages, **kwargs):
        return {"role": "assistant", "content": "loaded answer"}


def test_agent_solve_parses_openai_style_tool_calls(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "alpha.txt").write_text("alpha facts live here")

    corpus = Corpus(doc_dir)
    corpus.ingest()

    seen_queries = []
    original_search = corpus.search

    def wrapped_search(query, top_k=10, **kwargs):
        seen_queries.append(query)
        return original_search(query, top_k=top_k, **kwargs)

    corpus.search = wrapped_search

    client = StubLLMClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "alpha"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "final answer"},
        ]
    )

    agent = Agent(base_model="stub", corpus=corpus, api_base="http://example", api_key="k")
    agent._llm_client = client

    answer = agent.solve("What is alpha?", max_steps=2)

    assert answer == "final answer"
    assert seen_queries == ["alpha"]
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"


def test_agent_solve_parallel_rollouts_runs_end_to_end(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "alpha.txt").write_text("alpha facts live here")

    corpus = Corpus(doc_dir)
    client = StubLLMClient(
        [
            {"role": "assistant", "content": "candidate one"},
            {"role": "assistant", "content": "candidate one"},
            {"role": "assistant", "content": "aggregated answer"},
        ]
    )

    agent = Agent(base_model="stub", corpus=corpus, api_base="http://example", api_key="k")
    agent._llm_client = client

    answer = agent.solve("What is alpha?", parallel_rollouts=2, max_steps=1)

    assert answer == "aggregated answer"


def test_agent_load_supports_local_inference_via_lazy_engine_init(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "base_model": "stub-model",
                "project": "demo",
                "iterations": 1,
                "stats": [],
            }
        )
    )

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "alpha.txt").write_text("alpha facts live here")

    def fake_get_model_engine(self):
        self._model_engine = StubLocalEngine()
        return self._model_engine

    monkeypatch.setattr(Agent, "_get_model_engine", fake_get_model_engine)

    agent = Agent.load(str(project_dir), corpus=str(doc_dir))

    answer = agent.solve("What is alpha?", max_steps=1)

    assert answer == "loaded answer"

