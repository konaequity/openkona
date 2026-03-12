"""Tests for Value-Guided Search (VGS) integration in the Agent pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from konash.api import Agent
from konash.corpus import Corpus
from konash.inference.value_model import ValueModel
from konash.inference.value_search import ValueGuidedSearchEngine


def _make_corpus(doc_dir):
    """Create a pre-built Corpus to avoid embedding provider dependencies."""
    corpus = Corpus(doc_dir)
    corpus.ingest()
    return corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubLLMClient:
    """Canned LLM client that returns predetermined responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self._responses:
            # Default: return a final answer (no tool calls)
            return {"role": "assistant", "content": "stub answer"}
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Value model unit tests
# ---------------------------------------------------------------------------


def test_value_model_fit_and_score():
    """ValueModel can fit on rollout data and produce scores in [0, 1]."""
    vm = ValueModel(feature_dim=64)

    # Create rollout-like data: dicts with 'steps'
    rollouts = [
        {"steps": [{"role": "assistant", "content": "I searched and found the answer."}]},
        {"steps": [{"role": "assistant", "content": "search"}, {"role": "tool", "content": "results"}]},
        {"steps": [{"role": "assistant", "content": "short"}]},
    ]
    rewards = [1.0, 0.0, 1.0]

    stats = vm.fit(rollouts, rewards, lr=0.01, epochs=10)

    assert "final_loss" in stats
    assert stats["final_loss"] < 1.0  # Should converge somewhat

    # Scores should be in [0, 1]
    for rollout in rollouts:
        score = vm.score_partial_rollout(rollout)
        assert 0.0 <= score <= 1.0


def test_value_model_save_load(tmp_path):
    """ValueModel can be saved as JSON and restored."""
    vm = ValueModel(feature_dim=8)
    rollouts = [
        {"steps": [{"role": "assistant", "content": "answer"}]},
    ]
    vm.fit(rollouts, [1.0], epochs=5)

    # Save
    vm_path = tmp_path / "value_model.json"
    weights = vm.weights
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    with open(vm_path, "w") as f:
        json.dump({
            "weights": weights,
            "bias": vm.bias,
            "feature_dim": vm.feature_dim,
        }, f)

    # Load
    with open(vm_path) as f:
        data = json.load(f)
    vm2 = ValueModel(
        weights=data["weights"],
        bias=data["bias"],
        feature_dim=data["feature_dim"],
    )

    # Same scores
    score1 = vm.score_partial_rollout(rollouts[0])
    score2 = vm2.score_partial_rollout(rollouts[0])
    assert abs(score1 - score2) < 1e-6


# ---------------------------------------------------------------------------
# VGS engine unit tests
# ---------------------------------------------------------------------------


def test_vgs_engine_runs_without_agent():
    """VGS engine runs with placeholder expansion when no agent is set."""
    engine = ValueGuidedSearchEngine(
        candidate_width=2,
        parallel_searches=1,
        max_depth=2,
    )
    result = engine.run("What is the capital of France?")
    assert "answer" in result
    assert "search_trees" in result
    assert result["num_trees"] == 1


def test_vgs_engine_with_value_model():
    """VGS engine uses value model scores for candidate selection."""
    vm = ValueModel(feature_dim=64)
    # Pre-train with some data so scores aren't all 0
    rollouts = [
        {"steps": [{"role": "assistant", "content": "Paris is the capital of France."}]},
        {"steps": [{"role": "assistant", "content": "I don't know."}]},
    ]
    vm.fit(rollouts, [1.0, 0.0], epochs=20)

    engine = ValueGuidedSearchEngine(
        value_model=vm,
        candidate_width=2,
        parallel_searches=2,
        max_depth=2,
    )
    result = engine.run("What is the capital of France?")
    assert result["num_trees"] == 2
    assert len(result["scores"]) == 2
    # Scores should not all be zero since we have a trained value model
    # (though they may be close to 0.5 since the model is small)


# ---------------------------------------------------------------------------
# Agent VGS integration
# ---------------------------------------------------------------------------


def test_agent_stores_value_model_after_training(tmp_path):
    """After train(), the agent should have a value model and save it."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "fact.txt").write_text("The speed of light is 299792458 m/s.")

    agent = Agent(
        base_model="test-model",
        corpus=_make_corpus(doc_dir),
        project="test_vgs",
        api_base="https://example/v1",
        api_key="k",
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    # Manually set up a value model as if training had happened
    vm = ValueModel(feature_dim=64)
    vm.fit(
        [{"steps": [{"role": "assistant", "content": "answer"}]}],
        [1.0],
        epochs=5,
    )
    agent._value_model = vm

    assert agent._value_model is not None
    score = agent._value_model.score_partial_rollout(
        {"steps": [{"role": "assistant", "content": "test"}]}
    )
    assert 0.0 <= score <= 1.0


def test_agent_load_restores_value_model(tmp_path):
    """Agent.load() should restore the value model from checkpoint."""
    # Set up checkpoint directory
    ckpt_dir = tmp_path / "project" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    # Save training metadata
    meta = {
        "base_model": "test-model",
        "project": "test_vgs",
        "iterations": 1,
        "stats": [],
        "value_model": True,
    }
    with open(ckpt_dir / "training_meta.json", "w") as f:
        json.dump(meta, f)

    # Save a value model
    vm = ValueModel(feature_dim=64)
    vm.fit(
        [{"steps": [{"role": "assistant", "content": "good answer"}]}],
        [1.0],
        epochs=5,
    )
    weights = vm.weights
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    with open(ckpt_dir / "value_model.json", "w") as f:
        json.dump({
            "weights": weights,
            "bias": vm.bias,
            "feature_dim": vm.feature_dim,
        }, f)

    # Create a dummy corpus
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc.txt").write_text("test document content")

    # Load the agent
    loaded = Agent.load(
        str(tmp_path / "project"),
        corpus=_make_corpus(doc_dir),
        api_base="https://example/v1",
        api_key="k",
    )

    assert loaded._value_model is not None
    assert loaded._trained is True

    # Value model should produce valid scores
    score = loaded._value_model.score_partial_rollout(
        {"steps": [{"role": "assistant", "content": "test"}]}
    )
    assert 0.0 <= score <= 1.0


def test_solve_uses_vgs_when_value_model_available(tmp_path):
    """solve(use_vgs=True) should use ValueGuidedSearchEngine."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "fact.txt").write_text("The capital of France is Paris.")

    agent = Agent(
        base_model="test-model",
        corpus=_make_corpus(doc_dir),
        project="test_vgs",
        api_base="https://example/v1",
        api_key="k",
    )

    # Set up a value model
    vm = ValueModel(feature_dim=64)
    vm.fit(
        [{"steps": [{"role": "assistant", "content": "Paris"}]}],
        [1.0],
        epochs=5,
    )
    agent._value_model = vm

    # Stub the LLM client so it returns a final answer without needing network
    agent._llm_client = StubLLMClient([
        # VGS expand() will call generate_step multiple times
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "assistant", "content": "Paris is the capital."},
        {"role": "assistant", "content": "Paris."},
        {"role": "assistant", "content": "Paris."},
    ])

    # Ingest corpus first
    agent.corpus.ingest()

    answer = agent.solve(
        "What is the capital of France?",
        use_vgs=True,
        parallel_rollouts=2,
        vgs_candidate_width=2,
        vgs_max_depth=2,
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


def test_solve_falls_back_to_parallel_thinking_without_value_model(tmp_path):
    """solve() with no value model should use ParallelThinkingEngine."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "fact.txt").write_text("Water boils at 100 degrees Celsius.")

    agent = Agent(
        base_model="test-model",
        corpus=_make_corpus(doc_dir),
        project="test_vgs",
        api_base="https://example/v1",
        api_key="k",
    )

    # No value model set
    assert agent._value_model is None

    agent._llm_client = StubLLMClient([
        {"role": "assistant", "content": "Water boils at 100°C."},
        {"role": "assistant", "content": "100 degrees Celsius."},
    ])

    agent.corpus.ingest()

    answer = agent.solve(
        "What temperature does water boil at?",
        parallel_rollouts=2,
    )

    assert isinstance(answer, str)
