from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from konash.api import Agent
from konash.corpus import Corpus
from konash.inference.parallel import ParallelThinkingEngine
from konash.harness.strategy import ParallelThinkingStrategy
from konash.synthesis.rollouts import Rollout, RolloutGroup


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

    corpus = Corpus(doc_dir)
    corpus.ingest()

    def fake_get_model_engine(self):
        self._model_engine = StubLocalEngine()
        return self._model_engine

    monkeypatch.setattr(Agent, "_get_model_engine", fake_get_model_engine)

    agent = Agent.load(str(project_dir), corpus=corpus)

    answer = agent.solve("What is alpha?", max_steps=1)

    assert answer == "loaded answer"


def test_train_runs_local_synthesis_then_cloud_oapl(tmp_path, monkeypatch):
    """Iteration 1: synthesis + rollouts local, OAPL on cloud."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "alpha.txt").write_text("alpha facts live here")

    corpus = Corpus(doc_dir)
    corpus.ingest()
    agent = Agent(base_model="stub", corpus=corpus, api_base="http://example", api_key="k")

    # Patch synthesis to return empty (simulating no QA pairs generated)
    from konash.synthesis.qa import QuestionAnswerSynthesizer
    monkeypatch.setattr(
        QuestionAnswerSynthesizer, "synthesize",
        lambda self, **kwargs: [],
    )

    # Patch generate_fn to avoid network calls
    monkeypatch.setattr(
        agent, "_get_generate_fn",
        lambda: (lambda messages, **kw: {"role": "assistant", "content": ""}),
    )

    # train() should complete without calling cloud (no rollout data)
    result = agent.train(iterations=1, synthesis_calls=1, verbose=False)

    assert result["iterations"] == 1
    assert result["stats"] == []  # no stats since synthesis produced nothing


def test_agent_defers_local_embed_model_when_prebuilt_index_exists(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "financebench"
    corpus_dir.mkdir()
    (corpus_dir / "prebuilt_index.npz").write_bytes(b"stub")

    def fail_make_embed_fn(self):
        raise AssertionError("local embed model should not be loaded eagerly")

    monkeypatch.setattr(Agent, "_make_embed_fn", fail_make_embed_fn)

    agent = Agent(
        base_model="stub",
        corpus=corpus_dir,
        api_base="http://example",
        api_key="k",
    )

    assert agent.embedding_provider == "local"


# ------------------------------------------------------------------
# Parallel thinking concurrency tests
# ------------------------------------------------------------------


def test_parallel_thinking_engine_runs_rollouts_concurrently():
    """Verify rollouts run in parallel, not sequentially.

    Each stub rollout sleeps for 0.1s.  With N=5 sequential rollouts that
    would take >=0.5s.  Running concurrently should take ~0.1s.
    """
    active_threads: list[str] = []
    lock = threading.Lock()

    class SlowAgent:
        def generate_rollout(self, query, **kwargs):
            with lock:
                active_threads.append(threading.current_thread().name)
            time.sleep(0.1)
            return {"query": query, "final_answer": "answer", "steps": []}

    engine = ParallelThinkingEngine(agent=SlowAgent(), num_rollouts=5)
    start = time.monotonic()
    rollouts = engine.generate_parallel_rollouts("test query", num_rollouts=5)
    elapsed = time.monotonic() - start

    assert len(rollouts) == 5
    # If truly parallel, should finish well under 5 * 0.1s = 0.5s
    assert elapsed < 0.35, f"Took {elapsed:.2f}s — rollouts appear sequential"
    # Multiple threads should have been used
    unique_threads = set(active_threads)
    assert len(unique_threads) > 1, "All rollouts ran on the same thread"


def test_parallel_thinking_strategy_runs_rollouts_concurrently():
    """Same concurrency check for the harness-level strategy."""
    active_threads: list[str] = []
    lock = threading.Lock()

    class SlowEnvironment:
        def __init__(self):
            self.conversation_history = []
            self._step_count = 0

        def reset(self, prompt=""):
            self.conversation_history = [{"role": "user", "content": prompt}]
            self._step_count = 0

        def run_episode(self, agent, **kwargs):
            with lock:
                active_threads.append(threading.current_thread().name)
            time.sleep(0.1)
            return {"final_answer": "answer", "steps": []}

        def __deepcopy__(self, memo):
            return SlowEnvironment()

    class StubAgent:
        pass

    strategy = ParallelThinkingStrategy(num_rollouts=5, max_steps=1)
    env = SlowEnvironment()

    start = time.monotonic()
    results = strategy.spawn_parallel_rollouts(
        prompt="test", agent=StubAgent(), environment=env
    )
    elapsed = time.monotonic() - start

    assert len(results) == 5
    assert elapsed < 0.35, f"Took {elapsed:.2f}s — rollouts appear sequential"
    unique_threads = set(active_threads)
    assert len(unique_threads) > 1, "All rollouts ran on the same thread"


def test_parallel_thinking_engine_preserves_rollout_order():
    """Rollouts should be returned in index order regardless of completion order."""
    import random

    class VariableAgent:
        def generate_rollout(self, query, rollout_index=0, **kwargs):
            time.sleep(random.uniform(0.01, 0.05))
            return {"query": query, "final_answer": f"answer_{rollout_index}",
                    "rollout_index": rollout_index, "steps": []}

    engine = ParallelThinkingEngine(agent=VariableAgent(), num_rollouts=8)
    rollouts = engine.generate_parallel_rollouts("test", num_rollouts=8)

    for i, rollout in enumerate(rollouts):
        assert rollout["final_answer"] == f"answer_{i}", (
            f"Rollout at position {i} has answer '{rollout['final_answer']}' — order not preserved"
        )


# ------------------------------------------------------------------
# Rollout generator uses harness (not custom loop)
# ------------------------------------------------------------------


def test_rollout_generator_uses_environment_harness():
    """Verify generate_single() runs through Environment.run_episode(),
    not the old custom _reason_with_llm() loop.

    The LLM receives proper tool_calls and tool results in its conversation
    history, not flat evidence summaries.
    """
    from konash.synthesis.rollouts import RolloutGenerator

    call_log = []

    class StubSearchTool:
        def search(self, query, top_k=20):
            return [{"text": f"Result for: {query}", "score": 0.9}]

    def stub_llm(messages, **kwargs):
        call_log.append(messages)
        # Check if tools are being passed (harness path)
        tools = kwargs.get("tools")
        if tools:
            # First call: make a tool call
            if len(call_log) == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "test query"}',
                        },
                    }],
                }
        # Subsequent calls: give final answer
        return {"role": "assistant", "content": "The answer is 42."}

    gen = RolloutGenerator(
        search_tool=StubSearchTool(),
        llm_fn=stub_llm,
        max_steps=5,
    )

    rollout = gen.generate_single("What is the answer?")

    assert rollout.final_answer is not None
    assert len(rollout.steps) > 0
    # The harness stores conversation history in metadata
    assert "history" in rollout.metadata
    history = rollout.metadata["history"]
    # History should have proper message roles, not flat evidence summaries
    roles = [m.get("role") for m in history]
    assert "user" in roles
    assert "assistant" in roles


def test_rollout_generator_produces_multi_turn_conversation():
    """The LLM should see tool results as proper messages, not flat summaries."""
    from konash.synthesis.rollouts import RolloutGenerator

    conversation_snapshots = []

    class StubSearchTool:
        def search(self, query, top_k=20):
            return [{"text": "Albert Einstein was born in Ulm.", "score": 0.95}]

    call_count = 0

    def tracking_llm(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        conversation_snapshots.append(list(messages))

        if call_count == 1 and kwargs.get("tools"):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "Einstein birthplace"}',
                    },
                }],
            }
        return {"role": "assistant", "content": "Albert Einstein was born in Ulm."}

    gen = RolloutGenerator(
        search_tool=StubSearchTool(),
        llm_fn=tracking_llm,
        max_steps=5,
    )
    rollout = gen.generate_single("Where was Einstein born?")

    assert rollout.final_answer == "Albert Einstein was born in Ulm."
    # After the tool call, the LLM should see the tool result in its history
    if len(conversation_snapshots) > 1:
        second_call = conversation_snapshots[1]
        roles_in_second = [m.get("role") for m in second_call]
        assert "tool" in roles_in_second, (
            f"Tool results not in conversation history. Roles: {roles_in_second}"
        )


# ------------------------------------------------------------------
# Quality filter receives rollout attempts
# ------------------------------------------------------------------


def test_quality_filter_receives_rollout_attempts():
    """Verify that run_stage_two() passes rollout attempts to the quality filter."""
    from konash.synthesis.pipeline import SynthesisPipeline
    from konash.synthesis.qa import SyntheticExample
    from konash.synthesis.rollouts import Rollout, RolloutGroup
    from konash.synthesis.filters import QualityFilter

    received_attempts = []

    class TrackingQualityFilter(QualityFilter):
        def apply(self, examples, reference_documents=None, rollout_attempts=None):
            received_attempts.append(rollout_attempts)
            return list(examples)  # pass everything through

    class StubRolloutGenerator:
        def generate_group(self, prompt, reference_answer=None, num_rollouts=8, qa_idx=0):
            rollouts = [
                Rollout(
                    steps=[{"step": 0, "type": "answer"}],
                    final_answer=f"answer_{i}",
                    passed=(i % 2 == 0),
                )
                for i in range(num_rollouts)
            ]
            return RolloutGroup(
                prompt=prompt,
                reference_answer=reference_answer,
                rollouts=rollouts,
            )

    pipeline = SynthesisPipeline(
        rollout_generator=StubRolloutGenerator(),
        quality_filter=TrackingQualityFilter(),
    )
    pipeline.synthetic_examples = [
        SyntheticExample(question="What is X?", answer="Y"),
    ]

    pipeline.run_stage_two(num_rollouts=4)

    assert len(received_attempts) == 1, "Quality filter was not called"
    attempts = received_attempts[0]
    assert attempts is not None, "rollout_attempts was None — not passed through"
    assert len(attempts) > 0, "rollout_attempts was empty"
    # Each attempt should have answer, score, passed
    for attempt in attempts[0]:
        assert "answer" in attempt
        assert "score" in attempt
        assert "passed" in attempt
