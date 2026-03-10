from __future__ import annotations

from konash.harness.environment import Environment
from konash.plugins.compression import CompressionPlugin
from konash.plugins.control import StepBudgetPlugin, ToolGatePlugin


class StubAgent:
    def __init__(self, response):
        self.response = response
        self.seen_history = None

    def generate_step(self, history, **kwargs):
        self.seen_history = list(history)
        return dict(self.response)

    def extract_final_answer(self, history):
        for message in reversed(history):
            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]
        return None


def test_environment_compute_reward_uses_final_answer():
    calls = []

    def reward_fn(prediction, reference=None, **kwargs):
        calls.append((prediction, reference, kwargs.get("final_answer")))
        return 1.0 if prediction == "final answer" and reference == "gold" else 0.0

    env = Environment(reward_functions=[reward_fn])
    agent = StubAgent({"role": "assistant", "content": "final answer"})
    env.reset(prompt="question")

    result = env.run_episode(agent, reference_answer="gold", max_steps=1)

    assert result["reward"] == 1.0
    assert calls == [("final answer", "gold", "final answer")]


def test_environment_applies_compression_history_override_before_generation():
    plugin = CompressionPlugin(threshold_tokens=1, target_tokens=1)
    env = Environment(plugins=[plugin])
    agent = StubAgent({"role": "assistant", "content": "done"})
    env.reset(prompt="this prompt is long enough to trigger compression")
    env.conversation_history.append(
        {"role": "assistant", "content": "older context that should be summarized"}
    )

    env.step(agent)

    assert agent.seen_history[0]["role"] == "user"
    assert agent.seen_history[1]["role"] == "assistant"
    assert "Compressed:" in agent.seen_history[1]["content"]


def test_environment_resets_plugins_between_episodes():
    plugin = StepBudgetPlugin(max_steps=1)
    env = Environment(plugins=[plugin])
    agent = StubAgent({"role": "assistant", "content": "done"})

    env.reset(prompt="one")
    first = env.run_episode(agent, max_steps=2)
    assert first["steps"] == 1
    assert plugin.exhausted is False

    env.reset(prompt="two")
    second = env.run_episode(agent, max_steps=2)
    assert second["steps"] == 1


def test_environment_blocks_denied_tool_calls():
    agent = StubAgent(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
    )
    seen_calls = []
    env = Environment(
        plugins=[ToolGatePlugin(denied_tools={"search"})],
        tool_executor=lambda tool_call: seen_calls.append(tool_call) or {
            "role": "tool",
            "content": "executed",
        },
    )
    env.reset(prompt="question")

    step = env.step(agent)

    assert seen_calls == []
    assert step["tool_results"][0]["name"] == "tool_gate"
    assert "not permitted" in step["tool_results"][0]["content"]
