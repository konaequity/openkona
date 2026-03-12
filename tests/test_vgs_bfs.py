"""Tests for VGS BFS logic — expand, score, aggregate, and parallel BFS
without requiring any API calls or external services."""

import pytest

from konash.inference.value_search import ValueGuidedSearchEngine


class TestExpand:

    def test_expand_without_agent_returns_k_candidates(self):
        engine = ValueGuidedSearchEngine(candidate_width=3, max_depth=2)
        state = {"query": "test", "steps": [], "terminal": False}
        candidates = engine.expand(state, k=3)
        assert len(candidates) == 3

    def test_expand_preserves_query(self):
        engine = ValueGuidedSearchEngine(candidate_width=2, max_depth=2)
        state = {"query": "my question", "steps": [], "terminal": False}
        candidates = engine.expand(state)
        for c in candidates:
            assert c["query"] == "my question"

    def test_expand_appends_step(self):
        engine = ValueGuidedSearchEngine(candidate_width=2, max_depth=2)
        state = {"query": "q", "steps": [{"content": "step1"}], "terminal": False}
        candidates = engine.expand(state)
        for c in candidates:
            assert len(c["steps"]) == 2


class TestScoreCandidates:

    def test_no_value_model_returns_zeros(self):
        engine = ValueGuidedSearchEngine()
        candidates = [
            {"steps": [{"content": "a"}]},
            {"steps": [{"content": "b"}]},
        ]
        scores = engine.score_candidates(candidates)
        assert scores == [0.0, 0.0]

    def test_empty_candidates(self):
        engine = ValueGuidedSearchEngine()
        assert engine.score_candidates([]) == []


class TestAggregate:

    def test_no_aggregator_picks_highest_score(self):
        engine = ValueGuidedSearchEngine()
        result = engine.aggregate(
            ["answer_a", "answer_b", "answer_c"],
            scores=[0.3, 0.9, 0.1],
        )
        assert result == "answer_b"

    def test_no_scores_returns_first(self):
        engine = ValueGuidedSearchEngine()
        result = engine.aggregate(["first", "second"])
        assert result == "first"

    def test_all_zero_scores_returns_first(self):
        engine = ValueGuidedSearchEngine()
        result = engine.aggregate(
            ["first", "second"],
            scores=[0.0, 0.0],
        )
        assert result == "first"

    def test_empty_candidates_returns_empty(self):
        engine = ValueGuidedSearchEngine()
        assert engine.aggregate([]) == ""


class TestRunParallelBFS:

    def test_single_tree(self):
        engine = ValueGuidedSearchEngine(
            candidate_width=2, max_depth=2,
        )
        trees = engine.run_parallel_bfs("test query", num_trees=1)
        assert len(trees) == 1
        tree = trees[0]
        assert "best_trajectory" in tree
        assert "tree_index" in tree
        assert tree["tree_index"] == 0

    def test_multiple_trees(self):
        engine = ValueGuidedSearchEngine(
            candidate_width=2, max_depth=1,
        )
        trees = engine.run_parallel_bfs("test query", num_trees=3)
        assert len(trees) == 3
        indices = [t["tree_index"] for t in trees]
        assert sorted(indices) == [0, 1, 2]

    def test_run_returns_answer(self):
        engine = ValueGuidedSearchEngine(
            candidate_width=2, parallel_searches=2, max_depth=2,
        )
        result = engine.run("What is 2+2?")
        assert "answer" in result
        assert "search_trees" in result
        assert "scores" in result
        assert result["num_trees"] == 2


class TestBuildConversationHistory:

    def test_empty_state(self):
        history = ValueGuidedSearchEngine._build_conversation_history(
            {"query": "", "steps": []}
        )
        assert history == []

    def test_query_becomes_user_message(self):
        history = ValueGuidedSearchEngine._build_conversation_history(
            {"query": "Hello", "steps": []}
        )
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}

    def test_reasoning_steps(self):
        state = {
            "query": "q",
            "steps": [
                {"type": "reasoning", "content": "I think..."},
                {"type": "reasoning", "content": "Therefore..."},
            ],
        }
        history = ValueGuidedSearchEngine._build_conversation_history(state)
        assert len(history) == 3  # query + 2 reasoning
        assert history[1]["role"] == "assistant"

    def test_tool_call_with_result(self):
        state = {
            "query": "q",
            "steps": [
                {
                    "type": "tool_call",
                    "content": "search('Paris')",
                    "result": "Paris is the capital of France.",
                },
            ],
        }
        history = ValueGuidedSearchEngine._build_conversation_history(state)
        assert len(history) == 3  # query + tool_call + tool_result
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "tool"

    def test_string_steps(self):
        state = {"query": "q", "steps": ["step1", "step2"]}
        history = ValueGuidedSearchEngine._build_conversation_history(state)
        assert len(history) == 3
        assert history[1]["content"] == "step1"


class TestExtendState:

    def test_extends_without_mutation(self):
        state = {"query": "q", "steps": [{"content": "a"}], "terminal": False}
        new = ValueGuidedSearchEngine._extend_state(state, {"content": "b"})
        assert len(new["steps"]) == 2
        assert len(state["steps"]) == 1  # original unchanged

    def test_terminal_step_marks_state(self):
        state = {"query": "q", "steps": [], "terminal": False}
        new = ValueGuidedSearchEngine._extend_state(
            state, {"content": "done", "terminal": True}
        )
        assert new["terminal"] is True
        assert state["terminal"] is False


class TestExtractAnswer:

    def test_from_final_answer_field(self):
        state = {"final_answer": "Paris", "steps": []}
        assert ValueGuidedSearchEngine._extract_answer_from_state(state) == "Paris"

    def test_from_last_step_dict(self):
        state = {"steps": [{"content": "Paris"}]}
        assert ValueGuidedSearchEngine._extract_answer_from_state(state) == "Paris"

    def test_from_last_step_string(self):
        state = {"steps": ["Paris"]}
        assert ValueGuidedSearchEngine._extract_answer_from_state(state) == "Paris"

    def test_empty_state(self):
        assert ValueGuidedSearchEngine._extract_answer_from_state({"steps": []}) == ""
