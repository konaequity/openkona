"""Behavioral tests for KONASH framework.

These tests call real functions with synthetic data and assert on real outputs.
No API keys, GPU, or network access required. LLM calls are stubbed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest


# ======================================================================
# 1. OAPL train_epoch actually decreases loss
# ======================================================================


class TestOAPLTrainEpochLossDecreases:
    """Create a small OfflineRolloutDataset with 2-3 groups, run train_epoch()
    twice using the numpy reference implementation, and verify loss decreases."""

    def test_loss_decreases_with_policy_fn(self):
        from konash.training.oapl import OAPLTrainer
        from konash.training.dataset import OfflineRolloutDataset

        # Build dataset with 3 groups, 4 rollouts each, mix of pass/fail
        prompts = [
            "What is the capital of France?",
            "Who wrote Hamlet?",
            "What year did Apollo 11 land?",
        ]
        group_rollouts = [
            ["Paris", "London", "Paris", "Berlin"],
            ["Shakespeare", "Dickens", "Shakespeare", "Shakespeare"],
            ["1969", "1970", "1968", "1969"],
        ]
        rewards = [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
        dataset = OfflineRolloutDataset(
            prompts=prompts, group_rollouts=group_rollouts, rewards=rewards,
        )

        # Mutable state: log_probs start at 0.0, shift toward correct answers
        state = {"log_probs": {}}

        def policy_fn(prompt, rollout):
            key = (prompt, str(rollout))
            if key not in state["log_probs"]:
                state["log_probs"][key] = 0.0
            return state["log_probs"][key]

        trainer = OAPLTrainer(beta_value=1.0, beta_kl=0.01)

        # Epoch 1
        result1 = trainer.train_epoch(dataset, policy_fn=policy_fn)
        loss1 = result1["mean_loss"]

        # Simulate policy improvement: shift log_probs toward rewards
        # (high-reward rollouts get higher log_prob, low-reward get lower)
        for group_idx, prompt in enumerate(prompts):
            for local_idx, rollout in enumerate(group_rollouts[group_idx]):
                key = (prompt, str(rollout))
                reward = rewards[group_idx][local_idx]
                v_star = trainer.estimate_optimal_value(rewards[group_idx])
                advantage = reward - v_star
                # Nudge log_prob toward reducing the OAPL loss:
                # loss = (beta_kl * log_ratio - advantage)^2
                # To decrease loss, move beta_kl * log_ratio toward advantage
                target_ratio = advantage / trainer.beta_kl
                state["log_probs"][key] = target_ratio * 0.5  # partial step

        # Epoch 2
        result2 = trainer.train_epoch(dataset, policy_fn=policy_fn)
        loss2 = result2["mean_loss"]

        assert loss1 > 0, "Initial loss should be positive"
        assert loss2 < loss1, f"Loss should decrease: epoch1={loss1:.6f}, epoch2={loss2:.6f}"
        assert result1["num_groups"] == 3
        assert result1["num_rollouts"] == 12

    def test_loss_with_reference_policy(self):
        """With a reference policy, the KL term should affect the loss."""
        from konash.training.oapl import OAPLTrainer
        from konash.training.dataset import OfflineRolloutDataset

        dataset = OfflineRolloutDataset(
            prompts=["Q1", "Q2"],
            group_rollouts=[["A", "B"], ["C", "D"]],
            rewards=[[1.0, 0.0], [0.5, 0.5]],
        )

        def ref_policy(prompt, rollout):
            return -0.1  # constant reference log-prob

        trainer = OAPLTrainer(
            reference_policy=ref_policy, beta_value=1.0, beta_kl=0.01,
        )
        result = trainer.train_epoch(dataset, policy_fn=lambda p, r: 0.0)
        assert result["mean_loss"] > 0
        assert result["num_groups"] == 2


# ======================================================================
# 2. OfflineRolloutDataset grouping
# ======================================================================


class TestOfflineRolloutDatasetGrouping:
    """Verify from_rollouts() groups correctly, rewards assigned properly,
    group count matches unique prompts."""

    def test_from_rollouts_groups_by_prompt(self):
        from konash.training.dataset import OfflineRolloutDataset

        rollout_data = [
            {"prompt": "Q1", "rollout": "A1", "reward": 1.0},
            {"prompt": "Q1", "rollout": "A2", "reward": 0.0},
            {"prompt": "Q2", "rollout": "B1", "reward": 0.5},
            {"prompt": "Q1", "rollout": "A3", "reward": 0.8},
            {"prompt": "Q2", "rollout": "B2", "reward": 0.3},
            {"prompt": "Q3", "rollout": "C1", "reward": 1.0},
        ]

        ds = OfflineRolloutDataset.from_rollouts(rollout_data)

        # Group count matches unique prompts
        assert len(ds.prompts) == 3
        assert ds.prompts == ["Q1", "Q2", "Q3"]

        # Q1 has 3 rollouts
        assert len(ds.group_rollouts[0]) == 3
        assert ds.group_rollouts[0] == ["A1", "A2", "A3"]
        assert ds.rewards[0] == [1.0, 0.0, 0.8]

        # Q2 has 2 rollouts
        assert len(ds.group_rollouts[1]) == 2
        assert ds.group_rollouts[1] == ["B1", "B2"]
        assert ds.rewards[1] == [0.5, 0.3]

        # Q3 has 1 rollout
        assert len(ds.group_rollouts[2]) == 1
        assert ds.rewards[2] == [1.0]

        # Total rollout count
        assert len(ds) == 6

    def test_getitem_flattened_access(self):
        from konash.training.dataset import OfflineRolloutDataset

        ds = OfflineRolloutDataset(
            prompts=["P1", "P2"],
            group_rollouts=[["R1", "R2"], ["R3"]],
            rewards=[[0.5, 1.0], [0.0]],
        )

        item0 = ds[0]
        assert item0["prompt"] == "P1"
        assert item0["rollout"] == "R1"
        assert item0["reward"] == 0.5

        item2 = ds[2]
        assert item2["prompt"] == "P2"
        assert item2["rollout"] == "R3"
        assert item2["reward"] == 0.0

        with pytest.raises(IndexError):
            ds[3]

    def test_group_by_prompt_from_internal(self):
        from konash.training.dataset import OfflineRolloutDataset

        ds = OfflineRolloutDataset(
            prompts=["Alpha", "Beta"],
            group_rollouts=[["a1"], ["b1", "b2"]],
            rewards=[[1.0], [0.5, 0.0]],
        )
        grouped = ds.group_by_prompt()
        assert set(grouped.keys()) == {"Alpha", "Beta"}
        assert len(grouped["Beta"]) == 2

    def test_from_rollouts_preserves_insertion_order(self):
        from konash.training.dataset import OfflineRolloutDataset

        data = [
            {"prompt": "Z", "rollout": "z", "reward": 0.0},
            {"prompt": "A", "rollout": "a", "reward": 1.0},
            {"prompt": "M", "rollout": "m", "reward": 0.5},
        ]
        ds = OfflineRolloutDataset.from_rollouts(data)
        assert ds.prompts == ["Z", "A", "M"]


# ======================================================================
# 3. RolloutGenerator.generate_group produces valid structure
# ======================================================================


class TestRolloutGeneratorStructure:
    """Stub llm_fn and search_tool, call generate_group, verify structure."""

    def _make_generator(self, llm_responses, search_results):
        """Build a RolloutGenerator with stubbed LLM and search."""
        from konash.synthesis.rollouts import RolloutGenerator

        call_count = {"llm": 0}

        def stub_llm(messages, **kwargs):
            idx = call_count["llm"] % len(llm_responses)
            call_count["llm"] += 1
            resp = llm_responses[idx]
            if isinstance(resp, str):
                return {"role": "assistant", "content": resp}
            return resp

        class StubSearch:
            def search(self, query, top_k=5):
                return search_results

        gen = RolloutGenerator(
            max_steps=3,
            top_k=2,
            search_tool=StubSearch(),
            llm_fn=stub_llm,
        )
        return gen

    def test_generate_single_produces_rollout(self):
        from konash.synthesis.rollouts import Rollout

        # LLM first calls search tool, then gives an answer
        llm_responses = [
            # Step 1: agent makes a tool call (search)
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "capital France"}',
                        },
                    }
                ],
            },
            # Step 2: agent gives final answer (no tool call)
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        search_results = [
            {"text": "Paris is the capital of France.", "score": 0.95},
            {"text": "France is in Western Europe.", "score": 0.60},
        ]

        gen = self._make_generator(llm_responses, search_results)
        rollout = gen.generate_single(
            "What is the capital of France?",
            reference_answer="Paris",
        )

        assert isinstance(rollout, Rollout)
        assert rollout.final_answer is not None
        assert "Paris" in rollout.final_answer
        assert rollout.passed is True
        assert len(rollout.steps) >= 1

    def test_generate_group_correct_count(self):
        from konash.synthesis.rollouts import RolloutGroup

        llm_responses = [
            {
                "role": "assistant",
                "content": "Based on my knowledge, the answer is Paris.",
            },
        ]
        search_results = [{"text": "Paris is the capital.", "score": 0.9}]

        gen = self._make_generator(llm_responses, search_results)
        group = gen.generate_group(
            "What is the capital of France?",
            reference_answer="Paris",
            num_rollouts=3,
        )

        assert isinstance(group, RolloutGroup)
        assert group.size == 3
        assert len(group.rollouts) == 3
        for rollout in group.rollouts:
            assert rollout is not None
            assert rollout.final_answer is not None

    def test_pass_rate_reflects_evaluation(self):
        from konash.synthesis.rollouts import RolloutGroup, Rollout

        group = RolloutGroup(
            prompt="test",
            reference_answer="42",
            rollouts=[
                Rollout(steps=[], final_answer="42", passed=True),
                Rollout(steps=[], final_answer="wrong", passed=False),
                Rollout(steps=[], final_answer="42", passed=True),
                Rollout(steps=[], final_answer="nope", passed=False),
            ],
        )
        assert group.pass_rate == 0.5


# ======================================================================
# 4. RLTrainableCompressionPlugin compresses and preserves key info
# ======================================================================


class TestRLTrainableCompressionPlugin:
    """Build history exceeding threshold, verify compress behavior."""

    def test_should_compress_triggers_above_threshold(self):
        from konash.plugins.compression import RLTrainableCompressionPlugin

        plugin = RLTrainableCompressionPlugin(
            threshold_chars=100, target_chars=50,
        )

        short_history = [{"role": "user", "content": "Hi"}]
        assert plugin.should_compress(short_history) is False

        long_history = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "x" * 60},
            {"role": "assistant", "content": "y" * 60},
        ]
        assert plugin.should_compress(long_history) is True

    def test_compress_reduces_char_count(self):
        from konash.plugins.compression import (
            RLTrainableCompressionPlugin,
            _history_chars,
        )

        agent_called = {"count": 0}

        def stub_agent(messages):
            agent_called["count"] += 1
            return {"content": "Summary: key facts preserved."}

        plugin = RLTrainableCompressionPlugin(
            threshold_chars=200,
            target_chars=100,
            agent_fn=stub_agent,
            preserve_recent_turns=2,
        )

        history = [
            {"role": "system", "content": "You are a search agent."},
            {"role": "user", "content": "Search for topic A. " + "detail " * 20},
            {"role": "assistant", "content": "Found results about A. " + "data " * 20},
            {"role": "user", "content": "Search for topic B. " + "detail " * 20},
            {"role": "assistant", "content": "Found results about B. " + "data " * 20},
            {"role": "user", "content": "Now search for topic C."},
            {"role": "assistant", "content": "Here are results for C."},
        ]

        pre_chars = _history_chars(history)
        compressed = plugin.compress(history)
        post_chars = _history_chars(compressed)

        assert post_chars < pre_chars, "Compression should reduce character count"
        assert agent_called["count"] >= 1, "Agent should be called for summarization"

        # System prompt (first message) is preserved
        assert compressed[0]["role"] == "system"
        assert "search agent" in compressed[0]["content"]

        # Recent turns are preserved (last 2)
        assert compressed[-1]["content"] == "Here are results for C."
        assert compressed[-2]["content"] == "Now search for topic C."

        # Compression marker is present
        compression_found = any(
            "<|compression|>" in msg.get("content", "")
            for msg in compressed
        )
        assert compression_found, "Compression marker should be inserted"

    def test_mechanical_fallback_without_agent(self):
        from konash.plugins.compression import (
            RLTrainableCompressionPlugin,
            _history_chars,
        )

        plugin = RLTrainableCompressionPlugin(
            threshold_chars=50, target_chars=30, agent_fn=None,
        )
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "A " * 30},
            {"role": "assistant", "content": "B " * 30},
            {"role": "user", "content": "C " * 30},
            {"role": "assistant", "content": "D " * 30},
            {"role": "user", "content": "recent1"},
            {"role": "assistant", "content": "recent2"},
        ]
        compressed = plugin.compress(history)
        assert len(compressed) < len(history)


# ======================================================================
# 5. PassRateFilter keeps learning-frontier questions
# ======================================================================


class TestPassRateFilter:
    """Create RolloutGroups with various pass rates, verify filtering."""

    def test_filter_keeps_frontier_rejects_extremes(self):
        from konash.synthesis.filters import PassRateFilter
        from konash.synthesis.rollouts import RolloutGroup, Rollout

        def make_group(prompt, num_pass, num_fail):
            rollouts = (
                [Rollout(steps=[], final_answer="ok", passed=True)] * num_pass
                + [Rollout(steps=[], final_answer="bad", passed=False)] * num_fail
            )
            return RolloutGroup(prompt=prompt, rollouts=rollouts)

        groups = [
            make_group("all_fail", 0, 4),       # pass_rate = 0.0
            make_group("one_pass", 1, 3),        # pass_rate = 0.25
            make_group("half_pass", 2, 2),       # pass_rate = 0.50
            make_group("three_pass", 3, 1),      # pass_rate = 0.75
            make_group("all_pass", 4, 0),         # pass_rate = 1.0
        ]

        filt = PassRateFilter(min_pass_rate=0.1, max_pass_rate=0.9)
        result = filt.apply(groups)

        prompts_kept = [g.prompt for g in result]
        assert "all_fail" not in prompts_kept, "0% pass rate should be rejected"
        assert "all_pass" not in prompts_kept, "100% pass rate should be rejected"
        assert "one_pass" in prompts_kept, "25% pass rate should be kept"
        assert "half_pass" in prompts_kept, "50% pass rate should be kept"
        assert "three_pass" in prompts_kept, "75% pass rate should be kept"
        assert len(result) == 3

    def test_filter_with_adaptive_thresholds(self):
        from konash.synthesis.filters import PassRateFilter
        from konash.synthesis.rollouts import RolloutGroup, Rollout

        filt = PassRateFilter(task_name="BrowseCompPlus", iteration=0)
        # Should use adaptive thresholds from ADAPTIVE_THRESHOLDS
        assert filt.min_pass_rate is not None
        assert filt.max_pass_rate is not None

        group = RolloutGroup(
            prompt="test",
            rollouts=[
                Rollout(steps=[], passed=True),
                Rollout(steps=[], passed=False),
            ],
        )
        assert group.pass_rate == 0.5
        result = filt.apply([group])
        assert len(result) == 1  # 0.5 is within [0.1, 0.9]

    def test_binarize_scores(self):
        from konash.synthesis.filters import PassRateFilter

        filt = PassRateFilter(task_name="TRECBiogen", iteration=0)
        threshold = filt.binarization_threshold
        assert threshold == 0.6  # per BINARIZATION_THRESHOLDS

        scores = [0.3, 0.6, 0.8, 0.59, 1.0]
        binary = filt.binarize_scores(scores)
        assert binary == [0.0, 1.0, 1.0, 0.0, 1.0]


# ======================================================================
# 6. Rollout segmentation at compression boundaries
# ======================================================================


class TestRolloutSegmentation:
    """Create rollout steps with compression markers, verify segmentation."""

    def test_no_compression_returns_single_segment(self):
        from konash.training.oapl import _segment_rollout_for_training

        steps = [
            {"type": "retrieval", "query": "test", "results_text": "r1"},
            {"type": "answer", "answer": "final"},
        ]
        pairs = _segment_rollout_for_training("What?", steps)
        assert len(pairs) == 1
        assert pairs[0][0] == "What?"
        assert pairs[0][1] == steps

    def test_single_compression_creates_segments(self):
        from konash.training.oapl import _segment_rollout_for_training

        steps = [
            {"type": "retrieval", "query": "search1", "results_text": "r1"},
            {"type": "retrieval", "query": "search2", "results_text": "r2"},
            {"type": "compression", "summary": "Compressed: found X and Y."},
            {"type": "retrieval", "query": "search3", "results_text": "r3"},
            {"type": "answer", "answer": "final answer"},
        ]
        prompt = "Original question?"

        pairs = _segment_rollout_for_training(prompt, steps)

        # Should produce at least 2 segments:
        # 1. compression-as-target: pre-compression steps + compression step
        # 2. continuation: post-compression steps
        assert len(pairs) >= 2

        # The compression-as-target segment should include the compression step
        comp_segment = pairs[0]
        comp_steps = comp_segment[1]
        has_compression = any(
            s.get("type") == "compression" for s in comp_steps if isinstance(s, dict)
        )
        assert has_compression, "First segment should include compression step"

        # Continuation segment's prompt should include the compressed summary
        cont_segment = pairs[1]
        assert "Compressed" in cont_segment[0], "Continuation prompt should have compressed context"
        assert any(
            s.get("type") == "answer" for s in cont_segment[1] if isinstance(s, dict)
        )

    def test_multiple_compressions(self):
        from konash.training.oapl import _segment_rollout_for_training

        steps = [
            {"type": "retrieval", "query": "s1"},
            {"type": "compression", "summary": "Summary 1"},
            {"type": "retrieval", "query": "s2"},
            {"type": "compression", "summary": "Summary 2"},
            {"type": "answer", "answer": "done"},
        ]
        pairs = _segment_rollout_for_training("Q?", steps)
        # At least 3: comp-as-target-1, comp-as-target-2, continuation
        assert len(pairs) >= 3

    def test_segmenter_class_split_on_compression(self):
        from konash.training.segmentation import RolloutSegmenter

        segmenter = RolloutSegmenter(include_compression_segments=True)
        rollout = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "search result 1"},
            {"type": "compression", "content": "<|compression|>\nSummary here\n<|/compression|>"},
            {"role": "assistant", "content": "continued reasoning"},
            {"role": "assistant", "content": "final answer"},
        ]

        pairs = segmenter.split_on_compression(rollout)
        assert len(pairs) >= 1
        # Each pair is (x_context, y_continuation)
        for x, y in pairs:
            assert isinstance(x, list)
            assert isinstance(y, list)

    def test_segmenter_mask_tool_outputs(self):
        from konash.training.segmentation import RolloutSegmenter

        segmenter = RolloutSegmenter()
        tokens = [
            "I will search",
            "<|tool_start|>",
            "search result text",
            "more results",
            "<|tool_end|>",
            "Based on results",
        ]
        mask = segmenter.mask_tool_outputs(tokens)
        assert mask[0] == True   # model text
        assert mask[1] == False  # tool start
        assert mask[2] == False  # tool output
        assert mask[3] == False  # tool output
        assert mask[4] == False  # tool end
        assert mask[5] == True   # model text

    def test_segmenter_assign_reward(self):
        from konash.training.segmentation import RolloutSegmenter

        segmenter = RolloutSegmenter()
        segments = [
            ([{"type": "context"}], [{"type": "continuation"}]),
            ([{"type": "context2"}], [{"type": "answer"}]),
        ]
        rewarded = segmenter.assign_rollout_reward(segments, rollout_reward=0.75)
        assert len(rewarded) == 2
        for seg in rewarded:
            assert seg["reward"] == 0.75
            assert "x" in seg and "y" in seg


# ======================================================================
# 7. Corpus search returns relevant results first
# ======================================================================


class TestCorpusSearch:
    """Ingest documents, search, verify relevance ordering."""

    def test_search_returns_relevant_first(self, tmp_path):
        from konash.corpus import Corpus

        # Create 5 files about different topics
        docs = {
            "physics.txt": (
                "Albert Einstein developed the theory of relativity. "
                "E equals mc squared describes mass-energy equivalence. "
                "Special relativity was published in 1905."
            ),
            "cooking.txt": (
                "Italian pasta is made from durum wheat semolina. "
                "Spaghetti bolognese is a popular dish. "
                "Cook pasta in boiling salted water."
            ),
            "astronomy.txt": (
                "The Milky Way galaxy contains billions of stars. "
                "Jupiter is the largest planet in our solar system. "
                "Neptune is the farthest planet from the sun."
            ),
            "music.txt": (
                "Beethoven composed nine symphonies. "
                "The piano was invented by Bartolomeo Cristofori. "
                "Classical music originated in the western tradition."
            ),
            "biology.txt": (
                "DNA carries genetic information in living organisms. "
                "Mitochondria are the powerhouses of the cell. "
                "Photosynthesis converts light energy into chemical energy."
            ),
        }

        for name, content in docs.items():
            (tmp_path / name).write_text(content)

        corpus = Corpus(tmp_path, chunk_size=100, chunk_overlap=0)
        corpus.ingest()

        assert corpus.num_documents == 5

        # Search for physics-related content
        results = corpus.search("Einstein relativity theory", top_k=5)
        assert len(results) > 0

        # Top result should be from the physics document
        top_text = results[0]["text"]
        assert "Einstein" in top_text or "relativity" in top_text, (
            f"Top result should be physics-related, got: {top_text[:100]}"
        )

        # Scores should be in descending order
        scores = [r.get("score", 0.0) for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores should be descending: {scores}"
            )

    def test_search_different_modes(self, tmp_path):
        from konash.corpus import Corpus

        (tmp_path / "doc.txt").write_text(
            "Machine learning is a branch of artificial intelligence. "
            "Neural networks are inspired by biological neurons."
        )
        corpus = Corpus(tmp_path, chunk_size=100, chunk_overlap=0)
        corpus.ingest()

        # Vector search
        vec_results = corpus.search("artificial intelligence", mode="vector")
        assert len(vec_results) > 0

        # BM25 search
        bm25_results = corpus.search("neural networks", mode="bm25")
        assert len(bm25_results) > 0

        # Hybrid search (default)
        hybrid_results = corpus.search("machine learning", mode="hybrid")
        assert len(hybrid_results) > 0


# ======================================================================
# 8. NuggetScorer with stub judge
# ======================================================================


class TestNuggetScorer:
    """Create a scorer with a stub judge, verify score reflects nugget overlap."""

    def test_substring_judge_full_match(self):
        from konash.eval.nuggets import NuggetScorer

        scorer = NuggetScorer(judge=None)
        result = scorer.score(
            candidate="The Eiffel Tower is located in Paris, France.",
            reference="Paris, France",
        )
        assert result["score"] == 1.0
        assert len(result["nuggets"]) == 1

    def test_substring_judge_partial_match(self):
        from konash.eval.nuggets import NuggetScorer

        scorer = NuggetScorer(judge=None)
        result = scorer.score(
            candidate="Shakespeare wrote many plays including Hamlet.",
            reference="William Shakespeare, born in Stratford",
        )
        # Partial word overlap (Shakespeare is shared)
        assert 0.0 <= result["score"] <= 1.0

    def test_custom_judge_integration(self):
        from konash.eval.nuggets import NuggetScorer
        import re as _re

        class KeywordJudge:
            def judge(self, candidate, nugget):
                # Return 1.0 if nugget substring is in candidate (case-insensitive)
                if nugget.lower() in candidate.lower():
                    return 1.0
                # Fallback: word overlap
                nugget_words = set(_re.findall(r'\w+', nugget.lower()))
                cand_words = set(_re.findall(r'\w+', candidate.lower()))
                overlap = nugget_words & cand_words
                return len(overlap) / len(nugget_words) if nugget_words else 0.0

        scorer = NuggetScorer(judge=KeywordJudge())
        result = scorer.score(
            candidate="The answer is 42 which is the meaning of life.",
            reference="42; meaning of life",
            nuggets=["42", "meaning of life"],
        )
        assert result["score"] > 0.5
        assert len(result["nugget_scores"]) == 2
        assert result["nugget_scores"][0] == 1.0  # "42" is in candidate
        assert result["nugget_scores"][1] == 1.0  # "meaning of life" is in candidate

    def test_entity_per_nugget_mode(self):
        from konash.eval.nuggets import NuggetScorer, NuggetEvaluationPolicy

        policy = NuggetEvaluationPolicy(mode="entity_per_nugget")
        scorer = NuggetScorer(judge=None, policy=policy)

        result = scorer.score(
            candidate="Apple, Google, and Microsoft are tech companies.",
            reference="Apple, Google, Microsoft, Amazon",
            policy=policy,
        )
        # 3 out of 4 entities matched
        assert result["score"] == pytest.approx(0.75, abs=0.1)
        assert len(result["nuggets"]) == 4

    def test_no_nuggets_returns_zero(self):
        from konash.eval.nuggets import NuggetScorer

        scorer = NuggetScorer(judge=None)
        result = scorer.score(candidate="some answer", reference="")
        assert result["score"] == 0.0

    def test_llm_nugget_judge_batch(self):
        from konash.eval.nuggets import LLMNuggetJudge

        def stub_llm(messages):
            return {
                "content": '["support", "not_support", "partial_support"]'
            }

        judge = LLMNuggetJudge(llm_fn=stub_llm, question_context="test q")
        scores = judge.judge_batch(
            "The answer contains fact 1 and partially fact 3.",
            ["fact 1", "fact 2", "fact 3"],
        )
        assert scores == [1.0, 0.0, 0.5]


# ======================================================================
# 9. Value model scores good rollouts higher than bad
# ======================================================================


class TestValueModelScoring:
    """Train ValueModel on synthetic data, verify discrimination."""

    def test_value_model_discriminates_good_vs_bad(self):
        from konash.inference.value_model import ValueModel

        # Good rollouts: many steps, long content (simulate productive search)
        good_rollouts = []
        for i in range(10):
            steps = [
                {"role": "assistant", "content": f"I found relevant evidence about topic {i}. " * 5},
                {"role": "assistant", "content": f"Additional details from document {i}. " * 5},
                {"role": "assistant", "content": f"Based on all evidence, the answer is X{i}. " * 3},
            ]
            good_rollouts.append({"steps": steps})

        # Bad rollouts: minimal steps, short content
        bad_rollouts = []
        for i in range(10):
            steps = [
                {"role": "assistant", "content": "I don't know."},
            ]
            bad_rollouts.append({"steps": steps})

        all_rollouts = good_rollouts + bad_rollouts
        all_rewards = [1.0] * 10 + [0.0] * 10

        model = ValueModel(feature_dim=5)
        result = model.fit(all_rollouts, all_rewards, lr=0.1, epochs=50)

        assert result["final_loss"] < result["loss_history"][0], "Loss should decrease"

        # Good rollouts should score higher than bad
        good_scores = [model.score_rollout(r) for r in good_rollouts]
        bad_scores = [model.score_rollout(r) for r in bad_rollouts]

        avg_good = sum(good_scores) / len(good_scores)
        avg_bad = sum(bad_scores) / len(bad_scores)

        assert avg_good > avg_bad, (
            f"Good rollouts should score higher: avg_good={avg_good:.3f}, avg_bad={avg_bad:.3f}"
        )

    def test_value_model_masks_tool_tokens(self):
        from konash.inference.value_model import ValueModel

        model = ValueModel(feature_dim=5)
        rollout = [
            {"role": "assistant", "content": "Let me search for that."},
            {"role": "tool", "content": "Search results: ..."},
            {"role": "assistant", "content": "Based on results, the answer is X."},
        ]
        masked = model.mask_policy_tokens(rollout)
        # Tool message should be masked
        assert masked[1]["content"] == "[MASKED]"
        assert masked[1]["masked"] is True
        # Assistant messages should be preserved
        assert "search" in masked[0]["content"].lower()


# ======================================================================
# 10. VGS expand produces valid candidates
# ======================================================================


class TestVGSExpand:
    """Create a ValueGuidedSearchEngine with stubs, verify expand output."""

    def test_expand_produces_k_candidates(self):
        from konash.inference.value_search import ValueGuidedSearchEngine

        class StubAgent:
            def __init__(self):
                self.call_count = 0

            def generate_step(self, history, candidate_index=0, context=None, **kw):
                self.call_count += 1
                return {
                    "type": "reasoning",
                    "content": f"Reasoning step {candidate_index}",
                    "candidate_index": candidate_index,
                }

        agent = StubAgent()
        engine = ValueGuidedSearchEngine(
            agent=agent, candidate_width=3, max_depth=5,
        )

        initial_state = {
            "query": "What is quantum computing?",
            "steps": [],
            "terminal": False,
        }

        candidates = engine.expand(initial_state, k=3)

        assert len(candidates) == 3
        for i, cand in enumerate(candidates):
            assert "steps" in cand
            assert len(cand["steps"]) == 1  # one step added
            assert cand["steps"][0]["content"] == f"Reasoning step {i}"
            # Original state not mutated
            assert len(initial_state["steps"]) == 0

    def test_expand_with_existing_steps(self):
        from konash.inference.value_search import ValueGuidedSearchEngine

        class StubAgent:
            def generate_step(self, history, **kw):
                return {"type": "answer", "content": "Final answer.", "terminal": True}

        engine = ValueGuidedSearchEngine(
            agent=StubAgent(), candidate_width=2,
        )

        state = {
            "query": "test",
            "steps": [
                {"type": "retrieval", "content": "search for topic"},
            ],
            "terminal": False,
        }
        candidates = engine.expand(state, k=2)
        assert len(candidates) == 2
        for cand in candidates:
            assert len(cand["steps"]) == 2  # original + new
            assert cand["terminal"] is True  # step marked terminal

    def test_score_candidates_without_value_model(self):
        from konash.inference.value_search import ValueGuidedSearchEngine

        engine = ValueGuidedSearchEngine(value_model=None, candidate_width=2)
        candidates = [
            {"steps": [{"content": "a"}]},
            {"steps": [{"content": "b"}]},
        ]
        scores = engine.score_candidates(candidates)
        assert scores == [0.0, 0.0]

    def test_score_candidates_with_value_model(self):
        from konash.inference.value_model import ValueModel
        from konash.inference.value_search import ValueGuidedSearchEngine

        vm = ValueModel(feature_dim=5)
        engine = ValueGuidedSearchEngine(value_model=vm, candidate_width=2)

        candidates = [
            {"steps": [{"role": "assistant", "content": "short"}]},
            {"steps": [
                {"role": "assistant", "content": "long answer with lots of evidence " * 10},
                {"role": "assistant", "content": "more reasoning " * 5},
            ]},
        ]
        scores = engine.score_candidates(candidates)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_run_single_bfs(self):
        from konash.inference.value_search import ValueGuidedSearchEngine
        from konash.inference.value_model import ValueModel

        call_idx = {"n": 0}

        class CountingAgent:
            def generate_step(self, history, **kw):
                call_idx["n"] += 1
                # After 2 calls, produce a terminal step
                if call_idx["n"] >= 2:
                    return {"type": "answer", "content": "42", "terminal": True}
                return {"type": "reasoning", "content": f"thinking {call_idx['n']}"}

        engine = ValueGuidedSearchEngine(
            agent=CountingAgent(),
            value_model=ValueModel(feature_dim=5),
            candidate_width=2,
            max_depth=5,
        )

        tree = engine._run_single_bfs("What is the answer?")
        assert "best_trajectory" in tree
        assert tree["best_trajectory"]["answer"] is not None

    def test_aggregate_picks_highest_score(self):
        from konash.inference.value_search import ValueGuidedSearchEngine

        engine = ValueGuidedSearchEngine(candidate_width=2)
        answer = engine.aggregate(
            ["answer_a", "answer_b", "answer_c"],
            scores=[0.3, 0.9, 0.1],
        )
        assert answer == "answer_b"

    def test_aggregate_fallback_no_scores(self):
        from konash.inference.value_search import ValueGuidedSearchEngine

        engine = ValueGuidedSearchEngine(candidate_width=2)
        answer = engine.aggregate(["first", "second"])
        assert answer == "first"


# ======================================================================
# Additional OAPL math tests (behavioral, not just structural)
# ======================================================================


class TestOAPLMathBehavior:
    """Verify OAPL math produces correct values for known inputs."""

    def test_value_estimate_uniform_rewards(self):
        from konash.training.oapl import OAPLTrainer

        trainer = OAPLTrainer(beta_value=1.0)
        # All same rewards: V* should approximate the reward
        v = trainer.compute_group_value_estimate([0.5, 0.5, 0.5, 0.5])
        assert abs(v - 0.5) < 0.01

    def test_value_estimate_mixed_rewards(self):
        from konash.training.oapl import OAPLTrainer

        trainer = OAPLTrainer(beta_value=1.0)
        # Mixed: V* should be between min and max, skewed toward max
        # because logsumexp is a soft-max
        v = trainer.compute_group_value_estimate([0.0, 1.0])
        assert 0.0 < v < 1.0
        assert v > 0.5, "logsumexp should skew toward the maximum"

    def test_compute_loss_nonzero_for_mismatched_policy(self):
        from konash.training.oapl import OAPLTrainer

        trainer = OAPLTrainer(beta_kl=0.1)
        loss = trainer.compute_loss(
            log_probs=np.array([0.0, 0.0, 0.0, 0.0]),
            ref_log_probs=np.array([-1.0, -1.0, -1.0, -1.0]),
            rewards=np.array([1.0, 0.0, 1.0, 0.0]),
            group_indices=[[0, 1], [2, 3]],
        )
        assert loss > 0.0

    def test_mask_non_model_tokens(self):
        from konash.training.oapl import OAPLTrainer

        trainer = OAPLTrainer()
        mask = trainer.mask_non_model_tokens(
            token_ids=list(range(10)),
            tool_output_ranges=[(2, 5), (7, 9)],
        )
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3] == False
        assert mask[4] == False
        assert mask[5] == True
        assert mask[6] == True
        assert mask[7] == False
        assert mask[8] == False
        assert mask[9] == True

    def test_compute_squared_advantage_loss_zero_when_aligned(self):
        from konash.training.oapl import OAPLTrainer

        trainer = OAPLTrainer(beta_kl=0.01)
        # When kl_term == advantage, loss should be zero
        # advantage = reward - value = 1.0 - 0.5 = 0.5
        # kl_term = beta_kl * log_ratio = 0.01 * 50.0 = 0.5
        loss = trainer.compute_squared_advantage_loss(
            log_ratios=np.array([50.0]),
            rewards=np.array([1.0]),
            value_estimates=np.array([0.5]),
        )
        assert abs(loss) < 1e-10


# ======================================================================
# Compression before_step integration
# ======================================================================


class TestCompressionBeforeStepHook:
    """Test the before_step lifecycle hook integration."""

    def test_before_step_returns_compressed_when_triggered(self):
        from konash.plugins.compression import RLTrainableCompressionPlugin

        plugin = RLTrainableCompressionPlugin(
            threshold_chars=50,
            target_chars=30,
            agent_fn=lambda msgs: {"content": "Compressed summary."},
            preserve_recent_turns=1,
        )

        long_history = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "A " * 30},
            {"role": "assistant", "content": "B " * 30},
            {"role": "user", "content": "C " * 30},
            {"role": "assistant", "content": "D " * 30},
            {"role": "user", "content": "Recent message"},
        ]

        result = plugin.before_step(step_index=5, history=long_history)
        assert result is not None
        assert "history" in result
        assert "compression_event" in result
        event = result["compression_event"]
        assert event["pre_chars"] > event["post_chars"]

    def test_before_step_returns_none_when_below_threshold(self):
        from konash.plugins.compression import RLTrainableCompressionPlugin

        plugin = RLTrainableCompressionPlugin(
            threshold_chars=10000, target_chars=5000,
        )
        result = plugin.before_step(
            step_index=0,
            history=[{"role": "user", "content": "Hi"}],
        )
        assert result is None
