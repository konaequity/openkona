from __future__ import annotations

import json
import pytest
from argparse import Namespace
from types import SimpleNamespace

from konash.cli import _should_use_sleep_wake
from konash.synthesis.qa import QuestionAnswerSynthesizer, SyntheticExample
from scripts.train_oapl_unsloth import (
    _VectorSearchOnlyTool,
    _infer_karl_task_name,
    _load_eval_seed_examples,
    _load_eval_question_texts,
    _plan_synthesis_call_targets,
    _resolve_rollouts_per_example,
    _sample_seed_documents,
    _should_sleep_vllm_for_training,
    _spread_indices,
    _target_synthesis_examples,
    _trim_messages_for_context,
)
from konash.training.execution import plan_training_execution
from konash.training.project_state import configure_training_project
from konash.training.resume import (
    load_stage1_resume_examples,
    normalize_stage_resume_args,
    resolve_rollouts_artifact_path,
)


def test_remote_full_single_iteration():
    plan = plan_training_execution(iterations=1, synthesis_rollout_backend="remote_full")

    assert plan.synthesis_rollout_backend == "remote_full"
    assert plan.requires_remote_full_pipeline is True
    assert plan.requested_mode == "remote_full"


def test_remote_full_multi_iteration():
    plan = plan_training_execution(iterations=2, synthesis_rollout_backend="remote_full")

    assert plan.synthesis_rollout_backend == "remote_full"
    assert plan.requires_remote_full_pipeline is True
    assert plan.requested_mode == "remote_full"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Training only supports remote_full"):
        plan_training_execution(iterations=1, synthesis_rollout_backend="invalid")


def test_auto_is_rejected():
    with pytest.raises(ValueError, match="Training only supports remote_full"):
        plan_training_execution(iterations=1, synthesis_rollout_backend="auto")


def test_sleep_wake_enabled_for_multi_iteration():
    plan = plan_training_execution(
        iterations=3, synthesis_rollout_backend="remote_full", sleep_wake=True,
    )
    assert plan.supports_sleep_wake is True


def test_sleep_wake_disabled_for_single_iteration():
    plan = plan_training_execution(
        iterations=1, synthesis_rollout_backend="remote_full", sleep_wake=True,
    )
    assert plan.supports_sleep_wake is False


def test_sleep_wake_defaults_to_false():
    plan = plan_training_execution(iterations=2, synthesis_rollout_backend="remote_full")
    assert plan.supports_sleep_wake is False


def test_cli_prefers_sleep_wake_for_glm_on_h200():
    assert _should_use_sleep_wake("zai-org/GLM-4.5-Air-FP8", "H200") is True
    assert _should_use_sleep_wake("zai-org/GLM-4.5-Air-FP8", "H100") is True
    assert _should_use_sleep_wake("zai-org/GLM-5", "H200") is False


def test_target_synthesis_examples_uses_full_batch_without_cap():
    assert _target_synthesis_examples(current_count=0, max_examples=None) == 8


def test_target_synthesis_examples_respects_remaining_cap():
    assert _target_synthesis_examples(current_count=0, max_examples=3) == 3
    assert _target_synthesis_examples(current_count=2, max_examples=3) == 1


def test_target_synthesis_examples_returns_zero_when_done():
    assert _target_synthesis_examples(current_count=3, max_examples=3) == 0
    assert _target_synthesis_examples(current_count=4, max_examples=3) == 0


def test_plan_synthesis_call_targets_respects_cap():
    assert _plan_synthesis_call_targets(synthesis_calls=2, max_examples=12) == [6, 6]
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=3) == [1, 1, 1]


def test_plan_synthesis_call_targets_spreads_work_across_workers():
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=12) == [3, 3, 3, 3]
    assert _plan_synthesis_call_targets(synthesis_calls=4, max_examples=10) == [3, 3, 2, 2]


def test_plan_synthesis_call_targets_without_cap_uses_all_calls():
    assert _plan_synthesis_call_targets(synthesis_calls=3, max_examples=None) == [8, 8, 8]


def test_infer_karl_task_name_from_corpus_path():
    assert _infer_karl_task_name("/tmp/browsecomp-plus/documents") == "BrowseCompPlus"
    assert _infer_karl_task_name("/tmp/trec-biogen/documents") == "TRECBiogen"
    assert _infer_karl_task_name("/tmp/financebench/documents") is None


def test_load_eval_question_texts_reads_parent_of_documents(tmp_path):
    corpus_root = tmp_path / "browsecomp-plus"
    docs_dir = corpus_root / "documents"
    docs_dir.mkdir(parents=True)
    (corpus_root / "eval_questions.json").write_text(
        '[{"question": "Q1"}, {"question": "Q2"}, {"question": "Q1"}]',
        encoding="utf-8",
    )

    assert _load_eval_question_texts(str(docs_dir)) == ["Q1", "Q2"]


def test_load_eval_seed_examples_reads_answered_examples(tmp_path):
    corpus_root = tmp_path / "browsecomp-plus"
    docs_dir = corpus_root / "documents"
    docs_dir.mkdir(parents=True)
    (corpus_root / "eval_questions.json").write_text(
        json.dumps(
            [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
                {"question": "Q3", "answer": "A3"},
                {"question": "Q4", "answer": "A4"},
            ]
        ),
        encoding="utf-8",
    )

    examples = _load_eval_seed_examples(str(docs_dir), 2)

    assert len(examples) == 2
    assert all(isinstance(ex, SyntheticExample) for ex in examples)
    assert [(ex.question, ex.answer) for ex in examples] == [("Q1", "A1"), ("Q4", "A4")]


def test_spread_indices_evenly_across_collection():
    assert _spread_indices(10, 4) == [0, 3, 6, 9]
    assert _spread_indices(3, 5) == [0, 1, 2]


def test_sample_seed_documents_reads_spread_sources(tmp_path):
    files = []
    documents = []
    for idx, text in enumerate(["alpha text", "beta text", "gamma text", "delta text"]):
        path = tmp_path / f"doc{idx}.txt"
        path.write_text(text, encoding="utf-8")
        files.append(path)
        documents.append({"text": "", "source": str(path)})

    corpus = SimpleNamespace(documents=documents)

    sampled = _sample_seed_documents(corpus, 2)

    assert sampled == ["alpha text", "delta text"]


def test_vector_search_only_tool_forces_vector_mode():
    calls = []

    class DummyCorpus:
        def search(self, query, top_k=10, mode="hybrid"):
            calls.append((query, top_k, mode))
            return [{"text": "ok"}]

    tool = _VectorSearchOnlyTool(DummyCorpus())
    results = tool.search("Percy Jackson", top_k=5)

    assert results == [{"text": "ok"}]
    assert calls == [("Percy Jackson", 5, "vector")]


def test_resolve_rollouts_per_example_for_browsecomp():
    assert _resolve_rollouts_per_example(
        task_name="BrowseCompPlus",
        requested_rollouts=3,
    ) == 8
    assert _resolve_rollouts_per_example(
        task_name="TRECBiogen",
        requested_rollouts=6,
    ) == 6


def test_load_stage1_resume_examples_reads_dedup_payload(tmp_path):
    path = tmp_path / "stage1_deduped.json"
    path.write_text(
        '{"phase":"dedup","data":[{"question":"Q1","answer":"A1"}]}',
        encoding="utf-8",
    )

    resolved, phase, examples = load_stage1_resume_examples(str(path))

    assert resolved == str(path)
    assert phase == "dedup"
    assert [(ex.question, ex.answer) for ex in examples] == [("Q1", "A1")]


def test_resolve_rollouts_artifact_path_prefers_iter_rollouts(tmp_path):
    output_dir = tmp_path / "checkpoints"
    iter_dir = output_dir / "iter1"
    iter_dir.mkdir(parents=True)
    rollouts = iter_dir / "rollouts.json"
    rollouts.write_text("[]", encoding="utf-8")

    assert resolve_rollouts_artifact_path(str(output_dir)) == str(rollouts)


def test_normalize_stage_resume_args_maps_stage2_to_rollouts(tmp_path):
    output_dir = tmp_path / "checkpoints"
    iter_dir = output_dir / "iter2"
    iter_dir.mkdir(parents=True)
    rollouts = iter_dir / "rollouts.json"
    rollouts.write_text("[]", encoding="utf-8")
    args = Namespace(
        train_from_rollouts=None,
        rollouts=None,
        resume_stage2_from=str(output_dir),
        skip_rollouts=False,
        skip_synthesis=False,
        resume_stage1_from=None,
    )

    normalized = normalize_stage_resume_args(args)

    assert normalized.rollouts == str(rollouts)


def test_normalize_stage_resume_args_rejects_skip_synthesis_without_resume():
    args = Namespace(
        train_from_rollouts=None,
        rollouts=None,
        resume_stage2_from=None,
        skip_rollouts=False,
        skip_synthesis=True,
        resume_stage1_from=None,
    )

    with pytest.raises(SystemExit, match="--skip-synthesis"):
        normalize_stage_resume_args(args)


def test_configure_training_project_generates_browsecomp_name():
    configured = configure_training_project(
        base_model="zai-org/GLM-4.5-Air-FP8",
        corpora=["/tmp/browsecomp-plus/documents"],
        requested_output="./checkpoints",
        gpu_label="H200",
        rollouts_per_example=8,
        max_examples=12,
    )

    assert configured.project.startswith("browsecomp-glm45-h200-")
    assert "-r8-" in configured.project
    assert configured.project.endswith("-smoke")
    assert configured.display_name.startswith("BrowseComp-Plus")
    assert "R8" in configured.display_name
    assert configured.output_dir.endswith(f"/.konash/projects/{configured.project}/checkpoints")


def test_configure_training_project_keeps_explicit_project_and_output():
    configured = configure_training_project(
        base_model="zai-org/GLM-4.5-Air-FP8",
        corpora=["/tmp/financebench/documents"],
        requested_project="custom-project",
        requested_display_name="Custom Display",
        requested_output="/tmp/custom-checkpoints",
        rollouts_per_example=6,
        max_examples=32,
        run_tag="bringup",
    )

    assert configured.project == "custom-project"
    assert configured.display_name == "Custom Display"
    assert configured.output_dir == "/tmp/custom-checkpoints"


def test_should_sleep_vllm_for_intermediate_iteration():
    assert _should_sleep_vllm_for_training(iteration=0, total_iterations=2) is True
    assert _should_sleep_vllm_for_training(iteration=1, total_iterations=3) is True


def test_should_not_sleep_vllm_for_final_iteration():
    assert _should_sleep_vllm_for_training(iteration=0, total_iterations=1) is False
    assert _should_sleep_vllm_for_training(iteration=2, total_iterations=3) is False


def test_trim_messages_for_context_preserves_recent_history():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "u1 " + ("a" * 5000)},
        {"role": "assistant", "content": "a1 " + ("b" * 5000)},
        {"role": "tool", "content": "tool " + ("c" * 9000)},
        {"role": "user", "content": "latest question"},
    ]

    trimmed = _trim_messages_for_context(messages, max_context_tokens=2048)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "latest question"
    assert sum(len(m.get("content", "")) for m in trimmed) <= 4096
    assert any("[truncated]" in m.get("content", "") for m in trimmed)


def test_browsecomp_prompt_avoids_generic_topic_discovery():
    synth = QuestionAnswerSynthesizer(
        few_shot_examples=[SyntheticExample(question="Seed Q", answer="Seed A")],
        task_name="BrowseCompPlus",
        task_prompt=None,
        llm_fn=lambda messages, **kwargs: {"role": "assistant", "content": "SEARCH: x"},
    )

    system_prompt = synth._system_prompt()
    task_message = synth._task_message(2, ["Doc about a specific person and award."])

    assert "BrowseComp-Plus question-answer synthesis agent" in system_prompt
    assert "Do NOT issue broad topical searches" in system_prompt
    assert "discover what topics it covers" not in task_message
    assert "Do NOT start with broad topical queries" in task_message


def test_synthesis_prompts_do_not_enforce_minimum_search_count():
    generic = QuestionAnswerSynthesizer(
        llm_fn=lambda messages, **kwargs: {"role": "assistant", "content": "PROPOSE:\nQ: x\nA: y"},
    )
    browsecomp = QuestionAnswerSynthesizer(
        task_name="BrowseCompPlus",
        llm_fn=lambda messages, **kwargs: {"role": "assistant", "content": "PROPOSE:\nQ: x\nA: y"},
    )

    assert "MUST search at least" not in generic._system_prompt()
    assert "at least 3 times before proposing" not in generic._system_prompt()
    assert "Search for at least" not in generic._task_message(4, None)
    assert "at least" not in browsecomp._task_message(4, ["Doc about a named entity."])
