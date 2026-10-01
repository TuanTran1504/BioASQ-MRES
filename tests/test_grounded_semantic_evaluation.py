from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from src.utility.eval_types import EvalExample
from src.utility.eval_openai import OpenAICandidateGenerator
from src.utility.eval_runner import maybe_run_grounded_semantic_evaluation
from src.utility.evaluation import parse_args
from src.utility.grounded_semantic_eval import (
    GroundedSemanticJudge,
    _decode_judge_response,
    _judge_response_format,
    evaluate_grounded_semantics,
)
from src.notebook_workflows import plan
from src.notebook_workflows.runner import ROOT


def judge_args(**overrides):
    values = {
        "semantic_judge_model": "judge-test",
        "semantic_judge_max_new_calls": 10,
        "semantic_judge_api_key_file": "unused.txt",
        "semantic_judge_endpoint": "https://example.invalid",
        "semantic_judge_timeout_seconds": 10,
        "semantic_judge_request_delay_seconds": 0.0,
        "semantic_judge_max_retries": 0,
        "semantic_judge_cache_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_exact_semantic_answer_requires_snippet_support(tmp_path):
    supported = EvalExample(
        question_id="supported",
        question_type="factoid",
        body="Which protein is targeted?",
        instruction="Answer",
        resources=("PubMed ID: 1\n[BS]The drug targets Protein A.[ES]",),
        gold_output="[BE]Protein A[EE]",
        source_path="dev.json",
    )
    unsupported = EvalExample(
        question_id="unsupported",
        question_type="factoid",
        body="Which protein is targeted?",
        instruction="Answer",
        resources=("PubMed ID: 2\n[BS]The study measured survival.[ES]",),
        gold_output="[BE]Protein B[EE]",
        source_path="dev.json",
    )
    rows = [
        {"question_id": "supported", "question_type": "factoid", "prediction": "[BE]Protein A[EE]"},
        {"question_id": "unsupported", "question_type": "factoid", "prediction": "[BE]Protein B[EE]"},
    ]
    examples = {(row.question_id, "factoid"): row for row in (supported, unsupported)}
    summary = evaluate_grounded_semantics(
        prediction_rows=rows,
        examples_by_key=examples,
        args=judge_args(),
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        budget_state={"new_judge_calls": 0, "judge_retry_count": 0},
    )
    assert summary["semantic_accuracy"] == 1.0
    assert summary["grounded_semantic_accuracy"] == 0.5
    judgments = [
        json.loads(line)
        for line in (tmp_path / "out/candidate_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["evidence_label"] for row in judgments] == ["supported", "insufficient"]
    assert summary["shared_budget"]["new_judge_calls"] == 0


def test_nonexact_equivalent_answer_keeps_semantics_and_grounding_separate(tmp_path, monkeypatch):
    judge = GroundedSemanticJudge(
        args=judge_args(),
        cache_dir=tmp_path / "cache",
        budget_state={"new_judge_calls": 0, "judge_retry_count": 0},
    )
    monkeypatch.setattr(
        judge,
        "_call",
        lambda record: (
            {
                "semantic_label": "equivalent",
                "evidence_label": "insufficient",
                "evidence_ids": [],
                "relation_type": "abbreviation_expansion",
                "basis": "The wording is equivalent but the supplied snippet does not establish the answer.",
            },
            "api",
        ),
    )
    result = judge.judge(
        {
            "question_id": "q",
            "question": "What receptor?",
            "gold_aliases": ["CGRP receptor"],
            "candidate": "calcitonin gene-related peptide receptor",
            "rank": 1,
            "snippets": [],
        }
    )
    assert result["semantic_correct"] is True
    assert result["grounded_correct"] is False


def test_semantic_judge_uses_strict_enumerated_response_schema():
    response_format = _judge_response_format({"1.1", "2.1"})
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    properties = schema["properties"]
    assert set(properties) == set(schema["required"])
    semantic_relations = set(properties["semantic_relation"]["enum"])
    assert "equivalent:synonym" in semantic_relations
    assert "not_equivalent:narrower" in semantic_relations
    assert "uncertain:uncertain" in semantic_relations
    assert "equivalent:narrower" not in semantic_relations
    assert "not_equivalent:synonym" not in semantic_relations
    assert set(properties["evidence_label"]["enum"]) == {
        "supported",
        "contradicted",
        "insufficient",
    }
    assert properties["evidence_ids"]["items"]["enum"] == ["1.1", "2.1"]


def test_semantic_relation_decodes_to_consistent_public_fields():
    decoded = _decode_judge_response(
        {
            "semantic_relation": "equivalent:abbreviation_expansion",
            "evidence_label": "supported",
            "evidence_ids": ["1.1"],
            "basis": "The candidate expands the accepted abbreviation.",
        },
        {"1.1"},
    )
    assert decoded["semantic_label"] == "equivalent"
    assert decoded["relation_type"] == "abbreviation_expansion"


def test_insufficient_evidence_citations_do_not_discard_semantic_decision():
    decoded = _decode_judge_response(
        {
            "semantic_relation": "not_equivalent:narrower",
            "evidence_label": "insufficient",
            "evidence_ids": ["1.1"],
            "basis": "The candidate is narrower than the accepted answer.",
        },
        {"1.1"},
    )
    assert decoded["semantic_label"] == "not_equivalent"
    assert decoded["relation_type"] == "narrower"
    assert decoded["evidence_ids"] == []


def test_notebook_plan_supports_api_only_and_mixed_candidates(tmp_path):
    source = tmp_path / "dev.json"
    source.write_text("[]", encoding="utf-8")
    common = {
        "eval_input": [str(source)],
        "prompt_file": str(ROOT / "prompts/factoid_single_answer_aligned.json"),
        "model_ref": ["openai:gpt-test"],
        "max_new_api_calls": 5,
    }
    api_only = plan("local_evaluation", common, project_root=tmp_path)
    assert api_only["api"] is True
    assert api_only["gpu"] is False
    assert not api_only["missing"]

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    mixed = plan(
        "local_evaluation",
        {
            **common,
            "model_ref": [str(adapter), "openai:gpt-test"],
            "semantic_judge": True,
            "semantic_judge_max_new_calls": 20,
        },
        project_root=tmp_path,
    )
    assert mixed["api"] is True
    assert mixed["gpu"] is True
    command = " ".join(mixed["command"])
    assert "--model-ref openai:gpt-test" in command
    assert "--semantic-judge" in command


def test_cli_routes_prefixed_model_refs_to_openai(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_models",
            "--eval-input",
            "dev.json",
            "--model-ref",
            "local-adapter",
            "openai:gpt-test",
            "--openai-model",
            "gpt-test",
            "--max-new-api-calls",
            "5",
        ],
    )
    args = parse_args()
    assert args.model_ref == ["local-adapter"]
    assert args.openai_model == ["gpt-test"]


def test_grounded_judge_rejects_evidence_hidden_by_prompt_truncation(tmp_path):
    args = judge_args(semantic_judge=True)
    with pytest.raises(ValueError, match="could see"):
        maybe_run_grounded_semantic_evaluation(
            prediction_rows=[{
                "question_id": "q",
                "question_type": "factoid",
                "prediction": "[BE]alpha[EE]",
                "prompt_truncation": {"prompt_truncated": True},
            }],
            examples_by_key={},
            args=args,
            model_dir=tmp_path / "model",
            output_root=tmp_path,
            budget_state={"new_judge_calls": 0, "judge_retry_count": 0},
        )


def test_openai_candidate_cache_avoids_repeated_paid_call(tmp_path, monkeypatch):
    args = argparse.Namespace(
        do_sample=False,
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        num_generations=1,
        aggregation_strategy="union",
        aggregation_min_frequency=1,
        max_factoid_answers=5,
        max_new_api_calls=1,
        api_key_file="unused.txt",
        openai_endpoint="https://example.invalid",
        api_timeout_seconds=10,
        api_request_delay_seconds=0.0,
        api_max_retries=0,
    )
    example = EvalExample(
        question_id="q",
        question_type="factoid",
        body="Which protein?",
        instruction="Return [BE] answer [EE]",
        resources=("PubMed ID: 1\n[BS]Protein A.[ES]",),
        gold_output="[BE]Protein A[EE]",
        source_path="dev.json",
    )
    state = {"new_api_calls": 0, "retry_count": 0}
    first = OpenAICandidateGenerator(
        model="gpt-test", args=args, cache_dir=tmp_path / "cache", budget_state=state
    )
    monkeypatch.setattr(first, "_call", lambda payload: "[BE]Protein A[EE]")
    assert first.generate(example)[0] == "[BE]Protein A[EE]"

    second = OpenAICandidateGenerator(
        model="gpt-test", args=args, cache_dir=tmp_path / "cache", budget_state=state
    )
    monkeypatch.setattr(second, "_call", lambda payload: (_ for _ in ()).throw(AssertionError("API called")))
    prediction, _, telemetry = second.generate(example)
    assert prediction == "[BE]Protein A[EE]"
    assert telemetry[0]["origin"] == "cache"
