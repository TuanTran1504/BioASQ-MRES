from pathlib import Path
import json

import pytest

from cse_dpo.annotate_remaining_candidate_bank_questions import (
    DEFAULT_SUPPORTED_QUESTION_SOURCE,
    load_eligible_question_ids,
)
from cse_dpo.candidate_bank_class_judge import CandidateBankClassJudge, pair_dedupe_norm


def _record() -> dict:
    return {"snippets": [{"snippet_id": "1.1"}]}


def test_authoritative_supported_pool_has_1130_questions() -> None:
    ids = load_eligible_question_ids(DEFAULT_SUPPORTED_QUESTION_SOURCE)
    assert len(ids) == 1130


def test_c2_accepts_only_equivalent_relation_types() -> None:
    judge = CandidateBankClassJudge.__new__(CandidateBankClassJudge)
    value = {
        "class": "C2",
        "semantic_correct": True,
        "relation_type": "abbreviation_expansion",
        "evidence_support": "supported",
        "evidence_ids": ["1.1"],
        "error_type": "none",
        "basis": "The abbreviation and expansion denote the same entity.",
    }
    assert judge._validate(value, _record())["class"] == "C2"

    broader = {**value, "relation_type": "broader"}
    with pytest.raises(ValueError, match="equivalent relation_type"):
        judge._validate(broader, _record())


def test_c1_accepts_related_but_non_equivalent_answer() -> None:
    judge = CandidateBankClassJudge.__new__(CandidateBankClassJudge)
    value = {
        "class": "C1",
        "semantic_correct": False,
        "relation_type": "part_whole",
        "evidence_support": "supported",
        "evidence_ids": ["1.1"],
        "error_type": "part_whole",
        "basis": "The candidate is a component rather than the requested enzyme.",
    }
    assert judge._validate(value, _record())["class"] == "C1"


def test_deterministic_exact_key_preserves_punctuation() -> None:
    assert pair_dedupe_norm(">25") != pair_dedupe_norm("25")
    assert pair_dedupe_norm("CGRP receptor") == pair_dedupe_norm("cgrp   receptor")


def _retry_record():
    return {
        "question_id": "q", "question": "Which enzyme?", "gold_aliases": ["enzyme"],
        "candidate": "enzyme subunit", "candidate_output": "[BE]enzyme subunit[EE]",
        "source_model": "test", "response_id": "q-1", "sample_id": 1,
        "snippets": [{"snippet_id": "1.1", "pubmed_id": "1", "text": "enzyme subunit"}],
    }


def _invalid_judgment():
    return {
        "class": "C2", "semantic_correct": True, "relation_type": "part_whole",
        "evidence_support": "supported", "evidence_ids": ["1.1"], "error_type": "none",
        "basis": "The candidate is a component of the enzyme.",
    }


@pytest.mark.parametrize("resume", [False, True])
def test_retries_supply_rejected_response_and_class_rules(tmp_path, monkeypatch, resume):
    record = _retry_record()
    judge = CandidateBankClassJudge([record], tmp_path, tmp_path / "unused-key")
    bad = _invalid_judgment()
    corrected = {**bad, "class": "C1", "semantic_correct": False, "error_type": "part_whole"}
    key = judge._cache_key(record)
    if resume:
        (judge.error_dir / f"{key}.json").write_text(json.dumps({
            "error": "C2 must use an equivalent relation_type", "last_invalid_judgment": bad,
        }), encoding="utf-8")
    prompts = []
    replies = iter([corrected] if resume else [bad, corrected])

    def call(prompt, api_key):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(judge, "_call", call)
    monkeypatch.setattr("cse_dpo.candidate_bank_class_judge.time.sleep", lambda seconds: None)
    result = judge._annotate_one(record, "unused")
    assert result["class"] == "C1" and result["origin"] == "api"
    assert judge.new_api_calls == (1 if resume else 2)
    assert json.dumps(bad) in prompts[-1]
    assert "part_whole and extra_qualifier cannot be C2" in prompts[-1]
    assert "Do not preserve a class or change a relation merely to pass validation" in prompts[-1]
    # Valid results remain reusable even when a prior error file exists.
    assert judge._annotate_one(record, "unused")["origin"] == "cache"
    assert len(prompts) == (1 if resume else 2)


def test_exhausted_correction_never_caches_or_relabels_invalid_response(tmp_path, monkeypatch):
    record = _retry_record()
    judge = CandidateBankClassJudge([record], tmp_path, tmp_path / "unused-key", max_retries=1)
    monkeypatch.setattr(judge, "_call", lambda *args: _invalid_judgment())
    monkeypatch.setattr("cse_dpo.candidate_bank_class_judge.time.sleep", lambda seconds: None)
    result = judge._annotate_one(record, "unused")
    assert result["origin"] == "error" and result["class"] == "UNCERTAIN"
    assert judge.new_api_calls == 2
    assert not list(judge.cache_dir.iterdir())
    saved = json.loads(next(judge.error_dir.iterdir()).read_text(encoding="utf-8"))
    assert saved["last_invalid_judgment"] == _invalid_judgment()
