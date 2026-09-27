from pathlib import Path

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
        "related": True,
        "equivalence": "equivalent",
        "relation_type": "abbreviation_expansion",
        "essential_qualifiers_preserved": True,
        "evidence_support": "supported",
        "evidence_ids": ["1.1"],
        "error_type": "none",
        "confidence": "high",
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
        "related": True,
        "equivalence": "not_equivalent",
        "relation_type": "part_whole",
        "essential_qualifiers_preserved": False,
        "evidence_support": "supported",
        "evidence_ids": ["1.1"],
        "error_type": "part_whole",
        "confidence": "high",
        "basis": "The candidate is a component rather than the requested enzyme.",
    }
    assert judge._validate(value, _record())["class"] == "C1"


def test_deterministic_exact_key_preserves_punctuation() -> None:
    assert pair_dedupe_norm(">25") != pair_dedupe_norm("25")
    assert pair_dedupe_norm("CGRP receptor") == pair_dedupe_norm("cgrp   receptor")
