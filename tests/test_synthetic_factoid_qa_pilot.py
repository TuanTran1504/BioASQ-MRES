import copy

import pytest

from cse_dpo.build_synthetic_factoid_qa_pilot import (
    normalized_span_contains,
    validate_generated,
)


def _source(marked_text: str = "Drug X inhibits Protein Y. Pathway Z mediates repair.") -> dict:
    return {"target": {"marked_spans": [marked_text]}}


def _items() -> list[dict]:
    return [
        {
            "question": "Which treatment inhibits Protein Y?",
            "answer": "Drug X",
            "answer_type": "drug_or_treatment",
            "required_relation": "treatment that inhibits Protein Y",
            "essential_qualifiers": [],
            "support_quote": "Drug X inhibits Protein Y.",
            "basis": "The evidence explicitly states the inhibition relation.",
        },
        {
            "question": "Which pathway mediates repair?",
            "answer": "Pathway Z",
            "answer_type": "biological_process",
            "required_relation": "pathway that mediates repair",
            "essential_qualifiers": [],
            "support_quote": "Pathway Z mediates repair.",
            "basis": "The evidence explicitly identifies the repair pathway.",
        },
    ]


def _envelope(items=None) -> dict:
    return {
        "eligible": True,
        "eligibility_basis": "Two different explicit biomedical relations are present.",
        "items": _items() if items is None else items,
    }


def test_accepts_valid_two_relation_generation() -> None:
    assert validate_generated(_envelope(), _source())["eligible"] is True


def test_accepts_explicitly_ineligible_source() -> None:
    value = {
        "eligible": False,
        "eligibility_basis": "Only one usable factoid relation is stated.",
        "items": [],
    }
    assert validate_generated(value, _source())["items"] == []


def test_rejects_list_answer() -> None:
    source = _source("Twist, Tilt, Rise, Roll, Shift, and Slide are parameters. Pathway Z mediates repair.")
    items = _items()
    items[0].update({
        "question": "Which parameters describe the motion?",
        "answer": "Twist, Tilt, Rise, Roll, Shift, and Slide",
        "answer_type": "other",
        "support_quote": "Twist, Tilt, Rise, Roll, Shift, and Slide are parameters.",
    })
    with pytest.raises(ValueError, match="list rather than one factoid"):
        validate_generated(_envelope(items), source)


def test_rejects_metadata_question() -> None:
    items = _items()
    items[0]["question"] = "Which treatment does the article identify as inhibiting Protein Y?"
    with pytest.raises(ValueError, match="document metadata"):
        validate_generated(_envelope(items), _source())


def test_rejects_reordered_answer_leakage() -> None:
    source = _source(
        "Brodifacoum-laced synthetic marijuana toxicity was reported. Pathway Z mediates repair."
    )
    items = _items()
    items[0].update({
        "question": "Which toxicity is synthetic marijuana brodifacoum laced?",
        "answer": "Brodifacoum-laced synthetic marijuana toxicity",
        "answer_type": "disease_or_condition",
        "support_quote": "Brodifacoum-laced synthetic marijuana toxicity was reported.",
    })
    with pytest.raises(ValueError, match="repeats all answer tokens"):
        validate_generated(_envelope(items), source)


def test_rejects_process_type_assigned_to_method() -> None:
    source = _source("Fluorescence imaging reveals Protein Y. Pathway Z mediates repair.")
    items = _items()
    items[0].update({
        "question": "Which method reveals Protein Y?",
        "answer": "Fluorescence imaging",
        "answer_type": "biological_process",
        "support_quote": "Fluorescence imaging reveals Protein Y.",
    })
    with pytest.raises(ValueError, match="conflicts with its answer_type"):
        validate_generated(_envelope(items), source)


def test_rejects_two_descriptions_of_same_relation() -> None:
    items = _items()
    items[0]["required_relation"] = "target protein inhibition relation"
    items[1]["required_relation"] = "protein target inhibition relation"
    with pytest.raises(ValueError, match="required relations are too similar"):
        validate_generated(_envelope(items), _source())


def test_directional_containment_preserves_boundaries() -> None:
    assert normalized_span_contains(
        "Ublituximab is an anti-CD20 antibody", "anti-CD20 antibody"
    )
    assert not normalized_span_contains("ABCD", "ABC")
    assert not normalized_span_contains("prostate cancer", "castration-resistant prostate cancer")
