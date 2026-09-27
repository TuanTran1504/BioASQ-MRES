import json

from cse_dpo.candidate_bank_class_judge import build_pairs, pair_dedupe_norm


def _record(candidate: str, response_id: str) -> dict:
    return {
        "question_id": "q1",
        "question": "At what age does onset occur?",
        "gold_aliases": ["after age 25"],
        "candidate": candidate,
        "candidate_output": f"[BE]{candidate}[EE]",
        "source_model": "test-model",
        "response_id": response_id,
        "sample_id": response_id,
        "bank_prompt": "prompt",
        "snippets": [{"snippet_id": "1.1", "text": "Onset occurs after age 25."}],
    }


def _judgment(label: str) -> dict:
    return {
        "class": label,
        "semantic_correct": label != "C1",
        "related": True,
        "evidence_support": "supported",
        "evidence_ids": ["1.1"],
        "error_type": "none" if label != "C1" else "wrong value",
        "confidence": "high",
        "basis": "test",
        "origin": "test",
    }


def test_pair_dedupe_norm_preserves_punctuation() -> None:
    assert pair_dedupe_norm(">25") != pair_dedupe_norm("25")
    assert pair_dedupe_norm("EWS/FLI-1") != pair_dedupe_norm("EWS/FLI1")
    assert pair_dedupe_norm("  Migraine ") == pair_dedupe_norm("migraine")


def test_build_pairs_keeps_punctuation_distinct_candidates(tmp_path) -> None:
    records = [_record(">25", "r1"), _record("25", "r2")]
    judgments = [_judgment("C1"), _judgment("C1")]

    summary = build_pairs(
        records,
        judgments,
        tmp_path,
        inject_gold_c3=True,
        require_c3_extractive=True,
        gold_c3_policy="always_first_extractive_alias",
    )

    assert summary["pair_dedupe_normalization"] == "casefold_whitespace_punctuation_preserved_v1"
    assert summary["slate_class_counts"] == {"C3": 1, "C1": 2}
    assert summary["pair_class_counts"] == {"C3>C1": 2}
    pairs = [json.loads(line) for line in (tmp_path / "dpo_pairs.jsonl").read_text().splitlines()]
    assert {row["rejected_candidate"] for row in pairs} == {">25", "25"}
