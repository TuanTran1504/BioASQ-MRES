"""Integrity and leakage controls for the inspected official-form pilot export."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_generator_form_pilot as pilot


def examples():
    return [{"pair_id": f"{m}-{i}", "question_id": f"q{i}", "model_key": m, "split": "train",
        "objective": "sampled_response_coverage", "review_category": "benchmark_form_preference",
        "synthetic_chosen": False, "prompt": [{"role": "system", "content": "Use evidence"},
            {"role": "user", "content": "Question and evidence only"}],
        "chosen": ' {"answers":[{"answer":"X","relation_type":"original"}]} ',
        "rejected": '{"answers":[{"answer":"X name","relation_type":"original"}]}',
        "chosen_candidates": [{"answer": "X", "official_accepted": True}],
        "rejected_candidates": [{"answer": "X name", "official_accepted": False}],
        "chosen_candidate_count": 1, "rejected_candidate_count": 1,
        "snippets": [{"id": "1", "text": "The source says X is also called X name."}],
        "counterfactual_ensemble_coverage_gain": 0}
        for m in pilot.data.MODELS for i in range(8)]


def inspection(rows):
    return {"objective": "official_answer_form_coverage", "seed": 3407, "questions_per_model": 24,
        "decisions": {q: {"decision": "retain", "reason": "Same entity, supported names",
            "evidence": {"snippet_id": "1", "quote": "X is also called X name"}}
            for q in pilot.sample_questions(rows)}}


def test_dedup_preserves_exact_native_strings_and_objective_provenance():
    rows = examples()
    duplicate = {**rows[0], "pair_id": "another-draw", "objective": "greedy_recovery"}
    result, exclusions, _ = pilot.select([*rows, duplicate], inspection(rows))
    row = next(p for p in result if rows[0]["pair_id"] in p["source_pair_ids"])
    assert row["chosen"] == rows[0]["chosen"]  # Intentional leading/trailing whitespace survives.
    assert set(row["source_pair_ids"]) == {rows[0]["pair_id"], "another-draw"}
    assert set(row["source_objectives"]) == {"greedy_recovery", "sampled_response_coverage"}
    assert len(result) == len(rows)
    assert len(exclusions) == 1


def test_unreviewed_and_held_questions_cannot_enter_pilot():
    rows = examples()
    review = inspection(rows)
    review["decisions"]["q0"]["decision"] = "hold"
    extra = {**rows[0], "pair_id": "outside", "question_id": "outside"}
    # The inspection panel itself is fixed; unexpected unreviewed source questions
    # can exist only when they fall outside the sampled panel.
    with pytest.raises(ValueError, match="entire prespecified sample"):
        pilot.select([*rows, extra], review)
    selected, exclusions, _ = pilot.select(rows, review)
    assert not any(p["question_id"] == "q0" for p in selected)
    assert sum(e["reason"] == "assistant_scope_or_evidence_hold" for e in exclusions) == 3


@pytest.mark.parametrize("change", ["missing_quote", "false_positive", "negative_is_accepted", "synthetic"])
def test_bad_evidence_or_preference_direction_fails_closed(change):
    rows, review = examples(), inspection(examples())
    if change == "missing_quote":
        review["decisions"]["q0"]["evidence"]["quote"] = "invented evidence"
    elif change == "false_positive":
        rows[0]["chosen_candidates"][0]["official_accepted"] = False
    elif change == "negative_is_accepted":
        rows[0]["rejected_candidates"][0]["official_accepted"] = True
    else:
        rows[0]["synthetic_chosen"] = True
    with pytest.raises(ValueError):
        pilot.select(rows, review)


def test_question_holdout_is_shared_across_models_and_order_independent():
    rows = examples()
    a, b = pilot.assign_splits(rows), pilot.assign_splits(list(reversed(rows)))
    assert a == b
    assert set(a.values()) == {"train", "inner_validation"}
    for m in pilot.data.MODELS:
        train = {p["question_id"] for p in rows if p["model_key"] == m and a[p["question_id"]] == "train"}
        holdout = {p["question_id"] for p in rows if p["model_key"] == m and a[p["question_id"]] != "train"}
        assert not train & holdout
        assert len(holdout) >= 2


def test_export_control_exposure_and_native_strings_then_detect_tamper(tmp_path, monkeypatch):
    rows = examples()
    monkeypatch.setattr(pilot, "load_verified", lambda *args: copy.deepcopy(rows))
    source, reviewed, selector = (tmp_path / n for n in ("source", "reviewed", "selector"))
    for p in (source, reviewed, selector):
        pilot.data.write(p / "manifest.json", {"fixture": p.name})
    pilot.data.jsonl(reviewed / "benchmark_form_preference.jsonl", rows)
    review = {**inspection(rows), "review_manifest_sha256": pilot.data.digest(reviewed / "manifest.json"),
        "shortlist_sha256": pilot.data.digest(reviewed / "benchmark_form_preference.jsonl")}
    review_file = tmp_path / "inspection.json"
    pilot.data.write(review_file, review)
    output, archive = tmp_path / "pilot", tmp_path / "pilot.tar.gz"
    result = pilot.build(source, reviewed, selector, output, review_file, archive)
    assert result["training_ready"] is False
    assert result["training_launched"] is False
    for m in pilot.data.MODELS:
        for split in ("train", "inner_validation"):
            dpo = pilot.data.records(output / m / f"dpo_{split}.jsonl")
            sft = pilot.data.records(output / m / f"sft_{split}.jsonl")
            assert len(dpo) == len(sft)
            assert [r["chosen"] for r in dpo] == [r["messages"][-1]["content"] for r in sft]
            assert all(set(r["prompt"][1]) == {"role", "content"} for r in dpo)
    assert archive.exists()
    target = output / "qwen3/dpo_train.jsonl"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen selector input changed"):
        pilot.validate(output)


def test_small_or_empty_model_panel_cannot_be_exported():
    with pytest.raises(ValueError, match="Too few inspected questions"):
        pilot.assign_splits([p for p in examples() if p["model_key"] != "qwen3"])
