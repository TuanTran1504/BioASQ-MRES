"""Full-panel pair decisions must preserve clean siblings and fail closed on edits."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_generator_form_pilot as pilot
from test_generator_form_pilot import examples, inspection


def full_review(rows):
    result = inspection(rows)
    result.update(scope="all_pairs", expected_pair_count=len(rows),
                  expected_question_count=len({p["question_id"] for p in rows}))
    for q, rule in result["decisions"].items():
        records = [p for p in rows if p["question_id"] == q]
        surfaces = {c["answer"] for p in records for side in ("chosen", "rejected") for c in p[side + "_candidates"]}
        rule.update(surfaces_sha256=pilot.identity(sorted(surfaces)),
                    pair_ids_sha256=pilot.identity(sorted(p["pair_id"] for p in records)))
    return result


def test_surface_hold_does_not_discard_clean_siblings_for_same_question():
    rows = examples()
    rows[0]["rejected_candidates"][0]["answer"] = "a different entity"
    review = full_review(rows)
    review["decisions"]["q0"].update(decision="conditional", blocked_surfaces=["a different entity"])
    kept, held, panel = pilot.select(rows, review)
    assert len(panel) == 8
    assert len(kept) == len(rows) - 1
    assert len([p for p in kept if p["question_id"] == "q0"]) == 2
    assert held[0]["pair_id"] == rows[0]["pair_id"]


def test_bad_expansion_on_chosen_side_also_holds_whole_response_pair():
    rows = examples()
    rows[0]["chosen_candidates"].append({"answer": "a different entity", "official_accepted": False})
    review = full_review(rows)
    review["decisions"]["q0"].update(decision="conditional", blocked_surfaces=["a different entity"])
    kept, held, _ = pilot.select(rows, review)
    assert rows[0]["pair_id"] not in {p["pair_id"] for p in kept}
    assert held[0]["flagged_surfaces"] == ["a different entity"]


def test_allowlist_cannot_let_unreviewed_variants_through():
    rows = examples()
    rows[0]["rejected_candidates"].append({"answer": "extra qualifier", "official_accepted": False})
    review = full_review(rows)
    review["decisions"]["q0"].update(decision="conditional", allow_surfaces=["X", "X name"])
    kept, held, _ = pilot.select(rows, review)
    assert len(kept) == len(rows) - 1
    assert held[0]["flagged_surfaces"] == ["extra qualifier"]


def test_interval_groups_require_both_entire_lists_to_stay_in_one_group():
    rows = examples()
    q0 = [p for p in rows if p["question_id"] == "q0"]
    q0[1]["chosen_candidates"][0]["answer"] = "Y"
    q0[1]["rejected_candidates"][0]["answer"] = "Y name"
    q0[2]["rejected_candidates"][0]["answer"] = "Y name"
    review = full_review(rows)
    review["decisions"]["q0"].update(decision="conditional",
        equivalence_groups={"first": ["X", "X name"], "second": ["Y", "Y name"]})
    kept, held, _ = pilot.select(rows, review)
    assert {p["pair_id"] for p in q0[:2]} <= {p["pair_id"] for p in kept}
    assert held[0]["pair_id"] == q0[2]["pair_id"]
    assert held[0]["reason"] == "different_or_unreviewed_equivalence_group"


@pytest.mark.parametrize("change", ["missing_pair", "unseen_surface", "stale_rule", "invented_quote"])
def test_changed_full_panel_or_evidence_cannot_silently_reuse_decisions(change):
    rows = examples()
    review = full_review(rows)
    if change == "missing_pair":
        rows.pop()
    elif change == "unseen_surface":
        rows[0]["rejected_candidates"][0]["answer"] = "unseen"
    elif change == "stale_rule":
        review["decisions"]["q0"].update(decision="conditional", blocked_surfaces=["not present"])
    else:
        review["decisions"]["q0"]["evidence"]["quote"] = "not in source"
    with pytest.raises(ValueError):
        pilot.select(rows, review)


def test_full_export_all_decisions_controls_and_tamper_detection(tmp_path, monkeypatch):
    rows = examples()
    for p in rows:
        p.update(question="What is the entity?", chosen_id="chosen-" + p["pair_id"],
                 rejected_id="rejected-" + p["pair_id"])
    monkeypatch.setattr(pilot, "load_verified", lambda *args: copy.deepcopy(rows))
    source, reviewed, selector = (tmp_path / n for n in ("source", "reviewed", "selector"))
    for directory in (source, reviewed, selector):
        pilot.data.write(directory / "manifest.json", {"fixture": directory.name})
    pilot.data.jsonl(reviewed / "benchmark_form_preference.jsonl", rows)
    config = full_review(rows)
    config.update(review_manifest_sha256=pilot.data.digest(reviewed / "manifest.json"),
                  shortlist_sha256=pilot.data.digest(reviewed / "benchmark_form_preference.jsonl"))
    config_path, output = tmp_path / "inspection.json", tmp_path / "output"
    pilot.data.write(config_path, config)
    summary = pilot.build(source, reviewed, selector, output, config_path)
    assert summary["inspected_pairs"] == len(rows)
    assert summary["uninspected_pair_records"] == 0
    assert summary["training_ready"] is False
    decisions = pilot.data.records(output / "pair_decisions.jsonl")
    assert {r["pair_id"] for r in decisions} == {r["pair_id"] for r in rows}
    decisions[0]["decision"] = "hold"
    pilot.data.jsonl(output / "pair_decisions.jsonl", decisions)
    # Even an updated output-file checksum cannot change a reconstructed decision.
    manifest = pilot.data.read(output / "manifest.json")
    manifest["files_sha256"]["pair_decisions.jsonl"] = pilot.data.digest(output / "pair_decisions.jsonl")
    pilot.data.write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Pair decision differs"):
        pilot.validate(output)
