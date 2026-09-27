from __future__ import annotations

import pytest

from scripts.prepare_factoid_snippet_sft import build_splits, supported_aliases


def question(index, answers=None, snippets=None):
    return {
        "id": f"q{index:03}", "type": "factoid", "body": "Which expression?",
        "exact_answer": [answers or ["alpha", "beta"]],
        "snippets": snippets or [{"text": "alpha and beta", "document": "http://pubmed/123"}],
    }


def test_support_uses_snippet_text_and_whole_tokens():
    q = question(0, ["123", "alpha", "bet", "beta"], [{"text": "ALPHA, beta.", "document": "http://pubmed/123"}])
    assert supported_aliases(q) == ["alpha", "beta"]


def test_support_cannot_cross_snippet_boundaries():
    q = question(0, ["alpha beta"], [{"text": "alpha"}, {"text": "beta"}])
    assert supported_aliases(q) == []


def test_filter_variants_preserve_their_matching_policy():
    q = question(0, ["alpha", "beta", "absent"], [{"text": "ALPHA, beta."}])
    assert supported_aliases(q, "normalized") == ["alpha", "beta"]
    assert supported_aliases(q, "literal") == ["beta"]
    assert supported_aliases(q, "none") == ["alpha", "beta", "absent"]


def test_all_accepted_aliases_are_considered():
    q = question(0, ["one", "two", "three", "four", "five", "alpha"])
    assert supported_aliases(q) == ["alpha"]


def test_split_is_deterministic_disjoint_and_expands_only_training():
    qs = [question(i) for i in range(20)]
    test = [question(100)]
    train, dev, manifest = build_splits(qs, test, dev_ratio=0.2)
    assert (train, dev, manifest) == build_splits(list(reversed(qs)), test, dev_ratio=0.2)
    train_ids = {row["source_question_id"] for row in train}
    dev_ids = {row["id"] for row in dev}
    assert len(train_ids) == 16 and len(train) == 32 and len(dev) == 4
    assert not train_ids & dev_ids
    assert train_ids | dev_ids == {q["id"] for q in qs}
    assert "q100" not in train_ids | dev_ids
    assert all(row["output"].count("[BE]") == 1 for row in train)
    assert all(row["output"].count("[BE]") == 2 for row in dev)


def test_dev_keeps_accepted_aliases_without_snippet_matches():
    qs = [question(i, ["alpha", "accepted synonym"]) for i in range(10)]
    train, dev, _ = build_splits(qs, [])
    assert all(row["output"] == "[BE]alpha[EE]" for row in train)
    assert "accepted synonym" in dev[0]["output"]


def test_rejects_train_test_overlap_and_duplicate_questions():
    qs = [question(i) for i in range(10)]
    with pytest.raises(ValueError, match="overlap"):
        build_splits(qs, [qs[0]])
    with pytest.raises(ValueError, match="unique"):
        build_splits(qs + [qs[0]], [])


def test_split_first_retains_unsupported_dev_and_filters_only_training():
    qs = [question(i) for i in range(20)]
    _, initial_dev, initial_manifest = build_splits(qs, [], dev_ratio=.2)
    dev_ids = {r["id"] for r in initial_dev}
    removed_train_id = initial_manifest["train_question_ids"][0]
    for q in qs:
        if q["id"] in dev_ids | {removed_train_id}:
            q["snippets"] = [{"text": "No accepted answer here."}]
    train, dev, manifest = build_splits(qs, [], dev_ratio=.2)
    assert {r["id"] for r in dev} == dev_ids
    assert len(dev) == 4 and all(r["supported_aliases"] == [] for r in dev)
    assert [r["output"] for r in dev] == [r["output"] for r in initial_dev]
    assert removed_train_id not in {r["source_question_id"] for r in train}
    assert manifest["train_questions_before_filter"] == 16
    assert manifest["train_questions"] == 15
    assert manifest["train_filtered_out_question_ids"] == [removed_train_id]
    assert manifest["excluded_question_ids"] == [removed_train_id]
    assert manifest["dev_questions_without_normalized_snippet_match"] == 4
    assert (train, dev, manifest) == build_splits(list(reversed(qs)), [], dev_ratio=.2)


def test_dev_membership_and_gold_are_independent_of_train_filter():
    qs = [question(i, ["ALPHA", "absent"], [{"text": "alpha"}]) if i % 2
          else question(i) for i in range(40)]
    results = [build_splits(qs, [], filter_mode=mode) for mode in ("normalized", "literal", "none")]
    dev_gold = [{r["id"]: r["output"] for r in dev} for _, dev, _ in results]
    assert dev_gold[0] == dev_gold[1] == dev_gold[2]
    assert len({len(train) for train, _, _ in results}) > 1


def test_legacy_filter_first_still_excludes_unsupported_dev_questions():
    qs = [question(i) if i < 10 else question(i, ["absent"]) for i in range(20)]
    _, dev, manifest = build_splits(qs, [], dev_ratio=.2, split_order="filter_first")
    assert len(dev) == 2 and all(r["supported_aliases"] for r in dev)
    assert manifest["split_order"] == "filter_first"
    assert manifest["dev_questions_without_normalized_snippet_match"] == 0


def test_empty_filtered_training_is_rejected_without_resampling_dev():
    qs = [question(i, ["absent"]) for i in range(10)]
    with pytest.raises(ValueError, match="No training questions remain"):
        build_splits(qs, [])
