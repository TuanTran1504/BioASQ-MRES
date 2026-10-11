"""Check fold isolation, complete denominators and paired pool comparisons."""
import copy

import numpy as np
import pytest

from scripts import compare_expansion_pool_rerankers as rerank


def candidate(q, answer, label, model="llama31", rank=1):
    return {"question_id": q, "answer": answer, "label": label,
            "sources": [model], "source_ranks": {model: rank},
            "relation_types": ["original"], "surface_operations": [],
            "is_format_variant": False}


def loaded_sources():
    examples = {q: {"question_id": q, "question": q, "snippets": [], "gold_aliases": ["correct"]}
                for q in ("a", "empty")}
    loaded = []
    for model in rerank.MODELS:
        arms = {}
        for policy in rerank.POLICIES:
            arms[f"{policy}_sampling10"] = {
                "scores": {q: {} for q in examples},
                "candidates": {"a": [{"answer": "correct", "official_accepted": True,
                                       "rank": 1, "relation_type": "original"}], "empty": []}}
        loaded.append(({"model_key": model}, copy.deepcopy(examples), arms, {}))
    return loaded


def test_combined_pools_keep_sources_and_do_not_add_candidates():
    examples, pools = rerank.prepare_pools(loaded_sources(), "sampling10")
    assert set(examples) == {"a", "empty"}
    assert all(len(rows) == 1 for rows in pools.values())
    row = pools["sft"][0]
    assert row["sources"] == list(rerank.MODELS)
    assert row["source_ranks"] == dict.fromkeys(rerank.MODELS, 1)
    assert row["answer"] == "correct" and row["label"] == 1
    assert "policy" not in row


@pytest.mark.parametrize("failure", ["missing_model", "different_evidence", "conflicting_label"])
def test_pool_validation(failure):
    loaded = loaded_sources()
    if failure == "missing_model":
        loaded.pop()
    elif failure == "different_evidence":
        loaded[1][1]["a"]["snippets"] = [{"text": "different"}]
    else:
        loaded[1][2]["sft_sampling10"]["candidates"]["a"][0]["official_accepted"] = False
    with pytest.raises(ValueError):
        rerank.prepare_pools(loaded, "sampling10")


def test_round_robin_deduplicates_and_ignores_gold():
    rows = [candidate("q", "L", 0), candidate("q", "Q", 1, "qwen3"),
            candidate("q", "L2", 1, rank=2)]
    assert [r["answer"] for r in rerank.round_robin(rows)] == ["L", "Q", "L2"]
    flipped = [{**r, "label": 1 - r["label"]} for r in rows]
    assert [r["answer"] for r in rerank.round_robin(flipped)] == ["L", "Q", "L2"]


def test_question_policy_weights_balance_large_pools():
    rows = [candidate("a", str(i), 0) for i in range(4)] + [candidate("b", "b", 1), candidate("a", "x", 1)]
    policies = ["sft"] * 5 + ["dpo"]
    weights = rerank.question_weights(rows, policies)
    assert weights.mean() == pytest.approx(1)
    assert weights[:4].sum() == pytest.approx(weights[4])
    assert weights[:4].sum() == pytest.approx(weights[5])


def test_empty_pools_remain_in_denominator():
    metrics, questions = rerank.evaluate_rankings({"empty": [], "covered": [candidate("covered", "yes", 1)]})
    assert len(questions) == 2
    assert metrics["mrr_at5"] == metrics["coverage_full_pool"] == 0.5


def test_cv_holds_out_questions_from_both_policies(monkeypatch):
    examples = {f"q{i}": {} for i in range(6)}
    pools = {p: [candidate(q, answer, label, rank=rank)
                 for q in examples if q != "q5"
                 for rank, (answer, label) in enumerate((("wrong", 0), ("correct", 1)), 1)]
             for p in rerank.POLICIES}
    fitted, tested = [], []

    def fit(rows, examples_arg, labels, sample_weight=None):
        train = {r["question_id"] for r in rows}
        fitted.append(train)
        assert np.isfinite(sample_weight).all()
        assert all(set(e) == {"question", "snippets"} for e in examples_arg.values())
        return train

    def score(train, rows, examples_arg):
        test = {r["question_id"] for r in rows}
        assert not train & test
        tested.append(test)
        # Deliberately independent of gold labels, which must only score ranks.
        return np.asarray([float(r["answer"] == "correct") for r in rows])

    monkeypatch.setattr(rerank, "fit_components", fit)
    monkeypatch.setattr(rerank, "score_components", score)
    summary, questions, rankings = rerank.cross_validate(examples, pools, folds=3, resamples=40)
    assert len(fitted) == 9 and len(tested) == 18
    for fold in summary["folds"]:
        assert not set(fold["training_question_ids"]) & set(fold["test_question_ids"])
    assert all(len(rows) == 6 for rows in questions.values())
    assert summary["metrics"]["trained_sft_on_sft"]["mrr_at5"] == pytest.approx(5 / 6)
    assert summary["metrics"]["round_robin_on_sft"]["mrr_at5"] == pytest.approx(5 / 12)
    assert len(summary["metrics"]) == 8  # Six trained/test pairs and two controls.
    for name, contrast in summary["paired_contrasts"].items():
        if "reranker_minus" not in name:
            assert contrast["metrics"]["mrr_at5"]["difference"] == 0
    assert len(rankings) == 60


def test_real_logistic_fit_uses_training_vocabulary_only():
    examples = {q: {"question": "Which answer fits?", "snippets": [{"text": "correct wrong"}]}
                for q in ("a", "b", "heldout")}
    rows = [candidate(q, answer, label) for q in ("a", "b")
            for answer, label in (("correct", 1), ("wrong", 0))]
    components = rerank.fit_components(rows, examples, [r["label"] for r in rows],
                                       sample_weight=np.ones(len(rows)))
    heldout = [candidate("heldout", "neverobservedtoken", 0)]
    scores = rerank.score_components(components, heldout, examples)
    assert np.isfinite(scores).all()
    assert "neverobservedtoken" not in components["word"].vocabulary_
