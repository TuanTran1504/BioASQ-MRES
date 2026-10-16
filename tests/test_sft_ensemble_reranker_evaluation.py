"""Frozen inference, identical pools, test separation and provenance checks."""
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_sft_ensemble_rerankers as evaluator
from scripts import train_sft_ensemble_reranker as trainer
from scripts.analyze_sft_ensemble_test import summarize_pools


def candidate(q, answer, label):
    return {"question_id": q, "answer": answer, "label": label,
            "sources": list(evaluator.MODELS), "source_ranks": dict.fromkeys(evaluator.MODELS, 1 if not label else 2),
            "sample_frequencies": dict.fromkeys(evaluator.MODELS, 0.2 if not label else 0.8),
            "relation_types": ["original"]}


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "trained"; directory.mkdir()
    preparation = tmp_path / "prepared"; preparation.mkdir()
    comparison = tmp_path / "comparison"; comparison.mkdir()
    examples = {f"test{i}": {"question": "Which biomedical answer?", "snippets": [{"text": "valid correct"}]}
                for i in range(95)}
    gold = {q: {"batch": "B1" if i < 26 else "B2" if i < 53 else "B3" if i < 73 else "B4"}
            for i, q in enumerate(examples)}
    pools = {f"{a}_{m}": [candidate(q, answer, label) for q in list(examples)[:-1]
             for answer, label in (("invalid wrong", 0), ("valid correct", 1))]
             for a in evaluator.ARMS for m in evaluator.MODES}
    for split, n in (("train", 1296), ("validation", 144)):
        trainer.write_jsonl(directory / f"{split}_examples.jsonl", [{"question_id": f"{split}{i}"} for i in range(n)])
    fit_examples = {f"train{i}": {"question": "Which biomedical answer?", "snippets": [{"text": "valid correct"}]}
                    for i in range(5)}
    fit_rows = [{**candidate(q, answer, label), "surface_operations": [], "is_format_variant": False}
                for q in fit_examples for answer, label in (("invalid wrong", 0), ("valid correct", 1))]
    components = trainer.fit_components(trainer.feature_rows(fit_rows), fit_examples,
                                       np.array([r["label"] for r in fit_rows]))
    fitted = {s: {"components": components, "selected_C": 1.0, "mean_validation_mrr_at5": 0.5,
                   "trials": [], "training_arms": list(arms),
                   "selection_conditions": [f"{a}_{m}" for a in arms for m in trainer.MODES]}
              for s, arms in trainer.SELECTORS.items()}
    selectors = trainer.save_selectors(directory, fitted)
    prepared = {"code_hashes": {"scripts/train_candidate_reranker_pilot.py":
                evaluator.digest(evaluator.ROOT / "scripts/train_candidate_reranker_pilot.py")},
                "original_banks": {}, "max5_banks": {}}
    inference = {"sources": {m: {} for m in evaluator.MODELS}}
    test_submission = tmp_path / "test-prepared"; test_submission.mkdir()
    sources = {}
    for model in evaluator.MODELS:
        archive = tmp_path / f"test-{model}"; archive.mkdir()
        evaluator.write(archive / "manifest.json", {"submission_dir": str(test_submission)})
        sources[f"{model}_greedy"] = {"directory": str(archive)}
        for arm in evaluator.ARMS:
            source = {"adapter_files_sha256": {"weights": f"{model}-{arm}"},
                      "base_snapshot": model, "system_prompt": arm}
            bank = tmp_path / f"bank-{model}-{arm}"; bank.mkdir()
            evaluator.write(bank / "manifest.json", {"status": "complete", "model_key": model, "source": source})
            prepared[f"{arm}_banks"][model] = str(bank)
            inference["sources"][model][arm] = {"source": source}
    evaluator.write(test_submission / "manifest.json", inference)
    evaluator.write(comparison / "manifest.json", {"sources": sources})
    evaluator.write(preparation / "prepared.json", prepared)
    state = {"status": "complete", "selectors": selectors, "question_counts": {"train": 1296, "validation": 144},
             "sklearn_version": evaluator.sklearn.__version__, "submission": str(preparation),
             "prepared_sha256": evaluator.digest(preparation / "prepared.json"),
             "selected_C": dict.fromkeys(selectors, 1.0),
             "artifacts_sha256": {p.relative_to(directory).as_posix(): evaluator.digest(p)
                                  for p in directory.rglob("*") if p.is_file()}}
    evaluator.write(directory / "manifest.json", state)
    baseline, _, _ = summarize_pools(pools, examples, gold)
    original = {"question_count": 95, "conditions": baseline, "source_costs": {"saved": True},
                "ensemble_costs": {"saved": True}, "settings": {}, "limitations": ["Inspected test"]}
    return directory, comparison, prepared, examples, pools, gold, original


def test_load_three_frozen_models_and_reject_test_membership_overlap(bundle):
    directory, comparison, _, examples, *_ = bundle
    state, payloads, _ = evaluator.load_frozen(directory, examples, comparison)
    assert set(payloads) == set(trainer.SELECTORS)
    assert state["selected_C"] == dict.fromkeys(payloads, 1.0)
    with pytest.raises(ValueError, match="overlapping"):
        evaluator.load_frozen(directory, {**examples, "train0": {}}, comparison)


@pytest.mark.parametrize("problem", ["weights", "sklearn", "feature_code", "source"])
def test_reject_changed_models_runtime_features_and_generator_sources(bundle, problem):
    directory, comparison, prepared, examples, *_ = bundle
    state = evaluator.read(directory / "manifest.json")
    if problem == "weights":
        with (directory / "original/reranker.joblib").open("ab") as handle:
            handle.write(b"changed")
    elif problem == "sklearn":
        state["sklearn_version"] = "different"
        evaluator.write(directory / "manifest.json", state)
    elif problem == "feature_code":
        prepared["code_hashes"]["scripts/train_candidate_reranker_pilot.py"] = "different"
        path = Path(state["submission"]) / "prepared.json"
        evaluator.write(path, prepared)
        state["prepared_sha256"] = evaluator.digest(path)
        evaluator.write(directory / "manifest.json", state)
    else:
        bank = Path(prepared["max5_banks"]["llama31"])
        meta = evaluator.read(bank / "manifest.json")
        meta["source"]["adapter_files_sha256"] = {"weights": "different"}
        evaluator.write(bank / "manifest.json", meta)
    with pytest.raises(ValueError):
        evaluator.load_frozen(directory, examples, comparison)


def test_full_evaluation_28_conditions_disjoint_frozen_no_gold_in_features(bundle, monkeypatch, tmp_path):
    directory, comparison, _, examples, pools, gold, original = bundle
    source_hashes = {"manifest.json": evaluator.digest(comparison / "manifest.json")}
    monkeypatch.setattr(evaluator, "load_comparison", lambda path: (original, pools, examples, gold, source_hashes))
    real_score = evaluator.score_components
    def score(components, rows, contexts):
        assert all("label" not in r and "gold_aliases" not in r for r in rows)
        assert all(set(e) == {"question", "snippets"} for e in contexts.values())
        return real_score(components, rows, contexts)
    monkeypatch.setattr(evaluator, "score_components", score)
    before = copy.deepcopy(pools)
    weights_before = evaluator.digest(directory / "max5/reranker.joblib")
    output = tmp_path / "evaluated"
    evaluator.evaluate(comparison, directory, output, resamples=20)
    result = evaluator.read(output / "comparison_summary.json")
    assert len(result["conditions"]) == 28
    assert len(result["paired_contrasts"]) == 56
    for name, baseline in original["conditions"].items():
        assert result["conditions"][name] == baseline
    for pool in pools:
        values = [c["metrics"]["coverage_full_pool"] for n, c in result["conditions"].items() if n.startswith(pool + "_")]
        assert set(values) == {94/95}
    conditions = result["conditions"]["max5_sampling10_max5_reranker"]
    assert conditions["metrics"]["mrr_at5"] == pytest.approx(94/95)
    assert {b: v["question_count"] for b, v in conditions["per_batch"].items()} == {"B1": 26, "B2": 27, "B3": 20, "B4": 22}
    assert result["ensemble_costs"] == original["ensemble_costs"]
    assert evaluator.read(output / "manifest.json")["status"] == "complete"
    assert pools == before
    assert evaluator.digest(directory / "max5/reranker.joblib") == weights_before
    with pytest.raises(FileExistsError):
        evaluator.evaluate(comparison, directory, output, resamples=20)
