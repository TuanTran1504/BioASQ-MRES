"""Lossless miss extraction, frozen score explanations and artifact checks."""
import copy
from pathlib import Path
import tarfile

import pytest

from scripts import export_sft_reranker_misses as audit
from test_sft_ensemble_reranker_evaluation import bundle, candidate


@pytest.fixture
def scored(bundle, monkeypatch, tmp_path):
    directory, comparison, _, examples, pools, gold, original = bundle
    qid = next(iter(examples))
    for name, rows in pools.items():
        rows[:] = [r for r in rows if r["question_id"] != qid]
        for i in range(6):
            r = candidate(qid, f"valid correct variant {i}", 0)
            r["source_ranks"] = dict.fromkeys(audit.ev.MODELS, 2)
            r["sample_frequencies"] = dict.fromkeys(audit.ev.MODELS, 0.8)
            rows.append(r)
        r = candidate(qid, "invalid wrong", 1)
        r["source_ranks"] = dict.fromkeys(audit.ev.MODELS, 1)
        r["sample_frequencies"] = dict.fromkeys(audit.ev.MODELS, 0.0)
        rows.append(r)
    for q in gold:
        gold[q]["gold_aliases"] = ["invalid wrong" if q == qid else "valid correct"]
    original["conditions"], _, _ = audit.ev.summarize_pools(pools, examples, gold)
    hashes = {"manifest.json": audit.ev.digest(comparison / "manifest.json")}
    monkeypatch.setattr(audit.ev, "load_comparison", lambda path: (original, pools, examples, gold, hashes))
    evaluated = tmp_path / "evaluated"
    audit.ev.evaluate(comparison, directory, evaluated, resamples=20)
    summary = audit.ev.read(evaluated / "comparison_summary.json")
    scores = audit.ev.read(evaluated / "comparison_per_question.json")
    rankings = audit.ev.records(evaluated / "rankings.jsonl")
    _, payloads, _ = audit.ev.load_frozen(directory, examples, comparison)
    return directory, comparison, examples, pools, gold, evaluated, summary, scores, rankings, payloads, qid


def test_misses_include_every_missed_question_and_keep_semantics_unresolved(scored):
    _, _, examples, pools, gold, _, summary, scores, rankings, payloads, qid = scored
    before = copy.deepcopy(rankings)
    groups, names = audit.validate_rankings(rankings, pools, examples, summary, scores)
    cases, counts = audit.collect_cases(groups, names, examples, gold, payloads)
    assert len(counts) == 12
    assert all(c["selection_misses"] == 1 and c["question_ids"] == [qid] for c in counts.values())
    assert [c["question_id"] for c in cases] == [qid]
    assert len(cases[0]["missed_conditions"]) == 12
    assert cases[0]["review"]["status"] == "unreviewed"
    assert cases[0]["review"]["top5_semantically_valid_but_unaccepted"] is None
    for name in counts:
        d = cases[0]["conditions"][name]
        assert d["first_accepted_rank"] == 7 and len(d["top_five"]) == 5
        assert all(r["label"] == 0 for r in d["top_five"])
        assert d["score_gap_fifth_minus_best_accepted"] >= 0
        assert sum(d["gap_by_feature_block"].values()) == pytest.approx(d["score_gap_fifth_minus_best_accepted"])
        for ex in d["score_explanations"].values():
            assert ex["score"] == pytest.approx(ex["intercept"] + sum(ex["block_totals"].values()))
    assert rankings == before


@pytest.mark.parametrize("problem", ["label", "rank", "membership", "metric", "score"])
def test_corrupt_saved_rankings_or_scores_fail(scored, problem):
    _, _, examples, pools, _, _, summary, scores, rankings, _, _ = scored
    if problem == "label":
        rankings[0]["label"] = 1 - rankings[0]["label"]
    elif problem == "rank":
        rankings[0]["rank"] = 999
    elif problem == "membership":
        rankings.pop()
    elif problem == "metric":
        name = next(iter(summary["conditions"]))
        summary["conditions"][name]["metrics"]["mrr_at5"] = -1
    else:
        next(r for r in rankings if r["condition"].endswith("_reranker"))["score"] = float("nan")
    with pytest.raises(ValueError):
        audit.validate_rankings(rankings, pools, examples, summary, scores)


def test_export_portable_bundle_no_training_or_new_matching(scored, tmp_path):
    directory, comparison, examples, _, _, evaluated, _, _, _, _, qid = scored
    sources = audit.ev.read(comparison / "manifest.json")["sources"]
    for info in sources.values():
        source = Path(info["directory"])
        state = audit.ev.read(source / "manifest.json")
        state["runs"] = {a: a for a in audit.ev.ARMS}
        for arm in audit.ev.ARMS:
            (source / arm).mkdir()
            audit.ev.write_jsonl(source / arm / "generations.jsonl", [
                {"question_id": qid, "samples": [{"draw": 1, "raw_response": "saved response"}]},
                {"question_id": "not-selected", "samples": []}])
        audit.ev.write(source / "manifest.json", state)
    # The source fixture creates greedy archive handles only. Add matching
    # sampling handles for this export's trace collection.
    for model in audit.ev.MODELS:
        sources[f"{model}_sampling10"] = sources[f"{model}_greedy"]
    comparison_state = audit.ev.read(comparison / "manifest.json")
    comparison_state["sources"] = sources
    audit.ev.write(comparison / "manifest.json", comparison_state)
    state = audit.ev.read(evaluated / "manifest.json")
    state["source_file_hashes"]["manifest.json"] = audit.ev.digest(comparison / "manifest.json")
    audit.ev.write(evaluated / "manifest.json", state)
    before = audit.ev.digest(directory / "max5/reranker.joblib")
    output, archive = tmp_path / "audit", tmp_path / "audit.tar.gz"
    audit.export(evaluated, output, archive)
    result = audit.ev.read(output / "summary.json")
    assert result["exported_unique_questions"] == 1 and not result["training_ready"]
    assert len(audit.ev.records(output / "source_generations.jsonl")) == 12
    manifest = audit.ev.read(output / "manifest.json")
    assert manifest["status"] == "complete"
    assert not manifest["training"] and not manifest["generation"] and not manifest["paid_judging"]
    audit.check_hashes(output, manifest["output_sha256"])
    with tarfile.open(archive) as handle:
        assert {p.name for p in output.iterdir()} == set(handle.getnames())
    assert audit.ev.digest(directory / "max5/reranker.joblib") == before
    with pytest.raises(FileExistsError):
        audit.export(evaluated, output, archive)
