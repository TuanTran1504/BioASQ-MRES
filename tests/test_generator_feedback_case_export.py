"""Case audit reconstructs saved rankings and explains them without new labels."""
import copy
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from scripts import export_generator_feedback_cases as audit
from scripts import train_candidate_reranker_pilot as features


def generations(qid, answers, draws):
    return {"question_id": qid, "samples": [{"draw": d, "raw_response": "saved JSON",
        "answers": [{"answer": answer, "relation_type": "original" if i == 0 else "synonym"}
                    for i, answer in enumerate(answers)]} for d in range(1, draws+1)]}


def make_bundle(tmp_path, extra_questions=0):
    examples = {q: {"question_id": q, "question": "Which entity?", "gold_aliases": ["Core"],
                   "snippets": [{"text": "Core is the entity; Wrong is different."}]}
                for q in ["regression", "improvement", "unchanged", *[f"unchanged{i}" for i in range(extra_questions)]]}
    train_examples = {"fit1": examples["regression"], "fit2": examples["regression"]}
    train_rows = [{"question_id": q, "answer": answer, "sources": ["llama31"],
                   "source_ranks": {"llama31": rank}, "sample_frequencies": {"llama31": 1.},
                   "relation_types": ["original"], "surface_operations": [], "is_format_variant": False}
                  for q in train_examples for rank, answer in enumerate(("Core", "Wrong"), 1)]
    components = features.fit_components(train_rows, train_examples, np.array([1, 0, 1, 0]))
    raw, baseline, dpo = {}, {}, {}
    for mode in audit.ev.MODES:
        draws = 1 if mode == "greedy" else 10
        for model in audit.ev.MODELS:
            rows = [generations(q, ["Core"] if model == "llama31" and q != "improvement" else ["Wrong"], draws) for q in examples]
            baseline[f"{model}_{mode}"] = rows
            raw.update({(model, "sft", mode, r["question_id"]): r for r in rows})
        dpo[mode] = [generations(q, ["Wrong"] if q == "regression" else ["Core"], draws) for q in examples]
        raw.update({("llama31", "dpo", mode, r["question_id"]): r for r in dpo[mode]})
    pools = audit.ev.make_pools(baseline, dpo)
    def matcher(ee, cc, output, jar_path):
        return {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in cc}
    conditions, scores, rankings = audit.ev.score_pools(pools, examples,
        {"selector": "max5", "components": components}, tmp_path, Path("unused"), matcher)
    summary = {"question_count": len(examples), "conditions": conditions}
    return examples, raw, pools, components, summary, scores, rankings


@pytest.fixture
def bundle(tmp_path):
    return make_bundle(tmp_path)


def test_changed_questions_regression_first_and_exact_score_contributions(bundle):
    examples, raw, pools, components, summary, scores, rankings = bundle
    groups = audit.check_rankings(rankings, pools, examples, summary, scores)
    cases = audit.build_cases(groups, examples, raw, components)
    assert [c["question_id"] for c in cases] == ["regression", "improvement"]
    assert all(len(c["changes"]) == 2 for c in cases)
    assert cases[0]["candidate_changes"]["sampling10"]["removed_expressions"] == ["Core"]
    assert cases[1]["candidate_changes"]["sampling10"]["added_expressions"] == ["Core"]
    for case in cases:
        for mode in audit.ev.MODES:
            for candidate in case["candidate_changes"][mode]["focus_candidates"]:
                for policy in ("sft", "dpo"):
                    row = candidate[policy]
                    if row is not None:
                        explanation = row["score_explanation"]
                        assert explanation["intercept"] + sum(explanation["block_totals"].values()) == pytest.approx(row["score"])
                if "score_difference" in candidate:
                    assert sum(candidate["score_block_differences"].values()) == pytest.approx(candidate["score_difference"])
    report = audit.render_report(cases)
    assert "regression" in report and "Officially accepted" in report and "Llama draws / initial draws" in report


@pytest.mark.parametrize("field", ["rank", "label", "score", "frequency", "reported_metric"])
def test_corrupt_rankings_or_metrics_are_rejected(bundle, field):
    examples, _, pools, _, summary, scores, rankings = copy.deepcopy(bundle)
    rankings = copy.deepcopy(rankings)  # Disk archives do not share nested dicts with pool rows.
    row = next(r for r in rankings if r["condition"].endswith("_reranker"))
    if field == "rank": row["rank"] = 99
    elif field == "label": row["label"] = 1-row["label"]
    elif field == "score": row["score"] = float("nan")
    elif field == "frequency": row["sample_frequencies"]["llama31"] = .17
    else: summary["conditions"][row["condition"]]["metrics"]["mrr_at5"] = -.1
    with pytest.raises(ValueError):
        audit.check_rankings(rankings, pools, examples, summary, scores)


def test_per_draw_occurrence_counts_distinguish_initial_and_later_variants():
    row = {"samples": [
        {"draw": 1, "answers": [{"answer": "Core", "relation_type": "original"}, {"answer": "alias", "relation_type": "synonym"}]},
        {"draw": 2, "answers": [{"answer": "alias", "relation_type": "original"}, {"answer": "Core", "relation_type": "synonym"}]},
        {"draw": 3, "answers": []}]}
    counts = audit.occurrences(row, "CORE")
    assert counts["draw_count"] == 2 and counts["initial_draw_count"] == 1
    assert [r["position"] for r in counts["occurrences"]] == [1, 2]


def test_score_explanation_detects_weights_that_do_not_reproduce_saved_score(bundle):
    examples, raw, pools, components, summary, scores, rankings = bundle
    groups = audit.check_rankings(rankings, pools, examples, summary, scores)
    components["model"].intercept_ += 1
    with pytest.raises(ValueError, match="reproduces|reproduce"):
        audit.build_cases(groups, examples, raw, components)


def test_exporter_starts_with_clean_python_import_state():
    script = Path(__file__).resolve().parents[1] / "scripts/export_generator_feedback_cases.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--comparison" in result.stdout and "--archive" in result.stdout


def test_complete_export_reconstructs_pools_and_builds_small_download(tmp_path, monkeypatch):
    examples, raw, pools, components, summary, scores, rankings = make_bundle(tmp_path, 157)
    submission = tmp_path / "submission"; submission.mkdir()
    comparison = tmp_path / "comparison"; comparison.mkdir()
    generation = tmp_path / "generation"; generation.mkdir()
    baseline_runs, sources, rows_by_path = {}, {}, {}
    for model in audit.ev.MODELS:
        sources[model] = {"adapter": model}
        for mode in audit.ev.MODES:
            path = tmp_path / f"{model}_{mode}"
            baseline_runs[f"{model}_{mode}"] = str(path)
            rows_by_path[str(path)] = [r for (m, p, md, q), r in raw.items() if (m, p, md) == (model, "sft", mode)]
    hashes = {}
    for mode in audit.ev.MODES:
        path = generation / mode; path.mkdir()
        audit.ev.write_jsonl(path / "generations.jsonl", [r for (m, p, md, q), r in raw.items() if (m, p, md) == ("llama31", "dpo", mode)])
        hashes[mode] = {"generations.jsonl": audit.ev.digest(path / "generations.jsonl")}
        rows_by_path[str(path)] = audit.ev.records(path / "generations.jsonl")
    prepared = {"baseline_runs": baseline_runs, "sources": sources, "dpo_source": {"adapter": "dpo"},
                "rerankers": str(tmp_path / "reranker"), "reranker_hashes": {}}
    audit.ev.write(submission / "prepared.json", prepared)
    audit.ev.write_jsonl(submission / "dev_gold.jsonl", list(examples.values()))
    audit.ev.write(generation / "manifest.json", {"status": "complete", "smoke_test": False,
        "expected_questions": 160, "runs": {m: m for m in audit.ev.MODES}, "arm_hashes": hashes})
    audit.ev.write(comparison / "comparison_summary.json", summary)
    audit.ev.write(comparison / "comparison_per_question.json", scores)
    audit.ev.write_jsonl(comparison / "rankings.jsonl", rankings)
    for name, rows in pools.items(): audit.ev.write_jsonl(comparison / f"{name}_candidates.jsonl", rows)
    audit.ev.write(comparison / "manifest.json", {"status": "complete", "submission": str(submission),
        "generation": str(generation), "prepared_sha256": audit.ev.digest(submission / "prepared.json"),
        "generation_manifest_sha256": audit.ev.digest(generation / "manifest.json"),
        "output_sha256": {p.name: audit.ev.digest(p) for p in comparison.iterdir()}})
    monkeypatch.setattr(audit, "validate_prepared", lambda path: prepared)
    monkeypatch.setattr(audit.ev, "load_generation", lambda path, *args: rows_by_path[str(path)])
    monkeypatch.setattr(audit.ev, "load_selector", lambda *args: ({"components": components}, {}))
    export, archive = tmp_path / "export", tmp_path / "download.tar.gz"
    audit.export(comparison, export, archive)
    assert audit.ev.read(export / "summary.json")["question_ids"] == ["regression", "improvement"]
    raw_export = audit.ev.records(export / "source_generations.jsonl")
    assert len(raw_export) == 16 and {r["question_id"] for r in raw_export} == {"regression", "improvement"}
    manifest = audit.ev.read(export / "manifest.json")
    audit.ev.verify_files(export, manifest["output_sha256"])
    with tarfile.open(archive) as handle:
        assert set(handle.getnames()) == {"cases.jsonl", "source_generations.jsonl", "report.md", "summary.json", "manifest.json"}
