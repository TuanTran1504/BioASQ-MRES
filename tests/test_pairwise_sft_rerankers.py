"""Within-question learning, selection isolation and unchanged-pool controls."""
import copy
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import compare_pairwise_sft_rerankers as pw


def row(q, answer, label, frequency=None):
    return {"question_id": q, "answer": answer, "label": label,
            "sources": ["llama31"], "source_ranks": {"llama31": 1 if label else 2},
            "sample_frequencies": {"llama31": (0.8 if label else 0.2) if frequency is None else frequency},
            "relation_types": ["original"], "surface_operations": [], "is_format_variant": False}


@pytest.fixture
def small():
    examples = {"train": {f"t{i}": {"question": "Which biomedical answer?", "snippets": [{"text": "correct answer"}]}
                          for i in range(5)},
                "validation": {"v": {"question": "validationonlyword", "snippets": []},
                               "empty": {"question": "No candidates", "snippets": []}}}
    pools = {split: {f"{a}_{m}": [row(q, answer, label) for q in ee if q != "empty"
                                 for answer, label in (("correct answer", 1), ("incorrect answer", 0))]
                     for a in pw.rr.ARMS for m in pw.rr.MODES} for split, ee in examples.items()}
    baseline = {}
    for selector, arms in pw.rr.SELECTORS.items():
        best, _ = pw.rr.fit_selector(pools, examples, [1.0], arms)
        baseline[selector] = {"components": best[2], "selected_C": 1.0, "selector": selector,
                              "training_arms": list(arms)}
    return examples, pools, baseline


def test_pairs_stay_inside_question_and_pool_with_balanced_group_weights():
    rows = [row("a", a, y) for a, y in (("yes", 1), ("no", 0), ("other", 0))]
    rows += [row("a", "second pool", 1), row("a", "bad second pool", 0), row("b", "all negative", 0)]
    strata = ["one"] * 3 + ["two"] * 2 + ["one"]
    pp, nn, weights, stats = pw.make_pairs(rows, strata)
    assert len(pp) == 3 and stats["skipped_question_pools"] == 1
    totals = {}
    for p, n, w in zip(pp, nn, weights):
        assert rows[p]["label"] == 1 and rows[n]["label"] == 0
        assert rows[p]["question_id"] == rows[n]["question_id"]
        assert strata[p] == strata[n]
        totals[strata[p]] = totals.get(strata[p], 0) + w
    assert totals == {"one": 1.0, "two": 1.0}


def test_pair_cap_is_reproducible_and_never_samples_validation():
    rows = [row("fit", f"candidate {i}", int(i < 10)) for i in range(30)]
    a = pw.make_pairs(rows, ["pool"] * len(rows), cap=7)
    b = pw.make_pairs(rows, ["pool"] * len(rows), cap=7)
    for aa, bb in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(aa, bb)
    assert len(set(zip(a[0], a[1]))) == 7
    assert a[3]["capped_question_pools"] == 1
    with pytest.raises(ValueError, match="No within-question"):
        pw.make_pairs([row("q", "unmatched", 0)], ["pool"])


def test_real_pairwise_fit_preserves_features_excludes_validation_and_gold(small, monkeypatch):
    examples, pools, baseline = small
    original = copy.deepcopy(pools)
    real_matrix = pw.feature_matrix
    def checked(components, rows, ee, fit=False):
        assert fit and set(ee) == set(examples["train"])
        return real_matrix(components, rows, ee, fit)
    monkeypatch.setattr(pw, "feature_matrix", checked)
    fitted = pw.fit_pairwise(pools, examples, ("max5",), [0.1, 1.0])
    assert fitted["selected_C"] == 0.1  # Validation ties resolve to first C.
    assert fitted["pair_statistics"]["unordered_pairs"] == 10
    c = fitted["components"]
    assert c["model"].fit_intercept is False and c["model"].class_weight is None
    assert "validationonlyword" not in c["word"].vocabulary_
    assert "relation=original" in c["meta"].get_feature_names_out()
    base = baseline["max5"]["components"]
    for name in ("word", "char", "meta"):
        np.testing.assert_array_equal(c[name].get_feature_names_out(), base[name].get_feature_names_out())
    metrics, _, _ = pw.rank(fitted, pools["validation"], examples["validation"])
    assert all(v["metrics"]["mrr_at5"] == 0.5 for v in metrics.values())
    # Scoring must not depend on any gold labels or aliases supplied in input.
    altered = {n: [{**r, "label": 1-r["label"], "gold_aliases": ["poison"]} for r in rr]
               for n, rr in pools["validation"].items()}
    poisoned = {q: {**e, "gold_aliases": ["poison"]} for q, e in examples["validation"].items()}
    _, _, first = pw.rank(fitted, pools["validation"], examples["validation"])
    _, _, second = pw.rank(fitted, altered, poisoned)
    assert [(r["answer"], r["score"], r["rank"]) for r in first] == [(r["answer"], r["score"], r["rank"]) for r in second]
    assert pools == original


def test_pairwise_weights_learn_preference_on_new_question(small):
    examples, pools, _ = small
    fitted = pw.fit_pairwise(pools, examples, ("max5",), [1.0])
    # Repeated words are unseen, but frequency and evidence metadata generalize.
    ee = {"new": {"question": "Novel question", "snippets": []}}
    candidates = [row("new", "unseen red", 0), row("new", "unseen blue", 1)]
    _, _, ranked = pw.rank(fitted, {"pool": candidates}, ee)
    assert ranked[0]["answer"] == "unseen blue"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_typography_only_penalty_retains_all_candidates_and_numeric_distinctions():
    assert pw.typography_signature("X-inactivation") == pw.typography_signature("X inactivation.")
    for a, b in (("1-2", "12"), ("0.1%", "0.1"), ("2-3 years", "2-2.5 years"),
                 ("IL-6", "IL-8"), ("45,X", "45,X/46,XX"), ("TNF-alpha", "IL-6")):
        assert pw.typography_signature(a) != pw.typography_signature(b)
    answers = ["X-inactivation", "X inactivation.", "X inactivation", "another form", "other", "sixth"]
    rows = [row("q", a, int(i == 1)) for i, a in enumerate(answers)]
    before = copy.deepcopy(rows)
    result = pw.diversify(rows, 0.3)
    assert result[0] == rows[0] and result != rows
    assert sorted(r["answer"] for r in result) == sorted(answers)
    assert len(result) == len(rows) and rows == before
    assert pw.diversify(rows, 0) == rows
    # Selection never reads acceptance labels.
    inverted = [{**r, "label": 1-r["label"]} for r in rows]
    assert [r["answer"] for r in pw.diversify(inverted, 0.3)] == [r["answer"] for r in result]


def test_validation_penalty_ties_keep_disabled_policy(small):
    examples, pools, baseline = small
    selected = pw.select_penalty(baseline["max5"], pools["validation"], examples["validation"], ("max5",))
    assert selected["selected_penalty"] == 0


def test_complete_run_freezes_before_test_and_preserves_baseline(small, monkeypatch, tmp_path):
    examples, pools, baseline = small
    baseline_dir = tmp_path / "baseline"; baseline_dir.mkdir()
    comparison = tmp_path / "comparison"; comparison.mkdir()
    submission = tmp_path / "submission"; submission.mkdir()
    pw.ev.write(submission / "prepared.json", {"baseline": str(baseline_dir), "comparison": str(comparison),
                "baseline_hashes": {}, "comparison_hashes": {}, "code_hashes": {},
                "C_grid": [0.1, 1.0], "pair_cap": 128, "penalties": [0, 0.3]})
    test = {"test": {"question": "testonlyword", "snippets": []}, "emptytest": {"question": "empty", "snippets": []}}
    test_pools = {f"{a}_{m}": [row("test", "correct answer", 1), row("test", "incorrect answer", 0)]
                  for a in pw.ev.ARMS for m in pw.ev.MODES}
    monkeypatch.setattr(pw.ev, "load_comparison", lambda p: ({}, test_pools, test,
                        {q: {"batch": "B1"} for q in test}, {}))
    monkeypatch.setattr(pw.ev, "load_frozen", lambda *args: ({}, baseline, {}))
    monkeypatch.setattr(pw, "load_training", lambda *args: (examples, pools))
    output = tmp_path / "output"
    real_compare = pw.compare
    def checked(pp, ee, models, selections, gold, resamples):
        assert len(list(output.glob("*.joblib"))) == 3
        assert (output / "validation_selection.json").exists()
        for path in output.glob("*.joblib"):
            saved = joblib.load(path)
            assert "testonlyword" not in saved["components"]["word"].vocabulary_
        return real_compare(pp, ee, models, selections, gold, resamples)
    monkeypatch.setattr(pw, "compare", checked)
    before = copy.deepcopy(pools)
    pw.run(submission, output, resamples=20)
    assert pw.ev.read(output / "manifest.json")["status"] == "complete"
    result = pw.ev.read(output / "test_summary.json")
    assert len(result["conditions"]) == 48 and len(result["paired_contrasts"]) == 48
    assert all(v["metrics"]["coverage_full_pool"] == 0.5 for v in result["conditions"].values())
    assert all(v["per_batch"]["B1"]["question_count"] == 2 for v in result["conditions"].values())
    assert pools == before
    for selector in baseline:
        reference, _, _ = pw.ev.rank_frozen(baseline[selector], test_pools, test)
        for pool in test_pools:
            assert result["conditions"][f"{pool}_{selector}_logistic_plain"]["metrics"] == reference[f"{pool}_{selector}_reranker"]["metrics"]
    with pytest.raises(FileExistsError):
        pw.run(submission, output, 20)


def test_prepare_pins_inputs_code_protocol_and_job(small, monkeypatch, tmp_path):
    examples, _, baseline = small
    repository = tmp_path / "repository"
    jobs = repository / "gadi_sft_8b_starter/jobs"; jobs.mkdir(parents=True)
    (jobs / "compare_pairwise_sft_rerankers.pbs").write_text("frozen PBS")
    monkeypatch.setattr(pw.ev, "load_comparison", lambda *args: ({}, {}, {}, {}, {"input": "hash"}))
    monkeypatch.setattr(pw.ev, "load_frozen", lambda *args: ({}, baseline, {"model": "hash"}))
    monkeypatch.setattr(pw, "load_training", lambda *args: (examples, {}))
    monkeypatch.setattr(pw.rr, "snapshot_code", lambda *args: {"worker": "hash"})
    submission = tmp_path / "submission"
    pw.prepare(repository, tmp_path / "baseline", tmp_path / "comparison", submission)
    prepared = pw.ev.read(submission / "prepared.json")
    assert prepared["baseline_hashes"] == {"model": "hash"}
    assert prepared["comparison_hashes"] == {"input": "hash"}
    assert prepared["penalties"][0] == 0 and prepared["pair_cap"] == 128
    assert prepared["job_sha256"] == pw.ev.digest(submission / "job.pbs")


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_validates_then_submits_only_one_frozen_cpu_job(tmp_path, fail):
    import os
    import shutil
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    source = pw.ROOT / "gadi_sft_8b_starter/scripts/submit_pairwise_sft_rerankers.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"; printf 'frozen job' > "$1/job.pbs"; return 0
    fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '400.gadi-pbs\\n'; }
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "jobs.log"
    result = subprocess.run([executable, path.as_posix()], capture_output=True, text=True,
                            env={**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(),
                                 "TEST_FAIL": str(int(fail))}, timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(log.read_text().splitlines()) == 1
        assert "job.pbs" in log.read_text()
        record, = tmp_path.glob("outputs/pairwise_sft_rerankers/submission-*/jobs.tsv")
        assert record.read_text().splitlines()[-1] == "comparison\t400.gadi-pbs"
    job = (pw.ROOT / "gadi_sft_8b_starter/jobs/compare_pairwise_sft_rerankers.pbs").read_text()
    assert "#PBS -q normal" in job and "ngpus" not in job
    assert "/code/scripts/compare_pairwise_sft_rerankers.py" in job
