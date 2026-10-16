"""Relation-only intervention, split isolation and frozen experiment checks."""
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import ablate_reranker_relation_labels as ab
from scripts.train_candidate_reranker_pilot import score_components


def candidate(q, answer, label):
    return {"question_id": q, "answer": answer, "label": label,
            "sources": ["llama31"], "source_ranks": {"llama31": 1 if label else 2},
            "sample_frequencies": {"llama31": 0.8 if label else 0.2},
            "relation_types": ["synonym" if label else "nomenclature_variant"],
            "surface_operations": [], "is_format_variant": False}


@pytest.fixture
def small():
    examples = {"train": {f"t{i}": {"question": "Which biomedical entity?", "snippets": [{"text": "correct answer"}]}
                          for i in range(5)},
                "validation": {"v": {"question": "validationonlytoken", "snippets": []},
                               "empty": {"question": "Empty question", "snippets": []}}}
    pools = {s: {f"{a}_{m}": [candidate(q, answer, label) for q in ee if q != "empty"
                              for answer, label in (("correct answer", 1), ("incorrect answer", 0))]
                 for a in ab.rr.ARMS for m in ab.rr.MODES} for s, ee in examples.items()}
    baseline = {}
    for selector, arms in ab.rr.SELECTORS.items():
        best, _ = ab.rr.fit_selector(pools, examples, [1.0], arms)
        baseline[selector] = {"components": best[2], "selected_C": 1.0, "selector": selector,
                              "training_arms": list(arms), "schema": "sft-ensemble-logistic-v1"}
    return examples, pools, baseline


def test_ablation_changes_only_relation_types_and_does_not_mutate(small):
    _, pools, _ = small
    before = copy.deepcopy(pools["train"])
    blind = ab.without_relations(before)
    for name, rows in before.items():
        for old, new in zip(rows, blind[name]):
            assert new == {**old, "relation_types": []}
    assert before == pools["train"]
    blind["original_greedy"][0]["sample_frequencies"]["llama31"] = 0
    assert pools["train"]["original_greedy"][0]["sample_frequencies"]["llama31"] == 0.8


def test_real_fit_ignores_tags_gold_and_validation_vocabulary(small, monkeypatch):
    examples, pools, baseline = small
    real_fit = ab.fit_components
    def checked(rows, contexts, labels, **kwargs):
        assert set(contexts) == set(examples["train"])
        assert all("label" not in r and "gold_aliases" not in r and not r["relation_types"] for r in rows)
        return real_fit(rows, contexts, labels, **kwargs)
    monkeypatch.setattr(ab, "fit_components", checked)
    fitted, trials = ab.fit_ablation(pools, examples, baseline, [0.1, 1.0])
    assert len(fitted) == 6
    for selector in baseline:
        assert fitted[f"{selector}_fixed_C"]["selected_C"] == 1.0
        assert fitted[f"{selector}_validation_selected"]["selected_C"] == 0.1  # Validation tie: first C.
        assert len(trials[selector]) == 2
    for payload in fitted.values():
        components = payload["components"]
        assert not any(n.startswith("relation=") for n in components["meta"].get_feature_names_out())
        assert "validationonlytoken" not in components["word"].vocabulary_
        before = ab.rr.feature_rows(pools["validation"]["max5_sampling4"])
        after = [{**r, "relation_types": ["invented_tag", "original"]} for r in before]
        np.testing.assert_array_equal(score_components(components, before, examples["validation"]),
                                      score_components(components, after, examples["validation"]))
    result, questions = ab.compare(pools["validation"], examples["validation"], baseline, fitted, 20)
    assert len(result["conditions"]) == 36
    assert len(result["paired_contrasts"]) == 24
    assert all(v["metrics"]["coverage_full_pool"] == 0.5 for v in result["conditions"].values())
    assert all(len(v) == 2 for v in questions.values())


def test_fixed_C_must_be_in_archived_grid(small):
    examples, pools, baseline = small
    with pytest.raises(ValueError, match="outside"):
        ab.fit_ablation(pools, examples, baseline, [0.1])


def test_frozen_input_tampering_and_paths_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("original")
    hashes = {"input.json": ab.ev.digest(path)}
    ab.verify_files(tmp_path, hashes)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        ab.verify_files(tmp_path, hashes)
    with pytest.raises(ValueError, match="changed"):
        ab.verify_files(tmp_path, {"../outside": "x"})


def test_complete_run_freezes_models_before_test_and_preserves_sources(small, monkeypatch, tmp_path):
    examples, pools, baseline = small
    directory = tmp_path / "baseline"; directory.mkdir()
    comparison = tmp_path / "comparison"; comparison.mkdir()
    submission = tmp_path / "submission"; submission.mkdir()
    original_submission = tmp_path / "original"; original_submission.mkdir()
    ab.ev.write(original_submission / "prepared.json", {"C_grid": [0.1, 1.0]})
    state = {"submission": str(original_submission)}
    ab.ev.write(submission / "prepared.json", {"baseline": str(directory), "comparison": str(comparison),
                "baseline_hashes": {}, "comparison_hashes": {}, "code_hashes": {}})
    test = {"testq": {"question": "testonlytoken", "snippets": []}}
    test_pools = {f"{a}_{m}": [candidate("testq", "correct answer", 1), candidate("testq", "incorrect answer", 0)]
                  for a in ab.ev.ARMS for m in ab.ev.MODES}
    monkeypatch.setattr(ab.ev, "load_comparison", lambda path: ({}, test_pools, test, {"testq": {"batch": "B1"}}, {}))
    monkeypatch.setattr(ab.ev, "load_frozen", lambda *args: (state, baseline, {}))
    monkeypatch.setattr(ab, "load_training", lambda *args: (examples, pools))
    output = tmp_path / "output"
    real_compare = ab.compare
    def checked(pp, ee, bb, fitted, resamples):
        assert len(list(output.glob("*.joblib"))) == 6
        assert (output / "validation_selection.json").exists()
        assert all("testonlytoken" not in p["components"]["word"].vocabulary_ for p in fitted.values())
        return real_compare(pp, ee, bb, fitted, resamples)
    monkeypatch.setattr(ab, "compare", checked)
    before = copy.deepcopy(pools)
    ab.run(submission, output, 20)
    assert ab.ev.read(output / "manifest.json")["status"] == "complete"
    result = ab.ev.read(output / "test_summary.json")
    assert result["question_count"] == 1 and len(result["conditions"]) == 36
    assert all(c["per_batch"]["B1"]["question_count"] == 1 for c in result["conditions"].values())
    assert before == pools
    with pytest.raises(FileExistsError):
        ab.run(submission, output, 20)


def test_launcher_one_frozen_CPU_job_and_no_extra_generation():
    starter = ab.ROOT / "gadi_sft_8b_starter"
    launcher = (starter / "scripts/submit_reranker_relation_ablation.sh").read_text()
    job = (starter / "jobs/ablate_reranker_relation_labels.pbs").read_text()
    assert launcher.count("qsub ") == 1
    assert launcher.index("--mode prepare") < launcher.index("qsub ")
    assert '"$submission/job.pbs"' in launcher
    assert "#PBS -q normal" in job and "#PBS -l ncpus=1" in job
    assert "/code/scripts/ablate_reranker_relation_labels.py" in job
    assert "ngpus" not in job


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_stops_on_failed_validation_and_records_actual_job(tmp_path, fail):
    import os
    import shutil
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    source = ab.ROOT / "gadi_sft_8b_starter/scripts/submit_reranker_relation_ablation.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift
      mkdir -p "$1"
      printf 'frozen job' > "$1/job.pbs"
      return 0
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
        records = list(tmp_path.glob("outputs/reranker_relation_ablation/submission-*/jobs.tsv"))
        assert len(records) == 1 and records[0].read_text().splitlines()[-1] == "ablation\t400.gadi-pbs"
