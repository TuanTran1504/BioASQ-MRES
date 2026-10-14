"""Verify pooled provenance, normalised votes, held-out selection and PBS dependencies."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_sft_ensemble_reranker as rr


def response(model, arm, draw, answer="correct", split="train", qid="q"):
    raw = (f"Answer: [BE]{answer}[EE]" if arm == "original" else
           json.dumps({"answers": [{"answer": answer, "relation_type": "original"},
                                   {"answer": "alternate", "relation_type": "synonym"}]}))
    return {"model_key": model, "arm": arm, "draw": draw, "raw_response": raw,
            "split": split, "question_id": qid, "prompt": [
                {"role": "system", "content": arm},
                {"role": "user", "content": json.dumps({"question": "Question?", "snippets": []})}]}


def candidate(qid, answer, label):
    return {"question_id": qid, "answer": answer, "label": label,
            "sources": ["llama31"], "source_ranks": {"llama31": 1},
            "sample_frequencies": {"llama31": 1}, "relation_types": ["original"],
            "surface_operations": [], "is_format_variant": False}


def test_pool_modes_dedup_and_normalised_model_votes():
    rows = [response(m, a, d) for m in rr.MODELS for a in rr.ARMS for d in range(5)]
    examples = rr.contexts_from_rows(rows)
    pools, diag = rr.build_pools(rows, examples)
    assert len(pools["train"]["original_greedy"]) == 1
    assert len(pools["train"]["original_sampling4"]) == 1
    assert len(pools["train"]["max5_sampling4"]) == 2
    for row in pools["train"]["max5_sampling4"]:
        assert row["sources"] == list(rr.MODELS)
        assert row["sample_frequencies"] == dict.fromkeys(rr.MODELS, 1.0)
        assert row["source_ranks"] == dict.fromkeys(rr.MODELS, 1 if row["answer"] == "correct" else 2)
        assert "label" not in row
    assert diag["train/max5/sampling4/responses"] == 12


def test_greedy_not_inserted_into_sampling_and_failed_draws_not_removed_from_support():
    rows = [response("llama31", "original", d, "only_greedy" if d == 0 else "sample") for d in range(5)]
    rows[2]["raw_response"] = "malformed"
    pools, diag = rr.build_pools(rows, rr.contexts_from_rows(rows))
    a = pools["train"]["original_sampling4"]
    assert [r["answer"] for r in a] == ["sample"]
    assert a[0]["sample_frequencies"]["llama31"] == 0.75
    assert diag["train/original/sampling4/empty_responses"] == 1


def test_source_rank_is_first_unique_draw_order_and_caps_expansion():
    rows = [response("llama31", "original", 1, "first"), response("llama31", "original", 2, "second")]
    extra = response("qwen3", "max5", 1)
    extra["raw_response"] = json.dumps({"answers": [{"answer": str(i), "relation_type": "original" if i == 0 else "synonym"} for i in range(8)]})
    rows.append(extra)
    pools, _ = rr.build_pools(rows, rr.contexts_from_rows(rows))
    assert [r["source_ranks"]["llama31"] for r in pools["train"]["original_sampling4"]] == [1, 2]
    assert len(pools["train"]["max5_sampling4"]) == 5


@pytest.mark.parametrize("kind", ["evidence", "split", "gold_prompt"])
def test_cross_formulation_evidence_and_question_split_checks(kind):
    rows = [response("llama31", "original", 0), response("qwen3", "max5", 0)]
    if kind == "evidence":
        rows[1]["prompt"][1]["content"] = json.dumps({"question": "Different?", "snippets": []})
    elif kind == "split":
        rows[1]["split"] = "validation"
    else:
        rows[1]["prompt"].append({"role": "assistant", "content": "gold answer"})
    with pytest.raises(ValueError):
        rr.contexts_from_rows(rows)


def test_official_labels_are_separate_from_features(monkeypatch, tmp_path):
    examples = {"train": {"q": {"question_id": "q", "question": "Question?", "snippets": []}}}
    row = candidate("q", "accepted", 0)
    pools = {"train": {"original_greedy": [row], "max5_greedy": [copy.deepcopy(row)]}}
    def matcher(ex, cand, report, jar_path):
        assert ex[0]["gold_aliases"] == ["accepted"]
        assert len(cand) == 1
        return {("q", "accepted"): True}
    monkeypatch.setattr(rr, "official_candidate_matches", matcher)
    rr.label_pools(pools, examples, {"q": ["accepted"]}, tmp_path, Path("fake.jar"))
    assert all(r["label"] == 1 for rs in pools["train"].values() for r in rs)
    assert "label" not in rr.feature_rows(pools["train"]["original_greedy"])[0]
    assert "gold_aliases" not in examples["train"]["q"]


def test_real_fit_keeps_validation_out_of_vocabulary_and_persists(tmp_path):
    examples = {"train": {}, "validation": {"v": {"question": "validationuniqueword", "snippets": []},
                                           "empty": {"question": "empty", "snippets": []}}}
    pools = {s: {f"{a}_{m}": [] for a in rr.ARMS for m in rr.MODES} for s in examples}
    for i in range(5):
        q = f"q{i}"
        examples["train"][q] = {"question": "Shared biomedical question", "snippets": []}
        for rows in pools["train"].values():
            rows.extend([candidate(q, "valid correct", 1), candidate(q, "invalid wrong", 0)])
    for rows in pools["validation"].values():
        rows.extend([candidate("v", "invalid wrong", 0), candidate("v", "valid correct", 1)])
    best, trials = rr.fit_selector(pools, examples, [0.1, 1.0])
    score, C, components, metrics, questions, _ = best
    assert score == 0.5 and C == 0.1  # Earliest tie; empty question retained.
    assert "validationuniqueword" not in components["word"].vocabulary_
    assert len(trials) == 2 and len(metrics) == 16
    assert all(len(v) == 2 for v in questions.values())
    rr.joblib.dump({"components": components}, tmp_path / "selector.joblib")
    loaded = rr.joblib.load(tmp_path / "selector.joblib")
    before = rr.score_components(components, rr.feature_rows(pools["validation"]["original_greedy"]), examples["validation"])
    after = rr.score_components(loaded["components"], rr.feature_rows(pools["validation"]["original_greedy"]), examples["validation"])
    np.testing.assert_array_equal(before, after)


def test_ranking_cannot_change_full_pool_coverage(monkeypatch):
    monkeypatch.setattr(rr, "score_components", lambda components, rows, examples: np.arange(len(rows)))
    pools = {"original_sampling4": [candidate("q", "wrong", 0), candidate("q", "correct", 1)]}
    metrics, _, _ = rr.rank_conditions({}, pools, {"q": {}, "empty": {}})
    assert {m["coverage_full_pool"] for m in metrics.values()} == {0.5}


def test_latest_records_preserve_other_models_on_single_model_retry(tmp_path):
    root = tmp_path / "outputs/max5_dpo_banks_8b"
    root.mkdir(parents=True)
    (root / "submission-old.tsv").write_text("model\tstage\tjob_id\n" + "".join(f"{m}\tfull\t{100+i}.gadi-pbs\n" for i, m in enumerate(rr.MODELS)))
    retry = root / "submission-new.tsv"
    retry.write_text("model\tstage\tjob_id\nministral3\tfull\t200.gadi-pbs\n")
    import os
    os.utime(retry, (2000000000, 2000000000))
    paths, jobs = rr.latest_max5_banks(tmp_path)
    assert jobs == {"llama31": "100.gadi-pbs", "qwen3": "101.gadi-pbs", "ministral3": "200.gadi-pbs"}
    assert paths["ministral3"].name == "ministral3-generate-200.gadi-pbs"


def test_cpu_job_waits_for_pending_banks_but_not_completed(monkeypatch, tmp_path):
    paths = {m: tmp_path / m for m in rr.MODELS}
    paths["llama31"].mkdir()
    (paths["llama31"] / "manifest.json").write_text('{"status":"complete"}')
    jobs = dict(zip(rr.MODELS, ("100.gadi-pbs", "101.gadi-pbs", "102.gadi-pbs")))
    calls = []
    def qstat(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="job_state = Q", stderr="")
    monkeypatch.setattr(rr.subprocess, "run", qstat)
    assert rr.pending_dependencies(paths, jobs) == ["101.gadi-pbs", "102.gadi-pbs"]
    assert len(calls) == 2
    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="job_state = F\nExit_status = 1", stderr=""))
    with pytest.raises(ValueError, match="exit=1"):
        rr.pending_dependencies(paths, jobs)


def test_snapshot_is_unchanged_after_checkout_edit(tmp_path):
    repository = tmp_path / "repository"
    path = repository / "scripts/worker.py"
    path.parent.mkdir(parents=True)
    path.write_text("old code")
    snapshot = tmp_path / "snapshot"
    hashes = rr.snapshot_code(repository, snapshot)
    path.write_text("new code")
    assert (snapshot / "scripts/worker.py").read_text() == "old code"
    assert hashes["scripts/worker.py"] == rr.digest(snapshot / "scripts/worker.py")


@pytest.mark.parametrize("dependency,fail", [("101.gadi-pbs:102.gadi-pbs", False), ("", False), ("", True)])
def test_launcher_submits_one_cpu_job_with_only_pending_dependencies(tmp_path, dependency, fail):
    import os
    import shutil
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    source = rr.ROOT / "gadi_sft_8b_starter/scripts/submit_sft_ensemble_reranker.sh"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift
      mkdir -p "$1"
      printf '%s' "$TEST_DEPENDENCY" > "$1/dependencies.txt"
      return 0
    fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '300.gadi-pbs\\n'; }
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "jobs.log"
    result = subprocess.run([executable, path.as_posix()], capture_output=True, text=True,
                            env={**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(), "TEST_FAIL": str(int(fail)),
                                 "TEST_DEPENDENCY": dependency}, timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        jobs = log.read_text().splitlines()
        assert len(jobs) == 1
        assert "jobs/train_sft_ensemble_reranker.pbs" in jobs[0]
        if dependency:
            assert f"-W depend=afterok:{dependency}" in jobs[0]
        else:
            assert "-W" not in jobs[0]


def test_real_official_matcher_labels_only_existing_candidates(tmp_path):
    import shutil
    jar = rr.ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    if not shutil.which("java") or not jar.exists():
        pytest.skip("Java evaluator unavailable")
    examples = {"train": {"q": {"question_id": "q", "question": "Which protein?", "snippets": []}}}
    pools = {"train": {"original_greedy": [candidate("q", "alpha protein", 0), candidate("q", "beta protein", 0)]}}
    rr.label_pools(pools, examples, {"q": ["alpha protein"]}, tmp_path, jar)
    assert [r["label"] for r in pools["train"]["original_greedy"]] == [1, 0]
    assert len(pools["train"]["original_greedy"]) == 2
