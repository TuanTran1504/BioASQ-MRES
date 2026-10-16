"""Question-disjoint feedback, fixed-context interventions and audit-only output."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np
import pytest

from scripts import prepare_reranker_guided_dpo as pilot
import max5_dpo_data as data


def config():
    return pilot.read(pilot.DEFAULT_CONFIG)


def raw_row(qid, model, draw, extra=False):
    answers = [{"answer": "correct name", "relation_type": "original"}]
    if extra:
        answers.append({"answer": "bad alternative", "relation_type": "synonym"})
    raw = json.dumps({"answers": answers})
    return {"question_id": qid, "model_key": model, "split": "train", "arm": "max5", "draw": draw,
            "raw_response": raw, "response_id": data.response_id(model, qid, draw, raw),
            "output_tokens": 20 if extra else 10,
            "prompt": [{"role": "system", "content": "Generate up to FIVE equivalent answers."},
                       {"role": "user", "content": json.dumps({"question": f"Which entity uniquetoken{qid}?",
                        "snippets": [{"id": "s1", "text": "The answer is correct name."}]})}]}


def dataset(n=10):
    cfg = config()
    raw, examples, labels = [], {}, {}
    for i in range(n):
        qid = str(i)
        slot = pilot.replacement_slot(qid, cfg["seed"], 4)
        raw.extend(raw_row(qid, m, d, extra=m == "llama31" and d == slot)
                   for m in pilot.MODELS for d in range(5))
        context = json.loads(raw[-1]["prompt"][1]["content"])
        examples[qid] = {"question_id": qid, **context}
        labels[(qid, "correct name")] = 1
        labels[(qid, "bad alternative")] = 0
    return cfg, raw, examples, labels


def imperfect_score(components, rows, examples):
    assert all("label" not in r and "gold_aliases" not in r for r in rows)
    assert all("gold_aliases" not in e for e in examples.values())
    return np.asarray([1.0 if r["answer"] == "bad alternative" else 0.5 for r in rows])


def test_only_one_focal_slot_changes_and_other_eleven_responses_are_identical(monkeypatch):
    cfg, raw, examples, labels = dataset(1)
    before = copy.deepcopy(raw)
    options = pilot.alternatives(raw, examples, labels, cfg, data)["0"]
    slot = pilot.replacement_slot("0", cfg["seed"], 4)
    assert len(options) == 5
    assert len({tuple(o["fixed_response_ids"]) for o in options}) == 1
    assert len(options[0]["fixed_response_ids"]) == 11
    assert sum(o["baseline"] for o in options) == 1
    assert next(o for o in options if o["baseline"])["response"]["draw"] == slot
    # Replacing the unique bad expansion changes the pool and normalised draw support.
    bad = next(o for o in options if o["baseline"])
    assert next(r for r in bad["pool"] if r["answer"] == "bad alternative")["sample_frequencies"] == {"llama31": 0.25}
    assert all(r["answer"] != "bad alternative" for r in options[0]["pool"])
    assert raw == before
    monkeypatch.setattr(pilot, "score_components", imperfect_score)
    scored = pilot.score_options({}, {"0": options}, examples)
    assert next(r for r in scored if r["baseline"])["mrr_at5"] == 0.5
    assert next(r for r in scored if r["draw"] == 0)["mrr_at5"] == 1.0


def test_pair_uses_reranked_outcome_preserves_initial_and_requires_semantic_review(monkeypatch):
    cfg, raw, examples, labels = dataset(1)
    options = pilot.alternatives(raw, examples, labels, cfg, data)
    monkeypatch.setattr(pilot, "score_components", imperfect_score)
    scored = [{**r, "fold": 1} for r in pilot.score_options({}, options, examples)]
    primary, other, _ = pilot.preference_candidates(scored, {r["response_id"]: r for r in raw}, cfg)
    assert len(primary) == 1 and not other  # Identical repeated responses are deduplicated.
    pair = primary[0]
    assert pair["reciprocal_rank_gain"] == 0.5
    assert data.answers(pair["chosen"])[0] == data.answers(pair["rejected"])[0]
    assert pair["category"] == "same_accepted_initial"
    assert pair["semantic_review_status"] == "pending" and pair["training_ready"] is False
    assert len(pair["fixed_response_ids"]) == 11 and pair["synthetic_chosen"] is False


@pytest.mark.parametrize("problem", ["validation", "missing_draw", "different_prompt"])
def test_preference_contexts_reject_nonfitting_incomplete_or_changed_inputs(problem):
    cfg, raw, examples, labels = dataset(1)
    if problem == "validation": raw[0]["split"] = "validation"
    elif problem == "missing_draw": raw.pop()
    else: raw[0]["prompt"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        pilot.alternatives(raw, examples, labels, cfg, data)


@pytest.mark.parametrize("problem", ["tie", "coverage_loss", "form_loss", "different_initial", "ceiling"])
def test_preference_guards_and_separate_initial_changes(monkeypatch, problem):
    cfg, raw, examples, labels = dataset(1)
    options = pilot.alternatives(raw, examples, labels, cfg, data)
    monkeypatch.setattr(pilot, "score_components", imperfect_score)
    scored = [{**r, "fold": 1} for r in pilot.score_options({}, options, examples)]
    if problem == "tie":
        for r in scored: r["mrr_at5"] = 0.5
    else:
        for r in scored:
            if not r["baseline"]:
                if problem == "coverage_loss": r["coverage_full_pool"] = 0
                if problem == "form_loss": r["response_summary"]["accepted_form_count"] = 0
                if problem == "different_initial": r["response_summary"]["initial_answer"] = "other accepted expression"
                if problem == "ceiling": r["response_summary"] = {"eligible": False, "reason": "output_token_ceiling"}
    primary, other, _ = pilot.preference_candidates(scored, {r["response_id"]: r for r in raw}, cfg)
    assert not primary
    if problem == "different_initial":
        assert other and all(p["category"] == "different_accepted_initial" for p in other)
    else:
        assert not other


def test_full_audit_fits_only_other_questions_and_never_exports_training_ready_files(tmp_path, monkeypatch):
    cfg, raw, examples, labels = dataset()
    rebuilt, _ = pilot.rr.build_pools(raw, {"train": examples})
    pools = rebuilt["train"]
    for rows in pools.values():
        for r in rows: r["label"] = labels[(r["question_id"], r["answer"])]
    repository = tmp_path / "repo"; repository.mkdir()
    banks = {}
    for model in pilot.MODELS:
        path = tmp_path / model; path.mkdir()
        pilot.write(path / "manifest.json", {"model_key": model})
        pilot.write_jsonl(path / "responses.jsonl", [r for r in raw if r["model_key"] == model])
        banks[model] = {"path": str(path), "manifest_sha256": pilot.digest(path / "manifest.json"),
                        "responses_sha256": pilot.digest(path / "responses.jsonl")}
    config_path = tmp_path / "config.json"; pilot.write(config_path, cfg)
    submission = tmp_path / "submission"; submission.mkdir()
    pilot.write(submission / "prepared.json", {"repository": str(repository), "rerankers": "unused",
        "configuration": cfg, "config_path": str(config_path), "config_sha256": pilot.digest(config_path),
        "code_hashes": {}, "reranker_hashes": {}, "banks": banks})
    monkeypatch.setattr(pilot, "load_inputs", lambda *args: (examples, pools, raw, labels, {}, banks, data))
    monkeypatch.setattr(pilot, "score_components", imperfect_score)
    original_fit = pilot.fit_components
    fitted = []
    def fit(rows, contexts, targets, **kwargs):
        assert {r["question_id"] for r in rows} <= set(contexts)
        assert all("label" not in r for r in rows)
        fitted.append(set(contexts))
        return original_fit(rows, contexts, targets, **kwargs)
    monkeypatch.setattr(pilot, "fit_components", fit)
    output = tmp_path / "audit"
    pilot.run(submission, output)
    state = pilot.read(output / "manifest.json")
    assert state["status"] == "review_required" and state["training_ready"] is False
    assert state["summary"]["primary_questions"] == 10
    assert state["summary"]["scored_alternatives"] == 50
    assert state["summary"]["baseline_oof_mrr_at5"] == 0.5
    assert state["summary"]["best_saved_alternative_oracle_mrr_at5"] == 1.0
    folds = pilot.read(output / "folds.json")
    assert len(fitted) == len(folds) == 5
    seen = []
    for fold in folds:
        train, held = set(fold["fitting_question_ids"]), set(fold["held_out_question_ids"])
        assert not train & held and train | held == set(examples)
        seen.extend(held)
        components = joblib.load(output / f"fold-{fold['fold']}.joblib")["components"]
        assert not any(f"uniquetoken{q}" in components["word"].vocabulary_ for q in held)
    assert sorted(seen) == sorted(examples)
    assert not (output / "train.jsonl").exists() and not (output / "validation.jsonl").exists()
    reviews = pilot.records(output / "review_responses.jsonl")
    assert reviews and all(c["official_accepted"] is not None for r in reviews for c in r["candidates"])
    assert all(not r["reviewer"] for r in reviews)
    with pytest.raises(FileExistsError):
        pilot.run(submission, output)


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_submits_one_cpu_audit_and_never_training(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    source = pilot.ROOT / "gadi_sft_8b_starter/scripts/submit_reranker_guided_dpo_audit.sh"
    scripts = tmp_path / "scripts"; scripts.mkdir()
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift; mkdir -p "$1"; return 0; fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '400.gadi-pbs\\n'; }
'''
    launcher = scripts / source.name
    launcher.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "jobs"
    result = subprocess.run([executable, launcher.as_posix()], capture_output=True, text=True, timeout=30,
        env={**os.environ, "TEST_FAIL": str(int(fail)), "TEST_USER": "test", "TEST_LOG": log.as_posix()})
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(log.read_text().splitlines()) == 1
        assert "jobs/audit_reranker_guided_dpo.pbs" in log.read_text()
        assert "No DPO training" in result.stdout
