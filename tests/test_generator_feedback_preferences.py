"""Natural response preferences, frozen feedback and partition isolation."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import prepare_generator_feedback_preferences as gen
import max5_dpo_data as data


def response(q, model, draw, answers, split="train", tokens=30):
    raw = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                   for i, a in enumerate(answers)]})
    return {"question_id": q, "split": split, "model_key": model, "draw": draw,
            "raw_response": raw, "response_id": data.response_id(model, q, draw, raw), "output_tokens": tokens,
            "prompt": [{"role": "system", "content": "Generate up to FIVE answers."},
                       {"role": "user", "content": json.dumps({"question": "Which biomedical entity?",
                        "snippets": [{"id": "s1", "text": "correct name is the entity"}]})}]}


def outcome(row, labels, mrr=1.0, fold=1):
    split_labels = {(q, a): v for (s, q, a), v in labels.items() if s == row["split"]}
    return {"question_id": row["question_id"], "response_id": row["response_id"],
            "response_summary": gen.audit.response_summary(row, split_labels, data), "fold": fold,
            "fixed_response_ids": [f"fixed{i}" for i in range(11)], "replacement_slot": 1,
            "mrr_at5": mrr, "coverage_at5": int(mrr > 0), "coverage_full_pool": 1}


def test_ranking_and_compactness_are_separate_natural_pairs():
    a = response("q", "llama31", 0, ["correct name"])
    b = response("q", "llama31", 1, ["correct name", "unneeded alternative"])
    labels = {("train", "q", "correct name"): 1, ("train", "q", "unneeded alternative"): 0}
    rows = {r["response_id"]: r for r in (a, b)}
    scored = [outcome(a, labels, 1), outcome(b, labels, 0.5)]
    pairs, _ = gen.build_preferences(scored, rows, labels, "train", "llama31", data)
    assert len(pairs["ranking_gain"]) == len(pairs["compactness"]) == 1
    p = pairs["ranking_gain"][0]
    assert p["chosen"] == a["raw_response"] and p["rejected"] == b["raw_response"]
    assert p["category"] == "same_accepted_initial"
    assert not p["training_ready"] and not p["synthetic_chosen"]
    assert p["semantic_review_status"] == "pending"
    scored[1]["mrr_at5"] = 1
    pairs, _ = gen.build_preferences(scored, rows, labels, "train", "llama31", data)
    assert not pairs["ranking_gain"] and len(pairs["compactness"]) == 1


def test_compactness_keeps_accepted_forms_and_retention_controls_prefer_longer():
    short = response("q", "llama31", 0, ["unlisted correct alias"])
    long = response("q", "llama31", 1, ["unlisted correct alias", "accepted expression"])
    labels = {("train", "q", "unlisted correct alias"): 0, ("train", "q", "accepted expression"): 1}
    rows = {r["response_id"]: r for r in (short, long)}
    pairs, _ = gen.build_preferences([outcome(short, labels), outcome(long, labels)], rows,
                                     labels, "train", "llama31", data)
    assert not pairs["compactness"] and not pairs["ranking_gain"]
    assert len(pairs["retain_needed_expansion"]) == 1
    assert pairs["retain_needed_expansion"][0]["chosen"] == long["raw_response"]
    # Even if companions retain coverage, do not remove a focal accepted form.
    another = response("q", "llama31", 2, ["accepted expression"])
    second = response("q", "llama31", 3, ["accepted expression", "second accepted"])
    labels[("train", "q", "second accepted")] = 1
    rows = {r["response_id"]: r for r in (another, second)}
    pairs, _ = gen.build_preferences([outcome(another, labels), outcome(second, labels, 0.5)], rows,
                                     labels, "train", "llama31", data)
    assert not pairs["compactness"] and not pairs["ranking_gain"]


def test_excludes_tag_only_changes_ceiling_and_worse_ranking():
    a = response("q", "llama31", 0, ["correct name", "alias"])
    parsed = json.loads(a["raw_response"])
    parsed["answers"][1]["relation_type"] = "nomenclature_variant"
    b = {**a, "draw": 1, "raw_response": json.dumps(parsed)}
    b["response_id"] = data.response_id("llama31", "q", 1, b["raw_response"])
    labels = {("train", "q", "correct name"): 1, ("train", "q", "alias"): 0}
    rows = {r["response_id"]: r for r in (a, b)}
    pairs, diag = gen.build_preferences([outcome(a, labels), outcome(b, labels, 0.5)], rows,
                                        labels, "train", "llama31", data)
    assert not any(pairs.values()) and diag["tag_only_or_same_ordered_answers"] == 2
    short = response("q", "llama31", 2, ["correct name"])
    rows[short["response_id"]] = short
    pairs, _ = gen.build_preferences([outcome(short, labels, 0.5), outcome(a, labels)], rows,
                                     labels, "train", "llama31", data)
    assert not pairs["compactness"]
    ceiling = {**short, "output_tokens": 512}
    rows[short["response_id"]] = ceiling
    pairs, _ = gen.build_preferences([outcome(ceiling, labels), outcome(a, labels, 0.5)], rows,
                                     labels, "train", "llama31", data)
    assert not any(pairs.values())


def test_changed_initial_is_explicit_and_context_mismatch_is_rejected():
    a = response("q", "llama31", 0, ["correct name"])
    b = response("q", "llama31", 1, ["incorrect entity"])
    labels = {("train", "q", "correct name"): 1, ("train", "q", "incorrect entity"): 0}
    rows = {r["response_id"]: r for r in (a, b)}
    scored = [outcome(a, labels), outcome(b, labels, 0)]
    pairs, _ = gen.build_preferences(scored, rows, labels, "train", "llama31", data)
    assert len(pairs["ranking_gain"]) == 1 and not pairs["compactness"]
    assert pairs["ranking_gain"][0]["category"] == "changed_initial_expression"
    scored[1]["fixed_response_ids"] = ["changed"]
    with pytest.raises(ValueError, match="context"):
        gen.build_preferences(scored, rows, labels, "train", "llama31", data)
    with pytest.raises(ValueError, match="split"):
        gen.build_preferences(scored, rows, labels, "validation", "llama31", data)


def dataset():
    config = gen.audit.read(gen.audit.DEFAULT_CONFIG)
    examples, raw, labels = {s: {} for s in ("train", "validation")}, [], {}
    for split, count in (("train", 10), ("validation", 2)):
        for i in range(count):
            q = f"{split}{i}"
            rows = [response(q, model, d, ["correct name"] if d == 0 else ["correct name", "alternative"], split)
                    for model in gen.MODELS for d in range(5)]
            raw.extend(rows)
            examples[split][q] = {"question_id": q, **json.loads(rows[0]["prompt"][1]["content"])}
            labels[(split, q, "correct name")] = 1
            labels[(split, q, "alternative")] = 0
    return config, examples, raw, labels


@pytest.mark.parametrize("focal", gen.MODELS)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_options_keep_native_split_prompt_and_eleven_companions(split, focal):
    config, examples, raw, labels = dataset()
    original = copy.deepcopy(raw)
    options = gen.make_options(raw, examples[split], labels, split, focal, config, data)
    assert len(options) == len(examples[split])
    for q, values in options.items():
        assert len(values) == 5 and sum(v["baseline"] for v in values) == 1
        assert len({tuple(v["fixed_response_ids"]) for v in values}) == 1
        assert all(len(v["fixed_response_ids"]) == 11 for v in values)
        assert all(v["response"]["split"] == split and v["response"]["model_key"] == focal for v in values)
        assert all(r["question_id"] == q for v in values for r in v["pool"])
    assert original == raw


def folds_on_disk(directory, examples, config):
    qids = sorted(examples)
    folds, hashes = [], {}
    for fold, (fit, held) in enumerate(gen.KFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(qids), 1):
        membership = {"fitting_question_ids": sorted(qids[i] for i in fit),
                      "held_out_question_ids": sorted(qids[i] for i in held)}
        path = directory / f"fold-{fold}.joblib"
        joblib.dump({"components": {}, "C": config["reranker_C"], **membership}, path)
        hashes[path.name] = gen.audit.digest(path)
        folds.append({"fold": fold, **membership, "model_sha256": hashes[path.name]})
    gen.audit.write(directory / "folds.json", folds)
    hashes["folds.json"] = gen.audit.digest(directory / "folds.json")
    gen.audit.write(directory / "manifest.json", {"status": "review_required", "configuration": config,
                    "question_count": len(examples), "output_sha256": hashes})
    return folds


def test_fold_weights_and_question_membership_verified(tmp_path):
    config, examples, _, _ = dataset()
    folds_on_disk(tmp_path, examples["train"], config)
    gen.validate_folds(tmp_path, examples["train"], config)
    with (tmp_path / "fold-1.joblib").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="weights changed"):
        gen.validate_folds(tmp_path, examples["train"], config)


def test_run_uses_oof_only_for_fitting_and_final_frozen_model_for_validation(monkeypatch, tmp_path):
    config, examples, raw, labels = dataset()
    previous = tmp_path / "previous"; previous.mkdir()
    folds = folds_on_disk(previous, examples["train"], config)
    submission = tmp_path / "submission"; submission.mkdir()
    gen.audit.write(submission / "prepared.json", {"repository": "unused", "rerankers": "unused",
                    "previous_audit": str(previous), "code_hashes": {}, "reranker_hashes": {}, "banks": {},
                    "fold_hashes": {}, "audit_config": config, "models": ["llama31"],
                    "cap_per_question_per_objective": 2})
    final = {"components": {"validation_feedback": True}}
    monkeypatch.setattr(gen, "load_sources", lambda *args: (config, examples, raw, labels, folds, {}, {}, {}, final, data))
    def score(components, options, ee):
        is_validation = "validation_feedback" in components
        assert all(q.startswith("validation" if is_validation else "train") for q in ee)
        assert set(options) == set(ee)
        result = []
        for q, values in options.items():
            for option in values:
                r = option["response"]
                scored = outcome(r, labels, 1 if r["draw"] == 0 else 0.5)
                result.append({**scored, "fixed_response_ids": option["fixed_response_ids"],
                               "baseline": option["baseline"], "draw": r["draw"]})
        return result
    monkeypatch.setattr(gen.audit, "score_options", score)
    output = tmp_path / "output"
    gen.run(submission, output)
    state = gen.audit.read(output / "manifest.json")
    assert state["status"] == "review_required" and not state["training_ready"] and not state["DPO_training"]
    for split in examples:
        pairs = gen.audit.records(output / f"llama31/{split}_ranking_gain_review_candidates.jsonl")
        assert pairs and all(p["split"] == split and p["question_id"] in examples[split] for p in pairs)
        assert all(not p["training_ready"] and p["semantic_review_status"] == "pending" for p in pairs)
    reviews = gen.audit.records(output / "review_responses.jsonl")
    original = {r["response_id"]: r for r in raw}
    for r in reviews:
        assert all(r[k] == v for k, v in original[r["response_id"]].items())
    assert not (output / "train.jsonl").exists()
    assert not (output / "validation.jsonl").exists()
    with pytest.raises(FileExistsError):
        gen.run(submission, output)


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_defaults_to_llama_and_submits_one_cpu_job(tmp_path, fail):
    import os
    import shutil
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    source = gen.ROOT / "gadi_sft_8b_starter/scripts/submit_generator_feedback_preferences.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  printf '%s\\n' "$*" > "$TEST_ARGS"
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"; printf 'frozen job' > "$1/job.pbs"; return 0
    fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '401.gadi-pbs\\n'; }
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log, args = tmp_path / "jobs.log", tmp_path / "args.txt"
    result = subprocess.run([executable, path.as_posix()], capture_output=True, text=True,
                            env={**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(),
                                 "TEST_ARGS": args.as_posix(), "TEST_FAIL": str(int(fail))}, timeout=30)
    assert "--models llama31" in args.read_text()
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(log.read_text().splitlines()) == 1
        assert "job.pbs" in log.read_text()
        record, = tmp_path.glob("outputs/generator_feedback_dpo/submission-*/jobs.tsv")
        assert record.read_text().splitlines()[-1] == "preferences\t401.gadi-pbs"
