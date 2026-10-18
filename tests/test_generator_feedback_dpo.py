"""Reviewed generator feedback: blinded judging, provenance and training isolation."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import generator_feedback_dpo_data as data
import run_generator_feedback_dpo as runner
from scripts import judge_generator_feedback as judge


def test_expanded_review_reindexes_only_unjudged_units():
    row = raw("q", "train", 1, ["Core", "alias", "other"], {"Core"})
    jobs, units = judge.prepare([row])
    known = {jobs[0]["unit_ids"][0]: {"cached": True}}
    remaining = judge.missing_jobs(jobs, known)
    assert len(remaining) == 1 and not set(remaining[0]["unit_ids"]) & set(known)
    public = json.loads(remaining[0]["payload"]["messages"][1]["content"])
    assert [c["index"] for c in public["candidates"]] == list(range(len(public["candidates"])))
    assert "accepted_training_aliases" not in public
    assert len(remaining[0]["unit_ids"]) + len(known) == len(units)
    assert remaining[0]["request_sha256"] != jobs[0]["request_sha256"]


def test_expanded_pool_ties_preserve_other_pair_guards():
    a = raw("q", "train", 0, ["Core"], {"Core"})
    b = raw("q", "train", 1, ["Core", "alias"], {"Core"})
    p = pair(a, b, "compactness")
    p["chosen_outcome"]["pool_candidates"] = p["rejected_outcome"]["pool_candidates"]
    originals = {r["response_id"]: r for r in (a,b)}
    with pytest.raises(ValueError): data.check_pair(p, originals)
    data.check_pair(p, originals, allow_pool_ties=True)
    p["chosen_outcome"]["mrr_at5"] = 0
    with pytest.raises(ValueError): data.check_pair(p, originals, allow_pool_ties=True)


def raw(q, split, draw, answers, accepted):
    text = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                  for i, a in enumerate(answers)]})
    r = {"model_key": "llama31", "question_id": q, "split": split, "draw": draw,
         "raw_response": text, "response_id": data.data.response_id("llama31", q, draw, text),
         "output_tokens": 20 + 10*len(answers), "excluded_reason": "", "reviewer": "", "review_notes": "",
         "prompt": [{"role": "system", "content": "Return five answers."}, {"role": "user", "content": json.dumps({
             "question": "Which entity?", "snippets": [{"id": "s1", "text": "Core and alias name the same concept."}]})}],
         "candidates": [{"answer": a, "relation_type": "original" if i == 0 else "synonym",
                         "official_accepted": a in accepted} for i, a in enumerate(answers)]}
    return r


def pair(a, b, arm):
    outcome = lambda r, size: {"mrr_at5": 1.0, "coverage_full_pool": 1, "pool_candidates": size,
                              "fixed_response_ids": [f"fixed{i}" for i in range(11)]}
    return {"question_id": a["question_id"], "split": a["split"], "model_key": "llama31", "objective": arm,
            "category": "same_accepted_initial" if arm == "compactness" else "same_unaccepted_initial",
            "chosen_id": a["response_id"], "rejected_id": b["response_id"], "chosen": a["raw_response"],
            "rejected": b["raw_response"], "prompt": a["prompt"], "chosen_outcome": outcome(a, 4),
            "rejected_outcome": outcome(b, 5), "fixed_response_ids": [f"fixed{i}" for i in range(11)],
            "training_ready": False, "semantic_review_status": "pending"}


def annotate(r):
    r = copy.deepcopy(r)
    r["reviewer"] = "LLM:gpt-4.1-mini-2025-04-14"
    for c in r["candidates"]:
        c["judge_decision"] = {"correctness": "correct", "support": "supported", "equivalence": "equivalent",
            "relation_valid": "valid", "confidence": "high", "validation_issues": [],
            "evidence": [{"snippet_id": "s1", "quote": "Core and alias name the same concept."}],
            "rationale": "The supplied evidence establishes the expression."}
    return r


@pytest.fixture
def package(tmp_path):
    source, judgment = tmp_path / "source", tmp_path / "judge"
    source.mkdir(); judgment.mkdir()
    rows = []
    for split, count in (("train", 5), ("validation", 2)):
        files = {arm: [] for arm in ("compactness", "retain_needed_expansion", "ranking_gain")}
        for i in range(count):
            q = f"{split}{i}"
            a, b = raw(q, split, 0, ["Core"], {"Core"}), raw(q, split, 1, ["Core", "alias"], {"Core"})
            c, d = raw(q, split, 2, ["same concept", "Core"], {"Core"}), raw(q, split, 3, ["same concept"], {"Core"})
            rows.extend((a, b, c, d))
            files["compactness"].append(pair(a, b, "compactness"))
            files["retain_needed_expansion"].append(pair(c, d, "retain_needed_expansion"))
        for arm, pp in files.items():
            data.write_rows(source / f"llama31/{split}_{arm}_review_candidates.jsonl", pp)
    data.write_rows(source / "review_responses.jsonl", rows)
    hashes = {p.relative_to(source).as_posix(): data.digest(p) for p in source.rglob("*") if p.is_file()}
    data.write(source / "manifest.json", {"status": "review_required", "models": ["llama31"], "output_sha256": hashes})
    data.write_rows(judgment / "reviewed.jsonl", [annotate(r) for r in rows])
    data.write(judgment / "manifest.json", {"status": "complete", "source_manifest_sha256": data.digest(source / "manifest.json"),
               "reviewed_sha256": data.digest(judgment / "reviewed.jsonl")})
    config = data.read(data.DEFAULT_CONFIG)
    config.update(minimum_train_questions=1, minimum_validation_questions=1)
    config_path = tmp_path / "config.json"
    data.write(config_path, config)
    return source, judgment, rows, config, config_path


def test_judge_is_blind_to_acceptance_preference_and_model_identity(package):
    source, _, _, _, _ = package
    _, pairs, rows = judge.load_shortlist(source)
    jobs, units = judge.prepare(rows)
    assert pairs and jobs and units
    for j in jobs:
        public = json.loads(j["payload"]["messages"][1]["content"])
        assert set(public) == {"question", "snippets", "candidates"}
        assert "accepted_training_aliases" not in public
        assert all(set(c) == {"index", "original", "answer", "declared_relation_type"} for c in public["candidates"])
        assert j["payload"]["model"] == "gpt-4.1-mini-2025-04-14"
        assert j["payload"]["store"] is False
        assert j["maximum_cost_reserve_usd"] > 0


def test_reconstructs_balanced_question_capped_pairs_and_preserves_validation(package):
    source, judgment, _, config, _ = package
    rows, summary = data.choose_pairs(source, data.records(judgment / "reviewed.jsonl"), config)
    assert len(rows["train"]) == 10 and len(rows["validation"]) == 4
    assert summary["counts"]["train"]["selected_objectives"] == {"compactness": 5, "retain_needed_expansion": 5}
    assert not {r["question_id"] for r in rows["train"]} & {r["question_id"] for r in rows["validation"]}
    assert all(r["training_ready"] and r["semantic_review_status"].startswith("LLM_high") for rr in rows.values() for r in rr)


def test_rejects_changed_response_uncertainty_and_wrong_preferred_answer(package):
    source, judgment, _, config, _ = package
    rows = data.records(judgment / "reviewed.jsonl")
    bad = copy.deepcopy(rows); bad[0]["raw_response"] += " altered"
    with pytest.raises(ValueError, match="raw generation"):
        data.choose_pairs(source, bad, config)
    bad = copy.deepcopy(rows); bad[0]["candidates"][0]["judge_decision"]["confidence"] = "low"
    _, summary = data.choose_pairs(source, bad, config)
    assert summary["excluded"]["unresolved_or_conflicting_response_review"] >= 1
    bad = copy.deepcopy(rows); bad[0]["candidates"][0]["judge_decision"]["correctness"] = "incorrect"
    _, summary = data.choose_pairs(source, bad, config)
    assert summary["excluded"]


def test_pair_guards_retain_accepted_expression_and_fixed_context(package):
    source, _, rows, _, _ = package
    _, pairs, _ = judge.load_shortlist(source)
    by_id = {r["response_id"]: r for r in rows}
    p = next(p for p in pairs if p["objective"] == "compactness")
    bad = copy.deepcopy(p); bad["chosen_outcome"]["mrr_at5"] = 0
    with pytest.raises(ValueError, match="preservation"):
        data.check_pair(bad, by_id)
    bad = copy.deepcopy(p); bad["chosen_outcome"]["fixed_response_ids"] = ["different"]
    with pytest.raises(ValueError, match="preservation"):
        data.check_pair(bad, by_id)
    bad = copy.deepcopy(p); bad["chosen_outcome"]["pool_candidates"] = 5
    with pytest.raises(ValueError, match="shrink"):
        data.check_pair(bad, by_id)


def test_ready_package_rebuild_and_tamper_detection(package, tmp_path):
    source, judgment, _, config, config_path = package
    output = tmp_path / "ready"
    data.build(source, judgment, output, config_path)
    manifest, splits = data.validate(output, config)
    assert manifest["status"] == "ready" and splits["train"]
    (output / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        data.validate(output, config)


def test_refuses_insufficient_distinct_questions(package, tmp_path):
    source, judgment, _, config, path = package
    config["minimum_train_questions"] = 999
    data.write(path, config)
    with pytest.raises(ValueError, match="Insufficient reviewed train"):
        data.build(source, judgment, tmp_path / "out", path)


def test_multi_model_review_builds_separate_focal_packages_without_cross_model_pairs(package, tmp_path):
    source, judgment, original, config, _ = package
    annotations = data.records(judgment / "reviewed.jsonl")
    for model in ("qwen3", "ministral3"):
        identities = {}
        for row in original:
            copied = copy.deepcopy(row)
            copied["model_key"] = model
            copied["response_id"] = data.data.response_id(model, row["question_id"], row["draw"], row["raw_response"])
            identities[row["response_id"]] = copied["response_id"]
            annotations.append(annotate(copied))
        for split in ("train", "validation"):
            for objective in ("compactness", "retain_needed_expansion", "ranking_gain"):
                pp = data.records(source / f"llama31/{split}_{objective}_review_candidates.jsonl")
                pp = [{**p, "model_key": model, "chosen_id": identities[p["chosen_id"]], "rejected_id": identities[p["rejected_id"]]} for p in pp]
                data.write_rows(source / f"{model}/{split}_{objective}_review_candidates.jsonl", pp)
    data.write_rows(source / "review_responses.jsonl", annotations)
    data.write(source / "manifest.json", {"status":"review_required","models":["llama31","qwen3","ministral3"],
        "output_sha256":{p.relative_to(source).as_posix():data.digest(p) for p in source.rglob("*") if p.is_file() and p.name != "manifest.json"}})
    data.write_rows(judgment / "reviewed.jsonl", annotations)
    data.write(judgment / "manifest.json", {"status":"complete","review_scope":"all_compactness",
        "source_manifest_sha256":data.digest(source / "manifest.json"),"reviewed_sha256":data.digest(judgment / "reviewed.jsonl")})
    _, all_pairs, all_rows = judge.load_shortlist(source,"all_compactness")
    assert {p["model_key"] for p in all_pairs} == {"llama31","qwen3","ministral3"}
    assert len(all_rows) == len(annotations)
    for model in ("qwen3", "ministral3"):
        cfg = {**config,"model_key":model,"review_scope":"all_compactness"}
        path = tmp_path / (model+".json"); data.write(path,cfg)
        out = tmp_path / (model+"-preferences")
        data.build(source,judgment,out,path)
        manifest, splits = data.validate(out,cfg)
        assert manifest["model_key"] == model
        assert {p["model_key"] for rr in splits.values() for p in rr} == {model}
        with pytest.raises(ValueError): data.validate(out,{**cfg,"model_key":"llama31"})
    with pytest.raises(ValueError): judge.load_shortlist(source,"all_compactness","wrong")


def test_judging_new_source_reuses_cumulative_budget_without_reset(package, tmp_path):
    source, _, _, _, _ = package
    budget = tmp_path / "existing-budget"
    ledger = {"attempts":[{"charged_or_reserved_usd":1.0278268,"status":"complete"}]}
    judge.shared.save(budget / "spend-ledger.json",ledger)
    output = tmp_path / "new-judge"
    judge.run(source,output,scope="all_compactness",budget_directory=budget)
    state = data.read(output / "manifest.json")
    assert state["budget_directory"] == str(budget.resolve()) and state["max_usd"] == 10
    assert data.read(budget / "spend-ledger.json") == ledger
    with pytest.raises(ValueError,match="changed judging plan"):
        judge.run(source,output,scope="all_compactness")


@pytest.mark.parametrize("model", ["llama31", "qwen3", "ministral3"])
def test_gpu_worker_uses_reviewed_validator_and_stays_separate_from_test(monkeypatch, tmp_path, model):
    import types
    submission = tmp_path / "submission"; submission.mkdir()
    pref = tmp_path / "preferences"; pref.mkdir()
    data.write(pref / "manifest.json", {"status": "ready"})
    config = data.read(data.DEFAULT_CONFIG)
    config["model_key"] = model
    source = {"adapter": "exact_sft_adapter", "adapter_files_sha256": {}}
    splits = {"train": [{"question_id": "fitting"}], "validation": [{"question_id": "validation"}]}
    data.write(submission / "prepared.json", {"code_hashes": {}, "preferences": str(pref), "repository": str(tmp_path),
          "preferences_manifest_sha256": data.digest(pref / "manifest.json"), "configuration": config,
          "source": source, "native_banks": {}})
    monkeypatch.setattr(runner, "validate_native", lambda *args: ({"pair_counts": {"train": 1, "validation": 1}}, splits, source, {"banks": {}}))
    monkeypatch.setattr(runner, "version", lambda *args: "test")
    monkeypatch.setattr(runner, "activate_portable_helpers", lambda: None)
    monkeypatch.setattr(data, "validate", lambda *args: (None, splits))
    called = []
    def train(args, cfg, ss, output, state, *, preference_validator):
        assert ss == source and args.model == model and state["model_key"] == model
        manifest, pp = preference_validator(args.preferences, cfg)
        assert pp is splits and manifest["pair_counts"]["train"] == 1
        assert not cfg["epoch_validation_mrr"]
        called.append(True)
        state.update(phase="save_adapter", global_steps=2)
    monkeypatch.setitem(sys.modules, "run_expansion_dpo_8b", types.SimpleNamespace(train=train))
    output = tmp_path / "output"
    runner.run(submission, output, smoke=True)
    assert called and data.read(output / "manifest.json")["status"] == "complete"


def test_portable_helpers_resolve_after_evaluation_src_imports():
    import subprocess
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, 'gadi_sft_8b_starter/scripts')
import run_generator_feedback_dpo as r
import src.utility.eval_openai
r.activate_portable_helpers()
from importlib.util import find_spec
for name in ('training_execution', 'text_tokenizer', 'adapter_save'):
    spec = find_spec('src.utility.' + name)
    assert Path(spec.origin).resolve().is_relative_to((Path.cwd()/'gadi_sft_8b_starter/src').resolve())
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_preparation_pins_portable_training_helpers(monkeypatch, tmp_path):
    repo = tmp_path / "repository"
    starter = repo / "gadi_sft_8b_starter"
    (starter / "configs").mkdir(parents=True)
    data.write(starter / "configs/generator_feedback_dpo.json", data.read(data.DEFAULT_CONFIG))
    (starter / "jobs").mkdir()
    (starter / "jobs/train_generator_feedback_dpo.pbs").write_text("frozen PBS")
    helper = starter / "src/utility/training_execution.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("pinned portable helper")
    pref = tmp_path / "prefs"; pref.mkdir()
    data.write(pref / "manifest.json", {})
    monkeypatch.setattr(runner, "validate_native", lambda *args: ({}, {"train": [1], "validation": [2]}, {}, {"banks": {}}))
    monkeypatch.setattr(runner, "snapshot_code", lambda *args: {})
    submission = tmp_path / "submission"
    runner.prepare(repo, pref, submission)
    prepared = data.read(submission / "prepared.json")
    relative = "gadi_sft_8b_starter/src/utility/training_execution.py"
    assert prepared["code_hashes"][relative] == data.digest(submission / "code" / relative)
    helper.write_text("later checkout edit")
    assert (submission / "code" / relative).read_text() == "pinned portable helper"


@pytest.mark.parametrize("model_key", ["llama31", "qwen3", "ministral3"])
def test_native_bank_and_exact_raw_response_checks(monkeypatch, tmp_path, model_key):
    import types
    pref = tmp_path / "pref"; (pref / "source").mkdir(parents=True)
    generation = tmp_path / "generation.json"; data.write(generation, {})
    source = {"adapter": "exact_max5_sft"}
    native = raw("q", "train", 0, ["Core"], {"Core"})
    native["model_key"] = model_key
    native["response_id"] = data.data.response_id(model_key,"q",0,native["raw_response"])
    data.write_rows(pref / "annotations.jsonl", [native])
    banks, pinned = {}, {}
    for model in ("llama31", "qwen3", "ministral3"):
        path = tmp_path / model; path.mkdir()
        data.write(path / "manifest.json", {})
        data.write_rows(path / "responses.jsonl", [native])
        pinned[model] = {"path": str(path), "manifest_sha256": data.digest(path / "manifest.json"),
                         "responses_sha256": data.digest(path / "responses.jsonl")}
        banks[model] = {"manifest": {"source": source if model == model_key else {"adapter":"wrong_"+model}}}
    feedback = tmp_path / "feedback"; feedback.mkdir()
    data.write(feedback / "prepared.json", {"banks": pinned})
    data.write(pref / "source/manifest.json", {"submission": str(feedback), "prepared_sha256": data.digest(feedback / "prepared.json")})
    splits = {"train": [native] * 36, "validation": [native]}
    monkeypatch.setattr(data, "validate", lambda *args: ({}, splits))
    def selected(config, model):
        assert model == model_key
        return source
    fake = types.SimpleNamespace(DEFAULT_CONFIG=generation, MODELS=tuple(banks),
           pack_banks=lambda *args: (banks, [native]), selected_source=selected)
    monkeypatch.setattr(runner, "configure_data_root", lambda *args: fake)
    config = {**data.read(data.DEFAULT_CONFIG),"model_key":model_key}
    assert runner.validate_native(pref, config, tmp_path)[2] == source
    changed = {**native, "raw_response": "changed completion"}
    data.write_rows(pref / "annotations.jsonl", [changed])
    with pytest.raises(ValueError, match="original native bank"):
        runner.validate_native(pref, config, tmp_path)
    data.write_rows(pref / "annotations.jsonl", [native])
    (tmp_path / "qwen3/responses.jsonl").write_text("changed bank")
    with pytest.raises(ValueError, match="bank changed"):
        runner.validate_native(pref, config, tmp_path)


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_validates_before_two_dependent_jobs(tmp_path, fail):
    import os
    import shutil
    import subprocess
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    pref = tmp_path / "preferences"; pref.mkdir()
    source = ROOT / "gadi_sft_8b_starter/scripts/submit_generator_feedback_dpo.sh"
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
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '402.gadi-pbs\\n'; }
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "jobs.log"
    result = subprocess.run([executable, path.as_posix(), pref.as_posix()], capture_output=True, text=True,
                            env={**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(),
                                 "TEST_FAIL": str(int(fail))}, timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        lines = log.read_text().splitlines()
        assert len(lines) == 2 and "afterok:402.gadi-pbs" in lines[1]
        assert all("job.pbs" in line for line in lines)
