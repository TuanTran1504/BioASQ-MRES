"""Coverage preferences: fitting isolation, real marginal coverage, raw lineage."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from scripts import audit_generator_coverage_dpo as coverage
from scripts import prepare_answer_selector_data as sd


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "COUNTS", {"train": 2, "validation": 1})
    examples = {s: [{"question_id": q, "question": "Which marker?", "snippets": [{"id": "s", "text": "Marker HER-2."}],
                     "gold_aliases": ["HER-2", "HER2"]} for q in qs]
                for s, qs in (("train", ("t0", "t1")), ("validation", ("v",)))}
    raw = []
    for split, ee in examples.items():
        for e in ee:
            prompt = [{"role": "system", "content": "Native max-five generator instructions."},
                      {"role": "user", "content": json.dumps({k: e[k] for k in ("question", "snippets")})}]
            for model in sd.MODELS:
                for draw in range(5):
                    # t0: a sole Llama success; t1: multiple accepted aliases/draws.
                    success = (model == "llama31" and draw == 1) or (e["question_id"] == "t1" and model == "qwen3" and draw in (1, 2))
                    answer = ("HER2" if model == "qwen3" else "HER-2") if success else f"decoy {draw}"
                    raw.append({"split": split, "question_id": e["question_id"], "model_key": model,
                        "draw": draw, "response_id": f"{e['question_id']}-{model}-{draw}", "prompt": prompt,
                        "raw_response": json.dumps({"answers": [{"answer": answer, "relation_type": "original"}]}), "output_tokens": 25})
    pools, _, response_audit = sd.pool_responses(raw, examples)
    for split, modes in pools.items():
        for rows in modes.values():
            for row in rows:
                row["official_accepted"] = row["answer"] in ("HER-2", "HER2")
    directory = tmp_path / "source"
    sd.construct_dataset(pools, examples, directory)
    sd.jsonl(directory / "response_audit.jsonl", response_audit)
    sd.write(directory / "heldout_ids.json", {"dev160": ["dev"], "test95": ["test"]})
    banks = {}
    for model in sd.MODELS:
        bank = tmp_path / "banks" / model
        sd.jsonl(bank / "responses.jsonl", [r for r in raw if r["model_key"] == model])
        sd.write(bank / "manifest.json", {"model_key": model})
        banks[model] = {"directory": str(bank), "manifest_sha256": sd.digest(bank / "manifest.json"),
                        "responses_sha256": sd.digest(bank / "responses.jsonl")}
    sd.write(directory / "source_provenance.json", {"native_banks": banks})
    sd.write(directory / "manifest.json", {"status": "complete", "schema": "max5-answer-selector-audit-v1",
        "question_counts": sd.COUNTS, "output_sha256": {p.name: sd.digest(p) for p in directory.iterdir() if p.is_file()}})
    return directory, raw


def indexed(directory):
    ii, ss, rr, protected, _ = coverage.load_inputs(directory)
    return coverage.index_responses(ii, ss, rr, protected)


def test_true_coverage_gain_distinct_from_companion_miss_and_response_success(tmp_path, monkeypatch):
    directory, raw = fixture(tmp_path, monkeypatch)
    questions, aliases, rows, slots = indexed(directory)
    coverage.attach_native([r for r in raw if r["split"] == "train"], questions, rows)
    summary, pairs, _ = coverage.build_audit(questions, aliases, rows, slots)
    assert summary["ensemble"]["sampling4_covered"] == 2
    assert summary["ensemble"]["greedy_covered"] == 0
    assert summary["models"]["llama31"]["sampled_mixed_questions"] == 2
    assert summary["models"]["llama31"]["sampled_mixed_companions_uncovered_questions"] == 1
    for p in pairs:
        assert p["split"] == "train" and not p["training_ready"]
        assert p["prompt"] and p["chosen"] and p["rejected"]
        assert any(c["official_accepted"] for c in p["chosen_candidates"])
        assert not any(c["official_accepted"] for c in p["rejected_candidates"])
        if p["objective"] == "sampled_response_coverage":
            assert len(p["fixed_response_ids"]) == 11
            assert p["chosen_id"] not in p["fixed_response_ids"]
            assert p["counterfactual_ensemble_coverage_gain"] == (p["question_id"] == "t0")
        else:
            assert len(p["fixed_response_ids"]) == 2 and p["rejected_draw"] == 0
    assert all(len([p for p in pairs if (p["model_key"], p["question_id"], p["objective"]) == key]) <= 2
               for key in {(p["model_key"], p["question_id"], p["objective"]) for p in pairs})
    assert summary["native_text_available"] and not summary["training_ready"]


def test_failed_ceiling_and_repaired_draws_stay_in_denominator_not_pairs(tmp_path, monkeypatch):
    directory, _ = fixture(tmp_path, monkeypatch)
    ii, ss, rr, protected, _ = coverage.load_inputs(directory)
    for r in rr:
        if r["model_key"] == "llama31" and r["draw"] == 1:
            r["warnings"] = ["output_token_ceiling"]
            r["output_tokens"] = 512
    q, a, rows, slots = coverage.index_responses(ii, ss, rr, protected)
    summary, pairs, _ = coverage.build_audit(q, a, rows, slots)
    assert summary["ensemble"]["sampling4_covered"] == 2
    assert summary["models"]["llama31"]["sampled_requests"] == 8
    assert summary["models"]["llama31"]["ineligible_sampled_requests"] == 2
    assert not any(p["model_key"] == "llama31" for p in pairs)
    assert not summary["native_text_available"]


@pytest.mark.parametrize("change", ["heldout", "missing_response", "duplicate_response", "wrong_mode", "wrong_labels", "missing_alias", "missing_position"])
def test_rejects_mismatched_membership_or_provenance(tmp_path, monkeypatch, change):
    directory, _ = fixture(tmp_path, monkeypatch)
    ii, ss, rr, protected, _ = coverage.load_inputs(directory)
    if change == "heldout": protected["test95"].append("t0")
    elif change == "missing_response": rr.pop()
    elif change == "duplicate_response": rr.append(copy.deepcopy(rr[0]))
    elif change == "wrong_mode": ss[0]["pool"] = "max5_sampling4"
    elif change == "wrong_labels": ss[0]["candidates"][0]["official_accepted"] = 1
    elif change == "missing_alias": ss[0]["gold_aliases"] = []
    elif change == "missing_position": ss[0]["candidates"][0]["occurrences"] = []
    with pytest.raises(ValueError): coverage.index_responses(ii, ss, rr, protected)


@pytest.mark.parametrize("change", ["metadata", "evidence", "surface", "incomplete", "duplicate"])
def test_native_binding_rejects_changed_prompt_raw_or_identity(tmp_path, monkeypatch, change):
    directory, raw = fixture(tmp_path, monkeypatch)
    q, _, rows, _ = indexed(directory)
    native = copy.deepcopy([r for r in raw if r["split"] == "train"])
    if change == "metadata": native[0]["draw"] = 4
    elif change == "evidence": native[0]["prompt"][1]["content"] = '{}'
    elif change == "surface": native[0]["raw_response"] = '{"answers": [{"answer":"changed", "relation_type":"original"}]}'
    elif change == "incomplete": native.pop()
    elif change == "duplicate": native.append(native[0])
    with pytest.raises(ValueError): coverage.attach_native(native, q, rows)


def test_frozen_portable_audit_and_native_bank_hash_tamper(tmp_path, monkeypatch):
    directory, _ = fixture(tmp_path, monkeypatch)
    coverage.frozen_audit(directory)
    (directory / "train_supervision.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"): coverage.frozen_audit(directory)


def test_complete_cpu_freeze_run_archive_keeps_validation_out(tmp_path, monkeypatch):
    directory, _ = fixture(tmp_path, monkeypatch)
    import scripts.train_sft_ensemble_reranker as rr
    def snapshot(repo, destination):
        for name in ("audit_generator_coverage_dpo.py", "prepare_answer_selector_data.py"):
            target = destination / "scripts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(coverage.ROOT / "scripts" / name, target)
        return {}
    monkeypatch.setattr(rr, "snapshot_code", snapshot)
    submission = tmp_path / "submission"
    coverage.prepare(coverage.ROOT, directory, submission)
    assert {r["split"] for r in sd.records(submission / "native_responses.jsonl")} == {"train"}
    assert not list(submission.glob("validation*"))
    out = tmp_path / "out"
    archive = tmp_path / "review.tar.gz"
    coverage.run(out, submission=submission, archive=archive)
    manifest = sd.read(out / "manifest.json")
    sd.verify_files(out, manifest["output_sha256"])
    assert manifest["status"] == "complete" and not manifest["training"]
    summary = sd.read(out / "summary.json")
    assert summary["validation_pairs"] == summary["test95_pairs"] == summary["dev160_pairs"] == 0
    assert summary["native_text_available"]
    assert {p["question_id"] for p in sd.records(out / "preference_candidates.jsonl")} <= {"t0", "t1"}
    with tarfile.open(archive) as handle:
        assert "report.md" in handle.getnames() and not any(n.startswith("/") for n in handle.getnames())
    with pytest.raises(ValueError): coverage.run(out, submission=submission)
    # Same pairs/counters when native text is unavailable, without inventing text.
    local = tmp_path / "local"
    coverage.run(local, audit=directory)
    other = sd.read(local / "summary.json")
    assert other["models"] == summary["models"] and not other["native_text_available"]
    assert {p["pair_id"] for p in sd.records(local / "preference_candidates.jsonl")} == {p["pair_id"] for p in sd.records(out / "preference_candidates.jsonl")}


def test_submission_job_uses_cpu_and_no_training_or_judging():
    job = (coverage.ROOT / "gadi_sft_8b_starter/jobs/audit_generator_coverage_dpo.pbs").read_text()
    wrapper = (coverage.ROOT / "gadi_sft_8b_starter/scripts/submit_generator_coverage_dpo_audit.sh").read_text()
    assert "#PBS -q normal" in job and "ngpus" not in job
    assert "--mode audit" in job and "--mode prepare" in wrapper
    assert "qsub" in wrapper and "generator-coverage-dpo-audit-" in job


@pytest.mark.parametrize("failure", ["none", "prepare", "qsub"])
def test_launcher_one_cpu_job_or_no_submission_on_failure(tmp_path, failure):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    stub = r'''
module() { :; }
source() { :; }
python3() {
  [[ "$FAILURE" != prepare ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift
      mkdir -p "$1"
      printf '# fake CPU job\n' > "$1/job.pbs"
      return 0
    fi
    shift
  done
  return 1
}
qsub() {
  [[ "$FAILURE" != qsub ]] || return 1
  printf '%s\n' "$*" >> "$TEST_LOG"
  printf '901.gadi-pbs\n'
}
'''
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / "submit.sh"
    original = (coverage.ROOT / "gadi_sft_8b_starter/scripts/submit_generator_coverage_dpo_audit.sh").read_text()
    launcher.write_text(original.replace("${USER}", "${TEST_USER}").replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "qsub.log"
    result = subprocess.run([executable, launcher.as_posix()], capture_output=True, text=True, timeout=30,
        env={**os.environ, "FAILURE": failure, "TEST_USER": "fixture", "TEST_LOG": log.as_posix()})
    records = list(tmp_path.glob("outputs/generator_coverage_dpo/submission-*/jobs.tsv"))
    if failure == "none":
        assert result.returncode == 0, result.stderr
        assert len(records) == 1 and "audit\t901.gadi-pbs" in records[0].read_text()
        assert len(log.read_text().splitlines()) == 1
    else:
        assert result.returncode != 0 and not records and not log.exists()
