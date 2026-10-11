"""Verify prompt isolation, candidate caps, paired gains/losses and base-only jobs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from run_base_candidate_budget import five_prompt
import run_base_candidate_budget as runner
from run_extractive_expansion_8b import parse_equivalent_response
from prepare_matched_8b_sft import write, digest, records
from scripts.analyze_base_candidate_budget import compare, validate_experiment


def response(count):
    return json.dumps({"answers": [{"answer": str(i), "relation_type": "original" if i == 0 else "synonym"}
                                  for i in range(count)]})


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_prompt_changes_only_two_budget_instructions():
    ten = (ROOT / "gadi_sft_8b_starter/prompts/equivalent_expansion_v1.txt").read_text()
    five = five_prompt(ten)
    assert "up to FIVE\nanswers TOTAL" in five and "Do not pad to five:" in five
    assert five.replace("FIVE", "TEN").replace("pad to five:", "pad to ten:") == ten
    with pytest.raises(ValueError):
        five_prompt("Ask for ten answers")


def test_parser_caps_five_records_surplus_and_keeps_historical_ten_default():
    accepted, rejected, issues, compliant = parse_equivalent_response(response(7), 5)
    assert len(accepted) == 5 and len(rejected) == 2 and not compliant
    assert "answer_count_out_of_range" in issues
    assert all(r["reason"] == "unique_candidate_limit_exceeded" for r in rejected)
    assert len(parse_equivalent_response(response(7))[0]) == 7
    assert len(parse_equivalent_response(response(10), 10)[0]) == 10
    with pytest.raises(ValueError):
        parse_equivalent_response(response(5), 3)


def test_equal_budget_and_full_pool_are_different_gain_loss_questions():
    before = [{"question_id": "gain", "coverage_at5": 0, "coverage_at10": 1},
              {"question_id": "loss", "coverage_at5": 1, "coverage_at10": 1},
              {"question_id": "fail", "coverage_at5": 0, "coverage_at10": 0}]
    after = [{"question_id": "gain", "coverage_at5": 1},
             {"question_id": "loss", "coverage_at5": 0},
             {"question_id": "fail", "coverage_at5": 0}]
    same = compare(before, after, "coverage_at5", resamples=200)
    full = compare(before, after, "coverage_at10", resamples=200)
    assert same["gained"] == 1 and same["lost"] == 1 and same["net_questions"] == 0
    assert full["gained"] == 0 and full["lost"] == 1 and full["net_questions"] == -1
    assert same["question_count"] == 3 and same["gained_question_ids"] == ["gain"]
    assert same["max10_covered"] == same["max5_covered"] == 1
    with pytest.raises(ValueError):
        compare(before, after[:-1], "coverage_at5", resamples=200)


def archive(tmp_path):
    submission = tmp_path / "prepared"
    submission.mkdir()
    ten = (ROOT / "gadi_sft_8b_starter/prompts/equivalent_expansion_v1.txt").read_text()
    for k, text in ((10, ten), (5, five_prompt(ten))):
        (submission / f"prompt_max{k}.txt").write_text(text, encoding="utf-8")
    source = {"base_snapshot": "/cache/base/snapshots/revision", "model_loader": "fast_language_model",
              "execution_mode": "default", "chat_template_kwargs": {"enable_thinking": False}}
    prepared = {"models": {"qwen3": source}, "input_sha256": "data",
                "prompt_sha256": {str(k): digest(submission / f"prompt_max{k}.txt") for k in (10, 5)}}
    write(submission / "manifest.json", prepared)
    directory = tmp_path / "experiment"
    directory.mkdir()
    manifest = {"status": "complete", "model_key": "qwen3", "smoke_test": False,
                "expected_questions": 160, "provenance": source,
                "prepared_manifest": str(submission / "manifest.json"),
                "prepared_manifest_sha256": digest(submission / "manifest.json"), "runs": {}, "arm_hashes": {}}
    panel = [{"question_id": str(i), "question": "Which?", "snippets": [], "gold_aliases": ["0"]} for i in range(160)]
    for k in (10, 5):
        name = f"max{k}"
        path = directory / name
        path.mkdir()
        write(path / "config.json", {"model_name": source["base_snapshot"], "max_candidates": k,
            "temperature": 0, "response_mode": "equivalent", "require_all_snippets": True,
            "local_files_only": True, "gold_blind_generation": True, "question_count": 160,
            "model_loader": source["model_loader"], "execution_mode": "default",
            "chat_template_kwargs": source["chat_template_kwargs"], "max_seq_length": 6144,
            "max_new_tokens": 512, "input_sha256": "data", "prompt_sha256": prepared["prompt_sha256"][str(k)],
            "prompt": str(submission / f"prompt_max{k}.txt")})
        jsonl(path / "examples.jsonl", panel)
        generation, candidates, rejected = [], [], []
        for i in range(160):
            raw = response(7) if i else "format failure"
            generation.append({"question_id": str(i), "raw_response": raw, "snippets_truncated": False,
                               "included_snippets": 0, "total_snippets": 0, "request_count": 1})
            if i:
                candidates.extend({"question_id": str(i), "position": c["raw_position"], **c}
                                  for c in parse_equivalent_response(raw, k)[0])
                rejected.extend({"question_id": str(i), **c} for c in parse_equivalent_response(raw, k)[1])
        jsonl(path / "generations.jsonl", generation)
        jsonl(path / "candidates.jsonl", candidates)
        jsonl(path / "invalid_candidates.jsonl", rejected)
        write(path / "status.json", {"status": "complete"})
        manifest["runs"][name] = name
        manifest["arm_hashes"][name] = {n: digest(path / n) for n in (
            "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
    write(directory / "manifest.json", manifest)
    return directory, manifest


def test_validator_replays_caps_and_retains_all_failed_questions(tmp_path):
    directory, manifest = archive(tmp_path)
    checked, runs = validate_experiment(directory)
    assert checked["expected_questions"] == 160
    assert len(records(runs["max5"] / "generations.jsonl")) == 160
    jsonl(runs["max5"] / "candidates.jsonl", [])
    manifest["arm_hashes"]["max5"]["candidates.jsonl"] = digest(runs["max5"] / "candidates.jsonl")
    write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="raw-response"):
        validate_experiment(directory)


@pytest.mark.parametrize("field,value", [("model_name", "/some/sft/adapter"), ("temperature", 0.8), ("max_new_tokens", 256)])
def test_validator_rejects_adapter_or_unmatched_settings(tmp_path, field, value):
    directory, manifest = archive(tmp_path)
    path = directory / "max5/config.json"
    config = json.loads(path.read_text())
    config[field] = value
    write(path, config)
    manifest["arm_hashes"]["max5"]["config.json"] = digest(path)
    write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="settings differ"):
        validate_experiment(directory)


def test_runner_loads_base_for_both_conditions_and_scores_smoke(tmp_path, monkeypatch):
    directory, manifest = archive(tmp_path)
    submission = Path(manifest["prepared_manifest"]).parent
    source = manifest["provenance"]
    prepared = {"input": "data.jsonl", "max_seq_length": 6144, "max_new_tokens": 512,
                "seed": 3407, "evaluation": "base only"}
    monkeypatch.setattr(runner, "validate_prepared", lambda *a: (prepared, source))
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        if "--model-name" not in command:
            assert "analyze_base_candidate_budget.py" in command[1] and "--allow-smoke" in command
            return
        assert command[command.index("--model-name") + 1] == source["base_snapshot"]
        config = json.loads(Path(command[command.index("--config") + 1]).read_text())
        parent = Path(command[command.index("--output-parent") + 1])
        path = parent / "generated"
        path.mkdir()
        for name in ("config.json", "status.json"):
            write(path / name, config if name == "config.json" else {"status": "complete"})
        for name in ("examples.jsonl", "candidates.jsonl", "invalid_candidates.jsonl"):
            jsonl(path / name, [])
        jsonl(path / "generations.jsonl", [{"raw_response": response(5)} for _ in range(4)])

    monkeypatch.setattr(runner.subprocess, "run", launch)
    output = tmp_path / "new_run"
    runner.run(submission, "qwen3", output, True)
    assert len(calls) == 3
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.run(submission, "qwen3", output, True)


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_submission_uses_smoke_dependencies_and_never_submits_training(tmp_path, fail_preflight):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "bundle/scripts"
    scripts.mkdir(parents=True)
    name = "submit_base_candidate_budget.sh"
    text = (ROOT / "gadi_sft_8b_starter/scripts" / name).read_text().replace("${USER}", "${TEST_USER}")
    script = scripts / name
    script.write_text(text, newline="\n")
    log, counter = tmp_path / "calls.txt", tmp_path / "count.txt"
    counter.write_text("0")
    environment = {**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(),
                   "TEST_COUNTER": counter.as_posix(), "FAIL_PREFLIGHT": str(int(fail_preflight))}
    stub = r'''
module() { :; }
source() { :; }
python() {
  if [[ "$FAIL_PREFLIGHT" == 1 ]]; then return 1; fi
  while (( $# )); do
    if [[ "$1" == --submission-dir ]]; then mkdir -p "$2"; break; fi
    shift
  done
}
qsub() {
  printf '%s\n' "$*" >> "$TEST_LOG"
  value=$(cat "$TEST_COUNTER")
  value=$((value + 1))
  printf '%s' "$value" > "$TEST_COUNTER"
  printf '%s.gadi-pbs\n' "$value"
}
task_script="$1"
shift
. "$task_script"
'''
    process = subprocess.run([executable, "-c", stub, "stub", script.as_posix()],
                             env=environment, capture_output=True, text=True)
    if fail_preflight:
        assert process.returncode and not log.exists()
    else:
        assert process.returncode == 0, process.stderr
        calls = log.read_text().splitlines()
        assert len(calls) == 6 and all("evaluate_base_candidate_budget.pbs" in c for c in calls)
        for i in range(3):
            assert f"depend=afterok:{2*i+1}.gadi-pbs" in calls[2*i+1]
