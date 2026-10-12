"""Check one-answer prompt, base-only generation, matching and paired comparisons."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_base_candidate_budget as fixtures
import run_base_single_answer as runner
import scripts.analyze_base_single_answer as analysis
from run_extractive_expansion_8b import parse_equivalent_response
from prepare_matched_8b_sft import write, digest, read


def test_one_prompt_preserves_grounding_and_json_without_expansion():
    ten = (ROOT / "gadi_sft_8b_starter/prompts/equivalent_expansion_v1.txt").read_text()
    one = runner.one_prompt(ten)
    assert one.startswith(ten.split("First choose your single best answer concept.")[0])
    assert "exactly ONE" in one and '"relation_type": "original"' in one
    assert "Answer: [BE]" not in one and "Apply only these transformations" not in one
    with pytest.raises(ValueError):
        runner.one_prompt("Unknown prompt")


def test_parser_one_candidate_cap_and_failure():
    accepted, rejected, issues, compliant = parse_equivalent_response(fixtures.response(5), 1)
    assert len(accepted) == 1 and len(rejected) == 4 and not compliant
    assert accepted[0]["answer"] == "0" and "answer_count_out_of_range" in issues
    assert len(parse_equivalent_response(fixtures.response(5))[0]) == 5
    with pytest.raises(ValueError):
        parse_equivalent_response("unparseable", 1)


def test_paired_mrr_and_coverage_include_losses_and_failures():
    one = [{"question_id": q, "mrr_at5": value, "coverage_at1": value,
            "coverage_at5": value, "coverage_at10": value} for q, value in (("a", 1.0), ("b", 0.0), ("c", 0.0))]
    expanded = [{**one[0], "mrr_at5": 0.0, "coverage_at1": 0.0, "coverage_at5": 0.0, "coverage_at10": 1.0},
                {**one[1], "mrr_at5": 0.5, "coverage_at5": 1.0, "coverage_at10": 1.0}, one[2]]
    compared = analysis.paired_results(one, expanded, "max10", resamples=100)
    assert compared["mrr_at5"]["max10_minus_max1"] == pytest.approx(-1/6)
    assert compared["coverage_at5"]["gained"] == compared["coverage_at5"]["lost"] == 1
    assert compared["coverage_at10"]["gained"] == 1 and compared["coverage_at10"]["lost"] == 0
    assert compared["coverage_at5"]["question_count"] == 3


def single_archive(tmp_path, monkeypatch):
    baseline, old = fixtures.archive(tmp_path)
    baseline_runs = {k: baseline / v for k, v in old["runs"].items()}
    source = {"provenance": old["provenance"], "baseline": str(baseline),
              "baseline_manifest_sha256": digest(baseline / "manifest.json"),
              "input": "data.jsonl", "input_sha256": "data", "prompt_sha256": "one-prompt",
              "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407}
    submission = tmp_path / "single-submission"
    submission.mkdir()
    write(submission / "manifest.json", {"ready": True})
    prompt = submission / "one.txt"
    prompt.write_text("Single prompt")
    source["prompt_sha256"] = digest(prompt)
    monkeypatch.setattr(analysis, "validate_submission", lambda *a: ({}, source, baseline_runs))
    experiment = tmp_path / "single"
    path = experiment / "max1"
    path.mkdir(parents=True)
    provenance = source["provenance"]
    config = {"model_name": provenance["base_snapshot"], "model_loader": provenance["model_loader"],
              "chat_template_kwargs": provenance["chat_template_kwargs"], "execution_mode": "default",
              "max_seq_length": 6144, "max_new_tokens": 512, "temperature": 0,
              "max_candidates": 1, "response_mode": "equivalent", "require_all_snippets": True,
              "local_files_only": True, "gold_blind_generation": True, "question_count": 160,
              "input_sha256": "data", "prompt_sha256": source["prompt_sha256"], "seed": 3407, "prompt": str(prompt)}
    write(path / "config.json", config)
    fixtures.jsonl(path / "examples.jsonl", fixtures.records(baseline_runs["max10"] / "examples.jsonl"))
    generations, candidates, rejected = [], [], []
    for i in range(160):
        raw = fixtures.response(2) if i else "unparseable"
        generations.append({"question_id": str(i), "raw_response": raw, "request_count": 1,
                            "snippets_truncated": False, "included_snippets": 0, "total_snippets": 0})
        if i:
            accepted, invalid, _, _ = parse_equivalent_response(raw, 1)
            candidates.extend({"question_id": str(i), "position": r["raw_position"], **r} for r in accepted)
            rejected.extend({"question_id": str(i), **r} for r in invalid)
    fixtures.jsonl(path / "generations.jsonl", generations)
    fixtures.jsonl(path / "candidates.jsonl", candidates)
    fixtures.jsonl(path / "invalid_candidates.jsonl", rejected)
    write(path / "status.json", {"status": "complete"})
    manifest = {"status": "complete", "smoke_test": False, "expected_questions": 160, "model_key": "qwen3",
                "prepared_manifest": str(submission / "manifest.json"),
                "prepared_manifest_sha256": digest(submission / "manifest.json"),
                "baseline": source["baseline"], "baseline_manifest_sha256": source["baseline_manifest_sha256"],
                "provenance": provenance, "runs": {"max1": "max1"},
                "arm_hashes": {"max1": {n: digest(path / n) for n in (
                    "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}}}
    write(experiment / "manifest.json", manifest)
    return experiment, manifest


def test_single_validator_retains_failures_and_rejects_dropped_candidates(tmp_path, monkeypatch):
    directory, manifest = single_archive(tmp_path, monkeypatch)
    _, runs, ids = analysis.validate_single(directory)
    assert len(ids) == 160 and len(fixtures.records(runs["max1"] / "candidates.jsonl")) == 159
    fixtures.jsonl(runs["max1"] / "candidates.jsonl", [])
    manifest["arm_hashes"]["max1"]["candidates.jsonl"] = digest(runs["max1"] / "candidates.jsonl")
    write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="raw parsing"):
        analysis.validate_single(directory)


@pytest.mark.parametrize("field,value", [("model_name", "/sft/adapter"), ("temperature", 0.8), ("max_candidates", 5)])
def test_single_validator_rejects_wrong_checkpoint_or_protocol(tmp_path, monkeypatch, field, value):
    directory, manifest = single_archive(tmp_path, monkeypatch)
    path = directory / "max1/config.json"
    config = read(path)
    config[field] = value
    write(path, config)
    manifest["arm_hashes"]["max1"]["config.json"] = digest(path)
    write(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="inference settings"):
        analysis.validate_single(directory)


def test_runner_generates_only_one_arm_from_base_and_scores(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    write(submission / "manifest.json", {})
    source = {"provenance": {"base_snapshot": "/cache/base", "model_loader": "fast_model",
                           "chat_template_kwargs": {}, "execution_mode": "eager"},
              "baseline": "/completed/baseline", "baseline_manifest_sha256": "old",
              "input": "data", "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407}
    monkeypatch.setattr(runner, "validate_submission", lambda *a: ({"evaluation": "base"}, source, {}))
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        if "--model-name" not in command:
            assert "analyze_base_single_answer.py" in command[1] and "--allow-smoke" in command
            return
        assert command[command.index("--model-name")+1] == "/cache/base"
        config = read(Path(command[command.index("--config")+1]))
        assert config["max_candidates"] == 1 and config["temperature"] == 0
        parent = Path(command[command.index("--output-parent")+1])
        path = parent / "generated"
        path.mkdir()
        write(path / "status.json", {"status": "complete"})
        write(path / "config.json", config)
        for name in ("examples.jsonl", "candidates.jsonl", "invalid_candidates.jsonl"):
            fixtures.jsonl(path / name, [])
        fixtures.jsonl(path / "generations.jsonl", [{"raw_response": fixtures.response(1)} for _ in range(4)])

    monkeypatch.setattr(runner.subprocess, "run", launch)
    output = tmp_path / "new"
    runner.run(submission, "ministral3", output, True)
    assert len(calls) == 2 and read(output / "manifest.json")["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.run(submission, "ministral3", output, True)


@pytest.mark.parametrize("fail", [False, True])
def test_submission_dependencies_and_fail_closed_preparation(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "bundle/scripts"
    scripts.mkdir(parents=True)
    script = scripts / "submit_base_single_answer.sh"
    script.write_text((ROOT / "gadi_sft_8b_starter/scripts" / script.name).read_text().replace("${USER}", "${TEST_USER}"), newline="\n")
    log, counter = tmp_path / "log.txt", tmp_path / "count.txt"
    counter.write_text("0")
    env = {**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(), "TEST_COUNTER": counter.as_posix(), "TEST_FAIL": str(int(fail))}
    stub = r'''
module() { :; }
source() { :; }
python() {
  if [[ "$TEST_FAIL" == 1 ]]; then return 1; fi
  while (( $# )); do
    if [[ "$1" == --submission-dir ]]; then mkdir -p "$2"; break; fi
    shift
  done
}
qsub() {
  printf '%s\n' "$*" >> "$TEST_LOG"
  value=$(cat "$TEST_COUNTER")
  value=$((value+1))
  printf '%s' "$value" > "$TEST_COUNTER"
  printf '%s.gadi-pbs\n' "$value"
}
task_script="$1"
shift
. "$task_script"
'''
    result = subprocess.run([executable, "-c", stub, "stub", script.as_posix()], env=env, capture_output=True, text=True)
    if fail:
        assert result.returncode and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        lines = log.read_text().splitlines()
        assert len(lines) == 6 and all("evaluate_base_single_answer.pbs" in r for r in lines)
        for i in range(3):
            assert f"depend=afterok:{2*i+1}.gadi-pbs" in lines[2*i+1]
