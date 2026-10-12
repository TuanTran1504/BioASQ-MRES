"""Full input planning and matched tagged-answer comparison invariants."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import run_base_original_prompt as runner
import run_single_answer_sampling as sampling
from prepare_matched_8b_sft import write, digest
from scripts import analyze_base_original_prompt as analysis
from single_answer_format_recovery import recover_single_answer


class Tokenizer:
    def __init__(self, offset=0):
        self.offset = offset

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + messages[1]["content"]

    def __call__(self, *, text, **kwargs):
        return {"input_ids": [ord(c) + self.offset for c in text]}


def source(tmp_path, native=32768):
    write(tmp_path / "config.json", {"text_config": {"max_position_embeddings": native}})
    return {"base_snapshot": str(tmp_path), "adapter": "adapter", "system_prompt": "Answer: [BE]expression[EE]",
            "chat_template_kwargs": {}, "model_loader": "fast_language_model", "execution_mode": "default"}


def example(qid="q", size=9000):
    return {"question_id": qid, "question": "Which?", "gold_aliases": ["15"],
            "snippets": [{"snippet_id": "s", "text": "x" * size}]}


def test_full_inputs_over_8192_are_sized_to_actual_need_and_runtime_checked(tmp_path):
    src = source(tmp_path)
    rows = [example(), example("short", 20)]
    plan = runner.full_input_plan(src, rows, 512, lambda _: Tokenizer())
    assert plan["max_prompt_tokens"] > 8192
    assert plan["max_seq_length"] == plan["max_prompt_tokens"] + 512
    config = {**plan, "max_new_tokens": 512, "mark_snippets": False}
    prepared = sampling.prepare_prompts(Tokenizer(), src["system_prompt"], rows, config)
    assert prepared[0][2] == plan["max_prompt_tokens"]
    with pytest.raises(ValueError, match="input tokens differ"):
        sampling.prepare_prompts(Tokenizer(1), src["system_prompt"], rows, config)


def test_input_plan_rejects_template_or_native_context_mismatch(tmp_path):
    src = source(tmp_path, native=100)
    with pytest.raises(ValueError, match="no truncation"):
        runner.full_input_plan(src, [example()], 512, lambda _: Tokenizer())
    with pytest.raises(ValueError, match="different input tokens"):
        runner.full_input_plan(src, [example()], 512, lambda path: Tokenizer(int(path == "adapter")))


def test_submission_uses_frozen_worker_after_live_code_changes(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    for relative in ("scripts/run_single_answer_sampling.py", "scripts/run_extractive_expansion_8b.py", "src/utility/helper.py"):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# prepared version\n")
    monkeypatch.setattr(runner, "ROOT", bundle)
    submission = tmp_path / "submission"
    snapshot = runner.snapshot_inference_code(submission)
    worker = submission / "inference_code/scripts/run_single_answer_sampling.py"
    config = tmp_path / "config.json"
    write(config, {})
    provenance = {"base_snapshot": "unchanged"}
    prepared = {"ready": True, "config": str(config), "config_sha256": digest(config),
        "code_snapshot": snapshot, "worker_sha256": digest(worker),
        "sources": {"qwen3": {"provenance": provenance}}}
    write(submission / "manifest.json", prepared)
    monkeypatch.setattr(runner, "validate_training_pair", lambda *a: {"original": provenance})
    (bundle / "scripts/run_single_answer_sampling.py").write_text("# later Git pull\n")
    (bundle / "src/utility/helper.py").write_text("# later helper update\n")
    assert runner.validate_submission(submission, "qwen3")[0] == prepared
    assert runner.inference_worker(submission, prepared) == worker
    assert worker.read_text() == "# prepared version\n"
    (submission / "inference_code/src/utility/helper.py").write_text("# altered frozen helper\n")
    with pytest.raises(ValueError, match="Frozen inference code changed"):
        runner.validate_submission(submission, "qwen3")


def test_frozen_code_imports_helpers_from_saved_bundle_in_fresh_process(tmp_path):
    snapshot = runner.snapshot_inference_code(tmp_path)
    worker = tmp_path / "inference_code/scripts/run_single_answer_sampling.py"
    prepared = {"code_snapshot": snapshot, "worker_sha256": digest(worker)}
    runner.inference_worker(tmp_path, prepared)
    code = r'''
import sys
from pathlib import Path
code_root = Path(sys.argv[1])
sys.path.insert(0, str(code_root / "scripts"))
import run_single_answer_sampling as worker
import run_extractive_expansion_8b as generation
from src.utility import training_execution
assert worker.ROOT == code_root
assert generation.ROOT == code_root
assert Path(training_execution.__file__).resolve().is_relative_to(code_root)
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "inference_code")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("raw,expected", [
    ("Answer: 15", "15"),
    ("Answer: Experimental autoimmune encephalomyelitis (EAE)", "Experimental autoimmune encephalomyelitis (EAE)"),
    ("Answer: Autosomal dominant polycystic kidney disease (ADPKD) [EE]", "Autosomal dominant polycystic kidney disease (ADPKD)"),
    ("Answer: galactosylceramide beta-galactosidase", "galactosylceramide beta-galactosidase"),
    ("Answer: [BE]15", "15"),
    ("Answer: [BE]15[EE]", "15"),
    ("Answer: 15. Explanation follows.", "15. Explanation follows."),
    ("15", None),
    ("Answer: 15\nExplanation follows", None),
    ("Answer: [BE]15[EE] [BE]16[EE]", None),
    ("Answer: Answer: 15", None),
    ("Answer: [EE]", None),
])
def test_format_recovery_keeps_whole_expression_and_rejects_ambiguous_structure(raw, expected):
    answer, method = recover_single_answer(raw)
    assert answer == expected
    if expected is None:
        assert method == "unrecoverable"


def test_zero_parse_smoke_is_measured_as_format_failure_not_blocked(tmp_path, monkeypatch):
    src = source(tmp_path)
    examples = [example(str(i), 10) for i in range(4)]
    plan = runner.full_input_plan(src, examples, 512, lambda _: Tokenizer())
    submission = tmp_path / "submission"
    submission.mkdir()
    write(submission / "manifest.json", {})
    evaluation = {"input": "dev.jsonl", "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407, "input_sha256": "data"}
    monkeypatch.setattr(runner, "validate_submission", lambda *a: ({}, evaluation, {"provenance": src, "input_plan": plan}))
    monkeypatch.setattr(runner, "inference_worker", lambda *a: Path("frozen-worker.py"))
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        if "--output-dir" not in command:
            assert "--allow-smoke" in command
            return
        directory = Path(command[command.index("--output-dir")+1])
        directory.mkdir()
        write(directory / "config.json", {})
        write(directory / "status.json", {"status": "complete"})
        json_rows = [{"question_id": str(i), "samples": [{"raw_response": "Answer: 15", "answer": None}]} for i in range(4)]
        (directory / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in json_rows))
        (directory / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
        (directory / "candidates.jsonl").write_text("")
    monkeypatch.setattr(runner.subprocess, "run", launch)
    output = tmp_path / "smoke"
    runner.run(submission, "llama31", output, True)
    state = json.loads((output / "manifest.json").read_text())
    assert state["status"] == "complete" and len(calls) == 3
    assert state["smoke_parse_counts"] == {"base": 0, "sft": 0}
    assert set(state["runs"]) == {"base", "sft"}


def test_recovered_score_is_separate_and_does_not_rewrite_strict_outputs(tmp_path, monkeypatch):
    archive(tmp_path)
    path = tmp_path / "base/generations.jsonl"
    generations = [json.loads(line) for line in path.read_text().splitlines()]
    for row in generations:
        row["samples"][0]["raw_response"] = "Answer: 15"
        row["samples"][0]["answer"] = None
        row["parse_error"] = "failed"
    path.write_text("".join(json.dumps(r) + "\n" for r in generations))
    (tmp_path / "base/candidates.jsonl").write_text("")
    state = json.loads((tmp_path / "manifest.json").read_text())
    for name in ("generations.jsonl", "candidates.jsonl"):
        state["arm_hashes"]["base"][name] = digest(tmp_path / "base" / name)
    write(tmp_path / "manifest.json", state)
    before = {name: digest(tmp_path / "base" / name) for name in ("generations.jsonl", "candidates.jsonl")}
    from src.notebook_workflows import local_expansion
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
        {(r["question_id"], r["answer"]): r["answer"] == "15" for r in candidates})
    result = analysis.analyze(tmp_path, Path("unused.jar"))
    assert result["conditions"]["base"]["metrics"]["mrr_at5"] == 0
    assert result["format_recovered_conditions"]["base"]["metrics"]["mrr_at5"] == 1
    assert result["format_recovered_conditions"]["base"]["format_recovered_count"] == 160
    assert result["paired_contrasts"]["mrr_at5"]["sft_minus_base"] == .75
    assert result["format_recovered_paired_contrasts"]["mrr_at5"]["sft_minus_base"] == -.25
    assert before == {name: digest(tmp_path / "base" / name) for name in before}


def archive(tmp_path):
    src = source(tmp_path)
    examples = [example(str(i), 10) for i in range(160)]
    plan = runner.full_input_plan(src, examples, 512, lambda _: Tokenizer())
    prompt = tmp_path / "original_prompt.txt"
    prompt.write_bytes(src["system_prompt"].encode())
    settings = runner.arm_config({"input": "data.jsonl", "max_seq_length": 6144,
        "max_new_tokens": 512, "seed": 3407}, {"original": src}, "original_greedy", prompt)
    settings.update(max_seq_length=plan["max_seq_length"], expected_prompt_tokens=plan["expected_prompt_tokens"],
                    expected_prompt_ids_sha256=plan["expected_prompt_ids_sha256"], input_token_policy="complete_evidence_auto_size")
    manifest = {"status": "complete", "model_key": "llama31", "smoke_test": False, "expected_questions": 160,
        "settings": settings, "provenance": src, "input_plan": plan, "input_sha256": "data",
        "prompt_sha256": digest(prompt), "runs": {}, "arm_hashes": {}, "evaluation": "exploratory"}
    for name in ("base", "sft"):
        run = tmp_path / name
        run.mkdir()
        config = {**settings, "model_name": src["base_snapshot"] if name == "base" else src["adapter"],
            "question_count": 160, "input_sha256": "data", "prompt_sha256": digest(prompt),
            "parser_version": sampling.PARSER_VERSION, "gold_blind_generation": True, "local_files_only": True}
        write(run / "config.json", config)
        write(run / "status.json", {"status": "complete"})
        generations, candidates = [], []
        for i, row in enumerate(examples):
            # Format failures remain in both arms; SFT recovers some questions.
            raw = "Answer: [BE]15[EE]" if i % 2 == 0 or name == "sft" and i % 4 == 1 else "bad format"
            sample = {"draw": 1, "seed": sampling.sample_seed(str(i), 1), "raw_response": raw,
                "answer": "15" if raw != "bad format" else None, "output_tokens": 5, "generation_seconds": 1}
            generations.append({"question_id": str(i), "samples": [sample], "snippets_truncated": False,
                "request_count": 1, "included_snippets": 1, "total_snippets": 1, "parse_error": None if sample["answer"] else "failed",
                "prompt_tokens": plan["expected_prompt_tokens"][str(i)], "input_tokens": plan["expected_prompt_tokens"][str(i)],
                "output_tokens": 5, "generation_seconds": 1})
            candidates.extend({"question_id": str(i), **r} for r in sampling.unique_candidates([sample]))
        for filename, rows in (("examples.jsonl", examples), ("generations.jsonl", generations), ("candidates.jsonl", candidates)):
            (run / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifest["runs"][name] = name
        manifest["arm_hashes"][name] = {n: digest(run / n) for n in (
            "config.json", "status.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl")}
    write(tmp_path / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize("mutation", ["prompt", "model", "evidence", "seed", "parse", "input_size", "truncation", "missing_question"])
def test_analysis_rejects_unmatched_or_corrupt_runs(tmp_path, mutation):
    manifest = archive(tmp_path)
    analysis.validate_experiment(tmp_path)
    if mutation == "prompt":
        (tmp_path / "original_prompt.txt").write_text("different prompt")
    elif mutation == "model":
        path = tmp_path / "base/config.json"
        data = json.loads(path.read_text())
        data["model_name"] = "wrong backbone"
        write(path, data)
        manifest["arm_hashes"]["base"][path.name] = digest(path)
    else:
        name = "examples.jsonl" if mutation == "evidence" else "generations.jsonl"
        path = tmp_path / "base" / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mutation == "evidence": rows[0]["snippets"][0]["text"] = "different"
        elif mutation == "seed": rows[0]["samples"][0]["seed"] = 0
        elif mutation == "parse": rows[0]["samples"][0]["answer"] = "fake"
        elif mutation == "input_size": rows[0]["prompt_tokens"] -= 1
        elif mutation == "truncation": rows[0]["snippets_truncated"] = True
        elif mutation == "missing_question": rows.pop()
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest["arm_hashes"]["base"][name] = digest(path)
    write(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        analysis.validate_experiment(tmp_path)


def test_scoring_preserves_failure_denominators_and_reports_correct_direction(tmp_path, monkeypatch):
    archive(tmp_path)
    from src.notebook_workflows import local_expansion
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
        {(r["question_id"], r["answer"]): r["answer"] == "15" for r in candidates})
    result = analysis.analyze(tmp_path, Path("unused.jar"))
    assert result["conditions"]["base"]["metrics"]["mrr_at5"] == .5
    assert result["conditions"]["sft"]["metrics"]["mrr_at5"] == .75
    assert result["paired_contrasts"]["mrr_at5"]["sft_minus_base"] == .25
    assert result["coverage_transitions"]["base_covered"] == 80
    assert result["coverage_transitions"]["sft_covered"] == 120
    assert result["coverage_transitions"]["gained"] == 40
    assert result["coverage_transitions"]["lost"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_submission_all_models_smoke_dependencies_and_preflight_before_qsub(tmp_path, fail):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    text = (ROOT / "gadi_sft_8b_starter/scripts/submit_base_original_prompt.sh").read_text().replace("${USER}", "${TEST_USER}")
    stub = r'''
module() { :; }
source() { :; }
python() {
  printf 'preflight\n' >> "$TEST_LOG"
  if [[ "$TEST_FAIL" == 1 ]]; then return 1; fi
  while (( $# )); do
    if [[ "$1" == --submission-dir ]]; then mkdir -p "$2"; break; fi
    shift
  done
}
qsub() {
  printf 'qsub %s\n' "$*" >> "$TEST_LOG"
  n=$(cat "$TEST_COUNT")
  printf '%s' "$((n+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\n' "$n"
}
'''
    script = scripts / "submit.sh"
    script.write_text(text.replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log, count = tmp_path / "log", tmp_path / "count"
    count.write_text("1")
    completed = subprocess.run([bash, script.as_posix()], text=True, capture_output=True,
        env={**os.environ, "TEST_USER": "test", "TEST_LOG": log.as_posix(), "TEST_COUNT": count.as_posix(), "TEST_FAIL": str(int(fail))})
    assert log.exists(), completed.stderr
    lines = log.read_text().splitlines()
    assert lines[0] == "preflight"
    jobs = lines[1:]
    if fail:
        assert completed.returncode and not jobs
    else:
        assert completed.returncode == 0, completed.stderr
        assert len(jobs) == 6
        for i, model in enumerate(("llama31", "qwen3", "ministral3")):
            assert f"MODEL_KEY={model}" in jobs[2*i]
            assert "SMOKE_TEST=1" in jobs[2*i]
            assert f"depend=afterok:{2*i+1}.gadi-pbs" in jobs[2*i+1]
