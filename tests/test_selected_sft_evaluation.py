"""Dev separation, frozen selected weights, pool replay and dependent submissions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import run_selected_sft_evaluation as runner
import run_single_answer_sampling as sampling
from prepare_matched_8b_sft import write, digest, read
from scripts import analyze_selected_sft_evaluation as analysis


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_protocol_rejects_dev_overlap_and_changed_prepared_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    submission = tmp_path / "submission"
    dataset = tmp_path / "dataset/expansion"
    dev = tmp_path / "dev.jsonl"
    jsonl(dev, [{"question_id": str(i)} for i in range(160)])
    for split, qid in (("train", "fit"), ("validation", "val")):
        jsonl(dataset / f"{split}.jsonl", [{"question_id": qid}])
    save(dataset / "manifest.json", {"outer_dev_sha256": digest(dev), "output_sha256": {
        f"{split}.jsonl": digest(dataset / f"{split}.jsonl") for split in ("train", "validation")}})
    save(submission / "training_matrix.json", {"output_dataset": "dataset", "outer_dev": "dev.jsonl", "expansion_candidate_limit": 5})
    save(submission / "base_pins.json", {})
    (submission / "expansion_prompt.txt").write_text("Up to FIVE")
    jsonl(submission / "validation_examples.jsonl", [{"question_id": "val"}])
    prepared = {"status": "ready", "models": ["qwen3"], "configuration": {"selection": {"max_candidates": 5}},
        "training_matrix_sha256": digest(submission / "training_matrix.json"),
        "base_pins_sha256": digest(submission / "base_pins.json"), "expansion_prompt_sha256": digest(submission / "expansion_prompt.txt"),
        "validation_examples_sha256": digest(submission / "validation_examples.jsonl")}
    save(submission / "manifest.json", prepared)
    runner.validate_protocol(submission, ["qwen3"])
    jsonl(dataset / "train.jsonl", [{"question_id": "0"}])
    meta = read(dataset / "manifest.json")
    meta["output_sha256"]["train.jsonl"] = digest(dataset / "train.jsonl")
    save(dataset / "manifest.json", meta)
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_protocol(submission, ["qwen3"])


@pytest.mark.parametrize("failure", [None, "smoke", "weights", "winner"])
def test_selected_checkpoint_must_be_frozen_validation_winner(tmp_path, monkeypatch, failure):
    training, selection, submission = [tmp_path / name for name in ("training", "selection", "submission")]
    saved = {"base_snapshot": "base", "base_revision": "revision"}
    prepared = {"configuration": {"selection": {"max_candidates": 5}}}
    save(training / "status.json", {"status": "completed"})
    save(submission / "manifest.json", prepared)
    monkeypatch.setattr(runner, "validate_protocol", lambda *a: (prepared, {"expansion_candidate_limit": 5}, [], Path("dev")))
    monkeypatch.setattr(runner, "validate_run", lambda *a: (prepared, saved, []))
    scores = []
    for epoch, loss, coverage in ((1, .4, .7), (2, .3, .6), (3, .5, .5)):
        path = training / f"trainer_output/checkpoint-{epoch*81}"
        save(path / "trainer_state.json", {"epoch": epoch, "global_step": epoch*81,
            "log_history": [{"epoch": epoch, "step": epoch*81, "eval_loss": loss}]})
        save(path / "adapter_config.json", {"base_model_name_or_path": "base"})
        (path / "adapter_model.safetensors").write_bytes(f"epoch-{epoch}".encode())
        scores.append({"epoch": epoch, "eval_loss": loss, "metrics": {"coverage_full_pool": coverage, "coverage_at10": coverage}})
    selected = {}
    for criterion, epoch in (("best_eval_loss", 2), ("best_coverage", 1)):
        path = selection / "selected_adapters" / criterion
        save(path / "adapter_config.json", {"base_model_name_or_path": "base"})
        (path / "adapter_model.safetensors").write_bytes(f"epoch-{epoch}".encode())
        save(path / "training_complete.json", {"status": "completed", "model_key": "qwen3", "smoke_test": False, "selection_name": criterion,
                                               "training_configuration": saved})
        selected[criterion] = {"path": str(path), "epoch": epoch,
                              "files_sha256": {p.name: digest(p) for p in path.iterdir()}}
    state = {"status": "complete", "smoke_test": failure == "smoke", "model_key": "qwen3", "validation_questions": 144,
        "training_run": str(training), "training_status_sha256": digest(training / "status.json"),
        "submission_manifest_sha256": digest(submission / "manifest.json"), "selection": prepared["configuration"]["selection"],
        "selected_adapters": selected, "checkpoint_scores": scores}
    if failure == "weights":
        path = Path(selected["best_coverage"]["path"]) / "adapter_model.safetensors"
        path.write_bytes(b"wrong epoch weights")
        selected["best_coverage"]["files_sha256"][path.name] = digest(path)
    elif failure == "winner":
        selected["best_coverage"]["epoch"] = 2
    save(selection / "manifest.json", state)
    if failure:
        with pytest.raises(ValueError): runner.validate_selection(training, selection, submission, "qwen3")
    else:
        _, sources, _, _, limit = runner.validate_selection(training, selection, submission, "qwen3")
        assert sources["best_coverage"]["epoch"] == 1 and sources["best_eval_loss"]["epoch"] == 2
        assert limit == 5


def archive(directory, same=False):
    prompt = directory / "expansion_prompt.txt"
    prompt.write_text("up to FIVE")
    common = {"input": "dev.jsonl", "prompt": str(prompt), "max_candidates": 5,
        "max_seq_length": 522, "max_new_tokens": 512, "seed": 3407, "mark_snippets": False,
        "require_all_snippets": True, "model_loader": "fast_language_model", "chat_template_kwargs": {},
        "execution_mode": "default", "prompt_version": "selected-expansion-sft-dev160-v1"}
    state = {"status": "complete", "model_key": "qwen3", "expected_questions": 160,
        "input_sha256": "panel", "prompt_sha256": digest(prompt), "common_settings": common,
        "input_plan": {"expected_prompt_tokens": {str(i): 10 for i in range(160)}},
        "sources": {name: {"adapter": f"adapter-{name}", "epoch": 1 if same else (1 if name == "best_coverage" else 2),
            "generation_files_sha256": {"weights": "same" if same else name}} for name in runner.CRITERIA},
        "runs": {}, "reused_from": {}, "arm_hashes": {}, "evaluation": "exploratory"}
    examples = [{"question_id": str(i), "question": "Which?", "gold_aliases": ["gold"],
        "snippets": [{"snippet_id": "s", "text": "gold evidence"}]} for i in range(160)]
    for criterion in runner.CRITERIA:
        for mode in runner.MODES:
            name = f"{criterion}_{mode}"
            if same and criterion == "best_coverage":
                original = f"best_eval_loss_{mode}"
                state["runs"][name] = state["runs"][original]
                state["reused_from"][name] = original
                state["arm_hashes"][name] = state["arm_hashes"][original]
                continue
            path = directory / name
            path.mkdir()
            count = 1 if mode == "greedy" else 10
            config = {**common, "model_name": state["sources"][criterion]["adapter"],
                "response_mode": "equivalent_greedy" if count == 1 else "equivalent_sampling",
                "temperature": 0 if count == 1 else .8, "num_generations": count, "question_count": 160,
                "input_sha256": "panel", "prompt_sha256": digest(prompt), "gold_blind_generation": True, "local_files_only": True}
            if count == 10:
                config.update(top_p=.95, top_k=0)
            save(path / "config.json", config)
            save(path / "status.json", {"status": "complete"})
            jsonl(path / "examples.jsonl", examples)
            generations, candidates = [], []
            for i in range(160):
                success = i % 2 == 0 or criterion == "best_coverage" and i % 4 == 1
                samples = []
                for draw in range(1, count+1):
                    raw = json.dumps({"answers": [{"answer": "gold", "relation_type": "original"}]}) if success else "invalid"
                    answers = [{"answer": "gold", "relation_type": "original", "raw_position": 1}] if success else []
                    samples.append({"draw": draw, "seed": sampling.sample_seed(str(i), draw), "raw_response": raw,
                        "answers": answers, "parse_error": None if success else "failed", "generation_seconds": 1, "output_tokens": 10})
                generations.append({"question_id": str(i), "samples": samples, "request_count": count,
                    "snippets_truncated": False, "included_snippets": 1, "total_snippets": 1, "prompt_tokens": 10,
                    "parse_error": None if success else "failed", "generation_seconds": count, "input_tokens": count*10, "output_tokens": count*10})
                candidates.extend({"question_id": str(i), **r} for r in sampling.expansion_candidates(samples))
            jsonl(path / "generations.jsonl", generations)
            jsonl(path / "candidates.jsonl", candidates)
            jsonl(path / "invalid_candidates.jsonl", [])
            state["runs"][name] = name
            state["arm_hashes"][name] = {n: digest(path / n) for n in (
                "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
    save(directory / "manifest.json", state)
    return state


@pytest.mark.parametrize("mutation", [None, "seed", "pool", "truncation", "budget", "prompt", "incomplete"])
def test_analysis_preserves_full_panel_and_rejects_protocol_changes(tmp_path, mutation):
    state = archive(tmp_path)
    name = "best_eval_loss_sampling10"
    path = tmp_path / name
    if mutation == "prompt":
        (tmp_path / "expansion_prompt.txt").write_text("different")
    elif mutation:
        filename = "config.json" if mutation == "budget" else "candidates.jsonl" if mutation == "pool" else "generations.jsonl"
        if filename == "config.json":
            data = read(path / filename)
            data["max_candidates"] = 10
            save(path / filename, data)
        else:
            rows = [json.loads(line) for line in (path / filename).read_text().splitlines()]
            if mutation == "pool": rows[0]["answer"] = "invented"
            elif mutation == "seed": rows[0]["samples"][0]["seed"] = 0
            elif mutation == "truncation": rows[0]["snippets_truncated"] = True
            elif mutation == "incomplete": rows.pop()
            jsonl(path / filename, rows)
        state["arm_hashes"][name][filename] = digest(path / filename)
        save(tmp_path / "manifest.json", state)
    if mutation:
        with pytest.raises(ValueError): analysis.validate_experiment(tmp_path)
    else:
        analysis.validate_experiment(tmp_path)


@pytest.mark.parametrize("same", [False, True])
def test_scoring_both_modes_and_same_checkpoint_reuse(tmp_path, monkeypatch, same):
    archive(tmp_path, same)
    from src.notebook_workflows import local_expansion
    calls = []
    def matches(examples, candidates, *args, **kwargs):
        calls.append(1)
        return {(r["question_id"], r["answer"]): r["answer"] == "gold" for r in candidates}
    monkeypatch.setattr(local_expansion, "official_candidate_matches", matches)
    result = analysis.analyze(tmp_path, Path("unused.jar"))
    assert len(calls) == (2 if same else 4)
    assert result["conditions"]["best_eval_loss_greedy"]["metrics"]["mrr_at5"] == .5
    assert result["conditions"]["best_coverage_sampling10"]["metrics"]["mrr_at5"] == (.5 if same else .75)
    assert result["paired_contrasts"]["coverage_selected_minus_loss_selected_sampling10"]["mrr_at5"]["coverage_selected_minus_loss_selected"] == (0 if same else .25)
    if same:
        state = read(tmp_path / "manifest.json")
        state["sources"]["best_coverage"]["generation_files_sha256"] = {"weights": "different"}
        save(tmp_path / "manifest.json", state)
        with pytest.raises(ValueError, match="identical checkpoint"):
            analysis.validate_experiment(tmp_path)


def test_attachment_uses_selection_dependencies_and_skips_existing_jobs(tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash: pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    text = (ROOT / "gadi_sft_8b_starter/scripts/submit_sft_selected_evaluation.sh").read_text().replace("${USER}", "${TEST_USER}")
    stub = r'''
module() { :; }
source() { :; }
python() { :; }
qsub() {
  printf '%s\n' "$*" >> "$TEST_LOG"
  n=$(cat "$TEST_COUNT")
  printf '%s' "$((n+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\n' "$n"
}
'''
    script = scripts / "attach.sh"
    script.write_text(text.replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    record = tmp_path / "jobs.tsv"
    record.write_text("model\tstage\tjob_id\tdepends_on\ttraining_run\n" + "".join(
        f"{model}\tselect\t{100+i}.gadi-pbs\t99.gadi-pbs\t/training/{model}\n" for i, model in enumerate(("llama31", "qwen3", "ministral3"))))
    count, log = tmp_path / "count", tmp_path / "log"
    count.write_text("200")
    env = {**os.environ, "TEST_USER": "test", "TEST_COUNT": count.as_posix(), "TEST_LOG": log.as_posix()}
    for attempt in range(2):
        result = subprocess.run([bash, script.as_posix(), record.as_posix()], env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    for i, call in enumerate(calls):
        assert f"depend=afterok:{100+i}.gadi-pbs" in call
        assert f"selection-{100+i}.gadi-pbs" in call
    assert len(record.read_text().splitlines()) == 7
