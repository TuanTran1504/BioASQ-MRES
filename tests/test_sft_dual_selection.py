"""Test independent checkpoint selection and validation separation without GPUs."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import prepare_sft_dual_selection as prepare
import select_sft_epoch_checkpoints as select
import run_single_answer_sampling as sampling


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def epoch(root, number, loss):
    checkpoint = root / "trainer_output" / f"checkpoint-{number * 81}"
    write(checkpoint / "trainer_state.json", {"epoch": float(number), "global_step": number * 81,
        "log_history": [{"epoch": float(number), "step": number * 81, "eval_loss": loss}]})
    write(checkpoint / "adapter_config.json", {"base_model_name_or_path": "pinned-base"})
    (checkpoint / "adapter_model.safetensors").write_bytes(f"epoch-{number}".encode())
    return checkpoint


def test_loss_and_coverage_choose_different_epochs():
    rows = [{"epoch": 1, "eval_loss": 0.4, "metrics": {"coverage_full_pool": 0.5, "coverage_at10": 0.5}},
            {"epoch": 2, "eval_loss": 0.3, "metrics": {"coverage_full_pool": 0.6, "coverage_at10": 0.55}},
            {"epoch": 3, "eval_loss": 0.2, "metrics": {"coverage_full_pool": 0.55, "coverage_at10": 0.5}}]
    winners = select.choose_winners(rows)
    assert winners["best_coverage"]["epoch"] == 2
    assert winners["best_eval_loss"]["epoch"] == 3
    rows[2]["metrics"] = {"coverage_full_pool": 0.6, "coverage_at10": 0.6}
    assert select.choose_winners(rows)["best_coverage"]["epoch"] == 3
    rows[2]["metrics"] = dict(rows[1]["metrics"])
    assert select.choose_winners(rows)["best_coverage"]["epoch"] == 2


def test_epoch_selection_requires_all_epochs_and_own_loss(tmp_path):
    epoch(tmp_path, 1, 0.3)
    last = epoch(tmp_path, 3, 0.2)
    with pytest.raises(ValueError, match="loss-pruned"):
        select.epoch_checkpoints(tmp_path, 3)
    epoch(tmp_path, 2, 0.25)
    rows = select.epoch_checkpoints(tmp_path, 3)
    assert [r["eval_loss"] for r in rows] == [0.3, 0.25, 0.2]
    state = json.loads((last / "trainer_state.json").read_text())
    state["log_history"][0]["step"] = 81
    write(last / "trainer_state.json", state)
    with pytest.raises(ValueError, match="own epoch"):
        select.epoch_checkpoints(tmp_path, 3)


def test_staging_preserves_checkpoint_weights_and_native_tokenizer(tmp_path):
    checkpoint = epoch(tmp_path, 1, 0.3)
    source = tmp_path / "adapter"
    write(source / "adapter_config.json", {})
    (source / "adapter_model.safetensors").write_bytes(b"loss-winner-not-this-epoch")
    write(source / "tokenizer_config.json", {"chat_template": "native"})
    write(source / "training_complete.json", {"old_loss_selection": True})
    destination = tmp_path / "staged"
    select.stage_adapter(checkpoint, source, destination)
    assert (destination / "adapter_model.safetensors").read_bytes() == b"epoch-1"
    assert json.loads((destination / "tokenizer_config.json").read_text())["chat_template"] == "native"
    assert not (destination / "training_complete.json").exists()
    with pytest.raises(FileExistsError):
        select.stage_adapter(checkpoint, source, destination)


def validation_row(qid="v"):
    return {"question_id": qid, "messages": [
        {"role": "system", "content": "Expand"},
        {"role": "user", "content": json.dumps({"question": "Which?", "snippets": [{"id": "s1", "text": "evidence"}]})},
        {"role": "assistant", "content": json.dumps({"answers": [{"answer": "teacher variant"}]})}]}


def test_validation_gold_is_official_aliases_not_teacher_variants():
    examples = prepare.validation_examples([validation_row()], [{"question_id": "v", "internal_split": "validation",
                                                               "gold_aliases": ["official answer"]}])
    assert examples[0]["gold_aliases"] == ["official answer"]
    assert "teacher variant" not in json.dumps(examples)
    assert examples[0]["snippets"] == [{"snippet_id": "s1", "text": "evidence"}]


@pytest.mark.parametrize("failure", ["fitting", "missing", "duplicate", "gold_in_prompt"])
def test_validation_panel_rejects_bad_partition_or_prompt(failure):
    rows = [validation_row()]
    audit = [{"question_id": "v", "internal_split": "validation", "gold_aliases": ["answer"]}]
    if failure == "fitting":
        audit[0]["internal_split"] = "train"
    elif failure == "missing":
        audit[0]["question_id"] = "other"
    elif failure == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        user = json.loads(rows[0]["messages"][1]["content"])
        user["gold_aliases"] = ["answer"]
        rows[0]["messages"][1]["content"] = json.dumps(user)
    with pytest.raises(ValueError):
        prepare.validation_examples(rows, audit)


@pytest.mark.parametrize("module_path", ["src/utility/training.py", "gadi_sft_8b_starter/src/utility/training.py"])
def test_retention_reaches_actual_trainer_arguments(module_path):
    # Run the actual function with a fake training config; no Unsloth import.
    tree = ast.parse((ROOT / module_path).read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_training_arguments")

    class Config:
        def __init__(self, evaluation_strategy=None, **kwargs):
            self.kwargs = kwargs

    import argparse
    environment = {"argparse": argparse, "Any": Any, "inspect": inspect,
        "resolve_selection_metric": lambda a: "eval_loss", "uses_generated_selection": lambda a: False,
        "resolve_save_strategy": lambda a: "epoch", "resolve_eval_strategy": lambda *a, **k: "epoch",
        "is_bfloat16_supported": lambda: False, "SFTConfig": Config, "TrainingArguments": Config}
    exec(compile(ast.Module(body=[function], type_ignores=[]), module_path, "exec"), environment)
    args = SimpleNamespace(per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, warmup_steps=10, num_train_epochs=3,
        learning_rate=0.0002, logging_steps=1, weight_decay=0.01, seed=3407, output_dir="new")
    assert environment["build_training_arguments"](args, True).kwargs["save_total_limit"] == 1
    args.save_total_limit = 3
    result = environment["build_training_arguments"](args, True).kwargs
    assert result["save_total_limit"] == 3
    assert result["metric_for_best_model"] == "eval_loss"
    assert result["greater_is_better"] is False


def test_selection_exports_two_winners_without_retraining(tmp_path, monkeypatch):
    directory = tmp_path / "training"
    for n, loss in ((1, 0.3), (2, 0.25), (3, 0.2)):
        epoch(directory, n, loss)
    write(directory / "adapter/tokenizer_config.json", {})
    write(directory / "status.json", {})
    submission = tmp_path / "submission"
    write(submission / "manifest.json", {})
    prepared = {"configuration": {"selection": {"num_generations": 10}}}
    saved = {"model_loader": "fast_language_model", "chat_template_kwargs": {},
             "base_snapshot": "base", "base_revision": "revision"}
    monkeypatch.setattr(select, "validate_run", lambda *a: (prepared, saved, [{"question_id": "v"}]))
    launched = []

    def launch(command, **kwargs):
        launched.append(command)
        run = Path(command[command.index("--output-dir") + 1])
        for name in ("config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "selection_scores.json"):
            write(run / name, {})

    def score(run, qids):
        assert qids == ["v"]
        number = int(run.parent.name.split("-")[1])
        return {"coverage_full_pool": {1: 0.5, 2: 0.75, 3: 0.6}[number],
                "coverage_at10": 0.5, "parse_success_rate": 1}, run / "analysis-official"

    monkeypatch.setattr(select.subprocess, "run", launch)
    monkeypatch.setattr(select, "score_generation", score)
    output = tmp_path / "selection"
    select.run_selection(directory, submission, output, "qwen3", False)
    state = json.loads((output / "manifest.json").read_text())
    assert state["status"] == "complete"
    assert len(launched) == 3 and all("--preflight-all" in cmd for cmd in launched)
    assert state["selected_adapters"]["best_coverage"]["epoch"] == 2
    assert state["selected_adapters"]["best_eval_loss"]["epoch"] == 3
    assert (output / "selected_adapters/best_coverage/adapter_model.safetensors").read_bytes() == b"epoch-2"
    assert (output / "selected_adapters/best_eval_loss/adapter_model.safetensors").read_bytes() == b"epoch-3"
    assert (directory / "trainer_output/checkpoint-81/adapter_model.safetensors").read_bytes() == b"epoch-1"
    with pytest.raises(FileExistsError):
        select.run_selection(directory, submission, output, "qwen3", False)


@pytest.mark.parametrize("fail_preflight,reuse,stored_path", [(False, False, False), (True, False, False),
                                                          (False, True, False), (False, True, True)])
def test_submission_dependencies_and_preflight_fail_closed(tmp_path, fail_preflight, reuse, stored_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    bundle = tmp_path / "bundle"
    scripts = bundle / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "submit_sft_dual_selection_8b.sh"
    text = (ROOT / "gadi_sft_8b_starter/scripts" / script.name).read_text()
    script.write_text(text.replace("${USER}", "${TEST_USER}"), newline="\n")
    log = tmp_path / "qsub_calls.txt"
    counter = tmp_path / "counter.txt"
    counter.write_text("0")
    environment = {**os.environ, "TEST_LOG": log.as_posix(), "TEST_COUNTER": counter.as_posix(),
                   "FAIL_PREFLIGHT": "1" if fail_preflight else "0", "TEST_USER": "scheduler-test"}
    arguments = []
    if reuse:
        old_record = tmp_path / "old-jobs.tsv"
        old_record.write_text("model\tstage\tjob_id\tdepends_on\ttraining_run\n" + "".join(
            f"{model}\tsmoke\t{job}.gadi-pbs\t\t" + (f"/saved/{model}-original-smoke" if stored_path else "") + "\n"
            for model, job in (
                ("llama31", 180987381), ("qwen3", 180987384), ("ministral3", 180987387))))
        arguments = ["--reuse-smoke-record", old_record.as_posix()]
    stub = r'''
module() { :; }
source() { :; }
python() {
  if [[ "$1" == scripts/prepare_sft_dual_selection.py ]]; then
    if [[ "$FAIL_PREFLIGHT" == 1 ]]; then return 1; fi
    shift
    while (( $# )); do
      if [[ "$1" == --submission-dir ]]; then mkdir -p "$2"; break; fi
      shift
    done
  fi
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
    process = subprocess.run([executable, "-c", stub, "stub", script.as_posix(), *arguments],
                             env=environment, text=True, capture_output=True)
    if fail_preflight:
        assert process.returncode != 0
        assert not log.exists()
    else:
        assert process.returncode == 0, process.stderr
        lines = log.read_text().splitlines()
        assert len(lines) == 9
        for i, model in enumerate(("llama31", "qwen3", "ministral3")):
            assert model in lines[3 * i]
            assert f"depend=afterok:{3 * i + 1}.gadi-pbs" in lines[3 * i + 1]
            assert f"depend=afterok:{3 * i + 2}.gadi-pbs" in lines[3 * i + 2]
            assert ("REUSE_TRAINING_RUN=" in lines[3 * i]) == reuse
            if stored_path:
                assert f"REUSE_TRAINING_RUN=/saved/{model}-original-smoke" in lines[3 * i]
            assert "REUSE_TRAINING_RUN=" not in lines[3 * i + 1]
        jobs = next(bundle.glob("outputs/sft_dual_selection_8b/submission-*/jobs.tsv"))
        assert len(jobs.read_text().splitlines()) == 10


def test_prompt_preflight_catches_overflow_outside_smoke_subset(monkeypatch):
    examples = [{"question_id": str(i), "question": "short"} for i in range(4)]
    examples.append({"question_id": "long", "question": "x" * 100})
    monkeypatch.setattr(sampling, "render_prompt", lambda t, p, e, k: e["question"])
    monkeypatch.setattr(sampling, "tokenize_text", lambda t, text, **k: len(text))
    monkeypatch.setattr(sampling, "input_token_count", lambda value: value)
    config = {"max_seq_length": 80, "max_new_tokens": 10, "mark_snippets": False}
    assert len(sampling.prepare_prompts(None, "prompt", examples[:4], config)) == 4
    with pytest.raises(ValueError, match="long: all snippets exceed"):
        sampling.prepare_prompts(None, "prompt", examples, config)


def test_validation_budget_handles_prompt_that_exceeded_old_dev_limit(monkeypatch):
    config = json.loads((ROOT / "gadi_sft_8b_starter/configs/sft_dual_selection_8b.json").read_text())["selection"]
    assert config["max_seq_length"] == 8192 and config["max_new_tokens"] == 512
    examples = [{"question_id": "long-validation", "question": "x" * 6627}]
    monkeypatch.setattr(sampling, "render_prompt", lambda t, p, e, k: e["question"])
    monkeypatch.setattr(sampling, "tokenize_text", lambda t, text, **k: len(text))
    monkeypatch.setattr(sampling, "input_token_count", lambda value: value)
    with pytest.raises(ValueError):
        sampling.prepare_prompts(None, "prompt", examples, {**config, "max_seq_length": 6144})
    prepared = sampling.prepare_prompts(None, "prompt", examples, config)
    assert prepared[0][2] == 6627
    with pytest.raises(ValueError):
        sampling.prepare_prompts(None, "prompt", [{**examples[0], "question": "x" * 7681}], config)


def test_smoke_selects_longest_prompts_after_complete_panel_preflight():
    rows = [({"question_id": str(i)}, "text", count) for i, count in enumerate((100, 300, 200, 600, 500, 400))]
    chosen = sampling.choose_prepared(rows, 4, True)
    assert [r[2] for r in chosen] == [600, 500, 400, 300]
    assert sampling.choose_prepared(rows, 4) == rows[:4]


def test_selection_smoke_accepts_longest_subset_and_preserves_panel(tmp_path, monkeypatch):
    directory = tmp_path / "training"
    epoch(directory, 1, 0.3)
    write(directory / "adapter/tokenizer_config.json", {})
    write(directory / "status.json", {})
    submission = tmp_path / "submission"
    write(submission / "manifest.json", {})
    examples = [{"question_id": str(i), "question": f"question {i}"} for i in range(8)]
    selected = examples[4:]
    prepared = {"configuration": {"selection": {"num_generations": 10, "max_seq_length": 8192}}}
    saved = {"model_loader": "fast_language_model", "chat_template_kwargs": {},
             "base_snapshot": "base", "base_revision": "revision"}
    monkeypatch.setattr(select, "validate_run", lambda *a: (prepared, saved, examples))

    def launch(command, **kwargs):
        assert "--longest-first" in command and "--preflight-all" in command
        output = Path(command[command.index("--output-dir") + 1])
        for name in ("config.json", "generations.jsonl", "candidates.jsonl", "selection_scores.json"):
            write(output / name, {})
        (output / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))

    def score(path, ids):
        assert ids == ["4", "5", "6", "7"]
        return {"coverage_full_pool": 0.5, "coverage_at10": 0.5, "parse_success_rate": 1}, path / "official"

    monkeypatch.setattr(select.subprocess, "run", launch)
    monkeypatch.setattr(select, "score_generation", score)
    output = tmp_path / "retry"
    select.run_selection(directory, submission, output, "qwen3", True)
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    assert len(examples) == 8
