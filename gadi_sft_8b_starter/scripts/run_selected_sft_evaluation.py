#!/usr/bin/env python3
"""Evaluate validation-selected SFT checkpoints on disjoint dev160 inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, read, records, write, digest
from select_sft_epoch_checkpoints import validate_run, choose_winners, epoch_checkpoints
from run_base_original_prompt import full_input_plan

CRITERIA = ("best_eval_loss", "best_coverage")
MODES = ("greedy", "sampling10")


def validate_protocol(submission, models):
    prepared = read(submission / "manifest.json")
    if prepared["status"] != "ready" or not models or not set(models) <= set(prepared["models"]):
        raise ValueError("Submission not prepared for these models")
    matrix_path = submission / "training_matrix.json"
    if digest(matrix_path) != prepared["training_matrix_sha256"]:
        raise ValueError("Prepared matrix changed")
    matrix = read(matrix_path)
    for name, field in (("base_pins.json", "base_pins_sha256"),
                        ("expansion_prompt.txt", "expansion_prompt_sha256"),
                        ("validation_examples.jsonl", "validation_examples_sha256")):
        if digest(submission / name) != prepared[field]:
            raise ValueError("Prepared inputs or pins changed")
    limit = matrix.get("expansion_candidate_limit", 10)
    if limit != prepared["configuration"]["selection"].get("max_candidates", 10):
        raise ValueError("Training and evaluation candidate budgets differ")
    dataset = ROOT / matrix["output_dataset"] / "expansion"
    development = ROOT / matrix["outer_dev"]
    examples = records(development)
    ids = {r["question_id"] for r in examples}
    manifest = read(dataset / "manifest.json")
    if (len(examples) != 160 or len(ids) != 160 or digest(development) != manifest["outer_dev_sha256"]):
        raise ValueError("Expected the pinned dev160 panel")
    for split in ("train", "validation"):
        path = dataset / f"{split}.jsonl"
        if digest(path) != manifest["output_sha256"][path.name]:
            raise ValueError("Fitting/validation data changed")
        rows = records(path)
        if ids & {r["question_id"] for r in rows}:
            raise ValueError("Development questions overlap fitting or validation")
    if ids & {r["question_id"] for r in records(submission / "validation_examples.jsonl")}:
        raise ValueError("Development questions were used in checkpoint selection")
    return prepared, matrix, examples, development


def validate_selection(training, selection, submission, model):
    prepared, matrix, examples, development = validate_protocol(submission, [model])
    _, saved, _ = validate_run(training, submission, model, False)
    selected = read(selection / "manifest.json")
    if (selected["status"] != "complete" or selected["smoke_test"] or selected["model_key"] != model
            or selected["validation_questions"] != 144 or Path(selected["training_run"]).resolve() != training.resolve()
            or selected["training_status_sha256"] != digest(training / "status.json")
            or selected["submission_manifest_sha256"] != digest(submission / "manifest.json")
            or selected["selection"] != prepared["configuration"]["selection"]
            or set(selected["selected_adapters"]) != set(CRITERIA)):
        raise ValueError("Expected completed full internal-validation selection for this training run")
    epochs = {row["epoch"]: row for row in epoch_checkpoints(training, 3)}
    scores = selected["checkpoint_scores"]
    if {r["epoch"] for r in scores} != {1, 2, 3} or len(scores) != 3:
        raise ValueError("All three epochs must have been selected on internal validation")
    if any(row["eval_loss"] != epochs[row["epoch"]]["eval_loss"] for row in scores):
        raise ValueError("Selection losses differ from saved training epoch losses")
    sources = {}
    for criterion, winner in choose_winners(scores).items():
        recorded = selected["selected_adapters"][criterion]
        adapter = Path(recorded["path"]).resolve()
        if not adapter.is_relative_to(selection.resolve()) or recorded["epoch"] != winner["epoch"]:
            raise ValueError("Selected adapter differs from validation winner")
        for filename, expected in recorded["files_sha256"].items():
            if digest(adapter / filename) != expected:
                raise ValueError("Selected adapter changed")
        for filename, expected in epochs[winner["epoch"]]["adapter_sha256"].items():
            if digest(adapter / filename) != expected:
                raise ValueError("Selected weights differ from saved training epoch")
        if read(adapter / "adapter_config.json")["base_model_name_or_path"] != saved["base_snapshot"]:
            raise ValueError("Selected adapter uses a different backbone")
        metadata = read(adapter / "training_complete.json")
        if (metadata["status"] != "completed" or metadata["model_key"] != model
                or metadata["smoke_test"] or metadata["selection_name"] != criterion
                or metadata["training_configuration"] != saved):
            raise ValueError("Selected checkpoint metadata differ")
        sources[criterion] = {"adapter": str(adapter), "epoch": winner["epoch"],
            "files_sha256": recorded["files_sha256"],
            "generation_files_sha256": {k: v for k, v in recorded["files_sha256"].items() if k != "training_complete.json"}}
    return saved, sources, examples, development, matrix.get("expansion_candidate_limit", 10)


def run(training, selection, submission, output, model):
    saved, sources, examples, development, limit = validate_selection(training, selection, submission, model)
    prompt = (submission / "expansion_prompt.txt").read_text(encoding="utf-8").strip()
    # Allocation follows actual complete inputs; no artificial input cap.
    source = {**saved, "system_prompt": prompt, "adapter": sources["best_eval_loss"]["adapter"]}
    plan = full_input_plan(source, examples, 512)
    output.mkdir(parents=True, exist_ok=False)
    prompt_path = output / "expansion_prompt.txt"
    prompt_path.write_bytes(prompt.encode("utf-8"))
    common = {"input": str(development), "prompt": str(prompt_path),
        "max_candidates": limit, "max_seq_length": plan["max_seq_length"], "max_new_tokens": 512,
        "expected_prompt_tokens": plan["expected_prompt_tokens"], "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
        "input_token_policy": "complete_evidence_auto_size",
        "model_loader": saved["model_loader"], "chat_template_kwargs": saved["chat_template_kwargs"],
        "execution_mode": saved.get("execution_mode", "default"), "seed": 3407, "mark_snippets": False,
        "require_all_snippets": True, "prompt_version": "selected-expansion-sft-dev160-v1"}
    state = {"status": "running", "model_key": model, "expected_questions": 160,
        "training_run": str(training), "selection_run": str(selection),
        "selection_manifest_sha256": digest(selection / "manifest.json"), "sources": sources,
        "common_settings": common, "input_plan": plan, "input_sha256": digest(development),
        "prompt_sha256": digest(prompt_path), "runs": {}, "reused_from": {}, "arm_hashes": {},
        "evaluation": "Exploratory dev160 evaluation; checkpoint decisions frozen on 144 internal-validation questions"}
    write(output / "manifest.json", state)
    try:
        for criterion in CRITERIA:
            for mode in MODES:
                name = f"{criterion}_{mode}"
                original = f"best_eval_loss_{mode}"
                if criterion == "best_coverage" and sources[criterion]["generation_files_sha256"] == sources["best_eval_loss"]["generation_files_sha256"]:
                    state["runs"][name] = state["runs"][original]
                    state["reused_from"][name] = original
                    state["arm_hashes"][name] = state["arm_hashes"][original]
                    print(f"{name}: same checkpoint/tokenizer as {original}; reusing exact generations", flush=True)
                    continue
                settings = {**common, "response_mode": "equivalent_greedy" if mode == "greedy" else "equivalent_sampling",
                    "temperature": 0 if mode == "greedy" else .8, "num_generations": 1 if mode == "greedy" else 10}
                if mode == "sampling10":
                    settings.update(top_p=.95, top_k=0)
                configuration = output / f"{name}_config.json"
                write(configuration, settings)
                run_path = output / name
                subprocess.run([sys.executable, str(ROOT / "scripts/run_single_answer_sampling.py"),
                    "--config", str(configuration), "--model-name", sources[criterion]["adapter"],
                    "--output-dir", str(run_path), "--limit", "160", "--preflight-all"], check=True, cwd=ROOT)
                if read(run_path / "status.json")["status"] != "complete" or len(records(run_path / "generations.jsonl")) != 160:
                    raise ValueError("Incomplete development generation")
                state["runs"][name] = name
                state["arm_hashes"][name] = {n: digest(run_path / n) for n in (
                    "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
                write(output / "manifest.json", state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/analyze_selected_sft_evaluation.py"), str(output)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate-protocol", "run"), default="run")
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen3", "ministral3"])
    parser.add_argument("--model", choices=("llama31", "qwen3", "ministral3"))
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--selection-run", type=Path)
    args = parser.parse_args()
    if args.mode == "validate-protocol":
        validate_protocol(args.submission_dir.resolve(), args.models)
        print("Validated disjoint 160-question development evaluation protocol")
    else:
        if not args.model or not args.training_run or not args.selection_run:
            parser.error("run requires --model, --training-run and --selection-run")
        output = args.training_run.resolve() / f"dev-evaluation-{os.environ.get('PBS_JOBID', 'manual')}"
        run(args.training_run.resolve(), args.selection_run.resolve(), args.submission_dir.resolve(), output, args.model)


if __name__ == "__main__":
    main()
