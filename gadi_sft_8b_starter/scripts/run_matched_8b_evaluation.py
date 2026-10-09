#!/usr/bin/env python3
"""Validate a matched SFT pair, then run three gold-blind inference arms."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, digest, read, records, resolved_config, validate, write
from run_original_qwen25_inference import validate_smoke_outputs
from matched_8b_provenance import validate_provenance

DEFAULT_CONFIG = ROOT / "configs/matched_8b_evaluation.json"
CONDITIONS = ("original_greedy", "original_sampling10", "expansion_greedy")


def validate_training_pair(config, key):
    matrix_path = ROOT / config["training_matrix"]
    matrix = read(matrix_path)
    validate(matrix)
    input_path = ROOT / config["input"]
    if digest(input_path) != config["input_sha256"]:
        raise ValueError("Development data differ from the pinned 160-question export")
    if input_path.resolve() != (ROOT / matrix["outer_dev"]).resolve():
        raise ValueError("Evaluation must use the training matrix's disjoint outer dev")
    examples = records(input_path)
    if len(examples) != 160 or len({row["question_id"] for row in examples}) != 160:
        raise ValueError("Expected 160 unique development questions")
    if key not in config["models"]:
        raise ValueError(f"No completed adapter pair configured for {key}")
    provenance, states = {}, {}
    for formulation in ("original", "expansion"):
        directory = (ROOT / config["models"][key][formulation]).resolve()
        state = read(directory / "status.json")
        completion = read(directory / "adapter/training_complete.json")
        if state["status"] != "completed" or state["smoke_test"] or completion["status"] != "completed":
            raise ValueError(f"Expected completed full training: {directory}")
        if state["selected_train_examples"] != 1296 or state["selected_validation_examples"] != 144:
            raise ValueError("Completed training did not use the full matched fitting/validation sets")
        saved = state["configuration"]
        expected = resolved_config(matrix, key, formulation)
        if any(saved.get(field) != value for field, value in expected.items()):
            raise ValueError("Saved training configuration differs from the matched matrix")
        if saved["matrix_sha256"] != digest(matrix_path):
            raise ValueError("Training matrix hash differs")
        snapshot = Path(saved["base_snapshot"])
        if snapshot.name != saved["base_revision"] or not (snapshot / "config.json").is_file():
            raise ValueError("Pinned backbone snapshot is unavailable")
        adapter = directory / "adapter"
        for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
            if not (adapter / filename).is_file() or not (adapter / filename).stat().st_size:
                raise ValueError(f"Missing/empty adapter file: {adapter / filename}")
        if read(adapter / "adapter_config.json")["base_model_name_or_path"] != str(snapshot):
            raise ValueError("Adapter backbone differs from the pinned training snapshot")
        training_rows = records(ROOT / saved["train_input"])
        validation_rows = records(ROOT / saved["eval_input"])
        for split, field in (("train", "train_input"), ("validation", "eval_input")):
            if state["dataset_validation"][split + "_sha256"] != digest(ROOT / saved[field]):
                raise ValueError("Training data hashes differ from the completed run")
        system_prompt = training_rows[0]["messages"][0]["content"]
        if any(row["messages"][0]["content"] != system_prompt for row in training_rows + validation_rows):
            raise ValueError("Training system prompts differ within a formulation")
        files = {path.name: digest(path) for path in adapter.iterdir() if path.is_file()}
        provenance[formulation] = {
            "training_run": directory.name, "adapter": str(adapter), "adapter_files_sha256": files,
            "status_sha256": digest(directory / "status.json"),
            "base_snapshot": str(snapshot), "base_revision": saved["base_revision"],
            "matrix_sha256": saved["matrix_sha256"], "system_prompt": system_prompt,
            "execution_mode": saved.get("execution_mode", "default"),
            "model_loader": saved["model_loader"], "chat_template_kwargs": saved["chat_template_kwargs"]}
        states[formulation] = saved
    for field in ("base_snapshot", "base_revision", "matrix_sha256", "model_loader", "chat_template_kwargs"):
        if states["original"][field] != states["expansion"][field]:
            raise ValueError(f"Training pair differs in {field}")
    print(f"Validated {key}: completed matched adapters; same pinned backbone; 160 disjoint dev questions", flush=True)
    return provenance


def arm_config(config, provenance, condition, prompt_path):
    formulation = "expansion" if condition.startswith("expansion_") else "original"
    source = provenance[formulation]
    result = {"input": config["input"], "prompt": str(prompt_path),
              "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
              "max_seq_length": config["max_seq_length"], "max_new_tokens": config["max_new_tokens"],
              "require_all_snippets": True, "seed": config["seed"], "expected_questions": 160,
              "prompt_version": "matched-8b-" + formulation + "-native-v1", "mark_snippets": False,
              "temperature": 0, "num_generations": 1}
    if condition == "expansion_greedy":
        result["response_mode"] = "equivalent"
    else:
        result["response_mode"] = "single_answer_greedy"
    if condition == "original_sampling10":
        result.update(response_mode="single_answer_sampling", num_generations=10, temperature=0.8, top_p=0.95)
    if condition == "expansion_sampling10":
        result.update(response_mode="equivalent_sampling", num_generations=10, temperature=0.8, top_p=0.95, top_k=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("validate", "run"), default="validate")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--expansion-sampling-only", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    provenance = validate_training_pair(config, args.model)
    baseline = None
    if args.expansion_sampling_only:
        baseline = (ROOT / config["baseline_experiments"][args.model]).resolve()
        previous = read(baseline / "manifest.json")
        expected = {"status": "complete", "smoke_test": False, "expected_questions": 160,
                    "model_key": args.model, "input_sha256": config["input_sha256"]}
        differences = [field for field, value in expected.items() if previous.get(field) != value]
        if set(previous["runs"]) != set(CONDITIONS):
            differences.append("runs")
        if differences:
            raise ValueError("Baseline evaluation differs in: " + ", ".join(differences))
        validate_provenance(previous["provenance"], provenance)
    if args.mode == "validate":
        return
    name = args.run_name or f"{args.model}-8b-evaluation-{os.environ.get('PBS_JOBID', 'manual')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a directory name")
    output = ROOT / ("outputs/expansion_sampling_8b" if args.expansion_sampling_only else "outputs/matched_8b_evaluation") / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "model_key": args.model, "smoke_test": args.smoke_test,
                "expected_questions": 4 if args.smoke_test else 160, "runs": {},
                "provenance": provenance, "input_sha256": config["input_sha256"],
                "evaluation_config_sha256": digest(args.config),
                "evaluation": "development-only matched SFT formulation comparison",
                "selection": "first five unique candidates in response/draw order"}
    write(output / "manifest.json", manifest)
    if baseline is not None:
        manifest.update(baseline_experiment=str(baseline), baseline_manifest_sha256=digest(baseline / "manifest.json"),
                        selection="first five unique candidates in draw then within-response order",
                        evaluation="development-only ten-sample expansion SFT comparison")
        write(output / "manifest.json", manifest)
    try:
        for condition in (("expansion_sampling10",) if args.expansion_sampling_only else CONDITIONS):
            formulation = "expansion" if condition.startswith("expansion_") else "original"
            prompt_path = output / (formulation + "_prompt.txt")
            prompt_path.write_text(provenance[formulation]["system_prompt"], encoding="utf-8")
            configuration = arm_config(config, provenance, condition, prompt_path)
            path = output / (condition + "_config.json")
            write(path, configuration)
            command = [sys.executable, str(ROOT / "scripts" / (
                "run_extractive_expansion_8b.py" if condition == "expansion_greedy" else "run_single_answer_sampling.py")),
                "--config", str(path), "--model-name", provenance[formulation]["adapter"],
                "--limit", str(manifest["expected_questions"])]
            target = output / condition
            if condition == "expansion_greedy":
                target.mkdir()
                command.extend(["--output-parent", str(target), "--run-name", name])
            else:
                command.extend(["--output-dir", str(target)])
            subprocess.run(command, check=True)
            if condition == "expansion_greedy":
                children = list(target.iterdir())
                if len(children) != 1:
                    raise ValueError("Expected one expansion run")
                target = children[0]
            state = read(target / "status.json")
            rows = records(target / "generations.jsonl")
            if (state["status"] != "complete" or state["completed_questions"] != manifest["expected_questions"]
                    or len(rows) != manifest["expected_questions"]):
                raise ValueError(f"Incomplete inference arm: {condition}")
            manifest["runs"][condition] = target.relative_to(output).as_posix()
            write(output / "manifest.json", manifest)
        if args.smoke_test:
            validate_smoke_outputs({key: output / value for key, value in manifest["runs"].items()})
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", manifest)
    print(f"Matched 8B evaluation complete: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
