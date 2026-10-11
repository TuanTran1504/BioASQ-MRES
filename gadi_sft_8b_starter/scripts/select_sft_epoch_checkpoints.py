#!/usr/bin/env python3
"""Score retained expansion SFT epochs and export coverage/loss winners separately."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, digest, read, records, resolved_config, write

sys.path.insert(0, str(ROOT.parent))
from scripts.analyze_matched_qwen_expansion import ranking_metrics

JAR = ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"


def epoch_checkpoints(directory, expected_epochs):
    result = []
    for path in (directory / "trainer_output").glob("checkpoint-*"):
        if not path.is_dir():
            continue
        path = path.resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint escaped training run")
        state = read(path / "trainer_state.json")
        step = int(state["global_step"])
        epoch = float(state["epoch"])
        if path.name != f"checkpoint-{step}" or not epoch.is_integer():
            raise ValueError("Expected complete epoch checkpoints with matching global steps")
        losses = [float(row["eval_loss"]) for row in state["log_history"]
                  if "eval_loss" in row and row.get("step") == step
                  and abs(float(row["epoch"]) - epoch) < 1e-8]
        if len(losses) != 1 or not math.isfinite(losses[0]):
            raise ValueError("Checkpoint lacks one finite loss evaluated at its own epoch")
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / filename).is_file() or not (path / filename).stat().st_size:
                raise ValueError(f"Missing checkpoint adapter: {path / filename}")
        result.append({"path": path, "epoch": int(epoch), "global_step": step, "eval_loss": losses[0],
                       "adapter_sha256": {f: digest(path / f) for f in ("adapter_config.json", "adapter_model.safetensors")}})
    result.sort(key=lambda row: row["epoch"])
    if [r["epoch"] for r in result] != list(range(1, expected_epochs + 1)):
        raise ValueError("Every epoch must be retained; refusing selection from a loss-pruned subset")
    return result


def choose_winners(rows):
    if not rows:
        raise ValueError("No scored checkpoints")
    coverage = max(rows, key=lambda r: (r["metrics"]["coverage_full_pool"],
                                       r["metrics"]["coverage_at10"], -r["epoch"]))
    loss = min(rows, key=lambda r: (r["eval_loss"], r["epoch"]))
    return {"best_coverage": coverage, "best_eval_loss": loss}


def stage_adapter(checkpoint, tokenizer_source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(checkpoint / filename, destination / filename)
    # Trainer snapshots may omit the text tokenizer. Use the run's saved native
    # tokenizer, which is identical for all epochs, without copying loss weights.
    for path in tokenizer_source.iterdir():
        if path.is_file() and path.name not in {"adapter_config.json", "adapter_model.safetensors", "training_complete.json"}:
            shutil.copy2(path, destination / path.name)
    if not (destination / "tokenizer_config.json").is_file():
        raise ValueError("Missing native tokenizer for the selected checkpoint")


def score_generation(run, expected_ids):
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    generations = records(run / "generations.jsonl")
    if (read(run / "status.json")["status"] != "complete" or len(generations) != len(expected_ids)
            or {r["question_id"] for r in generations} != set(expected_ids)):
        raise ValueError("Incomplete validation generation; missing questions cannot be dropped")
    if any(r["snippets_truncated"] or r["included_snippets"] != r["total_snippets"]
           or len(r["samples"]) != 10 for r in generations):
        raise ValueError("Validation evidence or sample budget differs")
    report, diagnostics, scored = analyze_local_expansion(run, jar_path=JAR)
    metrics, per_question = ranking_metrics(scored)
    for source, row in zip(scored, per_question):
        row["coverage_full_pool"] = float(bool(source["matching_answers"]))
    metrics["coverage_full_pool"] = sum(r["coverage_full_pool"] for r in per_question) / len(per_question)
    metrics["mean_unique_candidates"] = diagnostics["mean_unique_candidates"]
    metrics["parse_success_rate"] = diagnostics["parse_success_rate"]
    write(run / "selection_scores.json", {"metrics": metrics, "questions": per_question,
                                            "official_report": str(report)})
    return metrics, report


def validate_run(directory, submission, model, smoke):
    prepared = read(submission / "manifest.json")
    if prepared["status"] != "ready" or model not in prepared["models"]:
        raise ValueError("Submission is not ready for this model")
    matrix_path = submission / "training_matrix.json"
    if prepared["training_matrix_sha256"] != digest(matrix_path):
        raise ValueError("Prepared training matrix changed")
    for filename, field in (("validation_examples.jsonl", "validation_examples_sha256"),
                            ("expansion_prompt.txt", "expansion_prompt_sha256")):
        if prepared[field] != digest(submission / filename):
            raise ValueError("Validation inputs or prompt changed after preparation")
    matrix = read(matrix_path)
    state = read(directory / "status.json")
    saved = state["configuration"]
    expected = resolved_config(matrix, model, "expansion")
    if (state["status"] != "completed" or bool(state["smoke_test"]) != smoke
            or any(saved.get(k) != v for k, v in expected.items())
            or saved["matrix_sha256"] != digest(matrix_path)):
        raise ValueError("Expected a completed expansion SFT run of the prepared matrix")
    pin = read(submission / "base_pins.json")
    if prepared["base_pins_sha256"] != digest(submission / "base_pins.json") or pin["matrix_sha256"] != digest(matrix_path):
        raise ValueError("Base pin matrix changed")
    base = pin["models"][model]
    if saved["base_snapshot"] != base["snapshot"] or saved["base_revision"] != base["revision"]:
        raise ValueError("Training did not use the pinned original backbone")
    if not smoke and (state["selected_train_examples"] != 1296 or state["selected_validation_examples"] != 144):
        raise ValueError("Full run must use all fitting and validation questions")
    if saved.get("execution_mode", "default") != ("eager" if model == "ministral3" else "default"):
        raise ValueError("Unexpected model execution mode")
    if state["dataset_validation"]["validation_sha256"] != digest(ROOT / saved["eval_input"]):
        raise ValueError("Teacher validation targets changed after training")
    examples = records(submission / "validation_examples.jsonl")
    ids = [r["question_id"] for r in examples]
    if len(ids) != 144 or len(set(ids)) != 144 or set(ids) != {r["question_id"] for r in records(ROOT / saved["eval_input"])}:
        raise ValueError("Generation panel must be exactly the 144 loss-validation questions")
    return prepared, saved, examples


def run_selection(directory, submission, output, model, smoke):
    prepared, saved, examples = validate_run(directory, submission, model, smoke)
    checkpoints = epoch_checkpoints(directory, 1 if smoke else 3)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "model_key": model, "smoke_test": smoke,
             "selector_sha256": digest(Path(__file__)),
             "generation_worker_sha256": digest(ROOT / "scripts/run_single_answer_sampling.py"),
             "training_run": str(directory), "training_status_sha256": digest(directory / "status.json"),
             "submission_manifest_sha256": digest(submission / "manifest.json"),
             "validation_questions": 4 if smoke else 144,
             "selection": prepared["configuration"]["selection"],
             "checkpoint_scores": [], "selected_adapters": {}}
    write(output / "manifest.json", state)
    try:
        for checkpoint in checkpoints:
            epoch_root = output / f"epoch-{checkpoint['epoch']}"
            adapter = epoch_root / "adapter"
            stage_adapter(checkpoint["path"], directory / "adapter", adapter)
            config = {**state["selection"], "input": str(submission / "validation_examples.jsonl"),
                "prompt": str(submission / "expansion_prompt.txt"),
                "response_mode": "equivalent_sampling", "model_loader": saved["model_loader"],
                "chat_template_kwargs": saved["chat_template_kwargs"],
                "execution_mode": saved.get("execution_mode", "default"),
                "prompt_version": "expansion-sft-internal-validation-native-v1"}
            config_path = epoch_root / "sampling_config.json"
            write(config_path, config)
            inference = epoch_root / "sampling10"
            subprocess.run([sys.executable, str(ROOT / "scripts/run_single_answer_sampling.py"),
                "--config", str(config_path), "--model-name", str(adapter), "--output-dir", str(inference),
                "--limit", "4" if smoke else "144", "--preflight-all",
                *(["--longest-first"] if smoke else [])], cwd=ROOT, check=True)
            expected_ids = [e["question_id"] for e in examples]
            if smoke:
                selected = records(inference / "examples.jsonl")
                selected_ids = [e["question_id"] for e in selected]
                by_id = {e["question_id"]: e for e in examples}
                if (len(selected_ids) != 4 or len(set(selected_ids)) != 4
                        or any(e != by_id.get(e["question_id"]) for e in selected)):
                    raise ValueError("Smoke must generate four intact internal-validation questions")
                expected_ids = selected_ids
            metrics, report = score_generation(inference, expected_ids)
            row = {**{k: str(v) if isinstance(v, Path) else v for k, v in checkpoint.items()},
                   "metrics": metrics, "inference": str(inference), "official_report": str(report),
                   "scored_artifact_sha256": {name: digest(inference / name)
                       for name in ("config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "selection_scores.json")}}
            state["checkpoint_scores"].append(row)
            write(output / "manifest.json", state)
            print(f"Epoch {row['epoch']}: eval loss={row['eval_loss']:.6f}, "
                  f"full-pool coverage={metrics['coverage_full_pool']:.4f}, "
                  f"coverage@10={metrics['coverage_at10']:.4f}", flush=True)
        if smoke:
            if not any(r["metrics"]["parse_success_rate"] > 0 for r in state["checkpoint_scores"]):
                raise RuntimeError("Smoke produced no parseable expansion response")
        for name, winner in choose_winners(state["checkpoint_scores"]).items():
            adapter = output / "selected_adapters" / name
            stage_adapter(Path(winner["path"]), directory / "adapter", adapter)
            metadata = {"status": "completed", "smoke_test": smoke, "selection_name": name,
                        "model_key": model, "checkpoint": winner,
                        "base_snapshot": saved["base_snapshot"], "base_revision": saved["base_revision"],
                        "training_configuration": saved, "selection_configuration": state["selection"]}
            write(adapter / "training_complete.json", metadata)
            state["selected_adapters"][name] = {"path": str(adapter), "epoch": winner["epoch"],
                "global_step": winner["global_step"], "eval_loss": winner["eval_loss"], "metrics": winner["metrics"],
                "files_sha256": {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}}
        state["status"] = "complete"
        print(json.dumps(state["selected_adapters"], indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "manifest.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama31", "qwen3", "ministral3"), required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Verify saved training and epoch snapshots without loading a GPU model")
    args = parser.parse_args()
    if args.validate_only:
        validate_run(args.training_run.resolve(), args.submission_dir.resolve(), args.model, args.smoke_test)
        epoch_checkpoints(args.training_run.resolve(), 1 if args.smoke_test else 3)
        print(f"Validated reusable {'smoke' if args.smoke_test else 'full'} training: {args.training_run}", flush=True)
        return
    if not args.output_dir:
        parser.error("--output-dir is required unless --validate-only is used")
    run_selection(args.training_run.resolve(), args.submission_dir.resolve(), args.output_dir.resolve(), args.model, args.smoke_test)


if __name__ == "__main__":
    main()
