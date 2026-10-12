#!/usr/bin/env python3
"""Pin a fresh expansion SFT run and its gold-blind validation-generation panel."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, digest, expansion_rows, load_source, read, records, write
from run_matched_8b_evaluation import validate_training_pair

DEFAULT_CONFIG = ROOT / "configs/sft_dual_selection_8b.json"
sys.path.insert(0, str(ROOT.parent))


def validation_examples(rows, audit):
    labels = {r["question_id"]: r for r in audit}
    if len(labels) != len(audit):
        raise ValueError("Duplicate audit question IDs")
    output = []
    ids = set()
    for row in sorted(rows, key=lambda r: r["question_id"]):
        qid = row["question_id"]
        if qid in ids:
            raise ValueError("Duplicate validation question")
        ids.add(qid)
        if qid not in labels or labels[qid]["internal_split"] != "validation":
            raise ValueError("Validation question not in the audited validation partition")
        messages = row["messages"]
        user = json.loads(messages[1]["content"])
        if set(user) != {"question", "snippets"}:
            raise ValueError("Generation inputs must contain only question and snippets")
        aliases = labels[qid]["gold_aliases"]
        if not isinstance(aliases, list) or not aliases or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError("Missing official validation aliases")
        output.append({"question_id": qid, "question": user["question"],
                       "snippets": [{"snippet_id": s["id"], "text": s["text"]} for s in user["snippets"]],
                       "gold_aliases": aliases})
    return output


def prepare(config_path, destination, models):
    config = read(config_path)
    if not models or len(set(models)) != len(models) or not set(models) <= set(config["models"]):
        raise ValueError("Use unique configured model names")
    if destination.exists():
        raise FileExistsError(destination)
    matrix_path = ROOT / config["training_matrix"]
    matrix = read(matrix_path)
    if matrix.get("expansion_candidate_limit", 10) != config["selection"].get("max_candidates", 10):
        raise ValueError("Training targets and validation generation must use the same candidate budget")
    source, splits, _ = load_source(matrix)
    audit_path = source / "audit.jsonl"
    if digest(audit_path) != read(source / "manifest.json")["output_sha256"]["audit.jsonl"]:
        raise ValueError("Audited gold alias file differs from the pinned source")
    audit = records(audit_path)
    if {r["question_id"] for r in audit} != {r["question_id"] for rows in splits.values() for r in rows}:
        raise ValueError("Audit question membership differs from the fitting/validation split")
    aliases = {r["question_id"]: r["gold_aliases"] for r in audit}
    splits = {split: expansion_rows(rows, matrix) for split, rows in splits.items()}
    if any(json.loads(r["messages"][-1]["content"])["answers"][0]["answer"] not in aliases[r["question_id"]]
           for rows in splits.values() for r in rows):
        raise ValueError("Every fitting/validation target must retain its first official alias")
    examples = validation_examples(splits["validation"], audit)
    if len(examples) != 144:
        raise ValueError("Expected all 144 internal-validation questions")
    prompt = splits["validation"][0]["messages"][0]["content"]
    if any(r["messages"][0]["content"] != prompt for rows in splits.values() for r in rows):
        raise ValueError("Expansion system prompts differ")
    epochs = float(matrix["common"]["num_train_epochs"])
    if epochs != 3 or matrix["common"]["save_strategy"] != "epoch" or matrix["common"]["early_stopping_patience"] != 0:
        raise ValueError("This protocol requires three complete epochs and no early stopping")
    baseline = read(ROOT / config["baseline_evaluation_config"])
    historical_path = config.get("historical_training_matrix", config["training_matrix"])
    if baseline["training_matrix"] != historical_path:
        raise ValueError("Historical base pins refer to a different training matrix")
    historical = read(ROOT / historical_path)
    for field in ("source_dataset", "source_manifest_sha256", "outer_dev", "single_answer_prompt", "common", "models"):
        if matrix[field] != historical[field]:
            raise ValueError(f"Budget experiment changes historical training settings: {field}")
    provenance = {model: validate_training_pair(baseline, model)["expansion"] for model in models}
    from src.utility.bioasq_official import _conda_executable, _official_adapter_classes
    subprocess.run([*_conda_executable("java"), "-version"], check=True)
    _official_adapter_classes(ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    fresh = copy.deepcopy(matrix)
    fresh["common"]["save_total_limit"] = 3
    destination.mkdir(parents=True, exist_ok=False)
    matrix_output = destination / "training_matrix.json"
    write(matrix_output, fresh)
    pins = {"matrix_sha256": digest(matrix_output), "models": {model: {
        "model_name": matrix["models"][model]["model_name"],
        "snapshot": source["base_snapshot"], "revision": source["base_revision"]}
        for model, source in provenance.items()}}
    write(destination / "base_pins.json", pins)
    (destination / "validation_examples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in examples), encoding="utf-8", newline="\n")
    (destination / "expansion_prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    write(destination / "manifest.json", {"status": "ready", "models": models,
        "configuration": config, "configuration_sha256": digest(config_path),
        "training_matrix_sha256": digest(matrix_output), "source_matrix_sha256": digest(matrix_path),
        "historical_matrix_sha256": digest(ROOT / historical_path),
        "base_pins_sha256": digest(destination / "base_pins.json"),
        "validation_examples_sha256": digest(destination / "validation_examples.jsonl"),
        "expansion_prompt_sha256": digest(destination / "expansion_prompt.txt"),
        "audit_sha256": digest(audit_path), "historical_sources": provenance,
        "fitting_questions": 1296, "validation_questions": 144,
        "checkpoint_policy": "One fresh base-initialized training run, two independent selections; all three epoch checkpoints retained"})
    print(f"Prepared {destination}: 1296 fitting / 144 validation; ten-draw coverage and eval-loss selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen3", "ministral3"])
    args = parser.parse_args()
    prepare(args.config.resolve(), args.submission_dir.resolve(), args.models)


if __name__ == "__main__":
    main()
