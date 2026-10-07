#!/usr/bin/env python3
"""Validate and run matched base/SFT expansion on the fixed development set."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUNS = {
    "05b": "qwen25-05b-expansion-sft-180598547.gadi-pbs",
    "3b": "qwen25-3b-expansion-sft-180598549.gadi-pbs",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(model_size, adapter=None):
    config_path = ROOT / f"configs/equivalent_expansion_qwen25_{model_size}.json"
    config = read(config_path)
    training = ROOT / "outputs/expansion_sft" / TRAINING_RUNS[model_size]
    adapter = Path(adapter).resolve() if adapter else training / "adapter"
    completion = read(adapter / "training_complete.json")
    status = read(adapter.parent / "status.json")
    adapter_config = read(adapter / "adapter_config.json")
    if completion.get("status") != "completed" or status.get("smoke_test") is not False:
        raise ValueError("Expected a completed full-training adapter, not a smoke adapter")
    if status["model"] != config["model_name"]:
        raise ValueError("The adapter and baseline refer to different backbones")
    if not any(adapter.glob("adapter_model.*")):
        raise FileNotFoundError(f"Missing adapter weights: {adapter}")
    if not adapter_config.get("base_model_name_or_path"):
        raise ValueError("Adapter metadata has no base-model reference")
    examples = jsonl(ROOT / config["input"])
    dev_ids = {row["question_id"] for row in examples}
    if len(examples) != 160 or len(dev_ids) != 160:
        raise ValueError("Expected exactly 160 unique development questions")
    for name, count in (("train", 1296), ("validation", 144)):
        key = "train_input" if name == "train" else "eval_input"
        path = (ROOT / status["configuration"][key]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != status["dataset_validation"][f"{name}_sha256"]:
            raise ValueError(f"{name} data differs from the completed training run")
        rows = jsonl(path)
        ids = {row["question_id"] for row in rows}
        if len(rows) != count or len(ids) != count or ids & dev_ids:
            raise ValueError(f"Unexpected {name} count or overlap with development questions")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_expansion_bundle.py"),
                    "--config", str(config_path)], check=True)
    print(f"Validated {model_size}: completed adapter; 160 dev questions disjoint from fitting/validation", flush=True)
    return config_path, config, adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=TRAINING_RUNS, required=True)
    parser.add_argument("--mode", choices=("validate", "run"), default="validate")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config_path, config, adapter = validate(args.model_size, args.adapter)
    if args.mode == "validate":
        return
    name = args.run_name or f"qwen25-{args.model_size}-dev160-{os.environ.get('PBS_JOBID', 'manual')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a directory name")
    output = ROOT / "outputs/matched_qwen25" / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "model_size": args.model_size,
                "training_run": adapter.parent.name, "smoke_test": args.smoke_test,
                "expected_questions": 4 if args.smoke_test else 160,
                "selection": "original generated candidate order; final top five", "runs": {}}
    manifest_path = output / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for condition, model in (("base", config["model_name"]), ("sft", str(adapter))):
            parent = output / condition
            parent.mkdir()
            command = [sys.executable, str(ROOT / "scripts/run_extractive_expansion_8b.py"),
                       "--config", str(config_path), "--model-name", model,
                       "--output-parent", str(parent), "--run-name", f"{name}-{condition}"]
            if args.smoke_test:
                command.extend(["--limit", "4"])
            print(f"Starting {condition}: {model}", flush=True)
            subprocess.run(command, check=True)
            runs = list(parent.iterdir())
            if len(runs) != 1:
                raise ValueError(f"Expected one {condition} output directory")
            state = read(runs[0] / "status.json")
            if state["status"] != "complete" or state["completed_questions"] != manifest["expected_questions"]:
                raise ValueError(f"Incomplete {condition} generation")
            manifest["runs"][condition] = runs[0].relative_to(output).as_posix()
            save()
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    print(f"Matched generation complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
