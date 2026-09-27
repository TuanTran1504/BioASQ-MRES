#!/usr/bin/env python3
"""Launch a reproducible 8B BioASQ factoid SFT run from a named configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BUNDLE_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("full_resources_per_alias", "evidence_per_supported_alias"),
        help="Named SFT data/configuration variant.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique label for this run. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Bundle-relative directory for manifests, checkpoints, and adapters.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("BIOASQ_MODEL_NAME"),
        help="Override the base model or cached Hugging Face model identifier.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model downloads. Compute jobs should normally use the pre-cached model instead.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small, one-epoch validation job before committing GPU time to a full run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training command without starting training.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    args = parse_args()
    config_path = CONFIG_DIR / f"{args.variant}.json"
    require_file(config_path, "configuration")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    train_input = BUNDLE_ROOT / config["train_input"]
    eval_input = BUNDLE_ROOT / config["eval_input"]
    prompt_file = BUNDLE_ROOT / "prompts/factoid_single_answer_aligned.json"
    java_jar = BUNDLE_ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    for path, label in ((train_input, "training data"), (eval_input, "dev data"), (prompt_file, "prompt registry"), (java_jar, "BioASQ evaluator")):
        require_file(path, label)

    subset_ids = config.get("generated_metric_subset_ids")
    subset_path = BUNDLE_ROOT / subset_ids if subset_ids else None
    if subset_path is not None:
        require_file(subset_path, "generated metric subset IDs")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"bioasq-llama31-8b-{args.variant}-{timestamp}"
    output_root = (BUNDLE_ROOT / args.output_root).resolve()
    model_name = args.model_name or config["model_name"]

    command = [
        sys.executable,
        "-m",
        "src.utility.answer_gen_ft",
        "--train-input",
        str(train_input),
        "--eval-input",
        str(eval_input),
        "--question-types",
        "factoid",
        "--validation-ratio",
        "0.0",
        "--model-name",
        model_name,
        "--chat-template",
        config["chat_template"],
        "--prompt-format",
        "chat",
        "--prompt-file",
        str(prompt_file),
        "--prompt",
        config["prompt"],
        "--max-resources",
        "0",
        "--max-resource-chars",
        "0",
        "--max-factoid-answers",
        str(config["max_factoid_answers"]),
        "--max-seq-length",
        str(config["max_seq_length"]),
        "--per-device-train-batch-size",
        str(config["per_device_train_batch_size"]),
        "--per-device-eval-batch-size",
        str(config["per_device_eval_batch_size"]),
        "--gradient-accumulation-steps",
        str(config["gradient_accumulation_steps"]),
        "--warmup-steps",
        str(config["warmup_steps"]),
        "--num-train-epochs",
        str(config["num_train_epochs"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--logging-steps",
        str(config["logging_steps"]),
        "--save-strategy",
        "steps",
        "--save-steps",
        str(config["save_steps"]),
        "--lora-r",
        str(config["lora_r"]),
        "--lora-alpha",
        str(config["lora_alpha"]),
        "--lora-dropout",
        str(config["lora_dropout"]),
        "--dataset-num-proc",
        "1",
        "--selection-metric",
        "generated_primary_score",
        "--selection-max-seq-length",
        str(config["selection_max_seq_length"]),
        "--selection-max-new-tokens",
        str(config["selection_max_new_tokens"]),
        "--selection-num-generations",
        "1",
        "--selection-aggregation-strategy",
        "union",
        "--early-stopping-patience",
        str(config["early_stopping_patience"]),
        "--early-stopping-threshold",
        str(config["early_stopping_threshold"]),
        "--bioasq-java-jar",
        str(java_jar),
        "--seed",
        str(config["seed"]),
        "--run-name",
        run_name,
        "--artifacts-root",
        str(output_root),
        "--registry-path",
        str(output_root / "models/registry.json"),
        "--resume-from-checkpoint",
        "auto",
    ]
    if subset_path is not None:
        command.extend(("--generated-metric-subset-ids", str(subset_path)))
    if not args.allow_download:
        command.append("--local-files-only")
    if args.smoke_test:
        command.extend(
            (
                "--max-train-samples",
                "8",
                "--max-eval-samples",
                "8",
                "--num-train-epochs",
                "1",
                "--save-steps",
                "4",
                "--selection-max-new-tokens",
                "32",
            )
        )

    print("Bundle root:", BUNDLE_ROOT)
    print("Variant:", args.variant)
    print("Run name:", run_name)
    print("Output root:", output_root)
    print("Resolved command:")
    print(" ".join(command))
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=BUNDLE_ROOT, check=True)


if __name__ == "__main__":
    main()
