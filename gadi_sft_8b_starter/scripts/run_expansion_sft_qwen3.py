#!/usr/bin/env python3
"""Validate or train Qwen3-8B on controlled multi-answer expansion records.

The input is the question-level chat JSONL produced by
scripts/build_expansion_sft_dataset.py. System and user tokens are masked; only
the assistant JSON response contributes to the SFT loss. The native Qwen3 chat
template is used with thinking disabled. No record is silently truncated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_ROOT.parent
DEFAULT_CONFIG = BUNDLE_ROOT / "configs/expansion_sft_qwen3_8b.json"
RELATIONS = {
    "original",
    "synonym",
    "abbreviation_expansion",
    "nomenclature_variant",
    "spelling_or_inflection",
    "numerically_equivalent",
    "harmless_formatting",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str | Path, base: Path = BUNDLE_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def normalized_surface(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def validate_rows(rows: list[dict[str, Any]], label: str, formulation: str = "expansion") -> dict[str, Any]:
    if formulation not in {"expansion", "original"}:
        raise ValueError(f"Unknown SFT formulation: {formulation}")
    question_ids: list[str] = []
    answer_counts: Counter[int] = Counter()
    relation_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        qid = str(row.get("question_id", "")).strip()
        messages = row.get("messages")
        if not qid or not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{label} row {index}: expected question_id and three messages")
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{label} row {index}: expected system/user/assistant roles")
        if any(not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
            raise ValueError(f"{label} row {index}: every message needs nonempty text content")
        if formulation == "original":
            target = messages[-1]["content"]
            match = re.fullmatch(r"Answer: \[BE\](.*?)\[EE\]", target, flags=re.DOTALL)
            if (not match or not match[1].strip() or target.count("[BE]") != 1
                    or target.count("[EE]") != 1):
                raise ValueError(f"{label} row {index}: expected one Answer: [BE]expression[EE] target")
            question_ids.append(qid)
            answer_counts[1] += 1
            continue
        try:
            target = json.loads(messages[-1]["content"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {index}: assistant target is not JSON") from exc
        if not isinstance(target, dict) or set(target) != {"answers"}:
            raise ValueError(f"{label} row {index}: target must contain only the answers field")
        answers = target["answers"]
        if not isinstance(answers, list) or not 1 <= len(answers) <= 10:
            raise ValueError(f"{label} row {index}: answers must contain 1 to 10 items")
        seen: set[str] = set()
        original_count = 0
        for answer_index, answer in enumerate(answers):
            if not isinstance(answer, dict) or set(answer) != {"answer", "relation_type"}:
                raise ValueError(f"{label} row {index}: malformed answer item")
            surface = str(answer["answer"]).strip()
            relation = str(answer["relation_type"]).strip()
            if not surface or relation not in RELATIONS:
                raise ValueError(f"{label} row {index}: invalid answer surface or relation")
            key = normalized_surface(surface)
            if key in seen:
                raise ValueError(f"{label} row {index}: duplicate answer surface {surface!r}")
            seen.add(key)
            original_count += relation == "original"
            relation_counts[relation] += 1
            if answer_index == 0 and relation != "original":
                raise ValueError(f"{label} row {index}: first answer must be original")
        if original_count != 1:
            raise ValueError(f"{label} row {index}: expected exactly one original answer")
        question_ids.append(qid)
        answer_counts[len(answers)] += 1
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{label}: duplicate question IDs")
    return {
        "questions": len(rows),
        "question_ids": set(question_ids),
        "answer_count_histogram": dict(sorted(answer_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def validate_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = resolve_path(config["train_input"])
    eval_path = resolve_path(config["eval_input"])
    manifest_path = resolve_path(config["dataset_manifest"])
    for path in (train_path, eval_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("output_sha256", {})
    for path in (train_path, eval_path):
        expected = expected_hashes.get(path.name)
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"Dataset hash mismatch for {path}: expected={expected!r}, actual={actual}"
            )

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    formulation = config.get("formulation", "expansion")
    train_summary = validate_rows(train_rows, "train", formulation)
    eval_summary = validate_rows(eval_rows, "validation", formulation)
    overlap = train_summary.pop("question_ids") & eval_summary.pop("question_ids")
    if overlap:
        raise ValueError(f"Train/validation question overlap: {sorted(overlap)[:5]}")
    summary = {
        "train": train_summary,
        "validation": eval_summary,
        "train_validation_overlap": 0,
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(eval_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_outer_dev_questions": manifest.get("outer_dev_questions"),
        "dataset_official_test_overlap": manifest.get("official_test_overlap"),
    }
    return train_rows, eval_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "train"), default="validate")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", default="outputs/expansion_sft")
    parser.add_argument("--model-name")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def token_preflight(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_seq_length: int,
    chat_template_kwargs: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    lengths: list[tuple[int, str]] = []
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        lengths.append((len(token_ids), row["question_id"]))
    overflow = [(length, qid) for length, qid in lengths if length > max_seq_length]
    ordered = sorted(length for length, _ in lengths)
    summary = {
        "examples": len(rows),
        "maximum_tokens": max(ordered, default=0),
        "median_tokens": ordered[len(ordered) // 2] if ordered else 0,
        "overflow_count": len(overflow),
        "overflow_examples": [
            {"question_id": qid, "tokens": length} for length, qid in overflow[:20]
        ],
    }
    return summary, {qid: length for length, qid in lengths}


def audit_response_masks(dataset: Any, tokenizer: Any, response_part: str) -> dict[str, int]:
    """Fail before optimisation if response masking omits prompts or all targets."""
    supervised_tokens = 0
    for index, record in enumerate(dataset):
        ids, labels = list(record["input_ids"]), list(record["labels"])
        active = [position for position, label in enumerate(labels) if label != -100]
        if len(ids) != len(labels) or not active:
            raise ValueError(f"Response masking row {index}: no trainable assistant tokens")
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if response_part not in decoded:
            raise ValueError(f"Response masking row {index}: missing native assistant delimiter")
        expected_prefix = decoded.rsplit(response_part, 1)[0] + response_part
        masked_prefix = tokenizer.decode(ids[:active[0]], skip_special_tokens=False)
        if not masked_prefix.startswith(expected_prefix):
            raise ValueError(f"Response masking row {index}: prompt tokens contribute to loss")
        if any(labels[position] != ids[position] for position in active):
            raise ValueError(f"Response masking row {index}: labels differ from assistant tokens")
        supervised_tokens += len(active)
    return {"examples": len(dataset), "supervised_tokens": supervised_tokens}


def train(args: argparse.Namespace, config: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], validation: dict[str, Any]) -> None:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if str(BUNDLE_ROOT) not in sys.path:
        sys.path.insert(0, str(BUNDLE_ROOT))

    # Import the CUDA/Unsloth stack only for an actual training invocation.
    import unsloth
    import torch
    from src.utility.adapter_save import save_adapter_and_tokenizer
    from src.utility.dataset_builder import prepare_dataset
    from src.utility.training import (
        build_trainer,
        load_model_and_tokenizer,
        save_training_curves,
        train_trainer,
    )

    run_name = args.run_name or f"qwen3-8b-expansion-sft-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    run_dir = resolve_path(Path(args.output_root) / run_name)
    trainer_dir = run_dir / "trainer_output"
    adapter_dir = run_dir / "adapter"
    status_path = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (adapter_dir / "training_complete.json").exists():
        raise FileExistsError(f"Run is already complete: {run_dir}")

    max_train = args.max_train_samples
    max_eval = args.max_eval_samples
    epochs = float(config["num_train_epochs"])
    if args.smoke_test:
        max_train = max_train or 16
        max_eval = max_eval or 8
        epochs = 1.0

    model_name = args.model_name or config["model_name"]
    chat_kwargs = dict(config.get("chat_template_kwargs", {}))

    training_args = argparse.Namespace(
        model_name=model_name,
        model_loader=config.get("model_loader", "fast_language_model"),
        max_seq_length=int(config["max_seq_length"]),
        dtype=None,
        no_4bit=False,
        local_files_only=not args.allow_download,
        lora_r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        seed=int(config["seed"]),
        prompt_format="chat",
        chat_template=str(config["chat_template"]),
        preserve_native_chat_template=True,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=(
            min(4, int(config["gradient_accumulation_steps"]))
            if args.smoke_test
            else int(config["gradient_accumulation_steps"])
        ),
        warmup_steps=(0 if args.smoke_test else int(config["warmup_steps"])),
        num_train_epochs=epochs,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=int(config["logging_steps"]),
        save_strategy=str(config["save_strategy"]),
        save_steps=50,
        eval_strategy="auto",
        eval_steps=None,
        output_dir=str(trainer_dir),
        selection_metric="eval_loss",
        dataset_num_proc=1,
        early_stopping_patience=int(config["early_stopping_patience"]),
        early_stopping_threshold=float(config["early_stopping_threshold"]),
        instruction_part=config.get("instruction_part"),
        response_part=config.get("response_part"),
        response_template="\nAnswer:",
        response_template_trim_tokens=0,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_dtype="float32",
    )

    status = {
        "status": "initializing",
        "run_name": run_name,
        "created_at": utc_now(),
        "model": model_name,
        "configuration": config,
        "dataset_validation": validation,
        "requested_max_train_examples": max_train,
        "requested_max_validation_examples": max_eval,
        "smoke_test": args.smoke_test,
        "package_versions": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "unsloth")},
    }
    write_json(status_path, status)

    try:
        model, tokenizer = load_model_and_tokenizer(training_args)
        train_preflight, train_lengths = token_preflight(
            train_rows,
            tokenizer,
            max_seq_length=training_args.max_seq_length,
            chat_template_kwargs=chat_kwargs,
            label="train",
        )
        eval_preflight, eval_lengths = token_preflight(
            eval_rows,
            tokenizer,
            max_seq_length=training_args.max_seq_length,
            chat_template_kwargs=chat_kwargs,
            label="validation",
        )
        preflight = {"train": train_preflight, "validation": eval_preflight}
        overflowing = {
            label: summary
            for label, summary in preflight.items()
            if summary["overflow_count"]
        }
        if overflowing:
            status.update(
                {
                    "status": "failed_preflight",
                    "failed_at": utc_now(),
                    "token_preflight": preflight,
                    "error": (
                        "ValueError: examples exceed "
                        f"max_seq_length={training_args.max_seq_length}; "
                        "refusing to truncate evidence or targets"
                    ),
                }
            )
            write_json(status_path, status)
            details = ", ".join(
                f"{label}={summary['overflow_count']}"
                for label, summary in overflowing.items()
            )
            raise ValueError(
                f"{details} examples exceed max_seq_length={training_args.max_seq_length}; "
                "refusing to truncate evidence or targets"
            )
        # A smoke test should exercise the highest-memory examples rather than
        # an arbitrary prefix of the data.
        selected_train = sorted(
            train_rows,
            key=lambda row: train_lengths[row["question_id"]],
            reverse=True,
        )[:max_train] if max_train else train_rows
        selected_eval = sorted(
            eval_rows,
            key=lambda row: eval_lengths[row["question_id"]],
            reverse=True,
        )[:max_eval] if max_eval else eval_rows
        status.update(
            {
                "status": "preflight_complete",
                "token_preflight": preflight,
                "selected_train_examples": len(selected_train),
                "selected_validation_examples": len(selected_eval),
                "smoke_selection": "longest_examples" if args.smoke_test else "all_examples",
            }
        )
        write_json(status_path, status)

        train_dataset = prepare_dataset(
            selected_train,
            tokenizer=tokenizer,
            num_proc=1,
            max_seq_length=training_args.max_seq_length,
            prompt_format="chat",
            chat_template_kwargs=chat_kwargs,
        )
        eval_dataset = prepare_dataset(
            selected_eval,
            tokenizer=tokenizer,
            num_proc=1,
            max_seq_length=training_args.max_seq_length,
            prompt_format="chat",
            chat_template_kwargs=chat_kwargs,
        )
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_rows=[],
            args=training_args,
        )
        if config.get("audit_response_masks"):
            from src.utility.training import resolve_response_only_parts
            _, response_part = resolve_response_only_parts(
                training_args.chat_template, training_args.instruction_part, training_args.response_part)
            status["response_mask_audit"] = {
                "train": audit_response_masks(trainer.train_dataset, tokenizer, response_part),
                "validation": audit_response_masks(trainer.eval_dataset, tokenizer, response_part)}
            write_json(status_path, status)
        status["phase"] = "trainer_train"
        write_json(status_path, status)
        result = train_trainer(trainer, training_args)
        status["phase"] = "final_validation"
        write_json(status_path, status)
        eval_metrics = dict(trainer.evaluate())
        status["phase"] = "save_adapter"
        write_json(status_path, status)
        save_adapter_and_tokenizer(
            model,
            tokenizer,
            adapter_dir,
            save_dtype=training_args.save_dtype,
        )
        status["phase"] = "save_training_curves"
        write_json(status_path, status)
        artifacts = save_training_curves(trainer, run_dir)
        completion = {
            "status": "completed",
            "completed_at": utc_now(),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "train_metrics": dict(getattr(result, "metrics", {}) or {}),
            "eval_metrics": eval_metrics,
            "adapter_dir": str(adapter_dir),
            "training_artifacts": artifacts,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        }
        write_json(adapter_dir / "training_complete.json", completion)
        status.update(completion)
        status["phase"] = "completed"
        write_json(status_path, status)
        print(json.dumps(completion, indent=2, default=str))
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(status_path, status)
        raise


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_rows, eval_rows, validation = validate_inputs(config)
    validation["config"] = str(config_path)
    validation["config_sha256"] = sha256_file(config_path)
    print(json.dumps(validation, indent=2))
    if args.mode == "train":
        train(args, config, train_rows, eval_rows, validation)


if __name__ == "__main__":
    main()
