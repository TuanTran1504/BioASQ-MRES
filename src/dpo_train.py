from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
for path_entry in (PROJECT_ROOT, SRC_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))


def prime_unsloth_runtime() -> None:
    """Import Unsloth early so it can patch Transformers before other model loads."""
    try:
        import unsloth  # noqa: F401
    except Exception:
        return


prime_unsloth_runtime()

from datasets import Dataset
from transformers import TrainerCallback
from utility.bioasq_official import evaluate_with_bioasq_java
from utility.adapter_save import save_adapter_and_tokenizer
from utility.config import QUESTION_INSTRUCTIONS
from utility.data import build_output, clean_text, list_record_resources
from utility.eval_models import aggregate_generated_answers, first_model_device
from utility.eval_types import EvalExample


CHAT_ASSISTANT_MARKER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
PLAIN_ANSWER_MARKERS = ("\n# Answer:", "\nAnswer:", "# Answer:", "Answer:")
DEFAULT_POLICY_ADAPTER_NAME = "default"
DEFAULT_REFERENCE_ADAPTER_NAME = "reference"
DEFAULT_DPO_LOSS_TYPE = "sigmoid"
ADADPO_LOSS_TYPE = "adadpo"
ADADPO_DEFAULT_CEILING = 2.0


def _extract_last_rule(instruction: str) -> str:
    lines = [line.strip() for line in instruction.splitlines() if line.strip()]
    rule_lines = [line[1:].strip() for line in lines if line.startswith("-")]
    if rule_lines:
        return rule_lines[-1]
    return lines[-1] if lines else ""


FORMAT_REMINDERS = {
    "list": (
        f"Output reminder: {_extract_last_rule(QUESTION_INSTRUCTIONS['list'])} "
        "Never output [BS] or [ES]. Put exactly one answer item inside each [BI]...[EI] span."
    ),
    "factoid": (
        f"Output reminder: {_extract_last_rule(QUESTION_INSTRUCTIONS['factoid'])} "
        "Never output [BS] or [ES]."
    ),
    "yesno": f"Output reminder: {_extract_last_rule(QUESTION_INSTRUCTIONS['yesno'])}",
    }


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_float_list(value: str | None) -> list[float] | None:
    items = parse_csv_list(value)
    if not items:
        return None
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise ValueError(f"Expected a comma-separated float list, got: {value!r}") from exc


def normalize_loss_types(loss_types: Sequence[str] | str | None) -> list[str]:
    if loss_types is None:
        return [DEFAULT_DPO_LOSS_TYPE]
    if isinstance(loss_types, str):
        normalized = parse_csv_list(loss_types)
    else:
        normalized = [str(item).strip() for item in loss_types if str(item).strip()]
    return normalized or [DEFAULT_DPO_LOSS_TYPE]


def resolve_selection_metric(args: argparse.Namespace) -> str:
    metric = clean_text(getattr(args, "selection_metric", "eval_loss")).lower() or "eval_loss"
    if metric == "auto":
        return "eval_loss"
    return metric


def uses_generated_selection(args: argparse.Namespace) -> bool:
    return resolve_selection_metric(args).startswith("generated_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO adapter from CSE-DPO preference pairs.")
    parser.add_argument("--preference-input", required=True, help="Preference-pair JSONL with prompt/chosen/rejected columns.")
    parser.add_argument(
        "--model-name",
        default="Artifacts/models/runs/20260613-132508-llama32-3b-bioasq/adapter",
        help="Base model or existing adapter to continue from.",
    )
    parser.add_argument(
        "--reference-model-name",
        default=None,
        help=(
            "Optional explicit frozen DPO reference model or adapter path. "
            "When omitted, TRL falls back to its default PEFT reference behavior."
        ),
    )
    parser.add_argument(
        "--precompute-ref-log-probs",
        action="store_true",
        help=(
            "Precompute reference-model log probabilities before training so the "
            "reference model can be released afterwards to save GPU memory."
        ),
    )
    parser.add_argument(
        "--precompute-ref-batch-size",
        type=int,
        default=None,
        help=(
            "Optional batch size used when precomputing reference log probabilities. "
            "Defaults to the train/eval per-device batch size when omitted."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Trainer output directory.")
    parser.add_argument("--save-model-dir", required=True, help="Final adapter/tokenizer output directory.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Optional checkpoint directory to resume from. "
            "Pass an explicit checkpoint path or 'latest' to resume from the newest checkpoint under --output-dir."
        ),
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=192,
        help=(
            "Cap for chosen/rejected completion tokens before collation. "
            "Defaults to 192 to reduce long rejected completions dominating DPO behavior."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument(
        "--split-by",
        choices=["question", "pair"],
        default="question",
        help=(
            "How to create the validation split. 'question' keeps all pairs from the "
            "same question together and is recommended for cleaner dev estimates."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--loss-type",
        default=DEFAULT_DPO_LOSS_TYPE,
        help=(
            "Comma-separated DPO loss types. Examples: 'sigmoid', 'adadpo', or 'adadpo,sft'. "
            "Built-in TRL loss types remain available; this script also adds a custom 'adadpo' loss."
        ),
    )
    parser.add_argument(
        "--loss-weights",
        default=None,
        help=(
            "Optional comma-separated weights aligned with --loss-type, e.g. '1.0,0.2' for "
            "'adadpo,sft'. When omitted, each loss contributes weight 1.0."
        ),
    )
    parser.add_argument(
        "--rpo-alpha",
        type=float,
        default=None,
        help=(
            "Optional chosen-response NLL mixing weight from TRL's RPO support. "
            "Useful when you want extra pressure to promote the chosen response directly."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Optional DPO label smoothing between 0.0 and 0.5. "
            "Applies to standard sigmoid DPO and the custom AdaDPO loss."
        ),
    )
    parser.add_argument(
        "--use-weighting",
        action="store_true",
        help="Enable TRL's WPO-style per-example weighting on top of the chosen loss type(s).",
    )
    parser.add_argument(
        "--reference-free",
        action="store_true",
        help="Ignore the reference model in the pairwise objective and use TRL's reference-free mode.",
    )
    parser.add_argument(
        "--adadpo-ceiling",
        type=float,
        default=ADADPO_DEFAULT_CEILING,
        help=(
            "Clipping ceiling C for the adaptive chosen coefficient in AdaDPO. "
            "The paper uses C=2.0."
        ),
    )
    parser.add_argument(
        "--adadpo-balance-space",
        choices=["reference", "policy"],
        default="reference",
        help=(
            "How AdaDPO forms the adaptive chosen/rejected ratio. "
            "'reference' matches the paper's default Stable AdaDPO implementation."
        ),
    )
    parser.add_argument(
        "--adadpo-length-normalize",
        dest="adadpo_length_normalize",
        action="store_true",
        default=True,
        help="Use Stable AdaDPO's per-token average log-probability ratio when computing the adaptive coefficient.",
    )
    parser.add_argument(
        "--no-adadpo-length-normalize",
        dest="adadpo_length_normalize",
        action="store_false",
        help="Disable length normalization and use the unnormalized AdaDPO ratio instead.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--selection-metric",
        default="eval_loss",
        help=(
            "Checkpoint-selection metric. Use 'eval_loss' for the current DPO objective, "
            "or a generated_* metric to rank checkpoints by downstream BioASQ quality."
        ),
    )
    parser.add_argument(
        "--selection-eval-input",
        nargs="+",
        default=None,
        help=(
            "Prepared or raw gold dataset used to score generated checkpoint outputs by question_id. "
            "Required when --selection-metric uses a generated_* metric."
        ),
    )
    parser.add_argument(
        "--selection-max-seq-length",
        type=int,
        default=None,
        help=(
            "Optional prompt-token cap used only during generated-dev checkpoint selection. "
            "Defaults to --max-seq-length when omitted."
        ),
    )
    parser.add_argument(
        "--selection-max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens for generated-dev checkpoint selection.",
    )
    parser.add_argument(
        "--selection-num-generations",
        type=int,
        default=1,
        help="Number of generations per dev question during checkpoint selection.",
    )
    parser.add_argument(
        "--selection-aggregation-strategy",
        choices=["union", "frequency"],
        default="union",
        help="How to aggregate multiple dev generations for list/factoid checkpoint selection.",
    )
    parser.add_argument(
        "--selection-aggregation-min-frequency",
        type=int,
        default=2,
        help="Minimum sample frequency when --selection-aggregation-strategy frequency is used.",
    )
    parser.add_argument(
        "--selection-do-sample",
        dest="selection_do_sample",
        action="store_true",
        default=False,
        help="Enable sampling during generated-dev checkpoint selection.",
    )
    parser.add_argument(
        "--no-selection-do-sample",
        dest="selection_do_sample",
        action="store_false",
        help="Disable sampling during generated-dev checkpoint selection.",
    )
    parser.add_argument(
        "--selection-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generated-dev checkpoint selection.",
    )
    parser.add_argument(
        "--selection-top-p",
        type=float,
        default=0.9,
        help="Top-p for generated-dev checkpoint selection.",
    )
    parser.add_argument(
        "--summary-reference-mode",
        choices=["first", "best", "all-mean"],
        default="first",
        help="Legacy generation setting; official Phase-B checkpoint scoring excludes summary questions.",
    )
    parser.add_argument(
        "--reinforce-output-format",
        dest="reinforce_output_format",
        action="store_true",
        default=True,
        help=(
            "Repeat a short output-format reminder near the answer stub for list/factoid/yesno prompts. "
            "This helps preserve BioASQ tagging when long prompts are truncated from the left."
        ),
    )
    parser.add_argument(
        "--no-reinforce-output-format",
        dest="reinforce_output_format",
        action="store_false",
        help="Disable prompt-tail output-format reminders.",
    )
    parser.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=None,
        help="Optionally call torch.cuda.empty_cache every N steps to reduce cache fragmentation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help=(
            "Optional early-stopping patience in evaluation calls. Use 0 to disable. "
            "Requires a validation split."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum eval-loss improvement required by early stopping.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-logits-to-keep",
        dest="use_logits_to_keep",
        action="store_true",
        default=True,
        help=(
            "Only compute logits for the completion span instead of the full prompt+completion sequence. "
            "This is usually much more memory-efficient for prompt-heavy DPO."
        ),
    )
    parser.add_argument(
        "--no-use-logits-to-keep",
        dest="use_logits_to_keep",
        action="store_false",
        help="Disable completion-only logits and compute full-sequence logits instead.",
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="Only load models from the local Hugging Face cache.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=None)
    parser.add_argument(
        "--save-dtype",
        choices=["auto", "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
        default="float32",
        help=(
            "Dtype used when saving the final adapter. Defaults to float32 for "
            "stable, portable LoRA artifacts. Use 'auto' to preserve the current "
            "in-memory dtype."
        ),
    )
    args = parser.parse_args()
    args.loss_type = normalize_loss_types(args.loss_type)
    args.loss_weights = parse_csv_float_list(args.loss_weights)

    if args.loss_weights is not None and len(args.loss_weights) != len(args.loss_type):
        raise ValueError(
            "When --loss-weights is provided, it must have the same number of values as --loss-type. "
            f"Received loss_type={args.loss_type!r} and loss_weights={args.loss_weights!r}."
        )
    selection_metric = resolve_selection_metric(args)
    supported_selection_metrics = {
        "eval_loss",
        "generated_mean_f1",
        "generated_primary_score",
        "generated_macro_primary_score",
    }
    if selection_metric not in supported_selection_metrics:
        raise ValueError(
            f"Unsupported --selection-metric {args.selection_metric!r}. "
            f"Choose from {sorted(supported_selection_metrics)}."
        )
    if uses_generated_selection(args) and not args.selection_eval_input:
        raise ValueError(
            "--selection-eval-input is required when --selection-metric uses a generated_* metric."
        )
    if args.selection_num_generations <= 0:
        raise ValueError("--selection-num-generations must be a positive integer.")
    if args.selection_aggregation_min_frequency <= 0:
        raise ValueError("--selection-aggregation-min-frequency must be a positive integer.")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError(f"--label-smoothing must be in [0.0, 0.5), got {args.label_smoothing}.")
    if args.adadpo_ceiling <= 0.0:
        raise ValueError(f"--adadpo-ceiling must be > 0, got {args.adadpo_ceiling}.")
    return args


def infer_answer_style(chosen: str, rejected: str) -> str | None:
    combined = f"{chosen}\n{rejected}"
    if "[BI]" in combined and "[EI]" in combined:
        return "list"
    if "[BE]" in combined and "[EE]" in combined:
        return "factoid"

    chosen_normalized = chosen.strip().lower()
    rejected_normalized = rejected.strip().lower()
    yesno_values = {"yes", "no"}
    if chosen_normalized in yesno_values and rejected_normalized in yesno_values:
        return "yesno"
    return None


def inject_prompt_tail_reminder(prompt: str, reminder: str) -> tuple[str, str]:
    if not reminder.strip():
        return prompt, "unchanged"
    if reminder in prompt:
        return prompt, "already_present"

    for marker in (CHAT_ASSISTANT_MARKER, *PLAIN_ANSWER_MARKERS):
        marker_index = prompt.rfind(marker)
        if marker_index == -1:
            continue
        prefix = prompt[:marker_index].rstrip()
        suffix = prompt[marker_index:]
        separator = "\n\n" if prefix else ""
        return f"{prefix}{separator}{reminder}{suffix}", "inserted_before_answer"

    separator = "\n\n" if prompt.rstrip() else ""
    return f"{prompt.rstrip()}{separator}{reminder}", "appended_to_end"


def read_pairs(path: Path, reinforce_output_format: bool = True) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    answer_style_counts = {key: 0 for key in ["list", "factoid", "yesno", "other"]}
    reminder_insert_counts = {key: 0 for key in ["inserted_before_answer", "appended_to_end", "already_present", "unchanged"]}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            chosen = str(row.get("chosen") or "").strip()
            rejected = str(row.get("rejected") or "").strip()
            if prompt and chosen and rejected:
                answer_style = infer_answer_style(chosen, rejected) or "other"
                answer_style_counts[answer_style] += 1
                reminder_status = "unchanged"
                if reinforce_output_format:
                    reminder = FORMAT_REMINDERS.get(answer_style)
                    if reminder:
                        prompt, reminder_status = inject_prompt_tail_reminder(prompt, reminder)
                reminder_insert_counts[reminder_status] += 1
                rows.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "pair_id": str(row.get("pair_id") or ""),
                        "pair_type": str(row.get("pair_type") or ""),
                        "question_id": str(row.get("question_id") or "").strip(),
                    }
                )
    if not rows:
        raise ValueError(f"No usable preference rows found in {path}")
    summary = {
        "rows": len(rows),
        "reinforce_output_format": bool(reinforce_output_format),
        "answer_style_counts": answer_style_counts,
        "reminder_insert_counts": reminder_insert_counts,
    }
    return rows, summary


def _row_group_key(row: dict[str, str], row_index: int) -> str:
    question_id = str(row.get("question_id") or "").strip()
    if question_id:
        return question_id

    prompt = str(row.get("prompt") or "").strip()
    if prompt:
        return prompt

    pair_id = str(row.get("pair_id") or "").strip()
    if pair_id:
        return pair_id

    return f"row-{row_index}"


def split_rows(
    rows: list[dict[str, str]],
    validation_ratio: float,
    seed: int,
    split_by: str = "question",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if validation_ratio <= 0 or len(rows) < 2:
        return rows, []

    normalized_split_by = str(split_by or "question").strip().lower()
    if normalized_split_by == "pair":
        dataset = Dataset.from_list(rows).train_test_split(test_size=validation_ratio, seed=seed)
        return list(dataset["train"]), list(dataset["test"])

    if normalized_split_by != "question":
        raise ValueError(f"Unsupported split mode: {split_by}")

    group_keys = [_row_group_key(row, row_index=index) for index, row in enumerate(rows)]
    unique_group_keys = list(dict.fromkeys(group_keys))
    if len(unique_group_keys) < 2:
        return rows, []

    eval_group_count = int(round(len(unique_group_keys) * validation_ratio))
    eval_group_count = max(1, eval_group_count)
    eval_group_count = min(len(unique_group_keys) - 1, eval_group_count)

    rng = random.Random(seed)
    shuffled_group_keys = list(unique_group_keys)
    rng.shuffle(shuffled_group_keys)
    eval_group_keys = set(shuffled_group_keys[:eval_group_count])

    train_rows: list[dict[str, str]] = []
    eval_rows: list[dict[str, str]] = []
    for row, group_key in zip(rows, group_keys):
        if group_key in eval_group_keys:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def count_unique_questions(rows: list[dict[str, str]]) -> int:
    keys = {_row_group_key(row, row_index=index) for index, row in enumerate(rows)}
    return len(keys)


def build_selection_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.max_seq_length = int(
        getattr(args, "selection_max_seq_length", 0)
        or 0
    )
    eval_args.max_new_tokens = int(getattr(args, "selection_max_new_tokens", 512) or 512)
    eval_args.num_generations = int(getattr(args, "selection_num_generations", 1) or 1)
    eval_args.aggregation_strategy = str(
        getattr(args, "selection_aggregation_strategy", "union") or "union"
    )
    eval_args.aggregation_min_frequency = int(
        getattr(args, "selection_aggregation_min_frequency", 2) or 2
    )
    eval_args.do_sample = bool(getattr(args, "selection_do_sample", False))
    eval_args.temperature = float(getattr(args, "selection_temperature", 0.7) or 0.7)
    eval_args.top_p = float(getattr(args, "selection_top_p", 0.9) or 0.9)
    eval_args.summary_reference_mode = str(
        getattr(args, "summary_reference_mode", "first") or "first"
    )
    eval_args.use_cache = False
    eval_args.empty_cuda_cache_per_generation = True
    eval_args.bioasq_java_jar = str(
        PROJECT_ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    )
    eval_args.bioasq_java_heap = "512m"
    eval_args.bioasq_java_version = 5
    return eval_args


def resolve_selection_eval_paths(path_values: Sequence[str] | None) -> list[Path]:
    resolved_paths: list[Path] = []
    for raw_value in path_values or []:
        candidate = Path(str(raw_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Selection eval input does not exist: {candidate}")
        if candidate.is_dir():
            child_paths = [
                path
                for path in sorted(candidate.glob("*.json"))
                if not path.name.endswith(".summary.json")
            ]
            if not child_paths:
                raise FileNotFoundError(
                    f"No .json files found under selection eval input directory: {candidate}"
                )
            resolved_paths.extend(child_paths)
            continue
        resolved_paths.append(candidate)
    if not resolved_paths:
        raise ValueError("No selection eval inputs were resolved.")
    return resolved_paths


def build_gold_output_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_summary_answers=5,
        max_factoid_answers=5,
        max_list_items=100,
    )


def load_selection_gold_examples(paths: Sequence[Path]) -> dict[str, EvalExample]:
    gold_examples: dict[str, EvalExample] = {}
    output_args = build_gold_output_args()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            for index, row in enumerate(payload):
                if not isinstance(row, dict):
                    continue
                question_id = clean_text(row.get("id", "")) or f"{path.name}:{index}"
                question_type = clean_text(row.get("type", "")).lower()
                body = clean_text(row.get("input_1", ""))
                gold_output = clean_text(row.get("output", ""))
                instruction = clean_text(row.get("instruction", "")) or QUESTION_INSTRUCTIONS.get(question_type, "")
                if not question_type or not gold_output:
                    continue
                gold_examples.setdefault(
                    question_id,
                    EvalExample(
                        question_id=question_id,
                        question_type=question_type,
                        body=body,
                        instruction=instruction,
                        resources=tuple(resource for resource in list_record_resources(row) if clean_text(resource)),
                        gold_output=gold_output,
                        source_path=str(path),
                        raw_question=None,
                    ),
                )
            continue

        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"Unsupported selection eval input format: {path}")

        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                continue
            question_id = clean_text(question.get("id", "")) or f"{path.name}:{index}"
            question_type = clean_text(question.get("type", "")).lower()
            body = clean_text(question.get("body", ""))
            gold_output = clean_text(build_output(question, output_args))
            instruction = QUESTION_INSTRUCTIONS.get(question_type, "")
            if not question_type or not gold_output:
                continue
            gold_examples.setdefault(
                question_id,
                EvalExample(
                    question_id=question_id,
                    question_type=question_type,
                    body=body,
                    instruction=instruction,
                    resources=(),
                    gold_output=gold_output,
                    source_path=str(path),
                    raw_question=question,
                ),
            )

    if not gold_examples:
        raise ValueError("No gold evaluation examples were loaded from --selection-eval-input.")
    return gold_examples


def build_generated_selection_rows(
    eval_rows: Sequence[dict[str, str]],
    gold_examples: dict[str, EvalExample],
) -> tuple[list[dict[str, Any]], list[str]]:
    selection_rows: list[dict[str, Any]] = []
    missing_question_ids: list[str] = []
    seen_keys: set[str] = set()

    for index, row in enumerate(eval_rows):
        prompt = str(row.get("prompt") or "").strip()
        question_id = clean_text(row.get("question_id", ""))
        if not prompt:
            continue

        dedupe_key = question_id or _row_group_key(row, row_index=index)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        if not question_id or question_id not in gold_examples:
            if question_id:
                missing_question_ids.append(question_id)
            continue

        selection_rows.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "example": gold_examples[question_id],
            }
        )

    return selection_rows, sorted(set(missing_question_ids))


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str | None) -> str | None:
    requested = str(resume_from_checkpoint or "").strip()
    if not requested:
        return None

    if requested.lower() == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint_path = get_last_checkpoint(output_dir)
        return checkpoint_path

    if requested.lower() == "latest":
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint_path = get_last_checkpoint(output_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found under output_dir={output_dir!r} to resume from.")
        return checkpoint_path

    checkpoint_path = Path(requested)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_dtype(dtype_name: str | None) -> Any:
    if dtype_name in {None, ""}:
        return None
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]


def normalize_optional_model_name(model_name: str | None) -> str | None:
    normalized = str(model_name or "").strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "off", "false", "null"}:
        return None
    return normalized


def require_cuda_for_unsloth() -> None:
    import torch

    if torch.cuda.is_available():
        return

    message = """
DPO training with Unsloth needs a visible CUDA GPU, but torch reports:
  cuda_available = False
  cuda_device_count = 0

Please run this script in a GPU-enabled shell/container, then verify with:
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
  nvidia-smi

If you are already on a GPU machine, this usually means the NVIDIA driver,
CUDA runtime, or container GPU passthrough is not visible to Python.
""".strip()
    print(message, file=sys.stderr)
    raise SystemExit(2)


def load_existing_adapter_config(model_name: str) -> dict[str, Any] | None:
    candidate_paths = []
    model_path = Path(model_name)
    if model_path.exists():
        if model_path.is_dir():
            candidate_paths.append(model_path / "adapter_config.json")
            candidate_paths.append(model_path / "adapter" / "adapter_config.json")
        else:
            candidate_paths.append(model_path)

    for config_path in candidate_paths:
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid adapter config at {config_path}") from exc
    return None


def adapter_base_model_name(model_name: str | None) -> str | None:
    normalized_name = normalize_optional_model_name(model_name)
    if normalized_name is None:
        return None
    adapter_config = load_existing_adapter_config(normalized_name)
    if adapter_config is None:
        return None
    base_model_name = str(adapter_config.get("base_model_name_or_path") or "").strip()
    return base_model_name or None


def resolve_reference_strategy(args: argparse.Namespace) -> dict[str, Any]:
    reference_model_name = normalize_optional_model_name(args.reference_model_name)
    if reference_model_name is None:
        return {
            "mode": "trl_default",
            "reference_model_name": None,
            "policy_adapter_name": None,
            "reference_adapter_name": None,
        }

    policy_base_model_name = adapter_base_model_name(args.model_name)
    reference_base_model_name = adapter_base_model_name(reference_model_name)
    if reference_base_model_name is not None:
        if policy_base_model_name is None or policy_base_model_name == reference_base_model_name:
            return {
                "mode": "shared_adapter",
                "reference_model_name": reference_model_name,
                "policy_adapter_name": DEFAULT_POLICY_ADAPTER_NAME,
                "reference_adapter_name": DEFAULT_REFERENCE_ADAPTER_NAME,
            }

    return {
        "mode": "separate_model",
        "reference_model_name": reference_model_name,
        "policy_adapter_name": None,
        "reference_adapter_name": None,
    }


def resolve_lora_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
        "use_rslora": False,
        "loftq_config": None,
    }

    existing_adapter_config = load_existing_adapter_config(args.model_name)
    if existing_adapter_config is None:
        return config

    adapted = {
        "r": int(existing_adapter_config.get("r", config["r"])),
        "lora_alpha": int(existing_adapter_config.get("lora_alpha", config["lora_alpha"])),
        "lora_dropout": float(existing_adapter_config.get("lora_dropout", config["lora_dropout"])),
        "target_modules": list(existing_adapter_config.get("target_modules") or config["target_modules"]),
        "bias": str(existing_adapter_config.get("bias", config["bias"])),
        "use_rslora": bool(existing_adapter_config.get("use_rslora", config["use_rslora"])),
        "loftq_config": existing_adapter_config.get("loftq_config", config["loftq_config"]),
    }

    if any(adapted[key] != config[key] for key in adapted):
        print(
            "Detected existing LoRA adapter config at model path; "
            "reusing its LoRA settings for continued DPO training.",
            flush=True,
        )
    return adapted


def generate_answer_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    question_type: str,
    args: argparse.Namespace,
) -> tuple[str, list[str]]:
    import gc
    import torch

    def generate_once() -> str:
        encode_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        max_seq_length = int(getattr(args, "max_seq_length", 0) or 0)
        previous_truncation_side = getattr(tokenizer, "truncation_side", None)
        try:
            if max_seq_length > 0:
                # Keep the assistant-generation prefix at the tail of the prompt
                # when long prompts need to be clipped.
                encode_kwargs.update({"truncation": True, "max_length": max_seq_length})
                if previous_truncation_side is not None:
                    tokenizer.truncation_side = "left"
            encoded = tokenizer(prompt, **encode_kwargs)
        finally:
            if previous_truncation_side is not None:
                tokenizer.truncation_side = previous_truncation_side
        device = first_model_device(model)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "do_sample": bool(args.do_sample),
            "use_cache": bool(getattr(args, "use_cache", True)),
        }
        if args.do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = clean_text(decoded)
        answer = cleaned[7:].strip() if cleaned.lower().startswith("answer:") else cleaned

        del generated_ids
        del output_ids
        del encoded
        gc.collect()

        if bool(getattr(args, "empty_cuda_cache_per_generation", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    num_generations = max(1, int(getattr(args, "num_generations", 1) or 1))
    samples = [generate_once() for _ in range(num_generations)]
    if num_generations == 1:
        return samples[0], samples
    return aggregate_generated_answers(samples, question_type=question_type, args=args), samples


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    lora_config = resolve_lora_config(args)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=resolve_dtype(args.dtype),
        load_in_4bit=not args.no_4bit,
        local_files_only=args.local_files_only,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=lora_config["use_rslora"],
        loftq_config=lora_config["loftq_config"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")
    return model, tokenizer


def maybe_attach_reference_adapter(model: Any, args: argparse.Namespace, reference_strategy: dict[str, Any]) -> Any:
    if reference_strategy.get("mode") != "shared_adapter":
        return model

    reference_model_name = str(reference_strategy["reference_model_name"])
    reference_adapter_name = str(reference_strategy["reference_adapter_name"] or DEFAULT_REFERENCE_ADAPTER_NAME)
    policy_adapter_name = str(reference_strategy["policy_adapter_name"] or DEFAULT_POLICY_ADAPTER_NAME)

    if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
        raise TypeError(
            "The loaded policy model does not expose PEFT adapter-management APIs, "
            "so shared-base reference adapters are not available for this run."
        )

    loaded_adapter_names = set(getattr(model, "peft_config", {}).keys())
    if reference_adapter_name not in loaded_adapter_names:
        model.load_adapter(
            reference_model_name,
            adapter_name=reference_adapter_name,
            is_trainable=False,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )

    model.set_adapter(policy_adapter_name, inference_mode=False)
    return model


def freeze_reference_model(model: Any) -> Any:
    if hasattr(model, "eval"):
        model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_reference_model(args: argparse.Namespace, reference_strategy: dict[str, Any]) -> Any | None:
    if reference_strategy.get("mode") != "separate_model":
        return None

    reference_model_name = str(reference_strategy["reference_model_name"])
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name=reference_model_name,
        max_seq_length=args.max_seq_length,
        dtype=resolve_dtype(args.dtype),
        load_in_4bit=not args.no_4bit,
        local_files_only=args.local_files_only,
    )
    return freeze_reference_model(model)


def resolve_dpo_trainer_class(loss_types: Sequence[str]) -> type[Any]:
    from trl import DPOTrainer

    if ADADPO_LOSS_TYPE not in loss_types:
        return DPOTrainer

    cached = getattr(resolve_dpo_trainer_class, "_cached_adadpo_trainer", None)
    if cached is not None:
        return cached

    import torch
    import torch.nn.functional as F

    class AdaDPOTrainer(DPOTrainer):
        def concatenated_forward(
            self,
            model: Any,
            batch: dict[str, Any],
            is_ref_model: bool = False,
        ) -> dict[str, Any]:
            output = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
            chosen_attention_mask = batch.get("chosen_attention_mask")
            rejected_attention_mask = batch.get("rejected_attention_mask")
            if chosen_attention_mask is not None and rejected_attention_mask is not None:
                output["chosen_lengths"] = chosen_attention_mask.sum(dim=1).clamp_min(1)
                output["rejected_lengths"] = rejected_attention_mask.sum(dim=1).clamp_min(1)
            return output

        def dpo_loss(
            self,
            chosen_logps: torch.FloatTensor,
            rejected_logps: torch.FloatTensor,
            ref_chosen_logps: torch.FloatTensor,
            ref_rejected_logps: torch.FloatTensor,
            loss_type: str = DEFAULT_DPO_LOSS_TYPE,
            model_output: dict[str, torch.FloatTensor] | None = None,
        ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
            if loss_type != ADADPO_LOSS_TYPE:
                return super().dpo_loss(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    loss_type,
                    model_output,
                )

            if model_output is None:
                raise ValueError("AdaDPO requires model_output so completion lengths are available.")

            device = self.accelerator.device
            chosen_logps = chosen_logps.to(device)
            rejected_logps = rejected_logps.to(device)

            if self.reference_free:
                ref_chosen_logps = torch.zeros_like(chosen_logps)
                ref_rejected_logps = torch.zeros_like(rejected_logps)
            else:
                ref_chosen_logps = ref_chosen_logps.to(device)
                ref_rejected_logps = ref_rejected_logps.to(device)

            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps

            balance_space = str(getattr(self.args, "adadpo_balance_space", "reference"))
            if balance_space == "policy":
                chosen_balance_terms = chosen_logps.detach()
                rejected_balance_terms = rejected_logps.detach()
            else:
                chosen_balance_terms = chosen_logratios.detach()
                rejected_balance_terms = rejected_logratios.detach()

            if bool(getattr(self.args, "adadpo_length_normalize", True)):
                chosen_lengths = model_output.get("chosen_lengths")
                rejected_lengths = model_output.get("rejected_lengths")
                if chosen_lengths is None or rejected_lengths is None:
                    raise ValueError(
                        "AdaDPO length normalization requires chosen/rejected completion lengths in model_output."
                    )
                chosen_lengths = chosen_lengths.to(device).float().clamp_min(1.0).detach()
                rejected_lengths = rejected_lengths.to(device).float().clamp_min(1.0).detach()
                chosen_balance_terms = chosen_balance_terms / chosen_lengths
                rejected_balance_terms = rejected_balance_terms / rejected_lengths

            adaptive_log_ratio = chosen_balance_terms - rejected_balance_terms
            beta_ceiling = float(getattr(self.args, "adadpo_ceiling", ADADPO_DEFAULT_CEILING))
            beta_w = self.beta * torch.exp(torch.clamp(adaptive_log_ratio, max=math.log(beta_ceiling)))

            delta = beta_w * chosen_logratios - self.beta * rejected_logratios
            losses = (
                -F.logsigmoid(delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-delta) * self.label_smoothing
            )

            chosen_rewards = (beta_w * chosen_logratios).detach()
            rejected_rewards = (self.beta * rejected_logratios).detach()
            return losses, chosen_rewards, rejected_rewards

    resolve_dpo_trainer_class._cached_adadpo_trainer = AdaDPOTrainer
    return AdaDPOTrainer


def build_dpo_config(args: argparse.Namespace, has_eval: bool, reference_strategy: dict[str, Any]) -> Any:
    from trl import DPOConfig
    from unsloth import is_bfloat16_supported

    reference_mode = str(reference_strategy.get("mode") or "trl_default")
    selection_metric = resolve_selection_metric(args)
    should_run_trainer_eval = has_eval and selection_metric == "eval_loss"
    signature = inspect.signature(DPOConfig.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": "none",
        "seed": args.seed,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "beta": args.beta,
        "loss_type": args.loss_type,
        "loss_weights": args.loss_weights,
        "rpo_alpha": args.rpo_alpha,
        "label_smoothing": args.label_smoothing,
        "use_weighting": args.use_weighting,
        "reference_free": args.reference_free,
        "max_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "remove_unused_columns": False,
        "use_logits_to_keep": args.use_logits_to_keep,
    }
    if "force_use_ref_model" in signature.parameters:
        kwargs["force_use_ref_model"] = reference_mode == "separate_model"
    if "model_adapter_name" in signature.parameters and reference_mode == "shared_adapter":
        kwargs["model_adapter_name"] = reference_strategy.get("policy_adapter_name") or DEFAULT_POLICY_ADAPTER_NAME
    if "ref_adapter_name" in signature.parameters and reference_mode == "shared_adapter":
        kwargs["ref_adapter_name"] = reference_strategy.get("reference_adapter_name") or DEFAULT_REFERENCE_ADAPTER_NAME
    if "precompute_ref_log_probs" in signature.parameters:
        kwargs["precompute_ref_log_probs"] = args.precompute_ref_log_probs
    if args.precompute_ref_batch_size is not None and "precompute_ref_batch_size" in signature.parameters:
        kwargs["precompute_ref_batch_size"] = args.precompute_ref_batch_size
    if args.torch_empty_cache_steps is not None and "torch_empty_cache_steps" in signature.parameters:
        kwargs["torch_empty_cache_steps"] = args.torch_empty_cache_steps
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps" if should_run_trainer_eval else "no"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps" if should_run_trainer_eval else "no"
    if should_run_trainer_eval and "eval_steps" in signature.parameters:
        kwargs["eval_steps"] = args.save_steps
    if should_run_trainer_eval:
        if "load_best_model_at_end" in signature.parameters:
            kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in signature.parameters:
            kwargs["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in signature.parameters:
            kwargs["greater_is_better"] = False

    return DPOConfig(**{key: value for key, value in kwargs.items() if key in signature.parameters})


class GeneratedDevSelectionCallback(TrainerCallback):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        eval_rows: Sequence[dict[str, str]],
    ) -> None:
        self.args = args
        self.eval_args = build_selection_eval_args(args)
        self.selection_metric = resolve_selection_metric(args)
        self.gold_examples = load_selection_gold_examples(
            resolve_selection_eval_paths(getattr(args, "selection_eval_input", None))
        )
        self.eval_rows, self.missing_question_ids = build_generated_selection_rows(
            eval_rows,
            self.gold_examples,
        )
        if not self.eval_rows:
            raise ValueError(
                "Generated checkpoint selection could not match any DPO eval questions to "
                "--selection-eval-input by question_id."
            )

        self.output_root = Path(args.output_dir).resolve().parent / "generated_dev_selection"
        self.best_model_dir = Path(args.save_model_dir)
        self.history_path = self.output_root / "history.json"
        self.best_summary_path = self.output_root / "best_summary.json"
        self.summary_by_step: dict[int, dict[str, Any]] = {}
        self.best_metric: float | None = None
        self.best_model_path: str | None = None
        self.best_summary: dict[str, Any] | None = None
        self.bad_rounds = 0
        self.last_scored_step: int | None = None

        matched_types = sorted({row["example"].question_type for row in self.eval_rows})
        if self.selection_metric == "generated_mean_f1" and "list" not in matched_types:
            raise ValueError(
                "--selection-metric generated_mean_f1 requires list questions in the matched eval split."
            )

        print(
            "Generated dev checkpoint selection: "
            f"matched_questions={len(self.eval_rows)} | "
            f"gold_pool={len(self.gold_examples)} | "
            f"metric={self.selection_metric}",
            flush=True,
        )
        if self.missing_question_ids:
            print(
                "Generated dev checkpoint selection skipped questions missing from "
                f"--selection-eval-input: {len(self.missing_question_ids)}",
                flush=True,
            )

    def _cleanup_cuda(self) -> None:
        gc.collect()
        try:
            import torch._dynamo as torch_dynamo

            torch_dynamo.reset()
        except Exception:
            pass
        try:
            import torch

            if not torch.cuda.is_available():
                return
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass
        except Exception:
            return

    def _set_inference_mode(self, model: Any) -> None:
        bound_handler = getattr(model, "for_inference", None)
        if callable(bound_handler):
            try:
                bound_handler()
                return
            except Exception:
                pass

        try:
            from unsloth import FastLanguageModel
        except Exception:
            return

        class_handler = getattr(FastLanguageModel, "for_inference", None)
        if callable(class_handler):
            try:
                class_handler(model)
            except Exception:
                pass

    def _set_training_mode(self, model: Any) -> None:
        bound_handler = getattr(model, "for_training", None)
        if callable(bound_handler):
            try:
                bound_handler()
                return
            except Exception:
                pass

        try:
            from unsloth import FastLanguageModel
        except Exception:
            return

        class_handler = getattr(FastLanguageModel, "for_training", None)
        if callable(class_handler):
            try:
                class_handler(model)
            except Exception:
                pass

    def _extract_metric(self, aggregate: dict[str, Any]) -> float | None:
        if self.selection_metric == "generated_mean_f1":
            return (
                aggregate.get("by_type", {})
                .get("list", {})
                .get("metrics", {})
                .get("mean_f1")
            )
        if self.selection_metric == "generated_primary_score":
            return aggregate.get("overall_average_primary_score")
        if self.selection_metric == "generated_macro_primary_score":
            return aggregate.get("overall_macro_average_primary_score")
        raise ValueError(f"Unsupported generated selection metric: {self.selection_metric}")

    def _evaluate_model(self, model: Any, tokenizer: Any) -> dict[str, Any]:
        prediction_rows = []
        was_training = bool(getattr(model, "training", False))
        self._set_inference_mode(model)
        if hasattr(model, "eval"):
            model.eval()
        self._cleanup_cuda()

        try:
            for selection_row in self.eval_rows:
                example = selection_row["example"]
                prediction, generation_samples = generate_answer_from_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=selection_row["prompt"],
                    question_type=example.question_type,
                    args=self.eval_args,
                )
                prediction_rows.append(
                    {
                        "question_id": example.question_id,
                        "question_type": example.question_type,
                        "body": example.body,
                        "source_path": example.source_path,
                        "prediction": prediction,
                        "generation_samples": generation_samples,
                        "gold_output": example.gold_output,
                    }
                )
        finally:
            self._set_training_mode(model)
            if was_training and hasattr(model, "train"):
                model.train()
            self._cleanup_cuda()

        examples_by_key = {
            (clean_text(row["example"].question_id), clean_text(row["example"].question_type).lower()): row["example"]
            for row in self.eval_rows
        }
        official = evaluate_with_bioasq_java(
            prediction_rows=prediction_rows,
            examples_by_key=examples_by_key,
            model_label="dpo-generated-dev-selection",
            model_dir=self.output_root / "official_eval_tmp",
            args=self.eval_args,
            include_per_question=True,
        )
        official_by_id = {row["question_id"]: row for row in official["per_question"]}
        for row in prediction_rows:
            row["score"] = official_by_id[row["question_id"]]
        aggregate = official["aggregate"]
        metric_value = self._extract_metric(aggregate)
        return {
            "metric_name": self.selection_metric,
            "metric_value": metric_value,
            "aggregate": aggregate,
            "question_count": len(prediction_rows),
            "scoring_backend": "bioasq_java",
        }

    def _log_history(self, state: Any, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate"]
        list_metrics = (
            aggregate.get("by_type", {})
            .get("list", {})
            .get("metrics", {})
        )
        log_entry: dict[str, Any] = {
            "epoch": state.epoch,
            "step": state.global_step,
            "generated_selection_metric": summary["metric_name"],
            summary["metric_name"]: summary["metric_value"],
            "generated_overall_average_primary_score": aggregate.get("overall_average_primary_score"),
            "generated_overall_macro_average_primary_score": aggregate.get("overall_macro_average_primary_score"),
            "generated_mean_precision": list_metrics.get("mean_precision"),
            "generated_mean_recall": list_metrics.get("mean_recall"),
            "generated_mean_f1": list_metrics.get("mean_f1"),
        }
        state.log_history.append(log_entry)

    def _persist_history(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        ordered_history = [
            self.summary_by_step[step]
            for step in sorted(self.summary_by_step)
        ]
        self.history_path.write_text(
            json.dumps(ordered_history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.best_summary is not None:
            self.best_summary_path.write_text(
                json.dumps(self.best_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _save_best_model(self, model: Any, tokenizer: Any) -> None:
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        save_adapter_and_tokenizer(
            model,
            tokenizer,
            self.best_model_dir,
            save_dtype=getattr(self.args, "save_dtype", "float32"),
        )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self.eval_rows or state.global_step == self.last_scored_step:
            return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            return control

        summary = self._evaluate_model(model=model, tokenizer=tokenizer)
        summary.update(
            {
                "epoch": state.epoch,
                "step": state.global_step,
            }
        )
        self.summary_by_step[int(state.global_step)] = summary
        self.last_scored_step = int(state.global_step)
        self._log_history(state, summary)
        self._persist_history()

        metric_value = summary.get("metric_value")
        improved = (
            isinstance(metric_value, (int, float))
            and (
                self.best_metric is None
                or float(metric_value) > float(self.best_metric) + float(getattr(self.args, "early_stopping_threshold", 0.0) or 0.0)
            )
        )

        if improved:
            self.best_metric = float(metric_value)
            self.best_model_path = str(self.best_model_dir)
            self.best_summary = summary
            self.bad_rounds = 0
            self._save_best_model(model=model, tokenizer=tokenizer)
            self._cleanup_cuda()
            self._persist_history()
            state.best_metric = float(metric_value)
            state.best_model_checkpoint = str(self.best_model_dir)
            print(
                f"Generated dev selection improved to {metric_value:.4f} "
                f"at step {state.global_step} (epoch {state.epoch}).",
                flush=True,
            )
        else:
            self.bad_rounds += 1
            print(
                "Generated dev selection did not improve "
                f"(best={self.best_metric}, current={metric_value}); "
                f"patience={self.bad_rounds}/{getattr(self.args, 'early_stopping_patience', 0)}.",
                flush=True,
            )

        if getattr(self.args, "early_stopping_patience", 0) > 0 and self.bad_rounds >= self.args.early_stopping_patience:
            control.should_training_stop = True
            print(
                f"Stopping early because {summary['metric_name']} has not improved "
                f"for {self.bad_rounds} checkpoint(s).",
                flush=True,
            )

        self._cleanup_cuda()
        return control

    @property
    def artifact_paths(self) -> dict[str, str]:
        paths = {}
        if self.history_path.exists():
            paths["generated_dev_history_json"] = str(self.history_path)
        if self.best_summary_path.exists():
            paths["generated_dev_best_summary_json"] = str(self.best_summary_path)
        return paths


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    eval_rows: Sequence[dict[str, str]],
    args: argparse.Namespace,
    reference_strategy: dict[str, Any],
    ref_model: Any | None = None,
) -> Any:
    dpo_args = build_dpo_config(
        args,
        has_eval=eval_dataset is not None and len(eval_dataset) > 0,
        reference_strategy=reference_strategy,
    )
    trainer_class = resolve_dpo_trainer_class(args.loss_type)
    signature = inspect.signature(trainer_class.__init__)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    if "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    if "ref_model" in signature.parameters:
        kwargs["ref_model"] = ref_model
    trainer = trainer_class(**kwargs)

    trainer.generated_metric_callback = None
    if uses_generated_selection(args) and eval_rows:
        callback = GeneratedDevSelectionCallback(args=args, eval_rows=eval_rows)
        trainer.add_callback(callback)
        trainer.generated_metric_callback = callback
    elif args.early_stopping_patience > 0 and eval_dataset is not None and len(eval_dataset) > 0:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    return trainer


def maybe_precompute_and_release_reference(trainer: Any) -> None:
    if not bool(getattr(trainer, "precompute_ref_log_probs", False)):
        return

    print("Precomputing reference log probabilities for the training split...", flush=True)
    trainer.get_train_dataloader()

    eval_dataset = getattr(trainer, "eval_dataset", None)
    if eval_dataset is not None and len(eval_dataset) > 0:
        print("Precomputing reference log probabilities for the evaluation split...", flush=True)
        trainer.get_eval_dataloader()

    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model = None
        print("Released reference model after precomputing reference log probabilities.", flush=True)

    try:
        trainer.accelerator.free_memory()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass
    except Exception:
        pass
    gc.collect()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_training_curves(trainer: Any, save_dir: Path) -> dict[str, str]:
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    save_json(save_dir / "training_log_history.json", log_history)
    if not log_history:
        return {}

    import pandas as pd

    frame = pd.DataFrame(log_history)
    frame.to_csv(save_dir / "training_metrics.csv", index=False)

    saved_paths: dict[str, str] = {
        "log_history_json": str(save_dir / "training_log_history.json"),
        "metrics_csv": str(save_dir / "training_metrics.csv"),
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return saved_paths

    plotted_sections = []
    if "loss" in frame.columns or "eval_loss" in frame.columns:
        plotted_sections.append(
            (
                "Loss",
                [
                    ("loss", "train loss"),
                    ("eval_loss", "eval loss"),
                ],
            )
        )
    if "learning_rate" in frame.columns:
        plotted_sections.append(
            (
                "Learning Rate",
                [
                    ("learning_rate", "learning rate"),
                ],
            )
        )
    generated_metric_columns = [
        column
        for column in (
            "generated_mean_f1",
            "generated_mean_precision",
            "generated_mean_recall",
            "generated_overall_average_primary_score",
            "generated_overall_macro_average_primary_score",
        )
        if column in frame.columns
    ]
    if generated_metric_columns:
        plotted_sections.append(
            (
                "Generated Dev Metrics",
                [(column, column.replace("_", " ")) for column in generated_metric_columns],
            )
        )

    reward_metric_keys = [
        key
        for key in frame.columns
        if isinstance(key, str) and ("reward" in key.lower() or "accuracy" in key.lower())
    ]
    if reward_metric_keys:
        plotted_sections.append(
            (
                "DPO Metrics",
                [(key, key) for key in sorted(reward_metric_keys)],
            )
        )

    if not plotted_sections:
        return saved_paths

    figure, axes = plt.subplots(len(plotted_sections), 1, figsize=(10, 4 * len(plotted_sections)), squeeze=False)
    x_steps = pd.to_numeric(frame.get("step"), errors="coerce") if "step" in frame.columns else None
    fallback_x = pd.Series(range(len(frame)))

    for axis, (title, metric_specs) in zip(axes[:, 0], plotted_sections):
        plotted_any = False
        for metric_key, metric_label in metric_specs:
            if metric_key not in frame.columns:
                continue
            y_values = pd.to_numeric(frame[metric_key], errors="coerce")
            valid_mask = y_values.notna()
            if not valid_mask.any():
                continue
            x_values = x_steps[valid_mask] if x_steps is not None else fallback_x[valid_mask]
            axis.plot(x_values, y_values[valid_mask], marker="o", linewidth=1.5, markersize=3, label=metric_label)
            plotted_any = True

        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        if plotted_any:
            axis.legend()
        else:
            axis.set_visible(False)

    figure.tight_layout()
    figure_path = save_dir / "training_curves.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved_paths["training_curves_png"] = str(figure_path)
    return saved_paths


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    args = parse_args()
    reference_strategy = resolve_reference_strategy(args)

    rows, preparation_summary = read_pairs(
        Path(args.preference_input),
        reinforce_output_format=args.reinforce_output_format,
    )
    if args.max_train_samples is not None:
        rows = rows[: args.max_train_samples]
        preparation_summary["rows_after_max_train_samples"] = len(rows)
    train_rows, eval_rows = split_rows(
        rows,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        split_by=args.split_by,
    )

    print(f"Prepared {len(train_rows):,} DPO training pairs.", flush=True)
    print(f"Prepared {len(eval_rows):,} DPO evaluation pairs.", flush=True)
    print(f"Training questions: {count_unique_questions(train_rows):,}", flush=True)
    print(f"Evaluation questions: {count_unique_questions(eval_rows):,}", flush=True)
    print(
        "Preference preparation: "
        f"styles={preparation_summary['answer_style_counts']} | "
        f"reminders={preparation_summary['reminder_insert_counts']}",
        flush=True,
    )
    print(
        "DPO objective: "
        f"loss_type={args.loss_type} | "
        f"loss_weights={args.loss_weights if args.loss_weights is not None else 'default(1.0)'} | "
        f"rpo_alpha={args.rpo_alpha} | "
        f"label_smoothing={args.label_smoothing}",
        flush=True,
    )
    if ADADPO_LOSS_TYPE in args.loss_type:
        print(
            "AdaDPO settings: "
            f"balance_space={args.adadpo_balance_space} | "
            f"length_normalize={args.adadpo_length_normalize} | "
            f"ceiling={args.adadpo_ceiling}",
            flush=True,
        )
    if not eval_rows:
        print(
            "Warning: no DPO evaluation split was prepared. "
            "Best-checkpoint selection and early stopping are disabled for this run.",
            file=sys.stderr,
            flush=True,
        )
    if uses_generated_selection(args) and not eval_rows:
        raise ValueError(
            "Generated checkpoint selection requires a non-empty DPO validation split. "
            "Increase --validation-ratio or disable generated selection for this run."
        )

    reference_model_name = str(reference_strategy.get("reference_model_name") or "")
    reference_mode = str(reference_strategy.get("mode") or "trl_default")
    if reference_mode == "trl_default":
        print(
            "Using TRL default PEFT reference behavior (adapter-off/base-model reference).",
            flush=True,
        )
    elif reference_mode == "shared_adapter":
        print(
            "Using shared-base DPO reference adapter "
            f"'{reference_strategy.get('reference_adapter_name')}' loaded from: {reference_model_name}",
            flush=True,
        )
    else:
        print(
            f"Using explicit frozen DPO reference model instance: {reference_model_name}",
            flush=True,
        )

    require_cuda_for_unsloth()
    model, tokenizer = load_model_and_tokenizer(args)
    model = maybe_attach_reference_adapter(model, args, reference_strategy)
    ref_model = load_reference_model(args, reference_strategy)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows and not uses_generated_selection(args) else None
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_rows=eval_rows,
        args=args,
        reference_strategy=reference_strategy,
        ref_model=ref_model,
    )
    maybe_precompute_and_release_reference(trainer)
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_checkpoint is not None:
        print(f"Resuming DPO training from checkpoint: {resume_checkpoint}", flush=True)
        stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        stats = trainer.train()
    print(stats)

    eval_metrics = {}
    generated_metric_callback = getattr(trainer, "generated_metric_callback", None)
    if generated_metric_callback is not None and generated_metric_callback.best_summary is not None:
        eval_metrics = {
            "selection_metric": generated_metric_callback.selection_metric,
            "generated_dev": generated_metric_callback.best_summary.get("aggregate", {}),
            "generated_dev_best_step": generated_metric_callback.best_summary.get("step"),
            "generated_dev_best_epoch": generated_metric_callback.best_summary.get("epoch"),
        }
        print(
            "Best generated-dev selection metrics: "
            f"{eval_metrics['selection_metric']}={generated_metric_callback.best_metric}",
            flush=True,
        )
    elif eval_dataset is not None and len(eval_dataset) > 0:
        eval_metrics = dict(trainer.evaluate())
        print(f"Best-model eval metrics: {eval_metrics}", flush=True)

    if not (
        uses_generated_selection(args)
        and generated_metric_callback is not None
        and generated_metric_callback.best_metric is not None
    ):
        Path(args.save_model_dir).mkdir(parents=True, exist_ok=True)
        save_adapter_and_tokenizer(
            model,
            tokenizer,
            args.save_model_dir,
            save_dtype=args.save_dtype,
        )
    artifact_paths = save_training_curves(trainer, Path(args.save_model_dir))
    if generated_metric_callback is not None:
        artifact_paths.update(generated_metric_callback.artifact_paths)

    metrics = dict(getattr(stats, "metrics", {}) or {})
    manifest = {
        "status": "completed",
        "preference_input": args.preference_input,
        "model_name": args.model_name,
        "reference_model_name": reference_model_name or None,
        "reference_mode": reference_mode,
        "policy_adapter_name": reference_strategy.get("policy_adapter_name"),
        "reference_adapter_name": reference_strategy.get("reference_adapter_name"),
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
        "precompute_ref_batch_size": args.precompute_ref_batch_size,
        "resume_from_checkpoint": resume_checkpoint,
        "split_by": args.split_by,
        "validation_ratio": args.validation_ratio,
        "selection_metric": resolve_selection_metric(args),
        "selection_eval_input": args.selection_eval_input,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "loss_weights": args.loss_weights,
        "rpo_alpha": args.rpo_alpha,
        "label_smoothing": args.label_smoothing,
        "use_weighting": args.use_weighting,
        "reference_free": args.reference_free,
        "adadpo_ceiling": args.adadpo_ceiling,
        "adadpo_balance_space": args.adadpo_balance_space,
        "adadpo_length_normalize": args.adadpo_length_normalize,
        "reinforce_output_format": args.reinforce_output_format,
        "max_seq_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "early_stopping_patience": args.early_stopping_patience,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_questions": count_unique_questions(train_rows),
        "eval_questions": count_unique_questions(eval_rows),
        "preference_preparation": preparation_summary,
        "metrics": metrics,
        "eval_metrics": eval_metrics,
        "best_model_checkpoint": (
            generated_metric_callback.best_model_path
            if generated_metric_callback is not None and generated_metric_callback.best_metric is not None
            else getattr(trainer.state, "best_model_checkpoint", None)
        ),
        "best_metric": (
            generated_metric_callback.best_metric
            if generated_metric_callback is not None and generated_metric_callback.best_metric is not None
            else getattr(trainer.state, "best_metric", None)
        ),
        "output_dir": args.output_dir,
        "save_model_dir": args.save_model_dir,
        "training_artifacts": artifact_paths,
    }
    save_json(Path(args.save_model_dir, "dpo_manifest.json"), manifest)
    print(f"Saved DPO adapter to {args.save_model_dir}")


if __name__ == "__main__":
    main()
