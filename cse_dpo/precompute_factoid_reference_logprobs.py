from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, summarize_numeric, write_json, write_jsonl
from cse_dpo.construct_factoid_baseline_data_views import (
    EXPECTED_CLASS_SIZE,
    build_augmented_rows,
    build_class_row,
    build_random_view_row,
    build_representative_row,
)
from cse_dpo.score_candidate_bank import (
    ScoringPayload,
    release_model,
    resolve_single_model_spec,
    score_payloads_with_model,
)
from src.utility.data import clean_text
from src.utility.eval_models import load_model_and_tokenizer_for_eval, prime_unsloth_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute frozen-reference sequence log-probabilities for all 48 factoid "
            "completions per question and emit reference-augmented dataset views."
        )
    )
    parser.add_argument(
        "--question-classes-jsonl",
        required=True,
        help="Question-class JSONL from construct_factoid_ranked_sequence_classes.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where reference-precomputed artifacts will be written.",
    )
    parser.add_argument(
        "--model-ref",
        required=True,
        help="Frozen SFT checkpoint used as the reference model.",
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry path used to resolve model refs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of completions to score per forward pass.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum prompt+completion sequence length for scoring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional question limit for smoke tests.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=250,
        help="Print progress after this many scored completions. Use 0 to disable.",
    )
    parser.add_argument(
        "--empty-cuda-cache-steps",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N scoring batches. Use 0 to disable.",
    )
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def stable_logsumexp(values: list[float]) -> float | None:
    if not values:
        return None
    max_value = max(values)
    if not math.isfinite(max_value):
        return None
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def validate_source_row(row: dict[str, Any]) -> None:
    preferred_sequences = list(row.get("preferred_sequences", []))
    rejected_sequences = list(row.get("rejected_sequences", []))
    if len(preferred_sequences) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {row.get('question_id')} has {len(preferred_sequences)} preferred sequences; "
            f"expected {EXPECTED_CLASS_SIZE}."
        )
    if len(rejected_sequences) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {row.get('question_id')} has {len(rejected_sequences)} rejected sequences; "
            f"expected {EXPECTED_CLASS_SIZE}."
        )


def build_scoring_payloads(
    question_rows: list[dict[str, Any]],
) -> tuple[list[ScoringPayload], list[dict[str, Any]]]:
    payloads: list[ScoringPayload] = []
    metadata: list[dict[str, Any]] = []
    payload_index = 0
    for question_row in question_rows:
        question_id = clean_text(question_row.get("question_id"))
        prompt = str(question_row.get("prompt", ""))
        for class_label, sequence_list in (
            ("preferred", question_row["preferred_sequences"]),
            ("rejected", question_row["rejected_sequences"]),
        ):
            for sequence in sequence_list:
                payloads.append(
                    ScoringPayload(
                        row_index=payload_index,
                        prompt=prompt,
                        completion_text=str(sequence["serialized"]),
                        append_eos=True,
                    )
                )
                metadata.append(
                    {
                        "payload_index": payload_index,
                        "question_id": question_id,
                        "class_label": class_label,
                        "permutation_index": int(sequence["permutation_index"]),
                        "serialized": str(sequence["serialized"]),
                    }
                )
                payload_index += 1
    return payloads, metadata


def attach_reference_scores(
    question_rows: list[dict[str, Any]],
    *,
    metadata_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    model_ref: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    completion_score_rows: list[dict[str, Any]] = []

    for meta, score in zip(metadata_rows, score_rows):
        key = (
            meta["question_id"],
            meta["class_label"],
            int(meta["permutation_index"]),
        )
        merged = {
            **meta,
            "reference_model_ref": model_ref,
            "reference_logp_sum": score.get("logp_sum"),
            "reference_logp_mean": score.get("logp_mean"),
            "reference_score_token_count": int(score.get("score_token_count", 0) or 0),
            "reference_prompt_token_count": int(score.get("prompt_token_count", 0) or 0),
            "reference_effective_prompt_token_count": int(
                score.get("effective_prompt_token_count", 0) or 0
            ),
            "reference_completion_token_count": int(score.get("completion_token_count", 0) or 0),
            "reference_effective_completion_token_count": int(
                score.get("effective_completion_token_count", 0) or 0
            ),
            "reference_prompt_truncated": bool(score.get("prompt_truncated", False)),
            "reference_completion_truncated": bool(score.get("completion_truncated", False)),
            "reference_scored_input_token_count": int(
                score.get("scored_input_token_count", 0) or 0
            ),
            "reference_scoring_status": str(score.get("status") or "scored"),
            "reference_eos_included": True,
            "reference_length_normalized": False,
        }
        scores_by_key[key] = merged
        completion_score_rows.append(merged)

    augmented_question_rows: list[dict[str, Any]] = []
    for question_row in question_rows:
        augmented = dict(question_row)
        preferred_sequences = []
        rejected_sequences = []
        preferred_logp_sums: list[float] = []
        rejected_logp_sums: list[float] = []

        for class_label, source_sequences, target_sequences, target_sums in (
            ("preferred", question_row["preferred_sequences"], preferred_sequences, preferred_logp_sums),
            ("rejected", question_row["rejected_sequences"], rejected_sequences, rejected_logp_sums),
        ):
            for sequence in source_sequences:
                key = (
                    clean_text(question_row.get("question_id")),
                    class_label,
                    int(sequence["permutation_index"]),
                )
                score = scores_by_key[key]
                sequence_with_reference = dict(sequence)
                sequence_with_reference.update(
                    {
                        "reference_model_ref": model_ref,
                        "reference_logp_sum": score["reference_logp_sum"],
                        "reference_logp_mean": score["reference_logp_mean"],
                        "reference_score_token_count": score["reference_score_token_count"],
                        "reference_prompt_token_count": score["reference_prompt_token_count"],
                        "reference_effective_prompt_token_count": score[
                            "reference_effective_prompt_token_count"
                        ],
                        "reference_completion_token_count": score[
                            "reference_completion_token_count"
                        ],
                        "reference_effective_completion_token_count": score[
                            "reference_effective_completion_token_count"
                        ],
                        "reference_prompt_truncated": score["reference_prompt_truncated"],
                        "reference_completion_truncated": score[
                            "reference_completion_truncated"
                        ],
                        "reference_scored_input_token_count": score[
                            "reference_scored_input_token_count"
                        ],
                        "reference_scoring_status": score["reference_scoring_status"],
                        "reference_eos_included": True,
                        "reference_length_normalized": False,
                    }
                )
                target_sequences.append(sequence_with_reference)
                if isinstance(score["reference_logp_sum"], (int, float)):
                    target_sums.append(float(score["reference_logp_sum"]))

        augmented["preferred_sequences"] = preferred_sequences
        augmented["rejected_sequences"] = rejected_sequences
        augmented["reference_model_ref"] = model_ref
        augmented["reference_eos_included"] = True
        augmented["reference_length_normalized"] = False
        augmented["preferred_class_reference_logp_sums"] = [
            sequence.get("reference_logp_sum") for sequence in preferred_sequences
        ]
        augmented["rejected_class_reference_logp_sums"] = [
            sequence.get("reference_logp_sum") for sequence in rejected_sequences
        ]
        augmented["preferred_class_reference_logsumexp"] = stable_logsumexp(preferred_logp_sums)
        augmented["rejected_class_reference_logsumexp"] = stable_logsumexp(rejected_logp_sums)
        if (
            isinstance(augmented["preferred_class_reference_logsumexp"], (int, float))
            and isinstance(augmented["rejected_class_reference_logsumexp"], (int, float))
        ):
            augmented["reference_logsumexp_margin"] = (
                float(augmented["preferred_class_reference_logsumexp"])
                - float(augmented["rejected_class_reference_logsumexp"])
            )
        else:
            augmented["reference_logsumexp_margin"] = None
        augmented_question_rows.append(augmented)

    return augmented_question_rows, completion_score_rows


def build_reference_views(
    question_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    representative_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    augmented_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for question_row in question_rows:
        representative = build_representative_row(question_row)
        permutation_index = int(representative["selected_permutation_index"])
        preferred_sequence = question_row["preferred_sequences"][permutation_index - 1]
        rejected_sequence = question_row["rejected_sequences"][permutation_index - 1]
        representative.update(
            {
                "reference_model_ref": question_row["reference_model_ref"],
                "reference_chosen_logp_sum": preferred_sequence["reference_logp_sum"],
                "reference_chosen_logp_mean": preferred_sequence["reference_logp_mean"],
                "reference_rejected_logp_sum": rejected_sequence["reference_logp_sum"],
                "reference_rejected_logp_mean": rejected_sequence["reference_logp_mean"],
                "reference_chosen_score_token_count": preferred_sequence[
                    "reference_score_token_count"
                ],
                "reference_rejected_score_token_count": rejected_sequence[
                    "reference_score_token_count"
                ],
                "reference_eos_included": True,
                "reference_length_normalized": False,
            }
        )
        if isinstance(representative["reference_chosen_logp_sum"], (int, float)) and isinstance(
            representative["reference_rejected_logp_sum"], (int, float)
        ):
            representative["reference_logp_margin"] = (
                float(representative["reference_chosen_logp_sum"])
                - float(representative["reference_rejected_logp_sum"])
            )
        else:
            representative["reference_logp_margin"] = None
        representative_rows.append(representative)

        random_view = build_random_view_row(
            question_row,
            random_view_seed=int(question_row.get("rank_assignment_seed", 0) or 0),
        )
        pair_options = []
        for preferred_sequence, rejected_sequence in zip(
            question_row["preferred_sequences"],
            question_row["rejected_sequences"],
        ):
            pair_option = {
                "permutation_index": int(preferred_sequence["permutation_index"]),
                "wrong_order": list(preferred_sequence["wrong_order"]),
                "chosen": preferred_sequence["serialized"],
                "rejected": rejected_sequence["serialized"],
                "reference_chosen_logp_sum": preferred_sequence["reference_logp_sum"],
                "reference_rejected_logp_sum": rejected_sequence["reference_logp_sum"],
                "reference_chosen_logp_mean": preferred_sequence["reference_logp_mean"],
                "reference_rejected_logp_mean": rejected_sequence["reference_logp_mean"],
                "reference_eos_included": True,
                "reference_length_normalized": False,
            }
            if isinstance(pair_option["reference_chosen_logp_sum"], (int, float)) and isinstance(
                pair_option["reference_rejected_logp_sum"], (int, float)
            ):
                pair_option["reference_logp_margin"] = (
                    float(pair_option["reference_chosen_logp_sum"])
                    - float(pair_option["reference_rejected_logp_sum"])
                )
            else:
                pair_option["reference_logp_margin"] = None
            pair_options.append(pair_option)
        random_view["reference_model_ref"] = question_row["reference_model_ref"]
        random_view["reference_eos_included"] = True
        random_view["reference_length_normalized"] = False
        random_view["pair_options"] = pair_options
        random_rows.append(random_view)

        for augmented in build_augmented_rows(question_row):
            permutation_index = int(augmented["permutation_index"])
            preferred_sequence = question_row["preferred_sequences"][permutation_index - 1]
            rejected_sequence = question_row["rejected_sequences"][permutation_index - 1]
            augmented.update(
                {
                    "reference_model_ref": question_row["reference_model_ref"],
                    "reference_chosen_logp_sum": preferred_sequence["reference_logp_sum"],
                    "reference_chosen_logp_mean": preferred_sequence["reference_logp_mean"],
                    "reference_rejected_logp_sum": rejected_sequence["reference_logp_sum"],
                    "reference_rejected_logp_mean": rejected_sequence["reference_logp_mean"],
                    "reference_chosen_score_token_count": preferred_sequence[
                        "reference_score_token_count"
                    ],
                    "reference_rejected_score_token_count": rejected_sequence[
                        "reference_score_token_count"
                    ],
                    "reference_eos_included": True,
                    "reference_length_normalized": False,
                }
            )
            if isinstance(augmented["reference_chosen_logp_sum"], (int, float)) and isinstance(
                augmented["reference_rejected_logp_sum"], (int, float)
            ):
                augmented["reference_logp_margin"] = (
                    float(augmented["reference_chosen_logp_sum"])
                    - float(augmented["reference_rejected_logp_sum"])
                )
            else:
                augmented["reference_logp_margin"] = None
            augmented_rows.append(augmented)

        class_row = build_class_row(question_row)
        class_row.update(
            {
                "reference_model_ref": question_row["reference_model_ref"],
                "preferred_class_reference_logp_sums": question_row[
                    "preferred_class_reference_logp_sums"
                ],
                "rejected_class_reference_logp_sums": question_row[
                    "rejected_class_reference_logp_sums"
                ],
                "preferred_class_reference_logsumexp": question_row[
                    "preferred_class_reference_logsumexp"
                ],
                "rejected_class_reference_logsumexp": question_row[
                    "rejected_class_reference_logsumexp"
                ],
                "reference_logsumexp_margin": question_row["reference_logsumexp_margin"],
                "reference_eos_included": True,
                "reference_length_normalized": False,
            }
        )
        class_rows.append(class_row)

    return representative_rows, random_rows, augmented_rows, class_rows


def main() -> int:
    args = parse_args()
    prime_unsloth_runtime()

    source_rows = [dict(row) for row in load_json_records(Path(args.question_classes_jsonl))]
    if args.limit is not None:
        source_rows = source_rows[: max(0, int(args.limit))]
    if not source_rows:
        raise ValueError("No question-class rows loaded from --question-classes-jsonl.")
    for row in source_rows:
        validate_source_row(row)

    payloads, metadata_rows = build_scoring_payloads(source_rows)
    model_spec = resolve_single_model_spec(
        model_ref=str(args.model_ref),
        args=args,
        project_root=PROJECT_ROOT,
    )
    model, tokenizer = load_model_and_tokenizer_for_eval(model_spec, args)
    try:
        score_rows = score_payloads_with_model(
            payloads=payloads,
            model=model,
            tokenizer=tokenizer,
            args=args,
            progress_label="reference-precompute",
        )
    finally:
        release_model(model, tokenizer)

    augmented_question_rows, completion_score_rows = attach_reference_scores(
        source_rows,
        metadata_rows=metadata_rows,
        score_rows=score_rows,
        model_ref=str(model_spec.ref),
    )
    representative_rows, random_rows, augmented_rows, class_rows = build_reference_views(
        augmented_question_rows
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    completion_scores_path = output_dir / "completion_reference_scores.jsonl"
    question_classes_path = output_dir / "question_classes_with_reference.jsonl"
    representative_path = output_dir / "representative_dpo_with_reference.jsonl"
    random_path = output_dir / "random_permutation_dpo_with_reference.jsonl"
    augmented_path = output_dir / "permutation_augmented_dpo_with_reference.jsonl"
    class_path = output_dir / "class_dpo_with_reference.jsonl"

    write_jsonl(completion_scores_path, completion_score_rows)
    write_jsonl(question_classes_path, augmented_question_rows)
    write_jsonl(representative_path, representative_rows)
    write_jsonl(random_path, random_rows)
    write_jsonl(augmented_path, augmented_rows)
    write_jsonl(class_path, class_rows)

    reference_logp_sums = [
        float(row["reference_logp_sum"])
        for row in completion_score_rows
        if isinstance(row.get("reference_logp_sum"), (int, float))
    ]
    class_preferred_logsumexp = [
        float(row["preferred_class_reference_logsumexp"])
        for row in class_rows
        if isinstance(row.get("preferred_class_reference_logsumexp"), (int, float))
    ]
    class_rejected_logsumexp = [
        float(row["rejected_class_reference_logsumexp"])
        for row in class_rows
        if isinstance(row.get("rejected_class_reference_logsumexp"), (int, float))
    ]
    prompt_truncation_count = sum(
        1 for row in completion_score_rows if bool(row.get("reference_prompt_truncated"))
    )
    completion_truncation_count = sum(
        1 for row in completion_score_rows if bool(row.get("reference_completion_truncated"))
    )
    status_counts: dict[str, int] = {}
    for row in completion_score_rows:
        status = clean_text(row.get("reference_scoring_status")) or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "input_question_classes_jsonl": str(args.question_classes_jsonl),
        "reference_model_ref": str(model_spec.ref),
        "question_count": len(augmented_question_rows),
        "completion_count": len(completion_score_rows),
        "completions_per_question": 48,
        "eos_included": True,
        "prompt_tokens_scored": False,
        "length_normalized": False,
        "batch_size": int(args.batch_size),
        "max_seq_length": int(args.max_seq_length),
        "scoring_status_counts": dict(sorted(status_counts.items())),
        "prompt_truncation_count": prompt_truncation_count,
        "completion_truncation_count": completion_truncation_count,
        "reference_logp_sum_distribution": summarize_numeric(reference_logp_sums),
        "preferred_class_reference_logsumexp_distribution": summarize_numeric(
            class_preferred_logsumexp
        ),
        "rejected_class_reference_logsumexp_distribution": summarize_numeric(
            class_rejected_logsumexp
        ),
        "outputs": {
            "completion_reference_scores_jsonl": str(completion_scores_path),
            "question_classes_with_reference_jsonl": str(question_classes_path),
            "representative_dpo_with_reference_jsonl": str(representative_path),
            "random_permutation_dpo_with_reference_jsonl": str(random_path),
            "permutation_augmented_dpo_with_reference_jsonl": str(augmented_path),
            "class_dpo_with_reference_jsonl": str(class_path),
        },
    }
    manifest = {
        "source_question_classes_jsonl": str(args.question_classes_jsonl),
        "reference_model_ref": str(model_spec.ref),
        "eos_included": True,
        "prompt_tokens_scored": False,
        "length_normalized": False,
        "views": [
            {"name": "representative_dpo_with_reference", "path": str(representative_path)},
            {"name": "random_permutation_dpo_with_reference", "path": str(random_path)},
            {"name": "permutation_augmented_dpo_with_reference", "path": str(augmented_path)},
            {"name": "class_dpo_with_reference", "path": str(class_path)},
        ],
        "summary_path": str(output_dir / "summary.json"),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
