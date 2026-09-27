from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import load_json_records, summarize_numeric, write_json, write_jsonl


EXPECTED_CLASS_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive representative, random-permutation, permutation-augmented, and "
            "class-DPO dataset views from ranked factoid sequence classes."
        )
    )
    parser.add_argument(
        "--question-classes-jsonl",
        required=True,
        help="Input JSONL emitted by construct_factoid_ranked_sequence_classes.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the derived dataset views will be written.",
    )
    parser.add_argument(
        "--random-view-seed",
        type=int,
        default=3407,
        help="Fixed seed recorded in the random-permutation view metadata.",
    )
    return parser.parse_args()


def representative_permutation_index(row: dict[str, Any]) -> int:
    assignment_index = int(row.get("rank_assignment_index", 0))
    return (assignment_index % EXPECTED_CLASS_SIZE) + 1


def validate_source_row(row: dict[str, Any]) -> None:
    preferred_sequences = list(row.get("preferred_sequences", []))
    rejected_sequences = list(row.get("rejected_sequences", []))
    if len(preferred_sequences) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {row.get('question_id')} has {len(preferred_sequences)} preferred sequences, "
            f"expected {EXPECTED_CLASS_SIZE}."
        )
    if len(rejected_sequences) != EXPECTED_CLASS_SIZE:
        raise ValueError(
            f"Question {row.get('question_id')} has {len(rejected_sequences)} rejected sequences, "
            f"expected {EXPECTED_CLASS_SIZE}."
        )


def build_representative_row(row: dict[str, Any]) -> dict[str, Any]:
    permutation_index = representative_permutation_index(row)
    preferred = row["preferred_sequences"][permutation_index - 1]
    rejected = row["rejected_sequences"][permutation_index - 1]
    return {
        "view_type": "representative_dpo",
        "question_id": row["question_id"],
        "split": row["split"],
        "prompt": row["prompt"],
        "chosen": preferred["serialized"],
        "rejected": rejected["serialized"],
        "gold": row["gold"],
        "accepted_gold": list(row["accepted_gold"]),
        "wrongs": list(row["wrongs"]),
        "rejected_rank": int(row["rejected_rank"]),
        "selected_permutation_index": permutation_index,
        "selected_wrong_order": list(preferred["wrong_order"]),
        "class_size": EXPECTED_CLASS_SIZE,
        "selection_strategy": "cyclic_by_rank_assignment_index",
        "rank_assignment_index": int(row.get("rank_assignment_index", 0)),
        "rank_assignment_seed": int(row.get("rank_assignment_seed", 0)),
    }


def build_random_view_row(row: dict[str, Any], *, random_view_seed: int) -> dict[str, Any]:
    pair_options: list[dict[str, Any]] = []
    for preferred, rejected in zip(row["preferred_sequences"], row["rejected_sequences"]):
        pair_options.append(
            {
                "permutation_index": int(preferred["permutation_index"]),
                "wrong_order": list(preferred["wrong_order"]),
                "chosen": preferred["serialized"],
                "rejected": rejected["serialized"],
            }
        )
    return {
        "view_type": "random_permutation_dpo",
        "question_id": row["question_id"],
        "split": row["split"],
        "prompt": row["prompt"],
        "gold": row["gold"],
        "accepted_gold": list(row["accepted_gold"]),
        "wrongs": list(row["wrongs"]),
        "rejected_rank": int(row["rejected_rank"]),
        "candidate_pair_count": EXPECTED_CLASS_SIZE,
        "sampling_strategy": "uniform_one_pair_per_question_per_epoch",
        "sampling_seed": int(random_view_seed),
        "pair_options": pair_options,
    }


def build_augmented_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    augmented_rows: list[dict[str, Any]] = []
    pair_loss_weight = 1.0 / EXPECTED_CLASS_SIZE
    for preferred, rejected in zip(row["preferred_sequences"], row["rejected_sequences"]):
        augmented_rows.append(
            {
                "view_type": "permutation_augmented_dpo",
                "question_id": row["question_id"],
                "split": row["split"],
                "prompt": row["prompt"],
                "chosen": preferred["serialized"],
                "rejected": rejected["serialized"],
                "gold": row["gold"],
                "accepted_gold": list(row["accepted_gold"]),
                "wrongs": list(row["wrongs"]),
                "rejected_rank": int(row["rejected_rank"]),
                "permutation_index": int(preferred["permutation_index"]),
                "wrong_order": list(preferred["wrong_order"]),
                "question_pair_count": EXPECTED_CLASS_SIZE,
                "pair_loss_weight_within_question": pair_loss_weight,
                "loss_reduction": "mean_over_question_pairs",
            }
        )
    return augmented_rows


def build_class_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_type": "class_dpo",
        "question_id": row["question_id"],
        "split": row["split"],
        "prompt": row["prompt"],
        "gold": row["gold"],
        "accepted_gold": list(row["accepted_gold"]),
        "wrongs": list(row["wrongs"]),
        "rejected_rank": int(row["rejected_rank"]),
        "preferred_class": [entry["serialized"] for entry in row["preferred_sequences"]],
        "rejected_class": [entry["serialized"] for entry in row["rejected_sequences"]],
        "preferred_class_size": EXPECTED_CLASS_SIZE,
        "rejected_class_size": EXPECTED_CLASS_SIZE,
        "aggregation": "logsumexp",
        "matched_permutation_indices": [
            int(entry["permutation_index"]) for entry in row["preferred_sequences"]
        ],
    }


def rejected_rank_counts(rows: list[dict[str, Any]], field: str = "rejected_rank") -> dict[str, int]:
    counts = Counter(int(row[field]) for row in rows)
    return {str(rank): int(counts.get(rank, 0)) for rank in (2, 3, 4, 5)}


def permutation_index_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(int(row[field]) for row in rows)
    return {str(index): int(counts.get(index, 0)) for index in range(1, EXPECTED_CLASS_SIZE + 1)}


def main() -> int:
    args = parse_args()
    source_rows = [dict(row) for row in load_json_records(Path(args.question_classes_jsonl))]
    for row in source_rows:
        validate_source_row(row)

    representative_rows = [
        build_representative_row(row)
        for row in source_rows
    ]
    random_rows = [
        build_random_view_row(row, random_view_seed=args.random_view_seed)
        for row in source_rows
    ]
    augmented_rows = [
        augmented_row
        for row in source_rows
        for augmented_row in build_augmented_rows(row)
    ]
    class_rows = [
        build_class_row(row)
        for row in source_rows
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_path = output_dir / "representative_dpo.jsonl"
    random_path = output_dir / "random_permutation_dpo.jsonl"
    augmented_path = output_dir / "permutation_augmented_dpo.jsonl"
    class_path = output_dir / "class_dpo.jsonl"

    write_jsonl(representative_path, representative_rows)
    write_jsonl(random_path, random_rows)
    write_jsonl(augmented_path, augmented_rows)
    write_jsonl(class_path, class_rows)

    augmented_pair_weights = [
        float(row["pair_loss_weight_within_question"]) for row in augmented_rows
    ]
    representative_summary = {
        "view_type": "representative_dpo",
        "input_question_classes_jsonl": str(args.question_classes_jsonl),
        "row_count": len(representative_rows),
        "question_count": len(representative_rows),
        "rejected_rank_counts": rejected_rank_counts(representative_rows),
        "selected_permutation_index_counts": permutation_index_counts(
            representative_rows,
            "selected_permutation_index",
        ),
        "selection_strategy": "cyclic_by_rank_assignment_index",
        "output_path": str(representative_path),
    }
    random_summary = {
        "view_type": "random_permutation_dpo",
        "input_question_classes_jsonl": str(args.question_classes_jsonl),
        "row_count": len(random_rows),
        "question_count": len(random_rows),
        "rejected_rank_counts": rejected_rank_counts(random_rows),
        "candidate_pair_count_per_question": EXPECTED_CLASS_SIZE,
        "sampling_strategy": "uniform_one_pair_per_question_per_epoch",
        "sampling_seed": int(args.random_view_seed),
        "output_path": str(random_path),
    }
    augmented_summary = {
        "view_type": "permutation_augmented_dpo",
        "input_question_classes_jsonl": str(args.question_classes_jsonl),
        "row_count": len(augmented_rows),
        "question_count": len(source_rows),
        "pairs_per_question": EXPECTED_CLASS_SIZE,
        "rejected_rank_counts": rejected_rank_counts(augmented_rows),
        "pair_loss_weight_summary": summarize_numeric(augmented_pair_weights),
        "loss_reduction": "mean_over_question_pairs",
        "output_path": str(augmented_path),
    }
    class_summary = {
        "view_type": "class_dpo",
        "input_question_classes_jsonl": str(args.question_classes_jsonl),
        "row_count": len(class_rows),
        "question_count": len(class_rows),
        "rejected_rank_counts": rejected_rank_counts(class_rows),
        "preferred_class_size": EXPECTED_CLASS_SIZE,
        "rejected_class_size": EXPECTED_CLASS_SIZE,
        "aggregation": "logsumexp",
        "output_path": str(class_path),
    }
    manifest = {
        "source_question_classes_jsonl": str(args.question_classes_jsonl),
        "question_count": len(source_rows),
        "views": [
            representative_summary,
            random_summary,
            augmented_summary,
            class_summary,
        ],
        "shared_source_records_only": True,
        "independent_candidate_selection_performed": False,
    }

    write_json(output_dir / "representative_dpo_summary.json", representative_summary)
    write_json(output_dir / "random_permutation_dpo_summary.json", random_summary)
    write_json(output_dir / "permutation_augmented_dpo_summary.json", augmented_summary)
    write_json(output_dir / "class_dpo_summary.json", class_summary)
    write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
