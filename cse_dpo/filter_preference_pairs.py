from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utility.data import clean_text

from .common import load_json_records, summarize_numeric, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter preference-pair JSONL files into controlled subsets for ablation studies."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input preference-pair JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Filtered output JSONL.")
    parser.add_argument("--summary-json", required=True, help="Filter summary JSON.")
    parser.add_argument(
        "--pair-type",
        action="append",
        default=[],
        help="Keep only rows whose pair_type matches one of these values. Repeatable.",
    )
    parser.add_argument(
        "--positive-source",
        action="append",
        default=[],
        help="Keep only rows whose positive_source matches one of these values. Repeatable.",
    )
    parser.add_argument(
        "--candidate-label",
        action="append",
        default=[],
        help="Keep only rows whose candidate_label matches one of these values. Repeatable.",
    )
    parser.add_argument("--min-delta-f1", type=float, default=None)
    parser.add_argument("--max-delta-f1", type=float, default=None)
    parser.add_argument("--min-delta-recall", type=float, default=None)
    parser.add_argument("--max-delta-recall", type=float, default=None)
    parser.add_argument("--min-delta-precision", type=float, default=None)
    parser.add_argument("--max-delta-precision", type=float, default=None)
    parser.add_argument("--max-semantic-set-edit-distance", type=int, default=None)
    parser.add_argument(
        "--require-equal-cardinality",
        action="store_true",
        help="Keep only rows where len(chosen_items) == len(rejected_items).",
    )
    parser.add_argument(
        "--max-entity-gap",
        type=int,
        default=None,
        help="Keep only rows whose absolute chosen/rejected item-count gap is <= this value.",
    )
    parser.add_argument(
        "--max-entity-ratio",
        type=float,
        default=None,
        help="Keep only rows whose larger/smaller item-count ratio is <= this value.",
    )
    return parser.parse_args()


def _as_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, int):
        return value
    return None


def _item_count(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _normalize_values(values: Sequence[str]) -> set[str]:
    return {clean_text(value) for value in values if clean_text(value)}


def keep_row(row: Mapping[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    pair_types = _normalize_values(args.pair_type)
    if pair_types and clean_text(row.get("pair_type")) not in pair_types:
        return False, "pair_type"

    positive_sources = _normalize_values(args.positive_source)
    if positive_sources and clean_text(row.get("positive_source")) not in positive_sources:
        return False, "positive_source"

    candidate_labels = _normalize_values(args.candidate_label)
    if candidate_labels and clean_text(row.get("candidate_label")) not in candidate_labels:
        return False, "candidate_label"

    delta_f1 = _as_float(row, "delta_f1")
    if args.min_delta_f1 is not None and (delta_f1 is None or delta_f1 < args.min_delta_f1):
        return False, "min_delta_f1"
    if args.max_delta_f1 is not None and (delta_f1 is None or delta_f1 > args.max_delta_f1):
        return False, "max_delta_f1"

    delta_recall = _as_float(row, "delta_recall")
    if args.min_delta_recall is not None and (delta_recall is None or delta_recall < args.min_delta_recall):
        return False, "min_delta_recall"
    if args.max_delta_recall is not None and (delta_recall is None or delta_recall > args.max_delta_recall):
        return False, "max_delta_recall"

    delta_precision = _as_float(row, "delta_precision")
    if args.min_delta_precision is not None and (
        delta_precision is None or delta_precision < args.min_delta_precision
    ):
        return False, "min_delta_precision"
    if args.max_delta_precision is not None and (
        delta_precision is None or delta_precision > args.max_delta_precision
    ):
        return False, "max_delta_precision"

    edit_distance = _as_int(row, "semantic_set_edit_distance")
    if args.max_semantic_set_edit_distance is not None and (
        edit_distance is None or edit_distance > args.max_semantic_set_edit_distance
    ):
        return False, "max_semantic_set_edit_distance"

    chosen_count = _item_count(row, "chosen_items")
    rejected_count = _item_count(row, "rejected_items")
    entity_gap = abs(chosen_count - rejected_count)
    if args.require_equal_cardinality and chosen_count != rejected_count:
        return False, "equal_cardinality"
    if args.max_entity_gap is not None and entity_gap > args.max_entity_gap:
        return False, "max_entity_gap"

    if args.max_entity_ratio is not None:
        larger = max(chosen_count, rejected_count)
        smaller = min(chosen_count, rejected_count)
        ratio = float("inf") if smaller == 0 and larger > 0 else (larger / smaller if smaller else 1.0)
        if ratio > args.max_entity_ratio:
            return False, "max_entity_ratio"

    return True, "kept"


def main() -> None:
    args = parse_args()
    rows = [dict(row) for row in load_json_records(Path(args.input_jsonl))]
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    dropped_counts = Counter()
    kept_rows: list[dict[str, Any]] = []
    for row in rows:
        keep, reason = keep_row(row, args)
        if keep:
            kept_rows.append(row)
        else:
            dropped_counts[reason] += 1

    write_jsonl(Path(args.output_jsonl), kept_rows)

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "input_count": len(rows),
        "output_count": len(kept_rows),
        "retention_rate": (len(kept_rows) / len(rows)) if rows else 0.0,
        "filters": {
            "pair_type": list(args.pair_type),
            "positive_source": list(args.positive_source),
            "candidate_label": list(args.candidate_label),
            "min_delta_f1": args.min_delta_f1,
            "max_delta_f1": args.max_delta_f1,
            "min_delta_recall": args.min_delta_recall,
            "max_delta_recall": args.max_delta_recall,
            "min_delta_precision": args.min_delta_precision,
            "max_delta_precision": args.max_delta_precision,
            "max_semantic_set_edit_distance": args.max_semantic_set_edit_distance,
            "require_equal_cardinality": bool(args.require_equal_cardinality),
            "max_entity_gap": args.max_entity_gap,
            "max_entity_ratio": args.max_entity_ratio,
        },
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "output_pair_type_counts": dict(
            sorted(Counter(clean_text(row.get("pair_type")) or "unknown" for row in kept_rows).items())
        ),
        "output_positive_source_counts": dict(
            sorted(Counter(clean_text(row.get("positive_source")) or "null" for row in kept_rows).items())
        ),
        "output_delta_f1": summarize_numeric(
            float(row["delta_f1"]) for row in kept_rows if isinstance(row.get("delta_f1"), (int, float))
        ),
        "output_delta_precision": summarize_numeric(
            float(row["delta_precision"]) for row in kept_rows if isinstance(row.get("delta_precision"), (int, float))
        ),
        "output_delta_recall": summarize_numeric(
            float(row["delta_recall"]) for row in kept_rows if isinstance(row.get("delta_recall"), (int, float))
        ),
    }
    write_json(Path(args.summary_json), summary)
    print(f"Wrote {len(kept_rows):,} filtered pairs to {args.output_jsonl}")


if __name__ == "__main__":
    main()
