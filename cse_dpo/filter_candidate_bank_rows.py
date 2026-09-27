from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import load_json_records, summarize_numeric, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter candidate-bank rows into a smaller JSONL subset for downstream experiments."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input candidate-bank JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Filtered output candidate-bank JSONL.")
    parser.add_argument("--summary-json", required=True, help="Summary JSON for the filtered subset.")
    parser.add_argument("--min-sample-id", type=int, default=None, help="Optional inclusive minimum sample_id.")
    parser.add_argument("--max-sample-id", type=int, default=None, help="Optional inclusive maximum sample_id.")
    parser.add_argument(
        "--require-parser-status",
        action="append",
        default=[],
        help="Optional parser_status values to keep. Repeatable.",
    )
    return parser.parse_args()


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, int):
        return False, "missing_sample_id"
    if args.min_sample_id is not None and sample_id < args.min_sample_id:
        return False, "min_sample_id"
    if args.max_sample_id is not None and sample_id > args.max_sample_id:
        return False, "max_sample_id"

    if args.require_parser_status:
        parser_status = row.get("parser_status")
        if parser_status not in set(args.require_parser_status):
            return False, "parser_status"

    return True, "kept"


def main() -> None:
    args = parse_args()
    rows = [dict(row) for row in load_json_records(Path(args.input_jsonl))]
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    kept_rows: list[dict[str, Any]] = []
    dropped_counts = Counter()
    for row in rows:
        keep, reason = keep_row(row, args)
        if keep:
            kept_rows.append(row)
        else:
            dropped_counts[reason] += 1

    if not kept_rows:
        raise ValueError("Filtering removed every row; refusing to write an empty candidate bank.")

    write_jsonl(Path(args.output_jsonl), kept_rows)

    sample_ids = [
        int(row["sample_id"])
        for row in kept_rows
        if isinstance(row.get("sample_id"), int)
    ]
    question_ids = {
        str(row["question_id"])
        for row in kept_rows
        if isinstance(row.get("question_id"), str) and row.get("question_id")
    }
    parser_status_counts = Counter(
        str(row.get("parser_status") or "missing")
        for row in kept_rows
    )
    sample_id_counts = Counter(sample_ids)

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "input_count": len(rows),
        "output_count": len(kept_rows),
        "retention_rate": len(kept_rows) / len(rows),
        "filters": {
            "min_sample_id": args.min_sample_id,
            "max_sample_id": args.max_sample_id,
            "require_parser_status": list(args.require_parser_status),
        },
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "question_count": len(question_ids),
        "sample_id_distribution": dict(sorted(sample_id_counts.items())),
        "sample_id_summary": summarize_numeric(sample_ids),
        "parser_status_distribution": dict(sorted(parser_status_counts.items())),
    }
    write_json(Path(args.summary_json), summary)
    print(f"Wrote {len(kept_rows):,} candidate-bank rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
