#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import parse_prediction_items
from src.utility.data import clean_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand prepared factoid rows with multiple [BE]...[EE] aliases into "
            "one row per alias while preserving question text and resources."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more prepared JSON input files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Prepared JSON output file.",
    )
    return parser.parse_args()


def load_prepared_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Prepared dataset must be a JSON list: {path}")

    required = {"instruction", "input_1", "output", "type"}
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(
                f"Prepared dataset row {index} in {path} is missing one of {sorted(required)}."
            )
        rows.append(dict(row))
    return rows


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for raw_value in values:
        value = clean_text(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def alias_row_id(base_id: str, *, row_index: int, alias_index: int, alias_count: int) -> str:
    fallback_id = base_id or f"prepared:{row_index}"
    if alias_count <= 1:
        return fallback_id

    width = max(2, len(str(alias_count)))
    return f"{fallback_id}__alias_{alias_index:0{width}d}_of_{alias_count:0{width}d}"


def expand_factoid_row(row: dict[str, Any], *, row_index: int) -> list[dict[str, Any]]:
    row_type = clean_text(row.get("type", "")).lower()
    if row_type != "factoid":
        return [dict(row)]

    aliases = unique_preserving_order(parse_prediction_items(str(row.get("output", "")), "factoid"))
    if not aliases:
        raise ValueError(
            f"Factoid row {row.get('id')!r} at position {row_index} does not contain any parsable answer."
        )

    base_id = clean_text(row.get("id", ""))
    alias_count = len(aliases)
    expanded_rows: list[dict[str, Any]] = []
    for alias_index, alias in enumerate(aliases, start=1):
        expanded = dict(row)
        expanded["id"] = alias_row_id(
            base_id,
            row_index=row_index,
            alias_index=alias_index,
            alias_count=alias_count,
        )
        expanded["output"] = f"[BE]{alias}[EE]"
        expanded["alias_index"] = alias_index
        expanded["alias_count"] = alias_count
        expanded["alias_text"] = alias
        if base_id and expanded["id"] != base_id:
            expanded["original_id"] = base_id
        expanded_rows.append(expanded)
    return expanded_rows


def main() -> None:
    args = parse_args()
    input_paths = [Path(raw_path).resolve() for raw_path in args.inputs]
    output_path = Path(args.output).resolve()

    output_rows: list[dict[str, Any]] = []
    total_input_rows = 0
    expanded_factoid_rows = 0
    extra_rows_added = 0

    for input_path in input_paths:
        rows = load_prepared_rows(input_path)
        total_input_rows += len(rows)
        for row_index, row in enumerate(rows, start=1):
            expanded_rows = expand_factoid_row(row, row_index=row_index)
            if clean_text(row.get("type", "")).lower() == "factoid":
                expanded_factoid_rows += 1
                extra_rows_added += max(0, len(expanded_rows) - 1)
            output_rows.extend(expanded_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved {len(output_rows):,} rows to {output_path}")
    print(f"Loaded {total_input_rows:,} prepared rows from {len(input_paths)} input file(s)")
    print(f"Expanded {expanded_factoid_rows:,} factoid rows into {len(output_rows):,} total rows")
    print(f"Added {extra_rows_added:,} extra alias-specific rows")


if __name__ == "__main__":
    main()
