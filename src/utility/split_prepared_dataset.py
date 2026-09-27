from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import clean_text


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a prepared answer-generation JSON list into fixed train/dev files "
            "without dropping extra metadata fields."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Project-relative or absolute path to a prepared JSON list.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Project-relative or absolute output directory for the split files.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of groups to place into the dev split when not using a reference split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for the split.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        choices=sorted(QUESTION_INSTRUCTIONS.keys()),
        help="Optional question types to keep before splitting.",
    )
    parser.add_argument(
        "--group-by",
        default="id",
        choices=["id", "row"],
        help="Split by question id or by individual row.",
    )
    parser.add_argument(
        "--reference-dev-input",
        default=None,
        help=(
            "Optional prepared dev JSON list whose question ids should define the dev split. "
            "Useful for keeping dev question ids aligned across datasets."
        ),
    )
    parser.add_argument(
        "--train-filename",
        default="train_set_answGEN.json",
        help="Filename for the output train split.",
    )
    parser.add_argument(
        "--dev-filename",
        default="dev_set_answGEN.json",
        help="Filename for the output dev split.",
    )
    parser.add_argument(
        "--summary-filename",
        default="split_summary.json",
        help="Filename for the split summary metadata.",
    )
    return parser.parse_args()


def read_prepared_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prepared dataset must be a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Prepared dataset row {index} is not an object in {path}")
        rows.append(row)
    return rows


def normalize_question_types(question_types: Iterable[str] | None) -> set[str] | None:
    if not question_types:
        return None
    normalized = {clean_text(question_type).lower() for question_type in question_types if clean_text(question_type)}
    return normalized or None


def filter_rows(rows: list[dict[str, Any]], allowed_types: set[str] | None) -> list[dict[str, Any]]:
    if not allowed_types:
        return list(rows)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_type = clean_text(row.get("type", "")).lower()
        if row_type in allowed_types:
            filtered.append(row)
    return filtered


def group_key_for_row(row: dict[str, Any], index: int, group_by: str) -> str:
    if group_by == "row":
        return f"row::{index}"
    question_id = clean_text(row.get("id", ""))
    return question_id or f"missing-id::{index}"


def build_groups(rows: list[dict[str, Any]], group_by: str) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows):
        key = group_key_for_row(row, index, group_by)
        groups.setdefault(key, []).append(row)
    return groups


def stratify_labels(groups: OrderedDict[str, list[dict[str, Any]]]) -> list[str] | None:
    labels: list[str] = []
    for group_rows in groups.values():
        row_types = sorted({clean_text(row.get("type", "")).lower() for row in group_rows if clean_text(row.get("type", ""))})
        labels.append("|".join(row_types) or "__unknown__")
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return None
    if any(labels.count(label) < 2 for label in unique_labels):
        return None
    return labels


def choose_dev_keys(
    groups: OrderedDict[str, list[dict[str, Any]]],
    *,
    validation_ratio: float,
    seed: int,
) -> set[str]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1.")
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups to create a train/dev split.")

    from sklearn.model_selection import train_test_split

    group_keys = list(groups.keys())
    labels = stratify_labels(groups)
    _, dev_keys = train_test_split(
        group_keys,
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return set(dev_keys)


def reference_dev_keys(
    path: Path,
    *,
    allowed_types: set[str] | None,
) -> set[str]:
    rows = filter_rows(read_prepared_rows(path), allowed_types)
    keys: set[str] = set()
    for index, row in enumerate(rows):
        question_id = clean_text(row.get("id", ""))
        keys.add(question_id or f"missing-id::{index}")
    return keys


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    allowed_types = normalize_question_types(args.question_types)

    rows = filter_rows(read_prepared_rows(input_path), allowed_types)
    groups = build_groups(rows, args.group_by)

    if args.reference_dev_input:
        dev_keys = reference_dev_keys(resolve_project_path(args.reference_dev_input), allowed_types=allowed_types)
        split_mode = "reference_dev_input"
    else:
        dev_keys = choose_dev_keys(groups, validation_ratio=args.validation_ratio, seed=args.seed)
        split_mode = "random_fixed"

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    matched_reference_groups = 0
    for index, row in enumerate(rows):
        key = group_key_for_row(row, index, args.group_by)
        if key in dev_keys:
            dev_rows.append(row)
        else:
            train_rows.append(row)
    if args.reference_dev_input:
        matched_reference_groups = len({group_key_for_row(row, index, args.group_by) for index, row in enumerate(rows) if group_key_for_row(row, index, args.group_by) in dev_keys})

    train_path = output_dir / args.train_filename
    dev_path = output_dir / args.dev_filename
    summary_path = output_dir / args.summary_filename

    write_rows(train_path, train_rows)
    write_rows(dev_path, dev_rows)

    summary = {
        "source_file": str(input_path),
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "split_mode": split_mode,
        "group_by": args.group_by,
        "question_types": sorted(allowed_types) if allowed_types else "all",
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "reference_dev_input": (
            str(resolve_project_path(args.reference_dev_input))
            if args.reference_dev_input
            else None
        ),
        "input_row_count": len(rows),
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
        "input_group_count": len(groups),
        "train_group_count": len(build_groups(train_rows, args.group_by)),
        "dev_group_count": len(build_groups(dev_rows, args.group_by)),
        "matched_reference_groups": matched_reference_groups if args.reference_dev_input else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(train_rows):,} train rows to {train_path}")
    print(f"Saved {len(dev_rows):,} dev rows to {dev_path}")
    print(f"Saved split summary to {summary_path}")


if __name__ == "__main__":
    main()
