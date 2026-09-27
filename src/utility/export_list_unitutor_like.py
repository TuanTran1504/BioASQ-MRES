from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.data import INPUT_FIELD_PATTERN, load_bioasq_records, save_prepared_records


DEFAULT_INPUT = "data/training13b.json"
DEFAULT_OUTPUT_DIR = "data/BioASQ_list_original_snippets_unitutor_like"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw BioASQ training data into a UNITOR-like prepared JSON list "
            "using only original BioASQ snippets for list questions."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Project-relative or absolute path to a raw BioASQ JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Project-relative or absolute directory where converted files will be written.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=0,
        help="Maximum number of resources to keep per question. Use 0 for all.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=0,
        help="Maximum characters per resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--resource-granularity",
        default="document",
        choices=["document", "snippet"],
        help="Whether to group evidence by PubMed document or keep individual snippets.",
    )
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=100,
        help="Maximum number of gold list items to keep in the tagged output.",
    )
    return parser.parse_args()


def build_conversion_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        question_types=["list"],
        max_resources=cli_args.max_resources,
        max_resource_chars=cli_args.max_resource_chars,
        resource_granularity=cli_args.resource_granularity,
        resource_selection="first",
        max_summary_answers=1,
        max_factoid_answers=5,
        max_list_items=cli_args.max_list_items,
    )


def max_resource_index(row: dict[str, Any]) -> int:
    max_index = 1
    for key, value in row.items():
        match = INPUT_FIELD_PATTERN.fullmatch(str(key))
        if match is None or not str(value or "").strip():
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index


def build_summary(
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    cli_args: argparse.Namespace,
) -> dict[str, Any]:
    resource_counts = [
        sum(
            1
            for key, value in row.items()
            if INPUT_FIELD_PATTERN.fullmatch(str(key))
            and int(INPUT_FIELD_PATTERN.fullmatch(str(key)).group(1)) >= 2
            and str(value or "").strip()
        )
        for row in rows
    ]
    max_input_index = max((max_resource_index(row) for row in rows), default=1)
    return {
        "source_file": str(input_path),
        "output_file": str(output_path),
        "question_types": ["list"],
        "record_count": len(rows),
        "max_resources": cli_args.max_resources,
        "max_resource_chars": cli_args.max_resource_chars,
        "max_list_items": cli_args.max_list_items,
        "resource_grouping": f"by_{cli_args.resource_granularity}",
        "resource_content": "Original BioASQ snippet text only, wrapped in [BS]...[ES] with no added abstract context.",
        "resource_header": "Each grouped resource keeps the PubMed ID header used by the repo's raw BioASQ SFT pipeline.",
        "average_nonempty_resource_fields": (
            sum(resource_counts) / len(resource_counts) if resource_counts else 0.0
        ),
        "max_nonempty_input_field_index": max_input_index,
    }


def main() -> None:
    cli_args = parse_args()
    input_path = resolve_project_path(cli_args.input)
    output_dir = resolve_project_path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conversion_args = build_conversion_args(cli_args)
    rows = load_bioasq_records(input_path, conversion_args)

    output_path = output_dir / "train_set_answGEN.json"
    summary_path = output_dir / "conversion_summary.json"

    save_prepared_records(rows, output_path)
    summary = build_summary(rows, input_path=input_path, output_path=output_path, cli_args=cli_args)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(rows):,} list examples to {output_path}")
    print(f"Saved conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
