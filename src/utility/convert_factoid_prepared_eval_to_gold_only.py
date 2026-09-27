from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import parse_prediction_items
from src.utility.data import load_prepared_records, save_prepared_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a prepared factoid eval set whose output contains ranked candidates "
            "into a gold-only version that keeps only the first factoid answer."
        )
    )
    parser.add_argument("--input-prepared", required=True, help="Prepared JSON input file.")
    parser.add_argument("--output-prepared", required=True, help="Prepared JSON output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_prepared).resolve()
    output_path = Path(args.output_prepared).resolve()

    rows = load_prepared_records(input_path)
    converted_rows: list[dict[str, object]] = []
    converted_factoid_rows = 0

    for row in rows:
        converted = dict(row)
        if str(row.get("type", "")).strip().lower() == "factoid":
            items = parse_prediction_items(str(row.get("output", "")), "factoid")
            if not items:
                raise ValueError(
                    f"Factoid row {row.get('id')!r} in {input_path} does not contain any parsable [BE]...[EE] answer."
                )
            converted["output"] = f"[BE]{items[0]}[EE]"
            converted_factoid_rows += 1
        converted_rows.append(converted)

    save_prepared_records(converted_rows, output_path)
    print(f"Saved {len(converted_rows):,} rows to {output_path}")
    print(f"Converted {converted_factoid_rows:,} factoid rows to gold-only outputs")


if __name__ == "__main__":
    main()
