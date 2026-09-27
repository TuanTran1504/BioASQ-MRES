from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import write_json, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced DPO preference-pair JSONL by downsampling pair types."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input preference-pair JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Balanced output JSONL.")
    parser.add_argument("--summary-json", required=True, help="Balance summary JSON.")
    parser.add_argument(
        "--target-per-type",
        type=int,
        default=None,
        help="Optional target count per pair type. Defaults to the smallest observed pair-type count.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for downsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_type = str(row.get("pair_type") or "").strip()
        if not pair_type:
            pair_type = "unknown"
        by_type[pair_type].append(row)

    if len(by_type) < 2:
        raise ValueError(f"Need at least two pair types to balance, found: {sorted(by_type)}")

    original_counts = {pair_type: len(values) for pair_type, values in sorted(by_type.items())}
    target = args.target_per_type if args.target_per_type is not None else min(original_counts.values())
    if target <= 0:
        raise ValueError("--target-per-type must be positive")

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()

    for pair_type, values in sorted(by_type.items()):
        shuffled = list(values)
        rng.shuffle(shuffled)
        kept = shuffled[: min(target, len(shuffled))]
        dropped = max(0, len(shuffled) - len(kept))
        selected.extend(kept)
        selected_counts[pair_type] = len(kept)
        dropped_counts[pair_type] = dropped

    rng.shuffle(selected)
    write_jsonl(Path(args.output_jsonl), selected)

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "seed": args.seed,
        "target_per_type": target,
        "original_pair_count": len(rows),
        "balanced_pair_count": len(selected),
        "original_pair_type_counts": original_counts,
        "balanced_pair_type_counts": dict(sorted(selected_counts.items())),
        "dropped_pair_type_counts": dict(sorted(dropped_counts.items())),
    }
    write_json(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
