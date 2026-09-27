from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import write_json, write_jsonl
from src.utility.data import clean_text
from src.utility.factoid_output_parsing import parse_factoid_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit factoid output parsing by comparing the existing tagged-first parser "
            "against the new format-agnostic parser on saved prediction files."
        )
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        help="Prediction file as LABEL=PATH. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-generation-samples",
        action="store_true",
        help="Expand generation_samples and audit every sampled output instead of only final predictions.",
    )
    parser.add_argument("--sample-per-stratum", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value).resolve()
        return path.stem, path
    label, path_value = value.split("=", 1)
    label = clean_text(label)
    if not label:
        raise ValueError(f"Missing label in --prediction: {value}")
    return label, Path(path_value).resolve()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def classify_parser_change(current_items: list[str], agnostic_items: list[str]) -> str:
    if current_items == agnostic_items:
        if len(current_items) <= 0:
            return "same_empty"
        if len(current_items) == 1:
            return "same_single"
        return "same_multi"
    if len(current_items) == 1 and len(agnostic_items) > 1:
        return "current_single_agnostic_multi"
    if len(current_items) > 1 and len(agnostic_items) == 1:
        return "current_multi_agnostic_single"
    if len(current_items) == len(agnostic_items) == 1:
        return "same_count_text_changed"
    if len(current_items) == 0 and len(agnostic_items) > 0:
        return "agnostic_recovers_items"
    if len(current_items) > 0 and len(agnostic_items) == 0:
        return "agnostic_loses_items"
    return "other_changed"


def build_audit_rows(
    *,
    label: str,
    prediction_path: Path,
    include_generation_samples: bool,
) -> list[dict[str, Any]]:
    prediction_rows = load_prediction_rows(prediction_path)
    audit_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        question_id = clean_text(row.get("question_id"))
        question_type = clean_text(row.get("question_type"))
        body = clean_text(row.get("body"))
        outputs: list[tuple[str, str]] = []
        if include_generation_samples:
            samples = row.get("generation_samples")
            if isinstance(samples, list) and samples:
                outputs.extend(
                    (f"sample_{index}", clean_text(sample))
                    for index, sample in enumerate(samples, start=1)
                    if clean_text(sample)
                )
        if not outputs:
            outputs.append(("prediction", clean_text(row.get("prediction"))))

        for output_key, output_text in outputs:
            current_items = parse_factoid_candidates(output_text, parser_mode="current")
            agnostic_items = parse_factoid_candidates(output_text, parser_mode="agnostic")
            audit_rows.append(
                {
                    "label": label,
                    "prediction_path": str(prediction_path),
                    "question_id": question_id,
                    "question_type": question_type,
                    "body": body,
                    "output_key": output_key,
                    "raw_output": output_text,
                    "current_items": current_items,
                    "agnostic_items": agnostic_items,
                    "current_item_count": len(current_items),
                    "agnostic_item_count": len(agnostic_items),
                    "parser_change_type": classify_parser_change(current_items, agnostic_items),
                }
            )
    return audit_rows


def sample_audit_rows(
    rows: list[dict[str, Any]],
    *,
    sample_per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["label"]), str(row["parser_change_type"]))].append(row)

    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        candidates = list(strata[key])
        rng.shuffle(candidates)
        sampled.extend(candidates[:sample_per_stratum])
    return sampled


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[str(row["label"])].append(row)

    by_label: dict[str, Any] = {}
    for label, label_rows in sorted(rows_by_label.items()):
        change_counts = Counter(str(row["parser_change_type"]) for row in label_rows)
        current_count_dist = Counter(int(row["current_item_count"]) for row in label_rows)
        agnostic_count_dist = Counter(int(row["agnostic_item_count"]) for row in label_rows)
        by_label[label] = {
            "row_count": len(label_rows),
            "change_type_counts": dict(sorted(change_counts.items())),
            "current_item_count_distribution": dict(sorted(current_count_dist.items())),
            "agnostic_item_count_distribution": dict(sorted(agnostic_count_dist.items())),
            "changed_row_count": sum(
                1
                for row in label_rows
                if str(row["parser_change_type"]).startswith("same_") is False
            ),
            "agnostic_more_items_count": sum(
                1
                for row in label_rows
                if int(row["agnostic_item_count"]) > int(row["current_item_count"])
            ),
        }

    return {
        "row_count": len(rows),
        "labels": by_label,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.prediction:
        raise ValueError("Pass at least one --prediction LABEL=PATH.")

    all_rows: list[dict[str, Any]] = []
    for value in args.prediction:
        label, prediction_path = parse_labeled_path(value)
        all_rows.extend(
            build_audit_rows(
                label=label,
                prediction_path=prediction_path,
                include_generation_samples=bool(args.include_generation_samples),
            )
        )

    sampled_rows = sample_audit_rows(
        all_rows,
        sample_per_stratum=int(args.sample_per_stratum),
        seed=int(args.seed),
    )
    summary = summarize_rows(all_rows)
    summary["sample_per_stratum"] = int(args.sample_per_stratum)
    summary["seed"] = int(args.seed)
    summary["include_generation_samples"] = bool(args.include_generation_samples)

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "audit_rows.jsonl", all_rows)
    write_jsonl(output_dir / "audit_sample.jsonl", sampled_rows)

    print(f"Wrote parser audit summary to {output_dir / 'summary.json'}")
    print(f"Wrote full audit rows to {output_dir / 'audit_rows.jsonl'}")
    print(f"Wrote sampled audit rows to {output_dir / 'audit_sample.jsonl'}")


if __name__ == "__main__":
    main()
