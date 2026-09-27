#!/usr/bin/env python
"""Build a Stage-2 dataset filtered by whether the rejected C2 is extractive.

By default, this keeps C3>C2 pairs where the rejected C2 answer is an exact
continuous normalized span in the PubMed resources. Use --drop-extractive-c2 to
instead keep the non-extractive C2 pairs.

Stage 1 and Stage 3 are copied unchanged so the output can be used directly as
a staged_root by the three-stage trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs"
DEFAULT_OUTPUT_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs_stage2_c2_extractive"

STAGE1_FILE = "dpo_stage1_concept_learning_all_pairs.jsonl"
STAGE2_FILE = "dpo_stage2_format_alignment_all_pairs.jsonl"
STAGE3_FILE = "dpo_stage3_hierarchical_ranking_all_pairs.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key, "")) for key in fieldnames})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"\[/?b?e\]", " ", value)
    # Weak normalization for exactness filters: keep hyphens, Greek letters,
    # and other biomedical surface markers instead of converting everything to
    # alphanumeric tokens. This makes span checks closer to BioASQ surface
    # matching while still ignoring case and whitespace differences.
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def prompt_resources_text(prompt: Any) -> str:
    text = str(prompt or "")
    if "PubMed resources:" in text:
        text = text.split("PubMed resources:", 1)[1]
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[0]
    return text


def contains_with_token_boundaries(container: str, containee: str) -> bool:
    if not container or not containee:
        return False
    if container == containee:
        return True
    return f" {containee} " in f" {container} "


def word_count(value: str) -> int:
    return 0 if not value else len(value.split())


def extractive_audit_record(row: dict[str, Any]) -> dict[str, Any]:
    chosen = row.get("chosen_candidate") or row.get("chosen")
    rejected = row.get("rejected_candidate") or row.get("rejected")
    resources_norm = normalize_text(prompt_resources_text(row.get("prompt")))
    chosen_norm = normalize_text(chosen)
    rejected_norm = normalize_text(rejected)
    chosen_extractive = contains_with_token_boundaries(resources_norm, chosen_norm)
    rejected_extractive = contains_with_token_boundaries(resources_norm, rejected_norm)
    return {
        "pair_id": row.get("pair_id"),
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "chosen_candidate": chosen,
        "rejected_candidate": rejected,
        "chosen_norm": chosen_norm,
        "rejected_norm": rejected_norm,
        "chosen_words": word_count(chosen_norm),
        "rejected_words": word_count(rejected_norm),
        "chosen_extractive_in_snippets": chosen_extractive,
        "rejected_c2_extractive_in_snippets": rejected_extractive,
        "chosen_evidence_support": row.get("chosen_evidence_support"),
        "rejected_evidence_support": row.get("rejected_evidence_support"),
        "chosen_source_model": row.get("chosen_source_model"),
        "rejected_source_model": row.get("rejected_source_model"),
        "chosen_class": row.get("chosen_class"),
        "rejected_class": row.get("rejected_class"),
    }


def stage_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "pairs": len(rows),
        "questions": len({row.get("question_id") for row in rows}),
        "pair_classes": dict(Counter(f"{row.get('chosen_class')}>{row.get('rejected_class')}" for row in rows)),
        "sha256": sha256(path),
        "path": str(path),
    }


def build_dataset(args: argparse.Namespace) -> None:
    source_root = Path(args.source_staged_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_src = source_root / STAGE1_FILE
    stage2_src = source_root / STAGE2_FILE
    stage3_src = source_root / STAGE3_FILE
    for path in [stage1_src, stage2_src, stage3_src]:
        if not path.exists():
            raise FileNotFoundError(path)

    stage2_rows = read_jsonl(stage2_src)
    selected_rows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []

    for row in stage2_rows:
        record = extractive_audit_record(row)
        all_records.append(record)
        keep = bool(record["rejected_c2_extractive_in_snippets"])
        if args.drop_extractive_c2:
            keep = not keep
        if keep:
            new_row = dict(row)
            new_row["stage2_c2_extractive_filter"] = {
                "mode": "drop_extractive_c2" if args.drop_extractive_c2 else "keep_extractive_c2",
                "rejected_c2_extractive_in_snippets": record["rejected_c2_extractive_in_snippets"],
                "chosen_extractive_in_snippets": record["chosen_extractive_in_snippets"],
                "chosen_norm": record["chosen_norm"],
                "rejected_norm": record["rejected_norm"],
            }
            selected_rows.append(new_row)
            selected_records.append(record)
        else:
            rejected_records.append(record)

    shutil.copy2(stage1_src, output_root / STAGE1_FILE)
    shutil.copy2(stage3_src, output_root / STAGE3_FILE)
    write_jsonl(output_root / STAGE2_FILE, selected_rows)

    write_csv((output_root / STAGE1_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE1_FILE))
    write_csv((output_root / STAGE2_FILE).with_suffix(".csv"), selected_rows)
    write_csv((output_root / STAGE3_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE3_FILE))
    write_csv(output_root / "stage2_c2_extractive_all_records.csv", all_records)
    write_csv(output_root / "stage2_c2_extractive_selected_records.csv", selected_records)
    write_jsonl(output_root / "stage2_c2_extractive_selected_records.jsonl", selected_records)
    write_csv(output_root / "stage2_c2_extractive_rejected_records.csv", rejected_records)

    all_c2_extractive_count = sum(1 for record in all_records if record["rejected_c2_extractive_in_snippets"])
    summary = {
        "description": "Staged curriculum with Stage 2 filtered by exact extractiveness of the rejected C2 answer in snippets. Stage 1 and Stage 3 are unchanged.",
        "source_staged_root": str(source_root),
        "mode": "drop_extractive_c2" if args.drop_extractive_c2 else "keep_extractive_c2",
        "source_stage2_pairs": len(stage2_rows),
        "source_stage2_questions": len({row.get("question_id") for row in stage2_rows}),
        "source_stage2_c2_extractive_pairs": all_c2_extractive_count,
        "source_stage2_c2_non_extractive_pairs": len(stage2_rows) - all_c2_extractive_count,
        "selected_stage2_pairs": len(selected_rows),
        "selected_stage2_questions": len({row.get("question_id") for row in selected_rows}),
        "removed_stage2_pairs": len(rejected_records),
        "selected_rejected_source_model_counts": dict(Counter(record.get("rejected_source_model") for record in selected_records)),
        "selected_rejected_evidence_support_counts": dict(Counter(record.get("rejected_evidence_support") for record in selected_records)),
        "stages": {
            STAGE1_FILE: stage_summary(output_root / STAGE1_FILE),
            STAGE2_FILE: stage_summary(output_root / STAGE2_FILE),
            STAGE3_FILE: stage_summary(output_root / STAGE3_FILE),
        },
        "audit_files": {
            "all_csv": str(output_root / "stage2_c2_extractive_all_records.csv"),
            "selected_csv": str(output_root / "stage2_c2_extractive_selected_records.csv"),
            "selected_jsonl": str(output_root / "stage2_c2_extractive_selected_records.jsonl"),
            "rejected_csv": str(output_root / "stage2_c2_extractive_rejected_records.csv"),
            "stage2_training_csv": str((output_root / STAGE2_FILE).with_suffix(".csv")),
            "summary": str(output_root / "summary.json"),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-staged-root", type=Path, default=DEFAULT_SOURCE_STAGED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--drop-extractive-c2", action="store_true", help="Keep non-extractive C2 pairs instead of extractive C2 pairs.")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
