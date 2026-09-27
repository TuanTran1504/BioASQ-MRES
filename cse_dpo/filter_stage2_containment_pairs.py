#!/usr/bin/env python
"""Build a Stage-2 dataset containing answer-containment and span-overlap pairs.

The input is the full staged curriculum root. Stage 1 and Stage 3 are copied
unchanged. Stage 2 is filtered to C3>C2 pairs where either:

1. the normalized chosen C3 answer contains the normalized rejected C2 answer;
2. the normalized rejected C2 answer contains the normalized chosen C3 answer;
3. the two answers overlap by a prefix/suffix, and their merged union is an
   exact continuous normalized span in the PubMed resources prompt.
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
DEFAULT_OUTPUT_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs_stage2_containment_overlap"

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


def normalize_answer(value: Any) -> str:
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


def word_count(value: str) -> int:
    return 0 if not value else len(value.split())


def contains_with_token_boundaries(container: str, containee: str) -> bool:
    if not container or not containee:
        return False
    if container == containee:
        return True
    return f" {containee} " in f" {container} "


def overlap_unions(left: str, right: str, min_overlap_words: int) -> list[dict[str, Any]]:
    left_tokens = left.split()
    right_tokens = right.split()
    max_overlap = min(len(left_tokens), len(right_tokens))
    unions: list[dict[str, Any]] = []
    for overlap in range(max_overlap, min_overlap_words - 1, -1):
        if left_tokens[-overlap:] == right_tokens[:overlap]:
            union_tokens = left_tokens + right_tokens[overlap:]
            unions.append({
                "overlap_words": overlap,
                "overlap_text": " ".join(left_tokens[-overlap:]),
                "merged_span_norm": " ".join(union_tokens),
                "overlap_direction": "chosen_prefix_to_rejected_suffix",
            })
        if right_tokens[-overlap:] == left_tokens[:overlap]:
            union_tokens = right_tokens + left_tokens[overlap:]
            unions.append({
                "overlap_words": overlap,
                "overlap_text": " ".join(right_tokens[-overlap:]),
                "merged_span_norm": " ".join(union_tokens),
                "overlap_direction": "rejected_prefix_to_chosen_suffix",
            })
    # Deduplicate while preserving strongest overlaps first.
    seen = set()
    unique = []
    for item in unions:
        key = (item["merged_span_norm"], item["overlap_direction"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def containment_record(row: dict[str, Any], min_shorter_words: int, include_overlap_span: bool, min_overlap_words: int) -> dict[str, Any] | None:
    chosen = row.get("chosen_candidate") or row.get("chosen")
    rejected = row.get("rejected_candidate") or row.get("rejected")
    chosen_norm = normalize_answer(chosen)
    rejected_norm = normalize_answer(rejected)
    resources_norm = normalize_answer(prompt_resources_text(row.get("prompt")))
    chosen_words = word_count(chosen_norm)
    rejected_words = word_count(rejected_norm)
    shorter_words = min(chosen_words, rejected_words)
    if shorter_words < min_shorter_words:
        return None

    chosen_contains_rejected = contains_with_token_boundaries(chosen_norm, rejected_norm)
    rejected_contains_chosen = contains_with_token_boundaries(rejected_norm, chosen_norm)
    merged_span_norm = ""
    overlap_words = 0
    overlap_text = ""
    overlap_direction = ""

    if chosen_norm == rejected_norm:
        direction = "exact_normalized_match"
    elif chosen_contains_rejected and rejected_contains_chosen:
        direction = "mutual_containment"
    elif chosen_contains_rejected:
        direction = "gold_contains_prediction"
    elif rejected_contains_chosen:
        direction = "prediction_contains_gold"
    else:
        direction = ""

    if not direction and include_overlap_span:
        for overlap in overlap_unions(chosen_norm, rejected_norm, min_overlap_words):
            if contains_with_token_boundaries(resources_norm, overlap["merged_span_norm"]):
                direction = "overlap_merged_span_in_snippets"
                merged_span_norm = overlap["merged_span_norm"]
                overlap_words = int(overlap["overlap_words"])
                overlap_text = overlap["overlap_text"]
                overlap_direction = overlap["overlap_direction"]
                break

    if not direction:
        return None

    return {
        "pair_id": row.get("pair_id"),
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "chosen_candidate": chosen,
        "rejected_candidate": rejected,
        "chosen_norm": chosen_norm,
        "rejected_norm": rejected_norm,
        "chosen_words": chosen_words,
        "rejected_words": rejected_words,
        "shorter_words": shorter_words,
        "longer_words": max(chosen_words, rejected_words),
        "word_length_delta": abs(chosen_words - rejected_words),
        "rejected_to_chosen_word_ratio": rejected_words / max(1, chosen_words),
        "containment_direction": direction,
        "merged_span_norm": merged_span_norm,
        "overlap_words": overlap_words,
        "overlap_text": overlap_text,
        "overlap_direction": overlap_direction,
        "merged_span_in_snippets": bool(merged_span_norm),
        "chosen_class": row.get("chosen_class"),
        "rejected_class": row.get("rejected_class"),
        "chosen_source_model": row.get("chosen_source_model"),
        "rejected_source_model": row.get("rejected_source_model"),
        "chosen_evidence_support": row.get("chosen_evidence_support"),
        "rejected_evidence_support": row.get("rejected_evidence_support"),
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

    for row in stage2_rows:
        record = containment_record(row, args.min_shorter_words, args.include_overlap_span, args.min_overlap_words)
        if record is None:
            rejected_records.append({
                "pair_id": row.get("pair_id"),
                "question_id": row.get("question_id"),
                "chosen_candidate": row.get("chosen_candidate") or row.get("chosen"),
                "rejected_candidate": row.get("rejected_candidate") or row.get("rejected"),
                "chosen_norm": normalize_answer(row.get("chosen_candidate") or row.get("chosen")),
                "rejected_norm": normalize_answer(row.get("rejected_candidate") or row.get("rejected")),
            })
            continue
        new_row = dict(row)
        new_row["stage2_containment_filter"] = {
            "containment_direction": record["containment_direction"],
            "chosen_norm": record["chosen_norm"],
            "rejected_norm": record["rejected_norm"],
            "chosen_words": record["chosen_words"],
            "rejected_words": record["rejected_words"],
            "min_shorter_words": args.min_shorter_words,
            "include_overlap_span": args.include_overlap_span,
            "min_overlap_words": args.min_overlap_words,
            "merged_span_norm": record["merged_span_norm"],
            "overlap_words": record["overlap_words"],
            "overlap_text": record["overlap_text"],
            "overlap_direction": record["overlap_direction"],
            "merged_span_in_snippets": record["merged_span_in_snippets"],
        }
        selected_rows.append(new_row)
        selected_records.append(record)

    shutil.copy2(stage1_src, output_root / STAGE1_FILE)
    shutil.copy2(stage3_src, output_root / STAGE3_FILE)
    write_jsonl(output_root / STAGE2_FILE, selected_rows)

    write_csv((output_root / STAGE1_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE1_FILE))
    write_csv((output_root / STAGE2_FILE).with_suffix(".csv"), selected_rows)
    write_csv((output_root / STAGE3_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE3_FILE))
    write_csv(output_root / "stage2_containment_selected_records.csv", selected_records)
    write_jsonl(output_root / "stage2_containment_selected_records.jsonl", selected_records)
    write_csv(output_root / "stage2_containment_rejected_records.csv", rejected_records)

    summary = {
        "description": "Staged curriculum with Stage 2 filtered to C3>C2 containment or snippet-backed overlap pairs. Stage 1 and Stage 3 are unchanged.",
        "source_staged_root": str(source_root),
        "min_shorter_words": args.min_shorter_words,
        "include_overlap_span": args.include_overlap_span,
        "min_overlap_words": args.min_overlap_words,
        "source_stage2_pairs": len(stage2_rows),
        "source_stage2_questions": len({row.get("question_id") for row in stage2_rows}),
        "selected_stage2_pairs": len(selected_rows),
        "selected_stage2_questions": len({row.get("question_id") for row in selected_rows}),
        "removed_stage2_pairs": len(rejected_records),
        "containment_direction_counts": dict(Counter(record["containment_direction"] for record in selected_records)),
        "rejected_source_model_counts": dict(Counter(record.get("rejected_source_model") for record in selected_records)),
        "stages": {
            STAGE1_FILE: stage_summary(output_root / STAGE1_FILE),
            STAGE2_FILE: stage_summary(output_root / STAGE2_FILE),
            STAGE3_FILE: stage_summary(output_root / STAGE3_FILE),
        },
        "audit_files": {
            "selected_csv": str(output_root / "stage2_containment_selected_records.csv"),
            "selected_jsonl": str(output_root / "stage2_containment_selected_records.jsonl"),
            "rejected_csv": str(output_root / "stage2_containment_rejected_records.csv"),
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
    parser.add_argument("--min-shorter-words", type=int, default=1, help="Minimum word count for the shorter normalized answer.")
    parser.add_argument("--include-overlap-span", action=argparse.BooleanOptionalAction, default=True, help="Include prefix/suffix overlap pairs if their merged span occurs in snippets.")
    parser.add_argument("--min-overlap-words", type=int, default=1, help="Minimum shared prefix/suffix words for overlap pairs.")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
