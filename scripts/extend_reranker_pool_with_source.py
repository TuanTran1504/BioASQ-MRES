#!/usr/bin/env python3
"""Add one candidate source to an existing, officially labeled reranker pool.

The base pool is assumed to have already been scored by the official BioASQ
matcher. Only genuinely new question/answer pairs from the added source are
sent to the matcher; existing labels and provenance are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_workflows.coverage_comparison import official_candidate_matches
from scripts.train_candidate_reranker_pilot import (
    clean,
    conservative_surface_variants,
    key,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_BASE_POOL = ROOT / (
    "Artifacts/reranker_pilot/20261002-023517-tfidf-logistic/"
    "candidate_pool_labeled.jsonl"
)
DEFAULT_EXAMPLES = ROOT / (
    "gadi_sft_8b_starter/outputs/model_comparison/"
    "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/examples.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "Artifacts/reranker_pilot"
DEFAULT_JAR = ROOT / (
    "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/"
    "BioASQEvaluation.jar"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_base_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one row from the existing labeled pool."""
    required = {"question_id", "answer", "sources", "source_ranks", "label"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Base pool row is missing fields: {sorted(missing)}")
    normalized = dict(row)
    normalized["question_id"] = str(row["question_id"])
    normalized["answer"] = clean(row["answer"])
    normalized["sources"] = [str(value) for value in row["sources"]]
    normalized["source_ranks"] = {
        str(source): int(rank) for source, rank in row["source_ranks"].items()
    }
    normalized["relation_types"] = list(row.get("relation_types", []))
    normalized["surface_operations"] = list(row.get("surface_operations", []))
    normalized["is_format_variant"] = bool(row.get("is_format_variant", False))
    normalized["label"] = int(bool(row["label"]))
    return normalized


def source_rows(path: Path, source: str, *, add_format_variants: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate source {source}: {path}")
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(path):
        answer = clean(raw.get("answer"))
        if not answer:
            continue
        if "question_id" not in raw:
            raise ValueError(f"Candidate source {source} has a row without question_id")
        position = raw.get("position", raw.get("rank", 1))
        row = {
            "question_id": str(raw["question_id"]),
            "answer": answer,
            "source": source,
            "source_rank": int(position),
            "relation_type": raw.get("relation_type") or raw.get("candidate_type") or "unknown",
            "is_format_variant": False,
            "surface_operation": None,
        }
        rows.append(row)
    if add_format_variants:
        originals = list(rows)
        for row in originals:
            for variant, operation in conservative_surface_variants(row["answer"]):
                rows.append({
                    **row,
                    "answer": variant,
                    "source": f"format::{source}",
                    "is_format_variant": True,
                    "surface_operation": operation,
                })
    return rows


def merge_metadata(target: dict[str, Any], row: dict[str, Any]) -> None:
    source = row["source"]
    if source not in target["sources"]:
        target["sources"].append(source)
    old_rank = target["source_ranks"].get(source)
    target["source_ranks"][source] = min(int(row["source_rank"]), int(old_rank or row["source_rank"]))
    relation = row.get("relation_type", "unknown")
    if relation not in target["relation_types"]:
        target["relation_types"].append(relation)
    operation = row.get("surface_operation")
    if operation and operation not in target["surface_operations"]:
        target["surface_operations"].append(operation)
    # Match the existing pool builder: a merged candidate is a format variant
    # only if every provenance path is a format variant.
    target["is_format_variant"] = target["is_format_variant"] and bool(row["is_format_variant"])


def merge_pool_metadata(target: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge a deduplicated pool-shaped row into another pool-shaped row."""
    for source, rank in row["source_ranks"].items():
        if source not in target["sources"]:
            target["sources"].append(source)
        old_rank = target["source_ranks"].get(source)
        target["source_ranks"][source] = min(int(rank), int(old_rank or rank))
    for relation in row.get("relation_types", []):
        if relation not in target["relation_types"]:
            target["relation_types"].append(relation)
    for operation in row.get("surface_operations", []):
        if operation not in target["surface_operations"]:
            target["surface_operations"].append(operation)
    target["is_format_variant"] = target["is_format_variant"] and bool(row["is_format_variant"])


def deduplicate_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        identity = (row["question_id"], key(row["answer"]))
        if identity not in merged:
            merged[identity] = {
                "question_id": row["question_id"],
                "answer": row["answer"],
                "sources": [],
                "source_ranks": {},
                "relation_types": [],
                "surface_operations": [],
                "is_format_variant": bool(row["is_format_variant"]),
            }
        merge_metadata(merged[identity], row)
    return list(merged.values())


def add_source_to_pool(
    base_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    matches: dict[tuple[str, str], bool],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in base_rows:
        identity = (row["question_id"], key(row["answer"]))
        if identity in merged:
            raise ValueError(f"Duplicate identity in base pool: {identity}")
        merged[identity] = row

    stats = Counter()
    for row in added_rows:
        identity = (row["question_id"], key(row["answer"]))
        if identity in merged:
            merge_pool_metadata(merged[identity], row)
            stats["merged_existing_candidates"] += 1
            continue
        row["label"] = int(matches[identity])
        merged[identity] = row
        stats["new_unique_candidates"] += 1
        stats["new_positive_candidates"] += int(row["label"])
    return list(merged.values()), dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, default=DEFAULT_BASE_POOL)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--java-jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--skip-format-variants", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_]+", args.source_name):
        raise ValueError("--source-name must contain only letters, digits and underscores")
    base_path = args.base_pool.expanduser()
    examples_path = args.examples.expanduser()
    source_path = args.source_candidates.expanduser()
    output_root = args.output_root.expanduser()
    jar_path = args.java_jar.expanduser()
    for path in (base_path, examples_path, source_path, jar_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    examples_list = read_jsonl(examples_path)
    examples = {str(row["question_id"]): row for row in examples_list}
    if len(examples) != len(examples_list):
        raise ValueError("Examples contain duplicate question IDs")

    base_rows = [normalize_base_row(row) for row in read_jsonl(base_path)]
    base_ids = {row["question_id"] for row in base_rows}
    if base_ids != set(examples):
        missing = sorted(set(examples) - base_ids)
        extra = sorted(base_ids - set(examples))
        raise ValueError(f"Base pool and examples disagree; missing={missing[:3]}, extra={extra[:3]}")

    raw_added = source_rows(
        source_path,
        args.source_name,
        add_format_variants=not args.skip_format_variants,
    )
    unknown_ids = sorted({row["question_id"] for row in raw_added} - set(examples))
    if unknown_ids:
        raise ValueError(f"Added source contains unknown question IDs: {unknown_ids[:5]}")
    added_rows = deduplicate_source_rows(raw_added)
    base_keys = {(row["question_id"], key(row["answer"])) for row in base_rows}
    rows_to_label = [row for row in added_rows if (row["question_id"], key(row["answer"])) not in base_keys]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = output_root / f"{timestamp}-extend-{args.source_name}"
    output.mkdir(parents=True, exist_ok=False)
    scorer_dir = output / "official_labels_new_source"
    scorer_dir.mkdir()

    matches = official_candidate_matches(examples_list, rows_to_label, scorer_dir, jar_path=jar_path) if rows_to_label else {}
    merged_rows, merge_stats = add_source_to_pool(base_rows, added_rows, matches)
    merged_rows.sort(key=lambda row: (row["question_id"], row["answer"].casefold()))
    base_labels = {
        (row["question_id"], key(row["answer"])): int(row["label"])
        for row in base_rows
    }

    positive_by_question = {
        qid for qid in examples if any(row["question_id"] == qid and row["label"] for row in merged_rows)
    }
    new_positive_questions = {
        row["question_id"] for row in rows_to_label if matches[(row["question_id"], row["answer"])]
    }
    summary = {
        "status": "complete",
        "method": "extend existing officially labeled reranker pool",
        "base_pool": str(base_path),
        "base_pool_sha256": sha256_file(base_path),
        "examples": str(examples_path),
        "examples_sha256": sha256_file(examples_path),
        "source_name": args.source_name,
        "source_candidates": str(source_path),
        "source_candidates_sha256": sha256_file(source_path),
        "format_variants": not args.skip_format_variants,
        "base_candidate_count": len(base_rows),
        "source_rows_before_deduplication": len(raw_added),
        "source_unique_candidates": len(added_rows),
        "source_new_candidates_to_score": len(rows_to_label),
        "new_positive_candidates": int(sum(matches.values())),
        "new_positive_questions": len(new_positive_questions),
        "new_positive_question_ids": sorted(new_positive_questions),
        "merged_candidate_count": len(merged_rows),
        "merged_positive_candidate_count": int(sum(row["label"] for row in merged_rows)),
        "answerable_pool_count": len(positive_by_question),
        "unanswerable_pool_count": len(examples) - len(positive_by_question),
        "merge_stats": merge_stats,
        "scorer": "official BioASQ Java matcher, per candidate",
    }
    write_jsonl(output / "candidate_pool_labeled.jsonl", merged_rows)
    write_jsonl(output / "new_source_labeled.jsonl", [
        {
            **row,
            "label": int(matches.get(
                (row["question_id"], row["answer"]),
                base_labels[(row["question_id"], key(row["answer"]))],
            )),
        }
        for row in added_rows
    ])
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()
