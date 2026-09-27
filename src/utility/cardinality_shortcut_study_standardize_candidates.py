from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, write_json
from cse_dpo.match_gold_groups import (
    build_matched_response,
    load_candidate_bank_records,
    load_question_examples,
)
from cse_dpo.normalize_set_answers import normalize_answer_surface
from cse_dpo.schemas import MatchedResponse
from src.utility.data import clean_text


DEFAULT_OUTPUT_DIR = "data/Cardinality_Shortcut_Study/standardized_candidates"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize candidate outputs for the Cardinality Shortcut Study into "
            "one row per candidate with frozen split labels and entity-level metrics."
        )
    )
    parser.add_argument(
        "--question-input",
        nargs="+",
        required=True,
        help="Raw BioASQ JSON or prepared JSON files containing the gold questions.",
    )
    parser.add_argument(
        "--candidate-input",
        nargs="+",
        required=True,
        help=(
            "Candidate-bank JSONL or saved predictions JSON files to standardize. "
            "Multiple files are allowed."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        required=True,
        help=(
            "CSV manifest produced by cardinality_shortcut_study_freeze_question_splits.py "
            "that maps question_id to train/validation/test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Project-relative or absolute output directory for standardized artifacts.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label used when loading the question examples.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=5,
        help="Resource cap used when reconstructing the gold question examples.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Resource text truncation cap used when reconstructing the gold question examples.",
    )
    parser.add_argument(
        "--gold-support-policy",
        default="all",
        choices=["all", "snippet"],
        help="Gold support policy passed through to the BioASQ gold-group loader.",
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing for untagged outputs.",
    )
    parser.add_argument(
        "--allow-fallback-pair-construction",
        action="store_true",
        help="Mark fallback-split responses as pair-eligible for downstream pair construction.",
    )
    return parser.parse_args()


def load_split_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"question_id", "split"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Split manifest must contain columns {sorted(required_fields)}: {path}"
            )
        for row in reader:
            question_id = clean_text(row.get("question_id", ""))
            split_name = clean_text(row.get("split", "")).lower()
            if not question_id or not split_name:
                continue
            if split_name not in {"train", "validation", "test"}:
                raise ValueError(f"Unsupported split label '{split_name}' in {path}")
            previous = assignments.get(question_id)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Question id '{question_id}' appears with conflicting splits "
                    f"('{previous}' and '{split_name}') in {path}"
                )
            assignments[question_id] = split_name
    if not assignments:
        raise ValueError(f"No split assignments were loaded from {path}")
    return assignments


def detect_candidate_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "candidate_bank"
    rows = load_json_records(path)
    if not rows:
        return "empty"
    sample = rows[0]
    if "raw_output" in sample or "response_id" in sample:
        return "candidate_bank"
    if "prediction" in sample:
        return "predictions"
    return "unknown"


def prediction_rows_to_candidate_bank_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    default_generator_checkpoint: str | None = None,
) -> int:
    rows = load_json_records(input_path)
    converted_rows: list[dict[str, Any]] = []
    per_question_counters: Counter[str] = Counter()

    for row in rows:
        question_id = clean_text(row.get("question_id") or row.get("id"))
        question_type = clean_text(row.get("question_type") or row.get("type") or "list").lower()
        prediction = clean_text(row.get("prediction"))
        if question_type != "list" or not question_id or not prediction:
            continue

        sample_id = per_question_counters[question_id]
        per_question_counters[question_id] += 1
        response_id = clean_text(row.get("response_id")) or f"{question_id}-prediction{sample_id}"
        generation_samples = row.get("generation_samples")
        generated_token_count = None
        if isinstance(generation_samples, Sequence) and not isinstance(generation_samples, (str, bytes)):
            sample_count = len(generation_samples)
            if sample_count > 0:
                generated_token_count = row.get("generated_token_count")

        converted_rows.append(
            {
                "dataset": clean_text(row.get("dataset") or "bioasq") or "bioasq",
                "question_id": question_id,
                "question_type": question_type,
                "sample_id": sample_id,
                "prompt": clean_text(row.get("prompt")),
                "question_text": clean_text(row.get("body") or row.get("question_text")),
                "evidence": row.get("evidence") if isinstance(row.get("evidence"), list) else [],
                "raw_output": prediction,
                "generated_token_count": generated_token_count,
                "generator_checkpoint": clean_text(
                    row.get("generator_checkpoint")
                    or row.get("model_ref")
                    or row.get("model")
                    or default_generator_checkpoint
                    or input_path.stem
                ),
                "response_id": response_id,
                "source_path": clean_text(row.get("source_path")) or str(input_path),
                "prompt_instruction": clean_text(row.get("prompt_instruction")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in converted_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return len(converted_rows)


def materialize_candidate_inputs(
    candidate_inputs: Sequence[str],
    output_dir: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    standardized_inputs: list[str] = []
    input_manifest: list[dict[str, Any]] = []
    converted_dir = output_dir / "_converted_candidate_inputs"
    converted_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in candidate_inputs:
        path = resolve_project_path(raw_path)
        format_name = detect_candidate_format(path)
        if format_name == "candidate_bank":
            standardized_inputs.append(str(path))
            input_manifest.append(
                {
                    "input_path": str(path),
                    "resolved_candidate_path": str(path),
                    "detected_format": format_name,
                    "converted_row_count": None,
                }
            )
            continue

        if format_name == "predictions":
            converted_path = converted_dir / f"{path.stem}.candidate_bank.jsonl"
            converted_count = prediction_rows_to_candidate_bank_jsonl(path, converted_path)
            standardized_inputs.append(str(converted_path))
            input_manifest.append(
                {
                    "input_path": str(path),
                    "resolved_candidate_path": str(converted_path),
                    "detected_format": format_name,
                    "converted_row_count": converted_count,
                }
            )
            continue

        raise ValueError(f"Unsupported candidate input format for {path} (detected as {format_name}).")

    return standardized_inputs, input_manifest


def split_output_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / f"standardized_candidates_{split_name}.jsonl"


def semantic_items(response: MatchedResponse) -> tuple[str, ...]:
    return tuple(candidate.surface for candidate in response.candidates)


def normalized_semantic_items(response: MatchedResponse) -> tuple[str, ...]:
    normalized_items: list[str] = []
    for item in semantic_items(response):
        normalized = normalize_answer_surface(item)
        if normalized:
            normalized_items.append(normalized)
    return tuple(normalized_items)


def matched_surface_items(response: MatchedResponse) -> tuple[str, ...]:
    return tuple(
        candidate.surface
        for candidate in response.candidates
        if candidate.matched_gold_group_id is not None
    )


def unmatched_surface_items(response: MatchedResponse) -> tuple[str, ...]:
    return tuple(
        candidate.surface
        for candidate in response.candidates
        if candidate.matched_gold_group_id is None
    )


def missing_gold_aliases(response: MatchedResponse, question: Any) -> tuple[str, ...]:
    missing: list[str] = []
    missing_ids = set(response.missing_gold_group_ids)
    for gold_group in question.gold_groups:
        if gold_group.group_id in missing_ids:
            missing.append(gold_group.canonical_alias)
    return tuple(missing)


def build_standardized_row(
    *,
    response: MatchedResponse,
    question: Any,
    split_name: str,
    candidate_source_path: str,
) -> dict[str, Any]:
    metrics = response.metrics
    return {
        "study_name": "Cardinality Shortcut Study",
        "question_id": question.question_id,
        "split": split_name,
        "dataset": question.dataset,
        "question_type": question.question_type,
        "question_text": question.question_text,
        "question_source_path": question.source_path,
        "candidate_source_path": candidate_source_path,
        "generator_checkpoint": response.record.generator_checkpoint,
        "response_id": response.record.response_id,
        "sample_id": response.record.sample_id,
        "prompt": response.record.prompt,
        "prompt_instruction": response.record.prompt_instruction,
        "raw_output": response.record.raw_output,
        "parser_status": response.parsed.status,
        "parser_warnings": list(response.parsed.warnings),
        "pair_eligible": response.pair_eligible,
        "pair_exclusion_reason": response.pair_exclusion_reason,
        "used_fallback_split": response.parsed.used_fallback_split,
        "begin_tag_count": response.parsed.begin_tag_count,
        "end_tag_count": response.parsed.end_tag_count,
        "empty_item_count": response.parsed.empty_item_count,
        "dropped_placeholder_count": response.parsed.dropped_placeholder_count,
        "generated_token_count": response.record.generated_token_count,
        "gold_entity_count": len(question.gold_groups),
        "entity_count": metrics.prediction_count,
        "token_length_chars": len(response.record.raw_output),
        "normalized_entity_set": list(normalized_semantic_items(response)),
        "predicted_entities": list(semantic_items(response)),
        "matched_entities": list(matched_surface_items(response)),
        "unmatched_entities": list(unmatched_surface_items(response)),
        "missing_gold_entities": list(missing_gold_aliases(response, question)),
        "matched_gold_group_ids": list(response.matched_gold_group_ids),
        "missing_gold_group_ids": list(response.missing_gold_group_ids),
        "tp": len(response.matched_gold_group_ids),
        "fp": metrics.prediction_count - len(response.matched_gold_group_ids),
        "fn": len(response.missing_gold_group_ids),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "invalid_addition_rate": metrics.invalid_addition_rate,
        "valid_omission_rate": metrics.valid_omission_rate,
        "has_duplicate_semantic_candidates": response.has_duplicate_semantic_candidates,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split") or "") for row in rows)
    parser_status_counts = Counter(str(row.get("parser_status") or "") for row in rows)
    checkpoint_counts = Counter(
        clean_text(str(row.get("generator_checkpoint") or "unknown")) or "unknown"
        for row in rows
    )
    f1_values = [float(row["f1"]) for row in rows if isinstance(row.get("f1"), (int, float))]
    entity_counts = [int(row["entity_count"]) for row in rows if isinstance(row.get("entity_count"), int)]
    gold_entity_counts = [int(row["gold_entity_count"]) for row in rows if isinstance(row.get("gold_entity_count"), int)]
    return {
        "row_count": len(rows),
        "split_distribution": dict(sorted(split_counts.items())),
        "parser_status_distribution": dict(sorted(parser_status_counts.items())),
        "generator_checkpoint_distribution": dict(sorted(checkpoint_counts.items())),
        "pair_eligible_count": sum(1 for row in rows if bool(row.get("pair_eligible"))),
        "fallback_split_count": sum(1 for row in rows if bool(row.get("used_fallback_split"))),
        "mean_f1": round(statistics.mean(f1_values), 6) if f1_values else None,
        "mean_entity_count": round(statistics.mean(entity_counts), 6) if entity_counts else None,
        "mean_gold_entity_count": round(statistics.mean(gold_entity_counts), 6) if gold_entity_counts else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    split_manifest_path = resolve_project_path(args.split_manifest)
    split_assignments = load_split_assignments(split_manifest_path)

    candidate_paths, input_manifest = materialize_candidate_inputs(args.candidate_input, output_dir)
    questions_by_id = load_question_examples(
        args.question_input,
        dataset_name=args.dataset_name,
        max_resources=args.max_resources,
        max_resource_chars=args.max_resource_chars,
        gold_support_policy=args.gold_support_policy,
    )
    candidate_records = load_candidate_bank_records(
        candidate_paths,
        questions_by_id=questions_by_id,
        dataset_name=args.dataset_name,
    )

    standardized_rows: list[dict[str, Any]] = []
    dropped_questions_without_split = 0
    for record in candidate_records:
        question = questions_by_id.get(record.question_id)
        if question is None:
            continue
        split_name = split_assignments.get(record.question_id)
        if split_name is None:
            dropped_questions_without_split += 1
            continue

        response = build_matched_response(
            record,
            question,
            allow_fallback_split=args.allow_fallback_split,
            allow_fallback_pair_construction=args.allow_fallback_pair_construction,
        )
        standardized_rows.append(
            build_standardized_row(
                response=response,
                question=question,
                split_name=split_name,
                candidate_source_path=record.source_path or "",
            )
        )

    standardized_rows.sort(
        key=lambda row: (
            str(row.get("split") or ""),
            str(row.get("question_id") or ""),
            str(row.get("generator_checkpoint") or ""),
            int(row.get("sample_id") or 0),
            str(row.get("response_id") or ""),
        )
    )

    combined_path = output_dir / "standardized_candidates_all.jsonl"
    train_path = split_output_path(output_dir, "train")
    validation_path = split_output_path(output_dir, "validation")
    test_path = split_output_path(output_dir, "test")
    summary_path = output_dir / "standardization_summary.json"
    inputs_manifest_path = output_dir / "candidate_input_manifest.json"

    write_jsonl(combined_path, standardized_rows)
    for split_name, split_path in (
        ("train", train_path),
        ("validation", validation_path),
        ("test", test_path),
    ):
        write_jsonl(split_path, [row for row in standardized_rows if row["split"] == split_name])

    summary = {
        "study_name": "Cardinality Shortcut Study",
        "step_name": "Candidate standardisation",
        "question_input": [str(resolve_project_path(path_value)) for path_value in args.question_input],
        "candidate_input": input_manifest,
        "split_manifest": str(split_manifest_path),
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "max_resources": args.max_resources,
        "max_resource_chars": args.max_resource_chars,
        "gold_support_policy": args.gold_support_policy,
        "allow_fallback_split": args.allow_fallback_split,
        "allow_fallback_pair_construction": args.allow_fallback_pair_construction,
        "dropped_questions_without_split": dropped_questions_without_split,
        "files": {
            "all": str(combined_path),
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "candidate_input_manifest": str(inputs_manifest_path),
        },
        "summary": summarize_rows(standardized_rows),
        "split_summaries": {
            split_name: summarize_rows([row for row in standardized_rows if row["split"] == split_name])
            for split_name in ("train", "validation", "test")
        },
    }
    write_json(inputs_manifest_path, input_manifest)
    write_json(summary_path, summary)

    print("Saved Cardinality Shortcut Study standardized candidates:")
    print(f"  all:        {combined_path}")
    print(f"  train:      {train_path}")
    print(f"  validation: {validation_path}")
    print(f"  test:       {test_path}")
    print(f"  summary:    {summary_path}")
    print(f"  rows:       {len(standardized_rows):,}")


if __name__ == "__main__":
    main()
