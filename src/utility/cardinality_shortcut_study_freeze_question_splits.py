from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import exact_answer_groups
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import clean_text
from src.utility.eval_types import EvalExample


DEFAULT_OUTPUT_DIR = "data/Cardinality_Shortcut_Study/question_splits"


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    question_type: str
    source_file: str
    gold_entity_count: int | None
    question: dict[str, Any]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze question-level train/validation/test splits for the Cardinality "
            "Shortcut Study before constructing any candidates or preference pairs."
        )
    )
    parser.add_argument(
        "--train-pool-input",
        nargs="+",
        required=True,
        help=(
            "One or more raw BioASQ JSON files that define the question pool used to "
            "sample the train/validation split."
        ),
    )
    parser.add_argument(
        "--test-input",
        nargs="*",
        default=None,
        help=(
            "Optional raw BioASQ JSON files to use as a fixed held-out test split. "
            "If omitted, the test split is sampled from --train-pool-input using --test-ratio."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Project-relative or absolute output directory for the frozen split files.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["list"],
        choices=sorted(QUESTION_INSTRUCTIONS.keys()),
        help="Question types to retain before splitting. Defaults to list questions.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of the non-test pool assigned to validation. "
            "This is applied after any fixed held-out test split is removed."
        ),
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of the train pool assigned to test when --test-input is omitted. "
            "Ignored if a fixed test split is provided."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for random splitting.",
    )
    return parser.parse_args()


def read_question_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Raw BioASQ file must contain a 'questions' list: {path}")
    return [question for question in questions if isinstance(question, dict)]


def make_eval_example(question: dict[str, Any], question_type: str) -> EvalExample:
    return EvalExample(
        question_id=clean_text(question.get("id", "")),
        question_type=question_type,
        body=clean_text(question.get("body", "")),
        instruction="",
        resources=(),
        gold_output="",
        source_path="",
        raw_question=question,
    )


def gold_entity_count(question: dict[str, Any], question_type: str) -> int | None:
    if question_type not in {"list", "factoid"}:
        return None
    groups = exact_answer_groups(make_eval_example(question, question_type), question_type)
    return len(groups)


def normalize_question_types(question_types: Iterable[str]) -> set[str]:
    normalized = {clean_text(question_type).lower() for question_type in question_types if clean_text(question_type)}
    if not normalized:
        raise ValueError("At least one non-empty question type must be provided.")
    return normalized


def load_question_records(
    paths: Sequence[str],
    *,
    allowed_types: set[str],
    split_label: str,
) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    seen_ids: dict[str, str] = {}
    for raw_path in paths:
        path = resolve_project_path(raw_path)
        for question in read_question_payload(path):
            question_type = clean_text(question.get("type", "")).lower()
            if question_type not in allowed_types:
                continue
            question_id = clean_text(question.get("id", ""))
            if not question_id:
                raise ValueError(f"Encountered a question without an id in {path}")
            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question id '{question_id}' found in {path} and {seen_ids[question_id]} "
                    f"while loading the {split_label} split source."
                )
            seen_ids[question_id] = str(path)
            records.append(
                QuestionRecord(
                    question_id=question_id,
                    question_type=question_type,
                    source_file=str(path),
                    gold_entity_count=gold_entity_count(question, question_type),
                    question=dict(question),
                )
            )
    return records


def stratify_labels(records: Sequence[QuestionRecord]) -> list[str] | None:
    labels = [record.question_type for record in records]
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return None
    if any(labels.count(label) < 2 for label in unique_labels):
        return None
    return labels


def split_records(
    records: Sequence[QuestionRecord],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[QuestionRecord], list[QuestionRecord], bool]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("Split ratios must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("Need at least 2 questions to create a split.")

    from sklearn.model_selection import train_test_split

    labels = stratify_labels(records)
    try:
        kept, holdout = train_test_split(
            list(records),
            test_size=holdout_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        return list(kept), list(holdout), labels is not None
    except ValueError:
        kept, holdout = train_test_split(
            list(records),
            test_size=holdout_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
        return list(kept), list(holdout), False


def sort_records(records: Sequence[QuestionRecord]) -> list[QuestionRecord]:
    return sorted(records, key=lambda record: record.question_id)


def build_question_payload(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    return {"questions": [record.question for record in records]}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest_csv(path: Path, split_records_map: dict[str, list[QuestionRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "split",
                "question_type",
                "source_file",
                "gold_entity_count",
            ],
        )
        writer.writeheader()
        for split_name, records in split_records_map.items():
            for record in records:
                writer.writerow(
                    {
                        "question_id": record.question_id,
                        "split": split_name,
                        "question_type": record.question_type,
                        "source_file": record.source_file,
                        "gold_entity_count": record.gold_entity_count,
                    }
                )


def split_stats(records: Sequence[QuestionRecord]) -> dict[str, Any]:
    type_distribution = Counter(record.question_type for record in records)
    source_distribution = Counter(record.source_file for record in records)
    gold_counts = [record.gold_entity_count for record in records if record.gold_entity_count is not None]
    gold_distribution = Counter(str(value) for value in gold_counts)
    return {
        "question_count": len(records),
        "question_type_distribution": dict(sorted(type_distribution.items())),
        "source_file_distribution": dict(sorted(source_distribution.items())),
        "gold_entity_count_distribution": dict(sorted(gold_distribution.items(), key=lambda item: int(item[0]))),
        "mean_gold_entity_count": (
            round(statistics.mean(gold_counts), 4)
            if gold_counts
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    allowed_types = normalize_question_types(args.question_types)
    output_dir = resolve_project_path(args.output_dir)

    train_pool_records = load_question_records(
        args.train_pool_input,
        allowed_types=allowed_types,
        split_label="train-pool",
    )
    if not train_pool_records:
        raise ValueError("No questions remained in --train-pool-input after filtering by question type.")

    fixed_test_records: list[QuestionRecord] = []
    fixed_test_paths = args.test_input or []
    if fixed_test_paths:
        fixed_test_records = load_question_records(
            fixed_test_paths,
            allowed_types=allowed_types,
            split_label="test",
        )
        if not fixed_test_records:
            raise ValueError("No questions remained in --test-input after filtering by question type.")
        overlap = sorted(
            {record.question_id for record in train_pool_records}
            & {record.question_id for record in fixed_test_records}
        )
        if overlap:
            raise ValueError(
                "Question ids overlap between the train pool and fixed held-out test split: "
                + ", ".join(overlap[:10])
                + (" ..." if len(overlap) > 10 else "")
            )

    if fixed_test_records:
        train_records, validation_records, train_validation_stratified = split_records(
            train_pool_records,
            holdout_ratio=args.validation_ratio,
            seed=args.seed,
        )
        test_records = fixed_test_records
        train_test_stratified = None
        split_mode = "fixed_test_plus_random_validation"
    else:
        train_validation_pool, test_records, train_test_stratified = split_records(
            train_pool_records,
            holdout_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_records, validation_records, train_validation_stratified = split_records(
            train_validation_pool,
            holdout_ratio=args.validation_ratio,
            seed=args.seed + 1,
        )
        split_mode = "random_train_validation_test"

    split_records_map = {
        "train": sort_records(train_records),
        "validation": sort_records(validation_records),
        "test": sort_records(test_records),
    }

    train_path = output_dir / "train_questions.json"
    validation_path = output_dir / "validation_questions.json"
    test_path = output_dir / "test_questions.json"
    train_ids_path = output_dir / "train_question_ids.json"
    validation_ids_path = output_dir / "validation_question_ids.json"
    test_ids_path = output_dir / "test_question_ids.json"
    manifest_path = output_dir / "question_split_manifest.csv"
    summary_path = output_dir / "split_summary.json"

    write_json(train_path, build_question_payload(split_records_map["train"]))
    write_json(validation_path, build_question_payload(split_records_map["validation"]))
    write_json(test_path, build_question_payload(split_records_map["test"]))
    write_json(train_ids_path, [record.question_id for record in split_records_map["train"]])
    write_json(validation_ids_path, [record.question_id for record in split_records_map["validation"]])
    write_json(test_ids_path, [record.question_id for record in split_records_map["test"]])
    write_manifest_csv(manifest_path, split_records_map)

    summary = {
        "study_name": "Cardinality Shortcut Study",
        "step_name": "Freeze question-level train/validation/test ids before pair construction",
        "split_mode": split_mode,
        "train_pool_input": [str(resolve_project_path(path_value)) for path_value in args.train_pool_input],
        "test_input": [str(resolve_project_path(path_value)) for path_value in fixed_test_paths],
        "output_dir": str(output_dir),
        "question_types": sorted(allowed_types),
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": None if fixed_test_paths else args.test_ratio,
        "stratified_by_question_type": {
            "train_validation": train_validation_stratified,
            "train_test": train_test_stratified,
        },
        "counts": {split_name: len(records) for split_name, records in split_records_map.items()},
        "files": {
            "train_questions": str(train_path),
            "validation_questions": str(validation_path),
            "test_questions": str(test_path),
            "train_question_ids": str(train_ids_path),
            "validation_question_ids": str(validation_ids_path),
            "test_question_ids": str(test_ids_path),
            "question_split_manifest_csv": str(manifest_path),
        },
        "splits": {
            split_name: split_stats(records)
            for split_name, records in split_records_map.items()
        },
    }
    write_json(summary_path, summary)

    print("Saved Cardinality Shortcut Study question splits:")
    print(f"  train:      {len(split_records_map['train']):,} questions -> {train_path}")
    print(f"  validation: {len(split_records_map['validation']):,} questions -> {validation_path}")
    print(f"  test:       {len(split_records_map['test']):,} questions -> {test_path}")
    print(f"  summary:    {summary_path}")


if __name__ == "__main__":
    main()
