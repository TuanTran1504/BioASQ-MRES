#!/usr/bin/env python3
"""Validate inputs or summarize the reproduced best list-question result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = PROJECT_ROOT / "reproducibility/best_list_system/expected_result.json"
DEFAULT_TRAIN = PROJECT_ROOT / "data/training13b.json"
DEFAULT_TESTS = [
    PROJECT_ROOT / f"data/Task13BTest/13B{batch}_golden.json"
    for batch in range(1, 5)
]
DEFAULT_JAR = (
    PROJECT_ROOT
    / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prerequisites or report a reproduced BioASQ list score."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inputs = subparsers.add_parser("inputs", help="Validate raw BioASQ inputs and evaluator.")
    inputs.add_argument("--train-input", default=str(DEFAULT_TRAIN))
    inputs.add_argument("--test-input", nargs="+", default=[str(path) for path in DEFAULT_TESTS])
    inputs.add_argument("--evaluator-jar", default=str(DEFAULT_JAR))
    inputs.add_argument("--expected-train-list-questions", type=int, default=1047)
    inputs.add_argument("--expected-test-list-questions", type=int, default=83)

    result = subparsers.add_parser("result", help="Compare an official score with the historical run.")
    result.add_argument("--scores", required=True, help="Path to official_scores.json.")
    result.add_argument("--expected", default=str(DEFAULT_EXPECTED))
    result.add_argument(
        "--strict-tolerance",
        type=float,
        default=None,
        help="Fail when absolute mean-F1 deviation exceeds this value.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_list_questions(path: Path) -> int:
    payload = load_json(path)
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list):
        raise ValueError(f"Expected a BioASQ object with a questions list: {path}")
    return sum(
        1
        for question in questions
        if isinstance(question, dict) and str(question.get("type", "")).lower() == "list"
    )


def validate_inputs(args: argparse.Namespace) -> int:
    train_path = Path(args.train_input).expanduser().resolve()
    test_paths = [Path(value).expanduser().resolve() for value in args.test_input]
    jar_path = Path(args.evaluator_jar).expanduser().resolve()
    required = [train_path, *test_paths, jar_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        print("Missing required reproduction inputs:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        print(
            "Raw BioASQ data is intentionally excluded from Git. See DATA.md.",
            file=sys.stderr,
        )
        return 2

    train_count = count_list_questions(train_path)
    test_counts = [count_list_questions(path) for path in test_paths]
    test_total = sum(test_counts)
    payload = {
        "status": "ok",
        "train_input": str(train_path),
        "train_list_questions": train_count,
        "test_inputs": [str(path) for path in test_paths],
        "test_list_questions_by_batch": test_counts,
        "test_list_questions": test_total,
        "evaluator_jar": str(jar_path),
    }
    print(json.dumps(payload, indent=2))

    errors: list[str] = []
    if train_count != args.expected_train_list_questions:
        errors.append(
            f"train list count is {train_count}; expected {args.expected_train_list_questions}"
        )
    if test_total != args.expected_test_list_questions:
        errors.append(
            f"test list count is {test_total}; expected {args.expected_test_list_questions}"
        )
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 2
    return 0


def report_result(args: argparse.Namespace) -> int:
    scores_path = Path(args.scores).expanduser().resolve()
    expected_path = Path(args.expected).expanduser().resolve()
    scores = load_json(scores_path)
    expected = load_json(expected_path)["evaluation"]

    list_metrics = scores["aggregate"]["by_type"]["list"]["metrics"]
    observed = {
        "question_count": scores.get("question_count"),
        "mean_precision": float(list_metrics["mean_precision"]),
        "mean_recall": float(list_metrics["mean_recall"]),
        "mean_f1": float(list_metrics["mean_f1"]),
    }
    delta = observed["mean_f1"] - float(expected["mean_f1"])
    report = {
        "status": "complete",
        "scores": str(scores_path),
        "observed": observed,
        "historical": {
            key: expected[key]
            for key in ("question_count", "mean_precision", "mean_recall", "mean_f1")
        },
        "mean_f1_delta": delta,
    }
    print(json.dumps(report, indent=2))

    tolerance = args.strict_tolerance
    if tolerance is not None and abs(delta) > tolerance:
        print(
            f"ERROR: |mean_f1_delta|={abs(delta):.9f} exceeds tolerance {tolerance}",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "inputs":
        return validate_inputs(args)
    return report_result(args)


if __name__ == "__main__":
    raise SystemExit(main())
