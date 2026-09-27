from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import exact_answer_groups, normalize_for_match
from src.utility.data import clean_text
from src.utility.eval_types import EvalExample


DEFAULT_INPUT_PATH = "data/training13b.json"
DEFAULT_OUTPUT_PATH = "data/training13b_list_full_gold_in_snippets.json"
DEFAULT_SUMMARY_PATH = "data/training13b_list_full_gold_in_snippets.summary.json"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a raw BioASQ dataset down to list questions whose full gold "
            "answer coverage is visible in the question snippets."
        )
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Project-relative or absolute path to the raw BioASQ JSON file.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Project-relative or absolute path for the filtered BioASQ JSON file.",
    )
    parser.add_argument(
        "--summary-path",
        default=DEFAULT_SUMMARY_PATH,
        help="Project-relative or absolute path for the filtering summary JSON file.",
    )
    parser.add_argument(
        "--question-type",
        default="list",
        choices=["list", "factoid", "yesno"],
        help="Question type to evaluate for snippet-level exact-answer coverage.",
    )
    parser.add_argument(
        "--min-gold-coverage",
        type=float,
        default=1.0,
        help="Minimum fraction of gold answer groups that must appear in the snippets.",
    )
    return parser.parse_args()


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


def iter_snippet_texts(question: dict[str, Any]) -> Iterable[str]:
    for snippet in question.get("snippets", []):
        if not isinstance(snippet, dict):
            continue
        text = clean_text(snippet.get("text", ""))
        if text:
            yield text


def gold_coverage_ratio(question: dict[str, Any], question_type: str) -> tuple[float, int, int]:
    gold_groups = exact_answer_groups(make_eval_example(question, question_type), question_type)
    if not gold_groups:
        return 0.0, 0, 0

    snippet_text = "\n".join(iter_snippet_texts(question))
    normalized_snippet_text = normalize_for_match(snippet_text)
    if not normalized_snippet_text:
        return 0.0, 0, len(gold_groups)

    visible_groups = 0
    for gold_group in gold_groups:
        for alias in gold_group:
            normalized_alias = normalize_for_match(clean_text(alias))
            if normalized_alias and normalized_alias in normalized_snippet_text:
                visible_groups += 1
                break

    return visible_groups / len(gold_groups), visible_groups, len(gold_groups)


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input_path)
    output_path = resolve_project_path(args.output_path)
    summary_path = resolve_project_path(args.summary_path)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Raw BioASQ file must contain a 'questions' list: {input_path}")

    target_type = clean_text(args.question_type).lower()
    kept_questions: list[dict[str, Any]] = []
    coverage_distribution: Counter[str] = Counter()
    total_target_questions = 0
    kept_target_questions = 0
    dropped_target_questions = 0

    for question in questions:
        if not isinstance(question, dict):
            continue

        question_type = clean_text(question.get("type", "")).lower()
        if question_type != target_type:
            continue

        total_target_questions += 1
        coverage_ratio, visible_groups, total_groups = gold_coverage_ratio(question, target_type)
        coverage_distribution[f"{coverage_ratio:.6f}"] += 1
        if coverage_ratio >= args.min_gold_coverage:
            kept_target_questions += 1
            kept_questions.append(question)
        else:
            dropped_target_questions += 1

    filtered_payload = dict(payload)
    filtered_payload["questions"] = kept_questions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(filtered_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "question_type": target_type,
        "min_gold_coverage": args.min_gold_coverage,
        "total_questions_in_input": len([question for question in questions if isinstance(question, dict)]),
        "total_target_questions": total_target_questions,
        "kept_target_questions": kept_target_questions,
        "dropped_target_questions": dropped_target_questions,
        "coverage_distribution": dict(sorted(coverage_distribution.items(), key=lambda item: float(item[0]), reverse=True)),
        "coverage_definition": (
            "A gold group counts as visible if any alias from its exact_answer group "
            "appears in the normalized concatenation of all snippet texts."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {kept_target_questions:,} {target_type} questions to {output_path}")
    print(f"Saved filtering summary to {summary_path}")


if __name__ == "__main__":
    main()
