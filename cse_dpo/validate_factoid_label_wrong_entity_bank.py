from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.utility.bioasq_format import normalize_for_match
from src.utility.data import clean_text
from src.utility.factoid_output_parsing import parse_factoid_candidates

from .common import load_json_records, summarize_numeric, write_json, write_jsonl
from .construct_factoid_label_wrong_entity_bank import (
    build_question_index,
    format_factoid,
    load_question_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze a factoid label-vs-wrong-entity question pool."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Question-level JSONL exported by construct_factoid_label_wrong_entity_bank.",
    )
    parser.add_argument(
        "--question-input",
        default=None,
        help=(
            "Prepared factoid JSON used to verify the complete accepted-gold list. "
            "Defaults to the shared question_source_path inside the input rows."
        ),
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Where to write the validation/freeze summary JSON.",
    )
    parser.add_argument(
        "--failures-jsonl",
        default=None,
        help="Optional JSONL file with one row per failing question.",
    )
    parser.add_argument(
        "--freeze-output-jsonl",
        default=None,
        help=(
            "Optional path to write the accepted frozen pool. Written only when all "
            "rows pass validation."
        ),
    )
    return parser.parse_args()


def expected_answer_stub(prompt: str) -> bool:
    normalized = clean_text(prompt)
    return normalized.endswith("Answer:") or normalized.endswith("# Answer:")


def resolve_question_input_path(rows: list[dict[str, Any]], explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    source_paths = {
        clean_text(row.get("question_source_path"))
        for row in rows
        if clean_text(row.get("question_source_path"))
    }
    if len(source_paths) != 1:
        raise ValueError(
            "Could not infer a unique question_input path from question_source_path fields."
        )
    return Path(next(iter(source_paths)))


def validate_row(
    row: dict[str, Any],
    *,
    source_question: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    question_id = clean_text(row.get("question_id"))
    prompt = str(row.get("prompt", ""))
    normalized_prompt = clean_text(prompt)
    question_text = clean_text(row.get("question_text"))
    snippets = [clean_text(value) for value in row.get("snippets", []) if clean_text(value)]

    canonical_gold_entity = clean_text(
        row.get("canonical_gold_entity") or row.get("chosen_entity")
    )
    canonical_gold_output = str(
        row.get("canonical_gold_output") or row.get("chosen_output") or format_factoid(canonical_gold_entity)
    )
    accepted_gold_entities = [
        clean_text(value)
        for value in row.get("accepted_gold_entities", row.get("gold_aliases", []))
        if clean_text(value)
    ]
    accepted_gold_outputs = [
        str(value)
        for value in row.get(
            "accepted_gold_outputs",
            [format_factoid(value) for value in accepted_gold_entities],
        )
        if clean_text(value)
    ]
    wrong_entities = [clean_text(value) for value in row.get("wrong_entities", []) if clean_text(value)]
    wrong_outputs = [
        str(value)
        for value in row.get("wrong_outputs", [format_factoid(value) for value in wrong_entities])
        if clean_text(value)
    ]

    source_accepted_gold_entities = list(source_question["gold_aliases"])
    source_accepted_gold_norms = {
        normalize_for_match(value) for value in source_accepted_gold_entities if normalize_for_match(value)
    }
    wrong_norms = [normalize_for_match(value) for value in wrong_entities if normalize_for_match(value)]

    parsed_canonical_output = parse_factoid_candidates(canonical_gold_output, parser_mode="current")
    parsed_wrong_outputs = [
        parse_factoid_candidates(output, parser_mode="current")
        for output in wrong_outputs
    ]
    parsed_accepted_gold_outputs = [
        parse_factoid_candidates(output, parser_mode="current")
        for output in accepted_gold_outputs
    ]
    parsed_combined_outputs = parse_factoid_candidates(
        " ".join([canonical_gold_output] + wrong_outputs),
        parser_mode="current",
    )

    checks = {
        "prompt_contains_question": bool(question_text and question_text in normalized_prompt),
        "prompt_contains_all_snippets": bool(snippets) and all(
            snippet in normalized_prompt for snippet in snippets
        ),
        "prompt_excludes_serialized_gold_labels": all(
            output not in prompt for output in accepted_gold_outputs
        ),
        "prompt_has_answer_stub_only": expected_answer_stub(prompt),
        "canonical_gold_present": bool(canonical_gold_entity),
        "canonical_gold_in_accepted_list": bool(canonical_gold_entity) and (
            normalize_for_match(canonical_gold_entity) in source_accepted_gold_norms
        ),
        "complete_accepted_gold_list_preserved": accepted_gold_entities == source_accepted_gold_entities,
        "exactly_four_wrong_entities": len(wrong_entities) == 4,
        "wrong_entities_distinct_normalized": len(set(wrong_norms)) == 4,
        "wrong_entities_do_not_match_any_accepted_gold_alias": all(
            normalized not in source_accepted_gold_norms for normalized in wrong_norms
        ),
        "canonical_output_parses": parsed_canonical_output == [canonical_gold_entity],
        "wrong_outputs_parse": len(parsed_wrong_outputs) == 4 and all(
            parsed == [entity]
            for parsed, entity in zip(parsed_wrong_outputs, wrong_entities)
        ),
        "accepted_gold_outputs_parse": len(parsed_accepted_gold_outputs) == len(accepted_gold_entities) and all(
            parsed == [entity]
            for parsed, entity in zip(parsed_accepted_gold_outputs, accepted_gold_entities)
        ),
        "combined_outputs_parse": parsed_combined_outputs == [canonical_gold_entity] + wrong_entities,
    }

    passed = all(checks.values())
    details = {
        "question_id": question_id,
        "question_text": question_text,
        "canonical_gold_entity": canonical_gold_entity,
        "accepted_gold_entities": accepted_gold_entities,
        "wrong_entities": wrong_entities,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "prompt_truncated": bool(row.get("prompt_truncated", False)),
        "snippet_count": len(snippets),
    }
    return passed, details


def main() -> int:
    args = parse_args()
    rows = [dict(row) for row in load_json_records(Path(args.input_jsonl))]
    question_input_path = resolve_question_input_path(rows, args.question_input)
    source_rows = load_question_records(question_input_path)
    _, question_index = build_question_index(
        source_rows,
        question_input_path=question_input_path,
    )

    failures: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()
    snippet_counts: list[int] = []
    prompt_truncated_count = 0

    for row in rows:
        question_id = clean_text(row.get("question_id"))
        if question_id not in question_index:
            failures.append(
                {
                    "question_id": question_id,
                    "question_text": clean_text(row.get("question_text")),
                    "failed_checks": ["question_id_missing_from_source"],
                }
            )
            failure_counter["question_id_missing_from_source"] += 1
            continue

        passed, details = validate_row(
            row,
            source_question=question_index[question_id],
        )
        snippet_counts.append(int(details["snippet_count"]))
        if details["prompt_truncated"]:
            prompt_truncated_count += 1
        if not passed:
            failures.append(details)
            for check_name in details["failed_checks"]:
                failure_counter[check_name] += 1

    summary = {
        "input_jsonl": str(Path(args.input_jsonl)),
        "question_input": str(question_input_path),
        "question_count": len(rows),
        "passed_question_count": len(rows) - len(failures),
        "failed_question_count": len(failures),
        "all_checks_passed": len(failures) == 0,
        "prompt_truncated_question_count": prompt_truncated_count,
        "snippet_count_summary": summarize_numeric(snippet_counts),
        "failure_counts": dict(sorted(failure_counter.items())),
        "validated_checks": [
            "prompt_contains_question",
            "prompt_contains_all_snippets",
            "prompt_excludes_serialized_gold_labels",
            "prompt_has_answer_stub_only",
            "canonical_gold_present",
            "canonical_gold_in_accepted_list",
            "complete_accepted_gold_list_preserved",
            "exactly_four_wrong_entities",
            "wrong_entities_distinct_normalized",
            "wrong_entities_do_not_match_any_accepted_gold_alias",
            "canonical_output_parses",
            "wrong_outputs_parse",
            "accepted_gold_outputs_parse",
            "combined_outputs_parse",
        ],
    }

    write_json(Path(args.summary_json), summary)
    if args.failures_jsonl:
        write_jsonl(Path(args.failures_jsonl), failures)
    if args.freeze_output_jsonl and not failures:
        write_jsonl(Path(args.freeze_output_jsonl), rows)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
