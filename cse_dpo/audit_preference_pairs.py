from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utility.data import clean_text

from .common import flatten_counter, load_json_records, summarize_numeric, truncate_multiline, write_json
from .match_gold_groups import load_question_examples
from .normalize_set_answers import normalize_answer_surface
from .schemas import PAIR_TYPE_NEGATIVE_ADDITION, PAIR_TYPE_VALID_OMISSION


def normalized_set_key(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(normalized for normalized in (normalize_answer_surface(item) for item in items) if normalized))


def load_pair_rows(path: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in load_json_records(Path(path))]
    for row in rows:
        row.setdefault("chosen_items", [])
        row.setdefault("rejected_items", [])
        row.setdefault("pair_type", "")
        row.setdefault("dataset", "")
        row.setdefault("question_id", "")
        row.setdefault("positive_source", None)
        row.setdefault("semantic_set_edit_distance", None)
        row.setdefault("delta_f1", None)
    return rows


def contradictory_pair_count(rows: Sequence[Mapping[str, Any]]) -> int:
    seen = set()
    contradictions = 0
    for row in rows:
        key = (
            clean_text(row.get("question_id")),
            normalized_set_key(row.get("chosen_items", [])),
            normalized_set_key(row.get("rejected_items", [])),
            clean_text(row.get("pair_type")),
        )
        reverse_key = (key[0], key[2], key[1], key[3])
        if reverse_key in seen:
            contradictions += 1
        seen.add(key)
    return contradictions


def duplicate_pair_count(rows: Sequence[Mapping[str, Any]]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = (
            clean_text(row.get("question_id")),
            normalized_set_key(row.get("chosen_items", [])),
            normalized_set_key(row.get("rejected_items", [])),
            clean_text(row.get("pair_type")),
        )
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    question_gold_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    pair_type_counts = Counter(clean_text(row.get("pair_type")) for row in rows)
    positive_source_counts = Counter(clean_text(row.get("positive_source")) or "null" for row in rows)
    dataset_counts = Counter(clean_text(row.get("dataset")) for row in rows)
    question_counts = Counter(clean_text(row.get("question_id")) for row in rows)
    edit_distance_counts = Counter(str(row.get("semantic_set_edit_distance")) for row in rows)

    by_gold_cardinality = Counter()
    if question_gold_counts:
        for question_id, count in question_counts.items():
            gold_count = question_gold_counts.get(question_id)
            if gold_count is not None:
                by_gold_cardinality[str(gold_count)] += count

    delta_f1_values = [float(row["delta_f1"]) for row in rows if isinstance(row.get("delta_f1"), (float, int))]
    distance_values = [
        int(row["semantic_set_edit_distance"])
        for row in rows
        if isinstance(row.get("semantic_set_edit_distance"), int)
    ]
    question_distribution = list(question_counts.values())

    clean_edit_distance = all(distance == 1 for distance in distance_values) if distance_values else False
    positive_delta = all(value > 0 for value in delta_f1_values) if delta_f1_values else False
    contradictions = contradictory_pair_count(rows)
    duplicates = duplicate_pair_count(rows)

    return {
        "pair_count": len(rows),
        "unique_questions": len(question_counts),
        "pair_type_counts": flatten_counter(pair_type_counts),
        "pairs_per_question": summarize_numeric(question_distribution),
        "semantic_edit_distance_distribution": flatten_counter(edit_distance_counts),
        "delta_f1_distribution": summarize_numeric(delta_f1_values),
        "set_edit_distance_distribution_numeric": summarize_numeric(distance_values),
        "duplicate_pair_count": duplicates,
        "contradictory_pair_count": contradictions,
        "positive_source_distribution": flatten_counter(positive_source_counts),
        "pair_count_by_gold_cardinality": flatten_counter(by_gold_cardinality),
        "pair_count_by_dataset": flatten_counter(dataset_counts),
        "check_interpretation": {
            "summary": (
                "These are structural consistency checks. Matcher-based checks certify consistency under the "
                "current matcher and do not independently prove semantic correctness."
            ),
            "delta_f1_scope": "Computed by rescoring chosen and rejected sets with the current matcher.",
            "semantic_edit_distance_scope": "Computed from the current matcher-derived semantic set key.",
        },
        "required_checks": {
            "all_primary_pairs_matcher_semantic_edit_distance_1": clean_edit_distance,
            "all_primary_pairs_positive_delta_f1_under_current_matcher": positive_delta,
            "no_contradictory_pairs_under_normalized_surface_key": contradictions == 0,
        },
    }


def render_pair_block(
    row: Mapping[str, Any],
    evidence_preview: Sequence[str],
) -> str:
    chosen = "\n".join(f"- {item}" for item in row.get("chosen_items", [])) or "- <empty>"
    rejected = "\n".join(f"- {item}" for item in row.get("rejected_items", [])) or "- <empty>"
    evidence_lines = "\n".join(f"- {item}" for item in evidence_preview) or "- <not available>"
    positive_source = clean_text(row.get("positive_source")) or "null"

    return (
        f"### {clean_text(row.get('pair_id'))}\n\n"
        f"Question: {clean_text(row.get('question_text'))}\n\n"
        f"Question source: {clean_text(row.get('question_source_path'))}\n\n"
        f"Evidence preview:\n{evidence_lines}\n\n"
        f"Pair type: {clean_text(row.get('pair_type'))}\n\n"
        f"Edited candidate: {clean_text(row.get('edited_candidate'))}\n\n"
        f"Chosen set:\n{chosen}\n\n"
        f"Rejected set:\n{rejected}\n\n"
        f"Delta precision: {float(row.get('delta_precision', 0.0)):.4f}\n\n"
        f"Delta recall: {float(row.get('delta_recall', 0.0)):.4f}\n\n"
        f"Delta F1: {float(row.get('delta_f1', 0.0)):.4f}\n\n"
        f"Candidate label: {clean_text(row.get('candidate_label'))}\n\n"
        f"Candidate label source: {clean_text(row.get('candidate_label_source'))}\n\n"
        f"Positive candidate source: {positive_source}\n"
    )


def build_manual_audit_markdown(
    rows: Sequence[Mapping[str, Any]],
    question_input: Sequence[str] | None,
    dataset_name: str,
    sample_per_type: int,
    seed: int,
) -> str:
    questions_by_id = (
        load_question_examples(question_input, dataset_name=dataset_name)
        if question_input
        else {}
    )
    rng = random.Random(seed)

    parts = [
        "# Manual Pair Audit",
        "",
        f"Requested sample size per pair type: {sample_per_type}",
        "",
        (
            "Note: Delta precision/recall/F1 values are recomputed under the current matcher. "
            "They are structural consistency signals, not independent semantic correctness judgments."
        ),
        "",
    ]

    for pair_type in (PAIR_TYPE_NEGATIVE_ADDITION, PAIR_TYPE_VALID_OMISSION):
        matching_rows = [dict(row) for row in rows if clean_text(row.get("pair_type")) == pair_type]
        rng.shuffle(matching_rows)
        selected_rows = matching_rows[:sample_per_type]
        parts.append(f"## {pair_type}")
        parts.append("")
        parts.append(f"Selected pairs: {len(selected_rows)} / {len(matching_rows)}")
        parts.append("")

        for row in selected_rows:
            question = questions_by_id.get(clean_text(row.get("question_id")))
            evidence_preview = truncate_multiline(question.evidence if question else [], max_chars_per_item=350)
            parts.append(render_pair_block(row, evidence_preview=evidence_preview))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and sample preference-pair audits for CSE-DPO experiments."
    )
    parser.add_argument(
        "--pairs-jsonl",
        required=True,
        help="Preference-pair JSONL or JSON file.",
    )
    parser.add_argument(
        "--output-summary",
        required=True,
        help="JSON path where the aggregate summary will be written.",
    )
    parser.add_argument(
        "--output-manual-audit",
        default=None,
        help="Optional markdown file showing sampled pairs for manual inspection.",
    )
    parser.add_argument(
        "--question-input",
        nargs="+",
        default=None,
        help="Optional raw/prepared question files used to render evidence previews.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label used when loading question metadata for the audit.",
    )
    parser.add_argument(
        "--sample-per-type",
        type=int,
        default=50,
        help="How many pairs of each type to include in the markdown audit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for audit sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_pair_rows(args.pairs_jsonl)
    question_gold_counts: dict[str, int] = {}
    if args.question_input:
        questions_by_id = load_question_examples(args.question_input, dataset_name=args.dataset_name)
        question_gold_counts = {
            question_id: len(question.gold_groups)
            for question_id, question in questions_by_id.items()
        }

    summary = build_summary(rows, question_gold_counts=question_gold_counts)
    write_json(Path(args.output_summary), summary)

    if args.output_manual_audit:
        markdown = build_manual_audit_markdown(
            rows,
            question_input=args.question_input,
            dataset_name=args.dataset_name,
            sample_per_type=args.sample_per_type,
            seed=args.seed,
        )
        Path(args.output_manual_audit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_manual_audit).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
