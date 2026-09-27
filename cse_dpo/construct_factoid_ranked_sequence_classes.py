from __future__ import annotations

import argparse
import itertools
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.utility.bioasq_format import normalize_for_bioasq_exact_match, normalize_for_match
from src.utility.data import clean_text
from src.utility.factoid_output_parsing import parse_factoid_candidates

from .common import load_json_records, summarize_numeric, write_json, write_jsonl


REJECTED_RANK_CYCLE = (2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign rejected ranks and construct preferred/rejected factoid sequence classes."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Frozen question-level factoid label/wrong-entity pool JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where split artifacts will be written.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        help="Split label stored in the emitted artifacts.",
    )
    parser.add_argument(
        "--rank-seed",
        type=int,
        default=3407,
        help="Seed used to shuffle question IDs before cyclic rejected-rank assignment.",
    )
    parser.add_argument(
        "--validation-input-jsonl",
        default=None,
        help="Optional validation JSONL to process with a separate seed.",
    )
    parser.add_argument(
        "--validation-split-name",
        default="validation",
        help="Split label used for the optional validation artifacts.",
    )
    parser.add_argument(
        "--validation-rank-seed",
        type=int,
        default=3408,
        help="Seed used for the optional validation split.",
    )
    return parser.parse_args()


def format_factoid(entity: str) -> str:
    return f"[BE]{clean_text(entity)}[EE]"


def serialize_sequence(entities: list[str]) -> str:
    return " ".join(format_factoid(entity) for entity in entities)


def token_length(text: str) -> int:
    return len(clean_text(text).split())


def assign_rejected_ranks(rows: list[dict[str, Any]], *, seed: int) -> tuple[dict[str, int], dict[str, int]]:
    question_ids = sorted(
        {
            clean_text(row.get("question_id"))
            for row in rows
            if clean_text(row.get("question_id"))
        }
    )
    shuffled_question_ids = list(question_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled_question_ids)

    rejected_rank_by_question: dict[str, int] = {}
    assignment_index_by_question: dict[str, int] = {}
    for index, question_id in enumerate(shuffled_question_ids):
        rejected_rank_by_question[question_id] = REJECTED_RANK_CYCLE[index % len(REJECTED_RANK_CYCLE)]
        assignment_index_by_question[question_id] = index
    return rejected_rank_by_question, assignment_index_by_question


def gold_rank(entities: list[str], gold: str) -> int | None:
    gold_key = normalize_for_bioasq_exact_match(gold)
    for index, entity in enumerate(entities, start=1):
        if normalize_for_bioasq_exact_match(entity) == gold_key:
            return index
    return None


def mrr_for_entities(entities: list[str], accepted_gold: list[str]) -> float:
    accepted = {
        normalize_for_bioasq_exact_match(entity)
        for entity in accepted_gold
        if normalize_for_bioasq_exact_match(entity)
    }
    for index, entity in enumerate(entities, start=1):
        if normalize_for_bioasq_exact_match(entity) in accepted:
            return 1.0 / index
    return 0.0


def build_question_classes(
    row: dict[str, Any],
    *,
    split_name: str,
    rejected_rank: int,
    rank_assignment_index: int,
    rank_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    question_id = clean_text(row.get("question_id"))
    prompt = str(row.get("prompt", ""))
    gold = clean_text(row.get("canonical_gold_entity") or row.get("chosen_entity"))
    accepted_gold = [
        clean_text(value)
        for value in row.get("accepted_gold_entities", row.get("gold_aliases", []))
        if clean_text(value)
    ]
    wrongs = [clean_text(value) for value in row.get("wrong_entities", []) if clean_text(value)]

    assigned_row = {
        "question_id": question_id,
        "split": split_name,
        "prompt": prompt,
        "gold": gold,
        "accepted_gold": accepted_gold,
        "wrongs": wrongs,
        "rejected_rank": rejected_rank,
        "rank_assignment_seed": rank_seed,
        "rank_assignment_index": rank_assignment_index,
        "question_text": clean_text(row.get("question_text")),
        "snippets": [clean_text(value) for value in row.get("snippets", []) if clean_text(value)],
    }

    preferred_sequences: list[dict[str, Any]] = []
    rejected_sequences: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for permutation_index, permutation in enumerate(itertools.permutations(wrongs), start=1):
        preferred_entities = [gold, *permutation]
        rejected_entities = list(permutation)
        rejected_entities.insert(rejected_rank - 1, gold)
        preferred_serialized = serialize_sequence(preferred_entities)
        rejected_serialized = serialize_sequence(rejected_entities)

        preferred_sequences.append(
            {
                "permutation_index": permutation_index,
                "wrong_order": list(permutation),
                "entities": preferred_entities,
                "serialized": preferred_serialized,
                "gold_rank": 1,
                "mrr": 1.0,
            }
        )
        rejected_sequences.append(
            {
                "permutation_index": permutation_index,
                "wrong_order": list(permutation),
                "entities": rejected_entities,
                "serialized": rejected_serialized,
                "gold_rank": rejected_rank,
                "mrr": 1.0 / rejected_rank,
            }
        )
        pair_rows.append(
            {
                "question_id": question_id,
                "split": split_name,
                "permutation_index": permutation_index,
                "rejected_rank": rejected_rank,
                "prompt": prompt,
                "gold": gold,
                "accepted_gold": accepted_gold,
                "wrongs": wrongs,
                "preferred_entities": preferred_entities,
                "preferred_serialized": preferred_serialized,
                "rejected_entities": rejected_entities,
                "rejected_serialized": rejected_serialized,
                "preferred_mrr": 1.0,
                "rejected_mrr": 1.0 / rejected_rank,
            }
        )

    question_class_row = dict(assigned_row)
    question_class_row.update(
        {
            "preferred_class_size": len(preferred_sequences),
            "rejected_class_size": len(rejected_sequences),
            "preferred_sequences": preferred_sequences,
            "rejected_sequences": rejected_sequences,
        }
    )
    return assigned_row, pair_rows, question_class_row


def validate_question_classes(question_row: dict[str, Any]) -> tuple[bool, dict[str, Any], int, int]:
    gold = clean_text(question_row.get("gold"))
    accepted_gold = [
        clean_text(value) for value in question_row.get("accepted_gold", []) if clean_text(value)
    ]
    wrongs = [clean_text(value) for value in question_row.get("wrongs", []) if clean_text(value)]
    rejected_rank = int(question_row.get("rejected_rank"))
    preferred_sequences = list(question_row.get("preferred_sequences", []))
    rejected_sequences = list(question_row.get("rejected_sequences", []))

    expected_norm_set = {
        normalize_for_match(value)
        for value in [gold, *wrongs]
        if normalize_for_match(value)
    }

    preferred_serialized_unique = {
        str(item.get("serialized", "")) for item in preferred_sequences if clean_text(item.get("serialized"))
    }
    rejected_serialized_unique = {
        str(item.get("serialized", "")) for item in rejected_sequences if clean_text(item.get("serialized"))
    }

    parse_total = 0
    parse_valid = 0
    parse_failures: list[str] = []
    same_candidate_set_ok = True
    gold_once_ok = True
    preferred_gold_rank_ok = True
    rejected_gold_rank_ok = True
    relative_wrong_order_ok = True
    preferred_mrr_ok = True
    rejected_mrr_ok = True

    for preferred, rejected in zip(preferred_sequences, rejected_sequences):
        preferred_entities = list(preferred.get("entities", []))
        rejected_entities = list(rejected.get("entities", []))
        preferred_serialized = str(preferred.get("serialized", ""))
        rejected_serialized = str(rejected.get("serialized", ""))

        for label, serialized, expected_entities in (
            ("preferred", preferred_serialized, preferred_entities),
            ("rejected", rejected_serialized, rejected_entities),
        ):
            parse_total += 1
            parsed_entities = parse_factoid_candidates(serialized, parser_mode="current")
            if parsed_entities == expected_entities and len(parsed_entities) == 5:
                parse_valid += 1
            else:
                parse_failures.append(label)

        preferred_norms = {
            normalize_for_match(value)
            for value in preferred_entities
            if normalize_for_match(value)
        }
        rejected_norms = {
            normalize_for_match(value)
            for value in rejected_entities
            if normalize_for_match(value)
        }
        if preferred_norms != expected_norm_set or rejected_norms != expected_norm_set:
            same_candidate_set_ok = False

        preferred_gold_occurrences = sum(
            1
            for value in preferred_entities
            if normalize_for_bioasq_exact_match(value) == normalize_for_bioasq_exact_match(gold)
        )
        rejected_gold_occurrences = sum(
            1
            for value in rejected_entities
            if normalize_for_bioasq_exact_match(value) == normalize_for_bioasq_exact_match(gold)
        )
        if preferred_gold_occurrences != 1 or rejected_gold_occurrences != 1:
            gold_once_ok = False

        if gold_rank(preferred_entities, gold) != 1:
            preferred_gold_rank_ok = False
        if gold_rank(rejected_entities, gold) != rejected_rank:
            rejected_gold_rank_ok = False

        preferred_without_gold = [
            value
            for value in preferred_entities
            if normalize_for_bioasq_exact_match(value) != normalize_for_bioasq_exact_match(gold)
        ]
        rejected_without_gold = [
            value
            for value in rejected_entities
            if normalize_for_bioasq_exact_match(value) != normalize_for_bioasq_exact_match(gold)
        ]
        if preferred_without_gold != rejected_without_gold:
            relative_wrong_order_ok = False

        if mrr_for_entities(preferred_entities, accepted_gold) != 1.0:
            preferred_mrr_ok = False
        if mrr_for_entities(rejected_entities, accepted_gold) != 1.0 / rejected_rank:
            rejected_mrr_ok = False

    checks = {
        "preferred_class_has_24_unique_sequences": len(preferred_serialized_unique) == 24,
        "rejected_class_has_24_unique_sequences": len(rejected_serialized_unique) == 24,
        "same_five_candidates_in_every_sequence": same_candidate_set_ok,
        "gold_occurs_exactly_once": gold_once_ok,
        "preferred_gold_rank_always_1": preferred_gold_rank_ok,
        "rejected_gold_rank_matches_assigned_rank": rejected_gold_rank_ok,
        "matched_pairs_preserve_relative_wrong_order": relative_wrong_order_ok,
        "parser_extracts_exactly_five_entities_from_all_sequences": parse_valid == parse_total,
        "preferred_mrr_always_1": preferred_mrr_ok,
        "rejected_mrr_matches_assigned_rank": rejected_mrr_ok,
    }
    details = {
        "question_id": clean_text(question_row.get("question_id")),
        "rejected_rank": rejected_rank,
        "failed_checks": [name for name, value in checks.items() if not value],
        "checks": checks,
        "parse_valid_sequences": parse_valid,
        "parse_total_sequences": parse_total,
        "parse_failures": parse_failures,
    }
    return all(checks.values()), details, parse_valid, parse_total


def process_split(
    *,
    input_jsonl: Path,
    output_dir: Path,
    split_name: str,
    rank_seed: int,
) -> dict[str, Any]:
    rows = [dict(row) for row in load_json_records(input_jsonl)]
    rejected_rank_by_question, assignment_index_by_question = assign_rejected_ranks(
        rows,
        seed=rank_seed,
    )

    assigned_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    question_class_rows: list[dict[str, Any]] = []

    for row in rows:
        question_id = clean_text(row.get("question_id"))
        assigned_row, question_pairs, question_class_row = build_question_classes(
            row,
            split_name=split_name,
            rejected_rank=rejected_rank_by_question[question_id],
            rank_assignment_index=assignment_index_by_question[question_id],
            rank_seed=rank_seed,
        )
        assigned_rows.append(assigned_row)
        pair_rows.extend(question_pairs)
        question_class_rows.append(question_class_row)

    failures: list[dict[str, Any]] = []
    parse_valid_sequences = 0
    parse_total_sequences = 0
    rejected_rank_counts: Counter[int] = Counter()
    candidate_token_lengths: list[int] = []

    for question_class_row in question_class_rows:
        rejected_rank_counts[int(question_class_row["rejected_rank"])] += 1
        candidate_token_lengths.extend(
            token_length(entity)
            for entity in [question_class_row["gold"], *question_class_row["wrongs"]]
        )
        passed, details, row_parse_valid, row_parse_total = validate_question_classes(
            question_class_row
        )
        parse_valid_sequences += row_parse_valid
        parse_total_sequences += row_parse_total
        if not passed:
            failures.append(details)

    summary = {
        "split_name": split_name,
        "input_jsonl": str(input_jsonl),
        "rank_seed": rank_seed,
        "question_count": len(question_class_rows),
        "assigned_question_count": len(assigned_rows),
        "pair_count": len(pair_rows),
        "rejected_rank_counts": {
            str(rank): int(rejected_rank_counts.get(rank, 0))
            for rank in REJECTED_RANK_CYCLE
        },
        "parse_validity_rate": (
            parse_valid_sequences / parse_total_sequences if parse_total_sequences else 0.0
        ),
        "parse_valid_sequences": parse_valid_sequences,
        "parse_total_sequences": parse_total_sequences,
        "failed_record_count": len(failures),
        "average_candidate_token_length": (
            sum(candidate_token_lengths) / len(candidate_token_lengths)
            if candidate_token_lengths
            else 0.0
        ),
        "candidate_token_length_summary": summarize_numeric(candidate_token_lengths),
        "all_checks_passed": len(failures) == 0,
        "validated_checks": [
            "preferred_class_has_24_unique_sequences",
            "rejected_class_has_24_unique_sequences",
            "same_five_candidates_in_every_sequence",
            "gold_occurs_exactly_once",
            "preferred_gold_rank_always_1",
            "rejected_gold_rank_matches_assigned_rank",
            "matched_pairs_preserve_relative_wrong_order",
            "parser_extracts_exactly_five_entities_from_all_sequences",
            "preferred_mrr_always_1",
            "rejected_mrr_matches_assigned_rank",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{split_name}_rank_assigned_questions.jsonl", assigned_rows)
    write_jsonl(output_dir / f"{split_name}_pair_records.jsonl", pair_rows)
    write_jsonl(output_dir / f"{split_name}_question_classes.jsonl", question_class_rows)
    write_jsonl(output_dir / f"{split_name}_failures.jsonl", failures)
    write_json(output_dir / f"{split_name}_summary.json", summary)

    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    summaries = [
        process_split(
            input_jsonl=Path(args.input_jsonl),
            output_dir=output_dir,
            split_name=args.split_name,
            rank_seed=args.rank_seed,
        )
    ]

    if args.validation_input_jsonl:
        summaries.append(
            process_split(
                input_jsonl=Path(args.validation_input_jsonl),
                output_dir=output_dir,
                split_name=args.validation_split_name,
                rank_seed=args.validation_rank_seed,
            )
        )

    write_json(output_dir / "manifest.json", {"splits": summaries})
    return 0 if all(summary["all_checks_passed"] for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
