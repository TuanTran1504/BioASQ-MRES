from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.model_registry import utc_now_iso
from src.utility.bioasq_format import normalize_for_match
from src.utility.data import clean_text, list_record_resources
from src.utility.factoid_output_parsing import parse_factoid_candidates

from .common import summarize_numeric, write_json, write_jsonl


SENTENCE_CUES = (
    " is ",
    " are ",
    " was ",
    " were ",
    " can ",
    " causes ",
    " cause ",
    " include ",
    " includes ",
    " leads to ",
    " associated with ",
    " treated with ",
    " approved for ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a factoid label-vs-wrong-entity export from a generated candidate bank."
        )
    )
    parser.add_argument(
        "--question-input",
        required=True,
        help="Prepared factoid JSON file used to define labels and resources.",
    )
    parser.add_argument(
        "--candidate-bank-jsonl",
        required=True,
        help="Candidate-bank JSONL produced by cse_dpo.generate_candidate_bank.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Question-level output JSONL path.",
    )
    parser.add_argument(
        "--pair-output-jsonl",
        default=None,
        help="Optional pair-level JSONL path with one chosen/rejected row per wrong entity.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path.",
    )
    parser.add_argument(
        "--wrongs-per-question",
        type=int,
        default=4,
        help="How many wrong entities to export per question.",
    )
    parser.add_argument(
        "--global-backfill-pool-size",
        type=int,
        default=256,
        help=(
            "When a question has too few within-question wrong entities, sample the "
            "remaining slots from the top-N global wrong-entity pool."
        ),
    )
    parser.add_argument(
        "--require-within-question-wrongs",
        action="store_true",
        help=(
            "Export only questions that provide every requested wrong entity from their "
            "own generated candidates; do not use global backfill."
        ),
    )
    parser.add_argument(
        "--allow-fewer-within-question-wrongs",
        action="store_true",
        help=(
            "Export questions with one to the requested number of locally generated "
            "wrong entities; skip questions with none and never use global backfill."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Deterministic seed used for diversified global backfill selection.",
    )
    parser.add_argument(
        "--max-entity-chars",
        type=int,
        default=160,
        help="Discard sampled entities longer than this many characters.",
    )
    parser.add_argument(
        "--max-entity-words",
        type=int,
        default=14,
        help="Discard sampled entities longer than this many whitespace-delimited words.",
    )
    return parser.parse_args()


def best_surface(surface_counts: Counter[str]) -> str:
    ranked = sorted(
        surface_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0].lower()),
    )
    return ranked[0][0]


def format_factoid(entity: str) -> str:
    return f"[BE]{clean_text(entity)}[EE]"


def prompt_present_snippets(prompt: str, snippets: list[str]) -> list[str]:
    normalized_prompt = clean_text(prompt)
    present = [
        clean_text(snippet)
        for snippet in snippets
        if clean_text(snippet) and clean_text(snippet) in normalized_prompt
    ]
    return present


def is_entity_like(entity: str, *, max_chars: int, max_words: int) -> bool:
    cleaned = clean_text(entity)
    if not cleaned:
        return False
    lowered = f" {cleaned.lower()} "
    word_count = len(cleaned.split())
    if max_chars > 0 and len(cleaned) > max_chars:
        return False
    if max_words > 0 and word_count > max_words:
        return False
    if cleaned.endswith((".", ";", ":")) and word_count > 4:
        return False
    if any(cue in lowered for cue in SENTENCE_CUES) and word_count > 4:
        return False
    return True


def load_question_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return [record for record in payload if isinstance(record, dict)]


def build_question_index(
    records: list[dict[str, Any]],
    *,
    question_input_path: Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_question_ids: list[str] = []
    question_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        question_id = clean_text(record.get("id"))
        if not question_id:
            continue

        gold_aliases = parse_factoid_candidates(
            clean_text(record.get("output")),
            parser_mode="current",
        )
        if not gold_aliases:
            continue

        chosen_entity = gold_aliases[0]
        gold_alias_norms = {
            normalize_for_match(alias)
            for alias in gold_aliases
            if normalize_for_match(alias)
        }
        question_by_id[question_id] = {
            "question_id": question_id,
            "question_text": clean_text(record.get("input_1")),
            "question_type": clean_text(record.get("type")) or "factoid",
            "instruction": str(record.get("instruction", "")),
            "gold_aliases": gold_aliases,
            "gold_alias_norms": gold_alias_norms,
            "chosen_entity": chosen_entity,
            "source_path": str(question_input_path),
            "fallback_snippets": list_record_resources(record),
        }
        ordered_question_ids.append(question_id)

    return ordered_question_ids, question_by_id


def ensure_candidate_stats(
    store: dict[str, dict[str, Any]],
    normalized: str,
    *,
    first_seen: int,
) -> dict[str, Any]:
    stats = store.get(normalized)
    if stats is not None:
        return stats

    stats = {
        "count": 0,
        "first_seen": first_seen,
        "surfaces": Counter(),
        "source_question_ids": set(),
    }
    store[normalized] = stats
    return stats


def sort_stats(stats_by_norm: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for normalized, stats in stats_by_norm.items():
        entity = best_surface(stats["surfaces"])
        ranked.append(
            {
                "entity": entity,
                "normalized": normalized,
                "count": int(stats["count"]),
                "first_seen": int(stats["first_seen"]),
                "source_question_count": len(stats["source_question_ids"]),
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["count"],
            item["first_seen"],
            len(item["entity"]),
            item["entity"].lower(),
        )
    )
    return ranked


def build_exports(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if args.require_within_question_wrongs and args.allow_fewer_within_question_wrongs:
        raise ValueError(
            "--require-within-question-wrongs and --allow-fewer-within-question-wrongs "
            "cannot be used together"
        )
    question_input_path = Path(args.question_input)
    candidate_bank_path = Path(args.candidate_bank_jsonl)

    question_records = load_question_records(question_input_path)
    ordered_question_ids, question_by_id = build_question_index(
        question_records,
        question_input_path=question_input_path,
    )

    local_wrong_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    global_wrong_stats: dict[str, dict[str, Any]] = {}
    prompt_by_question: dict[str, str] = {}
    prompt_instruction_by_question: dict[str, str] = {}
    evidence_by_question: dict[str, list[str]] = {}
    dataset_by_question: dict[str, str] = {}
    prompt_truncated_by_question: dict[str, bool] = {}
    sample_count_by_question: Counter[str] = Counter()
    global_first_seen = 0

    with candidate_bank_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {candidate_bank_path}:{line_number}")

            question_id = clean_text(row.get("question_id"))
            if question_id not in question_by_id:
                continue

            question_info = question_by_id[question_id]
            excluded_norms = question_info["gold_alias_norms"]
            sample_count_by_question[question_id] += 1

            if question_id not in prompt_by_question:
                prompt_by_question[question_id] = str(row.get("prompt", ""))
                prompt_instruction_by_question[question_id] = str(
                    row.get("prompt_instruction") or question_info["instruction"]
                )
                raw_evidence = [
                    clean_text(value) for value in row.get("evidence", []) if clean_text(value)
                ]
                evidence_by_question[question_id] = prompt_present_snippets(
                    prompt_by_question[question_id],
                    raw_evidence,
                ) or raw_evidence
                dataset_by_question[question_id] = clean_text(row.get("dataset")) or "bioasq-factoid"
                prompt_truncated_by_question[question_id] = bool(
                    row.get("generation_telemetry", {}).get("prompt_truncated")
                    or row.get("prompt_truncation", {}).get("prompt_truncated")
                )

            parsed_items = row.get("parsed_items")
            if not isinstance(parsed_items, list):
                parsed_items = parse_factoid_candidates(
                    clean_text(row.get("raw_output")),
                    parser_mode="agnostic",
                )

            for item in parsed_items:
                entity = clean_text(item)
                if not is_entity_like(
                    entity,
                    max_chars=args.max_entity_chars,
                    max_words=args.max_entity_words,
                ):
                    continue
                normalized = normalize_for_match(entity)
                if not normalized or normalized in excluded_norms:
                    continue

                per_question_store = local_wrong_stats[question_id]
                local_stats = ensure_candidate_stats(
                    per_question_store,
                    normalized,
                    first_seen=len(per_question_store),
                )
                local_stats["count"] += 1
                local_stats["surfaces"][entity] += 1
                local_stats["source_question_ids"].add(question_id)

                global_stats = ensure_candidate_stats(
                    global_wrong_stats,
                    normalized,
                    first_seen=global_first_seen,
                )
                global_stats["count"] += 1
                global_stats["surfaces"][entity] += 1
                global_stats["source_question_ids"].add(question_id)
                global_first_seen += 1

    global_ranked = sort_stats(global_wrong_stats)
    preferred_global_pool = global_ranked[: max(0, int(args.global_backfill_pool_size))]
    remaining_global_pool = global_ranked[len(preferred_global_pool) :]

    question_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    within_question_wrong_counts: list[int] = []
    selected_within_question_counts: list[int] = []
    global_backfill_counts: list[int] = []
    questions_requiring_backfill = 0
    questions_with_zero_within_question_wrongs = 0
    questions_with_full_within_question_wrongs = 0
    questions_skipped_for_insufficient_within_question_wrongs = 0
    questions_skipped_for_no_within_question_wrongs = 0

    for question_id in ordered_question_ids:
        question_info = question_by_id[question_id]
        local_ranked = sort_stats(local_wrong_stats.get(question_id, {}))
        within_question_wrong_count = len(local_ranked)
        within_question_wrong_counts.append(within_question_wrong_count)
        if within_question_wrong_count == 0:
            questions_with_zero_within_question_wrongs += 1
        if within_question_wrong_count >= args.wrongs_per_question:
            questions_with_full_within_question_wrongs += 1

        if (
            args.require_within_question_wrongs
            and within_question_wrong_count < args.wrongs_per_question
        ):
            questions_skipped_for_insufficient_within_question_wrongs += 1
            continue
        if args.allow_fewer_within_question_wrongs and within_question_wrong_count == 0:
            questions_skipped_for_no_within_question_wrongs += 1
            continue

        selected_negatives: list[dict[str, Any]] = []
        selected_norms: set[str] = set()
        for candidate in local_ranked:
            if len(selected_negatives) >= args.wrongs_per_question:
                break
            selected_norms.add(candidate["normalized"])
            selected_negatives.append(
                {
                    "entity": candidate["entity"],
                    "normalized": candidate["normalized"],
                    "selection_source": "within_question",
                    "within_question_count": candidate["count"],
                    "global_count": candidate["count"],
                    "source_question_count": candidate["source_question_count"],
                }
            )

        excluded_norms = set(question_info["gold_alias_norms"]) | selected_norms
        if (
            len(selected_negatives) < args.wrongs_per_question
            and not args.allow_fewer_within_question_wrongs
        ):
            questions_requiring_backfill += 1
            rng = random.Random(f"{args.seed}:{question_id}")
            shuffled_preferred = list(preferred_global_pool)
            rng.shuffle(shuffled_preferred)
            fallback_pool = shuffled_preferred + remaining_global_pool

            for candidate in fallback_pool:
                if len(selected_negatives) >= args.wrongs_per_question:
                    break
                if candidate["normalized"] in excluded_norms:
                    continue
                excluded_norms.add(candidate["normalized"])
                selected_negatives.append(
                    {
                        "entity": candidate["entity"],
                        "normalized": candidate["normalized"],
                        "selection_source": "global_backfill",
                        "within_question_count": 0,
                        "global_count": candidate["count"],
                        "source_question_count": candidate["source_question_count"],
                    }
                )

        if len(selected_negatives) < args.wrongs_per_question and not args.allow_fewer_within_question_wrongs:
            raise ValueError(
                f"Could not collect {args.wrongs_per_question} wrong entities for question {question_id}"
            )

        selected_within_question_count = sum(
            1 for item in selected_negatives if item["selection_source"] == "within_question"
        )
        global_backfill_count = len(selected_negatives) - selected_within_question_count
        selected_within_question_counts.append(selected_within_question_count)
        global_backfill_counts.append(global_backfill_count)

        prompt = prompt_by_question.get(question_id, "")
        prompt_instruction = prompt_instruction_by_question.get(
            question_id,
            question_info["instruction"],
        )
        snippets = evidence_by_question.get(question_id) or question_info["fallback_snippets"]
        dataset = dataset_by_question.get(question_id, "bioasq-factoid")

        question_row = {
            "dataset": dataset,
            "split": "train",
            "question_id": question_id,
            "question_type": question_info["question_type"],
            "question_text": question_info["question_text"],
            "instruction": question_info["instruction"],
            "prompt_instruction": prompt_instruction,
            "prompt": prompt,
            "prompt_truncated": bool(prompt_truncated_by_question.get(question_id, False)),
            "snippets": snippets,
            "question_source_path": question_info["source_path"],
            "candidate_bank_path": str(candidate_bank_path),
            "sample_count": int(sample_count_by_question.get(question_id, 0)),
            "canonical_gold_entity": question_info["chosen_entity"],
            "canonical_gold_output": format_factoid(question_info["chosen_entity"]),
            "accepted_gold_entities": list(question_info["gold_aliases"]),
            "accepted_gold_outputs": [
                format_factoid(alias) for alias in question_info["gold_aliases"]
            ],
            "chosen_entity": question_info["chosen_entity"],
            "chosen_output": format_factoid(question_info["chosen_entity"]),
            "gold_aliases": list(question_info["gold_aliases"]),
            "wrong_entities": [item["entity"] for item in selected_negatives],
            "wrong_outputs": [format_factoid(item["entity"]) for item in selected_negatives],
            "wrong_entity_metadata": [
                {
                    "entity": item["entity"],
                    "selection_source": item["selection_source"],
                    "within_question_count": int(item["within_question_count"]),
                    "global_count": int(item["global_count"]),
                    "source_question_count": int(item["source_question_count"]),
                }
                for item in selected_negatives
            ],
            "within_question_wrong_count": within_question_wrong_count,
            "selected_within_question_wrong_count": selected_within_question_count,
            "global_backfill_count": global_backfill_count,
        }
        question_rows.append(question_row)

        for wrong_index, item in enumerate(selected_negatives, start=1):
            pair_rows.append(
                {
                    "pair_id": f"{question_id}-wrong-{wrong_index}",
                    "dataset": dataset,
                    "split": "train",
                    "question_id": question_id,
                    "question_type": question_info["question_type"],
                    "question_text": question_info["question_text"],
                    "instruction": question_info["instruction"],
                    "prompt_instruction": prompt_instruction,
                    "prompt": prompt,
                    "prompt_truncated": bool(prompt_truncated_by_question.get(question_id, False)),
                    "snippets": snippets,
                    "question_source_path": question_info["source_path"],
                    "candidate_bank_path": str(candidate_bank_path),
                    "sample_count": int(sample_count_by_question.get(question_id, 0)),
                    "canonical_gold_entity": question_info["chosen_entity"],
                    "canonical_gold_output": format_factoid(question_info["chosen_entity"]),
                    "accepted_gold_entities": list(question_info["gold_aliases"]),
                    "chosen_entity": question_info["chosen_entity"],
                    "chosen": format_factoid(question_info["chosen_entity"]),
                    "rejected_entity": item["entity"],
                    "rejected": format_factoid(item["entity"]),
                    "rejected_rank": wrong_index,
                    "rejected_source": item["selection_source"],
                    "rejected_within_question_count": int(item["within_question_count"]),
                    "rejected_global_count": int(item["global_count"]),
                    "rejected_source_question_count": int(item["source_question_count"]),
                }
            )

    summary = {
        "created_at": utc_now_iso(),
        "question_input": str(question_input_path),
        "candidate_bank_jsonl": str(candidate_bank_path),
        "question_count": len(question_rows),
        "pair_count": len(pair_rows),
        "wrongs_per_question": int(args.wrongs_per_question),
        "require_within_question_wrongs": bool(args.require_within_question_wrongs),
        "allow_fewer_within_question_wrongs": bool(args.allow_fewer_within_question_wrongs),
        "global_backfill_pool_size": int(args.global_backfill_pool_size),
        "questions_with_full_within_question_wrongs": questions_with_full_within_question_wrongs,
        "questions_requiring_global_backfill": questions_requiring_backfill,
        "questions_with_zero_within_question_wrongs": questions_with_zero_within_question_wrongs,
        "questions_skipped_for_insufficient_within_question_wrongs": (
            questions_skipped_for_insufficient_within_question_wrongs
        ),
        "questions_skipped_for_no_within_question_wrongs": (
            questions_skipped_for_no_within_question_wrongs
        ),
        "within_question_wrong_count_summary": summarize_numeric(within_question_wrong_counts),
        "selected_within_question_wrong_count_summary": summarize_numeric(
            selected_within_question_counts
        ),
        "global_backfill_count_summary": summarize_numeric(global_backfill_counts),
        "global_wrong_pool_size": len(global_ranked),
        "max_entity_chars": int(args.max_entity_chars),
        "max_entity_words": int(args.max_entity_words),
    }
    return question_rows, pair_rows, summary


def main() -> int:
    args = parse_args()
    question_rows, pair_rows, summary = build_exports(args)
    write_jsonl(Path(args.output_jsonl), question_rows)
    if args.pair_output_jsonl:
        write_jsonl(Path(args.pair_output_jsonl), pair_rows)
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
