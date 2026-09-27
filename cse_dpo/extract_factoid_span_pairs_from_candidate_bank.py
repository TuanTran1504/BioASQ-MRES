"""Extract source-grounded span-mismatch candidates from an existing candidate bank.

The extractor is deliberately high precision: a generated alternative is retained
only when it and one gold alias occur literally in the same evidence resource and
their token spans have an unambiguous short, long, or boundary-shift relation.
Outputs are review artifacts, not immediately trainable DPO data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.build_factoid_extractive_span_pairs import (
    EDGE_STOPWORDS,
    candidate_is_clean,
    compact_normalized,
    find_literal_occurrences,
    gold_is_short_expression,
    output_completion,
    parse_gold_aliases,
    source_tokens,
    tokens_for_span,
    write_json,
    write_jsonl,
)
from src.utility.data import clean_text, list_record_resources


MARKED_SNIPPET_PATTERN = re.compile(r"\[BS\](.*?)\[ES\]", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--prepared-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-gold-tokens", type=int, default=12)
    parser.add_argument("--min-candidate-tokens", type=int, default=1)
    parser.add_argument("--min-candidate-chars", type=int, default=2)
    parser.add_argument("--min-retained-meaningful-gold-tokens", type=int, default=1)
    parser.add_argument(
        "--max-negatives-per-question",
        type=int,
        default=2,
        help="Export at most this many reviewed candidates per DPO prompt. Default: 2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def meaningful_overlap_count(tokens: Sequence[Any], gold_indices: Sequence[int], candidate_indices: Sequence[int]) -> int:
    overlap = set(gold_indices).intersection(candidate_indices)
    return sum(tokens[index].group(0).casefold() not in EDGE_STOPWORDS for index in overlap)


def relation_for_occurrences(
    tokens: Sequence[Any],
    gold_indices: Sequence[int],
    candidate_indices: Sequence[int],
    *,
    min_retained_meaningful_gold_tokens: int,
) -> tuple[str, int] | None:
    """Return a strict token-boundary relation and retained lexical-token count."""
    if not gold_indices or not candidate_indices:
        return None

    gold_first, gold_last = gold_indices[0], gold_indices[-1]
    candidate_first, candidate_last = candidate_indices[0], candidate_indices[-1]
    retained = meaningful_overlap_count(tokens, gold_indices, candidate_indices)

    if candidate_first >= gold_first and candidate_last <= gold_last:
        if candidate_first == gold_first and candidate_last == gold_last:
            return None
        return "too_short", retained
    if gold_first >= candidate_first and gold_last <= candidate_last:
        if candidate_first == gold_first and candidate_last == gold_last:
            return None
        return "too_long", retained

    # These candidates change both boundaries but retain lexical material from gold.
    if candidate_first > gold_first and candidate_last > gold_last and candidate_first <= gold_last:
        if retained >= min_retained_meaningful_gold_tokens:
            return "trim_left_extend_right", retained
    if candidate_first < gold_first and candidate_last < gold_last and candidate_last >= gold_first:
        if retained >= min_retained_meaningful_gold_tokens:
            return "extend_left_trim_right", retained
    return None


def source_context(resource: str, start: int, end: int, *, padding: int = 240) -> str:
    return resource[max(0, start - padding):min(len(resource), end + padding)]


def candidate_contains_other_gold_alias(candidate: str, aliases: Sequence[str], chosen_alias: str) -> bool:
    candidate_key = compact_normalized(candidate)
    chosen_key = compact_normalized(chosen_alias)
    for alias in aliases:
        alias_key = compact_normalized(alias)
        if alias_key and alias_key != chosen_key and alias_key in candidate_key:
            return True
    return False


def marked_snippets(record: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Return only individual marked snippets, never an entire multi-snippet resource."""
    snippets: list[tuple[int, int, str]] = []
    for resource_index, resource in enumerate(list_record_resources(dict(record)), start=1):
        for snippet_index, match in enumerate(MARKED_SNIPPET_PATTERN.finditer(resource), start=1):
            snippet = clean_text(match.group(1))
            if snippet:
                snippets.append((resource_index, snippet_index, snippet))
    return snippets


def first_aligned_row(
    *,
    question_id: str,
    question_text: str,
    candidate: str,
    gold_alias: str,
    snippets: Sequence[tuple[int, int, str]],
    min_retained_meaningful_gold_tokens: int,
) -> dict[str, Any] | None:
    for resource_index, snippet_index, resource in snippets:
        gold_occurrences = find_literal_occurrences(resource, gold_alias)
        candidate_occurrences = find_literal_occurrences(resource, candidate)
        if not gold_occurrences or not candidate_occurrences:
            continue

        tokens = source_tokens(resource)
        for gold_start, gold_end in gold_occurrences:
            gold_indices = tokens_for_span(tokens, gold_start, gold_end)
            for candidate_start, candidate_end in candidate_occurrences:
                candidate_indices = tokens_for_span(tokens, candidate_start, candidate_end)
                related = relation_for_occurrences(
                    tokens,
                    gold_indices,
                    candidate_indices,
                    min_retained_meaningful_gold_tokens=min_retained_meaningful_gold_tokens,
                )
                if related is None:
                    continue
                relation, retained = related
                return {
                    "question_id": question_id,
                    "question_text": question_text,
                    "chosen_entity": gold_alias,
                    "chosen": output_completion(gold_alias),
                    "rejected_entity": candidate,
                    "rejected": output_completion(candidate),
                    "span_direction": relation,
                    "resource_index": resource_index,
                    "snippet_index": snippet_index,
                    "gold_source_start": gold_start,
                    "gold_source_end": gold_end,
                    "candidate_source_start": candidate_start,
                    "candidate_source_end": candidate_end,
                    "retained_meaningful_gold_tokens": retained,
                    "source_context": source_context(
                        resource,
                        min(gold_start, candidate_start),
                        max(gold_end, candidate_end),
                    ),
                    "review_decision": "",
                    "review_notes": "",
                }
    return None


def write_review_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "question_id", "question_text", "chosen_entity", "rejected_entity", "span_direction",
        "resource_index", "sample_count", "sample_ids", "retained_meaningful_gold_tokens",
        "snippet_index",
        "gold_source_start", "gold_source_end", "candidate_source_start", "candidate_source_end",
        "source_context", "review_decision", "review_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)


def write_pre_audit_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "question_id", "chosen_entity", "rejected_entity", "span_direction", "resource_index",
        "snippet_index", "retained_meaningful_gold_tokens", "sample_count", "pre_audit_status", "pre_audit_note",
    ]
    audit_rows = []
    for row in rows:
        audit_rows.append({
            "question_id": row["question_id"],
            "chosen_entity": row["chosen_entity"],
            "rejected_entity": row["rejected_entity"],
            "span_direction": row["span_direction"],
            "resource_index": row["resource_index"],
            "snippet_index": row["snippet_index"],
            "retained_meaningful_gold_tokens": row["retained_meaningful_gold_tokens"],
            "sample_count": row["sample_count"],
            "pre_audit_status": "manual_review_required",
            "pre_audit_note": "Literal source alignment is verified, but the candidate may change biomedical meaning.",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(audit_rows)


def build_dpo_slates(rows: Sequence[Mapping[str, Any]], max_negatives_per_question: int) -> list[dict[str, Any]]:
    """Choose one target alias and a bounded, non-duplicative negative set per question."""
    by_question_and_gold: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_question_and_gold[str(row["question_id"])][str(row["chosen_entity"])].append(row)

    slates: list[dict[str, Any]] = []
    for question_id, by_gold in sorted(by_question_and_gold.items()):
        ranked_gold_groups = []
        for gold_alias, group in by_gold.items():
            directions = {str(row["span_direction"]) for row in group}
            repeated_samples = sum(int(row["sample_count"]) for row in group)
            ranked_gold_groups.append(((len(directions), len(group), repeated_samples, gold_alias.casefold()), gold_alias, group))
        _, gold_alias, group = max(ranked_gold_groups, key=lambda item: item[0])
        selected = sorted(
            group,
            key=lambda row: (
                -int(row["sample_count"]),
                str(row["span_direction"]),
                len(str(row["rejected_entity"])),
                str(row["rejected_entity"]).casefold(),
            ),
        )[:max(0, max_negatives_per_question)]
        if not selected:
            continue
        first = selected[0]
        slates.append({
            "question_id": question_id,
            "question_text": first["question_text"],
            "chosen": output_completion(gold_alias),
            "negatives": [row["rejected"] for row in selected],
            "negative_span_directions": [row["span_direction"] for row in selected],
            "candidate_bank_pair_count_for_selected_gold": len(group),
            "candidate_bank_selected_candidates": list(selected),
            "review_decision": "",
            "review_notes": "",
        })
    return slates


def main() -> None:
    args = parse_args()
    candidate_bank_path = Path(args.candidate_bank).expanduser().resolve()
    prepared_path = Path(args.prepared_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if args.min_retained_meaningful_gold_tokens < 1:
        raise ValueError("min_retained_meaningful_gold_tokens must be at least 1")
    if args.max_negatives_per_question < 1:
        raise ValueError("max_negatives_per_question must be at least 1")

    records = json.loads(prepared_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {prepared_path}")
    records_by_id = {clean_text(record.get("id")): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Prepared data must have unique non-empty question IDs")

    audit = Counter()
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    observed_sample_candidates: set[tuple[str, int, str]] = set()
    for bank_row in load_jsonl(candidate_bank_path):
        question_id = clean_text(bank_row.get("question_id"))
        record = records_by_id.get(question_id)
        if record is None:
            audit["bank_rows_missing_prepared_question"] += 1
            continue
        sample_id = int(bank_row.get("sample_id", -1))
        question_text = clean_text(record.get("input_1"))
        aliases = parse_gold_aliases(record.get("output"))
        snippets = marked_snippets(record)
        accepted_gold_keys = {compact_normalized(alias) for alias in aliases if compact_normalized(alias)}
        if not question_text or not aliases or not snippets:
            audit["questions_missing_required_fields"] += 1
            continue

        for raw_candidate in bank_row.get("parsed_items", []):
            candidate = clean_text(raw_candidate)
            candidate_key = compact_normalized(candidate)
            observed_key = (question_id, sample_id, candidate_key)
            if not candidate or not candidate_key or observed_key in observed_sample_candidates:
                continue
            observed_sample_candidates.add(observed_key)
            audit["parsed_candidate_observations"] += 1
            if candidate_key in accepted_gold_keys:
                audit["candidate_matches_any_gold_alias"] += 1
                continue
            if not candidate_is_clean(
                candidate,
                min_tokens=args.min_candidate_tokens,
                min_chars=args.min_candidate_chars,
            ):
                audit["candidate_rejected_unclean"] += 1
                continue

            accepted = False
            for gold_alias in aliases:
                if not gold_is_short_expression(gold_alias, max_gold_tokens=args.max_gold_tokens):
                    audit["gold_alias_rejected_sentence_like"] += 1
                    continue
                if candidate_contains_other_gold_alias(candidate, aliases, gold_alias):
                    audit["candidate_contains_other_gold_alias"] += 1
                    continue
                aligned = first_aligned_row(
                    question_id=question_id,
                    question_text=question_text,
                    candidate=candidate,
                    gold_alias=gold_alias,
                    snippets=snippets,
                    min_retained_meaningful_gold_tokens=args.min_retained_meaningful_gold_tokens,
                )
                if aligned is None:
                    continue
                key = (question_id, compact_normalized(gold_alias), candidate_key, aligned["span_direction"])
                existing = selected.get(key)
                if existing is None:
                    aligned["split"] = args.split
                    aligned["sample_ids"] = [sample_id]
                    aligned["sample_count"] = 1
                    selected[key] = aligned
                elif sample_id not in existing["sample_ids"]:
                    existing["sample_ids"].append(sample_id)
                    existing["sample_count"] += 1
                accepted = True
                break
            if not accepted:
                audit["candidate_not_high_precision_span_relation"] += 1

    rows = sorted(
        selected.values(),
        key=lambda row: (
            row["question_id"], row["span_direction"], row["chosen_entity"].casefold(),
            row["rejected_entity"].casefold(),
        ),
    )
    for row in rows:
        row["sample_ids"] = ",".join(str(sample_id) for sample_id in sorted(row["sample_ids"]))

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    review_slates = []
    for question_id, group in sorted(by_question.items()):
        first = group[0]
        review_slates.append({
            "question_id": question_id,
            "question_text": first["question_text"],
            "split": args.split,
            "candidates": group,
            "review_decision": "",
            "review_notes": "",
        })

    dpo_slates = build_dpo_slates(rows, args.max_negatives_per_question)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "span_mismatch_candidates.jsonl", rows)
    write_jsonl(output_dir / "question_slates_needing_review.jsonl", review_slates)
    write_jsonl(output_dir / "dpo_question_slates_needing_review.jsonl", dpo_slates)
    write_review_csv(output_dir / "manual_review.csv", rows)
    write_pre_audit_csv(output_dir / "pre_audit_v1" / "candidate_bank_span_pre_audit.csv", rows)
    summary = {
        "candidate_bank": str(candidate_bank_path),
        "prepared_json": str(prepared_path),
        "split": args.split,
        "prepared_question_count": len(records),
        "selected_candidate_count": len(rows),
        "selected_question_count": len(by_question),
        "dpo_slate_question_count": len(dpo_slates),
        "dpo_slate_pair_count": sum(len(slate["negatives"]) for slate in dpo_slates),
        "selected_by_span_direction": dict(sorted(Counter(row["span_direction"] for row in rows).items())),
        "limits": {
            "max_gold_tokens": args.max_gold_tokens,
            "min_candidate_tokens": args.min_candidate_tokens,
            "min_candidate_chars": args.min_candidate_chars,
            "min_retained_meaningful_gold_tokens": args.min_retained_meaningful_gold_tokens,
            "max_negatives_per_question": args.max_negatives_per_question,
        },
        "source_requirement": "The gold alias and rejected candidate must occur literally in the same individual [BS]...[ES] snippet.",
        "audit": dict(sorted(audit.items())),
        "review_requirement": (
            "Every candidate is source-aligned but must be manually reviewed before use in DPO training, "
            "because a formal span relation alone does not establish semantic equivalence."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
