"""Mine auditable literal boundary-shift DPO candidates for factoid extraction.

A boundary shift removes source tokens from one gold-span boundary while adding
source tokens at the opposite boundary. Every rejected candidate must retain at
least one meaningful (non-stopword) gold token. These pairs are deliberately
kept separate from strict subspan/superspan data and require manual approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Support both `python -m cse_dpo...` and direct script execution from the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.build_factoid_extractive_span_pairs import (
    EDGE_STOPWORDS,
    candidate_is_clean,
    compact_normalized,
    first_present_aliases,
    gold_is_short_expression,
    mining_resource_text,
    output_completion,
    parse_gold_aliases,
    source_span,
    source_tokens,
    tokens_for_span,
    write_json,
    write_jsonl,
)
from src.utility.data import clean_multiline_text, clean_text, list_record_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-negatives-per-question", type=int, default=2)
    parser.add_argument("--max-trimmed-tokens", type=int, default=2)
    parser.add_argument("--max-extended-tokens", type=int, default=2)
    parser.add_argument("--min-retained-meaningful-gold-tokens", type=int, default=1)
    parser.add_argument("--min-negative-tokens", type=int, default=2)
    parser.add_argument("--min-negative-chars", type=int, default=4)
    parser.add_argument("--max-gold-tokens", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def token_texts(tokens: Sequence[Any]) -> list[str]:
    return [token.group(0).casefold() for token in tokens]


def meaningful_token_count(tokens: Sequence[Any]) -> int:
    """Count retained lexical tokens while allowing biomedical abbreviations and numbers."""
    return sum(bool(token.group(0).strip()) and token.group(0).casefold() not in EDGE_STOPWORDS for token in tokens)


def mine_boundary_shift_candidates(
    resource: str,
    gold_start: int,
    gold_end: int,
    *,
    max_trimmed_tokens: int,
    max_extended_tokens: int,
    min_retained_meaningful_gold_tokens: int,
    min_tokens: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    """Return literal spans that trim one gold boundary and extend the opposite one."""
    tokens = source_tokens(resource)
    gold_indices = tokens_for_span(tokens, gold_start, gold_end)
    if len(gold_indices) < 2:
        return []

    first, last = gold_indices[0], gold_indices[-1]
    gold_tokens = tokens[first : last + 1]
    candidates: list[dict[str, Any]] = []

    # The candidate drops the beginning of the gold answer but continues after it.
    for trimmed in range(1, min(max_trimmed_tokens, len(gold_tokens) - 1) + 1):
        retained = gold_tokens[trimmed:]
        if meaningful_token_count(retained) < min_retained_meaningful_gold_tokens:
            continue
        for extended in range(1, max_extended_tokens + 1):
            left_index, right_index = first + trimmed, last + extended
            if right_index >= len(tokens):
                continue
            candidate = source_span(resource, tokens[left_index].start(), tokens[right_index].end())
            if candidate_is_clean(candidate, min_tokens=min_tokens, min_chars=min_chars):
                candidates.append({
                    "rejected": candidate,
                    "shift_type": "trim_left_extend_right",
                    "trimmed_tokens": trimmed,
                    "extended_tokens": extended,
                    "retained_meaningful_gold_tokens": meaningful_token_count(retained),
                })

    # The candidate starts before the gold answer but drops its ending.
    for trimmed in range(1, min(max_trimmed_tokens, len(gold_tokens) - 1) + 1):
        retained = gold_tokens[:-trimmed]
        if meaningful_token_count(retained) < min_retained_meaningful_gold_tokens:
            continue
        for extended in range(1, max_extended_tokens + 1):
            left_index, right_index = first - extended, last - trimmed
            if left_index < 0 or left_index > right_index:
                continue
            candidate = source_span(resource, tokens[left_index].start(), tokens[right_index].end())
            if candidate_is_clean(candidate, min_tokens=min_tokens, min_chars=min_chars):
                candidates.append({
                    "rejected": candidate,
                    "shift_type": "extend_left_trim_right",
                    "trimmed_tokens": trimmed,
                    "extended_tokens": extended,
                    "retained_meaningful_gold_tokens": meaningful_token_count(retained),
                })
    return candidates


def candidate_row(
    *,
    question_id: str,
    question: str,
    gold_alias: str,
    candidate: Mapping[str, Any],
    resource_index: int,
    source_start: int,
    source_end: int,
    source_resource: str,
) -> dict[str, Any]:
    context_start = max(0, source_start - 240)
    context_end = min(len(source_resource), source_end + 240)
    return {
        "question_id": question_id,
        "question_text": question,
        "chosen_entity": gold_alias,
        "chosen": output_completion(gold_alias),
        "rejected_entity": str(candidate["rejected"]),
        "rejected": output_completion(str(candidate["rejected"])),
        "span_direction": "boundary_shift",
        "boundary_shift_type": str(candidate["shift_type"]),
        "trimmed_tokens": int(candidate["trimmed_tokens"]),
        "extended_tokens": int(candidate["extended_tokens"]),
        "retained_meaningful_gold_tokens": int(candidate["retained_meaningful_gold_tokens"]),
        "resource_index": resource_index,
        "source_start": source_start,
        "source_end": source_end,
        "span_delta_tokens": int(candidate["extended_tokens"]) - int(candidate["trimmed_tokens"]),
        "source_context": source_resource[context_start:context_end],
        "review_decision": "",
        "review_notes": "",
    }


def mine_question(record: Mapping[str, Any], args: argparse.Namespace, audit: Counter[str]) -> list[dict[str, Any]]:
    question_id = clean_text(record.get("id"))
    question = clean_text(record.get("input_1"))
    resources = [mining_resource_text(resource) for resource in list_record_resources(dict(record))]
    aliases = parse_gold_aliases(record.get("output"))
    if not question_id or not question or not resources or not aliases:
        audit["questions_missing_required_fields"] += 1
        return []

    accepted_keys = {compact_normalized(alias) for alias in aliases if compact_normalized(alias)}
    present_by_alias = first_present_aliases(resources, aliases)
    if not present_by_alias:
        audit["questions_without_literal_gold_surface"] += 1
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for gold_alias, occurrences in present_by_alias.items():
        if not gold_is_short_expression(gold_alias, max_gold_tokens=max(1, args.max_gold_tokens)):
            audit["gold_aliases_rejected_sentence_like"] += 1
            continue
        other_gold_keys = accepted_keys - {compact_normalized(gold_alias)}
        for occurrence in occurrences:
            resource = resources[occurrence["resource_index"] - 1]
            for candidate in mine_boundary_shift_candidates(
                resource,
                occurrence["start"],
                occurrence["end"],
                max_trimmed_tokens=max(1, args.max_trimmed_tokens),
                max_extended_tokens=max(1, args.max_extended_tokens),
                min_retained_meaningful_gold_tokens=max(1, args.min_retained_meaningful_gold_tokens),
                min_tokens=max(1, args.min_negative_tokens),
                min_chars=max(1, args.min_negative_chars),
            ):
                rejected_key = compact_normalized(str(candidate["rejected"]))
                if not rejected_key or rejected_key in accepted_keys:
                    audit["candidates_rejected_matches_gold_alias"] += 1
                    continue
                if any(other_key and other_key in rejected_key for other_key in other_gold_keys):
                    audit["candidates_rejected_contains_other_gold_alias"] += 1
                    continue
                gold_key = compact_normalized(gold_alias)
                # Boundary shifts must overlap the gold span, but cannot be a strict sub/super span.
                if not gold_key or gold_key in rejected_key or rejected_key in gold_key:
                    audit["candidates_rejected_not_shifted_overlap"] += 1
                    continue
                key = (gold_alias, str(candidate["rejected"]), str(candidate["shift_type"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(candidate_row(
                    question_id=question_id,
                    question=question,
                    gold_alias=gold_alias,
                    candidate=candidate,
                    resource_index=occurrence["resource_index"],
                    source_start=occurrence["start"],
                    source_end=occurrence["end"],
                    source_resource=resource,
                ))
    audit["raw_candidates"] += len(rows)
    return rows


def select_question_candidates(rows: Sequence[Mapping[str, Any]], max_negatives: int) -> list[dict[str, Any]]:
    """Keep the closest candidate for each opposite-boundary shift type."""
    by_gold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold[str(row["chosen_entity"])].append(row)

    choices: list[tuple[tuple[int, int, str], list[dict[str, Any]]]] = []
    for gold_alias, group in by_gold.items():
        selected: list[dict[str, Any]] = []
        for shift_type in ("trim_left_extend_right", "extend_left_trim_right"):
            candidates = [row for row in group if row["boundary_shift_type"] == shift_type]
            if candidates:
                selected.append(dict(min(
                    candidates,
                    key=lambda row: (
                        int(row["trimmed_tokens"]) + int(row["extended_tokens"]),
                        len(str(row["rejected_entity"])),
                        str(row["rejected_entity"]).lower(),
                    ),
                )))
        if selected:
            choices.append(((-len(selected), sum(int(row["trimmed_tokens"]) + int(row["extended_tokens"]) for row in selected), gold_alias.lower()), selected))
    if not choices:
        return []
    _, selected = min(choices, key=lambda item: item[0])
    return selected[: max(0, max_negatives)]


def build_question_slate(record: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], input_path: Path, split: str) -> dict[str, Any]:
    first = selected[0]
    aliases = parse_gold_aliases(record.get("output"))
    return {
        "dataset": "bioasq-factoid",
        "split": split,
        "question_id": clean_text(record.get("id")),
        "question_type": clean_text(record.get("type")) or "factoid",
        "question_text": clean_text(record.get("input_1")),
        "instruction": clean_multiline_text(record.get("instruction")),
        "question_source_path": str(input_path),
        "prompt": "",
        "prompt_truncated": False,
        "snippets": list_record_resources(dict(record)),
        "canonical_gold_entity": str(first["chosen_entity"]),
        "canonical_gold_output": str(first["chosen"]),
        "accepted_gold_entities": aliases,
        "accepted_gold_outputs": [output_completion(alias) for alias in aliases],
        "chosen_entity": str(first["chosen_entity"]),
        "chosen_output": str(first["chosen"]),
        "chosen": str(first["chosen"]),
        "wrong_entities": [str(row["rejected_entity"]) for row in selected],
        "wrong_outputs": [str(row["rejected"]) for row in selected],
        "negatives": [str(row["rejected"]) for row in selected],
        "negative_metadata": [{
            "direction": "boundary_shift",
            "shift_type": row["boundary_shift_type"],
            "entity": row["rejected_entity"],
            "trimmed_tokens": row["trimmed_tokens"],
            "extended_tokens": row["extended_tokens"],
            "retained_meaningful_gold_tokens": row["retained_meaningful_gold_tokens"],
            "resource_index": row["resource_index"],
            "review_status": "needs_manual_review",
        } for row in selected],
        "pair_mining_method": "literal_source_grounded_boundary_shift",
        "review_status": "needs_manual_review",
    }


def write_review_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "pair_id", "question_id", "question_text", "chosen_entity", "rejected_entity",
        "boundary_shift_type", "trimmed_tokens", "extended_tokens", "retained_meaningful_gold_tokens",
        "resource_index", "source_start", "source_end", "source_context", "review_decision", "review_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_pre_audit(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "pair_id", "question_id", "question_text", "chosen_entity", "rejected_entity",
        "boundary_shift_type", "trimmed_tokens", "extended_tokens", "retained_meaningful_gold_tokens",
        "chosen_in_source_context", "rejected_in_source_context", "retains_meaningful_gold_token",
        "is_not_strict_subspan_or_superspan", "pre_audit_disposition", "pre_audit_rationale", "source_context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            gold_key = compact_normalized(str(row["chosen_entity"]))
            rejected_key = compact_normalized(str(row["rejected_entity"]))
            formal = {
                "chosen_in_source_context": gold_key in compact_normalized(str(row["source_context"])),
                "rejected_in_source_context": rejected_key in compact_normalized(str(row["source_context"])),
                "retains_meaningful_gold_token": int(row["retained_meaningful_gold_tokens"]) >= 1,
                "is_not_strict_subspan_or_superspan": gold_key not in rejected_key and rejected_key not in gold_key,
            }
            failed = [name for name, value in formal.items() if not value]
            payload = {
                **row,
                **formal,
                "pre_audit_disposition": "manual_review_high_risk" if not failed else "reject_formal",
                "pre_audit_rationale": "formal_checks_passed_shift_is_semantically_high_risk" if not failed else ";".join(failed),
            }
            writer.writerow({field: payload.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    record_ids = [clean_text(record.get("id")) for record in records if isinstance(record, Mapping)]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Input must contain unique question IDs, not a per-alias expansion.")

    audit: Counter[str] = Counter()
    all_candidates: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    slates: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            audit["invalid_records"] += 1
            continue
        candidates = mine_question(record, args, audit)
        all_candidates.extend(candidates)
        selected = select_question_candidates(candidates, max(1, args.max_negatives_per_question))
        if not selected:
            audit["questions_without_selected_pairs"] += 1
            continue
        slates.append(build_question_slate(record, selected, input_path, args.split))
        for index, row in enumerate(selected, start=1):
            pair_rows.append({
                **row,
                "pair_id": f"{row['question_id']}-boundary-shift-{row['boundary_shift_type']}-{index}",
                "dataset": "bioasq-factoid",
                "split": args.split,
                "question_type": clean_text(record.get("type")) or "factoid",
                "instruction": clean_multiline_text(record.get("instruction")),
                "snippets": list_record_resources(dict(record)),
                "accepted_gold_entities": parse_gold_aliases(record.get("output")),
                "rejected_source": "literal_source_boundary_shift_mining",
                "candidate_label": "needs_manual_review",
            })

    pair_rows.sort(key=lambda row: (str(row["question_id"]), str(row["boundary_shift_type"])))
    write_jsonl(output_dir / "all_mined_candidates.jsonl", all_candidates)
    write_jsonl(output_dir / "pairs_needing_review.jsonl", pair_rows)
    write_jsonl(output_dir / "question_slates_needing_review.jsonl", slates)
    write_review_csv(output_dir / "manual_review.csv", pair_rows)
    pre_audit_dir = output_dir / "pre_audit_v1"
    pre_audit_dir.mkdir(exist_ok=True)
    write_pre_audit(pre_audit_dir / "boundary_shift_pair_pre_audit.csv", pair_rows)

    direction_counts = Counter(row["boundary_shift_type"] for row in pair_rows)
    summary = {
        "input_json": str(input_path),
        "split": args.split,
        "input_question_count": len(records),
        "questions_with_selected_pairs": len(slates),
        "selected_pair_count": len(pair_rows),
        "selected_pair_count_by_shift_type": dict(sorted(direction_counts.items())),
        "all_mined_candidate_count": len(all_candidates),
        "limits": {
            "max_trimmed_tokens": args.max_trimmed_tokens,
            "max_extended_tokens": args.max_extended_tokens,
            "min_retained_meaningful_gold_tokens": args.min_retained_meaningful_gold_tokens,
            "min_negative_tokens": args.min_negative_tokens,
            "min_negative_chars": args.min_negative_chars,
            "max_gold_tokens": args.max_gold_tokens,
        },
        "audit": dict(sorted(audit.items())),
        "review_requirement": (
            "Every boundary-shift pair is high risk because it can alter meaning. Do not use these "
            "slates for DPO training until a reviewer marks each selected pair approved."
        ),
        "outputs": {
            "all_mined_candidates": str(output_dir / "all_mined_candidates.jsonl"),
            "pairs_needing_review": str(output_dir / "pairs_needing_review.jsonl"),
            "question_slates_needing_review": str(output_dir / "question_slates_needing_review.jsonl"),
            "manual_review_csv": str(output_dir / "manual_review.csv"),
            "pre_audit_csv": str(pre_audit_dir / "boundary_shift_pair_pre_audit.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
