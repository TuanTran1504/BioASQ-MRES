"""Mine auditable, extractive short/long span DPO candidates from SFT data.

This is intentionally conservative. It creates only source-grounded candidates
whose text is a strict normalized substring or superstring of a supported gold
alias. The output must be manually reviewed before it is used for DPO training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cse_dpo.normalize_set_answers import normalize_answer_surface
from src.utility.data import clean_multiline_text, clean_text, list_record_resources


ANSWER_PATTERN = re.compile(r"\[BE\](.*?)\[EE\]", flags=re.DOTALL | re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[+./'-][A-Za-z0-9]+)*")
EVIDENCE_MARKER_PATTERN = re.compile(r"\[(?:BS|ES)\]", flags=re.IGNORECASE)
EDGE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "be", "been", "being", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "in", "into", "is", "it", "its", "may", "might", "must",
    "of", "on", "or", "should", "that", "the", "their", "these", "this", "to", "via",
    "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would",
}
GENERIC_SINGLE_TOKENS = {
    "activity", "antibody", "channels", "disease", "function", "gene", "genes",
    "marker", "pathway", "protein", "proteins", "receptor", "receptors", "treatment",
}
WEAK_EXTENSION_TOKENS = {
    "cell", "cells", "disease", "disorder", "gene", "genes", "patient", "patients",
    "protein", "proteins", "study", "studies", "therapy", "treatment", "type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        required=True,
        help="Prepared, unique-question SFT JSON input; use the 1,130-question evidence-supported train file.",
    )
    parser.add_argument("--output-dir", required=True, help="New directory for mined slates, pairs, and review files.")
    parser.add_argument("--split", default="train", help="Split label written to outputs. Default: train.")
    parser.add_argument("--max-negatives-per-question", type=int, default=2)
    parser.add_argument("--max-shortened-tokens", type=int, default=2)
    parser.add_argument("--max-extended-tokens", type=int, default=2)
    parser.add_argument("--min-negative-tokens", type=int, default=2)
    parser.add_argument("--min-negative-chars", type=int, default=4)
    parser.add_argument(
        "--max-gold-tokens",
        type=int,
        default=12,
        help="Exclude sentence-like gold targets with more tokens than this. Default: 12.",
    )
    parser.add_argument(
        "--long-to-short-ratio",
        type=float,
        default=1.0,
        help="Keep at most this many long pairs per retained short pair. Default: 1.0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory. Existing output files are replaced.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def parse_gold_aliases(output: object) -> list[str]:
    aliases = [clean_text(match.group(1)) for match in ANSWER_PATTERN.finditer(str(output or ""))]
    aliases = [alias for alias in aliases if alias]
    if not aliases:
        fallback = clean_text(output)
        aliases = [fallback] if fallback else []

    unique: dict[str, str] = {}
    for alias in aliases:
        normalized = normalize_answer_surface(alias)
        if normalized and normalized not in unique:
            unique[normalized] = alias
    return list(unique.values())


def compact_normalized(text: str) -> str:
    return re.sub(r"\s+", "", normalize_answer_surface(text))


def is_strict_subspan(candidate: str, gold: str) -> bool:
    candidate_key = compact_normalized(candidate)
    gold_key = compact_normalized(gold)
    return bool(candidate_key and gold_key and candidate_key != gold_key and candidate_key in gold_key)


def is_strict_superspan(candidate: str, gold: str) -> bool:
    candidate_key = compact_normalized(candidate)
    gold_key = compact_normalized(gold)
    return bool(candidate_key and gold_key and candidate_key != gold_key and gold_key in candidate_key)


def find_literal_occurrences(resource: str, gold: str) -> list[tuple[int, int]]:
    """Find whole-token, whitespace-tolerant occurrences while preserving source offsets."""
    words = clean_text(gold).split()
    if not words:
        return []
    pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(word) for word in words) + r"(?![A-Za-z0-9])"
    return [(match.start(), match.end()) for match in re.finditer(pattern, resource, flags=re.IGNORECASE)]


def source_tokens(resource: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(resource))


def tokens_for_span(tokens: Sequence[re.Match[str]], start: int, end: int) -> list[int]:
    return [index for index, token in enumerate(tokens) if token.start() >= start and token.end() <= end]


def candidate_is_clean(candidate: str, *, min_tokens: int, min_chars: int) -> bool:
    cleaned = clean_text(candidate)
    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(cleaned)]
    if len(cleaned) < min_chars or len(tokens) < min_tokens:
        return False
    normalized_tokens = [token.lower() for token in tokens]
    if normalized_tokens[0] in EDGE_STOPWORDS or normalized_tokens[-1] in EDGE_STOPWORDS:
        return False
    if len(normalized_tokens) == 1 and normalized_tokens[0] in GENERIC_SINGLE_TOKENS:
        return False
    # A span crossing normal sentence punctuation is not an answer expression.
    if re.search(r"[.!?;:]\s", cleaned):
        return False
    if cleaned.count("(") != cleaned.count(")") or cleaned.count("[") != cleaned.count("]"):
        return False
    return True


def source_span(resource: str, start: int, end: int) -> str:
    return resource[start:end].strip(" \t\n.,;:()[]{}\"'")


def mining_resource_text(resource: str) -> str:
    """Remove evidence delimiters so they can never become answer-span tokens."""
    # A period creates a hard mining boundary between PubMed headers and marked
    # snippets, and between adjacent marked snippets in one resource field.
    return clean_multiline_text(EVIDENCE_MARKER_PATTERN.sub(". ", resource))


def gold_is_short_expression(gold: str, *, max_gold_tokens: int) -> bool:
    """Reject gold annotations that contradict the short-expression task format."""
    tokens = TOKEN_PATTERN.findall(gold)
    if not tokens or len(tokens) > max_gold_tokens:
        return False
    if tokens[0].lower() in EDGE_STOPWORDS or tokens[-1].lower() in EDGE_STOPWORDS:
        return False
    return not bool(re.search(r"[.!?;:]\s", clean_text(gold)))


def mine_short_candidates(
    source_gold: str,
    *,
    max_shortened_tokens: int,
    min_tokens: int,
    min_chars: int,
) -> list[tuple[str, int]]:
    tokens = source_tokens(source_gold)
    candidates: list[tuple[str, int]] = []
    if len(tokens) <= min_tokens:
        return candidates

    for removed in range(1, min(max_shortened_tokens, len(tokens) - min_tokens) + 1):
        for token_slice in (tokens[removed:], tokens[:-removed]):
            candidate = source_span(source_gold, token_slice[0].start(), token_slice[-1].end())
            if candidate_is_clean(candidate, min_tokens=min_tokens, min_chars=min_chars):
                candidates.append((candidate, removed))
    return candidates


def mine_long_candidates(
    resource: str,
    gold_start: int,
    gold_end: int,
    *,
    max_extended_tokens: int,
    min_tokens: int,
    min_chars: int,
) -> list[tuple[str, int]]:
    tokens = source_tokens(resource)
    gold_indices = tokens_for_span(tokens, gold_start, gold_end)
    candidates: list[tuple[str, int]] = []
    if not gold_indices:
        return candidates

    first_index, last_index = gold_indices[0], gold_indices[-1]
    for left_added in range(max_extended_tokens + 1):
        for right_added in range(max_extended_tokens + 1 - left_added):
            added = left_added + right_added
            if added == 0:
                continue
            left_index = first_index - left_added
            right_index = last_index + right_added
            if left_index < 0 or right_index >= len(tokens):
                continue
            candidate = source_span(resource, tokens[left_index].start(), tokens[right_index].end())
            if candidate_is_clean(candidate, min_tokens=min_tokens, min_chars=min_chars):
                candidates.append((candidate, added))
    return candidates


def output_completion(entity: str) -> str:
    return f"[BE]{clean_text(entity)}[EE]"


def long_candidate_strength(gold: str, candidate: str, span_delta_tokens: int) -> int:
    """Prefer technical modifiers over generic context continuations."""
    gold_tokens = [token.lower() for token in TOKEN_PATTERN.findall(gold)]
    candidate_tokens = [token.lower() for token in TOKEN_PATTERN.findall(candidate)]
    extension: list[str] = []
    if len(candidate_tokens) > len(gold_tokens):
        if candidate_tokens[-len(gold_tokens):] == gold_tokens:
            extension = candidate_tokens[: len(candidate_tokens) - len(gold_tokens)]
        elif candidate_tokens[: len(gold_tokens)] == gold_tokens:
            extension = candidate_tokens[len(gold_tokens):]

    score = 100 - 5 * max(0, span_delta_tokens)
    score -= 8 * sum(token in WEAK_EXTENSION_TOKENS for token in extension)
    score -= 5 * sum(token.isdigit() or re.fullmatch(r"[ivxlcdm]+", token) is not None for token in extension)
    score += 4 * sum(any(char.isdigit() for char in token) or "-" in token for token in extension)
    score += sum(len(token) >= 8 for token in extension)
    return score


def first_present_aliases(resources: Sequence[str], aliases: Sequence[str]) -> dict[str, list[dict[str, int]]]:
    found: dict[str, list[dict[str, int]]] = defaultdict(list)
    for alias in aliases:
        for resource_index, resource in enumerate(resources, start=1):
            for start, end in find_literal_occurrences(resource, alias):
                found[alias].append({"resource_index": resource_index, "start": start, "end": end})
    return dict(found)


def candidate_row(
    *,
    question_id: str,
    question: str,
    gold_alias: str,
    rejected: str,
    direction: str,
    resource_index: int,
    source_start: int,
    source_end: int,
    span_delta_tokens: int,
    source_resource: str,
) -> dict[str, Any]:
    context_start = max(0, source_start - 240)
    context_end = min(len(source_resource), source_end + 240)
    return {
        "question_id": question_id,
        "question_text": question,
        "chosen_entity": gold_alias,
        "chosen": output_completion(gold_alias),
        "rejected_entity": rejected,
        "rejected": output_completion(rejected),
        "span_direction": direction,
        "resource_index": resource_index,
        "source_start": source_start,
        "source_end": source_end,
        "span_delta_tokens": span_delta_tokens,
        "long_candidate_strength": long_candidate_strength(gold_alias, rejected, span_delta_tokens)
        if direction == "too_long" else None,
        "source_context": source_resource[context_start:context_end],
        "review_decision": "",
        "review_notes": "",
    }


def mine_question(
    record: Mapping[str, Any],
    *,
    max_shortened_tokens: int,
    max_extended_tokens: int,
    min_tokens: int,
    min_chars: int,
    max_gold_tokens: int,
    audit: Counter,
) -> list[dict[str, Any]]:
    question_id = clean_text(record.get("id"))
    question = clean_text(record.get("input_1"))
    resources = [mining_resource_text(resource) for resource in list_record_resources(dict(record))]
    aliases = parse_gold_aliases(record.get("output"))
    if not question_id or not question or not resources or not aliases:
        audit["questions_missing_required_fields"] += 1
        return []

    accepted_gold_keys = {compact_normalized(alias) for alias in aliases if compact_normalized(alias)}
    present_by_alias = first_present_aliases(resources, aliases)
    if not present_by_alias:
        audit["questions_without_literal_gold_surface"] += 1
        return []

    all_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for gold_alias, occurrences in present_by_alias.items():
        if not gold_is_short_expression(gold_alias, max_gold_tokens=max_gold_tokens):
            audit["gold_aliases_rejected_sentence_like"] += 1
            continue
        other_gold_keys = accepted_gold_keys - {compact_normalized(gold_alias)}
        for occurrence in occurrences:
            resource = resources[occurrence["resource_index"] - 1]
            gold_start, gold_end = occurrence["start"], occurrence["end"]
            source_gold = source_span(resource, gold_start, gold_end)

            for rejected, removed in mine_short_candidates(
                source_gold,
                max_shortened_tokens=max_shortened_tokens,
                min_tokens=min_tokens,
                min_chars=min_chars,
            ):
                if compact_normalized(rejected) in accepted_gold_keys:
                    audit["candidates_rejected_matches_gold_alias"] += 1
                    continue
                if any(other_key and other_key in compact_normalized(rejected) for other_key in other_gold_keys):
                    audit["candidates_rejected_contains_other_gold_alias"] += 1
                    continue
                if not is_strict_subspan(rejected, gold_alias):
                    audit["candidates_rejected_not_strict_short_span"] += 1
                    continue
                key = (gold_alias, rejected, "too_short")
                if key not in seen:
                    seen.add(key)
                    all_rows.append(candidate_row(
                        question_id=question_id, question=question, gold_alias=gold_alias,
                        rejected=rejected, direction="too_short", resource_index=occurrence["resource_index"],
                        source_start=gold_start, source_end=gold_end, span_delta_tokens=removed,
                        source_resource=resource,
                    ))

            for rejected, added in mine_long_candidates(
                resource, gold_start, gold_end,
                max_extended_tokens=max_extended_tokens,
                min_tokens=min_tokens,
                min_chars=min_chars,
            ):
                if compact_normalized(rejected) in accepted_gold_keys:
                    audit["candidates_rejected_matches_gold_alias"] += 1
                    continue
                if any(other_key and other_key in compact_normalized(rejected) for other_key in other_gold_keys):
                    audit["candidates_rejected_contains_other_gold_alias"] += 1
                    continue
                if not is_strict_superspan(rejected, gold_alias):
                    audit["candidates_rejected_not_strict_long_span"] += 1
                    continue
                key = (gold_alias, rejected, "too_long")
                if key not in seen:
                    seen.add(key)
                    all_rows.append(candidate_row(
                        question_id=question_id, question=question, gold_alias=gold_alias,
                        rejected=rejected, direction="too_long", resource_index=occurrence["resource_index"],
                        source_start=gold_start, source_end=gold_end, span_delta_tokens=added,
                        source_resource=resource,
                    ))

    audit["raw_candidates"] += len(all_rows)
    return all_rows


def select_question_candidates(rows: Sequence[dict[str, Any]], max_negatives: int) -> list[dict[str, Any]]:
    """Choose one gold alias and, at most, one closest candidate per direction."""
    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold[row["chosen_entity"]].append(row)

    ranked_groups: list[tuple[tuple[int, int, int, str], list[dict[str, Any]]]] = []
    for gold_alias, group in by_gold.items():
        by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_direction[row["span_direction"]].append(row)
        selected: list[dict[str, Any]] = []
        for direction in ("too_short", "too_long"):
            candidates = sorted(
                by_direction[direction],
                key=lambda row: (int(row["span_delta_tokens"]), len(row["rejected_entity"]), row["rejected_entity"].lower()),
            )
            if candidates:
                selected.append(candidates[0])
        if selected:
            direction_count = len({row["span_direction"] for row in selected})
            total_delta = sum(int(row["span_delta_tokens"]) for row in selected)
            ranked_groups.append(((direction_count, len(selected), -total_delta, gold_alias.lower()), selected))

    if not ranked_groups:
        return []
    _, selected = max(ranked_groups, key=lambda item: item[0])
    return selected[:max(0, max_negatives)]


def build_question_slate(
    record: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    input_path: Path,
    split: str,
) -> dict[str, Any]:
    first = selected[0]
    aliases = parse_gold_aliases(record.get("output"))
    resources = [mining_resource_text(resource) for resource in list_record_resources(dict(record))]
    negatives = [str(row["rejected"]) for row in selected]
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
        "snippets": resources,
        "canonical_gold_entity": str(first["chosen_entity"]),
        "canonical_gold_output": str(first["chosen"]),
        "accepted_gold_entities": aliases,
        "accepted_gold_outputs": [output_completion(alias) for alias in aliases],
        "chosen_entity": str(first["chosen_entity"]),
        "chosen_output": str(first["chosen"]),
        "chosen": str(first["chosen"]),
        "wrong_entities": [str(row["rejected_entity"]) for row in selected],
        "wrong_outputs": negatives,
        "negatives": negatives,
        "negative_metadata": [
            {
                "direction": row["span_direction"],
                "entity": row["rejected_entity"],
                "resource_index": row["resource_index"],
                "source_start": row["source_start"],
                "source_end": row["source_end"],
                "span_delta_tokens": row["span_delta_tokens"],
                "review_status": "needs_manual_review",
            }
            for row in selected
        ],
        "pair_mining_method": "literal_source_grounded_strict_span",
        "review_status": "needs_manual_review",
    }


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
    ids = [clean_text(record.get("id")) for record in records if isinstance(record, Mapping)]
    if len(ids) != len(set(ids)):
        raise ValueError("Input must contain unique question IDs. Use the 1,130-question evidence-grounded file, not the per-alias expansion.")

    audit: Counter = Counter()
    all_candidates: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    slates: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    direction_counts: Counter = Counter()
    records_by_id = {clean_text(record.get("id")): record for record in records if isinstance(record, Mapping)}

    for record in records:
        if not isinstance(record, Mapping):
            audit["invalid_records"] += 1
            continue
        candidates = mine_question(
            record,
            max_shortened_tokens=max(1, args.max_shortened_tokens),
            max_extended_tokens=max(1, args.max_extended_tokens),
            min_tokens=max(1, args.min_negative_tokens),
            min_chars=max(1, args.min_negative_chars),
            max_gold_tokens=max(1, args.max_gold_tokens),
            audit=audit,
        )
        all_candidates.extend(candidates)
        selected = select_question_candidates(candidates, max(1, args.max_negatives_per_question))
        if not selected:
            audit["questions_without_selected_pairs"] += 1
            continue

        selected_candidates.extend(selected)
        slates.append(build_question_slate(record, selected, input_path=input_path, split=args.split))
        for index, row in enumerate(selected, start=1):
            direction_counts[str(row["span_direction"])] += 1
            pair_rows.append({
                **row,
                "pair_id": f"{row['question_id']}-span-{row['span_direction']}-{index}",
                "dataset": "bioasq-factoid",
                "split": args.split,
                "question_type": clean_text(record.get("type")) or "factoid",
                "instruction": clean_multiline_text(record.get("instruction")),
                "snippets": list_record_resources(dict(record)),
                "accepted_gold_entities": parse_gold_aliases(record.get("output")),
                "rejected_source": "literal_source_span_mining",
                "candidate_label": "needs_manual_review",
            })

    short_pairs = [row for row in pair_rows if row["span_direction"] == "too_short"]
    long_pairs = [row for row in pair_rows if row["span_direction"] == "too_long"]
    long_pair_limit = max(0, int(len(short_pairs) * max(0.0, args.long_to_short_ratio)))
    retained_long_pairs = sorted(
        long_pairs,
        key=lambda row: (
            -int(row.get("long_candidate_strength") or 0),
            int(row["span_delta_tokens"]),
            len(str(row["rejected_entity"])),
            str(row["question_id"]),
        ),
    )[:long_pair_limit]
    pair_rows = sorted(
        [*short_pairs, *retained_long_pairs],
        key=lambda row: (str(row["question_id"]), str(row["span_direction"])),
    )
    selected_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        selected_by_question[str(row["question_id"])].append(row)
    slates = [
        build_question_slate(records_by_id[question_id], selected, input_path=input_path, split=args.split)
        for question_id, selected in sorted(selected_by_question.items())
    ]
    direction_counts = Counter(row["span_direction"] for row in pair_rows)

    all_candidates_path = output_dir / "all_mined_candidates.jsonl"
    pairs_path = output_dir / "pairs_needing_review.jsonl"
    slates_path = output_dir / "question_slates_needing_review.jsonl"
    review_path = output_dir / "manual_review.csv"
    write_jsonl(all_candidates_path, all_candidates)
    write_jsonl(pairs_path, pair_rows)
    write_jsonl(slates_path, slates)
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "pair_id", "question_id", "question_text", "chosen_entity", "rejected_entity",
            "span_direction", "resource_index", "source_start", "source_end", "span_delta_tokens",
            "long_candidate_strength", "source_context", "review_decision", "review_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    pair_counts_by_question = Counter(row["question_id"] for row in pair_rows)
    summary = {
        "input_json": str(input_path),
        "split": args.split,
        "input_question_count": len(records),
        "questions_with_selected_pairs": len(slates),
        "selected_pair_count": len(pair_rows),
        "selected_pair_count_by_direction": dict(sorted(direction_counts.items())),
        "selected_pairs_per_question": dict(sorted(Counter(pair_counts_by_question.values()).items())),
        "all_mined_candidate_count": len(all_candidates),
        "limits": {
            "max_negatives_per_question": args.max_negatives_per_question,
            "max_shortened_tokens": args.max_shortened_tokens,
            "max_extended_tokens": args.max_extended_tokens,
            "min_negative_tokens": args.min_negative_tokens,
            "min_negative_chars": args.min_negative_chars,
            "max_gold_tokens": args.max_gold_tokens,
            "long_to_short_ratio": args.long_to_short_ratio,
        },
        "direction_balancing": {
            "long_pairs_before_balancing": len(long_pairs),
            "long_pair_limit": long_pair_limit,
            "long_pairs_retained": len(retained_long_pairs),
            "ranking_definition": "conservative lexical score; it is not a semantic-quality guarantee",
        },
        "audit": dict(sorted(audit.items())),
        "review_requirement": (
            "All exported pairs are synthetic source-grounded candidates and require manual approval. "
            "Do not use question_slates_needing_review.jsonl as DPO training input until reviewed."
        ),
        "outputs": {
            "all_mined_candidates": str(all_candidates_path),
            "pairs_needing_review": str(pairs_path),
            "question_slates_needing_review": str(slates_path),
            "manual_review_csv": str(review_path),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
