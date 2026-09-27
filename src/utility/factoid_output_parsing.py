from __future__ import annotations

import re
from typing import Sequence

from .bioasq_format import normalize_for_match, parse_prediction_items, parse_tagged_items
from .data import clean_text


TAG_RE = re.compile(r"\[(?:BE|EE|BS|ES|BI|EI)\]")
NUMBERED_MARKER_RE = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=\s))(?:\d+[\.\)]|[-*])\s+")
LEADING_LABEL_RE = re.compile(r"^(?:answer|answers|candidate|candidates)\s*:\s*", flags=re.IGNORECASE)


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
    return items


def strip_factoid_markup(text: str) -> str:
    return clean_text(TAG_RE.sub(" ", text or ""))


def clean_factoid_candidate_text(text: str) -> str:
    cleaned = strip_factoid_markup(text)
    cleaned = LEADING_LABEL_RE.sub("", cleaned)
    # Remove true list markers like "1. answer" or "- answer", but preserve
    # biomedical strings that legitimately begin with digits, such as "53BP1",
    # "1:2000", or "3%".
    cleaned = re.sub(r"^\s*(?:\(?\d+\)?[\.\)]|[-*])\s+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" \t\r\n\"'`|")
    cleaned = re.sub(r"[;,:]+$", "", cleaned).strip()
    return cleaned


def split_numbered_factoid_segments(text: str) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    matches = list(NUMBERED_MARKER_RE.finditer(normalized))
    if len(matches) < 2:
        return []

    segments: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = clean_factoid_candidate_text(normalized[start:end])
        if segment:
            segments.append(segment)
    return dedupe_preserve_order(segments)


def parse_factoid_candidates(text: str, *, parser_mode: str) -> list[str]:
    normalized_mode = clean_text(parser_mode).lower() or "current"
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []

    if normalized_mode == "current":
        return dedupe_preserve_order(
            clean_factoid_candidate_text(item)
            for item in parse_prediction_items(cleaned_text, "factoid")
        )

    if normalized_mode != "agnostic":
        raise ValueError(f"Unsupported factoid parser mode: {parser_mode}")

    tagged_items = dedupe_preserve_order(
        clean_factoid_candidate_text(item)
        for item in parse_tagged_items(cleaned_text, "[BE]", "[EE]")
    )
    tagged_items = [item for item in tagged_items if item]
    if tagged_items:
        return tagged_items

    numbered_segments = split_numbered_factoid_segments(cleaned_text)
    if numbered_segments:
        return numbered_segments

    fallback = clean_factoid_candidate_text(cleaned_text)
    return [fallback] if fallback else []


def aggregate_factoid_candidates(
    samples: Sequence[str],
    *,
    strategy: str,
    min_frequency: int,
    max_candidates: int,
    parser_mode: str,
) -> list[str]:
    if not samples or max_candidates <= 0:
        return []

    parsed_by_sample = [
        parse_factoid_candidates(sample, parser_mode=parser_mode)
        for sample in samples
    ]

    normalized_strategy = clean_text(strategy).lower() or "union"
    if normalized_strategy == "union":
        seen: set[str] = set()
        aggregated: list[str] = []
        for sample_items in parsed_by_sample:
            for item in sample_items:
                key = normalize_for_match(item) or item.lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                aggregated.append(item)
                if len(aggregated) >= max_candidates:
                    return aggregated
        return aggregated

    if normalized_strategy == "frequency":
        representative_by_key: dict[str, str] = {}
        first_seen_order: dict[str, int] = {}
        frequency: dict[str, int] = {}
        observation_index = 0

        for sample_items in parsed_by_sample:
            sample_keys: set[str] = set()
            for item in sample_items:
                key = normalize_for_match(item) or item.lower()
                if not key or key in sample_keys:
                    continue
                sample_keys.add(key)
                if key not in representative_by_key:
                    representative_by_key[key] = item
                    first_seen_order[key] = observation_index
                observation_index += 1
            for key in sample_keys:
                frequency[key] = frequency.get(key, 0) + 1

        passing_keys = [
            key for key, count in frequency.items()
            if count >= max(1, int(min_frequency))
        ]
        passing_keys.sort(key=lambda key: first_seen_order[key])
        aggregated = [representative_by_key[key] for key in passing_keys[:max_candidates]]
        if aggregated:
            return aggregated

        return aggregate_factoid_candidates(
            samples,
            strategy="union",
            min_frequency=min_frequency,
            max_candidates=max_candidates,
            parser_mode=parser_mode,
        )

    raise ValueError(f"Unsupported factoid aggregation strategy: {strategy}")
