from __future__ import annotations

import argparse
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from src.utility.bioasq_format import normalize_for_match
from src.utility.data import clean_text

from .common import load_json_records, write_jsonl
from .schemas import ParsedListOutput, SemanticListItem, to_jsonable


BEGIN_TAG = "[BI]"
END_TAG = "[EI]"
TAGGED_ITEM_PATTERN = re.compile(re.escape(BEGIN_TAG) + r"(.*?)" + re.escape(END_TAG), flags=re.DOTALL)
PLACEHOLDER_ITEM_PATTERN = re.compile(r"^(?:item|expression)_\d+$", flags=re.IGNORECASE)
GREEK_TRANSLATION = str.maketrans(
    {
        "α": " alpha ",
        "β": " beta ",
        "γ": " gamma ",
        "δ": " delta ",
        "ε": " epsilon ",
        "κ": " kappa ",
        "λ": " lambda ",
        "μ": " mu ",
        "τ": " tau ",
        "ω": " omega ",
        "Α": " alpha ",
        "Β": " beta ",
        "Γ": " gamma ",
        "Δ": " delta ",
        "Ε": " epsilon ",
        "Κ": " kappa ",
        "Λ": " lambda ",
        "Μ": " mu ",
        "Τ": " tau ",
        "Ω": " omega ",
    }
)


def normalize_answer_surface(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", clean_text(text))
    cleaned = cleaned.translate(GREEK_TRANSLATION)
    cleaned = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", cleaned)
    return normalize_for_match(cleaned)


def normalized_surface_variants(text: str) -> tuple[str, ...]:
    normalized = normalize_answer_surface(text)
    if not normalized:
        return ()

    variants = [normalized]
    collapsed_spaces = normalized.replace(" ", "")
    if collapsed_spaces and collapsed_spaces != normalized:
        variants.append(collapsed_spaces)
    return tuple(dict.fromkeys(variants))


def normalize_evidence_text(values: Sequence[str]) -> str:
    return normalize_answer_surface(" ".join(clean_text(value) for value in values if clean_text(value)))


def is_format_placeholder_item(text: str) -> bool:
    return bool(PLACEHOLDER_ITEM_PATTERN.fullmatch(clean_text(text)))


def filter_placeholder_items(items: Iterable[str]) -> tuple[list[str], int]:
    filtered_items: list[str] = []
    dropped_count = 0
    for raw_item in items:
        item = clean_text(raw_item)
        if not item:
            continue
        if is_format_placeholder_item(item):
            dropped_count += 1
            continue
        filtered_items.append(item)
    return filtered_items, dropped_count


def parse_list_output(text: str, allow_fallback_split: bool = True) -> ParsedListOutput:
    raw_output = clean_text(text)
    if not raw_output:
        return ParsedListOutput(
            raw_output="",
            items=(),
            status="empty",
            warnings=(),
            begin_tag_count=0,
            end_tag_count=0,
            empty_item_count=0,
            dropped_placeholder_count=0,
            used_fallback_split=False,
        )

    begin_count = raw_output.count(BEGIN_TAG)
    end_count = raw_output.count(END_TAG)
    warnings: list[str] = []
    empty_item_count = 0

    extracted_items: list[str] = []
    for match in TAGGED_ITEM_PATTERN.finditer(raw_output):
        item = clean_text(match.group(1))
        if item:
            extracted_items.append(item)
        else:
            empty_item_count += 1

    extracted_items, dropped_placeholder_count = filter_placeholder_items(extracted_items)

    if empty_item_count:
        warnings.append("empty_items")
    if dropped_placeholder_count:
        warnings.append("placeholder_items_dropped")

    impossible_order = begin_count == 0 and end_count > 0
    mismatched_tags = begin_count != end_count
    if impossible_order:
        warnings.append("end_tag_without_begin_tag")
    if mismatched_tags:
        warnings.append("mismatched_tag_count")
    if begin_count > end_count:
        warnings.append("truncated_final_item")

    if begin_count and end_count and extracted_items:
        status = "malformed" if impossible_order or mismatched_tags else "ok"
        return ParsedListOutput(
            raw_output=raw_output,
            items=tuple(extracted_items),
            status=status,
            warnings=tuple(warnings),
            begin_tag_count=begin_count,
            end_tag_count=end_count,
            empty_item_count=empty_item_count,
            dropped_placeholder_count=dropped_placeholder_count,
            used_fallback_split=False,
        )

    if begin_count or end_count:
        return ParsedListOutput(
            raw_output=raw_output,
            items=tuple(extracted_items),
            status="malformed",
            warnings=tuple(warnings or ["unparseable_tag_structure"]),
            begin_tag_count=begin_count,
            end_tag_count=end_count,
            empty_item_count=empty_item_count,
            dropped_placeholder_count=dropped_placeholder_count,
            used_fallback_split=False,
        )

    fallback_items: list[str] = []
    if allow_fallback_split:
        fallback_items, dropped_from_fallback = filter_placeholder_items(
            clean_text(part) for part in re.split(r"\n|;", raw_output)
        )
        dropped_placeholder_count += dropped_from_fallback
        if dropped_from_fallback and "placeholder_items_dropped" not in warnings:
            warnings.append("placeholder_items_dropped")

    return ParsedListOutput(
        raw_output=raw_output,
        items=tuple(fallback_items),
        status="fallback_split" if fallback_items else "malformed",
        warnings=tuple(warnings + ["missing_tags"]) if fallback_items else tuple(warnings + ["missing_tags", "no_items_found"]),
        begin_tag_count=0,
        end_tag_count=0,
        empty_item_count=0,
        dropped_placeholder_count=dropped_placeholder_count,
        used_fallback_split=bool(fallback_items),
    )


def dedupe_semantic_items(items: Sequence[str]) -> list[SemanticListItem]:
    ordered: OrderedDict[str, SemanticListItem] = OrderedDict()
    for index, raw_item in enumerate(items):
        surface = clean_text(raw_item)
        normalized = normalize_answer_surface(surface)
        if not normalized:
            continue
        existing = ordered.get(normalized)
        if existing is None:
            ordered[normalized] = SemanticListItem(
                surface=surface,
                normalized=normalized,
                first_index=index,
                duplicate_count=1,
            )
            continue
        ordered[normalized] = SemanticListItem(
            surface=existing.surface,
            normalized=existing.normalized,
            first_index=existing.first_index,
            duplicate_count=existing.duplicate_count + 1,
        )
    return list(ordered.values())


def serialize_list_items(items: Sequence[str]) -> str:
    return " ".join(f"{BEGIN_TAG}{clean_text(item)}{END_TAG}" for item in items if clean_text(item))


def iter_normalized_rows(rows: Iterable[Mapping[str, object]], allow_fallback_split: bool) -> list[dict[str, object]]:
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        raw_output = clean_text(row.get("raw_output") if isinstance(row, Mapping) else "")
        parsed = parse_list_output(raw_output, allow_fallback_split=allow_fallback_split)
        semantic_items = dedupe_semantic_items(parsed.items)
        normalized_rows.append(
            {
                **dict(row),
                "parsed_items": list(parsed.items),
                "parser_status": parsed.status,
                "parser_warnings": list(parsed.warnings),
                "dropped_placeholder_count": parsed.dropped_placeholder_count,
                "semantic_items": [to_jsonable(item) for item in semantic_items],
            }
        )
    return normalized_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse and normalize list-style candidate-bank outputs. "
            "This stage flags malformed tagged outputs and collapses exact-normalized duplicates."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more candidate-bank JSON or JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL path where normalized rows will be written.",
    )
    parser.add_argument(
        "--disallow-fallback-split",
        action="store_true",
        help="Reject untagged list outputs instead of splitting on newline or semicolon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[Mapping[str, object]] = []
    for raw_path in args.input:
        rows.extend(load_json_records(Path(raw_path)))

    normalized_rows = iter_normalized_rows(
        rows,
        allow_fallback_split=not args.disallow_fallback_split,
    )
    write_jsonl(Path(args.output), normalized_rows)


if __name__ == "__main__":
    main()
