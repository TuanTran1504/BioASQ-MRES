"""Build compact DPO source inputs containing only pair-justifying gold snippets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.build_factoid_extractive_span_pairs import find_literal_occurrences, parse_gold_aliases, write_json
from src.utility.data import clean_text, list_record_resources


MARKED_SNIPPET_PATTERN = re.compile(r"\[BS\](.*?)\[ES\]", flags=re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"\[BE\](.*?)\[EE\]", flags=re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slates-jsonl", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def completion_entity(completion: object) -> str:
    matches = ANSWER_PATTERN.findall(str(completion or ""))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one [BE]...[EE] answer: {completion!r}")
    entity = clean_text(matches[0])
    if not entity:
        raise ValueError(f"Empty completion entity: {completion!r}")
    return entity


def resource_snippet(record: Mapping[str, Any], resource_index: int, snippet_index: int) -> str:
    resources = list_record_resources(dict(record))
    if not 1 <= resource_index <= len(resources):
        raise ValueError(f"Invalid resource index {resource_index} for {record.get('id')}")
    snippets = [clean_text(match.group(1)) for match in MARKED_SNIPPET_PATTERN.finditer(resources[resource_index - 1])]
    if not 1 <= snippet_index <= len(snippets):
        raise ValueError(f"Invalid snippet index {snippet_index} in resource {resource_index} for {record.get('id')}")
    return f"[BS]{snippets[snippet_index - 1]}[ES]"


def build_record(record: Mapping[str, Any], slate: Mapping[str, Any]) -> dict[str, Any]:
    chosen_entity = completion_entity(slate["chosen"])
    negatives = list(slate.get("negatives") or [])
    candidates = list(slate.get("candidate_bank_selected_candidates") or [])
    if len(negatives) != len(candidates):
        raise ValueError(f"Negative/candidate metadata count mismatch for {slate.get('question_id')}")

    selected: list[tuple[int, int, str]] = []
    for negative, candidate in zip(negatives, candidates):
        negative_entity = completion_entity(negative)
        if negative_entity != clean_text(candidate.get("rejected_entity")):
            raise ValueError(f"Negative metadata mismatch for {slate.get('question_id')}")
        if chosen_entity != clean_text(candidate.get("chosen_entity")):
            raise ValueError(f"Chosen metadata mismatch for {slate.get('question_id')}")
        resource_index = int(candidate["resource_index"])
        snippet_index = int(candidate["snippet_index"])
        snippet = resource_snippet(record, resource_index, snippet_index)
        if not find_literal_occurrences(snippet, chosen_entity):
            raise ValueError(f"Chosen answer is absent from selected snippet for {slate.get('question_id')}")
        if not find_literal_occurrences(snippet, negative_entity):
            raise ValueError(f"Negative answer is absent from selected snippet for {slate.get('question_id')}")
        selected.append((resource_index, snippet_index, snippet))

    unique_selected: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for item in selected:
        key = item[:2]
        if key not in seen:
            seen.add(key)
            unique_selected.append(item)

    compact = {
        key: value for key, value in record.items()
        if not re.fullmatch(r"input_\d+", str(key))
    }
    compact["input_1"] = clean_text(record.get("input_1"))
    compact["candidate_bank_dpo_gold_snippet_metadata"] = [
        {"resource_index": resource_index, "snippet_index": snippet_index}
        for resource_index, snippet_index, _ in unique_selected
    ]
    for input_index, (_, _, snippet) in enumerate(unique_selected, start=2):
        compact[f"input_{input_index}"] = snippet
    return compact


def main() -> None:
    args = parse_args()
    slates_path = Path(args.slates_jsonl).expanduser().resolve()
    source_path = Path(args.source_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    if (output_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite an existing compact DPO source export")

    source_records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_records, list):
        raise ValueError(f"Expected source JSON list: {source_path}")
    source_by_id = {clean_text(record.get("id")): record for record in source_records}
    if len(source_by_id) != len(source_records):
        raise ValueError("Source input has duplicate or empty question IDs")

    slates = load_jsonl(slates_path)
    compact_records = []
    for slate in slates:
        question_id = clean_text(slate.get("question_id"))
        record = source_by_id.get(question_id)
        if record is None:
            raise ValueError(f"Missing source record for {question_id}")
        compact_records.append(build_record(record, slate))

    if len({record["id"] for record in compact_records}) != len(compact_records):
        raise ValueError("Compact output unexpectedly contains duplicate question IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "split": args.split,
        "slates_jsonl": str(slates_path),
        "source_json": str(source_path),
        "output_json": str(output_path),
        "question_count": len(compact_records),
        "resource_selection": "Only individual [BS]...[ES] snippets that literally contain both the selected gold and rejected candidate.",
        "max_selected_snippets_per_question": max(
            (len(record["candidate_bank_dpo_gold_snippet_metadata"]) for record in compact_records),
            default=0,
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
