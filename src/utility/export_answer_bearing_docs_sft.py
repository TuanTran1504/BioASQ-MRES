from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import exact_answer_groups, normalize_for_match
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import (
    clean_multiline_text,
    clean_text,
    resolve_prompt_instructions,
    save_prepared_records,
    serialize_resource,
)
from src.utility.eval_types import EvalExample


DEFAULT_TRAIN_INPUT = ["data/training13b.json"]
DEFAULT_OUTPUT_DIR = "data/BioASQ_list_answer_bearing_docs_sft"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prepared list-only SFT dataset that keeps only document-level "
            "resources whose serialized text contains at least one BioASQ gold "
            "answer alias after normalization."
        )
    )
    parser.add_argument(
        "--train-input",
        nargs="+",
        default=DEFAULT_TRAIN_INPUT,
        help="One or more raw BioASQ JSON files used to build the train split.",
    )
    parser.add_argument(
        "--dev-input",
        nargs="*",
        default=None,
        help="Optional raw BioASQ JSON files used to build a dev split.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Project-relative or absolute directory where the prepared files will be written.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help=(
            "Maximum number of answer-bearing documents to keep per question after "
            "filtering. Use 0 to keep all answer-bearing documents."
        ),
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Maximum characters per serialized document resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--min-visible-items",
        type=int,
        default=1,
        help="Drop questions with fewer than this many visible gold groups after filtering.",
    )
    parser.add_argument(
        "--prompt-registry-path",
        default=None,
        help=(
            "Optional repo-relative prompt registry JSON path. If omitted, the "
            "built-in list instruction is used."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=(
            "Optional repo-relative JSON file describing one prompt bundle. This "
            "can be either a single inline prompt object or a full prompt registry."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt alias or prompt_id to resolve from the prompt registry.",
    )
    return parser.parse_args()


def iter_list_questions(paths: Sequence[Path]) -> Iterable[tuple[Path, Dict[str, Any]]]:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"Raw BioASQ file must contain a 'questions' list: {path}")
        for question in questions:
            if isinstance(question, dict) and clean_text(question.get("type", "")).lower() == "list":
                yield path, question


def make_eval_example(question: Dict[str, Any]) -> EvalExample:
    return EvalExample(
        question_id=clean_text(question.get("id", "")),
        question_type="list",
        body=clean_text(question.get("body", "")),
        instruction="",
        resources=(),
        gold_output="",
        source_path="",
        raw_question=question,
    )


def gold_groups_for_question(question: Dict[str, Any]) -> List[List[str]]:
    return exact_answer_groups(make_eval_example(question), "list")


def visible_aliases_in_text(gold_group: Sequence[str], text: str) -> List[str]:
    normalized_text = normalize_for_match(text)
    visible: List[str] = []
    seen: set[str] = set()
    for alias in gold_group:
        cleaned = clean_text(alias)
        alias_norm = normalize_for_match(cleaned)
        if not cleaned or not alias_norm or alias_norm in seen:
            continue
        if alias_norm in normalized_text:
            seen.add(alias_norm)
            visible.append(cleaned)
    return visible


def alias_specificity(alias: str) -> tuple[int, int, int, int]:
    normalized = normalize_for_match(alias)
    tokens = normalized.split() if normalized else []
    has_digits = int(any(char.isdigit() for char in alias))
    has_parenthetical = int("(" in alias or ")" in alias)
    return (has_digits, has_parenthetical, len(tokens), len(alias))


def choose_visible_alias(visible_aliases: Sequence[str]) -> Optional[str]:
    cleaned = [clean_text(alias) for alias in visible_aliases if clean_text(alias)]
    if not cleaned:
        return None
    return max(cleaned, key=alias_specificity)


def build_output_items(resources: Sequence[str], gold_groups: Sequence[Sequence[str]]) -> List[str]:
    items: List[str] = []
    combined_text = "\n".join(resources)
    for gold_group in gold_groups:
        chosen = choose_visible_alias(visible_aliases_in_text(gold_group, combined_text))
        if chosen:
            items.append(chosen)
    return items


def build_document_resources(question: Dict[str, Any], max_resource_chars: int) -> List[Tuple[str, str]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ordered_docs: List[str] = []

    for snippet in question.get("snippets", []):
        if not isinstance(snippet, dict):
            continue

        document_url = clean_text(snippet.get("document", ""))
        if not document_url:
            document_url = f"document_{len(ordered_docs) + 1}"

        if document_url not in grouped:
            grouped[document_url] = []
            ordered_docs.append(document_url)

        grouped[document_url].append(snippet)

    return [
        (document_url, serialize_resource(document_url, grouped[document_url], max_chars=max_resource_chars))
        for document_url in ordered_docs
    ]


def supported_gold_indices(resource_text: str, gold_groups: Sequence[Sequence[str]]) -> List[int]:
    supported: List[int] = []
    for gold_index, gold_group in enumerate(gold_groups):
        if choose_visible_alias(visible_aliases_in_text(gold_group, resource_text)):
            supported.append(gold_index)
    return supported


def build_record(
    question: Dict[str, Any],
    *,
    instruction: str,
    max_documents: int,
    max_resource_chars: int,
) -> Optional[Dict[str, Any]]:
    question_id = clean_text(question.get("id", ""))
    question_text = clean_text(question.get("body", ""))
    if not question_text:
        return None

    gold_groups = gold_groups_for_question(question)
    original_documents = build_document_resources(question, max_resource_chars=max_resource_chars)

    kept_documents: List[Tuple[str, str, List[int]]] = []
    for document_url, resource_text in original_documents:
        gold_indices = supported_gold_indices(resource_text, gold_groups)
        if gold_indices:
            kept_documents.append((document_url, resource_text, gold_indices))

    if max_documents > 0:
        kept_documents = kept_documents[:max_documents]

    kept_resource_texts = [resource_text for _, resource_text, _ in kept_documents]
    visible_items = build_output_items(kept_resource_texts, gold_groups)

    record: Dict[str, Any] = {
        "id": question_id,
        "type": "list",
        "instruction": instruction,
        "input_1": question_text,
        "output": " ".join(f"[BI]{item}[EI]" for item in visible_items),
        "visible_gold_count": len(visible_items),
        "original_gold_count": len(gold_groups),
        "kept_gold_ratio": len(visible_items) / len(gold_groups) if gold_groups else 0.0,
        "answer_bearing_document_count": len(kept_documents),
        "original_document_count": len(original_documents),
        "document_keep_ratio": (
            len(kept_documents) / len(original_documents) if original_documents else 0.0
        ),
        "source_question_id": question_id,
    }

    for index, resource_text in enumerate(kept_resource_texts, start=2):
        cleaned_resource = clean_multiline_text(resource_text)
        if cleaned_resource:
            record[f"input_{index}"] = cleaned_resource

    return record


def build_split(
    input_paths: Sequence[Path],
    *,
    instruction: str,
    min_visible_items: int,
    max_documents: int,
    max_resource_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    visible_counts: List[int] = []
    original_counts: List[int] = []
    kept_doc_counts: List[int] = []
    original_doc_counts: List[int] = []
    dropped_below_min_visible_items = 0
    total_questions = 0
    seen_ids: set[str] = set()

    for source_path, question in iter_list_questions(input_paths):
        total_questions += 1
        record = build_record(
            question,
            instruction=instruction,
            max_documents=max_documents,
            max_resource_chars=max_resource_chars,
        )
        if record is None:
            continue

        question_id = str(record.get("id") or f"{source_path.name}:{total_questions}")
        if question_id in seen_ids:
            continue
        seen_ids.add(question_id)

        visible_gold_count = int(record.get("visible_gold_count", 0) or 0)
        original_gold_count = int(record.get("original_gold_count", 0) or 0)
        answer_bearing_document_count = int(record.get("answer_bearing_document_count", 0) or 0)
        original_document_count = int(record.get("original_document_count", 0) or 0)

        if visible_gold_count < min_visible_items:
            dropped_below_min_visible_items += 1
            continue

        rows.append(record)
        visible_counts.append(visible_gold_count)
        original_counts.append(original_gold_count)
        kept_doc_counts.append(answer_bearing_document_count)
        original_doc_counts.append(original_document_count)

    summary = {
        "input_files": [str(path) for path in input_paths],
        "total_list_questions": total_questions,
        "kept_questions": len(rows),
        "dropped_below_min_visible_items": dropped_below_min_visible_items,
        "min_visible_items": min_visible_items,
        "average_visible_gold_count": statistics.mean(visible_counts) if visible_counts else 0.0,
        "average_original_gold_count": statistics.mean(original_counts) if original_counts else 0.0,
        "average_kept_gold_ratio": (
            statistics.mean(
                (visible / original) if original else 0.0
                for visible, original in zip(visible_counts, original_counts)
            )
            if visible_counts
            else 0.0
        ),
        "average_answer_bearing_document_count": (
            statistics.mean(kept_doc_counts) if kept_doc_counts else 0.0
        ),
        "average_original_document_count": (
            statistics.mean(original_doc_counts) if original_doc_counts else 0.0
        ),
        "average_document_keep_ratio": (
            statistics.mean(
                (kept / original) if original else 0.0
                for kept, original in zip(kept_doc_counts, original_doc_counts)
            )
            if kept_doc_counts
            else 0.0
        ),
    }
    return rows, summary


def write_summary(
    *,
    output_dir: Path,
    train_summary: dict[str, Any],
    dev_summary: Optional[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "dataset_name": "answer_bearing_docs_visible_gold_sft",
        "document_selection": "keep_document_if_any_gold_alias_is_visible_after_normalization",
        "target_policy": "visible_gold_only_over_kept_documents",
        "resource_granularity": "document",
        "max_documents": args.max_documents,
        "max_resource_chars": args.max_resource_chars,
        "min_visible_items": args.min_visible_items,
        "alias_policy": "most_explicit_visible_alias",
        "train": train_summary,
        "dev": dev_summary,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_instructions = resolve_prompt_instructions(args)
    instruction = clean_multiline_text(prompt_instructions.get("list") or QUESTION_INSTRUCTIONS["list"])

    train_input_paths = [resolve_project_path(path_value) for path_value in args.train_input]
    dev_input_paths = [resolve_project_path(path_value) for path_value in (args.dev_input or [])]

    train_rows, train_summary = build_split(
        train_input_paths,
        instruction=instruction,
        min_visible_items=args.min_visible_items,
        max_documents=args.max_documents,
        max_resource_chars=args.max_resource_chars,
    )
    save_prepared_records(train_rows, output_dir / "train_set_answGEN.json")

    dev_summary: Optional[dict[str, Any]] = None
    if dev_input_paths:
        dev_rows, dev_summary = build_split(
            dev_input_paths,
            instruction=instruction,
            min_visible_items=args.min_visible_items,
            max_documents=args.max_documents,
            max_resource_chars=args.max_resource_chars,
        )
        save_prepared_records(dev_rows, output_dir / "dev_set_answGEN.json")
        print(f"Saved {len(dev_rows):,} dev rows to {output_dir / 'dev_set_answGEN.json'}")

    write_summary(
        output_dir=output_dir,
        train_summary=train_summary,
        dev_summary=dev_summary,
        args=args,
    )

    print(f"Saved {len(train_rows):,} train rows to {output_dir / 'train_set_answGEN.json'}")
    print(f"Saved conversion summary to {output_dir / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
