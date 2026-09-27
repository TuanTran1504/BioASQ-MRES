from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.bioasq_format import exact_answer_groups, normalize_for_match
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import build_resources, clean_multiline_text, clean_text, resolve_prompt_instructions, save_prepared_records
from src.utility.eval_types import EvalExample


DEFAULT_TRAIN_INPUT = ["data/training13b.json"]
DEFAULT_OUTPUT_DIR = "data/BioASQ_list_mr5_visible_gold_sft"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prepared list-only SFT dataset where each target answer keeps "
            "only BioASQ gold groups that are visible in the current mr5 evidence "
            "(first 5 document-level resources with the current max_resource_chars)."
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
        "--max-resources",
        type=int,
        default=5,
        help="Maximum number of document-level resources to keep per question.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Maximum characters per serialized resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--resource-granularity",
        default="document",
        choices=["document", "snippet"],
        help="Resource grouping used when rebuilding the visible evidence.",
    )
    parser.add_argument(
        "--min-visible-items",
        type=int,
        default=1,
        help="Drop questions with fewer than this many visible gold groups.",
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


def visible_aliases_in_resources(gold_group: Sequence[str], resources: Sequence[str]) -> List[str]:
    evidence_text = normalize_for_match("\n".join(resources))
    visible: List[str] = []
    seen: set[str] = set()
    for alias in gold_group:
        cleaned = clean_text(alias)
        alias_norm = normalize_for_match(cleaned)
        if not cleaned or not alias_norm or alias_norm in seen:
            continue
        if alias_norm in evidence_text:
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
    for gold_group in gold_groups:
        visible_aliases = visible_aliases_in_resources(gold_group, resources)
        chosen = choose_visible_alias(visible_aliases)
        if chosen:
            items.append(chosen)
    return items


def build_record(
    question: Dict[str, Any],
    *,
    instruction: str,
    max_resources: int,
    max_resource_chars: int,
    resource_granularity: str,
) -> Optional[Dict[str, Any]]:
    question_id = clean_text(question.get("id", ""))
    question_text = clean_text(question.get("body", ""))
    if not question_text:
        return None

    resources = build_resources(
        question,
        max_resources=max_resources,
        max_resource_chars=max_resource_chars,
        question_text=question_text,
        resource_granularity=resource_granularity,
        resource_selection="first",
    )
    gold_groups = gold_groups_for_question(question)
    visible_items = build_output_items(resources, gold_groups)

    if not visible_items:
        return {
            "id": question_id,
            "type": "list",
            "instruction": instruction,
            "input_1": question_text,
            "output": "",
            "visible_gold_count": 0,
            "original_gold_count": len(gold_groups),
            "kept_gold_ratio": 0.0,
            "source_question_id": question_id,
        }

    record: Dict[str, Any] = {
        "id": question_id,
        "type": "list",
        "instruction": instruction,
        "input_1": question_text,
        "output": " ".join(f"[BI]{item}[EI]" for item in visible_items),
        "visible_gold_count": len(visible_items),
        "original_gold_count": len(gold_groups),
        "kept_gold_ratio": len(visible_items) / len(gold_groups) if gold_groups else 0.0,
        "source_question_id": question_id,
    }
    for index, resource in enumerate(resources, start=2):
        if clean_multiline_text(resource):
            record[f"input_{index}"] = clean_multiline_text(resource)
    return record


def build_split(
    input_paths: Sequence[Path],
    *,
    instruction: str,
    min_visible_items: int,
    max_resources: int,
    max_resource_chars: int,
    resource_granularity: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    visible_counts: List[int] = []
    original_counts: List[int] = []
    dropped_zero_visible = 0
    total_questions = 0
    seen_ids: set[str] = set()

    for source_path, question in iter_list_questions(input_paths):
        total_questions += 1
        record = build_record(
            question,
            instruction=instruction,
            max_resources=max_resources,
            max_resource_chars=max_resource_chars,
            resource_granularity=resource_granularity,
        )
        if record is None:
            continue

        question_id = str(record.get("id") or f"{source_path.name}:{total_questions}")
        if question_id in seen_ids:
            continue
        seen_ids.add(question_id)

        visible_gold_count = int(record.get("visible_gold_count", 0) or 0)
        original_gold_count = int(record.get("original_gold_count", 0) or 0)
        if visible_gold_count < min_visible_items:
            dropped_zero_visible += 1
            continue

        rows.append(record)
        visible_counts.append(visible_gold_count)
        original_counts.append(original_gold_count)

    summary = {
        "input_files": [str(path) for path in input_paths],
        "total_list_questions": total_questions,
        "kept_questions": len(rows),
        "dropped_below_min_visible_items": dropped_zero_visible,
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
        "dataset_name": "mr5_visible_gold_sft",
        "resource_selection": "first",
        "resource_granularity": args.resource_granularity,
        "max_resources": args.max_resources,
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
        max_resources=args.max_resources,
        max_resource_chars=args.max_resource_chars,
        resource_granularity=args.resource_granularity,
    )
    save_prepared_records(train_rows, output_dir / "train_set_answGEN.json")

    dev_summary: Optional[dict[str, Any]] = None
    if dev_input_paths:
        dev_rows, dev_summary = build_split(
            dev_input_paths,
            instruction=instruction,
            min_visible_items=args.min_visible_items,
            max_resources=args.max_resources,
            max_resource_chars=args.max_resource_chars,
            resource_granularity=args.resource_granularity,
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
