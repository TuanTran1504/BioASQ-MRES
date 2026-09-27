from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from transformers import AutoTokenizer


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "cse_dpo").exists():
            return candidate
    raise RuntimeError("Could not locate project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utility.data import clean_text, list_record_resources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style Softmax-DPO factoid dataset with single-answer prompts."
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        required=True,
        help="Frozen question-level wrong-entity records.",
    )
    parser.add_argument(
        "--single-answer-source",
        type=Path,
        required=True,
        help="Prepared single-answer training data used to reconstruct the prompt.",
    )
    parser.add_argument(
        "--model-ref",
        type=Path,
        required=True,
        help="Local tokenizer/model directory used to render the chat prompt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for train/validation Softmax-DPO JSONL outputs.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for the random split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used for the train/validation split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row cap after shuffling. 0 keeps all rows.",
    )
    parser.add_argument(
        "--same-question-4-only",
        action="store_true",
        help="Keep only rows with four within-question wrong entities.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Write all rows to validation_softmax_dpo.jsonl and leave the train split empty.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Write all rows to train_softmax_dpo.jsonl and leave the validation split empty.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_key(value: Any) -> str:
    return normalize_text(value).casefold()


def factoid_completion(entity: str) -> str:
    return f"[BE]{normalize_text(entity)}[EE]"


def build_user_message(question: str, resources: List[str]) -> str:
    parts = [f"Question: {clean_text(question)}"]
    if resources:
        parts.append("PubMed resources:")
        for index, resource in enumerate(resources, start=1):
            parts.append(f"Resource {index}:\n{resource}")
    return "\n\n".join(parts)


def render_prompt(tokenizer: Any, instruction: str, question: str, resources: List[str]) -> str:
    messages = [
        {"role": "system", "content": clean_text(instruction)},
        {"role": "user", "content": build_user_message(question=question, resources=resources)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt + "Answer:"


def validate_and_attach_prompt(
    row: Dict[str, Any],
    index: int,
    single_answer_index: Dict[str, Dict[str, Any]],
    tokenizer: Any,
    source_jsonl: Path,
    single_answer_source: Path,
) -> Dict[str, Any]:
    question_id = str(row.get("question_id") or row.get("id") or "").strip()
    if not question_id:
        raise ValueError(f"Row {index} is missing question_id")

    source_example = single_answer_index.get(question_id)
    if source_example is None:
        raise KeyError(
            f"Question {question_id} from {source_jsonl} was not found in {single_answer_source}"
        )

    gold = normalize_text(row.get("canonical_gold_entity") or row.get("chosen_entity") or row.get("gold"))
    wrongs = [normalize_text(x) for x in row.get("wrong_entities", []) if normalize_text(x)]
    accepted = [
        normalize_text(x)
        for x in row.get("accepted_gold_entities", row.get("gold_aliases", []))
        if normalize_text(x)
    ]
    if not gold:
        raise ValueError(f"Row {index} is missing canonical gold")
    if len(wrongs) < 1:
        raise ValueError(f"Row {index} must contain at least one wrong entity")

    normalized = {normalize_key(x) for x in [gold, *wrongs]}
    if len(normalized) != (1 + len(wrongs)):
        raise ValueError(f"Row {index} must contain distinct gold and wrong candidates")

    prompt = render_prompt(
        tokenizer=tokenizer,
        instruction=str(source_example.get("instruction", "")),
        question=str(source_example.get("input_1", "")),
        resources=list_record_resources(source_example),
    )
    if "Return exactly one short biomedical expression." not in prompt:
        raise ValueError(f"Question {question_id} did not render with the single-answer instruction")

    return {
        "question_id": question_id,
        "prompt": prompt,
        "question_text": normalize_text(source_example.get("input_1", row.get("question_text", ""))),
        "gold": gold,
        "accepted_gold": accepted or [gold],
        "wrongs": wrongs,
        "selected_within_question_wrong_count": int(
            row.get("selected_within_question_wrong_count", row.get("within_question_wrong_count", 0)) or 0
        ),
        "source_path": str(source_jsonl),
        "single_answer_source_path": str(single_answer_source),
    }


def split_rows(rows: List[Dict[str, Any]], validation_fraction: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to split train/validation")
    ordered = list(rows)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = max(1, round(len(ordered) * validation_fraction))
    n_val = min(n_val, len(ordered) - 1)
    return ordered[n_val:], ordered[:n_val]


def build_softmax_row(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    return {
        "view_type": "softmax_dpo",
        "question_id": row["question_id"],
        "split": split,
        "prompt": row["prompt"],
        "question_text": row["question_text"],
        "chosen": factoid_completion(row["gold"]),
        "negatives": [factoid_completion(entity) for entity in row["wrongs"]],
        "gold": row["gold"],
        "accepted_gold": list(row["accepted_gold"]),
        "wrongs": list(row["wrongs"]),
        "negative_count": len(row["wrongs"]),
        "selected_within_question_wrong_count": row["selected_within_question_wrong_count"],
        "source_path": row["source_path"],
        "single_answer_source_path": row["single_answer_source_path"],
    }


def main() -> int:
    args = parse_args()
    if args.validation_only and args.train_only:
        raise ValueError("Choose at most one of --validation-only or --train-only")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_ref),
        local_files_only=True,
        trust_remote_code=True,
    )

    frozen_rows = load_jsonl(args.source_jsonl)
    single_answer_rows = load_json(args.single_answer_source)
    single_answer_index = {
        str(row.get("id", "")).strip(): row
        for row in single_answer_rows
        if str(row.get("id", "")).strip()
    }

    rows = [
        validate_and_attach_prompt(
            row=row,
            index=index,
            single_answer_index=single_answer_index,
            tokenizer=tokenizer,
            source_jsonl=args.source_jsonl,
            single_answer_source=args.single_answer_source,
        )
        for index, row in enumerate(frozen_rows)
    ]

    if args.same_question_4_only:
        rows = [row for row in rows if row["selected_within_question_wrong_count"] == 4]

    if args.limit and args.limit > 0:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.limit]

    if args.validation_only:
        train_rows = []
        validation_rows = list(rows)
    elif args.train_only:
        train_rows = list(rows)
        validation_rows = []
    else:
        train_rows, validation_rows = split_rows(
            rows=rows,
            validation_fraction=float(args.validation_fraction),
            seed=int(args.seed),
        )
    train_dataset = [build_softmax_row(row, "train") for row in train_rows]
    validation_dataset = [build_softmax_row(row, "validation") for row in validation_rows]

    output_dir = args.output_dir
    train_path = output_dir / "train_softmax_dpo.jsonl"
    validation_path = output_dir / "validation_softmax_dpo.jsonl"
    summary_path = output_dir / "construction_summary.json"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(train_path, train_dataset)
    write_jsonl(validation_path, validation_dataset)

    summary = {
        "source_jsonl": str(args.source_jsonl),
        "single_answer_source": str(args.single_answer_source),
        "tokenizer_model_ref": str(args.model_ref),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "validation_only": bool(args.validation_only),
        "train_only": bool(args.train_only),
        "use_only_same_question_4": bool(args.same_question_4_only),
        "row_count": len(rows),
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
        "avg_negative_count": mean(row["negative_count"] for row in train_dataset + validation_dataset),
        "same_question_4_count": sum(row["selected_within_question_wrong_count"] == 4 for row in rows),
        "prompt_contract": {
            "instruction_type": "single_answer",
            "rendered_with_model_tokenizer": str(args.model_ref),
            "expects_exactly_one_answer": True,
            "completion_format": "[BE] short answer [EE]",
        },
        "files": {
            "train_softmax_dpo_jsonl": str(train_path),
            "validation_softmax_dpo_jsonl": str(validation_path),
        },
    }
    manifest = {
        "dataset_view": "softmax_dpo",
        "description": "Paper-style one-positive/multi-negative factoid Softmax-DPO dataset with single-answer prompts.",
        "summary_path": str(summary_path),
        "files": [str(train_path), str(validation_path)],
    }

    write_json(summary_path, summary)
    write_json(manifest_path, manifest)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
