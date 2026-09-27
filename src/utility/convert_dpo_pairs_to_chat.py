from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.data import clean_multiline_text, clean_text


QUESTION_MARKER = "# Question:"
RESOURCES_MARKER = "# PubMed resources:"
ANSWER_MARKER = "# Answer:"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def default_today_date() -> str:
    return date.today().strftime("%d %b %Y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DPO pair prompts from the repo's plain UNITOR-like format "
            "into a llama-3 chat-style prompt for chat-trained SFT adapters."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Project-relative or absolute input JSONL containing prompt/chosen/rejected pairs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Project-relative or absolute output JSONL for the converted pair file.",
    )
    parser.add_argument(
        "--cutting-knowledge-date",
        default="December 2023",
        help="Header value inserted into the chat-style system prompt.",
    )
    parser.add_argument(
        "--today-date",
        default=default_today_date(),
        help="Header value inserted into the chat-style system prompt.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default="Answer:",
        help="Assistant stub appended to the converted prompt.",
    )
    return parser.parse_args()


def split_resource_block(resource_text: str) -> List[str]:
    normalized = resource_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    lines = [line.rstrip() for line in normalized.split("\n") if line.strip()]
    resources: List[List[str]] = []
    current: List[str] = []

    def starts_new_resource(line: str) -> bool:
        return line.startswith("PubMed ID:") or line.startswith("Document:")

    for line in lines:
        if starts_new_resource(line) and current:
            resources.append(current)
            current = [line]
            continue
        if starts_new_resource(line):
            current = [line]
            continue
        current.append(line)

    if current:
        resources.append(current)

    joined_resources = [clean_multiline_text("\n".join(resource_lines)) for resource_lines in resources]
    joined_resources = [resource for resource in joined_resources if resource]
    if joined_resources:
        return joined_resources

    # Some historical pair files flatten all evidence into one plain text span
    # without document headers. Keep that as a single resource block.
    fallback = clean_multiline_text(normalized)
    return [fallback] if fallback else []


def parse_plain_prompt(prompt_text: str) -> Dict[str, Any]:
    normalized = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    if QUESTION_MARKER not in normalized or ANSWER_MARKER not in normalized:
        raise ValueError("Prompt does not look like the expected plain DPO format.")

    instruction_text, remainder = normalized.split(QUESTION_MARKER, 1)
    body_block, _answer_stub = remainder.rsplit(ANSWER_MARKER, 1)

    if RESOURCES_MARKER in body_block:
        question_text, resource_block = body_block.split(RESOURCES_MARKER, 1)
    else:
        question_text, resource_block = body_block, ""

    instruction = clean_multiline_text(instruction_text)
    question = clean_text(question_text)
    resources = split_resource_block(resource_block)
    return {
        "instruction": instruction,
        "question": question,
        "resources": resources,
    }


def build_chat_prompt(
    *,
    instruction: str,
    question: str,
    resources: Sequence[str],
    cutting_knowledge_date: str,
    today_date: str,
    assistant_prefix: str,
) -> str:
    system_parts = [
        f"Cutting Knowledge Date: {clean_text(cutting_knowledge_date)}",
        f"Today Date: {clean_text(today_date)}",
        clean_multiline_text(instruction),
    ]
    system_content = " ".join(part for part in system_parts if part)

    user_parts = [f"Question: {clean_text(question)}"]
    if resources:
        user_parts.append("PubMed resources:")
        for index, resource in enumerate(resources, start=1):
            user_parts.append(f"Resource {index}: {clean_multiline_text(resource)}")
    user_content = " ".join(part for part in user_parts if part)

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{clean_text(assistant_prefix)}"
    )
    return clean_text(prompt)


def convert_row(
    row: Dict[str, Any],
    *,
    cutting_knowledge_date: str,
    today_date: str,
    assistant_prefix: str,
) -> Dict[str, Any]:
    prompt_text = str(row.get("prompt") or "")
    if "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" in prompt_text:
        converted_prompt = clean_text(prompt_text)
    else:
        parsed = parse_plain_prompt(prompt_text)
        converted_prompt = build_chat_prompt(
            instruction=parsed["instruction"],
            question=parsed["question"],
            resources=parsed["resources"],
            cutting_knowledge_date=cutting_knowledge_date,
            today_date=today_date,
            assistant_prefix=assistant_prefix,
        )

    converted = dict(row)
    converted["prompt"] = converted_prompt
    converted["prompt_format"] = "chat"
    converted["chat_template"] = "llama-3"
    return converted


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)

    rows: List[Dict[str, Any]] = []
    converted_count = 0
    passthrough_chat_count = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            original_prompt = str(row.get("prompt") or "")
            converted_row = convert_row(
                row,
                cutting_knowledge_date=args.cutting_knowledge_date,
                today_date=args.today_date,
                assistant_prefix=args.assistant_prefix,
            )
            if converted_row["prompt"] != clean_text(original_prompt):
                converted_count += 1
            else:
                passthrough_chat_count += 1
            rows.append(converted_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read {len(rows):,} pair rows from {input_path}")
    print(f"Converted {converted_count:,} prompts to chat format")
    print(f"Passed through {passthrough_chat_count:,} prompts already in chat format")
    print(f"Wrote converted pair file to {output_path}")


if __name__ == "__main__":
    main()
