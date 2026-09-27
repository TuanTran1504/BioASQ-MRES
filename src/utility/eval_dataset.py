from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.model_registry import resolve_repo_path

from .config import QUESTION_INSTRUCTIONS
from .data import (
    EmbeddingResourceSelector,
    build_output,
    build_resources,
    build_unitor_prompt_text,
    clean_multiline_text,
    clean_text,
    list_record_resources,
    load_prepared_records,
    read_json,
    select_resource_texts,
)
from .eval_types import EvalExample


def resolve_eval_input_paths(args: argparse.Namespace, project_root: Path) -> List[Path]:
    if args.eval_input:
        return [
            resolve_repo_path(path_value, project_root=project_root) or Path(path_value)
            for path_value in args.eval_input
        ]

    default_dir = project_root / "data" / "Task13BTest"
    paths = sorted(default_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            "No evaluation files found. Pass --eval-input explicitly or populate data/Task13BTest."
        )
    return paths


def normalize_question_types(question_types: Iterable[str]) -> List[str]:
    return [clean_text(question_type).lower() for question_type in question_types if clean_text(question_type)]


def load_eval_examples(
    paths: Sequence[Path],
    args: argparse.Namespace,
    prompt_instructions: Mapping[str, str],
) -> List[EvalExample]:
    allowed_types = set(normalize_question_types(args.question_types))
    resource_selection = clean_text(getattr(args, "resource_selection", "first")).lower() or "first"
    resource_granularity = clean_text(getattr(args, "resource_granularity", "document")).lower() or "document"
    resource_window_mode = clean_text(getattr(args, "resource_window_mode", "single")).lower() or "single"
    resource_limit = 0 if resource_window_mode == "sequential" else args.max_resources
    resource_selector = (
        EmbeddingResourceSelector(
            clean_text(getattr(args, "resource_reranker_model", "")) or "sentence-transformers/all-MiniLM-L12-v2",
            article_model_name=clean_text(getattr(args, "resource_reranker_article_model", "")) or None,
            device=clean_text(getattr(args, "resource_reranker_device", "")) or "auto",
            batch_size=int(getattr(args, "resource_reranker_batch_size", 32) or 32),
            local_files_only=bool(getattr(args, "local_files_only", False)),
        )
        if resource_selection == "embedding"
        else None
    )
    examples: List[EvalExample] = []
    seen_ids = set()

    for path in paths:
        payload = read_json(path)

        if isinstance(payload, list):
            rows = load_prepared_records(path)
            for index, row in enumerate(rows):
                question_type = clean_text(row.get("type", "")).lower()
                if question_type not in allowed_types:
                    continue

                instruction = clean_multiline_text(
                    prompt_instructions.get(question_type)
                    or row.get("instruction")
                    or QUESTION_INSTRUCTIONS.get(question_type)
                )
                if not instruction:
                    continue

                question_id = clean_text(row.get("id", "")) or f"{path.name}:{index}"
                dedupe_key = (question_id, question_type)
                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)
                resources = select_resource_texts(
                    list_record_resources(row),
                    question_text=clean_text(row.get("input_1", "")),
                    max_resources=resource_limit,
                    resource_selection=resource_selection,
                    resource_selector=resource_selector,
                )

                examples.append(
                    EvalExample(
                        question_id=question_id,
                        question_type=question_type,
                        body=clean_text(row.get("input_1", "")),
                        instruction=instruction,
                        resources=tuple(resource for resource in resources if clean_text(resource)),
                        gold_output=clean_text(row.get("output", "")),
                        source_path=str(path),
                        raw_question=None,
                    )
                )
            continue

        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"Unsupported evaluation input format: {path}")

        for question in questions:
            if not isinstance(question, dict):
                continue

            question_type = clean_text(question.get("type", "")).lower()
            if question_type not in allowed_types:
                continue

            instruction = clean_multiline_text(prompt_instructions.get(question_type) or QUESTION_INSTRUCTIONS.get(question_type))
            body = clean_text(question.get("body", ""))
            if not instruction or not body:
                continue

            gold_output = clean_text(build_output(question, args))
            if not gold_output:
                continue

            resources = build_resources(
                question,
                max_resources=resource_limit,
                max_resource_chars=args.max_resource_chars,
                question_text=body,
                resource_granularity=resource_granularity,
                resource_selection=resource_selection,
                resource_selector=resource_selector,
            )
            question_id = clean_text(question.get("id", "")) or f"{path.name}:{len(examples)}"
            dedupe_key = (question_id, question_type)
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)

            examples.append(
                EvalExample(
                    question_id=question_id,
                    question_type=question_type,
                    body=body,
                    instruction=instruction,
                    resources=tuple(resource for resource in resources if clean_text(resource)),
                    gold_output=gold_output,
                    source_path=str(path),
                    raw_question=question,
                )
            )

    return examples if args.limit is None else examples[: args.limit]


def build_messages(example: EvalExample) -> List[Dict[str, str]]:
    user_parts = [f"Question: {example.body}"]
    resources = [resource for resource in example.resources if clean_text(resource)]
    if resources:
        user_parts.append("PubMed resources:")
        for index, resource in enumerate(resources, start=1):
            user_parts.append(f"Resource {index}:\n{resource}")

    messages = [{"role": "system", "content": example.instruction}]
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def render_manual_chat(messages: Sequence[Mapping[str, str]], chat_template: Optional[str]) -> str:
    normalized_template = clean_text(chat_template).lower()
    system_text = clean_text(next((message["content"] for message in messages if message.get("role") == "system"), ""))
    user_text = clean_text(next((message["content"] for message in messages if message.get("role") == "user"), ""))

    if normalized_template in {"llama-3", "llama3"}:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    if normalized_template in {"phi-4"}:
        return (
            f"<|im_start|>system<|im_sep|>{system_text}<|im_end|>"
            f"<|im_start|>user<|im_sep|>{user_text}<|im_end|>"
            "<|im_start|>assistant<|im_sep|>"
        )

    if normalized_template in {"phi-3", "phi-35", "phi-3.5"}:
        return f"<|system|>\n{system_text}<|end|>\n<|user|>\n{user_text}<|end|>\n<|assistant|>\n"

    return f"System:\n{system_text}\n\nUser:\n{user_text}\n\nAssistant:\n"


def render_prompt(
    tokenizer: Any,
    example: EvalExample,
    chat_template: Optional[str],
    prompt_format: Optional[str],
) -> str:
    if clean_text(prompt_format).lower() == "unitor_plain":
        return build_unitor_prompt_text(
            instruction=example.instruction,
            question=example.body,
            resources=example.resources,
        )

    messages = build_messages(example)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ) + "Answer:"
        except Exception:
            pass
    return render_manual_chat(messages, chat_template=chat_template) + "Answer:"
