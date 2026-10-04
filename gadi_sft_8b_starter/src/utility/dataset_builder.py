from __future__ import annotations

from typing import Any, Dict, List, Sequence

from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt

from .data import build_unitor_prompt_text, clean_multiline_text, clean_text, list_record_resources


def build_sharegpt_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    resources = list_record_resources(example)

    user_parts = [f"Question: {clean_text(example['input_1'])}"]
    if resources:
        user_parts.append("PubMed resources:")
        for idx, resource in enumerate(resources, start=1):
            user_parts.append(f"Resource {idx}:\n{resource}")

    conversation = []
    instruction = clean_text(example.get("instruction", ""))
    if instruction:
        conversation.append({"role": "system", "content": instruction})

    conversation.append({"role": "user", "content": "\n\n".join(user_parts)})
    conversation.append({"role": "assistant", "content": f"Answer: {clean_text(example['output'])}"})
    return {"conversations": conversation}


def formatting_prompts_func(
    examples: Dict[str, Any],
    tokenizer: Any,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, List[str]]:
    chat_template_kwargs = chat_template_kwargs or {}
    texts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        for conversation in examples["conversations"]
    ]
    return {"text": texts}


def build_unitor_text_record(example: Dict[str, Any], eos_token: str) -> Dict[str, str]:
    return {
        "text": build_unitor_prompt_text(
            instruction=clean_multiline_text(example.get("instruction", "")),
            question=clean_text(example.get("input_1", "")),
            resources=list_record_resources(example),
            answer=clean_multiline_text(example.get("output", "")),
            eos_token=eos_token,
        )
    }


def tokenize_texts(examples: Dict[str, Any], tokenizer: Any, max_seq_length: int) -> Dict[str, Any]:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )


def _find_last_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1

    last_index = -1
    window = len(pattern)
    for index in range(len(sequence) - window + 1):
        if list(sequence[index : index + window]) == list(pattern):
            last_index = index
    return last_index


def build_completion_masks(
    examples: Dict[str, Any],
    response_template_ids: Sequence[int],
    fallback_response_template_ids: Sequence[int] | None = None,
) -> Dict[str, List[List[int]]]:
    masks: List[List[int]] = []
    fallback_response_template_ids = fallback_response_template_ids or ()

    for input_ids in examples["input_ids"]:
        match_start = _find_last_subsequence(input_ids, response_template_ids)
        match_length = len(response_template_ids)

        if match_start < 0 and fallback_response_template_ids:
            match_start = _find_last_subsequence(input_ids, fallback_response_template_ids)
            match_length = len(fallback_response_template_ids)

        if match_start < 0:
            # Keep the example trainable rather than crashing the full run.
            # This should be rare and usually indicates truncation or marker drift.
            masks.append([1] * len(input_ids))
            continue

        completion_start = min(len(input_ids), match_start + match_length)
        masks.append(([0] * completion_start) + ([1] * (len(input_ids) - completion_start)))

    return {"completion_mask": masks}


def prepare_dataset(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    num_proc: int,
    max_seq_length: int,
    prompt_format: str,
    response_template: str | None = None,
    response_template_trim_tokens: int = 0,
    chat_template_kwargs: Dict[str, Any] | None = None,
) -> Dataset:
    num_proc = max(1, int(num_proc))
    is_message_dataset = bool(rows) and all(
        isinstance(row.get("messages"), list) for row in rows
    )
    if is_message_dataset:
        dataset = Dataset.from_list(
            [{"conversations": row["messages"]} for row in rows]
        )
    else:
        dataset = Dataset.from_list(rows)

    if clean_text(prompt_format).lower() == "unitor_plain":
        eos_token = clean_text(getattr(tokenizer, "eos_token", ""))
        full_response_template_ids = tokenizer.encode(str(response_template or ""), add_special_tokens=False)
        trimmed_response_template_ids = full_response_template_ids[max(0, int(response_template_trim_tokens or 0)) :]
        dataset = dataset.map(
            build_unitor_text_record,
            num_proc=num_proc,
            fn_kwargs={"eos_token": eos_token},
        )
        dataset = dataset.map(
            tokenize_texts,
            batched=True,
            num_proc=1,
            fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
        )
        dataset = dataset.map(
            build_completion_masks,
            batched=True,
            num_proc=1,
            fn_kwargs={
                "response_template_ids": trimmed_response_template_ids,
                "fallback_response_template_ids": full_response_template_ids,
            },
        )
        return dataset

    if not is_message_dataset:
        dataset = dataset.map(build_sharegpt_conversation, num_proc=num_proc)
    dataset = standardize_sharegpt(dataset)

    # Tokenizers / processors from transformers + unsloth are often not picklable
    # across dataset worker processes. Keep the tokenizer-dependent formatting step
    # single-process for compatibility across library versions.
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=1,
        fn_kwargs={
            "tokenizer": tokenizer,
            "chat_template_kwargs": chat_template_kwargs or {},
        },
    )

    # Pre-tokenize here so TRL sees an already processed dataset with input_ids
    # and skips its own tokenizer.map(...) stage, which is the source of the
    # dill pickling failure in some environments.
    dataset = dataset.map(
        tokenize_texts,
        batched=True,
        num_proc=1,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": max_seq_length},
    )
    return dataset
