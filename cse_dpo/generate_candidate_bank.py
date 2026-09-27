from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import clean_text
from src.utility.eval_dataset import load_eval_examples, render_prompt, resolve_eval_input_paths
from src.utility.eval_models import (
    first_model_device,
    load_model_and_tokenizer_for_eval,
    resolve_model_specs,
    tokenize_prompts_for_generation,
)
from src.utility.factoid_output_parsing import parse_factoid_candidates
from src.model_registry import get_project_root, slugify, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle

from .common import write_json, write_jsonl
from .normalize_set_answers import parse_list_output
from .schemas import CandidateBankRecord, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated answer-generation candidate banks for DPO pair construction."
        )
    )
    parser.add_argument(
        "--eval-input",
        nargs="+",
        required=True,
        help="Raw BioASQ JSON or prepared JSON files containing training questions.",
    )
    parser.add_argument(
        "--model-ref",
        nargs="+",
        default=None,
        help="Model registry alias/run id/path to sample from.",
    )
    parser.add_argument(
        "--all-registry-runs",
        action="store_true",
        help="Generate candidate banks for all completed answer_generation runs in the registry.",
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry path.",
    )
    parser.add_argument(
        "--prompt-registry-path",
        default=None,
        help="Optional repo-relative prompt registry JSON path.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional repo-relative JSON file describing one prompt bundle.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt alias or prompt id to resolve from the prompt registry.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where candidate-bank artifacts will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label stored in emitted rows.",
    )
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=16,
        help=(
            "Compatibility option used with --independent-banks when "
            "--samples-per-question-total is omitted."
        ),
    )
    parser.add_argument(
        "--samples-per-question-total",
        type=int,
        default=None,
        help=(
            "Total number of sampled outputs to generate per question. "
            "Preferred over the legacy banked sampling layout."
        ),
    )
    parser.add_argument(
        "--independent-banks",
        type=int,
        default=3,
        help=(
            "Compatibility option describing how many legacy banks to emulate "
            "when --samples-per-question-total is omitted."
        ),
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["list"],
        choices=["summary", "factoid", "list", "yesno"],
        help="Question types to load. Candidate-bank generation currently supports list and factoid.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=3,
        help="Maximum number of PubMed resources per question. Use 0 for all resources.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Maximum characters per serialized PubMed resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--resource-selection",
        default="first",
        choices=["first", "embedding"],
        help=(
            "How to choose which resources to include. 'first' keeps dataset order; "
            "'embedding' reranks by question-resource similarity before applying "
            "--max-resources."
        ),
    )
    parser.add_argument(
        "--resource-granularity",
        default="document",
        choices=["document", "snippet"],
        help=(
            "Whether candidate-generation resources are grouped by document or "
            "treated as individual snippets before selection."
        ),
    )
    parser.add_argument(
        "--resource-reranker-model",
        default="sentence-transformers/all-MiniLM-L12-v2",
        help=(
            "Embedding query model or single-encoder model used when "
            "--resource-selection embedding is enabled."
        ),
    )
    parser.add_argument(
        "--resource-reranker-article-model",
        default=None,
        help=(
            "Optional separate embedding model used for resources/documents when "
            "--resource-selection embedding is enabled. Use this for dual-encoder "
            "retrievers such as MedCPT."
        ),
    )
    parser.add_argument(
        "--resource-reranker-device",
        default="auto",
        help=(
            "Device used for the resource reranker, for example 'cpu', 'cuda', or "
            "'auto'."
        ),
    )
    parser.add_argument(
        "--resource-reranker-batch-size",
        type=int,
        default=32,
        help="Batch size used when encoding candidate resources for reranking.",
    )
    parser.add_argument(
        "--max-summary-answers",
        type=int,
        default=1,
        help="Compatibility option passed through to raw BioASQ loading.",
    )
    parser.add_argument(
        "--max-factoid-answers",
        type=int,
        default=5,
        help="Compatibility option passed through to raw BioASQ loading.",
    )
    parser.add_argument(
        "--max-list-items",
        type=int,
        default=100,
        help="Maximum number of list gold items when converting raw BioASQ questions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of loaded questions for smoke tests.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Model max sequence length.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Optional dtype override for model loading.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading for supported loaders.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to Transformers/PEFT fallback loaders.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Optional chat template override used when the model does not provide one.",
    )
    parser.add_argument(
        "--prompt-format",
        default=None,
        choices=["chat", "unitor_plain"],
        help="Optional prompt format override used when the model does not provide one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Base seed used to make sampling reproducible.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many questions to generate per forward pass. Increase to reduce runtime if VRAM allows.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force model/tokenizer loading from local files and Hugging Face cache only.",
    )
    parser.add_argument(
        "--factoid-parser-mode",
        choices=["current", "agnostic"],
        default="current",
        help=(
            "How to parse factoid outputs into candidate items for metadata. "
            "'current' mirrors the tagged-first parser; 'agnostic' also accepts numbered lists."
        ),
    )
    return parser.parse_args()


def _generate_outputs_for_prompts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    args: argparse.Namespace,
    generation_seed: int | None,
) -> list[tuple[str, int, dict[str, Any]]]:
    import torch

    max_seq_length = int(getattr(args, "max_seq_length", 0) or 0)
    encoded, prompt_telemetry = tokenize_prompts_for_generation(
        tokenizer,
        list(prompts),
        max_seq_length=max_seq_length,
        encode_kwargs={
            "return_tensors": "pt",
            "padding": True,
        },
    )
    device = first_model_device(model)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    if generation_seed is not None:
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    outputs: list[tuple[str, int, dict[str, Any]]] = []
    input_width = int(encoded["input_ids"].shape[-1])
    for row_index in range(len(prompts)):
        # In padded batched generation, decoder-only models append new tokens after the
        # full padded input width, not after each row's non-pad token count.
        generated_ids = output_ids[row_index][input_width:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = clean_text(decoded)
        if cleaned.lower().startswith("answer:"):
            cleaned = cleaned[7:].strip()
        generated_token_count = int(generated_ids.shape[-1])
        sample_telemetry = (
            prompt_telemetry[row_index]
            if row_index < len(prompt_telemetry)
            else {
                "prompt_token_count": input_width,
                "effective_prompt_token_count": input_width,
                "prompt_truncated": False,
                "truncated_token_count": 0,
                "max_seq_length": int(max_seq_length) if max_seq_length > 0 else None,
            }
        )
        outputs.append((cleaned, generated_token_count, sample_telemetry))

    return outputs


def summarize_prompt_truncation(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized_entries = [entry for entry in entries if isinstance(entry, Mapping)]
    truncated_generation_count = sum(1 for entry in normalized_entries if bool(entry.get("prompt_truncated")))
    return {
        "prompt_truncated": bool(truncated_generation_count > 0),
        "generation_count": len(normalized_entries),
        "truncated_generation_count": truncated_generation_count,
        "max_prompt_token_count": max(
            (int(entry.get("prompt_token_count") or 0) for entry in normalized_entries),
            default=0,
        ),
        "max_effective_prompt_token_count": max(
            (int(entry.get("effective_prompt_token_count") or 0) for entry in normalized_entries),
            default=0,
        ),
        "total_truncated_token_count": sum(
            int(entry.get("truncated_token_count") or 0) for entry in normalized_entries
        ),
        "max_truncated_token_count": max(
            (int(entry.get("truncated_token_count") or 0) for entry in normalized_entries),
            default=0,
        ),
    }


def build_cached_prompts(
    examples: Sequence[Any],
    tokenizer: Any,
    chat_template: str | None,
    prompt_format: str | None,
) -> dict[str, str]:
    return {
        example.question_id: render_prompt(
            tokenizer,
            example,
            chat_template=chat_template,
            prompt_format=prompt_format,
        )
        for example in examples
    }


def evaluate_paths(args: argparse.Namespace) -> list[Path]:
    project_root = get_project_root()
    return resolve_eval_input_paths(args, project_root=project_root)


def resolve_total_samples_per_question(args: argparse.Namespace) -> int:
    if args.samples_per_question_total is not None:
        if args.samples_per_question_total <= 0:
            raise ValueError("--samples-per-question-total must be positive.")
        if args.samples_per_question != 16 or args.independent_banks != 3:
            raise ValueError(
                "Use either --samples-per-question-total or the legacy "
                "--samples-per-question/--independent-banks combination, not both."
            )
        return args.samples_per_question_total

    if args.samples_per_question <= 0:
        raise ValueError("--samples-per-question must be positive.")
    if args.independent_banks <= 0:
        raise ValueError("--independent-banks must be positive.")
    return args.samples_per_question * args.independent_banks


def supported_question_types(args: argparse.Namespace) -> list[str]:
    requested = [
        clean_text(question_type).lower()
        for question_type in getattr(args, "question_types", []) or []
        if clean_text(question_type)
    ]
    if not requested:
        raise ValueError("At least one supported question type must be requested.")

    unsupported = sorted(set(requested) - {"list", "factoid"})
    if unsupported:
        raise ValueError(
            "Candidate-bank generation currently supports only list and factoid questions. "
            f"Unsupported: {unsupported}"
        )
    return requested


def parse_candidate_output(
    raw_output: str,
    *,
    question_type: str,
    factoid_parser_mode: str,
) -> dict[str, Any]:
    normalized_type = clean_text(question_type).lower()
    if normalized_type == "list":
        parsed = parse_list_output(raw_output, allow_fallback_split=True)
        return {
            "parsed_items": list(parsed.items),
            "parser_status": parsed.status,
            "parser_warnings": list(parsed.warnings),
            "dropped_placeholder_count": parsed.dropped_placeholder_count,
            "used_fallback_split": parsed.used_fallback_split,
        }

    if normalized_type == "factoid":
        parsed_items = parse_factoid_candidates(
            raw_output,
            parser_mode=factoid_parser_mode,
        )
        cleaned_output = clean_text(raw_output)
        parser_status = "ok" if parsed_items else ("empty" if not cleaned_output else "malformed")
        parser_warnings: list[str] = []
        if parser_status == "malformed":
            parser_warnings.append("no_factoid_candidates_found")
        return {
            "parsed_items": list(parsed_items),
            "parser_status": parser_status,
            "parser_warnings": parser_warnings,
            "dropped_placeholder_count": 0,
            "used_fallback_split": False,
        }

    raise ValueError(f"Unsupported question type for candidate parsing: {question_type}")


def write_candidate_bank_manifest(
    *,
    path: Path,
    model_summary: Mapping[str, Any],
    prompt_bundle: Mapping[str, Any],
    eval_paths: Sequence[Path],
    question_count: int,
    candidate_count: int,
    total_samples_per_question: int,
    prompt_truncation_summary: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    generation_summary: dict[str, Any] = {
        "samples_per_question_total": total_samples_per_question,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_truncation": dict(prompt_truncation_summary),
    }
    if args.samples_per_question_total is None:
        generation_summary["legacy_sampling_layout"] = {
            "samples_per_question": args.samples_per_question,
            "independent_banks": args.independent_banks,
        }

    write_json(
        path,
        {
            "created_at": utc_now_iso(),
            "dataset": {
                "name": args.dataset_name,
                "eval_input": [str(item) for item in eval_paths],
                "question_count": question_count,
                "question_types": sorted(
                    {
                        clean_text(question_type).lower()
                        for question_type in (getattr(args, "question_types", []) or [])
                        if clean_text(question_type)
                    }
                ),
                "max_resources": args.max_resources,
                "max_resource_chars": args.max_resource_chars,
                "resource_selection": args.resource_selection,
                "resource_granularity": args.resource_granularity,
                "resource_reranker_model": (
                    args.resource_reranker_model if args.resource_selection == "embedding" else None
                ),
                "resource_reranker_article_model": (
                    args.resource_reranker_article_model if args.resource_selection == "embedding" else None
                ),
                "resource_reranker_device": (
                    args.resource_reranker_device if args.resource_selection == "embedding" else None
                ),
            },
            "model": dict(model_summary),
            "prompt": {
                "prompt_id": prompt_bundle.get("prompt_id"),
                "source": prompt_bundle.get("source"),
                "registry_path": prompt_bundle.get("registry_path"),
            },
            "generation": generation_summary,
            "parser": {
                "factoid_parser_mode": (
                    str(args.factoid_parser_mode)
                    if "factoid" in {
                        clean_text(question_type).lower()
                        for question_type in (getattr(args, "question_types", []) or [])
                        if clean_text(question_type)
                    }
                    else None
                ),
            },
            "candidate_count": candidate_count,
        },
    )


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    total_samples_per_question = resolve_total_samples_per_question(args)
    requested_question_types = supported_question_types(args)
    prompt_path_value = args.prompt_file or args.prompt_registry_path
    prompt_registry_path = (
        None
        if prompt_path_value in {None, ""}
        else (project_root / prompt_path_value).resolve() if not Path(prompt_path_value).is_absolute() else Path(prompt_path_value)
    )
    prompt_bundle = resolve_prompt_bundle(
        registry_path=prompt_registry_path,
        prompt_ref=args.prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    eval_paths = evaluate_paths(args)
    examples = load_eval_examples(
        eval_paths,
        args,
        prompt_instructions=prompt_bundle["instructions"],
    )
    examples = [
        example
        for example in examples
        if clean_text(example.question_type).lower() in requested_question_types
    ]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(
            "No supported examples were loaded from --eval-input for "
            f"question types {requested_question_types}."
        )

    model_specs = resolve_model_specs(args, project_root=project_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_spec in model_specs:
        model, tokenizer = load_model_and_tokenizer_for_eval(model_spec, args)
        active_chat_template = model_spec.chat_template or clean_text(args.chat_template) or clean_text(prompt_bundle.get("chat_template", ""))
        active_prompt_format = (
            model_spec.prompt_format
            or clean_text(args.prompt_format)
            or clean_text(prompt_bundle.get("prompt_format", ""))
            or "chat"
        )
        prompt_by_question_id = build_cached_prompts(
            examples=examples,
            tokenizer=tokenizer,
            chat_template=active_chat_template or None,
            prompt_format=active_prompt_format or None,
        )

        model_dir = output_root / slugify(model_spec.label, fallback="model")
        model_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        prompt_truncation_entries: list[dict[str, Any]] = []

        for sample_id in range(total_samples_per_question):
            for batch_start in range(0, len(examples), args.batch_size):
                batch_examples = examples[batch_start : batch_start + args.batch_size]
                generation_seed = args.seed + (sample_id * 10_000) + batch_start
                batch_prompts = [prompt_by_question_id[example.question_id] for example in batch_examples]
                batch_outputs = _generate_outputs_for_prompts(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    args=args,
                    generation_seed=generation_seed,
                )
                for example, prompt, (raw_output, generated_token_count, generation_telemetry) in zip(batch_examples, batch_prompts, batch_outputs):
                    record = CandidateBankRecord(
                        dataset=args.dataset_name,
                        question_id=example.question_id,
                        sample_id=sample_id,
                        prompt=prompt,
                        question_text=example.body,
                        evidence=tuple(resource for resource in example.resources if clean_text(resource)),
                        raw_output=raw_output,
                        generated_token_count=generated_token_count,
                        generator_checkpoint=model_spec.ref,
                        response_id=f"{example.question_id}-sample{sample_id}",
                        source_path=example.source_path,
                        prompt_instruction=example.instruction,
                    )
                    row = to_jsonable(record)
                    row["question_type"] = example.question_type
                    row.update(
                        parse_candidate_output(
                            raw_output,
                            question_type=example.question_type,
                            factoid_parser_mode=str(args.factoid_parser_mode),
                        )
                    )
                    row["generation_telemetry"] = generation_telemetry
                    row["prompt_truncation"] = summarize_prompt_truncation([generation_telemetry])
                    rows.append(row)
                    prompt_truncation_entries.append(generation_telemetry)

        write_jsonl(model_dir / "candidate_bank.jsonl", rows)
        write_candidate_bank_manifest(
            path=model_dir / "manifest.json",
            model_summary={
                "label": model_spec.label,
                "ref": model_spec.ref,
                "source": model_spec.source,
                "load_target": model_spec.load_target,
                "chat_template": active_chat_template or None,
                "prompt_format": active_prompt_format or None,
            },
            prompt_bundle=prompt_bundle,
            eval_paths=eval_paths,
            question_count=len(examples),
            candidate_count=len(rows),
            total_samples_per_question=total_samples_per_question,
            prompt_truncation_summary=summarize_prompt_truncation(prompt_truncation_entries),
            args=args,
        )


if __name__ == "__main__":
    main()
