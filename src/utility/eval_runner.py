from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.model_registry import get_project_root, resolve_repo_path, slugify, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle

from .bioasq_format import build_bioasq_prediction_entry
from .bioasq_official import OFFICIAL_EXACT_TYPES, evaluate_with_bioasq_java
from .config import QUESTION_INSTRUCTIONS
from .data import clean_text
from .eval_dataset import load_eval_examples, resolve_eval_input_paths
from .eval_models import (
    aggregate_generated_answers,
    generate_answer_samples,
    load_model_and_tokenizer_for_eval,
    prime_unsloth_runtime,
    resolve_model_specs,
)
from .eval_types import EvalExample, ModelSpec


def maybe_run_grounded_semantic_evaluation(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    examples_by_key: Mapping[tuple[str, str], EvalExample],
    args: argparse.Namespace,
    model_dir: Path,
    output_root: Path,
    budget_state: dict[str, int],
) -> Dict[str, Any] | None:
    if not bool(getattr(args, "semantic_judge", False)):
        return None
    truncated_ids = [
        clean_text(row.get("question_id"))
        for row in prediction_rows
        if bool((row.get("prompt_truncation") or {}).get("prompt_truncated"))
    ]
    if truncated_ids:
        raise ValueError(
            "Grounded semantic evaluation requires the judge evidence to match what the "
            "candidate model could see. Reduce max_resources/max_resource_chars, use "
            "sequential resource windows, or increase max_seq_length. Truncated questions: "
            + ", ".join(truncated_ids[:10])
        )
    from .grounded_semantic_eval import evaluate_grounded_semantics

    return evaluate_grounded_semantics(
        prediction_rows=prediction_rows,
        examples_by_key=examples_by_key,
        args=args,
        output_dir=model_dir / "grounded_semantic",
        cache_dir=output_root / "_semantic_judge_cache",
        budget_state=budget_state,
    )


def seed_generation(seed: int, *, include_torch: bool = True) -> None:
    try:
        import numpy as np
    except Exception:
        np = None

    if include_torch:
        try:
            import torch
        except Exception:
            torch = None
    else:
        torch = None

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def prepare_output_root(args: argparse.Namespace, project_root: Path, prompt_id: str) -> Path:
    if args.output_dir:
        resolved = resolve_repo_path(args.output_dir, project_root=project_root)
        if resolved is None:
            raise ValueError("Could not resolve --output-dir.")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    root = project_root / "Artifacts" / "evaluations" / f"{utc_now_iso().replace(':', '').replace('+00:00', 'z')}-{slugify(prompt_id)}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def summarize_prompt_truncation(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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


def summarize_prediction_prompt_truncation(prediction_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    generation_entries = [
        entry
        for row in prediction_rows
        for entry in row.get("generation_telemetry", [])
        if isinstance(entry, Mapping)
    ]
    summary = summarize_prompt_truncation(generation_entries)
    summary["question_count"] = len(prediction_rows)
    summary["truncated_question_count"] = sum(
        1
        for row in prediction_rows
        if bool((row.get("prompt_truncation") or {}).get("prompt_truncated"))
    )
    return summary


def resource_window_mode(args: argparse.Namespace) -> str:
    return clean_text(getattr(args, "resource_window_mode", "single")).lower() or "single"


def effective_resource_window_step(args: argparse.Namespace) -> int:
    window_size = max(1, int(getattr(args, "max_resources", 0) or 1))
    window_step = int(getattr(args, "resource_window_step", 0) or 0)
    return window_size if window_step <= 0 else window_step


def build_resource_window_examples(
    example: EvalExample,
    args: argparse.Namespace,
) -> list[tuple[EvalExample, Dict[str, int]]]:
    resources = tuple(resource for resource in example.resources if clean_text(resource))
    normalized_example = replace(example, resources=resources)
    total_resources = len(resources)
    mode = resource_window_mode(args)
    if mode != "sequential" or total_resources == 0:
        return [
            (
                normalized_example,
                {
                    "window_index": 1,
                    "window_count": 1,
                    "resource_index_start": 1 if total_resources else 0,
                    "resource_index_end": total_resources,
                    "resource_count": total_resources,
                    "resource_total_count": total_resources,
                },
            )
        ]

    window_size = max(1, int(getattr(args, "max_resources", 0) or 1))
    window_step = effective_resource_window_step(args)
    spans: list[tuple[int, int]] = []
    for start in range(0, total_resources, window_step):
        stop = min(start + window_size, total_resources)
        if stop <= start:
            continue
        spans.append((start, stop))

    window_count = len(spans)
    return [
        (
            replace(normalized_example, resources=resources[start:stop]),
            {
                "window_index": index + 1,
                "window_count": window_count,
                "resource_index_start": start + 1,
                "resource_index_end": stop,
                "resource_count": stop - start,
                "resource_total_count": total_resources,
            },
        )
        for index, (start, stop) in enumerate(spans)
    ]


def aggregate_prediction_texts(
    predictions: Sequence[str],
    *,
    question_type: str,
    args: argparse.Namespace,
) -> str:
    cleaned_predictions = [clean_text(prediction) for prediction in predictions if clean_text(prediction)]
    if not cleaned_predictions:
        return ""
    if len(cleaned_predictions) == 1:
        return cleaned_predictions[0]
    return aggregate_generated_answers(cleaned_predictions, question_type=question_type, args=args)


def evaluate_single_model(
    model_spec: ModelSpec,
    examples: Sequence[EvalExample],
    args: argparse.Namespace,
    output_root: Path,
    prompt_bundle: Mapping[str, Any],
    supported_question_types: Sequence[str],
    semantic_budget_state: dict[str, int],
) -> Dict[str, Any]:
    model_dir = output_root / slugify(model_spec.label, fallback="model")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_spec.label} ({model_spec.load_target})")
    model, tokenizer = load_model_and_tokenizer_for_eval(model_spec, args)
    active_chat_template = model_spec.chat_template or clean_text(args.chat_template) or clean_text(prompt_bundle.get("chat_template", ""))
    active_prompt_format = model_spec.prompt_format or clean_text(args.prompt_format) or clean_text(prompt_bundle.get("prompt_format", "")) or "chat"

    prediction_rows = []
    window_mode = resource_window_mode(args)
    window_step = effective_resource_window_step(args) if window_mode == "sequential" else None
    examples_by_key = {
        (clean_text(example.question_id), clean_text(example.question_type).lower()): example
        for example in examples
    }
    for index, example in enumerate(examples, start=1):
        window_examples = build_resource_window_examples(example, args)
        window_rows = []
        for window_example, window_metadata in window_examples:
            if window_mode == "sequential":
                print(
                    f"[{model_spec.label}] {index}/{len(examples)} "
                    f"{example.question_type} {example.question_id} "
                    f"(window {window_metadata['window_index']}/{window_metadata['window_count']}, "
                    f"resources {window_metadata['resource_index_start']}-{window_metadata['resource_index_end']})"
                )
            else:
                print(f"[{model_spec.label}] {index}/{len(examples)} {example.question_type} {example.question_id}")

            prediction, generation_samples, generation_telemetry = generate_answer_samples(
                model=model,
                tokenizer=tokenizer,
                example=window_example,
                args=args,
                chat_template=active_chat_template,
                prompt_format=active_prompt_format,
            )
            window_rows.append(
                {
                    **window_metadata,
                    "prediction": prediction,
                    "generation_samples": generation_samples,
                    "generation_telemetry": generation_telemetry,
                    "prompt_truncation": summarize_prompt_truncation(generation_telemetry),
                }
            )

        aggregated_window_predictions = [row["prediction"] for row in window_rows]
        flattened_generation_telemetry = [
            entry
            for row in window_rows
            for entry in row.get("generation_telemetry", [])
            if isinstance(entry, Mapping)
        ]
        final_prediction = (
            aggregate_prediction_texts(
                aggregated_window_predictions,
                question_type=example.question_type,
                args=args,
            )
            if window_mode == "sequential"
            else aggregated_window_predictions[0]
        )
        prediction_row = {
            "question_id": example.question_id,
            "question_type": example.question_type,
            "body": example.body,
            "source_path": example.source_path,
            "prompt_instruction": example.instruction,
            "prediction": final_prediction,
            "generation_samples": (
                aggregated_window_predictions
                if window_mode == "sequential"
                else window_rows[0]["generation_samples"]
            ),
            "generation_telemetry": flattened_generation_telemetry,
            "prompt_truncation": summarize_prompt_truncation(flattened_generation_telemetry),
            "gold_output": example.gold_output,
        }
        if window_mode == "sequential":
            prediction_row.update(
                {
                    "resource_window_mode": window_mode,
                    "resource_window_size": int(args.max_resources or 0),
                    "resource_window_step": window_step,
                    "resource_window_count": len(window_rows),
                    "resource_total_count": window_rows[0]["resource_total_count"] if window_rows else 0,
                    "window_predictions": window_rows,
                }
            )
        prediction_rows.append(prediction_row)

    prompt_truncation_summary = summarize_prediction_prompt_truncation(prediction_rows)
    question_type_set = {
        clean_text(row.get("question_type", "")).lower()
        for row in prediction_rows
        if clean_text(row.get("question_type", ""))
    }
    exact_only = bool(question_type_set) and question_type_set.issubset(set(OFFICIAL_EXACT_TYPES))
    scoring_requested = str(args.score_backend) != "none"

    scorers: Dict[str, Any] = {}
    selected_backend: str | None = None
    selected_aggregate: Dict[str, Any] | None = None
    selected_batches: list[Dict[str, Any]] = []
    if scoring_requested:
        if not exact_only:
            raise ValueError(
                "Metric scoring now requires an exact-answer BioASQ Phase-B set "
                "(yesno, factoid, or list). Python metric scoring has been removed."
            )
        scorers = {
            "bioasq_java": evaluate_with_bioasq_java(
                prediction_rows=prediction_rows,
                examples_by_key=examples_by_key,
                model_label=model_spec.label,
                model_dir=model_dir,
                args=args,
                include_per_question=True,
            )
        }
        official_by_id = {row["question_id"]: row for row in scorers["bioasq_java"]["per_question"]}
        for row in prediction_rows:
            row["score"] = official_by_id[row["question_id"]]
        selected_backend = "bioasq_java"
        selected_payload = scorers[selected_backend]
        selected_aggregate = selected_payload["aggregate"]
        selected_batches = selected_payload["batches"]
    semantic_summary = maybe_run_grounded_semantic_evaluation(
        prediction_rows=prediction_rows,
        examples_by_key=examples_by_key,
        args=args,
        model_dir=model_dir,
        output_root=output_root,
        budget_state=semantic_budget_state,
    )
    score_payload = {
        "created_at": utc_now_iso(),
        "model": {
            "label": model_spec.label,
            "ref": model_spec.ref,
            "source": model_spec.source,
            "load_target": model_spec.load_target,
            "run_id": model_spec.run_id,
            "alias": model_spec.alias,
            "base_model": model_spec.base_model,
            "adapter_dir": model_spec.adapter_dir,
            "chat_template": active_chat_template or None,
            "prompt_format": active_prompt_format,
        },
        "prompt": {
            "prompt_id": prompt_bundle.get("prompt_id"),
            "name": prompt_bundle.get("name"),
            "source": prompt_bundle.get("source"),
            "registry_path": prompt_bundle.get("registry_path"),
            "prompt_format": clean_text(prompt_bundle.get("prompt_format", "")) or None,
        },
        "dataset": {
            "eval_input": sorted({row["source_path"] for row in prediction_rows}),
            "question_types": sorted({row["question_type"] for row in prediction_rows}),
            "question_count": len(prediction_rows),
            "max_resources": args.max_resources,
            "max_resource_chars": args.max_resource_chars,
            "resource_selection": args.resource_selection,
            "resource_granularity": args.resource_granularity,
            "resource_window_mode": window_mode,
            "resource_window_size": args.max_resources if window_mode == "sequential" else None,
            "resource_window_step": window_step if window_mode == "sequential" else None,
            "resource_reranker_model": args.resource_reranker_model if args.resource_selection == "embedding" else None,
            "resource_reranker_article_model": (
                args.resource_reranker_article_model if args.resource_selection == "embedding" else None
            ),
            "resource_reranker_device": args.resource_reranker_device if args.resource_selection == "embedding" else None,
        },
        "generation": {
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "num_generations": args.num_generations,
            "aggregation_strategy": args.aggregation_strategy,
            "aggregation_min_frequency": args.aggregation_min_frequency,
            "do_sample": bool(args.do_sample),
            "seed": int(getattr(args, "seed", 3407) or 3407),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "summary_reference_mode": args.summary_reference_mode,
            "prompt_truncation": prompt_truncation_summary,
        },
        "scoring": {
            "requested_backend": args.score_backend,
            "selected_backend": selected_backend,
            "available_backends": sorted(scorers.keys()),
            "exact_answer_official_only": exact_only,
            "performed": scoring_requested,
            "grounded_semantic_performed": semantic_summary is not None,
        },
        "grounded_semantic": semantic_summary,
        "aggregate": selected_aggregate,
        "batches": selected_batches,
        "scorers": scorers,
    }

    write_json(model_dir / "predictions.json", prediction_rows)
    write_json(
        model_dir / "bioasq_predictions.json",
        {
            "system": model_spec.label,
            "prompt_id": prompt_bundle.get("prompt_id"),
            "questions": [build_bioasq_prediction_entry(row) for row in prediction_rows],
        },
    )
    if scoring_requested or semantic_summary is not None:
        write_json(model_dir / "scores.json", score_payload)
    return {
        "model": score_payload["model"],
        "prompt": score_payload["prompt"],
        "dataset": score_payload["dataset"],
        "generation": score_payload["generation"],
        "scoring": score_payload["scoring"],
        "aggregate": selected_aggregate,
        "batches": selected_batches,
        "scorers": scorers,
        "grounded_semantic": semantic_summary,
        "paths": {
            "model_dir": str(model_dir),
            "predictions": str(model_dir / "predictions.json"),
            "scores": str(model_dir / "scores.json") if scoring_requested or semantic_summary is not None else None,
            "bioasq_predictions": str(model_dir / "bioasq_predictions.json"),
            "official_scores": (
                str(model_dir / "official_bioasq" / "official_scores.json")
                if scoring_requested and "bioasq_java" in scorers
                else None
            ),
            "grounded_semantic": (
                str(model_dir / "grounded_semantic" / "summary.json")
                if semantic_summary is not None
                else None
            ),
        },
    }


def evaluate_openai_model(
    *,
    model_name: str,
    examples: Sequence[EvalExample],
    args: argparse.Namespace,
    output_root: Path,
    prompt_bundle: Mapping[str, Any],
    candidate_budget_state: dict[str, int],
    semantic_budget_state: dict[str, int],
) -> Dict[str, Any]:
    """Generate and score one API model using the local evaluator's artifact schema."""
    from .eval_openai import OpenAICandidateGenerator

    label = f"openai-{model_name}"
    model_dir = output_root / slugify(label, fallback="openai-model")
    model_dir.mkdir(parents=True, exist_ok=True)
    generator = OpenAICandidateGenerator(
        model=model_name,
        args=args,
        cache_dir=model_dir / "generation_cache",
        budget_state=candidate_budget_state,
        read_cache_dir=(Path(args.openai_cache_dir) if args.openai_cache_dir else None),
    )
    prediction_rows: list[dict[str, Any]] = []
    window_mode = resource_window_mode(args)
    window_step = effective_resource_window_step(args) if window_mode == "sequential" else None
    for index, example in enumerate(examples, 1):
        window_rows: list[dict[str, Any]] = []
        for window_example, window_metadata in build_resource_window_examples(example, args):
            suffix = (
                f" window {window_metadata['window_index']}/{window_metadata['window_count']}"
                if window_mode == "sequential"
                else ""
            )
            print(
                f"[{label}] {index}/{len(examples)} {example.question_type} "
                f"{example.question_id}{suffix}"
            )
            prediction, samples, telemetry = generator.generate(window_example)
            window_rows.append(
                {
                    **window_metadata,
                    "prediction": prediction,
                    "generation_samples": samples,
                    "generation_telemetry": telemetry,
                }
            )
        window_predictions = [row["prediction"] for row in window_rows]
        final_prediction = (
            aggregate_prediction_texts(
                window_predictions,
                question_type=example.question_type,
                args=args,
            )
            if window_mode == "sequential"
            else window_predictions[0]
        )
        flattened_telemetry = [
            value
            for window in window_rows
            for value in window["generation_telemetry"]
        ]
        prediction_rows.append(
            {
                "question_id": example.question_id,
                "question_type": example.question_type,
                "body": example.body,
                "source_path": example.source_path,
                "prompt_instruction": example.instruction,
                "prediction": final_prediction,
                "generation_samples": (
                    window_predictions if window_mode == "sequential" else window_rows[0]["generation_samples"]
                ),
                "generation_telemetry": flattened_telemetry,
                "prompt_truncation": summarize_prompt_truncation(flattened_telemetry),
                "gold_output": example.gold_output,
                **(
                    {
                        "resource_window_mode": window_mode,
                        "resource_window_size": int(args.max_resources or 0),
                        "resource_window_step": window_step,
                        "resource_window_count": len(window_rows),
                        "resource_total_count": window_rows[0]["resource_total_count"] if window_rows else 0,
                        "window_predictions": window_rows,
                    }
                    if window_mode == "sequential"
                    else {}
                ),
            }
        )

    examples_by_key = {
        (clean_text(example.question_id), clean_text(example.question_type).lower()): example
        for example in examples
    }
    question_type_set = {clean_text(row["question_type"]).lower() for row in prediction_rows}
    exact_only = bool(question_type_set) and question_type_set.issubset(set(OFFICIAL_EXACT_TYPES))
    scoring_requested = str(args.score_backend) != "none"
    scorers: Dict[str, Any] = {}
    selected_aggregate = None
    selected_batches: list[Dict[str, Any]] = []
    if scoring_requested:
        if not exact_only:
            raise ValueError("Metric scoring requires an exact-answer BioASQ Phase-B set.")
        scorers["bioasq_java"] = evaluate_with_bioasq_java(
            prediction_rows=prediction_rows,
            examples_by_key=examples_by_key,
            model_label=label,
            model_dir=model_dir,
            args=args,
            include_per_question=True,
        )
        official_by_id = {row["question_id"]: row for row in scorers["bioasq_java"]["per_question"]}
        for row in prediction_rows:
            row["score"] = official_by_id[row["question_id"]]
        selected_aggregate = scorers["bioasq_java"]["aggregate"]
        selected_batches = scorers["bioasq_java"]["batches"]

    semantic_summary = maybe_run_grounded_semantic_evaluation(
        prediction_rows=prediction_rows,
        examples_by_key=examples_by_key,
        args=args,
        model_dir=model_dir,
        output_root=output_root,
        budget_state=semantic_budget_state,
    )
    model_payload = {
        "label": label,
        "ref": model_name,
        "source": "openai-api",
        "load_target": model_name,
        "run_id": None,
        "alias": None,
        "base_model": model_name,
        "adapter_dir": None,
        "chat_template": "openai-chat-completions",
        "prompt_format": "chat",
    }
    prompt_payload = {
        "prompt_id": prompt_bundle.get("prompt_id"),
        "name": prompt_bundle.get("name"),
        "source": prompt_bundle.get("source"),
        "registry_path": prompt_bundle.get("registry_path"),
        "prompt_format": clean_text(prompt_bundle.get("prompt_format", "")) or None,
    }
    generation_payload = {
        "backend": "openai",
        "max_new_tokens": args.max_new_tokens,
        "num_generations": args.num_generations,
        "aggregation_strategy": args.aggregation_strategy,
        "aggregation_min_frequency": args.aggregation_min_frequency,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "api_budget": dict(candidate_budget_state),
        "prompt_truncation": summarize_prediction_prompt_truncation(prediction_rows),
    }
    score_payload = {
        "created_at": utc_now_iso(),
        "model": model_payload,
        "prompt": prompt_payload,
        "dataset": {
            "eval_input": sorted({row["source_path"] for row in prediction_rows}),
            "question_types": sorted(question_type_set),
            "question_count": len(prediction_rows),
            "max_resources": args.max_resources,
            "max_resource_chars": args.max_resource_chars,
        },
        "generation": generation_payload,
        "scoring": {
            "requested_backend": args.score_backend,
            "selected_backend": "bioasq_java" if scoring_requested else None,
            "available_backends": sorted(scorers),
            "performed": scoring_requested,
            "grounded_semantic_performed": semantic_summary is not None,
        },
        "aggregate": selected_aggregate,
        "batches": selected_batches,
        "scorers": scorers,
        "grounded_semantic": semantic_summary,
    }
    write_json(model_dir / "predictions.json", prediction_rows)
    write_json(
        model_dir / "bioasq_predictions.json",
        {
            "system": label,
            "prompt_id": prompt_bundle.get("prompt_id"),
            "questions": [build_bioasq_prediction_entry(row) for row in prediction_rows],
        },
    )
    if scoring_requested or semantic_summary is not None:
        write_json(model_dir / "scores.json", score_payload)
    return {
        **score_payload,
        "paths": {
            "model_dir": str(model_dir),
            "predictions": str(model_dir / "predictions.json"),
            "scores": str(model_dir / "scores.json"),
            "bioasq_predictions": str(model_dir / "bioasq_predictions.json"),
            "official_scores": (
                str(model_dir / "official_bioasq" / "official_scores.json") if scoring_requested else None
            ),
            "grounded_semantic": (
                str(model_dir / "grounded_semantic" / "summary.json")
                if semantic_summary is not None
                else None
            ),
        },
    }


def run_evaluation(args: argparse.Namespace, supported_question_types: Sequence[str]) -> None:
    project_root = get_project_root()
    has_local_models = bool(args.model_ref or args.all_registry_runs)
    if has_local_models:
        prime_unsloth_runtime()
    seed_generation(
        int(getattr(args, "seed", 3407) or 3407),
        include_torch=has_local_models,
    )
    args.api_key_file = str(resolve_repo_path(args.api_key_file, project_root=project_root) or args.api_key_file)
    if args.openai_cache_dir:
        args.openai_cache_dir = str(
            resolve_repo_path(args.openai_cache_dir, project_root=project_root)
            or args.openai_cache_dir
        )
    args.semantic_judge_api_key_file = str(
        resolve_repo_path(args.semantic_judge_api_key_file, project_root=project_root)
        or args.semantic_judge_api_key_file
    )
    if args.semantic_judge_cache_dir:
        args.semantic_judge_cache_dir = str(
            resolve_repo_path(args.semantic_judge_cache_dir, project_root=project_root)
            or args.semantic_judge_cache_dir
        )
    prompt_path_value = args.prompt_file or args.prompt_registry_path
    prompt_registry_path = resolve_repo_path(prompt_path_value, project_root=project_root)
    prompt_bundle = resolve_prompt_bundle(
        registry_path=prompt_registry_path,
        prompt_ref=args.prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    eval_paths = resolve_eval_input_paths(args, project_root=project_root)
    examples = load_eval_examples(eval_paths, args, prompt_instructions=prompt_bundle["instructions"])
    if not examples:
        raise ValueError("No evaluation examples were prepared from the provided inputs.")

    model_specs = resolve_model_specs(args, project_root=project_root) if has_local_models else []
    output_root = prepare_output_root(args, project_root=project_root, prompt_id=str(prompt_bundle.get("prompt_id")))

    print(f"Prepared {len(examples)} evaluation examples.")
    print(f"Using prompt bundle: {prompt_bundle.get('prompt_id')} ({prompt_bundle.get('source')})")
    print(f"Writing evaluation artifacts to: {output_root}")

    candidate_budget_state = {"new_api_calls": 0, "retry_count": 0}
    semantic_budget_state = {"new_judge_calls": 0, "judge_retry_count": 0}
    summaries = [
        evaluate_single_model(
            model_spec=model_spec,
            examples=examples,
            args=args,
            output_root=output_root,
            prompt_bundle=prompt_bundle,
            supported_question_types=supported_question_types,
            semantic_budget_state=semantic_budget_state,
        )
        for model_spec in model_specs
    ]
    summaries.extend(
        evaluate_openai_model(
            model_name=model_name,
            examples=examples,
            args=args,
            output_root=output_root,
            prompt_bundle=prompt_bundle,
            candidate_budget_state=candidate_budget_state,
            semantic_budget_state=semantic_budget_state,
        )
        for model_name in (args.openai_model or [])
    )

    write_json(
        output_root / "manifest.json",
        {
            "created_at": utc_now_iso(),
            "output_root": str(output_root),
            "prompt": {
                "prompt_id": prompt_bundle.get("prompt_id"),
                "source": prompt_bundle.get("source"),
                "registry_path": prompt_bundle.get("registry_path"),
                "prompt_format": clean_text(prompt_bundle.get("prompt_format", "")) or None,
                "model_prompt_formats": sorted(
                    {
                        str(summary["model"].get("prompt_format"))
                        for summary in summaries
                        if summary["model"].get("prompt_format")
                    }
                ),
                "model_chat_templates": sorted(
                    {
                        str(summary["model"].get("chat_template"))
                        for summary in summaries
                        if summary["model"].get("chat_template")
                    }
                ),
            },
            "dataset": {
                "eval_input": [str(path) for path in eval_paths],
                "question_count": len(examples),
                "question_types": sorted({example.question_type for example in examples}),
                "summary_reference_mode": args.summary_reference_mode,
                "max_resources": args.max_resources,
                "max_resource_chars": args.max_resource_chars,
                "resource_selection": args.resource_selection,
                "resource_granularity": args.resource_granularity,
                "resource_window_mode": resource_window_mode(args),
                "resource_window_size": args.max_resources if resource_window_mode(args) == "sequential" else None,
                "resource_window_step": effective_resource_window_step(args) if resource_window_mode(args) == "sequential" else None,
                "resource_reranker_model": args.resource_reranker_model if args.resource_selection == "embedding" else None,
                "resource_reranker_article_model": (
                    args.resource_reranker_article_model if args.resource_selection == "embedding" else None
                ),
                "resource_reranker_device": args.resource_reranker_device if args.resource_selection == "embedding" else None,
            },
            "generation": {
                "max_seq_length": args.max_seq_length,
                "max_new_tokens": args.max_new_tokens,
                "num_generations": args.num_generations,
                "aggregation_strategy": args.aggregation_strategy,
                "aggregation_min_frequency": args.aggregation_min_frequency,
                "do_sample": bool(args.do_sample),
                "seed": int(getattr(args, "seed", 3407) or 3407),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "summary_reference_mode": args.summary_reference_mode,
            },
            "scoring": {
                "requested_backend": args.score_backend,
                "bioasq_java_jar": args.bioasq_java_jar if args.score_backend == "bioasq_java" else None,
                "bioasq_java_version": args.bioasq_java_version if args.score_backend == "bioasq_java" else None,
                "bioasq_java_heap": args.bioasq_java_heap if args.score_backend == "bioasq_java" else None,
                "grounded_semantic_enabled": bool(args.semantic_judge),
                "semantic_judge_model": args.semantic_judge_model if args.semantic_judge else None,
                "semantic_judge_rubric": (
                    "bioasq-grounded-semantic-eval-v1" if args.semantic_judge else None
                ),
                "semantic_judge_budget": dict(semantic_budget_state),
            },
            "openai_candidate_generation": {
                "models": list(args.openai_model or []),
                "budget": dict(candidate_budget_state),
            },
            "models": summaries,
        },
    )
    print(f"Saved evaluation manifest to {output_root / 'manifest.json'}")
