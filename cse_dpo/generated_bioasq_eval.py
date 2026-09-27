from __future__ import annotations

import argparse
import copy
import gc
import json
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import build_output, clean_text, list_record_resources
from src.utility.bioasq_format import exact_answer_groups, normalize_for_match, parse_prediction_items
from src.utility.bioasq_official import evaluate_with_bioasq_java
from src.utility.eval_models import first_model_device
from src.utility.eval_types import EvalExample


def build_gold_output_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_summary_answers=5,
        max_factoid_answers=5,
        max_list_items=100,
    )


def load_gold_examples(paths: Sequence[Path]) -> dict[str, EvalExample]:
    gold_examples: dict[str, EvalExample] = {}
    output_args = build_gold_output_args()

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            for index, row in enumerate(payload):
                if not isinstance(row, dict):
                    continue
                question_id = clean_text(row.get("id", "")) or f"{path.name}:{index}"
                question_type = clean_text(row.get("type", "")).lower()
                body = clean_text(row.get("input_1", ""))
                gold_output = clean_text(row.get("output", ""))
                instruction = clean_text(row.get("instruction", "")) or QUESTION_INSTRUCTIONS.get(question_type, "")
                if not question_type or not gold_output:
                    continue
                gold_examples.setdefault(
                    question_id,
                    EvalExample(
                        question_id=question_id,
                        question_type=question_type,
                        body=body,
                        instruction=instruction,
                        resources=tuple(resource for resource in list_record_resources(row) if clean_text(resource)),
                        gold_output=gold_output,
                        source_path=str(path),
                        raw_question=None,
                    ),
                )
            continue

        questions = payload.get("questions") if isinstance(payload, dict) else None
        if not isinstance(questions, list):
            raise ValueError(f"Unsupported generated-eval gold input format: {path}")

        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                continue
            question_id = clean_text(question.get("id", "")) or f"{path.name}:{index}"
            question_type = clean_text(question.get("type", "")).lower()
            body = clean_text(question.get("body", ""))
            gold_output = clean_text(build_output(question, output_args))
            instruction = QUESTION_INSTRUCTIONS.get(question_type, "")
            if not question_type or not gold_output:
                continue
            gold_examples.setdefault(
                question_id,
                EvalExample(
                    question_id=question_id,
                    question_type=question_type,
                    body=body,
                    instruction=instruction,
                    resources=(),
                    gold_output=gold_output,
                    source_path=str(path),
                    raw_question=question,
                ),
            )

    if not gold_examples:
        raise ValueError("No gold examples were loaded for generated BioASQ evaluation.")
    return gold_examples


def build_generated_eval_rows(
    eval_rows: Sequence[dict[str, Any]],
    gold_examples: dict[str, EvalExample],
) -> tuple[list[dict[str, Any]], list[str]]:
    selection_rows: list[dict[str, Any]] = []
    missing_question_ids: list[str] = []
    seen_question_ids: set[str] = set()

    for row in eval_rows:
        prompt = str(row.get("prompt") or "").strip()
        question_id = clean_text(row.get("question_id", ""))
        if not prompt or not question_id:
            continue
        if question_id in seen_question_ids:
            continue
        seen_question_ids.add(question_id)

        example = gold_examples.get(question_id)
        if example is None:
            missing_question_ids.append(question_id)
            continue

        selection_rows.append(
            {
                "question_id": question_id,
                "prompt": prompt,
                "example": example,
            }
        )

    return selection_rows, sorted(set(missing_question_ids))


def _official_score_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    return argparse.Namespace(
        bioasq_java_jar=str(
            project_root / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
        ),
        bioasq_java_heap="512m",
        bioasq_java_version=5,
    )


def generate_answer_from_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_seq_length: int,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generation_seed: int | None = None,
    empty_cuda_cache_per_generation: bool = False,
) -> str:
    encode_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    previous_truncation_side = getattr(tokenizer, "truncation_side", None)
    try:
        if max_seq_length > 0:
            encode_kwargs.update({"truncation": True, "max_length": max_seq_length})
            if previous_truncation_side is not None:
                tokenizer.truncation_side = "left"
        encoded = tokenizer(prompt, **encode_kwargs)
    finally:
        if previous_truncation_side is not None:
            tokenizer.truncation_side = previous_truncation_side

    device = first_model_device(model)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "do_sample": bool(do_sample),
        "use_cache": False,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)

    output_ids = None
    try:
        if generation_seed is not None:
            torch = __import__("torch")
            torch.manual_seed(int(generation_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(generation_seed))
        with __import__("torch").inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = clean_text(decoded)
        return cleaned[7:].strip() if cleaned.lower().startswith("answer:") else cleaned
    finally:
        del output_ids
        del encoded
        gc.collect()
        torch = __import__("torch")
        if bool(empty_cuda_cache_per_generation) and torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_factoid_self_consistency(samples: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """Choose the most frequent valid single-answer factoid sample.

    Samples that contain zero or multiple factoid items are excluded from the
    vote because this evaluation protocol requires exactly one answer.
    """
    votes: Counter[str] = Counter()
    representative_by_key: dict[str, str] = {}
    first_seen: dict[str, int] = {}

    for sample_index, sample in enumerate(samples):
        items = parse_prediction_items(sample, "factoid")
        if len(items) != 1:
            continue
        candidate = clean_text(items[0])
        key = normalize_for_match(candidate) or candidate.casefold()
        if not key:
            continue
        votes[key] += 1
        representative_by_key.setdefault(key, candidate)
        first_seen.setdefault(key, sample_index)

    if not votes:
        return clean_text(samples[0]) if samples else "", {
            "valid_single_answer_sample_count": 0,
            "unique_valid_candidate_count": 0,
            "selected_vote_count": 0,
            "selection_fallback": "first_raw_sample_no_valid_single_answer_vote",
        }

    selected_key = min(votes, key=lambda key: (-votes[key], first_seen[key]))
    return f"[BE] {representative_by_key[selected_key]} [EE]", {
        "valid_single_answer_sample_count": sum(votes.values()),
        "unique_valid_candidate_count": len(votes),
        "selected_vote_count": votes[selected_key],
        "selection_fallback": None,
    }


def evaluate_generated_bioasq(
    model: Any,
    tokenizer: Any,
    eval_rows: Sequence[dict[str, Any]],
    *,
    max_seq_length: int,
    max_new_tokens: int,
    summary_reference_mode: str = "all-mean",
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_generations: int = 1,
    aggregation_strategy: str = "first",
    sampling_seed: int | None = None,
    empty_cuda_cache_per_generation: bool = False,
    progress_factory: Any = None,
    progress_desc: str = "Generated validation",
    official_output_dir: Path | None = None,
    official_model_label: str = "generated-bioasq-eval",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if aggregation_strategy not in {"first", "self_consistency"}:
        raise ValueError(f"Unsupported aggregation strategy: {aggregation_strategy}")
    if num_generations > 1 and aggregation_strategy != "self_consistency":
        raise ValueError("Multiple generations require aggregation_strategy='self_consistency'.")
    if num_generations > 1 and not do_sample:
        raise ValueError("Multiple generations require do_sample=True.")

    rows_iter = eval_rows
    if progress_factory is not None:
        rows_iter = progress_factory(eval_rows, desc=progress_desc, leave=False)

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    metrics: list[dict[str, Any]] = []
    try:
        for row_index, row in enumerate(rows_iter):
            example = row["example"]
            samples = [
                generate_answer_from_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=row["prompt"],
                    max_seq_length=max_seq_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    generation_seed=(sampling_seed + row_index * num_generations + sample_index)
                    if sampling_seed is not None else None,
                    empty_cuda_cache_per_generation=empty_cuda_cache_per_generation,
                )
                for sample_index in range(num_generations)
            ]
            if aggregation_strategy == "self_consistency":
                prediction, selection = select_factoid_self_consistency(samples)
            else:
                prediction = samples[0]
                selection = {
                    "valid_single_answer_sample_count": 1 if parse_prediction_items(prediction, "factoid") else 0,
                    "unique_valid_candidate_count": 1 if parse_prediction_items(prediction, "factoid") else 0,
                    "selected_vote_count": 1 if parse_prediction_items(prediction, "factoid") else 0,
                    "selection_fallback": None,
                }
            metrics.append(
                {
                    "question_id": example.question_id,
                    "question_type": example.question_type,
                    "prediction": prediction,
                    "gold_output": example.gold_output,
                    "body": example.body,
                    "source_path": example.source_path,
                    "prediction_count": len(
                        parse_prediction_items(prediction, example.question_type)[:5]
                        if example.question_type == "factoid"
                        else parse_prediction_items(prediction, example.question_type)
                    ),
                    "gold_count": len(exact_answer_groups(example, example.question_type)),
                    "generation_sample_count": num_generations,
                    "valid_single_answer_sample_count": selection["valid_single_answer_sample_count"],
                    "unique_valid_candidate_count": selection["unique_valid_candidate_count"],
                    "selected_vote_count": selection["selected_vote_count"],
                    "selection_fallback": selection["selection_fallback"],
                    "generation_samples": samples,
                }
            )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()

    scorer_root = Path(official_output_dir) if official_output_dir is not None else Path(
        tempfile.mkdtemp(prefix="bioasq-official-generated-")
    )
    examples_by_key = {
        (row["example"].question_id, row["example"].question_type): row["example"]
        for row in eval_rows
    }
    official = evaluate_with_bioasq_java(
        prediction_rows=metrics,
        examples_by_key=examples_by_key,
        model_label=official_model_label,
        model_dir=scorer_root,
        args=_official_score_args(),
        include_per_question=True,
    )
    official_by_id = {row["question_id"]: row for row in official["per_question"]}
    for row in metrics:
        score = official_by_id[row["question_id"]]
        row.update({key: value for key, value in score.items() if key not in {"question_id", "question_type"}})
        row["scoring_backend"] = "bioasq_java"

    # Oracle-at-N remains a diagnostic, but its sample scores also come from the
    # official Java implementation. Synthetic IDs allow all samples to be
    # evaluated in one Java invocation without altering the source questions.
    if num_generations == 1:
        for row in metrics:
            row["oracle_any_sample_mrr"] = float(row.get("mrr", 0.0))
            row["oracle_any_sample_strict_accuracy"] = float(row.get("strict_accuracy", 0.0))
    else:
        sample_rows: list[dict[str, Any]] = []
        sample_examples: dict[tuple[str, str], EvalExample] = {}
        for row, source in zip(metrics, eval_rows):
            example = source["example"]
            for sample_index, sample in enumerate(row["generation_samples"]):
                sample_id = f"{example.question_id}__sample_{sample_index}"
                raw_question = copy.deepcopy(example.raw_question)
                if raw_question is not None:
                    raw_question["id"] = sample_id
                sample_example = replace(example, question_id=sample_id, raw_question=raw_question)
                sample_examples[(sample_id, example.question_type)] = sample_example
                sample_rows.append({
                    "question_id": sample_id, "question_type": example.question_type,
                    "body": example.body, "source_path": example.source_path,
                    "prediction": sample, "parent_question_id": example.question_id,
                })
        sample_official = evaluate_with_bioasq_java(
            prediction_rows=sample_rows,
            examples_by_key=sample_examples,
            model_label=f"{official_model_label}-samples",
            model_dir=scorer_root / "samples",
            args=_official_score_args(),
            include_per_question=True,
        )
        sample_scores: dict[str, list[dict[str, Any]]] = {}
        for score in sample_official["per_question"]:
            parent = score["question_id"].rsplit("__sample_", 1)[0]
            sample_scores.setdefault(parent, []).append(score)
        for row in metrics:
            scores = sample_scores.get(row["question_id"], [])
            row["oracle_any_sample_mrr"] = max((float(x.get("mrr", 0.0)) for x in scores), default=0.0)
            row["oracle_any_sample_strict_accuracy"] = max(
                (float(x.get("strict_accuracy", 0.0)) for x in scores), default=0.0
            )

    def _mean(key: str) -> float | None:
        values = [float(row[key]) for row in metrics if isinstance(row.get(key), (int, float))]
        return float(np.mean(values)) if values else None

    factoid_metrics = official["aggregate"].get("by_type", {}).get("factoid", {}).get("metrics", {})
    summary = {
        "question_count": len(metrics),
        "mrr": factoid_metrics.get("mrr"),
        "strict_accuracy": factoid_metrics.get("strict_accuracy"),
        "lenient_accuracy": factoid_metrics.get("lenient_accuracy"),
        "num_generations": int(num_generations),
        "aggregation_strategy": aggregation_strategy,
        "oracle_any_sample_mrr": _mean("oracle_any_sample_mrr"),
        "oracle_any_sample_strict_accuracy": _mean("oracle_any_sample_strict_accuracy"),
        "scoring_backend": "bioasq_java",
        "official_scores_path": str(scorer_root / "official_bioasq" / "official_scores.json"),
    }
    return summary, metrics
