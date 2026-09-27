from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, summarize_numeric, write_json, write_jsonl
from cse_dpo.score_candidate_bank import (
    ScoringPayload,
    release_model,
    resolve_single_model_spec,
    score_payloads_with_model,
)
from src.utility.data import clean_text
from src.utility.eval_models import load_model_and_tokenizer_for_eval, prime_unsloth_runtime


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def normalize_model_ref(model_ref: str) -> str:
    text = clean_text(model_ref)
    if not text:
        return text
    if text.lower() in {"none", "null"}:
        return ""

    path = Path(text)
    if path.is_absolute():
        return str(path)

    looks_like_local_path = "/" in text or "\\" in text or text.startswith(".")
    if not looks_like_local_path:
        return text

    if path.exists():
        return str(path.resolve())

    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score Cardinality Shortcut Study probe pairs with average answer-token "
            "log-likelihood so we can measure held-out candidate-ranking accuracy."
        )
    )
    parser.add_argument(
        "--probe-input",
        nargs="+",
        required=True,
        help="One or more natural/constructed probe-bank JSONL or JSON files.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where the per-probe scoring JSONL will be written.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Where the aggregate scoring summary JSON will be written.",
    )
    parser.add_argument(
        "--model-ref",
        required=True,
        help="Scoring model reference or local path.",
    )
    parser.add_argument(
        "--reference-model-ref",
        default=None,
        help=(
            "Optional reference model used to compute DPO-style implicit-reward margins "
            "with raw sequence log-probability sums."
        ),
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry path used to resolve model refs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of completions scored per forward pass.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help=(
            "Maximum prompt+completion sequence length used for scoring. Prompts are "
            "left-truncated first to preserve the completion span."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=100,
        help="Print progress after this many scored completions. Use 0 to disable.",
    )
    parser.add_argument(
        "--empty-cuda-cache-steps",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N scoring batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--margin-tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance used to call a score margin a tie.",
    )
    parser.add_argument(
        "--eos-policy",
        choices=["excluded", "included"],
        default="excluded",
        help="Whether to append a single EOS token to each candidate answer before scoring.",
    )
    parser.add_argument(
        "--ci-confidence-level",
        type=float,
        default=0.95,
        help="Confidence level used for reported accuracy intervals.",
    )
    parser.add_argument(
        "--question-bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples used for question-macro accuracy intervals.",
    )
    parser.add_argument(
        "--ci-seed",
        type=int,
        default=3407,
        help="Random seed used for deterministic confidence-interval resampling.",
    )
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_probe_rows(paths: Sequence[str], limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = resolve_project_path(raw_path)
        for row_index_within_file, raw_row in enumerate(load_json_records(path)):
            row = dict(raw_row)
            row["_probe_input_path"] = str(path)
            row["_probe_input_row_index"] = row_index_within_file
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_probe_payloads(
    rows: Sequence[Mapping[str, Any]],
    *,
    eos_policy: str,
) -> tuple[list[ScoringPayload], list[ScoringPayload], dict[str, Any]]:
    preferred_payloads: list[ScoringPayload] = []
    dispreferred_payloads: list[ScoringPayload] = []
    audit = Counter()
    append_eos = clean_text(eos_policy).lower() == "included"

    for row_index, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        preferred_answer = clean_text(row.get("preferred_answer"))
        dispreferred_answer = clean_text(row.get("dispreferred_answer"))

        missing_fields: list[str] = []
        if not prompt:
            missing_fields.append("prompt")
            audit["missing_prompt"] += 1
        if not preferred_answer:
            missing_fields.append("preferred_answer")
            audit["missing_preferred_answer"] += 1
        if not dispreferred_answer:
            missing_fields.append("dispreferred_answer")
            audit["missing_dispreferred_answer"] += 1

        if missing_fields:
            audit["invalid_probe_rows"] += 1
            continue

        preferred_payloads.append(
            ScoringPayload(
                row_index=row_index,
                prompt=prompt,
                completion_text=preferred_answer,
                append_eos=append_eos,
            )
        )
        dispreferred_payloads.append(
            ScoringPayload(
                row_index=row_index,
                prompt=prompt,
                completion_text=dispreferred_answer,
                append_eos=append_eos,
            )
        )
        audit["ready_probe_rows"] += 1

    audit["probe_row_count"] = len(rows)
    audit["preferred_payload_count"] = len(preferred_payloads)
    audit["dispreferred_payload_count"] = len(dispreferred_payloads)
    return preferred_payloads, dispreferred_payloads, dict(sorted(audit.items()))


def _result_value(result: Mapping[str, Any] | None, field_name: str) -> Any:
    if result is None:
        return None
    return result.get(field_name)


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def wilson_interval(successes: int, total: int, confidence_level: float) -> dict[str, float | int] | None:
    if total <= 0:
        return None
    z_value = NormalDist().inv_cdf(0.5 + (float(confidence_level) / 2.0))
    phat = float(successes) / float(total)
    denominator = 1.0 + (z_value * z_value) / float(total)
    center = (phat + (z_value * z_value) / (2.0 * float(total))) / denominator
    spread = (
        z_value
        * ((phat * (1.0 - phat) / float(total)) + ((z_value * z_value) / (4.0 * float(total) * float(total)))) ** 0.5
        / denominator
    )
    return {
        "confidence_level": float(confidence_level),
        "successes": int(successes),
        "total": int(total),
        "lower": max(0.0, center - spread),
        "upper": min(1.0, center + spread),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int] | None:
    if not values:
        return None
    if len(values) == 1:
        return {
            "confidence_level": float(confidence_level),
            "sample_count": 1,
            "bootstrap_samples": int(bootstrap_samples),
            "lower": float(values[0]),
            "upper": float(values[0]),
        }

    sample_count = len(values)
    rng = random.Random(int(seed))
    means: list[float] = []
    for _ in range(max(1, int(bootstrap_samples))):
        resampled = [float(values[rng.randrange(sample_count)]) for _ in range(sample_count)]
        means.append(sum(resampled) / float(sample_count))
    means.sort()

    alpha = max(0.0, min(1.0, 1.0 - float(confidence_level)))
    lower_rank = int(alpha / 2.0 * (len(means) - 1))
    upper_rank = int((1.0 - alpha / 2.0) * (len(means) - 1))
    return {
        "confidence_level": float(confidence_level),
        "sample_count": int(sample_count),
        "bootstrap_samples": int(len(means)),
        "lower": float(means[lower_rank]),
        "upper": float(means[upper_rank]),
    }


def attach_probe_scores(
    *,
    rows: list[dict[str, Any]],
    preferred_policy_results: Sequence[Mapping[str, Any]],
    dispreferred_policy_results: Sequence[Mapping[str, Any]],
    preferred_reference_results: Sequence[Mapping[str, Any]] | None,
    dispreferred_reference_results: Sequence[Mapping[str, Any]] | None,
    model_ref: str,
    reference_model_ref: str | None,
    tie_tolerance: float,
    eos_policy: str,
) -> None:
    preferred_policy_by_row = {int(result["row_index"]): dict(result) for result in preferred_policy_results}
    dispreferred_policy_by_row = {int(result["row_index"]): dict(result) for result in dispreferred_policy_results}
    preferred_reference_by_row = (
        {int(result["row_index"]): dict(result) for result in preferred_reference_results}
        if preferred_reference_results is not None
        else {}
    )
    dispreferred_reference_by_row = (
        {int(result["row_index"]): dict(result) for result in dispreferred_reference_results}
        if dispreferred_reference_results is not None
        else {}
    )

    for row_index, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        preferred_answer = clean_text(row.get("preferred_answer"))
        dispreferred_answer = clean_text(row.get("dispreferred_answer"))

        input_validation_errors: list[str] = []
        if not prompt:
            input_validation_errors.append("missing_prompt")
        if not preferred_answer:
            input_validation_errors.append("missing_preferred_answer")
        if not dispreferred_answer:
            input_validation_errors.append("missing_dispreferred_answer")

        row["scoring_model_ref"] = model_ref
        row["reference_scoring_model_ref"] = reference_model_ref
        row["scoring_metric"] = "average_answer_token_log_likelihood"
        row["scoring_prompt_tokens_excluded"] = True
        row["scoring_padding_tokens_excluded"] = True
        row["scoring_eos_treatment"] = eos_policy
        row["scoring_candidate_ranking_measurement"] = True
        row["input_validation_errors"] = input_validation_errors

        preferred_policy = preferred_policy_by_row.get(row_index)
        dispreferred_policy = dispreferred_policy_by_row.get(row_index)
        preferred_reference = preferred_reference_by_row.get(row_index)
        dispreferred_reference = dispreferred_reference_by_row.get(row_index)

        row["preferred_logp_sum"] = _result_value(preferred_policy, "logp_sum")
        row["preferred_logp_mean"] = _result_value(preferred_policy, "logp_mean")
        row["preferred_score_token_count"] = _result_value(preferred_policy, "score_token_count")
        row["preferred_prompt_token_count"] = _result_value(preferred_policy, "prompt_token_count")
        row["preferred_effective_prompt_token_count"] = _result_value(preferred_policy, "effective_prompt_token_count")
        row["preferred_completion_token_count"] = _result_value(preferred_policy, "completion_token_count")
        row["preferred_effective_completion_token_count"] = _result_value(
            preferred_policy, "effective_completion_token_count"
        )
        row["preferred_prompt_truncated"] = _result_value(preferred_policy, "prompt_truncated")
        row["preferred_completion_truncated"] = _result_value(preferred_policy, "completion_truncated")
        row["preferred_scoring_status"] = clean_text(_result_value(preferred_policy, "status")) or "scored"

        row["dispreferred_logp_sum"] = _result_value(dispreferred_policy, "logp_sum")
        row["dispreferred_logp_mean"] = _result_value(dispreferred_policy, "logp_mean")
        row["dispreferred_score_token_count"] = _result_value(dispreferred_policy, "score_token_count")
        row["dispreferred_prompt_token_count"] = _result_value(dispreferred_policy, "prompt_token_count")
        row["dispreferred_effective_prompt_token_count"] = _result_value(
            dispreferred_policy, "effective_prompt_token_count"
        )
        row["dispreferred_completion_token_count"] = _result_value(dispreferred_policy, "completion_token_count")
        row["dispreferred_effective_completion_token_count"] = _result_value(
            dispreferred_policy, "effective_completion_token_count"
        )
        row["dispreferred_prompt_truncated"] = _result_value(dispreferred_policy, "prompt_truncated")
        row["dispreferred_completion_truncated"] = _result_value(dispreferred_policy, "completion_truncated")
        row["dispreferred_scoring_status"] = clean_text(_result_value(dispreferred_policy, "status")) or "scored"

        preferred_mean = _numeric_or_none(row.get("preferred_logp_mean"))
        dispreferred_mean = _numeric_or_none(row.get("dispreferred_logp_mean"))
        if preferred_mean is not None and dispreferred_mean is not None:
            avg_margin = preferred_mean - dispreferred_mean
            row["avg_logprob_margin"] = avg_margin
            row["avg_logprob_ranked_preferred"] = avg_margin > tie_tolerance
            row["avg_logprob_ranking_tie"] = abs(avg_margin) <= tie_tolerance
        else:
            row["avg_logprob_margin"] = None
            row["avg_logprob_ranked_preferred"] = None
            row["avg_logprob_ranking_tie"] = None

        if input_validation_errors:
            row["scoring_status"] = "invalid_probe_row"
        elif preferred_mean is None or dispreferred_mean is None:
            row["scoring_status"] = "unscored"
        else:
            row["scoring_status"] = "scored"

        row["preferred_reference_logp_sum"] = _result_value(preferred_reference, "logp_sum")
        row["preferred_reference_logp_mean"] = _result_value(preferred_reference, "logp_mean")
        row["dispreferred_reference_logp_sum"] = _result_value(dispreferred_reference, "logp_sum")
        row["dispreferred_reference_logp_mean"] = _result_value(dispreferred_reference, "logp_mean")

        preferred_policy_sum = _numeric_or_none(row.get("preferred_logp_sum"))
        dispreferred_policy_sum = _numeric_or_none(row.get("dispreferred_logp_sum"))
        preferred_reference_sum = _numeric_or_none(row.get("preferred_reference_logp_sum"))
        dispreferred_reference_sum = _numeric_or_none(row.get("dispreferred_reference_logp_sum"))
        if (
            preferred_policy_sum is not None
            and dispreferred_policy_sum is not None
            and preferred_reference_sum is not None
            and dispreferred_reference_sum is not None
        ):
            preferred_reward = preferred_policy_sum - preferred_reference_sum
            dispreferred_reward = dispreferred_policy_sum - dispreferred_reference_sum
            reward_margin = preferred_reward - dispreferred_reward
            row["preferred_dpo_implicit_reward_sum"] = preferred_reward
            row["dispreferred_dpo_implicit_reward_sum"] = dispreferred_reward
            row["dpo_implicit_reward_margin_sum"] = reward_margin
            row["dpo_implicit_reward_ranked_preferred"] = reward_margin > tie_tolerance
            row["dpo_implicit_reward_ranking_tie"] = abs(reward_margin) <= tie_tolerance
        else:
            row["preferred_dpo_implicit_reward_sum"] = None
            row["dispreferred_dpo_implicit_reward_sum"] = None
            row["dpo_implicit_reward_margin_sum"] = None
            row["dpo_implicit_reward_ranked_preferred"] = None
            row["dpo_implicit_reward_ranking_tie"] = None


def summarize_slice(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float,
    ci_confidence_level: float,
    ci_seed: int,
    question_bootstrap_samples: int,
) -> dict[str, Any]:
    total_probe_count = len(rows)
    total_question_count = len({clean_text(row.get("question_id")) for row in rows if clean_text(row.get("question_id"))})

    scored_rows = [
        row
        for row in rows
        if isinstance(row.get("avg_logprob_margin"), (int, float))
    ]
    scored_question_ids = {
        clean_text(row.get("question_id"))
        for row in scored_rows
        if clean_text(row.get("question_id"))
    }

    preferred_logp_means = [
        float(row["preferred_logp_mean"])
        for row in scored_rows
        if isinstance(row.get("preferred_logp_mean"), (int, float))
    ]
    dispreferred_logp_means = [
        float(row["dispreferred_logp_mean"])
        for row in scored_rows
        if isinstance(row.get("dispreferred_logp_mean"), (int, float))
    ]
    margins = [float(row["avg_logprob_margin"]) for row in scored_rows]
    pair_micro_correct_count = sum(1 for margin in margins if margin > tie_tolerance)
    pair_micro_tie_count = sum(1 for margin in margins if abs(margin) <= tie_tolerance)
    pair_micro_incorrect_count = max(0, len(margins) - pair_micro_correct_count - pair_micro_tie_count)
    pair_micro_accuracy = (
        float(pair_micro_correct_count) / len(margins)
        if margins
        else None
    )
    pair_micro_tie_rate = (
        float(pair_micro_tie_count) / len(margins)
        if margins
        else None
    )

    question_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        question_id = clean_text(row.get("question_id"))
        if question_id:
            question_rows[question_id].append(row)

    question_level_accuracies: list[float] = []
    question_level_tie_rates: list[float] = []
    question_level_margins: list[float] = []
    for question_id, group_rows in sorted(question_rows.items()):
        group_margins = [float(row["avg_logprob_margin"]) for row in group_rows]
        if not group_margins:
            continue
        question_level_accuracies.append(
            sum(1.0 for margin in group_margins if margin > tie_tolerance) / len(group_margins)
        )
        question_level_tie_rates.append(
            sum(1.0 for margin in group_margins if abs(margin) <= tie_tolerance) / len(group_margins)
        )
        question_level_margins.append(sum(group_margins) / len(group_margins))

    prompt_truncation_count = sum(1 for row in scored_rows if row.get("preferred_prompt_truncated") or row.get("dispreferred_prompt_truncated"))
    completion_truncation_count = sum(
        1
        for row in scored_rows
        if row.get("preferred_completion_truncated") or row.get("dispreferred_completion_truncated")
    )
    question_macro_accuracy = (
        sum(question_level_accuracies) / len(question_level_accuracies)
        if question_level_accuracies
        else None
    )
    question_macro_tie_rate = (
        sum(question_level_tie_rates) / len(question_level_tie_rates)
        if question_level_tie_rates
        else None
    )
    question_macro_mean_margin = (
        sum(question_level_margins) / len(question_level_margins)
        if question_level_margins
        else None
    )

    return {
        "probe_count_total": total_probe_count,
        "probe_count_scored": len(scored_rows),
        "probe_count_unscored": total_probe_count - len(scored_rows),
        "question_count_total": total_question_count,
        "question_count_scored": len(scored_question_ids),
        "pair_micro_correct_count": pair_micro_correct_count,
        "pair_micro_incorrect_count": pair_micro_incorrect_count,
        "pair_micro_tie_count": pair_micro_tie_count,
        "pair_micro_accuracy": pair_micro_accuracy,
        "pair_micro_accuracy_ci": wilson_interval(
            pair_micro_correct_count,
            len(margins),
            ci_confidence_level,
        ),
        "pair_micro_tie_rate": pair_micro_tie_rate,
        "question_macro_accuracy": question_macro_accuracy,
        "question_macro_accuracy_ci": bootstrap_mean_interval(
            question_level_accuracies,
            confidence_level=ci_confidence_level,
            bootstrap_samples=question_bootstrap_samples,
            seed=ci_seed,
        ),
        "question_macro_question_count": len(question_level_accuracies),
        "question_macro_tie_rate": question_macro_tie_rate,
        "question_macro_mean_margin": question_macro_mean_margin,
        "avg_logprob_margin_distribution": summarize_numeric(margins),
        "preferred_logp_mean_distribution": summarize_numeric(preferred_logp_means),
        "dispreferred_logp_mean_distribution": summarize_numeric(dispreferred_logp_means),
        "preferred_score_token_count_distribution": summarize_numeric(
            float(row["preferred_score_token_count"])
            for row in scored_rows
            if isinstance(row.get("preferred_score_token_count"), (int, float))
        ),
        "dispreferred_score_token_count_distribution": summarize_numeric(
            float(row["dispreferred_score_token_count"])
            for row in scored_rows
            if isinstance(row.get("dispreferred_score_token_count"), (int, float))
        ),
        "prompt_truncation_count": prompt_truncation_count,
        "completion_truncation_count": completion_truncation_count,
    }


def summarize_by_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
    tie_tolerance: float,
    ci_confidence_level: float,
    ci_seed: int,
    question_bootstrap_samples: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = clean_text(row.get(key_name)) or "unknown"
        grouped[key].append(row)
    return {
        key: summarize_slice(
            group_rows,
            tie_tolerance=tie_tolerance,
            ci_confidence_level=ci_confidence_level,
            ci_seed=ci_seed,
            question_bootstrap_samples=question_bootstrap_samples,
        )
        for key, group_rows in sorted(grouped.items())
    }


def summarize_nested_by_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    outer_key_name: str,
    inner_key_name: str,
    tie_tolerance: float,
    ci_confidence_level: float,
    ci_seed: int,
    question_bootstrap_samples: int,
) -> dict[str, Any]:
    outer_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        outer_key = clean_text(row.get(outer_key_name)) or "unknown"
        outer_grouped[outer_key].append(row)
    return {
        outer_key: summarize_by_key(
            group_rows,
            key_name=inner_key_name,
            tie_tolerance=tie_tolerance,
            ci_confidence_level=ci_confidence_level,
            ci_seed=ci_seed,
            question_bootstrap_samples=question_bootstrap_samples,
        )
        for outer_key, group_rows in sorted(outer_grouped.items())
    }


def build_probe_source_semantic_operation_cells(
    rows: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float,
    ci_confidence_level: float,
    ci_seed: int,
    question_bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                clean_text(row.get("probe_source")) or "unknown",
                clean_text(row.get("semantic_operation")) or "unknown",
            )
        ].append(row)

    cells: list[dict[str, Any]] = []
    for (probe_source, semantic_operation), group_rows in sorted(grouped.items()):
        summary = summarize_slice(
            group_rows,
            tie_tolerance=tie_tolerance,
            ci_confidence_level=ci_confidence_level,
            ci_seed=ci_seed,
            question_bootstrap_samples=question_bootstrap_samples,
        )
        cells.append(
            {
                "probe_source": probe_source,
                "semantic_operation": semantic_operation,
                **summary,
            }
        )
    return cells


def summarize_probe_scores(
    *,
    rows: Sequence[Mapping[str, Any]],
    input_paths: Sequence[str],
    model_ref: str,
    reference_model_ref: str | None,
    build_payload_audit: Mapping[str, Any],
    tie_tolerance: float,
    eos_policy: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    status_counts = Counter(clean_text(row.get("scoring_status")) or "unknown" for row in rows)
    preferred_status_counts = Counter(clean_text(row.get("preferred_scoring_status")) or "unknown" for row in rows)
    dispreferred_status_counts = Counter(clean_text(row.get("dispreferred_scoring_status")) or "unknown" for row in rows)
    probe_source_counts = Counter(clean_text(row.get("probe_source")) or "unknown" for row in rows)
    direction_counts = Counter(clean_text(row.get("direction_label")) or "unknown" for row in rows)
    semantic_operation_counts = Counter(clean_text(row.get("semantic_operation")) or "unknown" for row in rows)
    input_path_counts = Counter(clean_text(row.get("_probe_input_path")) or "unknown" for row in rows)

    has_reference_scores = any(isinstance(row.get("dpo_implicit_reward_margin_sum"), (int, float)) for row in rows)

    summary = {
        "study_name": "Cardinality Shortcut Study",
        "artifact_type": "probe_pair_candidate_ranking_scores",
        "input_paths": [str(resolve_project_path(path_value)) for path_value in input_paths],
        "model_ref": model_ref,
        "reference_model_ref": reference_model_ref,
        "score_definition": {
            "name": "average_answer_token_log_likelihood",
            "mathematical_name": "s_avg",
            "prompt_tokens_excluded": True,
            "padding_tokens_excluded": True,
            "eos_treatment": eos_policy,
            "candidate_prompt_policy": "stored prompt reused exactly for both candidates within each probe",
            "interpretation": "candidate-ranking measurement rather than a literal free-generation choice",
        },
        "runtime_config": {
            "batch_size": int(args.batch_size),
            "max_seq_length": int(args.max_seq_length),
            "margin_tie_tolerance": float(tie_tolerance),
            "eos_policy": eos_policy,
            "ci_confidence_level": float(args.ci_confidence_level),
            "question_bootstrap_samples": int(args.question_bootstrap_samples),
            "ci_seed": int(args.ci_seed),
            "dtype": args.dtype,
            "no_4bit": bool(args.no_4bit),
            "device_map": str(args.device_map),
            "local_files_only": bool(args.local_files_only),
        },
        "build_payload_audit": dict(build_payload_audit),
        "probe_counts_by_input_path": dict(sorted(input_path_counts.items())),
        "probe_counts_by_source": dict(sorted(probe_source_counts.items())),
        "probe_counts_by_direction_label": dict(sorted(direction_counts.items())),
        "probe_counts_by_semantic_operation": dict(sorted(semantic_operation_counts.items())),
        "scoring_status_counts": dict(sorted(status_counts.items())),
        "preferred_scoring_status_counts": dict(sorted(preferred_status_counts.items())),
        "dispreferred_scoring_status_counts": dict(sorted(dispreferred_status_counts.items())),
        "overall": summarize_slice(
            rows,
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "by_probe_source": summarize_by_key(
            rows,
            key_name="probe_source",
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "by_direction_label": summarize_by_key(
            rows,
            key_name="direction_label",
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "by_semantic_operation": summarize_by_key(
            rows,
            key_name="semantic_operation",
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "by_probe_source_and_direction_label": summarize_nested_by_keys(
            rows,
            outer_key_name="probe_source",
            inner_key_name="direction_label",
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "by_probe_source_and_semantic_operation": summarize_nested_by_keys(
            rows,
            outer_key_name="probe_source",
            inner_key_name="semantic_operation",
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "probe_source_semantic_operation_cells": build_probe_source_semantic_operation_cells(
            rows,
            tie_tolerance=tie_tolerance,
            ci_confidence_level=float(args.ci_confidence_level),
            ci_seed=int(args.ci_seed),
            question_bootstrap_samples=int(args.question_bootstrap_samples),
        ),
        "notes": [
            "The score uses average answer-token log-likelihood computed over response tokens only.",
            "The stored probe prompt is reused exactly as saved in the probe bank; prompts are not reconstructed during scoring.",
            "pair_micro_accuracy_ci uses a Wilson interval over scored probe pairs.",
            "question_macro_accuracy_ci uses question-level bootstrap resampling.",
            "Raw DPO implicit-reward margins appear only when --reference-model-ref is provided.",
        ],
    }

    if has_reference_scores:
        dpo_margins = [
            float(row["dpo_implicit_reward_margin_sum"])
            for row in rows
            if isinstance(row.get("dpo_implicit_reward_margin_sum"), (int, float))
        ]
        summary["dpo_implicit_reward_margin_distribution"] = summarize_numeric(dpo_margins)

    return summary


def print_console_summary(summary: Mapping[str, Any]) -> None:
    overall = summary.get("overall", {}) if isinstance(summary.get("overall"), Mapping) else {}
    pair_micro_accuracy = overall.get("pair_micro_accuracy")
    question_macro_accuracy = overall.get("question_macro_accuracy")
    scored = overall.get("probe_count_scored")
    total = overall.get("probe_count_total")
    print(
        "Scored probe pairs: "
        f"{scored}/{total} | pair-micro accuracy={pair_micro_accuracy} | "
        f"question-macro accuracy={question_macro_accuracy}"
    )
    by_direction = summary.get("by_direction_label", {})
    if isinstance(by_direction, Mapping):
        for direction_label, direction_summary in sorted(by_direction.items()):
            if not isinstance(direction_summary, Mapping):
                continue
            print(
                f"  {direction_label}: pair-micro={direction_summary.get('pair_micro_accuracy')} | "
                f"question-macro={direction_summary.get('question_macro_accuracy')} | "
                f"scored={direction_summary.get('probe_count_scored')}/{direction_summary.get('probe_count_total')}"
            )


def main() -> None:
    args = parse_args()

    input_paths = [str(resolve_project_path(path_value)) for path_value in args.probe_input]
    output_jsonl_path = resolve_project_path(str(args.output_jsonl))
    summary_json_path = resolve_project_path(str(args.summary_json))
    requested_model_ref = clean_text(args.model_ref)
    resolved_model_ref = normalize_model_ref(requested_model_ref)
    requested_reference_model_ref = clean_text(args.reference_model_ref)
    resolved_reference_model_ref = normalize_model_ref(requested_reference_model_ref)

    rows = load_probe_rows(args.probe_input, limit=args.limit)
    preferred_payloads, dispreferred_payloads, build_payload_audit = build_probe_payloads(
        rows,
        eos_policy=str(args.eos_policy),
    )

    project_root = PROJECT_ROOT
    prime_unsloth_runtime()

    policy_spec = resolve_single_model_spec(
        model_ref=resolved_model_ref,
        args=args,
        project_root=project_root,
    )
    policy_model, policy_tokenizer = load_model_and_tokenizer_for_eval(policy_spec, args)
    try:
        preferred_policy_results = score_payloads_with_model(
            payloads=preferred_payloads,
            model=policy_model,
            tokenizer=policy_tokenizer,
            args=args,
            progress_label=f"{policy_spec.label}:preferred",
        )
        dispreferred_policy_results = score_payloads_with_model(
            payloads=dispreferred_payloads,
            model=policy_model,
            tokenizer=policy_tokenizer,
            args=args,
            progress_label=f"{policy_spec.label}:dispreferred",
        )
    finally:
        release_model(policy_model, policy_tokenizer)

    preferred_reference_results = None
    dispreferred_reference_results = None
    if resolved_reference_model_ref:
        reference_spec = resolve_single_model_spec(
            model_ref=resolved_reference_model_ref,
            args=args,
            project_root=project_root,
        )
        reference_model, reference_tokenizer = load_model_and_tokenizer_for_eval(reference_spec, args)
        try:
            preferred_reference_results = score_payloads_with_model(
                payloads=preferred_payloads,
                model=reference_model,
                tokenizer=reference_tokenizer,
                args=args,
                progress_label=f"{reference_spec.label}:preferred-ref",
            )
            dispreferred_reference_results = score_payloads_with_model(
                payloads=dispreferred_payloads,
                model=reference_model,
                tokenizer=reference_tokenizer,
                args=args,
                progress_label=f"{reference_spec.label}:dispreferred-ref",
            )
        finally:
            release_model(reference_model, reference_tokenizer)

    scored_rows = [dict(row) for row in rows]
    attach_probe_scores(
        rows=scored_rows,
        preferred_policy_results=preferred_policy_results,
        dispreferred_policy_results=dispreferred_policy_results,
        preferred_reference_results=preferred_reference_results,
        dispreferred_reference_results=dispreferred_reference_results,
        model_ref=requested_model_ref or str(policy_spec.ref),
        reference_model_ref=requested_reference_model_ref or None,
        tie_tolerance=float(args.margin_tie_tolerance),
        eos_policy=str(args.eos_policy),
    )

    summary = summarize_probe_scores(
        rows=scored_rows,
        input_paths=input_paths,
        model_ref=requested_model_ref or str(policy_spec.ref),
        reference_model_ref=requested_reference_model_ref or None,
        build_payload_audit=build_payload_audit,
        tie_tolerance=float(args.margin_tie_tolerance),
        eos_policy=str(args.eos_policy),
        args=args,
    )

    write_jsonl(output_jsonl_path, scored_rows)
    write_json(summary_json_path, summary)
    print_console_summary(summary)
    print(f"Wrote per-probe scores to {output_jsonl_path}")
    print(f"Wrote summary to {summary_json_path}")


if __name__ == "__main__":
    main()
