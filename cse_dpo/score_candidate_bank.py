from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_registry import get_project_root
from src.utility.data import clean_text
from src.utility.eval_models import (
    first_model_device,
    load_model_and_tokenizer_for_eval,
    prime_unsloth_runtime,
    resolve_model_specs,
)

from .common import finite_or_none, load_json_records, summarize_numeric, write_json, write_jsonl
from .normalize_set_answers import parse_list_output, serialize_list_items


@dataclass(frozen=True)
class ScoringPayload:
    row_index: int
    prompt: str
    completion_text: str
    append_eos: bool = False


@dataclass(frozen=True)
class PreparedSequence:
    row_index: int
    input_ids: list[int]
    completion_start: int
    prompt_token_count: int
    effective_prompt_token_count: int
    completion_token_count: int
    effective_completion_token_count: int
    prompt_truncated: bool
    completion_truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score candidate-bank responses with completion log-probabilities so they can be "
            "used for DCRM-style preference-pair construction."
        )
    )
    parser.add_argument(
        "--candidate-input",
        nargs="+",
        required=True,
        help="Candidate-bank JSONL or JSON files containing prompt and raw_output fields.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where the scored candidate-bank JSONL will be written.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Where the scoring summary JSON will be written.",
    )
    parser.add_argument(
        "--model-ref",
        required=True,
        help="Policy/scoring model reference used for completion log-probability scoring.",
    )
    parser.add_argument(
        "--reference-model-ref",
        default=None,
        help=(
            "Optional reference model used to compute DCRM-style score differences "
            "(policy log-prob minus reference log-prob)."
        ),
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry path used to resolve model refs.",
    )
    parser.add_argument(
        "--completion-source",
        choices=["raw", "parsed", "parsed_or_raw"],
        default="parsed_or_raw",
        help=(
            "What text to score as the completion. 'parsed' canonicalizes parsed list items, "
            "'raw' uses the raw model output, and 'parsed_or_raw' falls back to raw output when "
            "no parsed items are available."
        ),
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing when rebuilding parsed completions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of candidate completions scored per forward pass. Keep this small for long prompts.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help=(
            "Maximum prompt+completion sequence length used for scoring. Prompts are left-truncated "
            "first to preserve the completion span."
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
        default=250,
        help="Print progress after this many scored rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--empty-cuda-cache-steps",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N scoring batches. Use 0 to disable.",
    )
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_single_model_spec(
    *,
    model_ref: str,
    args: argparse.Namespace,
    project_root: Path,
) -> Any:
    resolution_args = argparse.Namespace(
        model_ref=[model_ref],
        all_registry_runs=False,
        registry_path=args.registry_path,
        chat_template=args.chat_template,
        prompt_format=args.prompt_format,
    )
    specs = resolve_model_specs(resolution_args, project_root=project_root)
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one resolved model for {model_ref}, found {len(specs)}.")
    return specs[0]


def load_candidate_rows(paths: Sequence[str], limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        for row in load_json_records(path):
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def canonical_completion_from_row(row: Mapping[str, Any], allow_fallback_split: bool) -> str:
    parsed_items_value = row.get("parsed_items")
    parsed_items: list[str] = []
    if isinstance(parsed_items_value, Sequence) and not isinstance(parsed_items_value, (str, bytes)):
        parsed_items = [clean_text(item) for item in parsed_items_value if clean_text(item)]
    if not parsed_items:
        raw_output = clean_text(row.get("raw_output") or row.get("prediction"))
        if raw_output:
            parsed = parse_list_output(raw_output, allow_fallback_split=allow_fallback_split)
            parsed_items = [clean_text(item) for item in parsed.items if clean_text(item)]
    return serialize_list_items(parsed_items)


def completion_text_for_row(row: Mapping[str, Any], args: argparse.Namespace) -> str:
    raw_output = clean_text(row.get("raw_output") or row.get("prediction"))
    parsed_text = canonical_completion_from_row(row, allow_fallback_split=bool(args.allow_fallback_split))
    completion_source = str(args.completion_source)
    if completion_source == "raw":
        return raw_output
    if completion_source == "parsed":
        return parsed_text
    return parsed_text or raw_output


def build_scoring_payloads(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> tuple[list[ScoringPayload], dict[str, int]]:
    payloads: list[ScoringPayload] = []
    counts = {
        "missing_prompt": 0,
        "empty_completion": 0,
        "ready": 0,
    }
    for row_index, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        if not prompt:
            counts["missing_prompt"] += 1
            continue
        completion_text = completion_text_for_row(row, args)
        if not completion_text:
            counts["empty_completion"] += 1
            continue
        payloads.append(
            ScoringPayload(
                row_index=row_index,
                prompt=prompt,
                completion_text=completion_text,
            )
        )
        counts["ready"] += 1
    return payloads, counts


def _tokenize_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    tokenized = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = tokenized.get("input_ids")
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], int):
        return [int(token_id) for token_id in input_ids]
    if hasattr(input_ids, "tolist"):
        payload = input_ids.tolist()
        if isinstance(payload, list) and payload and isinstance(payload[0], int):
            return [int(token_id) for token_id in payload]
    raise ValueError("Tokenizer did not return a flat input_ids sequence for scoring.")


def prepare_sequence(
    *,
    tokenizer: Any,
    payload: ScoringPayload,
    max_seq_length: int,
) -> PreparedSequence | None:
    prompt_ids = _tokenize_ids(tokenizer, payload.prompt, add_special_tokens=True)
    completion_ids = _tokenize_ids(tokenizer, payload.completion_text, add_special_tokens=False)
    if payload.append_eos:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (not completion_ids or int(completion_ids[-1]) != int(eos_token_id)):
            completion_ids = [*completion_ids, int(eos_token_id)]
    if not prompt_ids or not completion_ids:
        return None

    original_prompt_token_count = len(prompt_ids)
    original_completion_token_count = len(completion_ids)
    prompt_truncated = False
    completion_truncated = False

    if max_seq_length > 0:
        reserved_prompt_tokens = 1 if prompt_ids else 0
        max_completion_tokens = max(1, max_seq_length - reserved_prompt_tokens)
        if len(completion_ids) > max_completion_tokens:
            completion_ids = completion_ids[:max_completion_tokens]
            completion_truncated = True
        available_prompt_tokens = max(0, max_seq_length - len(completion_ids))
        if len(prompt_ids) > available_prompt_tokens:
            prompt_ids = prompt_ids[-available_prompt_tokens:] if available_prompt_tokens > 0 else []
            prompt_truncated = True

    if not prompt_ids or not completion_ids:
        return None

    return PreparedSequence(
        row_index=payload.row_index,
        input_ids=[*prompt_ids, *completion_ids],
        completion_start=len(prompt_ids),
        prompt_token_count=original_prompt_token_count,
        effective_prompt_token_count=len(prompt_ids),
        completion_token_count=original_completion_token_count,
        effective_completion_token_count=len(completion_ids),
        prompt_truncated=prompt_truncated,
        completion_truncated=completion_truncated,
    )


def score_prepared_batch(
    *,
    model: Any,
    tokenizer: Any,
    prepared_batch: Sequence[PreparedSequence],
) -> list[dict[str, Any]]:
    import torch

    if not prepared_batch:
        return []

    device = first_model_device(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    max_width = max(len(item.input_ids) for item in prepared_batch)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    completion_masks: list[list[float]] = []
    for item in prepared_batch:
        pad_width = max_width - len(item.input_ids)
        input_rows.append(item.input_ids + ([pad_token_id] * pad_width))
        attention_rows.append(([1] * len(item.input_ids)) + ([0] * pad_width))
        target_mask = [0.0] * max(0, max_width - 1)
        start_index = max(0, item.completion_start - 1)
        for position in range(start_index, len(item.input_ids) - 1):
            target_mask[position] = 1.0
        completion_masks.append(target_mask)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
    completion_mask = torch.tensor(completion_masks, dtype=torch.float32, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        selected_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_log_probs = selected_logits - torch.logsumexp(logits, dim=-1)
        masked_token_log_probs = token_log_probs * completion_mask
        score_sums = masked_token_log_probs.sum(dim=-1)
        score_counts = completion_mask.sum(dim=-1)
        score_means = score_sums / score_counts.clamp_min(1.0)

    results: list[dict[str, Any]] = []
    for batch_index, item in enumerate(prepared_batch):
        results.append(
            {
                "row_index": item.row_index,
                "logp_sum": finite_or_none(float(score_sums[batch_index].item())),
                "logp_mean": finite_or_none(float(score_means[batch_index].item())),
                "score_token_count": int(score_counts[batch_index].item()),
                "prompt_token_count": int(item.prompt_token_count),
                "effective_prompt_token_count": int(item.effective_prompt_token_count),
                "completion_token_count": int(item.completion_token_count),
                "effective_completion_token_count": int(item.effective_completion_token_count),
                "prompt_truncated": bool(item.prompt_truncated),
                "completion_truncated": bool(item.completion_truncated),
                "scored_input_token_count": len(item.input_ids),
            }
        )

    del outputs
    del logits
    del target_ids
    del selected_logits
    del token_log_probs
    del masked_token_log_probs
    del score_sums
    del score_counts
    del score_means
    del input_ids
    del attention_mask
    del completion_mask

    return results


def score_payloads_with_model(
    *,
    payloads: Sequence[ScoringPayload],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    progress_label: str,
) -> list[dict[str, Any]]:
    import torch

    results_by_row_index: dict[int, dict[str, Any]] = {}
    prepared_batch: list[PreparedSequence] = []
    scored_batches = 0
    for payload in payloads:
        prepared = prepare_sequence(
            tokenizer=tokenizer,
            payload=payload,
            max_seq_length=int(args.max_seq_length),
        )
        if prepared is None:
            results_by_row_index[payload.row_index] = {
                "row_index": payload.row_index,
                "logp_sum": None,
                "logp_mean": None,
                "score_token_count": 0,
                "prompt_token_count": 0,
                "effective_prompt_token_count": 0,
                "completion_token_count": 0,
                "effective_completion_token_count": 0,
                "prompt_truncated": False,
                "completion_truncated": False,
                "scored_input_token_count": 0,
                "status": "unscorable_sequence",
            }
            continue

        prepared_batch.append(prepared)
        if len(prepared_batch) < int(args.batch_size):
            continue

        for result in score_prepared_batch(model=model, tokenizer=tokenizer, prepared_batch=prepared_batch):
            results_by_row_index[result["row_index"]] = result
        prepared_batch = []
        scored_batches += 1
        if int(args.empty_cuda_cache_steps) > 0 and scored_batches % int(args.empty_cuda_cache_steps) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if int(args.verbose_every) > 0:
            scored_rows = len(results_by_row_index)
            if scored_rows % int(args.verbose_every) == 0:
                print(f"[{progress_label}] scored {scored_rows:,} / {len(payloads):,} candidate completions", flush=True)

    if prepared_batch:
        for result in score_prepared_batch(model=model, tokenizer=tokenizer, prepared_batch=prepared_batch):
            results_by_row_index[result["row_index"]] = result

    ordered_results: list[dict[str, Any]] = []
    for payload in payloads:
        ordered_results.append(
            results_by_row_index.get(
                payload.row_index,
                {
                    "row_index": payload.row_index,
                    "logp_sum": None,
                    "logp_mean": None,
                    "score_token_count": 0,
                    "prompt_token_count": 0,
                    "effective_prompt_token_count": 0,
                    "completion_token_count": 0,
                    "effective_completion_token_count": 0,
                    "prompt_truncated": False,
                    "completion_truncated": False,
                    "scored_input_token_count": 0,
                    "status": "missing_result",
                },
            )
        )
    return ordered_results


def release_model(model: Any, tokenizer: Any) -> None:
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def attach_scores_to_rows(
    *,
    rows: list[dict[str, Any]],
    payloads: Sequence[ScoringPayload],
    policy_results: Sequence[Mapping[str, Any]],
    reference_results: Sequence[Mapping[str, Any]] | None,
    policy_model_ref: str,
    reference_model_ref: str | None,
    completion_source: str,
) -> None:
    policy_by_row = {int(result["row_index"]): dict(result) for result in policy_results}
    reference_by_row = (
        {int(result["row_index"]): dict(result) for result in reference_results}
        if reference_results is not None
        else {}
    )
    completion_text_by_row = {payload.row_index: payload.completion_text for payload in payloads}

    for row_index, row in enumerate(rows):
        row["scoring_completion_source"] = completion_source
        row["scoring_completion_text"] = completion_text_by_row.get(row_index, "")

        policy = policy_by_row.get(row_index)
        if policy is None:
            row["policy_logp_sum"] = None
            row["policy_logp_mean"] = None
            row["scoring_status"] = "unscored"
            continue

        row["policy_scoring_model_ref"] = policy_model_ref
        row["policy_logp_sum"] = policy.get("logp_sum")
        row["policy_logp_mean"] = policy.get("logp_mean")
        row["scoring_prompt_token_count"] = int(policy.get("prompt_token_count", 0) or 0)
        row["scoring_effective_prompt_token_count"] = int(policy.get("effective_prompt_token_count", 0) or 0)
        row["scoring_completion_token_count"] = int(policy.get("completion_token_count", 0) or 0)
        row["scoring_effective_completion_token_count"] = int(policy.get("effective_completion_token_count", 0) or 0)
        row["scoring_input_token_count"] = int(policy.get("scored_input_token_count", 0) or 0)
        row["scoring_score_token_count"] = int(policy.get("score_token_count", 0) or 0)
        row["scoring_prompt_truncated"] = bool(policy.get("prompt_truncated", False))
        row["scoring_completion_truncated"] = bool(policy.get("completion_truncated", False))
        row["scoring_status"] = str(policy.get("status") or "scored")

        reference = reference_by_row.get(row_index)
        if reference_model_ref is not None:
            row["reference_scoring_model_ref"] = reference_model_ref
        if reference is not None:
            row["reference_logp_sum"] = reference.get("logp_sum")
            row["reference_logp_mean"] = reference.get("logp_mean")
            if isinstance(row.get("policy_logp_sum"), (int, float)) and isinstance(reference.get("logp_sum"), (int, float)):
                row["dcrm_score_sum"] = float(row["policy_logp_sum"]) - float(reference["logp_sum"])
            else:
                row["dcrm_score_sum"] = None
            if isinstance(row.get("policy_logp_mean"), (int, float)) and isinstance(reference.get("logp_mean"), (int, float)):
                row["dcrm_score"] = float(row["policy_logp_mean"]) - float(reference["logp_mean"])
            else:
                row["dcrm_score"] = None
        elif reference_model_ref is not None:
            row["reference_logp_sum"] = None
            row["reference_logp_mean"] = None
            row["dcrm_score_sum"] = None
            row["dcrm_score"] = None


def summarize_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    payload_counts: Mapping[str, int],
    policy_model_ref: str,
    reference_model_ref: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy_means = [float(row["policy_logp_mean"]) for row in rows if isinstance(row.get("policy_logp_mean"), (int, float))]
    policy_sums = [float(row["policy_logp_sum"]) for row in rows if isinstance(row.get("policy_logp_sum"), (int, float))]
    reference_means = [
        float(row["reference_logp_mean"])
        for row in rows
        if isinstance(row.get("reference_logp_mean"), (int, float))
    ]
    dcrm_scores = [float(row["dcrm_score"]) for row in rows if isinstance(row.get("dcrm_score"), (int, float))]
    scoring_status_counts: dict[str, int] = {}
    for row in rows:
        status = clean_text(row.get("scoring_status")) or "unknown"
        scoring_status_counts[status] = scoring_status_counts.get(status, 0) + 1

    prompt_truncation_count = sum(1 for row in rows if bool(row.get("scoring_prompt_truncated")))
    completion_truncation_count = sum(1 for row in rows if bool(row.get("scoring_completion_truncated")))

    return {
        "candidate_input": list(args.candidate_input),
        "output_jsonl": args.output_jsonl,
        "policy_model_ref": policy_model_ref,
        "reference_model_ref": reference_model_ref,
        "completion_source": str(args.completion_source),
        "row_count": len(rows),
        "payload_counts": dict(payload_counts),
        "scoring_status_counts": dict(sorted(scoring_status_counts.items())),
        "max_seq_length": int(args.max_seq_length),
        "batch_size": int(args.batch_size),
        "prompt_truncation_count": prompt_truncation_count,
        "completion_truncation_count": completion_truncation_count,
        "policy_logp_mean_distribution": summarize_numeric(policy_means),
        "policy_logp_sum_distribution": summarize_numeric(policy_sums),
        "reference_logp_mean_distribution": summarize_numeric(reference_means),
        "dcrm_score_distribution": summarize_numeric(dcrm_scores),
    }


def main() -> None:
    args = parse_args()
    prime_unsloth_runtime()
    project_root = get_project_root()

    rows = load_candidate_rows(args.candidate_input, limit=args.limit)
    if not rows:
        raise ValueError("No candidate rows were loaded from --candidate-input.")

    payloads, payload_counts = build_scoring_payloads(rows, args)
    if not payloads:
        raise ValueError(
            "No candidate rows were scorable. Check whether prompt/raw_output fields are present "
            "and whether --completion-source leaves any non-empty completions."
        )

    policy_model_spec = resolve_single_model_spec(
        model_ref=str(args.model_ref),
        args=args,
        project_root=project_root,
    )
    policy_model, policy_tokenizer = load_model_and_tokenizer_for_eval(policy_model_spec, args)
    policy_results = score_payloads_with_model(
        payloads=payloads,
        model=policy_model,
        tokenizer=policy_tokenizer,
        args=args,
        progress_label="policy",
    )
    release_model(policy_model, policy_tokenizer)

    reference_results: list[dict[str, Any]] | None = None
    if clean_text(args.reference_model_ref):
        reference_model_spec = resolve_single_model_spec(
            model_ref=str(args.reference_model_ref),
            args=args,
            project_root=project_root,
        )
        reference_model, reference_tokenizer = load_model_and_tokenizer_for_eval(reference_model_spec, args)
        reference_results = score_payloads_with_model(
            payloads=payloads,
            model=reference_model,
            tokenizer=reference_tokenizer,
            args=args,
            progress_label="reference",
        )
        release_model(reference_model, reference_tokenizer)

    attach_scores_to_rows(
        rows=rows,
        payloads=payloads,
        policy_results=policy_results,
        reference_results=reference_results,
        policy_model_ref=str(policy_model_spec.ref),
        reference_model_ref=clean_text(args.reference_model_ref) or None,
        completion_source=str(args.completion_source),
    )

    write_jsonl(Path(args.output_jsonl), rows)
    summary = summarize_rows(
        rows=rows,
        payload_counts=payload_counts,
        policy_model_ref=str(policy_model_spec.ref),
        reference_model_ref=clean_text(args.reference_model_ref) or None,
        args=args,
    )
    write_json(Path(args.summary_json), summary)

    print(f"Wrote {len(rows):,} scored candidate rows to {args.output_jsonl}")
    print(f"Wrote scoring summary to {args.summary_json}")


if __name__ == "__main__":
    main()
