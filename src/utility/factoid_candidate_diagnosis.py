from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import summarize_numeric, write_json, write_jsonl
from cse_dpo.score_candidate_bank import (
    ScoringPayload,
    release_model,
    resolve_single_model_spec,
    score_payloads_with_model,
)
from src.model_registry import get_project_root, resolve_repo_path, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle
from src.utility.bioasq_format import (
    exact_answer_groups,
    match_to_gold_group,
    normalize_for_match,
)
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import clean_text
from src.utility.bioasq_official import evaluate_with_bioasq_java
from src.utility.eval_dataset import load_eval_examples, render_prompt, resolve_eval_input_paths
from src.utility.eval_models import load_model_and_tokenizer_for_eval, prime_unsloth_runtime
from src.utility.eval_types import EvalExample
from src.utility.factoid_output_parsing import (
    clean_factoid_candidate_text,
    parse_factoid_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose factoid candidate discovery versus candidate ranking from saved "
            "validation predictions. The script reconstructs prompts, deduplicates "
            "candidate entities, scores each candidate with answer-token log-probability, "
            "and reports discovery/ranking summaries."
        )
    )
    parser.add_argument("--prediction-json", required=True, help="Saved predictions.json from evaluate_models.py.")
    parser.add_argument("--output-dir", required=True, help="Directory where diagnosis artifacts will be written.")
    parser.add_argument("--model-ref", required=True, help="Model reference used to score/rerank candidates.")
    parser.add_argument(
        "--candidate-extraction-mode",
        required=True,
        choices=["single_answer", "top_five"],
        help="How to interpret the saved generations when extracting candidate entities.",
    )
    parser.add_argument(
        "--single-answer-policy",
        choices=["permissive", "strict", "first_entity"],
        default="permissive",
        help=(
            "How to treat outputs in S1/S3 single-answer settings. 'permissive' keeps every "
            "parsed entity, 'strict' keeps outputs only when exactly one entity is parsed, "
            "and 'first_entity' keeps only the first parsed entity."
        ),
    )
    parser.add_argument(
        "--factoid-parser-mode",
        choices=["current", "agnostic"],
        default="current",
        help=(
            "How to parse factoid outputs before candidate construction. 'current' reproduces "
            "the existing tagged-first parser, while 'agnostic' adds numbered-list fallback parsing."
        ),
    )
    parser.add_argument(
        "--ranker",
        nargs="+",
        default=["union", "logp_mean", "logp_sum", "frequency", "generation_order"],
        choices=["union", "logp_mean", "logp_sum", "frequency", "generation_order"],
        help="Candidate ranking functions to summarize.",
    )
    parser.add_argument(
        "--official-ranker",
        nargs="+",
        default=[],
        choices=["union", "logp_mean", "logp_sum", "frequency", "generation_order"],
        help=(
            "Rerankers to materialize as new top-k predictions and evaluate with the "
            "official BioASQ evaluator."
        ),
    )
    parser.add_argument("--keep-top-k", type=int, default=5, help="How many reranked candidates to retain for top-k summaries.")
    parser.add_argument("--max-recall-k", type=int, default=5, help="Maximum K used in Recall@K summaries.")
    parser.add_argument(
        "--eos-policy",
        choices=["excluded", "included"],
        default="excluded",
        help="Whether to append one EOS token when scoring candidate completions.",
    )
    parser.add_argument(
        "--eval-input",
        nargs="+",
        required=True,
        help="Evaluation files used to create the saved predictions.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["factoid"],
        help="Question types to load from the evaluation inputs.",
    )
    parser.add_argument("--max-resources", type=int, default=5)
    parser.add_argument("--max-resource-chars", type=int, default=1200)
    parser.add_argument(
        "--resource-selection",
        default="first",
        choices=["first", "embedding"],
    )
    parser.add_argument(
        "--resource-granularity",
        default="document",
        choices=["document", "snippet"],
    )
    parser.add_argument(
        "--resource-window-mode",
        default="single",
        choices=["single", "sequential"],
    )
    parser.add_argument("--resource-window-step", type=int, default=0)
    parser.add_argument("--resource-reranker-model", default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--resource-reranker-article-model", default=None)
    parser.add_argument("--resource-reranker-device", default="auto")
    parser.add_argument("--resource-reranker-batch-size", type=int, default=32)
    parser.add_argument("--max-summary-answers", type=int, default=1)
    parser.add_argument("--max-factoid-answers", type=int, default=5)
    parser.add_argument("--max-list-items", type=int, default=100)
    parser.add_argument(
        "--summary-reference-mode",
        default="first",
        choices=["first", "all-max", "all-mean"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--registry-path", default="models/registry.json")
    parser.add_argument("--prompt-registry-path", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None, choices=["chat", "unitor_plain"])
    parser.add_argument(
        "--bioasq-java-jar",
        default="third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
    )
    parser.add_argument(
        "--bioasq-java-version",
        type=int,
        default=5,
        choices=[2, 3, 5, 8, 9],
    )
    parser.add_argument("--bioasq-java-heap", default="4G")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--verbose-every", type=int, default=100)
    parser.add_argument("--empty-cuda-cache-steps", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = normalize_for_match(cleaned) or cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(cleaned)
    return items


def classify_single_answer_status(parsed_items: Sequence[str]) -> tuple[str, bool, bool]:
    item_count = len(parsed_items)
    if item_count <= 0:
        return "empty", False, True
    if item_count == 1:
        return "single_item", True, False
    return "multi_item", False, True


def extract_candidates_from_output(
    text: str,
    *,
    mode: str,
    parser_mode: str,
    single_answer_policy: str,
) -> dict[str, Any]:
    parsed_items = parse_factoid_candidates(text, parser_mode=parser_mode)

    if mode == "single_answer":
        status, strict_valid, protocol_violation = classify_single_answer_status(parsed_items)
        selected_items = list(parsed_items)
        if single_answer_policy == "strict":
            selected_items = list(parsed_items) if len(parsed_items) == 1 else []
        elif single_answer_policy == "first_entity":
            selected_items = list(parsed_items[:1])
        elif single_answer_policy != "permissive":
            raise ValueError(f"Unsupported single-answer policy: {single_answer_policy}")
        return {
            "items": selected_items,
            "parsed_items": list(parsed_items),
            "status": status,
            "strict_valid": strict_valid,
            "protocol_violation": protocol_violation,
        }

    status = "nonempty" if parsed_items else "empty"
    return {
        "items": list(parsed_items),
        "parsed_items": list(parsed_items),
        "status": status,
        "strict_valid": bool(parsed_items),
        "protocol_violation": False,
    }


def first_gold_rank(candidate_texts: Sequence[str], gold_groups: Sequence[Sequence[str]]) -> int | None:
    for rank, candidate in enumerate(candidate_texts, start=1):
        if any(match_to_gold_group(candidate, gold_group) for gold_group in gold_groups):
            return rank
    return None


def direct_rank_from_prediction(
    prediction_text: str,
    gold_groups: Sequence[Sequence[str]],
    *,
    parser_mode: str,
    candidate_extraction_mode: str,
    single_answer_policy: str,
) -> int | None:
    extracted = extract_candidates_from_output(
        prediction_text,
        mode=candidate_extraction_mode,
        parser_mode=parser_mode,
        single_answer_policy=single_answer_policy,
    )
    candidates = [clean_factoid_candidate_text(item) for item in extracted.get("items", []) if clean_factoid_candidate_text(item)]
    return first_gold_rank(candidates, gold_groups)


def question_rankings(
    candidates: Sequence[Mapping[str, Any]],
    ranker_name: str,
) -> list[dict[str, Any]]:
    records = [dict(candidate) for candidate in candidates]
    def numeric_score(row: Mapping[str, Any], field_name: str) -> float:
        value = row.get(field_name)
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    if ranker_name == "logp_mean":
        return sorted(
            records,
            key=lambda row: (
                0 if row.get("policy_logp_mean") is None else 1,
                numeric_score(row, "policy_logp_mean"),
                int(row.get("occurrence_count") or 0),
                -int(row.get("first_seen_observation_order") or 0),
                str(row.get("normalized_candidate") or ""),
            ),
            reverse=True,
        )
    if ranker_name == "logp_sum":
        return sorted(
            records,
            key=lambda row: (
                0 if row.get("policy_logp_sum") is None else 1,
                numeric_score(row, "policy_logp_sum"),
                int(row.get("occurrence_count") or 0),
                -int(row.get("first_seen_observation_order") or 0),
                str(row.get("normalized_candidate") or ""),
            ),
            reverse=True,
        )
    if ranker_name == "frequency":
        return sorted(
            records,
            key=lambda row: (
                int(row.get("occurrence_count") or 0),
                0 if row.get("policy_logp_mean") is None else 1,
                numeric_score(row, "policy_logp_mean"),
                -int(row.get("first_seen_observation_order") or 0),
                str(row.get("normalized_candidate") or ""),
            ),
            reverse=True,
        )
    if ranker_name == "generation_order":
        return sorted(
            records,
            key=lambda row: (
                int(row.get("first_seen_observation_order") or 0),
                int(row.get("first_seen_output_rank") or 0),
                str(row.get("normalized_candidate") or ""),
            ),
        )
    raise KeyError(f"Unsupported ranker: {ranker_name}")


def union_ranking_from_prediction(
    prediction_text: str,
    *,
    gold_groups: Sequence[Sequence[str]],
    candidate_rows: Sequence[Mapping[str, Any]],
    parser_mode: str,
    candidate_extraction_mode: str,
    single_answer_policy: str,
    keep_top_k: int,
) -> dict[str, Any]:
    extracted = extract_candidates_from_output(
        prediction_text,
        mode=candidate_extraction_mode,
        parser_mode=parser_mode,
        single_answer_policy=single_answer_policy,
    )
    ordered_items = [
        clean_factoid_candidate_text(item)
        for item in extracted.get("items", [])
        if clean_factoid_candidate_text(item)
    ]
    ordered_items = dedupe_preserve_order(ordered_items)
    candidate_by_key = {
        normalize_for_match(str(candidate.get("candidate_text") or "")): candidate
        for candidate in candidate_rows
        if normalize_for_match(str(candidate.get("candidate_text") or ""))
    }
    top_candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered_items[:keep_top_k], start=1):
        key = normalize_for_match(item)
        matched_candidate = candidate_by_key.get(key or "")
        top_candidates.append(
            {
                "rank": rank,
                "candidate_text": item,
                "gold_match": any(match_to_gold_group(item, gold_group) for gold_group in gold_groups),
                "occurrence_count": int(matched_candidate.get("occurrence_count") or 0) if matched_candidate else 0,
                "policy_logp_mean": matched_candidate.get("policy_logp_mean") if matched_candidate else None,
                "policy_logp_sum": matched_candidate.get("policy_logp_sum") if matched_candidate else None,
            }
        )
    return {
        "gold_rank": first_gold_rank(ordered_items, gold_groups),
        "top_candidates": top_candidates,
    }


def ranking_summary(
    question_rows: Sequence[Mapping[str, Any]],
    *,
    ranker_name: str,
    keep_top_k: int,
    max_recall_k: int,
) -> dict[str, Any]:
    question_count = len(question_rows)
    rank1_count = 0
    rank2_to_k_count = 0
    below_k_but_present_count = 0
    absent_from_pool_count = 0
    recall_hits = {k: 0 for k in range(1, max_recall_k + 1)}
    reciprocal_ranks: list[float] = []

    for row in question_rows:
        ranking = row["rankings"][ranker_name]
        gold_rank = ranking.get("gold_rank")
        if gold_rank is None:
            absent_from_pool_count += 1
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(1.0 / float(gold_rank))
        if gold_rank == 1:
            rank1_count += 1
        elif gold_rank <= keep_top_k:
            rank2_to_k_count += 1
        else:
            below_k_but_present_count += 1
        for k in range(1, max_recall_k + 1):
            if gold_rank <= k:
                recall_hits[k] += 1

    return {
        "question_count": question_count,
        "gold_at_rank_1_count": rank1_count,
        "gold_at_rank_1_rate": (rank1_count / question_count) if question_count else 0.0,
        "gold_at_ranks_2_to_k_count": rank2_to_k_count,
        "gold_at_ranks_2_to_k_rate": (rank2_to_k_count / question_count) if question_count else 0.0,
        "gold_below_top_k_but_present_count": below_k_but_present_count,
        "gold_below_top_k_but_present_rate": (below_k_but_present_count / question_count) if question_count else 0.0,
        "gold_absent_from_pool_count": absent_from_pool_count,
        "gold_absent_from_pool_rate": (absent_from_pool_count / question_count) if question_count else 0.0,
        "mrr": (sum(reciprocal_ranks) / question_count) if question_count else 0.0,
        "recall_at_k": {
            str(k): (recall_hits[k] / question_count) if question_count else 0.0
            for k in range(1, max_recall_k + 1)
        },
    }


def submission_aligned_summary(
    official_payload: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = dict(official_payload.get("aggregate") or {})
    by_type = dict(aggregate.get("by_type") or {})
    factoid = dict(by_type.get("factoid") or {})
    return {
        "source": "official_saved_prediction",
        "question_count": int(factoid.get("question_count") or aggregate.get("question_count") or 0),
        "primary_metric": factoid.get("primary_metric") or "mrr",
        "metrics": dict(factoid.get("metrics") or {}),
        "note": (
            "Use these official BioASQ metrics as the submission-aligned headline results. "
            "The ranking, candidate_pool, and output_format sections remain parser-based diagnostics."
        ),
    }


def build_examples(
    *,
    args: argparse.Namespace,
    project_root: Path,
) -> tuple[dict[str, EvalExample], Mapping[str, Any], list[Path]]:
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
        raise ValueError("No evaluation examples were loaded for candidate diagnosis.")
    examples_by_id = {clean_text(example.question_id): example for example in examples}
    return examples_by_id, prompt_bundle, eval_paths


def inference_budget_summary(prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generation_counts: list[int] = []
    prompt_token_counts: list[float] = []
    effective_prompt_token_counts: list[float] = []
    truncated_generation_count = 0
    truncated_question_count = 0

    for row in prediction_rows:
        samples = row.get("generation_samples")
        if isinstance(samples, list):
            generation_counts.append(len(samples))
        else:
            generation_counts.append(1)
        prompt_truncation = row.get("prompt_truncation") or {}
        if bool(prompt_truncation.get("prompt_truncated")):
            truncated_question_count += 1
        telemetry = row.get("generation_telemetry")
        if not isinstance(telemetry, list):
            continue
        for entry in telemetry:
            if not isinstance(entry, Mapping):
                continue
            if isinstance(entry.get("prompt_token_count"), (int, float)):
                prompt_token_counts.append(float(entry["prompt_token_count"]))
            if isinstance(entry.get("effective_prompt_token_count"), (int, float)):
                effective_prompt_token_counts.append(float(entry["effective_prompt_token_count"]))
            if bool(entry.get("prompt_truncated")):
                truncated_generation_count += 1

    return {
        "question_count": len(prediction_rows),
        "total_generation_calls": sum(generation_counts),
        "generation_count_distribution": summarize_numeric(generation_counts),
        "prompt_token_count_distribution": summarize_numeric(prompt_token_counts),
        "effective_prompt_token_count_distribution": summarize_numeric(effective_prompt_token_counts),
        "truncated_generation_count": truncated_generation_count,
        "truncated_question_count": truncated_question_count,
    }


def attach_scoring(
    *,
    candidate_rows: list[dict[str, Any]],
    payloads: Sequence[ScoringPayload],
    results: Sequence[Mapping[str, Any]],
) -> None:
    results_by_index = {int(result["row_index"]): dict(result) for result in results}
    completion_text_by_index = {payload.row_index: payload.completion_text for payload in payloads}
    for row_index, row in enumerate(candidate_rows):
        row["scoring_completion_text"] = completion_text_by_index.get(row_index, "")
        result = results_by_index.get(row_index)
        if result is None:
            row["policy_logp_mean"] = None
            row["policy_logp_sum"] = None
            row["score_token_count"] = 0
            row["scoring_status"] = "missing_result"
            continue
        row["policy_logp_mean"] = result.get("logp_mean")
        row["policy_logp_sum"] = result.get("logp_sum")
        row["score_token_count"] = int(result.get("score_token_count", 0) or 0)
        row["prompt_token_count"] = int(result.get("prompt_token_count", 0) or 0)
        row["effective_prompt_token_count"] = int(result.get("effective_prompt_token_count", 0) or 0)
        row["completion_token_count"] = int(result.get("completion_token_count", 0) or 0)
        row["effective_completion_token_count"] = int(result.get("effective_completion_token_count", 0) or 0)
        row["prompt_truncated"] = bool(result.get("prompt_truncated", False))
        row["completion_truncated"] = bool(result.get("completion_truncated", False))
        row["scoring_status"] = str(result.get("status") or "scored")


def main() -> None:
    args = parse_args()
    prime_unsloth_runtime()
    project_root = get_project_root()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = Path(args.prediction_json).resolve()
    prediction_rows = load_prediction_rows(prediction_path)
    examples_by_id, prompt_bundle, eval_paths = build_examples(args=args, project_root=project_root)
    rankers = dedupe_preserve_order(["union", *args.ranker])

    model_spec = resolve_single_model_spec(
        model_ref=str(args.model_ref),
        args=args,
        project_root=project_root,
    )
    model, tokenizer = load_model_and_tokenizer_for_eval(model_spec, args)
    active_chat_template = model_spec.chat_template or clean_text(args.chat_template) or clean_text(prompt_bundle.get("chat_template", ""))
    active_prompt_format = model_spec.prompt_format or clean_text(args.prompt_format) or clean_text(prompt_bundle.get("prompt_format", "")) or "chat"

    candidate_rows: list[dict[str, Any]] = []
    payloads: list[ScoringPayload] = []
    question_rows: list[dict[str, Any]] = []
    sample_status_counts: Counter[str] = Counter()
    strict_valid_output_count = 0
    parseable_output_count = 0
    protocol_violation_count = 0
    prompt_mismatch_count = 0

    for question_index, prediction_row in enumerate(prediction_rows[: args.limit] if args.limit else prediction_rows, start=1):
        question_id = clean_text(prediction_row.get("question_id"))
        if not question_id:
            continue
        example = examples_by_id.get(question_id)
        if example is None:
            raise KeyError(f"Prediction row question_id '{question_id}' was not found in the reconstructed evaluation set.")

        saved_prompt_instruction = clean_text(prediction_row.get("prompt_instruction"))
        if saved_prompt_instruction and saved_prompt_instruction != clean_text(example.instruction):
            prompt_mismatch_count += 1

        prompt_text = render_prompt(
            tokenizer,
            example,
            chat_template=active_chat_template,
            prompt_format=active_prompt_format,
        )
        gold_groups = exact_answer_groups(example, "factoid")
        direct_prediction = clean_text(prediction_row.get("prediction"))

        if args.candidate_extraction_mode == "top_five":
            raw_outputs = [direct_prediction]
        else:
            generation_samples = prediction_row.get("generation_samples")
            if isinstance(generation_samples, list) and generation_samples:
                raw_outputs = [clean_text(sample) for sample in generation_samples if clean_text(sample)]
            else:
                raw_outputs = [direct_prediction] if direct_prediction else []

        candidate_map: dict[str, dict[str, Any]] = {}
        output_status_counts: Counter[str] = Counter()
        observation_order = 0
        valid_output_count = 0

        for output_index, raw_output in enumerate(raw_outputs, start=1):
            extracted = extract_candidates_from_output(
                raw_output,
                mode=str(args.candidate_extraction_mode),
                parser_mode=str(args.factoid_parser_mode),
                single_answer_policy=str(args.single_answer_policy),
            )
            status = str(extracted["status"])
            items = [clean_text(item) for item in extracted["items"] if clean_text(item)]
            output_status_counts[status] += 1
            sample_status_counts[status] += 1
            if extracted["strict_valid"]:
                strict_valid_output_count += 1
            if bool(extracted.get("protocol_violation")):
                protocol_violation_count += 1
            if items:
                parseable_output_count += 1
                valid_output_count += 1

            for output_rank, item in enumerate(items, start=1):
                normalized_candidate = normalize_for_match(item)
                if not normalized_candidate:
                    continue
                observation_order += 1
                candidate = candidate_map.get(normalized_candidate)
                if candidate is None:
                    candidate = {
                        "question_id": question_id,
                        "question_index": question_index,
                        "question_type": example.question_type,
                        "body": example.body,
                        "candidate_text": item,
                        "normalized_candidate": normalized_candidate,
                        "occurrence_count": 0,
                        "source_output_count": 0,
                        "source_output_indices": [],
                        "source_output_ranks": [],
                        "surface_forms": [],
                        "first_seen_output_index": output_index,
                        "first_seen_output_rank": output_rank,
                        "first_seen_observation_order": observation_order,
                    }
                    candidate_map[normalized_candidate] = candidate
                candidate["occurrence_count"] += 1
                if output_index not in candidate["source_output_indices"]:
                    candidate["source_output_indices"].append(output_index)
                    candidate["source_output_count"] += 1
                candidate["source_output_ranks"].append(output_rank)
                if item not in candidate["surface_forms"]:
                    candidate["surface_forms"].append(item)

        candidates_for_question = list(candidate_map.values())
        for candidate in candidates_for_question:
            candidate["gold_match"] = any(
                match_to_gold_group(candidate["candidate_text"], gold_group)
                for gold_group in gold_groups
            )
            candidate["prompt_instruction"] = example.instruction
            candidate["gold_output"] = example.gold_output
            candidate["policy_scoring_model_ref"] = str(model_spec.ref)

            payloads.append(
                ScoringPayload(
                    row_index=len(candidate_rows),
                    prompt=prompt_text,
                    completion_text=f"[BE] {candidate['candidate_text']} [EE]",
                    append_eos=(args.eos_policy == "included"),
                )
            )
            candidate_rows.append(candidate)

        question_record = {
            "question_id": question_id,
            "question_type": example.question_type,
            "body": example.body,
            "gold_output": example.gold_output,
            "source_path": example.source_path,
            "prompt_instruction": example.instruction,
            "saved_prediction": direct_prediction,
            "saved_generation_sample_count": len(raw_outputs),
            "output_status_counts": dict(sorted(output_status_counts.items())),
            "strict_valid_output_count": sum(
                count
                for status_name, count in output_status_counts.items()
                if status_name in {"single_item", "nonempty"}
            ),
            "protocol_violation_count": sum(
                count
                for status_name, count in output_status_counts.items()
                if status_name == "multi_item"
            ),
            "parseable_output_count": valid_output_count,
            "candidate_pool_count": len(candidates_for_question),
            "pool_gold_present": any(candidate["gold_match"] for candidate in candidates_for_question),
            "candidate_rows": candidates_for_question,
        }
        question_rows.append(question_record)

    if not payloads:
        release_model(model, tokenizer)
        raise ValueError("No candidate completions were extracted from the saved predictions.")

    policy_results = score_payloads_with_model(
        payloads=payloads,
        model=model,
        tokenizer=tokenizer,
        args=args,
        progress_label="factoid-candidate-diagnosis",
    )
    release_model(model, tokenizer)
    attach_scoring(candidate_rows=candidate_rows, payloads=payloads, results=policy_results)

    candidate_rows_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        candidate_rows_by_question.setdefault(str(row["question_id"]), []).append(row)

    for question_row in question_rows:
        ranked_candidates = candidate_rows_by_question.get(str(question_row["question_id"]), [])
        rankings: dict[str, Any] = {}
        example = examples_by_id[str(question_row["question_id"])]
        gold_groups = exact_answer_groups(example, "factoid")
        for ranker_name in rankers:
            if ranker_name == "union":
                rankings[ranker_name] = union_ranking_from_prediction(
                    str(question_row.get("saved_prediction") or ""),
                    gold_groups=gold_groups,
                    candidate_rows=ranked_candidates,
                    parser_mode=str(args.factoid_parser_mode),
                    candidate_extraction_mode=str(args.candidate_extraction_mode),
                    single_answer_policy=str(args.single_answer_policy),
                    keep_top_k=int(args.keep_top_k),
                )
                continue
            ordered_candidates = question_rankings(ranked_candidates, ranker_name=ranker_name)
            gold_rank = next(
                (rank for rank, candidate in enumerate(ordered_candidates, start=1) if bool(candidate.get("gold_match"))),
                None,
            )
            rankings[ranker_name] = {
                "gold_rank": gold_rank,
                "top_candidates": [
                    {
                        "rank": rank,
                        "candidate_text": candidate["candidate_text"],
                        "gold_match": bool(candidate.get("gold_match")),
                        "occurrence_count": int(candidate.get("occurrence_count") or 0),
                        "policy_logp_mean": candidate.get("policy_logp_mean"),
                        "policy_logp_sum": candidate.get("policy_logp_sum"),
                    }
                    for rank, candidate in enumerate(ordered_candidates[: args.keep_top_k], start=1)
                ],
            }
        question_row["rankings"] = rankings
        question_row["union_gold_rank"] = rankings["union"]["gold_rank"]
        question_row["candidate_rows"] = ranked_candidates

    pool_candidate_counts = [int(row["candidate_pool_count"]) for row in question_rows]
    pool_gold_present_count = sum(1 for row in question_rows if bool(row.get("pool_gold_present")))
    scoring_status_counts: Counter[str] = Counter(
        str(row.get("scoring_status") or "unknown")
        for row in candidate_rows
    )

    summary = {
        "created_at": utc_now_iso(),
        "prediction_json": str(prediction_path),
        "output_dir": str(output_dir),
        "question_count": len(question_rows),
        "candidate_count": len(candidate_rows),
        "model": {
            "ref": model_spec.ref,
            "label": model_spec.label,
            "source": model_spec.source,
            "load_target": model_spec.load_target,
            "chat_template": active_chat_template or None,
            "prompt_format": active_prompt_format,
        },
        "prompt": {
            "prompt_id": prompt_bundle.get("prompt_id"),
            "name": prompt_bundle.get("name"),
            "source": prompt_bundle.get("source"),
            "registry_path": prompt_bundle.get("registry_path"),
        },
        "dataset": {
            "eval_input": [str(path) for path in eval_paths],
            "question_types": sorted({row["question_type"] for row in question_rows}),
            "max_resources": int(args.max_resources),
            "max_resource_chars": int(args.max_resource_chars),
            "resource_selection": str(args.resource_selection),
            "resource_granularity": str(args.resource_granularity),
            "resource_window_mode": str(args.resource_window_mode),
            "prompt_instruction_mismatch_count": prompt_mismatch_count,
        },
        "candidate_extraction": {
            "mode": args.candidate_extraction_mode,
            "single_answer_policy": str(args.single_answer_policy),
            "factoid_parser_mode": str(args.factoid_parser_mode),
            "keep_top_k": int(args.keep_top_k),
            "max_recall_k": int(args.max_recall_k),
            "eos_policy": str(args.eos_policy),
            "rankers": list(rankers),
        },
        "candidate_pool": {
            "question_count": len(question_rows),
            "gold_present_count": pool_gold_present_count,
            "gold_present_rate": (pool_gold_present_count / len(question_rows)) if question_rows else 0.0,
            "gold_absent_count": len(question_rows) - pool_gold_present_count,
            "gold_absent_rate": ((len(question_rows) - pool_gold_present_count) / len(question_rows)) if question_rows else 0.0,
            "candidate_pool_count_distribution": summarize_numeric(pool_candidate_counts),
        },
        "output_format": {
            "total_outputs": sum(int(row["saved_generation_sample_count"]) for row in question_rows),
            "strict_valid_output_count": strict_valid_output_count,
            "parseable_output_count": parseable_output_count,
            "protocol_violation_count": protocol_violation_count,
            "status_counts": dict(sorted(sample_status_counts.items())),
        },
        "inference_budget": inference_budget_summary(prediction_rows[: args.limit] if args.limit else prediction_rows),
        "scoring": {
            "scoring_status_counts": dict(sorted(scoring_status_counts.items())),
            "policy_logp_mean_distribution": summarize_numeric(
                [
                    float(row["policy_logp_mean"])
                    for row in candidate_rows
                    if isinstance(row.get("policy_logp_mean"), (int, float))
                ]
            ),
            "policy_logp_sum_distribution": summarize_numeric(
                [
                    float(row["policy_logp_sum"])
                    for row in candidate_rows
                    if isinstance(row.get("policy_logp_sum"), (int, float))
                ]
            ),
        },
        "ranking": {
            ranker_name: ranking_summary(
                question_rows,
                ranker_name=ranker_name,
                keep_top_k=int(args.keep_top_k),
                max_recall_k=int(args.max_recall_k),
            )
            for ranker_name in rankers
        },
    }

    official_saved_prediction = evaluate_with_bioasq_java(
        prediction_rows=prediction_rows[: args.limit] if args.limit else prediction_rows,
        examples_by_key={
            (clean_text(example.question_id), clean_text(example.question_type).lower()): example
            for example in examples_by_id.values()
        },
        model_label=str(model_spec.label),
        model_dir=output_dir,
        args=args,
    )
    summary["official_saved_prediction"] = {
        "aggregate": official_saved_prediction["aggregate"],
        "paths": official_saved_prediction["paths"],
        "command": official_saved_prediction["command"],
    }

    official_reranked = {}
    official_prediction_rows = prediction_rows[: args.limit] if args.limit else prediction_rows
    question_rows_by_id = {
        clean_text(row["question_id"]): row
        for row in question_rows
    }
    examples_by_key = {
        (clean_text(example.question_id), clean_text(example.question_type).lower()): example
        for example in examples_by_id.values()
    }
    for reranker_name in dedupe_preserve_order(args.official_ranker):
        reranked_rows = []
        for original_row in official_prediction_rows:
            question_id = clean_text(original_row.get("question_id"))
            diagnosis_row = question_rows_by_id.get(question_id)
            if diagnosis_row is None:
                continue
            top_candidates = diagnosis_row["rankings"][reranker_name]["top_candidates"]
            prediction = " ".join(
                f"[BE]{clean_text(candidate['candidate_text'])}[EE]"
                for candidate in top_candidates
                if clean_text(candidate.get("candidate_text"))
            )
            reranked_row = dict(original_row)
            reranked_row["prediction"] = prediction
            reranked_rows.append(reranked_row)

        reranked_dir = output_dir / f"official_bioasq_{reranker_name}_reranked"
        reranked_payload = evaluate_with_bioasq_java(
            prediction_rows=reranked_rows,
            examples_by_key=examples_by_key,
            model_label=f"{model_spec.label}-{reranker_name}-reranked",
            model_dir=reranked_dir,
            args=args,
        )
        official_reranked[reranker_name] = {
            "aggregate": reranked_payload["aggregate"],
            "paths": reranked_payload["paths"],
            "command": reranked_payload["command"],
            "prediction_json": str(reranked_dir / "official_bioasq" / "overall.predictions.json"),
        }
    if official_reranked:
        summary["official_reranked_predictions"] = official_reranked

    summary["submission_aligned"] = submission_aligned_summary(official_saved_prediction)
    summary["metric_notes"] = {
        "headline": (
            "submission_aligned and official_saved_prediction use the official BioASQ "
            "Java evaluation and should be treated as the primary scores."
        ),
        "diagnostic": (
            "ranking, candidate_pool, and output_format quantify "
            "candidate discovery and reranking behavior; they are not identical to the "
            "official submission metric."
        ),
    }

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "question_diagnosis.jsonl", question_rows)
    write_jsonl(output_dir / "candidate_scores.jsonl", candidate_rows)

    print(f"Wrote factoid candidate diagnosis summary to {output_dir / 'summary.json'}")
    print(f"Wrote question-level diagnosis rows to {output_dir / 'question_diagnosis.jsonl'}")
    print(f"Wrote candidate score rows to {output_dir / 'candidate_scores.jsonl'}")


if __name__ == "__main__":
    main()
