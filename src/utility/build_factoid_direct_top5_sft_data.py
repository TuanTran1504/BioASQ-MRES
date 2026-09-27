from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.score_candidate_bank import (
    ScoringPayload,
    release_model,
    resolve_single_model_spec,
    score_payloads_with_model,
)
from src.model_registry import get_project_root, resolve_repo_path, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle
from src.utility.bioasq_format import exact_answer_groups, match_to_gold_group, normalize_for_match
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import build_resources, clean_multiline_text, clean_text, save_prepared_records, split_train_eval
from src.utility.eval_dataset import render_prompt
from src.utility.eval_models import generate_answer_samples, load_model_and_tokenizer_for_eval, prime_unsloth_runtime
from src.utility.eval_types import EvalExample
from src.utility.factoid_output_parsing import clean_factoid_candidate_text, parse_factoid_candidates


DEFAULT_TRAIN_INPUT = ["data/training13b.json"]
DEFAULT_OUTPUT_DIR = "data/BioASQ_factoid_sft_prepared/direct_top5_hard_negative_full_resources"
DEFAULT_MODEL_REF = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DEFAULT_PROMPT_FILE = "prompts/factoid_single_answer_aligned.json"
DEFAULT_SAMPLE_PROMPT = "factoid-single-answer-aligned-v1"
DEFAULT_TARGET_PROMPT = "factoid-top-five-eval-v1"
NEGATIVE_RANKING_RULE = "frequency_desc_then_logp_mean_desc_then_first_seen_asc_then_text_asc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build direct-top-5 factoid SFT training data by sampling many single-answer "
            "responses from a base model, removing gold-equivalent candidates, ranking the "
            "remaining hard negatives, and writing prepared train/eval JSON files."
        )
    )
    parser.add_argument(
        "--train-input",
        nargs="+",
        default=DEFAULT_TRAIN_INPUT,
        help="One or more raw BioASQ JSON files containing training questions.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prepared data, audit records, and summary JSON will be written.",
    )
    parser.add_argument(
        "--model-ref",
        default=DEFAULT_MODEL_REF,
        help="Base model used to sample candidate answers and score hard negatives.",
    )
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative model registry path used when resolving --model-ref aliases.",
    )
    parser.add_argument(
        "--sample-prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help="Prompt registry JSON used for the single-answer sampling prompt.",
    )
    parser.add_argument(
        "--sample-prompt",
        default=DEFAULT_SAMPLE_PROMPT,
        help="Prompt alias or prompt_id used for single-answer candidate sampling.",
    )
    parser.add_argument(
        "--target-prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help="Prompt registry JSON used for the direct-top-5 training target prompt.",
    )
    parser.add_argument(
        "--target-prompt",
        default=DEFAULT_TARGET_PROMPT,
        help="Prompt alias or prompt_id used for the direct-top-5 training target instruction.",
    )
    parser.add_argument(
        "--reuse-question-audit",
        default=None,
        help=(
            "Optional existing question_audit.jsonl path. When provided, reuse the saved "
            "sampled outputs per question instead of generating fresh 20-sample outputs."
        ),
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=20,
        help="How many sampled single-answer generations to draw per question.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling for candidate generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used when generating candidate answers.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter used when generating candidate answers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens generated per sampled answer.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=0,
        help="Maximum number of resources to keep per question. Use 0 for all available resources.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=0,
        help="Maximum characters per serialized resource. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--resource-granularity",
        choices=["document", "snippet"],
        default="document",
        help="How to rebuild BioASQ evidence resources.",
    )
    parser.add_argument(
        "--single-answer-policy",
        choices=["strict", "first_entity", "permissive"],
        default="strict",
        help=(
            "How to interpret each sampled single-answer output. 'strict' keeps outputs only "
            "when exactly one candidate is parsed."
        ),
    )
    parser.add_argument(
        "--factoid-parser-mode",
        choices=["current", "agnostic"],
        default="agnostic",
        help="Parser used to extract candidates from sampled outputs.",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=4,
        help="How many non-gold hard negatives to keep per question.",
    )
    parser.add_argument(
        "--negative-fill-strategy",
        choices=["leave_short", "drop", "global_pool"],
        default="leave_short",
        help=(
            "How to handle questions with too few question-specific non-gold negatives after "
            "sampling and deduplication. 'leave_short' keeps the question with fewer than five "
            "targets, 'drop' removes it, and 'global_pool' backfills missing slots from the "
            "global scored non-gold candidate pool."
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of prepared rows reserved for the eval split.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--eos-policy",
        choices=["excluded", "included"],
        default="excluded",
        help="Whether to append EOS when scoring negative candidate completions.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--verbose-every", type=int, default=25)
    parser.add_argument("--empty-cuda-cache-steps", type=int, default=0)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--prompt-format", default=None, choices=["chat", "unitor_plain"])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit for smoke tests.")
    args = parser.parse_args()
    args.do_sample = True if not bool(args.do_sample) else bool(args.do_sample)
    args.use_cache = True
    args.aggregation_strategy = "union"
    args.aggregation_min_frequency = 2
    return args


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def iter_factoid_questions(paths: Sequence[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"Raw BioASQ file must contain a 'questions' list: {path}")
        for question in questions:
            if isinstance(question, dict) and clean_text(question.get("type", "")).lower() == "factoid":
                yield path, question


def alias_specificity(alias: str) -> tuple[int, int, int, int]:
    normalized = normalize_for_match(alias)
    tokens = normalized.split() if normalized else []
    has_digits = int(any(char.isdigit() for char in alias))
    has_parenthetical = int("(" in alias or ")" in alias)
    return (has_digits, has_parenthetical, len(tokens), len(alias))


def choose_best_alias(aliases: Sequence[str]) -> str:
    cleaned = [clean_text(alias) for alias in aliases if clean_text(alias)]
    if not cleaned:
        return ""
    return max(cleaned, key=alias_specificity)


def select_canonical_gold_answer(gold_groups: Sequence[Sequence[str]], resources: Sequence[str]) -> tuple[str, int | None]:
    evidence_text = normalize_for_match("\n".join(resource for resource in resources if clean_text(resource)))
    visible_candidates: list[tuple[int, str]] = []
    fallback_candidates: list[tuple[int, str]] = []

    for group_index, gold_group in enumerate(gold_groups):
        aliases = [clean_text(alias) for alias in gold_group if clean_text(alias)]
        if not aliases:
            continue
        visible_aliases = [
            alias
            for alias in aliases
            if normalize_for_match(alias) and normalize_for_match(alias) in evidence_text
        ]
        if visible_aliases:
            visible_candidates.append((group_index, choose_best_alias(visible_aliases)))
        fallback_candidates.append((group_index, choose_best_alias(aliases)))

    if visible_candidates:
        return visible_candidates[0][1], visible_candidates[0][0]
    if fallback_candidates:
        return fallback_candidates[0][1], fallback_candidates[0][0]
    return "", None


def classify_extraction(parsed_items: Sequence[str]) -> tuple[str, bool, bool]:
    item_count = len(parsed_items)
    if item_count <= 0:
        return "empty", False, True
    if item_count == 1:
        return "single_item", True, False
    return "multi_item", False, True


def extract_single_answer_candidates(
    text: str,
    *,
    parser_mode: str,
    single_answer_policy: str,
) -> dict[str, Any]:
    parsed_items = parse_factoid_candidates(text, parser_mode=parser_mode)
    status, strict_valid, protocol_violation = classify_extraction(parsed_items)
    if single_answer_policy == "strict":
        selected_items = list(parsed_items) if len(parsed_items) == 1 else []
    elif single_answer_policy == "first_entity":
        selected_items = list(parsed_items[:1])
    elif single_answer_policy == "permissive":
        selected_items = list(parsed_items)
    else:
        raise ValueError(f"Unsupported single-answer policy: {single_answer_policy}")

    return {
        "parsed_items": list(parsed_items),
        "selected_items": [clean_factoid_candidate_text(item) for item in selected_items if clean_factoid_candidate_text(item)],
        "status": status,
        "strict_valid": strict_valid,
        "protocol_violation": protocol_violation,
    }


def build_eval_example(
    *,
    question: Mapping[str, Any],
    question_id: str,
    question_text: str,
    instruction: str,
    resources: Sequence[str],
    source_path: str,
) -> EvalExample:
    return EvalExample(
        question_id=question_id,
        question_type="factoid",
        body=question_text,
        instruction=instruction,
        resources=tuple(resource for resource in resources if clean_text(resource)),
        gold_output="",
        source_path=source_path,
        raw_question=dict(question),
    )


def negative_sort_key(candidate: Mapping[str, Any]) -> tuple[int, float, int, str]:
    occurrence_count = int(candidate.get("occurrence_count") or 0)
    logp_mean = candidate.get("policy_logp_mean")
    numeric_logp_mean = float(logp_mean) if isinstance(logp_mean, (int, float)) else float("-inf")
    first_seen = int(candidate.get("first_seen_observation_order") or 0)
    normalized_candidate = str(candidate.get("normalized_candidate") or "")
    return (
        -occurrence_count,
        -numeric_logp_mean,
        first_seen,
        normalized_candidate,
    )


def candidate_matches_gold_groups(candidate_text: str, gold_groups: Sequence[Sequence[str]]) -> bool:
    # Route all candidate-vs-gold checks through the shared BioASQ-like exact
    # matcher so this data-construction pipeline stays aligned with evaluation.
    return any(match_to_gold_group(candidate_text, gold_group) for gold_group in gold_groups)


def build_global_negative_pool(candidate_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    ordered_candidates = sorted(candidate_rows, key=negative_sort_key)
    unique_candidates: list[Mapping[str, Any]] = []
    seen_normalized: set[str] = set()
    for candidate in ordered_candidates:
        normalized_candidate = str(candidate.get("normalized_candidate") or "")
        if not normalized_candidate or normalized_candidate in seen_normalized:
            continue
        seen_normalized.add(normalized_candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def select_negative_candidates(
    *,
    question_id: str,
    gold_groups: Sequence[Sequence[str]],
    local_candidates: Sequence[Mapping[str, Any]],
    global_candidates: Sequence[Mapping[str, Any]],
    negative_count: int,
    fill_strategy: str,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    selected_candidates = list(local_candidates[:negative_count])
    selected_sources = ["local"] * len(selected_candidates)

    if len(selected_candidates) >= negative_count or fill_strategy in {"drop", "leave_short"}:
        return selected_candidates, selected_sources

    seen_normalized = {
        str(candidate.get("normalized_candidate") or "")
        for candidate in selected_candidates
        if str(candidate.get("normalized_candidate") or "")
    }

    for candidate in global_candidates:
        if len(selected_candidates) >= negative_count:
            break
        if str(candidate.get("question_id") or "") == question_id:
            continue

        normalized_candidate = str(candidate.get("normalized_candidate") or "")
        candidate_text = str(candidate.get("candidate_text") or "")
        if not normalized_candidate or not candidate_text:
            continue
        if normalized_candidate in seen_normalized:
            continue
        if candidate_matches_gold_groups(candidate_text, gold_groups):
            continue

        selected_candidates.append(candidate)
        selected_sources.append("global_pool")
        seen_normalized.add(normalized_candidate)

    return selected_candidates, selected_sources


def build_prepared_row(
    *,
    question_id: str,
    question_text: str,
    instruction: str,
    resources: Sequence[str],
    canonical_gold: str,
    negatives: Sequence[str],
) -> dict[str, Any]:
    target_items = [canonical_gold, *negatives]
    row: dict[str, Any] = {
        "id": question_id,
        "type": "factoid",
        "instruction": instruction,
        "input_1": question_text,
        "output": " ".join(f"[BE]{item}[EE]" for item in target_items),
    }
    for index, resource in enumerate(resources, start=2):
        cleaned = clean_multiline_text(resource)
        if cleaned:
            row[f"input_{index}"] = cleaned
    return row


def safe_mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    prime_unsloth_runtime()
    project_root = get_project_root()

    output_dir = resolve_repo_path(args.output_dir, project_root=project_root) or Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse_audit_path = (
        resolve_repo_path(args.reuse_question_audit, project_root=project_root) or Path(args.reuse_question_audit)
        if args.reuse_question_audit
        else None
    )

    sample_prompt_path = resolve_repo_path(args.sample_prompt_file, project_root=project_root)
    target_prompt_path = resolve_repo_path(args.target_prompt_file, project_root=project_root)
    sample_prompt_bundle = resolve_prompt_bundle(
        registry_path=sample_prompt_path,
        prompt_ref=args.sample_prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    target_prompt_bundle = resolve_prompt_bundle(
        registry_path=target_prompt_path,
        prompt_ref=args.target_prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )

    sample_instruction = clean_multiline_text(
        sample_prompt_bundle.get("instructions", {}).get("factoid") or QUESTION_INSTRUCTIONS["factoid"]
    )
    target_instruction = clean_multiline_text(
        target_prompt_bundle.get("instructions", {}).get("factoid") or QUESTION_INSTRUCTIONS["factoid"]
    )
    if not sample_instruction or not target_instruction:
        raise ValueError("Both sample and target factoid instructions must resolve to non-empty text.")

    model_spec = resolve_single_model_spec(
        model_ref=str(args.model_ref),
        args=args,
        project_root=project_root,
    )
    model, tokenizer = load_model_and_tokenizer_for_eval(model_spec, args)
    active_chat_template = (
        model_spec.chat_template
        or clean_text(sample_prompt_bundle.get("chat_template", ""))
        or None
    )
    active_prompt_format = (
        model_spec.prompt_format
        or clean_text(sample_prompt_bundle.get("prompt_format", ""))
        or "chat"
    )

    train_input_paths = [
        resolve_repo_path(path_value, project_root=project_root) or Path(path_value)
        for path_value in args.train_input
    ]
    reuse_audits_by_question: dict[str, dict[str, Any]] = {}
    if reuse_audit_path is not None:
        if not reuse_audit_path.exists():
            raise FileNotFoundError(f"Reuse audit file does not exist: {reuse_audit_path}")
        reuse_rows = load_jsonl_rows(reuse_audit_path)
        reuse_audits_by_question = {
            clean_text(row.get("question_id", "")): row
            for row in reuse_rows
            if clean_text(row.get("question_id", ""))
        }

    audits: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    scoring_payloads: list[ScoringPayload] = []
    question_builds: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    seen_question_ids: set[str] = set()
    total_questions = 0

    for source_path, question in iter_factoid_questions(train_input_paths):
        total_questions += 1
        if args.limit is not None and len(question_builds) + sum(drop_counts.values()) >= args.limit:
            break

        question_id = clean_text(question.get("id", "")) or f"{source_path.name}:{total_questions}"
        if question_id in seen_question_ids:
            continue
        seen_question_ids.add(question_id)

        question_text = clean_text(question.get("body", ""))
        if not question_text:
            drop_counts["missing_question_text"] += 1
            audits.append(
                {
                    "question_id": question_id,
                    "source_path": str(source_path),
                    "status": "dropped",
                    "drop_reason": "missing_question_text",
                }
            )
            continue

        resources = build_resources(
            question,
            max_resources=args.max_resources,
            max_resource_chars=args.max_resource_chars,
            question_text=question_text,
            resource_granularity=args.resource_granularity,
            resource_selection="first",
        )
        example = build_eval_example(
            question=question,
            question_id=question_id,
            question_text=question_text,
            instruction=sample_instruction,
            resources=resources,
            source_path=str(source_path),
        )
        gold_groups = exact_answer_groups(example, "factoid")
        canonical_gold, canonical_gold_group_index = select_canonical_gold_answer(gold_groups, resources)
        if not gold_groups or not canonical_gold:
            drop_counts["missing_canonical_gold"] += 1
            audits.append(
                {
                    "question_id": question_id,
                    "source_path": str(source_path),
                    "status": "dropped",
                    "drop_reason": "missing_canonical_gold",
                    "gold_group_count": len(gold_groups),
                }
            )
            continue

        prompt_text = render_prompt(
            tokenizer,
            example,
            chat_template=active_chat_template,
            prompt_format=active_prompt_format,
        )
        reuse_audit = reuse_audits_by_question.get(question_id)
        reused_existing_samples = False
        if reuse_audit is not None:
            sample_rows = reuse_audit.get("samples")
            if isinstance(sample_rows, list) and sample_rows:
                ordered_sample_rows = sorted(
                    (dict(sample_row) for sample_row in sample_rows if isinstance(sample_row, Mapping)),
                    key=lambda row: int(row.get("sample_index") or 0),
                )
                generation_samples = [
                    clean_text(sample_row.get("raw_output", ""))
                    for sample_row in ordered_sample_rows
                    if clean_text(sample_row.get("raw_output", ""))
                ]
                reused_existing_samples = bool(generation_samples)
            else:
                generation_samples = []
        else:
            generation_samples = []

        if not generation_samples:
            _prediction, generation_samples, _generation_telemetry = generate_answer_samples(
                model=model,
                tokenizer=tokenizer,
                example=example,
                args=args,
                chat_template=active_chat_template,
                prompt_format=active_prompt_format,
            )

        candidate_map: dict[str, dict[str, Any]] = {}
        sample_audits: list[dict[str, Any]] = []
        sample_status_counts: Counter[str] = Counter()
        observation_order = 0
        gold_matched_sample_count = 0

        for output_index, raw_output in enumerate(generation_samples, start=1):
            extracted = extract_single_answer_candidates(
                raw_output,
                parser_mode=str(args.factoid_parser_mode),
                single_answer_policy=str(args.single_answer_policy),
            )
            sample_status = str(extracted["status"])
            parsed_items = list(extracted["parsed_items"])
            selected_items = [clean_text(item) for item in extracted["selected_items"] if clean_text(item)]
            sample_status_counts[sample_status] += 1
            status_counts[sample_status] += 1

            sample_audits.append(
                {
                    "sample_index": output_index,
                    "raw_output": clean_text(raw_output),
                    "parsed_items": parsed_items,
                    "selected_items": selected_items,
                    "status": sample_status,
                    "strict_valid": bool(extracted["strict_valid"]),
                    "protocol_violation": bool(extracted["protocol_violation"]),
                }
            )

            for output_rank, candidate_text in enumerate(selected_items, start=1):
                normalized_candidate = normalize_for_match(candidate_text)
                if not normalized_candidate:
                    continue
                if candidate_matches_gold_groups(candidate_text, gold_groups):
                    gold_matched_sample_count += 1
                    continue

                observation_order += 1
                candidate = candidate_map.get(normalized_candidate)
                if candidate is None:
                    candidate = {
                        "question_id": question_id,
                        "candidate_text": candidate_text,
                        "normalized_candidate": normalized_candidate,
                        "occurrence_count": 0,
                        "surface_forms": [],
                        "source_output_indices": [],
                        "first_seen_output_index": output_index,
                        "first_seen_output_rank": output_rank,
                        "first_seen_observation_order": observation_order,
                    }
                    candidate_map[normalized_candidate] = candidate
                candidate["occurrence_count"] += 1
                if candidate_text not in candidate["surface_forms"]:
                    candidate["surface_forms"].append(candidate_text)
                if output_index not in candidate["source_output_indices"]:
                    candidate["source_output_indices"].append(output_index)

        non_gold_candidates = list(candidate_map.values())
        audit_record: dict[str, Any] = {
            "question_id": question_id,
            "source_path": str(source_path),
            "status": "pending_scoring",
            "question": question_text,
            "canonical_gold": canonical_gold,
            "canonical_gold_group_index": canonical_gold_group_index,
            "gold_groups": [[clean_text(alias) for alias in group if clean_text(alias)] for group in gold_groups],
            "resource_count": len([resource for resource in resources if clean_text(resource)]),
            "sample_generation_count": len(generation_samples),
            "sample_source": "reused_question_audit" if reused_existing_samples else "fresh_generation",
            "sample_status_counts": dict(sorted(sample_status_counts.items())),
            "gold_matched_sample_count": gold_matched_sample_count,
            "candidate_pool_size_before_scoring": len(non_gold_candidates),
            "samples": sample_audits,
        }

        for candidate in non_gold_candidates:
            candidate["policy_scoring_model_ref"] = str(model_spec.ref)
            scoring_payloads.append(
                ScoringPayload(
                    row_index=len(candidate_rows),
                    prompt=prompt_text,
                    completion_text=f"[BE] {candidate['candidate_text']} [EE]",
                    append_eos=(args.eos_policy == "included"),
                )
            )
            candidate_rows.append(candidate)

        question_builds.append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "resources": resources,
                "canonical_gold": canonical_gold,
                "gold_groups": [[clean_text(alias) for alias in group if clean_text(alias)] for group in gold_groups],
                "audit": audit_record,
                "candidates": non_gold_candidates,
            }
        )

        if int(args.verbose_every) > 0 and len(question_builds) % int(args.verbose_every) == 0:
            print(
                f"[direct-top5-data] prepared candidate pools for {len(question_builds):,} kept questions",
                flush=True,
            )

    if not question_builds:
        release_model(model, tokenizer)
        raise ValueError("No usable factoid questions were converted into direct-top-5 training rows.")

    scoring_results = score_payloads_with_model(
        payloads=scoring_payloads,
        model=model,
        tokenizer=tokenizer,
        args=args,
        progress_label="direct-top5-data",
    )
    release_model(model, tokenizer)

    results_by_index = {int(result["row_index"]): dict(result) for result in scoring_results}
    for row_index, candidate in enumerate(candidate_rows):
        result = results_by_index.get(row_index, {})
        candidate["policy_logp_mean"] = result.get("logp_mean")
        candidate["policy_logp_sum"] = result.get("logp_sum")
        candidate["score_token_count"] = int(result.get("score_token_count", 0) or 0)
        candidate["scoring_status"] = str(result.get("status") or "scored")

    global_negative_pool = build_global_negative_pool(candidate_rows)

    prepared_rows: list[dict[str, Any]] = []
    candidate_pool_sizes: list[float] = []
    selected_negative_occurrences: list[float] = []
    selected_negative_logp_means: list[float] = []
    shortfall_question_count = 0
    padded_question_count = 0
    padded_negative_slot_count = 0
    short_target_question_count = 0

    for question_build in question_builds:
        sorted_candidates = sorted(question_build["candidates"], key=negative_sort_key)
        negative_count = int(args.negative_count)
        local_shortfall = max(0, negative_count - len(sorted_candidates))
        if local_shortfall > 0:
            shortfall_question_count += 1

        selected_candidates, selected_sources = select_negative_candidates(
            question_id=str(question_build["question_id"]),
            gold_groups=question_build["gold_groups"],
            local_candidates=sorted_candidates,
            global_candidates=global_negative_pool,
            negative_count=negative_count,
            fill_strategy=str(args.negative_fill_strategy),
        )
        global_padding_count = sum(1 for source in selected_sources if source == "global_pool")
        if global_padding_count > 0:
            padded_question_count += 1
            padded_negative_slot_count += global_padding_count

        requires_full_length = str(args.negative_fill_strategy) in {"drop", "global_pool"}
        if len(selected_candidates) < int(args.negative_count) and requires_full_length:
            drop_counts["too_few_negatives_after_fill"] += 1
            audit = dict(question_build["audit"])
            audit["status"] = "dropped"
            audit["drop_reason"] = "too_few_negatives_after_fill"
            audit["candidate_pool_size_after_scoring"] = len(sorted_candidates)
            audit["local_negative_shortfall"] = local_shortfall
            audit["negative_fill_strategy"] = str(args.negative_fill_strategy)
            audit["global_padding_count"] = global_padding_count
            audits.append(audit)
            continue
        if len(selected_candidates) < int(args.negative_count):
            short_target_question_count += 1

        selected_negative_texts = [str(candidate["candidate_text"]) for candidate in selected_candidates]
        row = build_prepared_row(
            question_id=str(question_build["question_id"]),
            question_text=str(question_build["question_text"]),
            instruction=target_instruction,
            resources=question_build["resources"],
            canonical_gold=str(question_build["canonical_gold"]),
            negatives=selected_negative_texts,
        )
        prepared_rows.append(row)

        candidate_pool_sizes.append(float(len(sorted_candidates)))
        for candidate in selected_candidates:
            selected_negative_occurrences.append(float(candidate.get("occurrence_count") or 0))
            if isinstance(candidate.get("policy_logp_mean"), (int, float)):
                selected_negative_logp_means.append(float(candidate["policy_logp_mean"]))

        audit = dict(question_build["audit"])
        audit["status"] = "kept"
        audit["drop_reason"] = None
        audit["candidate_pool_size_after_scoring"] = len(sorted_candidates)
        audit["local_negative_shortfall"] = local_shortfall
        audit["negative_fill_strategy"] = str(args.negative_fill_strategy)
        audit["global_padding_count"] = global_padding_count
        audit["used_global_negative_padding"] = bool(global_padding_count)
        audit["would_have_been_dropped_without_padding"] = local_shortfall > 0
        audit["kept_short_target"] = len(selected_candidates) < int(args.negative_count)
        audit["negative_ranking_rule"] = NEGATIVE_RANKING_RULE
        audit["selected_negatives"] = [
            {
                "rank": rank,
                "candidate_text": candidate["candidate_text"],
                "selection_source": source,
                "occurrence_count": int(candidate.get("occurrence_count") or 0),
                "policy_logp_mean": candidate.get("policy_logp_mean"),
                "policy_logp_sum": candidate.get("policy_logp_sum"),
                "first_seen_output_index": int(candidate.get("first_seen_output_index") or 0),
                "first_seen_observation_order": int(candidate.get("first_seen_observation_order") or 0),
            }
            for rank, (candidate, source) in enumerate(zip(selected_candidates, selected_sources), start=2)
        ]
        audits.append(audit)

    if not prepared_rows:
        raise ValueError("Candidate pools were built, but no final prepared rows survived negative selection.")

    train_rows, eval_rows = split_train_eval(
        prepared_rows,
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
    )

    save_prepared_records(prepared_rows, output_dir / "all_prepared.json")
    save_prepared_records(train_rows, output_dir / "train_prepared.json")
    save_prepared_records(eval_rows, output_dir / "eval_prepared.json")
    write_jsonl(output_dir / "question_audit.jsonl", audits)

    summary = {
        "created_at": utc_now_iso(),
        "dataset_name": "factoid_direct_top5_hard_negative_sft",
        "train_input": [str(path) for path in train_input_paths],
        "output_dir": str(output_dir),
        "model": {
            "ref": model_spec.ref,
            "label": model_spec.label,
            "source": model_spec.source,
            "load_target": model_spec.load_target,
            "chat_template": active_chat_template,
            "prompt_format": active_prompt_format,
        },
        "sampling": {
            "sample_prompt_id": sample_prompt_bundle.get("prompt_id"),
            "sample_prompt_name": sample_prompt_bundle.get("name"),
            "target_prompt_id": target_prompt_bundle.get("prompt_id"),
            "target_prompt_name": target_prompt_bundle.get("name"),
            "num_generations": int(args.num_generations),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "single_answer_policy": str(args.single_answer_policy),
            "factoid_parser_mode": str(args.factoid_parser_mode),
            "eos_policy": str(args.eos_policy),
            "gold_matcher": "bioasq_like_exact_match_to_synonym_group",
            "sample_source": (
                "mixed_reuse_and_fresh"
                if reuse_audit_path is not None
                else "fresh_generation"
            ),
            "reuse_question_audit": str(reuse_audit_path) if reuse_audit_path is not None else None,
        },
        "resources": {
            "max_resources": int(args.max_resources),
            "max_resource_chars": int(args.max_resource_chars),
            "resource_granularity": str(args.resource_granularity),
        },
        "negative_selection": {
            "negative_count": int(args.negative_count),
            "ranking_rule": NEGATIVE_RANKING_RULE,
            "fill_strategy": str(args.negative_fill_strategy),
        },
        "split": {
            "validation_ratio": float(args.validation_ratio),
            "seed": int(args.seed),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "all_rows": len(prepared_rows),
        },
        "counts": {
            "total_factoid_questions_seen": total_questions,
            "kept_questions_before_scoring": len(question_builds),
            "final_prepared_rows": len(prepared_rows),
            "dropped_questions": dict(sorted(drop_counts.items())),
            "questions_with_local_negative_shortfall": shortfall_question_count,
            "questions_kept_with_global_padding": padded_question_count,
            "global_padding_negative_slots_used": padded_negative_slot_count,
            "questions_kept_with_short_targets": short_target_question_count,
            "sample_status_counts": dict(sorted(status_counts.items())),
            "candidate_rows_scored": len(candidate_rows),
        },
        "descriptives": {
            "mean_candidate_pool_size_after_scoring": safe_mean(candidate_pool_sizes),
            "mean_selected_negative_occurrence_count": safe_mean(selected_negative_occurrences),
            "mean_selected_negative_logp_mean": safe_mean(selected_negative_logp_means),
        },
        "artifacts": {
            "all_prepared_json": str(output_dir / "all_prepared.json"),
            "train_prepared_json": str(output_dir / "train_prepared.json"),
            "eval_prepared_json": str(output_dir / "eval_prepared.json"),
            "question_audit_jsonl": str(output_dir / "question_audit.jsonl"),
        },
    }
    write_json(output_dir / "build_summary.json", summary)

    print(f"Saved {len(prepared_rows):,} prepared rows to {output_dir / 'all_prepared.json'}")
    print(f"Saved {len(train_rows):,} train rows to {output_dir / 'train_prepared.json'}")
    print(f"Saved {len(eval_rows):,} eval rows to {output_dir / 'eval_prepared.json'}")
    print(f"Saved question audit to {output_dir / 'question_audit.jsonl'}")
    print(f"Saved summary to {output_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()
