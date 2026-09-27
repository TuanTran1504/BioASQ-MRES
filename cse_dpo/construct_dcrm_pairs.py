from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_registry import slugify
from src.utility.data import clean_text

from .audit_preference_pairs import build_summary, render_pair_block
from .common import load_json_records, summarize_numeric, truncate_multiline, write_json, write_jsonl
from .construct_set_edit_pairs import build_response_audit_summary, group_responses_by_question
from .construct_whole_response_dpo_pairs import (
    RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
    RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
    RESPONSE_DEDUPE_MODE_SEMANTIC,
    dedupe_and_trim_whole_response_pairs,
    response_dedupe_key,
    response_items,
)
from .match_gold_groups import score_item_surfaces, semantic_set_key_for_question
from .normalize_set_answers import clean_text as normalize_clean_text
from .normalize_set_answers import serialize_list_items
from .schemas import MatchedResponse, PreferencePair, QuestionExample, to_jsonable


PAIR_TYPE_WHOLE_RESPONSE_MODEL_SCORE = "whole_response_model_score"


def semantic_set_edit_distance(
    question: QuestionExample,
    left: Sequence[str],
    right: Sequence[str],
) -> int:
    return len(
        set(semantic_set_key_for_question(question, left)).symmetric_difference(
            set(semantic_set_key_for_question(question, right))
        )
    )


def score_lookup_key(row: Mapping[str, Any]) -> tuple[str, str]:
    question_id = clean_text(row.get("question_id") or row.get("id"))
    response_id = clean_text(row.get("response_id"))
    return question_id, response_id


def load_score_lookup(paths: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_path in paths:
        for row in load_json_records(Path(raw_path)):
            key = score_lookup_key(row)
            if not key[0] or not key[1]:
                continue
            lookup[key] = dict(row)
    return lookup


def numeric_score(row: Mapping[str, Any], field_name: str) -> float | None:
    value = row.get(field_name)
    if isinstance(value, (int, float)):
        value_float = float(value)
        if value_float == value_float and value_float not in {float("inf"), float("-inf")}:
            return value_float
    return None


def build_scored_whole_response_pair(
    *,
    pair_id: str,
    question: QuestionExample,
    chosen_response: MatchedResponse,
    rejected_response: MatchedResponse,
    pair_label: str,
    min_delta_f1: float,
    max_semantic_set_edit_distance: int | None = None,
    min_delta_recall: float | None = None,
    min_chosen_recall: float | None = None,
    max_entity_gap: int | None = None,
    max_entity_ratio: float | None = None,
) -> tuple[PreferencePair | None, str]:
    chosen_items = response_items(chosen_response)
    rejected_items = response_items(rejected_response)
    if not chosen_items or not rejected_items:
        return None, "empty_response"

    edit_distance = semantic_set_edit_distance(question, chosen_items, rejected_items)
    if edit_distance == 0:
        return None, "duplicate_response_sets"
    if max_semantic_set_edit_distance is not None and edit_distance > max_semantic_set_edit_distance:
        return None, "filtered_edit_distance_pairs"

    chosen_metrics = score_item_surfaces(question, chosen_items)
    rejected_metrics = score_item_surfaces(question, rejected_items)
    if chosen_metrics.f1 <= rejected_metrics.f1:
        return None, "score_metric_disagreement_pairs"

    delta_f1 = chosen_metrics.f1 - rejected_metrics.f1
    if delta_f1 < min_delta_f1:
        return None, "filtered_low_delta_pairs"

    delta_recall = chosen_metrics.recall - rejected_metrics.recall
    if min_delta_recall is not None and delta_recall < min_delta_recall:
        return None, "filtered_low_delta_recall_pairs"
    if min_chosen_recall is not None and chosen_metrics.recall < min_chosen_recall:
        return None, "filtered_low_chosen_recall_pairs"

    entity_gap = abs(len(chosen_items) - len(rejected_items))
    if max_entity_gap is not None and entity_gap > max_entity_gap:
        return None, "filtered_entity_gap_pairs"

    smaller = max(1, min(len(chosen_items), len(rejected_items)))
    entity_ratio = max(len(chosen_items), len(rejected_items)) / smaller
    if max_entity_ratio is not None and entity_ratio > max_entity_ratio:
        return None, "filtered_entity_ratio_pairs"

    return (
        PreferencePair(
            pair_id=pair_id,
            dataset=question.dataset,
            question_id=question.question_id,
            question_text=question.question_text,
            question_source_path=question.source_path,
            prompt=chosen_response.record.prompt or rejected_response.record.prompt,
            chosen=serialize_list_items(chosen_items),
            rejected=serialize_list_items(rejected_items),
            pair_type=PAIR_TYPE_WHOLE_RESPONSE_MODEL_SCORE,
            base_response_id=rejected_response.record.response_id,
            edited_candidate="",
            edited_candidate_normalized="",
            edited_gold_group_id=None,
            candidate_label=pair_label,
            candidate_label_source="complete_response_model_score_comparison",
            positive_source=None,
            semantic_set_edit_distance=edit_distance,
            chosen_items=chosen_items,
            rejected_items=rejected_items,
            chosen_precision=chosen_metrics.precision,
            chosen_recall=chosen_metrics.recall,
            chosen_f1=chosen_metrics.f1,
            rejected_precision=rejected_metrics.precision,
            rejected_recall=rejected_metrics.recall,
            rejected_f1=rejected_metrics.f1,
            delta_precision=chosen_metrics.precision - rejected_metrics.precision,
            delta_recall=delta_recall,
            delta_f1=delta_f1,
            generator_checkpoint=chosen_response.record.generator_checkpoint,
            sample_id=chosen_response.record.sample_id,
        ),
        "kept",
    )


def build_scored_pairs_for_group(
    *,
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    response_score_rows: Mapping[str, Mapping[str, Any]],
    score_field: str,
    pair_id_prefix: str,
    min_score_margin: float,
    min_delta_f1: float,
    min_response_f1: float,
    max_pairs_per_question: int,
    max_semantic_set_edit_distance: int | None,
    max_response_entities: int | None,
    min_delta_recall: float | None,
    min_chosen_recall: float | None,
    max_entity_gap: int | None,
    max_entity_ratio: float | None,
    response_dedupe_mode: str,
) -> tuple[list[PreferencePair], dict[str, Any], dict[str, dict[str, Any]]]:
    eligible_with_scores: list[tuple[MatchedResponse, float]] = []
    audit = Counter(
        {
            "responses_total": len(responses),
        }
    )

    for response in responses:
        items = response_items(response)
        if not response.pair_eligible or not items:
            continue
        if max_response_entities is not None and len(items) > max_response_entities:
            audit["filtered_large_responses"] += 1
            continue
        if response.metrics.f1 <= min_response_f1:
            audit["filtered_low_f1_responses"] += 1
            continue
        score_row = response_score_rows.get(response.record.response_id)
        if score_row is None:
            audit["responses_missing_score"] += 1
            continue
        score_value = numeric_score(score_row, score_field)
        if score_value is None:
            audit["responses_missing_score"] += 1
            continue
        eligible_with_scores.append((response, score_value))

    audit["responses_eligible"] = len(eligible_with_scores)
    seen_response_sets: set[tuple[str, ...]] = set()
    unique_responses: list[tuple[MatchedResponse, float]] = []
    for response, score_value in sorted(
        eligible_with_scores,
        key=lambda item: (
            item[1],
            item[0].metrics.f1,
            item[0].metrics.precision,
            item[0].metrics.recall,
            -item[0].metrics.prediction_count,
            normalize_clean_text(item[0].record.response_id),
        ),
        reverse=True,
    ):
        dedupe_key = response_dedupe_key(
            question,
            response_items(response),
            response_dedupe_mode=response_dedupe_mode,
        )
        if dedupe_key in seen_response_sets:
            audit["duplicate_response_sets"] += 1
            continue
        seen_response_sets.add(dedupe_key)
        unique_responses.append((response, score_value))

    audit["unique_eligible_response_sets"] = len(unique_responses)
    if "duplicate_response_sets" in audit:
        audit["duplicate_semantic_response_sets"] = audit["duplicate_response_sets"]

    pair_counter = 0
    ranked_pairs: list[tuple[tuple[Any, ...], PreferencePair]] = []
    pair_extras: dict[str, dict[str, Any]] = {}
    pair_label = f"whole_response_ranked_by_{slugify(score_field, fallback='score')}"
    for left_index, (left_response, left_score) in enumerate(unique_responses):
        for right_response, right_score in unique_responses[left_index + 1 :]:
            audit["candidate_response_pairs_total"] += 1
            if left_score >= right_score:
                chosen_response = left_response
                chosen_score = left_score
                rejected_response = right_response
                rejected_score = right_score
            else:
                chosen_response = right_response
                chosen_score = right_score
                rejected_response = left_response
                rejected_score = left_score

            score_margin = chosen_score - rejected_score
            if score_margin < min_score_margin:
                audit["filtered_low_score_margin_pairs"] += 1
                continue

            pair, reason = build_scored_whole_response_pair(
                pair_id=f"{pair_id_prefix}-score-{pair_counter:04d}",
                question=question,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                pair_label=pair_label,
                min_delta_f1=min_delta_f1,
                max_semantic_set_edit_distance=max_semantic_set_edit_distance,
                min_delta_recall=min_delta_recall,
                min_chosen_recall=min_chosen_recall,
                max_entity_gap=max_entity_gap,
                max_entity_ratio=max_entity_ratio,
            )
            pair_counter += 1
            if pair is None:
                audit[reason] += 1
                continue

            rank = (
                score_margin,
                chosen_score,
                pair.delta_f1,
                pair.chosen_f1,
                pair.chosen_precision,
                -pair.rejected_f1,
                pair.pair_id,
            )
            ranked_pairs.append((rank, pair))
            pair_extras[pair.pair_id] = {
                "score_field": score_field,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": score_margin,
            }

    selected = dedupe_and_trim_whole_response_pairs(
        question=question,
        ranked_pairs=ranked_pairs,
        max_pairs=max_pairs_per_question,
        response_dedupe_mode=response_dedupe_mode,
    )
    audit["emitted_whole_response_pairs"] = len(selected)
    return selected, dict(sorted(audit.items())), pair_extras


def build_scored_pair_audit_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    questions_by_id: Mapping[str, QuestionExample],
    sample_size: int,
    seed: int,
    score_field: str,
) -> str:
    rng = random.Random(seed)
    selected_rows = [dict(row) for row in rows]
    rng.shuffle(selected_rows)
    selected_rows = selected_rows[:sample_size]

    parts = [
        "# DCRM / Model-Scored Whole-Response Pair Audit",
        "",
        f"Selected pairs: {len(selected_rows)} / {len(rows)}",
        "",
        (
            f"Pairs are ranked by the response-level score field '{score_field}'. The chosen answer "
            "must also remain better than the rejected answer under the current matcher."
        ),
        "",
    ]
    for row in selected_rows:
        question = questions_by_id.get(clean_text(row.get("question_id")))
        evidence_preview = truncate_multiline(question.evidence if question else [], max_chars_per_item=350)
        parts.append(render_pair_block(row, evidence_preview=evidence_preview))
        parts.append(
            f"\nScore field: {clean_text(row.get('score_field')) or score_field}\n\n"
            f"Chosen score: {float(row.get('chosen_score', 0.0)):.6f}\n\n"
            f"Rejected score: {float(row.get('rejected_score', 0.0)):.6f}\n\n"
            f"Score margin: {float(row.get('score_margin', 0.0)):.6f}\n"
        )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct whole-response DPO pairs by ranking candidate-bank responses with a model "
            "score field such as dcrm_score or policy_logp_mean."
        )
    )
    parser.add_argument("--question-input", nargs="+", required=True)
    parser.add_argument("--candidate-input", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--manual-audit-md", required=True)
    parser.add_argument("--dataset-name", default="bioasq")
    parser.add_argument(
        "--score-field",
        default="dcrm_score",
        help="Candidate-bank field used to rank responses within each question.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.0,
        help="Only emit pairs whose chosen-minus-rejected score margin meets this threshold.",
    )
    parser.add_argument("--max-resources", type=int, default=3)
    parser.add_argument("--max-resource-chars", type=int, default=1200)
    parser.add_argument(
        "--gold-support-policy",
        choices=["all", "snippet"],
        default="all",
    )
    parser.add_argument(
        "--min-delta-f1",
        type=float,
        default=0.05,
        help="Only keep pairs whose matcher-based chosen F1 exceeds rejected F1 by at least this amount.",
    )
    parser.add_argument(
        "--min-response-f1",
        type=float,
        default=0.0,
        help="Only keep responses whose F1 is strictly greater than this threshold before pairing.",
    )
    parser.add_argument("--max-pairs-per-question", type=int, default=8)
    parser.add_argument("--max-semantic-set-edit-distance", type=int, default=None)
    parser.add_argument("--max-response-entities", type=int, default=None)
    parser.add_argument("--min-delta-recall", type=float, default=None)
    parser.add_argument("--min-chosen-recall", type=float, default=None)
    parser.add_argument("--max-entity-gap", type=int, default=None)
    parser.add_argument("--max-entity-ratio", type=float, default=None)
    parser.add_argument(
        "--response-dedupe-mode",
        choices=[
            RESPONSE_DEDUPE_MODE_SEMANTIC,
            RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
            RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
        ],
        default=RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
    )
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--allow-fallback-pair-construction", action="store_true")
    parser.add_argument("--allow-multiple-generator-checkpoints", action="store_true")
    parser.add_argument("--manual-audit-sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_by_id, responses_by_question, responses_by_question_and_checkpoint = group_responses_by_question(
        question_input=args.question_input,
        candidate_input=args.candidate_input,
        dataset_name=args.dataset_name,
        allow_fallback_split=bool(args.allow_fallback_split),
        allow_fallback_pair_construction=bool(args.allow_fallback_pair_construction),
        max_resources=int(args.max_resources),
        max_resource_chars=int(args.max_resource_chars),
        gold_support_policy=str(args.gold_support_policy),
        question_limit=args.limit,
    )

    generator_checkpoints = {
        clean_text(response.record.generator_checkpoint) or "unknown"
        for responses in responses_by_question.values()
        for response in responses
    }
    if len(generator_checkpoints) > 1 and not args.allow_multiple_generator_checkpoints:
        raise ValueError(
            "Multiple generator checkpoints were found in the candidate inputs. "
            "Construct pairs from one frozen generator checkpoint, or pass "
            "--allow-multiple-generator-checkpoints to partition by checkpoint. "
            f"Found: {sorted(generator_checkpoints)}"
        )

    score_lookup = load_score_lookup(args.candidate_input)
    all_pairs: list[PreferencePair] = []
    pair_audit = Counter()
    pair_extra_by_id: dict[str, dict[str, Any]] = {}

    for (question_id, checkpoint_key), responses in responses_by_question_and_checkpoint.items():
        question = questions_by_id[question_id]
        checkpoint_slug = slugify(checkpoint_key, fallback="checkpoint")
        pair_id_prefix = f"{slugify(question.dataset)}-{slugify(question.question_id)}-{checkpoint_slug}"
        response_score_rows = {
            response.record.response_id: score_lookup.get((response.record.question_id, response.record.response_id), {})
            for response in responses
        }
        pairs, audit, extras = build_scored_pairs_for_group(
            question=question,
            responses=responses,
            response_score_rows=response_score_rows,
            score_field=str(args.score_field),
            pair_id_prefix=pair_id_prefix,
            min_score_margin=float(args.min_score_margin),
            min_delta_f1=float(args.min_delta_f1),
            min_response_f1=float(args.min_response_f1),
            max_pairs_per_question=int(args.max_pairs_per_question),
            max_semantic_set_edit_distance=(
                int(args.max_semantic_set_edit_distance)
                if args.max_semantic_set_edit_distance is not None
                else None
            ),
            max_response_entities=(
                int(args.max_response_entities) if args.max_response_entities is not None else None
            ),
            min_delta_recall=(
                float(args.min_delta_recall) if args.min_delta_recall is not None else None
            ),
            min_chosen_recall=(
                float(args.min_chosen_recall) if args.min_chosen_recall is not None else None
            ),
            max_entity_gap=(
                int(args.max_entity_gap) if args.max_entity_gap is not None else None
            ),
            max_entity_ratio=(
                float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
            ),
            response_dedupe_mode=str(args.response_dedupe_mode),
        )
        all_pairs.extend(pairs)
        pair_audit.update(audit)
        pair_extra_by_id.update(extras)

    pair_rows: list[dict[str, Any]] = []
    for pair in sorted(all_pairs, key=lambda item: (item.question_id, item.pair_id)):
        row = to_jsonable(pair)
        row.update(pair_extra_by_id.get(pair.pair_id, {}))
        pair_rows.append(row)
    write_jsonl(Path(args.output_jsonl), pair_rows)

    question_gold_counts = {
        question_id: len(question.gold_groups)
        for question_id, question in questions_by_id.items()
    }
    summary = build_summary(pair_rows, question_gold_counts=question_gold_counts)
    summary["response_audit"] = build_response_audit_summary(responses_by_question)
    summary["model_score_pair_audit"] = dict(sorted(pair_audit.items()))
    summary["pair_construction_policy"] = {
        "pair_type": PAIR_TYPE_WHOLE_RESPONSE_MODEL_SCORE,
        "description": (
            "Rank complete candidate-bank responses by a model score field such as dcrm_score, "
            "then keep only score-ranked pairs whose chosen response still improves matcher-based F1."
        ),
        "score_field": str(args.score_field),
        "min_score_margin": float(args.min_score_margin),
        "min_delta_f1": float(args.min_delta_f1),
        "min_response_f1": float(args.min_response_f1),
        "max_pairs_per_question": int(args.max_pairs_per_question),
        "max_semantic_set_edit_distance": (
            int(args.max_semantic_set_edit_distance)
            if args.max_semantic_set_edit_distance is not None
            else None
        ),
        "max_response_entities": (
            int(args.max_response_entities) if args.max_response_entities is not None else None
        ),
        "min_delta_recall": (
            float(args.min_delta_recall) if args.min_delta_recall is not None else None
        ),
        "min_chosen_recall": (
            float(args.min_chosen_recall) if args.min_chosen_recall is not None else None
        ),
        "max_entity_gap": (
            int(args.max_entity_gap) if args.max_entity_gap is not None else None
        ),
        "max_entity_ratio": (
            float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
        ),
        "response_dedupe_mode": str(args.response_dedupe_mode),
        "max_resources": int(args.max_resources),
        "max_resource_chars": int(args.max_resource_chars),
        "gold_support_policy": str(args.gold_support_policy),
        "allow_fallback_split": bool(args.allow_fallback_split),
        "allow_fallback_pair_construction": bool(args.allow_fallback_pair_construction),
    }
    summary["score_margin_distribution"] = summarize_numeric(
        float(row["score_margin"]) for row in pair_rows if isinstance(row.get("score_margin"), (int, float))
    )
    summary["chosen_score_distribution"] = summarize_numeric(
        float(row["chosen_score"]) for row in pair_rows if isinstance(row.get("chosen_score"), (int, float))
    )
    summary["rejected_score_distribution"] = summarize_numeric(
        float(row["rejected_score"]) for row in pair_rows if isinstance(row.get("rejected_score"), (int, float))
    )
    summary["generator_checkpoint_distribution"] = summary["response_audit"]["generator_checkpoint_distribution"]
    write_json(Path(args.summary_json), summary)

    markdown = build_scored_pair_audit_markdown(
        rows=pair_rows,
        questions_by_id=questions_by_id,
        sample_size=int(args.manual_audit_sample_size),
        seed=int(args.seed),
        score_field=str(args.score_field),
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(pair_rows):,} model-scored whole-response pairs to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
