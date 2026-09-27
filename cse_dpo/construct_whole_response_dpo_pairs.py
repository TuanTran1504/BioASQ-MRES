from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_registry import slugify
from src.utility.data import clean_text

from .audit_preference_pairs import build_summary, render_pair_block
from .common import summarize_numeric, truncate_multiline, write_json, write_jsonl
from .construct_set_edit_pairs import build_response_audit_summary, group_responses_by_question
from .match_gold_groups import score_item_surfaces, semantic_set_key_for_question
from .normalize_set_answers import normalize_answer_surface, serialize_list_items
from .schemas import MatchedResponse, PreferencePair, QuestionExample, to_jsonable

PAIR_TYPE_WHOLE_RESPONSE_METRIC = "whole_response_metric"
RESPONSE_DEDUPE_MODE_SEMANTIC = "semantic"
RESPONSE_DEDUPE_MODE_EXACT_UNORDERED = "exact_unordered"
RESPONSE_DEDUPE_MODE_EXACT_ORDERED = "exact_ordered"


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


def response_items(response: MatchedResponse) -> tuple[str, ...]:
    return tuple(candidate.surface for candidate in response.candidates)


def exact_response_key(
    items: Sequence[str],
    *,
    preserve_order: bool,
) -> tuple[str, ...]:
    normalized_items = [
        normalized
        for item in items
        if (normalized := normalize_answer_surface(item))
    ]
    if not preserve_order:
        normalized_items = sorted(normalized_items)
    return tuple(normalized_items)


def response_dedupe_key(
    question: QuestionExample,
    items: Sequence[str],
    *,
    response_dedupe_mode: str,
) -> tuple[str, ...]:
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_SEMANTIC:
        return semantic_set_key_for_question(question, items)
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_EXACT_UNORDERED:
        return exact_response_key(items, preserve_order=False)
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_EXACT_ORDERED:
        return exact_response_key(items, preserve_order=True)
    raise ValueError(f"Unsupported response_dedupe_mode: {response_dedupe_mode}")


def response_rank_key(response: MatchedResponse) -> tuple[float, float, float, int, str]:
    metrics = response.metrics
    return (
        metrics.f1,
        metrics.precision,
        metrics.recall,
        -metrics.prediction_count,
        clean_text(response.record.response_id),
    )


def build_whole_response_pair(
    *,
    pair_id: str,
    question: QuestionExample,
    chosen_response: MatchedResponse,
    rejected_response: MatchedResponse,
    max_semantic_set_edit_distance: int | None = None,
    min_delta_recall: float | None = None,
    min_chosen_recall: float | None = None,
    max_entity_gap: int | None = None,
    max_entity_ratio: float | None = None,
) -> PreferencePair | None:
    chosen_items = response_items(chosen_response)
    rejected_items = response_items(rejected_response)
    if not chosen_items or not rejected_items:
        return None

    edit_distance = semantic_set_edit_distance(question, chosen_items, rejected_items)
    if edit_distance == 0:
        return None
    if max_semantic_set_edit_distance is not None and edit_distance > max_semantic_set_edit_distance:
        return None

    chosen_metrics = score_item_surfaces(question, chosen_items)
    rejected_metrics = score_item_surfaces(question, rejected_items)
    if chosen_metrics.f1 <= rejected_metrics.f1:
        return None
    delta_recall = chosen_metrics.recall - rejected_metrics.recall
    if min_delta_recall is not None and delta_recall < min_delta_recall:
        return None
    if min_chosen_recall is not None and chosen_metrics.recall < min_chosen_recall:
        return None

    entity_gap = abs(len(chosen_items) - len(rejected_items))
    if max_entity_gap is not None and entity_gap > max_entity_gap:
        return None
    smaller = max(1, min(len(chosen_items), len(rejected_items)))
    entity_ratio = max(len(chosen_items), len(rejected_items)) / smaller
    if max_entity_ratio is not None and entity_ratio > max_entity_ratio:
        return None

    return PreferencePair(
        pair_id=pair_id,
        dataset=question.dataset,
        question_id=question.question_id,
        question_text=question.question_text,
        question_source_path=question.source_path,
        prompt=chosen_response.record.prompt or rejected_response.record.prompt,
        chosen=serialize_list_items(chosen_items),
        rejected=serialize_list_items(rejected_items),
        pair_type=PAIR_TYPE_WHOLE_RESPONSE_METRIC,
        base_response_id=rejected_response.record.response_id,
        edited_candidate="",
        edited_candidate_normalized="",
        edited_gold_group_id=None,
        candidate_label="whole_response_ranked_by_f1",
        candidate_label_source="complete_response_metric_comparison",
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
        delta_recall=chosen_metrics.recall - rejected_metrics.recall,
        delta_f1=chosen_metrics.f1 - rejected_metrics.f1,
        generator_checkpoint=chosen_response.record.generator_checkpoint,
        sample_id=chosen_response.record.sample_id,
    )


def build_whole_response_pairs_for_group(
    *,
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    pair_id_prefix: str,
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
) -> tuple[list[PreferencePair], dict[str, Any]]:
    eligible_responses: list[MatchedResponse] = []
    filtered_large_responses = 0
    filtered_low_f1_responses = 0
    for response in responses:
        items = response_items(response)
        if not response.pair_eligible or not items:
            continue
        if max_response_entities is not None and len(items) > max_response_entities:
            filtered_large_responses += 1
            continue
        if response.metrics.f1 <= min_response_f1:
            filtered_low_f1_responses += 1
            continue
        eligible_responses.append(response)

    audit = Counter(
        {
            "responses_total": len(responses),
            "responses_eligible": len(eligible_responses),
            "filtered_large_responses": filtered_large_responses,
            "filtered_low_f1_responses": filtered_low_f1_responses,
        }
    )
    ranked_pairs: list[tuple[tuple[float, float, float, float, str], PreferencePair]] = []
    seen_response_sets: set[tuple[str, ...]] = set()
    unique_responses: list[MatchedResponse] = []

    for response in sorted(eligible_responses, key=response_rank_key, reverse=True):
        dedupe_key = response_dedupe_key(
            question,
            response_items(response),
            response_dedupe_mode=response_dedupe_mode,
        )
        if dedupe_key in seen_response_sets:
            audit["duplicate_response_sets"] += 1
            continue
        seen_response_sets.add(dedupe_key)
        unique_responses.append(response)

    audit["unique_eligible_response_sets"] = len(unique_responses)
    if "duplicate_response_sets" in audit:
        # Preserve the historical audit key for downstream notebooks that still expect it.
        audit["duplicate_semantic_response_sets"] = audit["duplicate_response_sets"]
    pair_counter = 0
    for left_index, left_response in enumerate(unique_responses):
        for right_response in unique_responses[left_index + 1 :]:
            if response_rank_key(left_response) >= response_rank_key(right_response):
                chosen_response = left_response
                rejected_response = right_response
            else:
                chosen_response = right_response
                rejected_response = left_response

            delta_f1 = chosen_response.metrics.f1 - rejected_response.metrics.f1
            audit["candidate_response_pairs_total"] += 1
            if delta_f1 < min_delta_f1:
                audit["filtered_low_delta_pairs"] += 1
                continue

            pair = build_whole_response_pair(
                pair_id=f"{pair_id_prefix}-whole-{pair_counter:04d}",
                question=question,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                max_semantic_set_edit_distance=max_semantic_set_edit_distance,
                min_delta_recall=min_delta_recall,
                min_chosen_recall=min_chosen_recall,
                max_entity_gap=max_entity_gap,
                max_entity_ratio=max_entity_ratio,
            )
            pair_counter += 1
            if pair is None:
                edit_distance = semantic_set_edit_distance(
                    question,
                    response_items(chosen_response),
                    response_items(rejected_response),
                )
                if max_semantic_set_edit_distance is not None and edit_distance > max_semantic_set_edit_distance:
                    audit["filtered_edit_distance_pairs"] += 1
                else:
                    audit["filtered_invalid_pairs"] += 1
                continue

            rank = (
                pair.delta_f1,
                pair.chosen_f1,
                pair.chosen_precision,
                -pair.rejected_f1,
                pair.pair_id,
            )
            ranked_pairs.append((rank, pair))

    selected = dedupe_and_trim_whole_response_pairs(
        question=question,
        ranked_pairs=ranked_pairs,
        max_pairs=max_pairs_per_question,
        response_dedupe_mode=response_dedupe_mode,
    )
    audit["emitted_whole_response_pairs"] = len(selected)
    return selected, dict(sorted(audit.items()))


def dedupe_and_trim_whole_response_pairs(
    *,
    question: QuestionExample,
    ranked_pairs: Sequence[tuple[tuple[Any, ...], PreferencePair]],
    max_pairs: int,
    response_dedupe_mode: str,
) -> list[PreferencePair]:
    seen = set()
    selected: list[PreferencePair] = []
    for _rank, pair in sorted(ranked_pairs, key=lambda item: item[0], reverse=True):
        key = (
            pair.question_id,
            response_dedupe_key(
                question,
                pair.chosen_items,
                response_dedupe_mode=response_dedupe_mode,
            ),
            response_dedupe_key(
                question,
                pair.rejected_items,
                response_dedupe_mode=response_dedupe_mode,
            ),
            pair.pair_type,
        )
        reverse_key = (key[0], key[2], key[1], key[3])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def build_whole_response_audit_markdown(
    rows: Sequence[Mapping[str, Any]],
    questions_by_id: Mapping[str, QuestionExample],
    sample_size: int,
    seed: int,
) -> str:
    rng = random.Random(seed)
    selected_rows = [dict(row) for row in rows]
    rng.shuffle(selected_rows)
    selected_rows = selected_rows[:sample_size]

    parts = [
        "# Whole-Response DPO Pair Audit",
        "",
        f"Selected pairs: {len(selected_rows)} / {len(rows)}",
        "",
        (
            "Whole-response pairs compare complete sampled answers. The chosen answer has higher "
            "BioASQ-style F1 than the rejected answer under the current matcher."
        ),
        "",
    ]
    for row in selected_rows:
        question = questions_by_id.get(clean_text(row.get("question_id")))
        evidence_preview = truncate_multiline(question.evidence if question else [], max_chars_per_item=350)
        parts.append(render_pair_block(row, evidence_preview=evidence_preview))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct whole-response DPO pairs by ranking complete candidate-bank rollouts by BioASQ F1."
    )
    parser.add_argument(
        "--question-input",
        nargs="+",
        required=True,
        help="Raw BioASQ JSON or prepared JSON files containing list questions and gold answers.",
    )
    parser.add_argument(
        "--candidate-input",
        nargs="+",
        required=True,
        help="Candidate-bank JSON/JSONL files from a frozen generator.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Preference-pair JSONL output path.")
    parser.add_argument("--summary-json", required=True, help="Audit summary JSON output path.")
    parser.add_argument("--manual-audit-md", required=True, help="Manual audit markdown output path.")
    parser.add_argument("--dataset-name", default="bioasq", help="Dataset label stored in emitted rows.")
    parser.add_argument(
        "--max-resources",
        type=int,
        default=3,
        help="Maximum number of PubMed resources to use when loading raw question evidence. Use 0 for all resources.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=1200,
        help="Maximum characters per serialized PubMed resource when loading raw question evidence. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--gold-support-policy",
        choices=["all", "snippet"],
        default="all",
        help="Use all gold groups, or only gold groups whose aliases appear in the loaded snippets.",
    )
    parser.add_argument(
        "--min-delta-f1",
        type=float,
        default=0.05,
        help="Only emit pairs where chosen F1 exceeds rejected F1 by at least this amount.",
    )
    parser.add_argument(
        "--min-response-f1",
        type=float,
        default=0.0,
        help="Only keep responses whose F1 is strictly greater than this threshold before pairing.",
    )
    parser.add_argument(
        "--max-pairs-per-question",
        type=int,
        default=8,
        help="Maximum whole-response pairs emitted per question/checkpoint group.",
    )
    parser.add_argument(
        "--max-semantic-set-edit-distance",
        type=int,
        default=None,
        help="Optional cap on semantic set edit distance between chosen and rejected whole responses.",
    )
    parser.add_argument(
        "--max-response-entities",
        type=int,
        default=None,
        help="Optional cap on the number of extracted entities allowed in any response used for pairing.",
    )
    parser.add_argument(
        "--min-delta-recall",
        type=float,
        default=None,
        help=(
            "Optional minimum recall improvement required from chosen over rejected. "
            "Use 0.0 to forbid recall regressions and a positive value to prefer recall-improving pairs."
        ),
    )
    parser.add_argument(
        "--min-chosen-recall",
        type=float,
        default=None,
        help="Optional absolute minimum recall required for the chosen response.",
    )
    parser.add_argument(
        "--max-entity-gap",
        type=int,
        default=None,
        help="Optional cap on the absolute entity-count difference between chosen and rejected responses.",
    )
    parser.add_argument(
        "--max-entity-ratio",
        type=float,
        default=None,
        help=(
            "Optional cap on the larger/smaller entity-count ratio between chosen and rejected responses. "
            "Useful for dropping extreme short-vs-long pairs."
        ),
    )
    parser.add_argument(
        "--response-dedupe-mode",
        choices=[
            RESPONSE_DEDUPE_MODE_SEMANTIC,
            RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
            RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
        ],
        default=RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
        help=(
            "How to deduplicate candidate responses before pairing and when trimming duplicate "
            "pairs. 'exact_unordered' keeps duplicate items but ignores item order, "
            "'exact_ordered' treats different orders as distinct, and 'semantic' preserves "
            "the previous question-aware semantic deduplication."
        ),
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing for untagged outputs.",
    )
    parser.add_argument(
        "--allow-fallback-pair-construction",
        action="store_true",
        help="Allow fallback-split responses to seed pairs. Malformed and empty responses remain excluded.",
    )
    parser.add_argument(
        "--allow-multiple-generator-checkpoints",
        action="store_true",
        help="Allow multiple generator checkpoints. Responses are still paired only within the same checkpoint.",
    )
    parser.add_argument(
        "--manual-audit-sample-size",
        type=int,
        default=80,
        help="How many pairs to include in the markdown audit.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed used for audit sampling.")
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit for smoke tests.")
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

    all_pairs: list[PreferencePair] = []
    pair_audit = Counter()
    for (question_id, _checkpoint), responses in responses_by_question_and_checkpoint.items():
        question = questions_by_id[question_id]
        checkpoint_slug = slugify(clean_text(responses[0].record.generator_checkpoint) or "checkpoint")
        pair_id_prefix = f"{slugify(question.dataset)}-{slugify(question.question_id)}-{checkpoint_slug}"
        pairs, audit = build_whole_response_pairs_for_group(
            question=question,
            responses=responses,
            pair_id_prefix=pair_id_prefix,
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

    pair_rows = [
        to_jsonable(pair)
        for pair in sorted(all_pairs, key=lambda item: (item.question_id, item.pair_id))
    ]
    write_jsonl(Path(args.output_jsonl), pair_rows)

    question_gold_counts = {
        question_id: len(question.gold_groups)
        for question_id, question in questions_by_id.items()
    }
    summary = build_summary(pair_rows, question_gold_counts=question_gold_counts)
    summary["response_audit"] = build_response_audit_summary(responses_by_question)
    summary["whole_response_pair_audit"] = dict(sorted(pair_audit.items()))
    summary["pair_construction_policy"] = {
        "pair_type": PAIR_TYPE_WHOLE_RESPONSE_METRIC,
        "description": "Rank complete candidate-bank responses by BioASQ-style F1 and prefer higher-F1 responses.",
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
    summary["whole_response_delta_precision_distribution"] = summarize_numeric(
        row["delta_precision"] for row in pair_rows
    )
    summary["whole_response_delta_recall_distribution"] = summarize_numeric(
        row["delta_recall"] for row in pair_rows
    )
    summary["generator_checkpoint_distribution"] = summary["response_audit"]["generator_checkpoint_distribution"]
    write_json(Path(args.summary_json), summary)

    markdown = build_whole_response_audit_markdown(
        rows=pair_rows,
        questions_by_id=questions_by_id,
        sample_size=int(args.manual_audit_sample_size),
        seed=int(args.seed),
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(pair_rows):,} whole-response DPO pairs to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
