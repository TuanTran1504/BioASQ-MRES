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
from .schemas import (
    PAIR_TYPE_GOLD_VS_SAMPLED_RESPONSE,
    MatchedResponse,
    PreferencePair,
    QuestionExample,
    to_jsonable,
)


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


def gold_items(question: QuestionExample) -> tuple[str, ...]:
    return tuple(
        clean_text(group.canonical_alias)
        for group in question.gold_groups
        if clean_text(group.canonical_alias)
    )


def response_items(response: MatchedResponse) -> tuple[str, ...]:
    return tuple(candidate.surface for candidate in response.candidates)


def response_rank_key(response: MatchedResponse) -> tuple[float, float, float, int, str]:
    metrics = response.metrics
    return (
        metrics.f1,
        metrics.precision,
        metrics.recall,
        -metrics.prediction_count,
        clean_text(response.record.response_id),
    )


def normalized_surface_key(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            normalized
            for normalized in (normalize_answer_surface(item) for item in items)
            if normalized
        )
    )


def build_gold_response_pair(
    *,
    pair_id: str,
    question: QuestionExample,
    chosen_items: Sequence[str],
    rejected_response: MatchedResponse,
    chosen_metrics: Any,
    edit_distance: int,
) -> PreferencePair | None:
    rejected_items = response_items(rejected_response)
    if not chosen_items or not rejected_items:
        return None

    if edit_distance == 0:
        return None

    rejected_metrics = rejected_response.metrics
    if chosen_metrics.f1 <= rejected_metrics.f1:
        return None

    return PreferencePair(
        pair_id=pair_id,
        dataset=question.dataset,
        question_id=question.question_id,
        question_text=question.question_text,
        question_source_path=question.source_path,
        prompt=rejected_response.record.prompt,
        chosen=serialize_list_items(chosen_items),
        rejected=serialize_list_items(rejected_items),
        pair_type=PAIR_TYPE_GOLD_VS_SAMPLED_RESPONSE,
        base_response_id=rejected_response.record.response_id,
        edited_candidate="",
        edited_candidate_normalized="",
        edited_gold_group_id=None,
        candidate_label="model_generated_rejected_response",
        candidate_label_source="candidate_bank_vs_gold_metric_comparison",
        positive_source="gold_answer",
        semantic_set_edit_distance=edit_distance,
        chosen_items=tuple(chosen_items),
        rejected_items=tuple(rejected_items),
        chosen_precision=chosen_metrics.precision,
        chosen_recall=chosen_metrics.recall,
        chosen_f1=chosen_metrics.f1,
        rejected_precision=rejected_metrics.precision,
        rejected_recall=rejected_metrics.recall,
        rejected_f1=rejected_metrics.f1,
        delta_precision=chosen_metrics.precision - rejected_metrics.precision,
        delta_recall=chosen_metrics.recall - rejected_metrics.recall,
        delta_f1=chosen_metrics.f1 - rejected_metrics.f1,
        generator_checkpoint=rejected_response.record.generator_checkpoint,
        sample_id=rejected_response.record.sample_id,
    )


def dedupe_and_trim_gold_response_pairs(
    *,
    question: QuestionExample,
    ranked_pairs: Sequence[tuple[tuple[float, float, float, int, str], PreferencePair]],
    max_pairs: int,
) -> tuple[list[PreferencePair], Counter]:
    audit = Counter()
    seen_semantic_keys = set()
    seen_normalized_keys = set()
    selected: list[PreferencePair] = []

    for _rank, pair in sorted(ranked_pairs, key=lambda item: item[0], reverse=True):
        semantic_key = (
            pair.question_id,
            semantic_set_key_for_question(question, pair.chosen_items),
            semantic_set_key_for_question(question, pair.rejected_items),
            pair.pair_type,
        )
        normalized_key = (
            pair.question_id,
            normalized_surface_key(pair.chosen_items),
            normalized_surface_key(pair.rejected_items),
            pair.pair_type,
        )
        if semantic_key in seen_semantic_keys:
            audit["filtered_duplicate_semantic_pairs_after_ranking"] += 1
            continue
        if normalized_key in seen_normalized_keys:
            audit["filtered_duplicate_normalized_pairs_after_ranking"] += 1
            continue

        seen_semantic_keys.add(semantic_key)
        seen_normalized_keys.add(normalized_key)
        selected.append(pair)
        if max_pairs > 0 and len(selected) >= max_pairs:
            break

    return selected, audit


def build_gold_response_pairs_for_group(
    *,
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    pair_id_prefix: str,
    min_delta_f1: float,
    max_pairs_per_question: int,
    min_chosen_f1: float,
    max_semantic_set_edit_distance: int,
    max_rejected_items: int,
    min_rejected_f1: float,
    max_rejected_f1: float,
    max_entity_gap: int | None,
    max_entity_ratio: float | None,
) -> tuple[list[PreferencePair], dict[str, Any]]:
    chosen_items = gold_items(question)
    chosen_semantic_key = semantic_set_key_for_question(question, chosen_items)
    chosen_metrics = score_item_surfaces(question, chosen_items)
    audit = Counter(
        {
            "responses_total": len(responses),
            "gold_reference_item_count": len(chosen_items),
        }
    )
    if not chosen_items:
        audit["filtered_questions_without_gold_items"] = 1
        return [], dict(sorted(audit.items()))
    if chosen_metrics.f1 < min_chosen_f1:
        audit["filtered_questions_below_min_chosen_f1"] = 1
        return [], dict(sorted(audit.items()))

    ranked_pairs: list[tuple[tuple[float, float, float, int, str], PreferencePair]] = []
    seen_rejected_semantic_sets: set[tuple[str, ...]] = set()
    pair_counter = 0

    for response in sorted(responses, key=response_rank_key, reverse=True):
        if not response.pair_eligible:
            audit["filtered_ineligible_responses"] += 1
            continue

        rejected_items = response_items(response)
        if not rejected_items:
            audit["filtered_empty_rejected_sets"] += 1
            continue
        if max_rejected_items > 0 and len(rejected_items) > max_rejected_items:
            audit["filtered_large_rejected_sets"] += 1
            continue
        if response.metrics.f1 < min_rejected_f1:
            audit["filtered_low_rejected_f1"] += 1
            continue
        if response.metrics.f1 > max_rejected_f1:
            audit["filtered_high_rejected_f1"] += 1
            continue

        entity_gap = abs(len(chosen_items) - len(rejected_items))
        if max_entity_gap is not None and entity_gap > max_entity_gap:
            audit["filtered_large_entity_gap"] += 1
            continue

        smaller = max(1, min(len(chosen_items), len(rejected_items)))
        entity_ratio = max(len(chosen_items), len(rejected_items)) / smaller
        if max_entity_ratio is not None and entity_ratio > max_entity_ratio:
            audit["filtered_large_entity_ratio"] += 1
            continue

        rejected_semantic_key = semantic_set_key_for_question(question, rejected_items)
        if rejected_semantic_key == chosen_semantic_key:
            audit["filtered_gold_equivalent_responses"] += 1
            continue
        if rejected_semantic_key in seen_rejected_semantic_sets:
            audit["filtered_duplicate_rejected_semantic_sets"] += 1
            continue

        edit_distance = semantic_set_edit_distance(question, chosen_items, rejected_items)
        if max_semantic_set_edit_distance > 0 and edit_distance > max_semantic_set_edit_distance:
            audit["filtered_large_semantic_edit_distance"] += 1
            continue

        delta_f1 = chosen_metrics.f1 - response.metrics.f1
        if delta_f1 < min_delta_f1:
            audit["filtered_low_delta_pairs"] += 1
            continue

        pair = build_gold_response_pair(
            pair_id=f"{pair_id_prefix}-goldresp-{pair_counter:04d}",
            question=question,
            chosen_items=chosen_items,
            rejected_response=response,
            chosen_metrics=chosen_metrics,
            edit_distance=edit_distance,
        )
        if pair is None:
            audit["filtered_invalid_pairs"] += 1
            continue

        pair_counter += 1
        seen_rejected_semantic_sets.add(rejected_semantic_key)
        rank = (
            pair.rejected_f1,
            pair.rejected_precision,
            pair.rejected_recall,
            -len(pair.rejected_items),
            pair.pair_id,
        )
        ranked_pairs.append((rank, pair))

    selected_pairs, dedupe_audit = dedupe_and_trim_gold_response_pairs(
        question=question,
        ranked_pairs=ranked_pairs,
        max_pairs=max_pairs_per_question,
    )

    audit["unique_non_gold_rejected_response_sets"] = len(ranked_pairs)
    audit.update(dedupe_audit)
    audit["emitted_gold_response_pairs"] = len(selected_pairs)
    return selected_pairs, dict(sorted(audit.items()))


def build_gold_response_audit_markdown(
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
        "# Gold-vs-Candidate DPO Pair Audit",
        "",
        f"Selected pairs: {len(selected_rows)} / {len(rows)}",
        "",
        (
            "Chosen answers are the canonical BioASQ gold lists. Rejected answers are unique "
            "candidate-bank responses sampled from the frozen generator."
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
        description=(
            "Construct paper-style model-based DPO pairs where the chosen answer is the "
            "BioASQ gold list and the rejected answer is a sampled candidate-bank response."
        )
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
        help="Candidate-bank JSON/JSONL or evaluation predictions JSON files.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Preference-pair JSONL output path.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Audit summary JSON output path.",
    )
    parser.add_argument(
        "--manual-audit-md",
        required=True,
        help="Manual audit markdown output path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label stored in emitted rows.",
    )
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
        help="Only emit pairs where the gold list exceeds the sampled response by at least this F1 margin.",
    )
    parser.add_argument(
        "--max-pairs-per-question",
        type=int,
        default=0,
        help="Maximum pairs emitted per question/checkpoint group. Use 0 to keep all unique non-gold responses.",
    )
    parser.add_argument(
        "--min-chosen-f1",
        type=float,
        default=1.0,
        help="Minimum matcher F1 the canonical gold chosen side must achieve. Default 1.0 drops noisy gold surfaces.",
    )
    parser.add_argument(
        "--max-semantic-set-edit-distance",
        type=int,
        default=20,
        help="Maximum matcher-derived semantic set edit distance allowed for emitted pairs. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-rejected-items",
        type=int,
        default=30,
        help="Maximum number of items allowed in the rejected response. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-rejected-f1",
        type=float,
        default=0.1,
        help="Minimum matcher F1 allowed for the rejected response.",
    )
    parser.add_argument(
        "--max-rejected-f1",
        type=float,
        default=0.8,
        help="Maximum matcher F1 allowed for the rejected response.",
    )
    parser.add_argument(
        "--max-entity-gap",
        type=int,
        default=None,
        help="Optional maximum absolute item-count gap between the gold chosen list and rejected response.",
    )
    parser.add_argument(
        "--max-entity-ratio",
        type=float,
        default=None,
        help="Optional maximum ratio between the larger and smaller item counts of the chosen/rejected sides.",
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
        pairs, audit = build_gold_response_pairs_for_group(
            question=question,
            responses=responses,
            pair_id_prefix=pair_id_prefix,
            min_delta_f1=float(args.min_delta_f1),
            max_pairs_per_question=int(args.max_pairs_per_question),
            min_chosen_f1=float(args.min_chosen_f1),
            max_semantic_set_edit_distance=int(args.max_semantic_set_edit_distance),
            max_rejected_items=int(args.max_rejected_items),
            min_rejected_f1=float(args.min_rejected_f1),
            max_rejected_f1=float(args.max_rejected_f1),
            max_entity_gap=(
                int(args.max_entity_gap) if args.max_entity_gap is not None else None
            ),
            max_entity_ratio=(
                float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
            ),
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
    summary["gold_response_pair_audit"] = dict(sorted(pair_audit.items()))
    summary["pair_construction_policy"] = {
        "pair_type": PAIR_TYPE_GOLD_VS_SAMPLED_RESPONSE,
        "description": (
            "Model-based preference pairs where the chosen side is the canonical BioASQ gold list "
            "and the rejected side is a sampled candidate-bank response."
        ),
        "chosen_source": "question.gold_groups.canonical_alias",
        "rejected_source": "candidate_bank_response",
        "min_delta_f1": float(args.min_delta_f1),
        "max_pairs_per_question": int(args.max_pairs_per_question),
        "min_chosen_f1": float(args.min_chosen_f1),
        "max_semantic_set_edit_distance": int(args.max_semantic_set_edit_distance),
        "max_rejected_items": int(args.max_rejected_items),
        "min_rejected_f1": float(args.min_rejected_f1),
        "max_rejected_f1": float(args.max_rejected_f1),
        "max_entity_gap": (
            int(args.max_entity_gap) if args.max_entity_gap is not None else None
        ),
        "max_entity_ratio": (
            float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
        ),
        "max_resources": int(args.max_resources),
        "max_resource_chars": int(args.max_resource_chars),
        "gold_support_policy": str(args.gold_support_policy),
        "allow_fallback_split": bool(args.allow_fallback_split),
        "allow_fallback_pair_construction": bool(args.allow_fallback_pair_construction),
        "checkpoint_partitioned": True,
    }
    summary["gold_chosen_f1_distribution"] = summarize_numeric(
        row["chosen_f1"] for row in pair_rows
    )
    summary["rejected_response_f1_distribution"] = summarize_numeric(
        row["rejected_f1"] for row in pair_rows
    )
    summary["rejected_response_precision_distribution"] = summarize_numeric(
        row["rejected_precision"] for row in pair_rows
    )
    summary["rejected_response_recall_distribution"] = summarize_numeric(
        row["rejected_recall"] for row in pair_rows
    )
    summary["generator_checkpoint_distribution"] = summary["response_audit"]["generator_checkpoint_distribution"]
    summary["required_checks"]["all_chosen_sides_perfect_f1_under_current_matcher"] = all(
        float(row["chosen_f1"]) == 1.0 for row in pair_rows
    )
    summary["required_checks"]["all_pairs_within_max_semantic_edit_distance_filter"] = (
        all(
            int(row["semantic_set_edit_distance"]) <= int(args.max_semantic_set_edit_distance)
            for row in pair_rows
        )
        if int(args.max_semantic_set_edit_distance) > 0
        else True
    )
    summary["required_checks"]["all_pairs_within_max_rejected_items_filter"] = (
        all(
            len(row["rejected_items"]) <= int(args.max_rejected_items)
            for row in pair_rows
        )
        if int(args.max_rejected_items) > 0
        else True
    )
    if args.max_entity_gap is not None:
        summary["required_checks"]["all_pairs_within_max_entity_gap_filter"] = all(
            abs(len(row["chosen_items"]) - len(row["rejected_items"])) <= int(args.max_entity_gap)
            for row in pair_rows
        )
    if args.max_entity_ratio is not None:
        summary["required_checks"]["all_pairs_within_max_entity_ratio_filter"] = all(
            (
                max(len(row["chosen_items"]), len(row["rejected_items"]))
                / max(1, min(len(row["chosen_items"]), len(row["rejected_items"])))
            )
            <= float(args.max_entity_ratio)
            for row in pair_rows
        )
    summary["check_interpretation"]["chosen_side_scope"] = (
        "Chosen sides are canonical gold-answer surfaces derived from the loaded BioASQ gold groups."
    )
    write_json(Path(args.summary_json), summary)

    markdown = build_gold_response_audit_markdown(
        rows=pair_rows,
        questions_by_id=questions_by_id,
        sample_size=int(args.manual_audit_sample_size),
        seed=int(args.seed),
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(pair_rows):,} gold-vs-candidate DPO pairs to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
