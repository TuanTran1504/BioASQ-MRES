from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utility.data import clean_text
from src.model_registry import slugify

from .audit_preference_pairs import build_manual_audit_markdown, build_summary
from .common import write_json, write_jsonl
from .match_gold_groups import (
    best_surface_semantic_match,
    build_matched_response,
    load_candidate_bank_records,
    load_question_examples,
    semantic_set_key_for_question,
    score_item_surfaces,
    surface_match_aliases,
)
from .normalize_set_answers import normalize_answer_surface, normalize_evidence_text, serialize_list_items
from .schemas import (
    LABEL_GOLD_MISSING,
    LABEL_METRIC_NEGATIVE,
    PAIR_TYPE_NEGATIVE_ADDITION,
    PAIR_TYPE_VALID_OMISSION,
    MatchedCandidate,
    MatchedResponse,
    PreferencePair,
    QuestionExample,
    to_jsonable,
)

NON_INFORMATIVE_SINGLE_TOKEN_OVERLAP_TOKENS = {
    "cancer",
    "cell",
    "cells",
    "disease",
    "disorder",
    "factor",
    "gene",
    "genes",
    "group",
    "marker",
    "markers",
    "protein",
    "proteins",
    "receptor",
    "receptors",
    "syndrome",
    "tumor",
    "tumour",
}


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


def token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in normalize_answer_surface(text).split() if token)


def normalized_phrase_in_text(phrase: str, evidence_text: str, evidence_text_compact: str) -> bool:
    cleaned_phrase = clean_text(phrase)
    if not cleaned_phrase:
        return False
    if " " in cleaned_phrase:
        return f" {cleaned_phrase} " in f" {evidence_text} "
    return f" {cleaned_phrase} " in f" {evidence_text} " or cleaned_phrase in evidence_text_compact


def is_informative_single_overlap_token(token: str) -> bool:
    cleaned = clean_text(token)
    if not cleaned or cleaned in NON_INFORMATIVE_SINGLE_TOKEN_OVERLAP_TOKENS:
        return False
    return len(cleaned) >= 6 or any(char.isdigit() for char in cleaned)


def token_sequence_contained(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if not shorter or len(shorter) > len(longer):
        return False
    width = len(shorter)
    for start in range(len(longer) - width + 1):
        if tuple(longer[start : start + width]) == tuple(shorter):
            return True
    return False


def surface_overlap_reason(
    question: QuestionExample,
    left: str,
    right: str,
) -> str | None:
    left_aliases = set(surface_match_aliases(left, question.evidence))
    right_aliases = set(surface_match_aliases(right, question.evidence))
    if left_aliases & right_aliases:
        return "expanded_alias_overlap"

    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return None

    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    if not token_sequence_contained(shorter, longer):
        return None
    if len(shorter) >= 2:
        return "token_subsequence_overlap"
    if is_informative_single_overlap_token(shorter[0]):
        return "token_subsequence_overlap"
    return None


def surface_represented_in_items(
    question: QuestionExample,
    surface: str,
    items: Sequence[str],
) -> str | None:
    best_match = best_surface_semantic_match(question, surface, items)
    if best_match is not None:
        return best_match.match_type
    for item in items:
        overlap_reason = surface_overlap_reason(question, surface, item)
        if overlap_reason is not None:
            return overlap_reason
    return None


def surface_overlaps_gold_alias(
    question: QuestionExample,
    surface: str,
) -> str | None:
    for gold_group in question.gold_groups:
        best_match = best_surface_semantic_match(question, surface, gold_group.aliases)
        if best_match is not None:
            return best_match.match_type
    for gold_group in question.gold_groups:
        for alias in gold_group.aliases:
            overlap_reason = surface_overlap_reason(question, surface, alias)
            if overlap_reason is not None:
                return overlap_reason
    return None


def candidate_is_evidence_supported(
    question: QuestionExample,
    surface: str,
    question_stats: Mapping[str, Any],
) -> bool:
    evidence_text = question_stats["evidence_text"]
    evidence_text_compact = question_stats["evidence_text_compact"]
    for alias in surface_match_aliases(surface, question.evidence):
        if normalized_phrase_in_text(alias, evidence_text=evidence_text, evidence_text_compact=evidence_text_compact):
            return True
    return False


def build_question_statistics(
    responses: Sequence[MatchedResponse],
    question: QuestionExample,
) -> dict[str, Any]:
    negative_frequency = Counter()
    valid_surface_by_gold_group = defaultdict(Counter)
    evidence_text = normalize_evidence_text(question.evidence)

    for response in responses:
        if not response.pair_eligible:
            continue
        for candidate in response.candidates:
            if candidate.label == LABEL_METRIC_NEGATIVE:
                negative_frequency[candidate.normalized] += 1
            elif candidate.matched_gold_group_id is not None:
                valid_surface_by_gold_group[candidate.matched_gold_group_id][candidate.surface] += 1

    return {
        "negative_frequency": negative_frequency,
        "valid_surface_by_gold_group": valid_surface_by_gold_group,
        "evidence_text": evidence_text,
        "evidence_text_compact": evidence_text.replace(" ", ""),
    }


def negative_candidate_priority(
    candidate: MatchedCandidate,
    question: QuestionExample,
    question_stats: Mapping[str, Any],
) -> tuple[float, int, int, int, str]:
    negative_frequency = question_stats["negative_frequency"]
    evidence_supported = candidate_is_evidence_supported(question, candidate.surface, question_stats)
    lexical_similarity = max(
        (
            token_overlap_score(candidate.normalized, alias)
            for gold_group in question.gold_groups
            for alias in gold_group.normalized_aliases
        ),
        default=0.0,
    )
    return (
        1 if not evidence_supported else 0,
        float(negative_frequency[candidate.normalized]),
        -int(round(lexical_similarity * 1000)),
        -candidate.duplicate_count,
        candidate.normalized,
    )


def choose_positive_surface(
    question: QuestionExample,
    response: MatchedResponse,
    gold_group_id: int,
    question_stats: Mapping[str, Any],
    allow_gold_alias_fallback: bool,
    allow_semantic_overlap_omission_pairs: bool,
    allow_evidence_unsupported_omission_pairs: bool,
) -> tuple[str | None, str | None, str | None]:
    rejected_items = [candidate.surface for candidate in response.candidates]
    surface_counter = question_stats["valid_surface_by_gold_group"].get(gold_group_id, Counter())
    overlap_filtered = False
    for surface, _count in surface_counter.most_common():
        normalized = normalize_answer_surface(surface)
        if not normalized:
            continue
        evidence_supported = candidate_is_evidence_supported(question, surface, question_stats)
        if not evidence_supported and not allow_evidence_unsupported_omission_pairs:
            continue
        overlap_reason = surface_represented_in_items(question, surface, rejected_items)
        if overlap_reason is not None and not allow_semantic_overlap_omission_pairs:
            overlap_filtered = True
            continue
        return surface, "sampled_valid_candidate", None

    if allow_gold_alias_fallback:
        gold_group = next(group for group in question.gold_groups if group.group_id == gold_group_id)
        normalized = normalize_answer_surface(gold_group.canonical_alias)
        if normalized:
            overlap_reason = surface_represented_in_items(question, gold_group.canonical_alias, rejected_items)
            if overlap_reason is not None and not allow_semantic_overlap_omission_pairs:
                overlap_filtered = True
            else:
                return gold_group.canonical_alias, "gold_alias_fallback", None

    if overlap_filtered:
        return None, None, "semantic_overlap_with_rejected_set"
    return None, None, "positive_surface_not_supported_by_evidence"


def build_pair(
    *,
    pair_id: str,
    pair_type: str,
    question: QuestionExample,
    response: MatchedResponse,
    chosen_items: Sequence[str],
    rejected_items: Sequence[str],
    edited_candidate: str,
    edited_candidate_normalized: str,
    edited_gold_group_id: int | None,
    candidate_label: str,
    candidate_label_source: str,
    positive_source: str | None,
) -> PreferencePair | None:
    edit_distance = semantic_set_edit_distance(question, chosen_items, rejected_items)
    if edit_distance != 1:
        return None

    chosen_metrics = score_item_surfaces(question, chosen_items)
    rejected_metrics = score_item_surfaces(question, rejected_items)
    if chosen_metrics.f1 <= rejected_metrics.f1:
        return None

    return PreferencePair(
        pair_id=pair_id,
        dataset=question.dataset,
        question_id=question.question_id,
        question_text=question.question_text,
        question_source_path=question.source_path,
        prompt=response.record.prompt,
        chosen=serialize_list_items(chosen_items),
        rejected=serialize_list_items(rejected_items),
        pair_type=pair_type,
        base_response_id=response.record.response_id,
        edited_candidate=edited_candidate,
        edited_candidate_normalized=edited_candidate_normalized,
        edited_gold_group_id=edited_gold_group_id,
        candidate_label=candidate_label,
        candidate_label_source=candidate_label_source,
        positive_source=positive_source,
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
        generator_checkpoint=response.record.generator_checkpoint,
        sample_id=response.record.sample_id,
    )


def build_response_audit_summary(
    responses_by_question: Mapping[str, Sequence[MatchedResponse]],
) -> dict[str, Any]:
    parser_status_counts = Counter()
    parser_warning_counts = Counter()
    exclusion_reason_counts = Counter()
    checkpoint_counts = Counter()
    total_responses = 0
    eligible_responses = 0
    responses_with_placeholder_drop = 0
    dropped_placeholder_total = 0

    for responses in responses_by_question.values():
        for response in responses:
            total_responses += 1
            parser_status_counts[response.parsed.status] += 1
            parser_warning_counts.update(response.parsed.warnings)
            checkpoint_counts[clean_text(response.record.generator_checkpoint) or "null"] += 1
            if response.parsed.dropped_placeholder_count:
                responses_with_placeholder_drop += 1
                dropped_placeholder_total += response.parsed.dropped_placeholder_count
            if response.pair_eligible:
                eligible_responses += 1
            elif response.pair_exclusion_reason:
                exclusion_reason_counts[response.pair_exclusion_reason] += 1

    return {
        "total_responses": total_responses,
        "pair_eligible_responses": eligible_responses,
        "pair_ineligible_responses": total_responses - eligible_responses,
        "parser_status_distribution": dict(sorted(parser_status_counts.items())),
        "parser_warning_distribution": dict(sorted(parser_warning_counts.items())),
        "pair_exclusion_reason_distribution": dict(sorted(exclusion_reason_counts.items())),
        "generator_checkpoint_distribution": dict(sorted(checkpoint_counts.items())),
        "responses_with_placeholder_drop": responses_with_placeholder_drop,
        "dropped_placeholder_total": dropped_placeholder_total,
    }


def build_independent_post_hoc_validation_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    questions_by_id: Mapping[str, QuestionExample],
) -> dict[str, Any]:
    negative_counts = Counter()
    omission_counts = Counter()
    negative_examples: dict[str, list[str]] = defaultdict(list)
    omission_examples: dict[str, list[str]] = defaultdict(list)

    for row in pair_rows:
        question_id = clean_text(row.get("question_id"))
        question = questions_by_id.get(question_id)
        if question is None:
            continue

        edited_candidate = clean_text(row.get("edited_candidate"))
        if not edited_candidate:
            continue

        evidence_text = normalize_evidence_text(question.evidence)
        question_stats = {
            "evidence_text": evidence_text,
            "evidence_text_compact": evidence_text.replace(" ", ""),
        }
        pair_id = clean_text(row.get("pair_id"))
        pair_type = clean_text(row.get("pair_type"))

        if pair_type == PAIR_TYPE_NEGATIVE_ADDITION:
            negative_counts["total_pairs"] += 1
            gold_overlap = False
            for gold_group in question.gold_groups:
                for alias in gold_group.aliases:
                    if surface_overlap_reason(question, edited_candidate, alias) is not None:
                        gold_overlap = True
                        break
                if gold_overlap:
                    break
            if gold_overlap:
                negative_counts["lexical_gold_overlap_pairs"] += 1
                if len(negative_examples["lexical_gold_overlap_pairs"]) < 10:
                    negative_examples["lexical_gold_overlap_pairs"].append(pair_id)

            evidence_supported = candidate_is_evidence_supported(question, edited_candidate, question_stats)
            if evidence_supported:
                negative_counts["evidence_supported_pairs"] += 1
                if len(negative_examples["evidence_supported_pairs"]) < 10:
                    negative_examples["evidence_supported_pairs"].append(pair_id)

            if not gold_overlap and not evidence_supported:
                negative_counts["pairs_passing_all_independent_screens"] += 1
            continue

        if pair_type == PAIR_TYPE_VALID_OMISSION:
            omission_counts["total_pairs"] += 1
            rejected_overlap = False
            for item in row.get("rejected_items", []):
                if surface_overlap_reason(question, edited_candidate, clean_text(item)) is not None:
                    rejected_overlap = True
                    break
            if rejected_overlap:
                omission_counts["lexical_overlap_with_rejected_pairs"] += 1
                if len(omission_examples["lexical_overlap_with_rejected_pairs"]) < 10:
                    omission_examples["lexical_overlap_with_rejected_pairs"].append(pair_id)

            evidence_supported = candidate_is_evidence_supported(question, edited_candidate, question_stats)
            if not evidence_supported:
                omission_counts["evidence_unsupported_pairs"] += 1
                if len(omission_examples["evidence_unsupported_pairs"]) < 10:
                    omission_examples["evidence_unsupported_pairs"].append(pair_id)

            if not rejected_overlap and evidence_supported:
                omission_counts["pairs_passing_all_independent_screens"] += 1

    def with_rates(counts: Counter, examples: Mapping[str, Sequence[str]]) -> dict[str, Any]:
        total = int(counts.get("total_pairs", 0))
        result: dict[str, Any] = dict(sorted(counts.items()))
        if total:
            for key, value in list(result.items()):
                if key == "total_pairs":
                    continue
                result[f"{key}_rate"] = value / total
        if examples:
            result["sample_pair_ids"] = {key: list(values) for key, values in sorted(examples.items())}
        return result

    return {
        "description": (
            "Conservative heuristic screens that do not call score_item_surfaces() or the pair-scoring F1 matcher. "
            "These counts are still heuristic and do not prove semantic correctness."
        ),
        "negative_addition": with_rates(negative_counts, negative_examples),
        "valid_omission": with_rates(omission_counts, omission_examples),
    }


def build_omission_recoverability_summary(
    grouped_responses: Mapping[tuple[str, str], Sequence[MatchedResponse]],
    questions_by_id: Mapping[str, QuestionExample],
    allow_semantic_overlap_omission_pairs: bool,
    allow_evidence_unsupported_omission_pairs: bool,
) -> dict[str, Any]:
    total_missing_group_instances = 0
    recoverable_missing_group_instances = 0
    unrecoverable_missing_group_instances = 0
    semantic_overlap_filtered_missing_group_instances = 0
    eligible_response_count = 0
    responses_with_any_missing_group = 0
    responses_with_any_recoverable_omission = 0
    responses_with_only_unrecoverable_omissions = 0
    recoverable_question_ids: set[str] = set()

    for (question_id, _checkpoint), responses in grouped_responses.items():
        question = questions_by_id[question_id]
        question_stats = build_question_statistics(responses, question)
        question_has_recoverable = False

        for response in responses:
            if not response.pair_eligible:
                continue

            eligible_response_count += 1
            if not response.missing_gold_group_ids:
                continue

            responses_with_any_missing_group += 1
            recoverable_for_response = 0
            for gold_group_id in response.missing_gold_group_ids:
                total_missing_group_instances += 1
                positive_surface, _positive_source, filter_reason = choose_positive_surface(
                    question=question,
                    response=response,
                    gold_group_id=gold_group_id,
                    question_stats=question_stats,
                    allow_gold_alias_fallback=False,
                    allow_semantic_overlap_omission_pairs=allow_semantic_overlap_omission_pairs,
                    allow_evidence_unsupported_omission_pairs=allow_evidence_unsupported_omission_pairs,
                )
                if positive_surface:
                    recoverable_missing_group_instances += 1
                    recoverable_for_response += 1
                    question_has_recoverable = True
                else:
                    unrecoverable_missing_group_instances += 1
                    if filter_reason == "semantic_overlap_with_rejected_set":
                        semantic_overlap_filtered_missing_group_instances += 1

            if recoverable_for_response:
                responses_with_any_recoverable_omission += 1
            else:
                responses_with_only_unrecoverable_omissions += 1

        if question_has_recoverable:
            recoverable_question_ids.add(question_id)

    recoverable_rate = (
        recoverable_missing_group_instances / total_missing_group_instances
        if total_missing_group_instances
        else 0.0
    )
    return {
        "eligible_response_count": eligible_response_count,
        "responses_with_any_missing_group": responses_with_any_missing_group,
        "responses_with_any_recoverable_omission": responses_with_any_recoverable_omission,
        "responses_with_only_unrecoverable_omissions": responses_with_only_unrecoverable_omissions,
        "question_count_with_any_recoverable_omission": len(recoverable_question_ids),
        "total_missing_gold_group_instances": total_missing_group_instances,
        "recoverable_missing_gold_group_instances": recoverable_missing_group_instances,
        "unrecoverable_missing_gold_group_instances": unrecoverable_missing_group_instances,
        "semantic_overlap_filtered_missing_group_instances": semantic_overlap_filtered_missing_group_instances,
        "recoverable_missing_group_rate": recoverable_rate,
    }


def build_negative_addition_pairs(
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    question_stats: Mapping[str, Any],
    max_pairs_per_question: int,
    pair_id_prefix: str,
    allow_evidence_supported_negative_pairs: bool,
    allow_gold_overlap_negative_pairs: bool,
    min_negative_addition_delta_f1: float,
) -> tuple[list[PreferencePair], dict[str, int]]:
    ranked_pairs: list[tuple[tuple[float, int, int, int, str], PreferencePair]] = []
    audit_counts = Counter()
    counter = 0
    for response in responses:
        if not response.pair_eligible:
            continue
        rejected_items = [candidate.surface for candidate in response.candidates]
        negative_candidates = [candidate for candidate in response.candidates if candidate.label == LABEL_METRIC_NEGATIVE]
        for candidate in sorted(
            negative_candidates,
            key=lambda item: negative_candidate_priority(item, question, question_stats),
            reverse=True,
        ):
            audit_counts["metric_negative_candidates_total"] += 1
            gold_overlap_reason = surface_overlaps_gold_alias(question, candidate.surface)
            if gold_overlap_reason is not None:
                audit_counts["gold_overlap_metric_negative_candidates"] += 1
            if gold_overlap_reason is not None and not allow_gold_overlap_negative_pairs:
                audit_counts["filtered_gold_overlap_metric_negative_candidates"] += 1
                continue

            evidence_supported = candidate_is_evidence_supported(question, candidate.surface, question_stats)
            if evidence_supported:
                audit_counts["evidence_supported_metric_negative_candidates"] += 1
            if evidence_supported and not allow_evidence_supported_negative_pairs:
                audit_counts["filtered_evidence_supported_metric_negative_candidates"] += 1
                continue

            audit_counts["metric_negative_candidates_after_filters"] += 1
            chosen_items = [item.surface for item in response.candidates if item.first_index != candidate.first_index]
            pair = build_pair(
                pair_id=f"{pair_id_prefix}-addneg-{counter:04d}",
                pair_type=PAIR_TYPE_NEGATIVE_ADDITION,
                question=question,
                response=response,
                chosen_items=chosen_items,
                rejected_items=rejected_items,
                edited_candidate=candidate.surface,
                edited_candidate_normalized=candidate.normalized,
                edited_gold_group_id=None,
                candidate_label=LABEL_METRIC_NEGATIVE,
                candidate_label_source="one_to_one_gold_group_matching",
                positive_source=None,
            )
            counter += 1
            if pair is None:
                continue
            if pair.delta_f1 < min_negative_addition_delta_f1:
                audit_counts["filtered_low_delta_negative_pairs"] += 1
                continue
            audit_counts["candidate_negative_pairs_after_filters"] += 1
            ranked_pairs.append((negative_candidate_priority(candidate, question, question_stats), pair))

    selected_pairs = dedupe_and_trim_pairs(
        question=question,
        ranked_pairs=ranked_pairs,
        max_pairs=max_pairs_per_question,
    )
    audit_counts["emitted_negative_addition_pairs"] = len(selected_pairs)
    return selected_pairs, dict(sorted(audit_counts.items()))


def build_valid_omission_pairs(
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    question_stats: Mapping[str, Any],
    max_pairs_per_question: int,
    allow_gold_alias_fallback: bool,
    pair_id_prefix: str,
    allow_semantic_overlap_omission_pairs: bool,
    allow_evidence_unsupported_omission_pairs: bool,
) -> tuple[list[PreferencePair], dict[str, int]]:
    ranked_pairs: list[tuple[tuple[int, int, int, str], PreferencePair]] = []
    audit_counts = Counter()
    counter = 0
    surface_counter = question_stats["valid_surface_by_gold_group"]

    for response in responses:
        if not response.pair_eligible:
            continue
        rejected_items = [candidate.surface for candidate in response.candidates]
        for gold_group_id in response.missing_gold_group_ids:
            audit_counts["missing_gold_group_instances_total"] += 1
            positive_surface, positive_source, filter_reason = choose_positive_surface(
                question=question,
                response=response,
                gold_group_id=gold_group_id,
                question_stats=question_stats,
                allow_gold_alias_fallback=allow_gold_alias_fallback,
                allow_semantic_overlap_omission_pairs=allow_semantic_overlap_omission_pairs,
                allow_evidence_unsupported_omission_pairs=allow_evidence_unsupported_omission_pairs,
            )
            if not positive_surface:
                if filter_reason == "semantic_overlap_with_rejected_set":
                    audit_counts["filtered_semantic_overlap_missing_gold_group_instances"] += 1
                elif filter_reason == "positive_surface_not_supported_by_evidence":
                    audit_counts["filtered_evidence_unsupported_missing_gold_group_instances"] += 1
                continue
            audit_counts["missing_gold_group_instances_with_positive_surface"] += 1
            chosen_items = rejected_items + [positive_surface]
            pair = build_pair(
                pair_id=f"{pair_id_prefix}-omit-{counter:04d}",
                pair_type=PAIR_TYPE_VALID_OMISSION,
                question=question,
                response=response,
                chosen_items=chosen_items,
                rejected_items=rejected_items,
                edited_candidate=positive_surface,
                edited_candidate_normalized=normalize_answer_surface(positive_surface),
                edited_gold_group_id=gold_group_id,
                candidate_label=LABEL_GOLD_MISSING,
                candidate_label_source="missing_gold_group_after_one_to_one_matching",
                positive_source=positive_source,
            )
            counter += 1
            if pair is None:
                continue
            audit_counts["candidate_valid_omission_pairs_after_filters"] += 1
            surface_count = int(surface_counter.get(gold_group_id, Counter()).get(positive_surface, 0))
            rank = (
                1 if positive_source == "sampled_valid_candidate" else 0,
                surface_count,
                int(round(pair.delta_f1 * 10000)),
                normalize_answer_surface(positive_surface),
            )
            ranked_pairs.append((rank, pair))

    selected_pairs = dedupe_and_trim_pairs(
        question=question,
        ranked_pairs=ranked_pairs,
        max_pairs=max_pairs_per_question,
    )
    audit_counts["emitted_valid_omission_pairs"] = len(selected_pairs)
    return selected_pairs, dict(sorted(audit_counts.items()))


def dedupe_and_trim_pairs(
    question: QuestionExample,
    ranked_pairs: Iterable[tuple[tuple[Any, ...], PreferencePair]],
    max_pairs: int,
) -> list[PreferencePair]:
    seen = set()
    contradictions = set()
    selected: list[tuple[tuple[Any, ...], PreferencePair]] = []

    for rank, pair in sorted(ranked_pairs, key=lambda item: item[0], reverse=True):
        key = (
            pair.question_id,
            semantic_set_key_for_question(question, pair.chosen_items),
            semantic_set_key_for_question(question, pair.rejected_items),
            pair.pair_type,
        )
        reverse_key = (key[0], key[2], key[1], key[3])
        if reverse_key in seen:
            contradictions.add(reverse_key)
            continue
        if key in seen:
            continue
        seen.add(key)
        selected.append((rank, pair))
        if len(selected) >= max_pairs:
            break

    return [pair for _rank, pair in selected]


def group_responses_by_question(
    question_input: Sequence[str],
    candidate_input: Sequence[str],
    dataset_name: str,
    allow_fallback_split: bool,
    allow_fallback_pair_construction: bool,
    max_resources: int,
    max_resource_chars: int,
    gold_support_policy: str,
    question_limit: int | None = None,
) -> tuple[dict[str, QuestionExample], dict[str, list[MatchedResponse]], dict[tuple[str, str], list[MatchedResponse]]]:
    questions_by_id = load_question_examples(
        paths=question_input,
        dataset_name=dataset_name,
        max_resources=max_resources,
        max_resource_chars=max_resource_chars,
        gold_support_policy=gold_support_policy,
    )
    if question_limit is not None:
        questions_by_id = dict(list(sorted(questions_by_id.items()))[:question_limit])
    records = load_candidate_bank_records(
        paths=candidate_input,
        questions_by_id=questions_by_id,
        dataset_name=dataset_name,
    )

    responses_by_question_and_checkpoint: dict[tuple[str, str], list[MatchedResponse]] = defaultdict(list)
    responses_by_question: dict[str, list[MatchedResponse]] = defaultdict(list)
    for record in records:
        question = questions_by_id[record.question_id]
        matched = build_matched_response(
            record=record,
            question=question,
            allow_fallback_split=allow_fallback_split,
            allow_fallback_pair_construction=allow_fallback_pair_construction,
        )
        responses_by_question[record.question_id].append(matched)
        checkpoint_key = clean_text(record.generator_checkpoint) or "unknown"
        responses_by_question_and_checkpoint[(record.question_id, checkpoint_key)].append(matched)
    return questions_by_id, responses_by_question, responses_by_question_and_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct minimal counterfactual set-edit preference pairs for BioASQ list questions."
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
        "--max-negative-addition-pairs-per-question",
        type=int,
        default=8,
        help="Cap on negative-addition pairs emitted per question.",
    )
    parser.add_argument(
        "--max-valid-omission-pairs-per-question",
        type=int,
        default=8,
        help="Cap on valid-omission pairs emitted per question.",
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing for untagged outputs.",
    )
    parser.add_argument(
        "--allow-fallback-pair-construction",
        action="store_true",
        help="Allow fallback-split responses to seed preference pairs. Malformed and empty responses remain excluded.",
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--allow-gold-alias-fallback",
        action="store_true",
        help="Allow omission repairs to fall back to a canonical gold alias when no sampled valid candidate exists, regardless of evidence support.",
    )
    fallback_group.add_argument(
        "--disallow-gold-alias-fallback",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-multiple-generator-checkpoints",
        action="store_true",
        help="Allow candidate inputs from multiple generator checkpoints. Responses are still grouped by checkpoint during pair construction.",
    )
    parser.add_argument(
        "--allow-evidence-supported-negative-pairs",
        action="store_true",
        help="Allow negative-addition pairs whose removed candidate is explicitly supported by the supplied evidence.",
    )
    parser.add_argument(
        "--allow-gold-overlap-negative-pairs",
        action="store_true",
        help="Allow negative-addition pairs whose removed candidate still overlaps a gold alias under the conservative semantic or lexical screen.",
    )
    parser.add_argument(
        "--allow-semantic-overlap-omission-pairs",
        action="store_true",
        help="Allow omission repairs even when the added surface is already semantically represented in the rejected set.",
    )
    parser.add_argument(
        "--allow-evidence-unsupported-omission-pairs",
        action="store_true",
        help="Allow omission repairs even when the added surface is not explicitly supported by the supplied evidence for that question.",
    )
    parser.add_argument(
        "--min-negative-addition-delta-f1",
        type=float,
        default=0.0,
        help="Drop negative-addition pairs whose F1 improvement is below this threshold.",
    )
    parser.add_argument(
        "--manual-audit-sample-per-type",
        type=int,
        default=50,
        help="How many pairs of each type to include in the manual audit markdown.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used when sampling the markdown audit file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_by_id, responses_by_question, responses_by_question_and_checkpoint = group_responses_by_question(
        question_input=args.question_input,
        candidate_input=args.candidate_input,
        dataset_name=args.dataset_name,
        allow_fallback_split=args.allow_fallback_split,
        allow_fallback_pair_construction=args.allow_fallback_pair_construction,
        max_resources=int(args.max_resources),
        max_resource_chars=int(args.max_resource_chars),
        gold_support_policy=str(args.gold_support_policy),
    )

    generator_checkpoints = {
        clean_text(response.record.generator_checkpoint) or "unknown"
        for responses in responses_by_question.values()
        for response in responses
    }
    if len(generator_checkpoints) > 1 and not args.allow_multiple_generator_checkpoints:
        raise ValueError(
            "Multiple generator checkpoints were found in the candidate inputs. "
            "Construct pairs from a single frozen generator checkpoint or pass "
            "--allow-multiple-generator-checkpoints to partition by checkpoint explicitly. "
            f"Found: {sorted(generator_checkpoints)}"
        )

    all_pairs: list[PreferencePair] = []
    negative_pair_filter_audit = Counter()
    omission_pair_filter_audit = Counter()
    allow_gold_alias_fallback = bool(args.allow_gold_alias_fallback and not args.disallow_gold_alias_fallback)
    for (question_id, _checkpoint), responses in responses_by_question_and_checkpoint.items():
        question = questions_by_id[question_id]
        question_stats = build_question_statistics(responses, question)
        checkpoint_slug = slugify(clean_text(responses[0].record.generator_checkpoint) or "checkpoint")
        pair_id_prefix = f"{slugify(question.dataset)}-{slugify(question.question_id)}-{checkpoint_slug}"
        negative_pairs, negative_audit = build_negative_addition_pairs(
            question=question,
            responses=responses,
            question_stats=question_stats,
            max_pairs_per_question=args.max_negative_addition_pairs_per_question,
            pair_id_prefix=pair_id_prefix,
            allow_evidence_supported_negative_pairs=bool(args.allow_evidence_supported_negative_pairs),
            allow_gold_overlap_negative_pairs=bool(args.allow_gold_overlap_negative_pairs),
            min_negative_addition_delta_f1=float(args.min_negative_addition_delta_f1),
        )
        all_pairs.extend(negative_pairs)
        negative_pair_filter_audit.update(negative_audit)

        omission_pairs, omission_audit = build_valid_omission_pairs(
            question=question,
            responses=responses,
            question_stats=question_stats,
            max_pairs_per_question=args.max_valid_omission_pairs_per_question,
            allow_gold_alias_fallback=allow_gold_alias_fallback,
            pair_id_prefix=pair_id_prefix,
            allow_semantic_overlap_omission_pairs=bool(args.allow_semantic_overlap_omission_pairs),
            allow_evidence_unsupported_omission_pairs=bool(args.allow_evidence_unsupported_omission_pairs),
        )
        all_pairs.extend(omission_pairs)
        omission_pair_filter_audit.update(omission_audit)

    pair_rows = [to_jsonable(pair) for pair in sorted(all_pairs, key=lambda item: (item.question_id, item.pair_type, item.pair_id))]
    write_jsonl(Path(args.output_jsonl), pair_rows)

    question_gold_counts = {
        question_id: len(question.gold_groups)
        for question_id, question in questions_by_id.items()
    }
    summary = build_summary(pair_rows, question_gold_counts=question_gold_counts)
    summary["response_audit"] = build_response_audit_summary(responses_by_question)
    summary["omission_recoverability"] = build_omission_recoverability_summary(
        grouped_responses=responses_by_question_and_checkpoint,
        questions_by_id=questions_by_id,
        allow_semantic_overlap_omission_pairs=bool(args.allow_semantic_overlap_omission_pairs),
        allow_evidence_unsupported_omission_pairs=bool(args.allow_evidence_unsupported_omission_pairs),
    )
    summary["pair_filter_audit"] = {
        "negative_addition": dict(sorted(negative_pair_filter_audit.items())),
        "valid_omission": dict(sorted(omission_pair_filter_audit.items())),
    }
    summary["independent_post_hoc_validation"] = build_independent_post_hoc_validation_summary(
        pair_rows=pair_rows,
        questions_by_id=questions_by_id,
    )
    negative_addition_policy = (
        "metric_gold_unmatched_evidence_supported_allowed"
        if args.allow_evidence_supported_negative_pairs
        else "strict_gold_unmatched_not_evidence_supported"
    )
    summary["pair_construction_policy"] = {
        "allow_fallback_split": bool(args.allow_fallback_split),
        "allow_fallback_pair_construction": bool(args.allow_fallback_pair_construction),
        "max_resources": int(args.max_resources),
        "max_resource_chars": int(args.max_resource_chars),
        "gold_support_policy": str(args.gold_support_policy),
        "allow_gold_alias_fallback": allow_gold_alias_fallback,
        "gold_alias_fallback_requires_evidence": False,
        "allow_multiple_generator_checkpoints": bool(args.allow_multiple_generator_checkpoints),
        "allow_evidence_supported_negative_pairs": bool(args.allow_evidence_supported_negative_pairs),
        "negative_addition_policy": negative_addition_policy,
        "allow_gold_overlap_negative_pairs": bool(args.allow_gold_overlap_negative_pairs),
        "allow_semantic_overlap_omission_pairs": bool(args.allow_semantic_overlap_omission_pairs),
        "allow_evidence_unsupported_omission_pairs": bool(args.allow_evidence_unsupported_omission_pairs),
        "min_negative_addition_delta_f1": float(args.min_negative_addition_delta_f1),
    }
    write_json(Path(args.summary_json), summary)

    markdown = build_manual_audit_markdown(
        pair_rows,
        question_input=args.question_input,
        dataset_name=args.dataset_name,
        sample_per_type=args.manual_audit_sample_per_type,
        seed=args.seed,
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
