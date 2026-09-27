from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, summarize_numeric, write_json, write_jsonl
from cse_dpo.construct_set_edit_pairs import (
    build_question_statistics,
    candidate_is_evidence_supported,
    choose_positive_surface,
    negative_candidate_priority,
    semantic_set_edit_distance,
    surface_overlaps_gold_alias,
)
from cse_dpo.match_gold_groups import (
    build_matched_response,
    load_candidate_bank_records,
    load_question_examples,
)
from cse_dpo.normalize_set_answers import normalize_answer_surface, serialize_list_items
from cse_dpo.schemas import (
    CandidateBankRecord,
    LABEL_METRIC_NEGATIVE,
    MatchedCandidate,
    MatchedResponse,
    QuestionExample,
)
from src.model_registry import slugify
from src.utility.data import clean_text


TARGET_OPERATIONS = (
    "fp_removal",
    "tp_restoration",
    "fp_to_tp_substitution",
)

EXPECTED_EDIT_DISTANCE = {
    "fp_removal": 1,
    "tp_restoration": 1,
    "fp_to_tp_substitution": 2,
}


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct validation/test constructed minimal-edit probe banks for the "
            "Cardinality Shortcut Study."
        )
    )
    parser.add_argument(
        "--question-input",
        nargs="+",
        required=True,
        help="Raw BioASQ JSON or prepared JSON files containing the gold questions.",
    )
    parser.add_argument(
        "--candidate-bank-jsonl",
        required=True,
        help="Candidate-bank JSONL used to recover the held-out base responses.",
    )
    parser.add_argument(
        "--standardized-input-jsonl",
        required=True,
        help="Standardized candidate JSONL for the requested split, e.g. standardized_candidates_validation.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the constructed minimal-edit probe bank and summary will be written.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Expected split label in the standardized candidate file.",
    )
    parser.add_argument(
        "--dataset-name",
        default="bioasq",
        help="Dataset label used when loading question examples.",
    )
    parser.add_argument(
        "--max-resources",
        type=int,
        default=0,
        help="Maximum number of resources to load from the raw questions. Use 0 for all resources.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=0,
        help="Maximum characters per serialized resource when loading raw question evidence. Use 0 for no truncation.",
    )
    parser.add_argument(
        "--gold-support-policy",
        choices=["all", "snippet"],
        default="all",
        help="Gold support policy passed to the project question loader.",
    )
    parser.add_argument(
        "--min-delta-f1",
        type=float,
        default=0.05,
        help="Minimum preferred-minus-dispreferred F1 margin required for a retained probe.",
    )
    parser.add_argument(
        "--max-probes-per-question-per-operation",
        type=int,
        default=4,
        help="Maximum retained probes per question/checkpoint group for each target operation.",
    )
    parser.add_argument(
        "--allow-fallback-split",
        action="store_true",
        help="Allow newline/semicolon fallback parsing when rebuilding matched responses from the base candidate outputs.",
    )
    parser.add_argument(
        "--allow-fallback-pair-construction",
        action="store_true",
        help="Allow fallback-split responses to seed constructed probes.",
    )
    parser.add_argument(
        "--allow-gold-alias-fallback",
        action="store_true",
        help="Allow TP-restoration and substitution probes to fall back to a canonical gold alias when no sampled valid positive surface exists.",
    )
    parser.add_argument(
        "--allow-multiple-generator-checkpoints",
        action="store_true",
        help="Allow base responses from multiple generator checkpoints. Construction still stays within each checkpoint group.",
    )
    parser.add_argument(
        "--allow-evidence-supported-negative-probes",
        action="store_true",
        help="Allow FP-removal and substitution probes that remove a negative entity explicitly supported by the evidence.",
    )
    parser.add_argument(
        "--allow-gold-overlap-negative-probes",
        action="store_true",
        help="Allow FP-removal and substitution probes that remove a negative entity overlapping a gold alias under the conservative overlap screens.",
    )
    parser.add_argument(
        "--allow-semantic-overlap-omission-probes",
        action="store_true",
        help="Allow TP-restoration and substitution probes even when the added surface is already semantically represented in the base response.",
    )
    parser.add_argument(
        "--allow-evidence-unsupported-omission-probes",
        action="store_true",
        help="Allow TP-restoration and substitution probes even when the added positive surface is not explicitly supported by the supplied evidence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Recorded seed value for reproducibility metadata.",
    )
    return parser.parse_args()


def load_allowed_split_records(
    path: Path,
    *,
    expected_split: str,
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    response_ids: set[str] = set()
    question_ids: set[str] = set()
    metadata_by_response_id: dict[str, dict[str, Any]] = {}
    for raw_row in load_json_records(path):
        row = dict(raw_row)
        split_name = clean_text(row.get("split")).lower()
        if split_name and split_name != expected_split:
            continue
        response_id = clean_text(row.get("response_id"))
        question_id = clean_text(row.get("question_id"))
        if not response_id or not question_id:
            continue
        response_ids.add(response_id)
        question_ids.add(question_id)
        metadata_by_response_id[response_id] = {
            "question_source_path": clean_text(row.get("question_source_path")),
            "generated_token_count": row.get("generated_token_count"),
            "pair_eligible": bool(row.get("pair_eligible")),
            "pair_exclusion_reason": clean_text(row.get("pair_exclusion_reason")),
            "parser_status": clean_text(row.get("parser_status")),
            "study_name": clean_text(row.get("study_name")),
        }
    return response_ids, question_ids, metadata_by_response_id


def group_key_for_response(response: MatchedResponse) -> str:
    checkpoint = clean_text(response.record.generator_checkpoint) or "unknown"
    return f"{response.record.question_id}::{checkpoint}"


def response_items(response: MatchedResponse) -> list[str]:
    ordered_candidates = sorted(response.candidates, key=lambda candidate: candidate.first_index)
    return [candidate.surface for candidate in ordered_candidates]


def response_counts(response: MatchedResponse) -> tuple[int, int, int]:
    true_positives = len(response.matched_gold_group_ids)
    false_positives = int(response.metrics.prediction_count) - true_positives
    false_negatives = len(response.missing_gold_group_ids)
    return true_positives, false_positives, false_negatives


def whitespace_token_count(text: str) -> int:
    return len([token for token in clean_text(text).split() if token])


def summarize_support(details: Sequence[Mapping[str, Any]]) -> str:
    if not details:
        return "not_applicable"
    supported_flags = [bool(item.get("evidence_supported")) for item in details]
    if all(supported_flags):
        return "supported"
    if not any(supported_flags):
        return "unsupported"
    return "mixed"


def changed_detail(
    *,
    question: QuestionExample,
    question_stats: Mapping[str, Any],
    surface: str,
    role: str,
    gold_overlap_reason: str | None = None,
    positive_source: str | None = None,
) -> dict[str, Any]:
    normalized_surface = normalize_answer_surface(surface)
    evidence_supported = candidate_is_evidence_supported(question, surface, question_stats)
    return {
        "role": role,
        "surface": clean_text(surface),
        "normalized_surface": normalized_surface,
        "evidence_supported": evidence_supported,
        "gold_overlap_reason": clean_text(gold_overlap_reason),
        "positive_source": clean_text(positive_source),
    }


def build_synthetic_response(
    *,
    question: QuestionExample,
    base_response: MatchedResponse,
    edited_items: Sequence[str],
    edited_response_ref: str,
) -> MatchedResponse:
    record = CandidateBankRecord(
        dataset=base_response.record.dataset,
        question_id=question.question_id,
        sample_id=base_response.record.sample_id,
        prompt=base_response.record.prompt,
        question_text=question.question_text,
        evidence=tuple(question.evidence),
        raw_output=serialize_list_items(edited_items),
        generated_token_count=None,
        generator_checkpoint=base_response.record.generator_checkpoint,
        response_id=edited_response_ref,
        source_path=base_response.record.source_path,
        prompt_instruction=base_response.record.prompt_instruction,
    )
    return build_matched_response(
        record=record,
        question=question,
        allow_fallback_split=False,
        allow_fallback_pair_construction=False,
    )


def operation_matches(
    *,
    semantic_operation: str,
    preferred: MatchedResponse,
    dispreferred: MatchedResponse,
    question: QuestionExample,
) -> bool:
    preferred_tp, preferred_fp, preferred_fn = response_counts(preferred)
    dispreferred_tp, dispreferred_fp, dispreferred_fn = response_counts(dispreferred)
    delta_tp = preferred_tp - dispreferred_tp
    delta_fp = preferred_fp - dispreferred_fp
    delta_fn = preferred_fn - dispreferred_fn
    preferred_items = response_items(preferred)
    dispreferred_items = response_items(dispreferred)
    edit_distance = semantic_set_edit_distance(question, preferred_items, dispreferred_items)
    expected_edit_distance = EXPECTED_EDIT_DISTANCE[semantic_operation]
    if edit_distance != expected_edit_distance:
        return False
    if preferred.metrics.f1 <= dispreferred.metrics.f1:
        return False

    if semantic_operation == "fp_removal":
        return (
            len(preferred_items) < len(dispreferred_items)
            and delta_tp == 0
            and delta_fp < 0
            and delta_fn == 0
        )
    if semantic_operation == "tp_restoration":
        return (
            len(preferred_items) > len(dispreferred_items)
            and delta_tp > 0
            and delta_fp == 0
            and delta_fn < 0
        )
    if semantic_operation == "fp_to_tp_substitution":
        return (
            len(preferred_items) == len(dispreferred_items)
            and delta_tp > 0
            and delta_fp < 0
            and delta_fn < 0
        )
    return False


def selection_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        abs(int(row.get("semantic_set_edit_distance") or 0) - int(EXPECTED_EDIT_DISTANCE[clean_text(row.get("semantic_operation"))])),
        int(row.get("absolute_token_length_gap") or 0),
        round(float(row.get("f1_margin") or 0.0), 6),
        -round(float(row.get("dispreferred_f1") or 0.0), 6),
        clean_text(row.get("base_response_id")),
        clean_text(row.get("changed_entity")),
    )


def dedupe_and_trim_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=selection_rank):
        key = (
            clean_text(row.get("question_group_id")),
            clean_text(row.get("semantic_operation")),
            tuple(clean_text(item) for item in row.get("preferred_items", []) or []),
            tuple(clean_text(item) for item in row.get("dispreferred_items", []) or []),
        )
        reverse_key = (key[0], key[1], key[3], key[2])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        selected.append(dict(row))
        if len(selected) >= max_rows:
            break
    return selected


def build_probe_row(
    *,
    question: QuestionExample,
    base_response: MatchedResponse,
    preferred: MatchedResponse,
    dispreferred: MatchedResponse,
    semantic_operation: str,
    changed_entity_details: Sequence[Mapping[str, Any]],
    positive_source: str | None,
    split: str,
) -> dict[str, Any]:
    preferred_items = response_items(preferred)
    dispreferred_items = response_items(dispreferred)
    preferred_answer = serialize_list_items(preferred_items)
    dispreferred_answer = serialize_list_items(dispreferred_items)
    preferred_tp, preferred_fp, preferred_fn = response_counts(preferred)
    dispreferred_tp, dispreferred_fp, dispreferred_fn = response_counts(dispreferred)
    preferred_entity_count = len(preferred_items)
    dispreferred_entity_count = len(dispreferred_items)
    smaller_entity_count = min(preferred_entity_count, dispreferred_entity_count)
    entity_ratio = (
        float("inf")
        if smaller_entity_count == 0 and max(preferred_entity_count, dispreferred_entity_count) > 0
        else max(preferred_entity_count, dispreferred_entity_count) / max(1, smaller_entity_count)
    )
    direction_label = (
        "shorter_preferred"
        if preferred_entity_count < dispreferred_entity_count
        else "longer_preferred"
        if preferred_entity_count > dispreferred_entity_count
        else "equal_preferred"
    )
    changed_entity = None
    if semantic_operation == "fp_to_tp_substitution":
        removed = next((item["surface"] for item in changed_entity_details if clean_text(item.get("role")) == "removed_negative"), None)
        added = next((item["surface"] for item in changed_entity_details if clean_text(item.get("role")) == "added_positive"), None)
        if removed and added:
            changed_entity = f"remove:{removed} -> add:{added}"
    elif changed_entity_details:
        changed_entity = clean_text(changed_entity_details[0].get("surface"))

    return {
        "probe_id": f"{clean_text(base_response.record.response_id)}-{semantic_operation}-constructed-minimal-edit",
        "split": split,
        "probe_source": "constructed_minimal_edit",
        "semantic_operation": semantic_operation,
        "direction_label": direction_label,
        "question_id": question.question_id,
        "question_group_id": group_key_for_response(base_response),
        "question_text": question.question_text,
        "question_source_path": question.source_path,
        "prompt": base_response.record.prompt,
        "evidence": list(question.evidence),
        "base_response_id": base_response.record.response_id,
        "base_sample_id": base_response.record.sample_id,
        "base_generator_checkpoint": clean_text(base_response.record.generator_checkpoint),
        "preferred_origin": "edited_from_base",
        "dispreferred_origin": "base_response",
        "preferred_response_ref": clean_text(preferred.record.response_id),
        "dispreferred_response_ref": clean_text(dispreferred.record.response_id),
        "preferred_answer": preferred_answer,
        "dispreferred_answer": dispreferred_answer,
        "preferred_items": preferred_items,
        "dispreferred_items": dispreferred_items,
        "changed_entity": changed_entity,
        "changed_entity_details": [dict(item) for item in changed_entity_details],
        "changed_entity_count": len(changed_entity_details),
        "changed_entity_support_status": summarize_support(changed_entity_details),
        "removed_negative_support_status": summarize_support(
            [item for item in changed_entity_details if clean_text(item.get("role")) == "removed_negative"]
        ),
        "added_positive_support_status": summarize_support(
            [item for item in changed_entity_details if clean_text(item.get("role")) == "added_positive"]
        ),
        "positive_source": clean_text(positive_source),
        "preferred_f1": preferred.metrics.f1,
        "dispreferred_f1": dispreferred.metrics.f1,
        "f1_margin": preferred.metrics.f1 - dispreferred.metrics.f1,
        "preferred_precision": preferred.metrics.precision,
        "preferred_recall": preferred.metrics.recall,
        "dispreferred_precision": dispreferred.metrics.precision,
        "dispreferred_recall": dispreferred.metrics.recall,
        "preferred_tp": preferred_tp,
        "preferred_fp": preferred_fp,
        "preferred_fn": preferred_fn,
        "dispreferred_tp": dispreferred_tp,
        "dispreferred_fp": dispreferred_fp,
        "dispreferred_fn": dispreferred_fn,
        "preferred_entity_count": preferred_entity_count,
        "dispreferred_entity_count": dispreferred_entity_count,
        "entity_count_delta": preferred_entity_count - dispreferred_entity_count,
        "absolute_entity_gap": abs(preferred_entity_count - dispreferred_entity_count),
        "entity_ratio": entity_ratio,
        "gold_entity_count": len(question.gold_groups),
        "preferred_token_length_chars": len(preferred_answer),
        "dispreferred_token_length_chars": len(dispreferred_answer),
        "token_length_delta": len(preferred_answer) - len(dispreferred_answer),
        "absolute_token_length_gap": abs(len(preferred_answer) - len(dispreferred_answer)),
        "preferred_answer_token_count_estimate": whitespace_token_count(preferred_answer),
        "dispreferred_answer_token_count_estimate": whitespace_token_count(dispreferred_answer),
        "base_generated_token_count": base_response.record.generated_token_count,
        "preferred_generated_token_count": None,
        "dispreferred_generated_token_count": base_response.record.generated_token_count,
        "semantic_set_edit_distance": semantic_set_edit_distance(question, preferred_items, dispreferred_items),
    }


def build_group_probe_candidates(
    *,
    question: QuestionExample,
    responses: Sequence[MatchedResponse],
    min_delta_f1: float,
    max_probes_per_question_per_operation: int,
    allow_gold_alias_fallback: bool,
    allow_evidence_supported_negative_probes: bool,
    allow_gold_overlap_negative_probes: bool,
    allow_semantic_overlap_omission_probes: bool,
    allow_evidence_unsupported_omission_probes: bool,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_stats = build_question_statistics(responses, question)
    candidate_rows_by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = Counter({"responses_total": len(responses)})

    for response in responses:
        if response.pair_eligible:
            audit["pair_eligible_responses"] += 1
        else:
            audit["filtered_pair_ineligible_responses"] += 1
            continue

        base_items = response_items(response)
        ordered_candidates = sorted(response.candidates, key=lambda candidate: candidate.first_index)
        negative_candidates = [
            candidate
            for candidate in ordered_candidates
            if candidate.label == LABEL_METRIC_NEGATIVE
        ]

        positive_surfaces: list[tuple[int, str, str]] = []
        for gold_group_id in response.missing_gold_group_ids:
            positive_surface, positive_source, filter_reason = choose_positive_surface(
                question=question,
                response=response,
                gold_group_id=gold_group_id,
                question_stats=question_stats,
                allow_gold_alias_fallback=allow_gold_alias_fallback,
                allow_semantic_overlap_omission_pairs=allow_semantic_overlap_omission_probes,
                allow_evidence_unsupported_omission_pairs=allow_evidence_unsupported_omission_probes,
            )
            if not positive_surface:
                if filter_reason == "semantic_overlap_with_rejected_set":
                    audit["filtered_semantic_overlap_positive_surfaces"] += 1
                elif filter_reason == "positive_surface_not_supported_by_evidence":
                    audit["filtered_evidence_unsupported_positive_surfaces"] += 1
                continue
            positive_surfaces.append((gold_group_id, positive_surface, clean_text(positive_source)))
            audit["recoverable_missing_gold_groups"] += 1

            edited_items = list(base_items) + [positive_surface]
            preferred_ref = f"{response.record.response_id}::tp-restoration::{slugify(positive_surface, fallback='entity')}"
            preferred_response = build_synthetic_response(
                question=question,
                base_response=response,
                edited_items=edited_items,
                edited_response_ref=preferred_ref,
            )
            if not operation_matches(
                semantic_operation="tp_restoration",
                preferred=preferred_response,
                dispreferred=response,
                question=question,
            ):
                audit["filtered_invalid_tp_restoration_candidates"] += 1
                continue
            delta_f1 = preferred_response.metrics.f1 - response.metrics.f1
            if delta_f1 < min_delta_f1:
                audit["filtered_low_delta_tp_restoration_candidates"] += 1
                continue

            detail = changed_detail(
                question=question,
                question_stats=question_stats,
                surface=positive_surface,
                role="added_positive",
                positive_source=positive_source,
            )
            candidate_rows_by_operation["tp_restoration"].append(
                build_probe_row(
                    question=question,
                    base_response=response,
                    preferred=preferred_response,
                    dispreferred=response,
                    semantic_operation="tp_restoration",
                    changed_entity_details=[detail],
                    positive_source=positive_source,
                    split=split,
                )
            )
            audit["candidate_tp_restoration_probes"] += 1

        for negative_candidate in sorted(
            negative_candidates,
            key=lambda candidate: negative_candidate_priority(candidate, question, question_stats),
            reverse=True,
        ):
            gold_overlap_reason = surface_overlaps_gold_alias(question, negative_candidate.surface)
            evidence_supported = candidate_is_evidence_supported(question, negative_candidate.surface, question_stats)
            if gold_overlap_reason and not allow_gold_overlap_negative_probes:
                audit["filtered_gold_overlap_negative_candidates"] += 1
                continue
            if evidence_supported and not allow_evidence_supported_negative_probes:
                audit["filtered_evidence_supported_negative_candidates"] += 1
                continue

            removal_items = [
                candidate.surface
                for candidate in ordered_candidates
                if candidate.first_index != negative_candidate.first_index
            ]
            preferred_ref = f"{response.record.response_id}::fp-removal::{slugify(negative_candidate.surface, fallback='entity')}"
            preferred_response = build_synthetic_response(
                question=question,
                base_response=response,
                edited_items=removal_items,
                edited_response_ref=preferred_ref,
            )
            if operation_matches(
                semantic_operation="fp_removal",
                preferred=preferred_response,
                dispreferred=response,
                question=question,
            ):
                delta_f1 = preferred_response.metrics.f1 - response.metrics.f1
                if delta_f1 >= min_delta_f1:
                    detail = changed_detail(
                        question=question,
                        question_stats=question_stats,
                        surface=negative_candidate.surface,
                        role="removed_negative",
                        gold_overlap_reason=gold_overlap_reason,
                    )
                    candidate_rows_by_operation["fp_removal"].append(
                        build_probe_row(
                            question=question,
                            base_response=response,
                            preferred=preferred_response,
                            dispreferred=response,
                            semantic_operation="fp_removal",
                            changed_entity_details=[detail],
                            positive_source=None,
                            split=split,
                        )
                    )
                    audit["candidate_fp_removal_probes"] += 1
                else:
                    audit["filtered_low_delta_fp_removal_candidates"] += 1
            else:
                audit["filtered_invalid_fp_removal_candidates"] += 1

            for gold_group_id, positive_surface, positive_source in positive_surfaces:
                substitution_items: list[str] = []
                for candidate in ordered_candidates:
                    if candidate.first_index == negative_candidate.first_index:
                        substitution_items.append(positive_surface)
                    else:
                        substitution_items.append(candidate.surface)
                preferred_ref = (
                    f"{response.record.response_id}::substitution::"
                    f"{slugify(negative_candidate.surface, fallback='remove')}::"
                    f"{slugify(positive_surface, fallback='add')}"
                )
                preferred_response = build_synthetic_response(
                    question=question,
                    base_response=response,
                    edited_items=substitution_items,
                    edited_response_ref=preferred_ref,
                )
                if not operation_matches(
                    semantic_operation="fp_to_tp_substitution",
                    preferred=preferred_response,
                    dispreferred=response,
                    question=question,
                ):
                    audit["filtered_invalid_substitution_candidates"] += 1
                    continue
                delta_f1 = preferred_response.metrics.f1 - response.metrics.f1
                if delta_f1 < min_delta_f1:
                    audit["filtered_low_delta_substitution_candidates"] += 1
                    continue

                changed_details = [
                    changed_detail(
                        question=question,
                        question_stats=question_stats,
                        surface=negative_candidate.surface,
                        role="removed_negative",
                        gold_overlap_reason=gold_overlap_reason,
                    ),
                    changed_detail(
                        question=question,
                        question_stats=question_stats,
                        surface=positive_surface,
                        role="added_positive",
                        positive_source=positive_source,
                    ),
                ]
                candidate_rows_by_operation["fp_to_tp_substitution"].append(
                    build_probe_row(
                        question=question,
                        base_response=response,
                        preferred=preferred_response,
                        dispreferred=response,
                        semantic_operation="fp_to_tp_substitution",
                        changed_entity_details=changed_details,
                        positive_source=positive_source,
                        split=split,
                    )
                )
                audit["candidate_fp_to_tp_substitution_probes"] += 1

    selected_rows: list[dict[str, Any]] = []
    for operation in TARGET_OPERATIONS:
        trimmed = dedupe_and_trim_rows(
            candidate_rows_by_operation.get(operation, []),
            max_rows=max_probes_per_question_per_operation,
        )
        selected_rows.extend(trimmed)
        audit[f"retained_{operation}"] = len(trimmed)
    audit["retained_probe_total"] = len(selected_rows)
    return selected_rows, dict(sorted(audit.items()))


def summarize_probe_bank(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    question_input: Sequence[str],
    candidate_bank_jsonl: str,
    standardized_input_jsonl: str,
    output_dir: str,
    seed: int,
    max_probes_per_question_per_operation: int,
    group_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    operation_counts = Counter(clean_text(row.get("semantic_operation")) for row in rows)
    direction_counts = Counter(clean_text(row.get("direction_label")) for row in rows)
    support_counts = Counter(clean_text(row.get("changed_entity_support_status")) for row in rows)
    positive_source_counts = Counter(clean_text(row.get("positive_source")) or "none" for row in rows)
    question_counts_by_operation: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        question_counts_by_operation[clean_text(row.get("semantic_operation"))].add(clean_text(row.get("question_id")))

    return {
        "study_name": "Cardinality Shortcut Study",
        "artifact_type": "constructed_minimal_edit_probe_bank",
        "split": split,
        "question_input": list(question_input),
        "candidate_bank_jsonl": candidate_bank_jsonl,
        "standardized_input_jsonl": standardized_input_jsonl,
        "output_dir": output_dir,
        "seed": seed,
        "probe_source": "constructed_minimal_edit",
        "selection_policy": {
            "max_probes_per_question_per_operation": max_probes_per_question_per_operation,
            "target_operations": list(TARGET_OPERATIONS),
            "expected_edit_distance": dict(EXPECTED_EDIT_DISTANCE),
            "ranking_order": [
                "edit_distance_deviation_from_target",
                "absolute_token_length_gap",
                "f1_margin",
                "dispreferred_f1",
            ],
        },
        "probe_count": len(rows),
        "question_count": len({clean_text(row.get("question_id")) for row in rows}),
        "probe_counts_by_operation": dict(sorted(operation_counts.items())),
        "probe_counts_by_direction": dict(sorted(direction_counts.items())),
        "probe_counts_by_changed_entity_support": dict(sorted(support_counts.items())),
        "positive_source_counts": dict(sorted(positive_source_counts.items())),
        "question_counts_by_operation": {
            operation: len(question_ids)
            for operation, question_ids in sorted(question_counts_by_operation.items())
        },
        "f1_margin_distribution": summarize_numeric(float(row.get("f1_margin") or 0.0) for row in rows),
        "semantic_set_edit_distance_distribution": summarize_numeric(
            int(row.get("semantic_set_edit_distance") or 0) for row in rows
        ),
        "absolute_token_length_gap_distribution": summarize_numeric(
            int(row.get("absolute_token_length_gap") or 0) for row in rows
        ),
        "gold_entity_count_distribution": summarize_numeric(
            int(row.get("gold_entity_count") or 0) for row in rows
        ),
        "group_audits": {
            question_group_id: dict(audit)
            for question_group_id, audit in sorted(group_audits.items())
        },
        "notes": [
            "This artifact contains constructed minimal-edit probes only.",
            "Each retained probe is created by editing a held-out base response from the requested split.",
            "The changed_entity_support_status field uses the project evidence-support helper rather than a manual annotation.",
        ],
    }


def write_empty_outputs(
    *,
    output_dir: Path,
    split: str,
    question_input: Sequence[str],
    candidate_bank_jsonl: str,
    standardized_input_jsonl: str,
    seed: int,
    max_probes_per_question_per_operation: int,
) -> None:
    combined_path = output_dir / f"constructed_minimal_edit_probes_{split}.jsonl"
    write_jsonl(combined_path, [])
    for operation in TARGET_OPERATIONS:
        write_jsonl(output_dir / f"constructed_minimal_edit_probes_{split}_{operation}.jsonl", [])
    write_json(
        output_dir / f"constructed_minimal_edit_probe_summary_{split}.json",
        {
            "study_name": "Cardinality Shortcut Study",
            "artifact_type": "constructed_minimal_edit_probe_bank",
            "split": split,
            "question_input": list(question_input),
            "candidate_bank_jsonl": candidate_bank_jsonl,
            "standardized_input_jsonl": standardized_input_jsonl,
            "output_dir": str(output_dir),
            "seed": seed,
            "probe_count": 0,
            "question_count": 0,
            "probe_counts_by_operation": {operation: 0 for operation in TARGET_OPERATIONS},
            "status": "empty_input",
            "selection_policy": {
                "max_probes_per_question_per_operation": max_probes_per_question_per_operation,
                "target_operations": list(TARGET_OPERATIONS),
            },
            "notes": [
                "No standardized candidate rows were available for the requested split.",
                "This is expected when the held-out test candidate bank has not yet been generated.",
            ],
        },
    )


def main() -> None:
    args = parse_args()

    standardized_input_path = resolve_project_path(str(args.standardized_input_jsonl))
    candidate_bank_path = resolve_project_path(str(args.candidate_bank_jsonl))
    output_dir = resolve_project_path(str(args.output_dir))

    allowed_response_ids, allowed_question_ids, _metadata_by_response_id = load_allowed_split_records(
        standardized_input_path,
        expected_split=str(args.split),
    )
    if not allowed_response_ids:
        write_empty_outputs(
            output_dir=output_dir,
            split=str(args.split),
            question_input=[str(resolve_project_path(path)) for path in args.question_input],
            candidate_bank_jsonl=str(candidate_bank_path),
            standardized_input_jsonl=str(standardized_input_path),
            seed=int(args.seed),
            max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
        )
        print(f"No standardized candidate rows available for split='{args.split}'. Wrote empty constructed-probe files to {output_dir}")
        return

    questions_by_id = load_question_examples(
        paths=[str(resolve_project_path(path)) for path in args.question_input],
        dataset_name=str(args.dataset_name),
        max_resources=int(args.max_resources),
        max_resource_chars=int(args.max_resource_chars),
        gold_support_policy=str(args.gold_support_policy),
    )
    questions_by_id = {
        question_id: question
        for question_id, question in questions_by_id.items()
        if question_id in allowed_question_ids
    }

    candidate_records = [
        record
        for record in load_candidate_bank_records(
            paths=[str(candidate_bank_path)],
            questions_by_id=questions_by_id,
            dataset_name=str(args.dataset_name),
        )
        if clean_text(record.response_id) in allowed_response_ids
    ]

    responses_by_group: dict[str, list[MatchedResponse]] = defaultdict(list)
    for record in candidate_records:
        question = questions_by_id.get(record.question_id)
        if question is None:
            continue
        response = build_matched_response(
            record=record,
            question=question,
            allow_fallback_split=bool(args.allow_fallback_split),
            allow_fallback_pair_construction=bool(args.allow_fallback_pair_construction),
        )
        responses_by_group[group_key_for_response(response)].append(response)

    generator_checkpoints = {
        clean_text(response.record.generator_checkpoint) or "unknown"
        for grouped in responses_by_group.values()
        for response in grouped
    }
    if len(generator_checkpoints) > 1 and not args.allow_multiple_generator_checkpoints:
        raise ValueError(
            "Multiple generator checkpoints were found in the filtered held-out responses. "
            "Construct probes from a single frozen generator checkpoint or pass "
            "--allow-multiple-generator-checkpoints to partition by checkpoint explicitly. "
            f"Found: {sorted(generator_checkpoints)}"
        )

    all_rows: list[dict[str, Any]] = []
    group_audits: dict[str, dict[str, Any]] = {}
    for question_group_id, responses in sorted(responses_by_group.items()):
        if not responses:
            continue
        question = questions_by_id[responses[0].record.question_id]
        selected_rows, audit = build_group_probe_candidates(
            question=question,
            responses=responses,
            min_delta_f1=float(args.min_delta_f1),
            max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
            allow_gold_alias_fallback=bool(args.allow_gold_alias_fallback),
            allow_evidence_supported_negative_probes=bool(args.allow_evidence_supported_negative_probes),
            allow_gold_overlap_negative_probes=bool(args.allow_gold_overlap_negative_probes),
            allow_semantic_overlap_omission_probes=bool(args.allow_semantic_overlap_omission_probes),
            allow_evidence_unsupported_omission_probes=bool(args.allow_evidence_unsupported_omission_probes),
            split=str(args.split),
        )
        all_rows.extend(selected_rows)
        group_audits[question_group_id] = audit

    all_rows = sorted(
        all_rows,
        key=lambda row: (
            clean_text(row.get("question_id")),
            clean_text(row.get("semantic_operation")),
            clean_text(row.get("probe_id")),
        ),
    )

    combined_path = output_dir / f"constructed_minimal_edit_probes_{args.split}.jsonl"
    write_jsonl(combined_path, all_rows)
    for operation in TARGET_OPERATIONS:
        operation_rows = [
            row
            for row in all_rows
            if clean_text(row.get("semantic_operation")) == operation
        ]
        write_jsonl(
            output_dir / f"constructed_minimal_edit_probes_{args.split}_{operation}.jsonl",
            operation_rows,
        )

    summary = summarize_probe_bank(
        all_rows,
        split=str(args.split),
        question_input=[str(resolve_project_path(path)) for path in args.question_input],
        candidate_bank_jsonl=str(candidate_bank_path),
        standardized_input_jsonl=str(standardized_input_path),
        output_dir=str(output_dir),
        seed=int(args.seed),
        max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
        group_audits=group_audits,
    )
    write_json(output_dir / f"constructed_minimal_edit_probe_summary_{args.split}.json", summary)

    print("Saved Cardinality Shortcut Study constructed minimal-edit probe bank:")
    print(f"  split: {args.split}")
    print(f"  combined: {combined_path}")
    for operation in TARGET_OPERATIONS:
        print(
            f"  {operation}: "
            f"{output_dir / f'constructed_minimal_edit_probes_{args.split}_{operation}.jsonl'}"
        )
    print(
        f"  summary: "
        f"{output_dir / f'constructed_minimal_edit_probe_summary_{args.split}.json'}"
    )


if __name__ == "__main__":
    main()
