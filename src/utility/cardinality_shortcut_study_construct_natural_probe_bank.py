from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, summarize_numeric, write_json, write_jsonl
from cse_dpo.normalize_set_answers import normalize_answer_surface, serialize_list_items
from src.model_registry import slugify
from src.utility.cardinality_shortcut_study_construct_direction_mixture_pairs import (
    RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
    RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
    RESPONSE_DEDUPE_MODE_SEMANTIC,
    CandidateRow,
    build_pair_row,
    parse_float,
    parse_float_list,
    parse_int,
    parse_sequence,
    response_dedupe_key,
    response_rank_key,
)
from src.utility.data import clean_text


TARGET_OPERATIONS = (
    "fp_removal",
    "tp_restoration",
    "fp_to_tp_substitution",
)

OPERATION_TO_EXPECTED_EDIT_DISTANCE = {
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
            "Construct natural held-out probe banks for the Cardinality Shortcut Study "
            "from standardized candidate rows and the matching candidate bank."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Standardized candidate JSONL for one split, e.g. standardized_candidates_validation.jsonl.",
    )
    parser.add_argument(
        "--candidate-bank-jsonl",
        required=True,
        help="Candidate-bank JSONL used to recover question prompts and evidence snippets.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the natural probe bank and summary files will be written.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Expected split label in the standardized candidate file.",
    )
    parser.add_argument(
        "--min-delta-f1",
        type=float,
        default=0.05,
        help="Minimum preferred-minus-dispreferred F1 margin required for a probe.",
    )
    parser.add_argument(
        "--min-response-f1",
        type=float,
        default=0.0,
        help="Minimum response F1 before a candidate can participate in probe construction.",
    )
    parser.add_argument(
        "--max-probes-per-question-per-operation",
        type=int,
        default=4,
        help="Maximum retained natural probes per question/checkpoint group for each semantic operation.",
    )
    parser.add_argument(
        "--max-response-entities",
        type=int,
        default=None,
        help="Optional cap on the number of entities allowed in a candidate response.",
    )
    parser.add_argument(
        "--max-semantic-set-edit-distance",
        type=int,
        default=None,
        help="Optional hard cap on semantic set edit distance between the two responses.",
    )
    parser.add_argument(
        "--max-entity-gap",
        type=int,
        default=None,
        help="Optional cap on the absolute entity-count difference between the two responses.",
    )
    parser.add_argument(
        "--max-entity-ratio",
        type=float,
        default=None,
        help="Optional cap on the larger/smaller entity-count ratio between the two responses.",
    )
    parser.add_argument(
        "--response-dedupe-mode",
        choices=[
            RESPONSE_DEDUPE_MODE_SEMANTIC,
            RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
            RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
        ],
        default=RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
        help="How to deduplicate candidate responses before pairing.",
    )
    parser.add_argument(
        "--f1-margin-bin-edges",
        default="0.05,0.10,0.20,0.30,0.50",
        help="Comma-separated bin edges for probe metadata.",
    )
    parser.add_argument(
        "--edit-size-bin-edges",
        default="1,2,3,4,6",
        help="Comma-separated bin edges for probe metadata.",
    )
    parser.add_argument(
        "--token-gap-bin-edges",
        default="20,50,100,200,400",
        help="Comma-separated bin edges for probe metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Recorded seed value for reproducibility metadata.",
    )
    return parser.parse_args()


def load_candidate_rows_allow_empty(path: Path, *, expected_split: str) -> list[CandidateRow]:
    raw_rows = load_json_records(path)
    if not raw_rows:
        return []

    candidate_rows: list[CandidateRow] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        split_name = clean_text(row.get("split")).lower()
        if split_name and split_name != expected_split:
            continue

        generator_checkpoint = clean_text(row.get("generator_checkpoint")) or "unknown"
        question_id = clean_text(row.get("question_id"))
        question_group_id = f"{question_id}::{generator_checkpoint}"
        predicted_entities = parse_sequence(row.get("predicted_entities"))
        normalized_set = parse_sequence(row.get("normalized_entity_set"))
        if not normalized_set:
            normalized_set = tuple(
                sorted(
                    {
                        normalized
                        for entity in predicted_entities
                        if (normalized := normalize_answer_surface(entity))
                    }
                )
            )

        candidate_rows.append(
            CandidateRow(
                dataset=clean_text(row.get("dataset")) or "bioasq",
                question_id=question_id,
                question_group_id=question_group_id,
                question_text=clean_text(row.get("question_text")),
                question_source_path=clean_text(row.get("question_source_path")),
                split=split_name or expected_split,
                response_id=clean_text(row.get("response_id")),
                sample_id=parse_int(row.get("sample_id")),
                prompt=clean_text(row.get("prompt")),
                raw_output=clean_text(row.get("raw_output")),
                generator_checkpoint=generator_checkpoint,
                pair_eligible=bool(row.get("pair_eligible")),
                pair_exclusion_reason=clean_text(row.get("pair_exclusion_reason")),
                predicted_entities=predicted_entities,
                normalized_entity_set=normalized_set,
                matched_entities=parse_sequence(row.get("matched_entities")),
                unmatched_entities=parse_sequence(row.get("unmatched_entities")),
                missing_gold_entities=parse_sequence(row.get("missing_gold_entities")),
                matched_gold_group_ids=tuple(parse_int(value) for value in row.get("matched_gold_group_ids", []) or []),
                missing_gold_group_ids=tuple(parse_int(value) for value in row.get("missing_gold_group_ids", []) or []),
                gold_entity_count=parse_int(row.get("gold_entity_count")),
                entity_count=parse_int(row.get("entity_count")),
                token_length_chars=parse_int(row.get("token_length_chars")),
                tp=parse_int(row.get("tp")),
                fp=parse_int(row.get("fp")),
                fn=parse_int(row.get("fn")),
                precision=parse_float(row.get("precision")),
                recall=parse_float(row.get("recall")),
                f1=parse_float(row.get("f1")),
            )
        )
    return candidate_rows


def load_standardized_response_metadata(
    path: Path,
    *,
    expected_split: str,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for raw_row in load_json_records(path):
        row = dict(raw_row)
        split_name = clean_text(row.get("split")).lower()
        if split_name and split_name != expected_split:
            continue
        response_id = clean_text(row.get("response_id"))
        if not response_id:
            continue
        metadata[response_id] = {
            "generated_token_count": parse_int(row.get("generated_token_count")),
            "parser_status": clean_text(row.get("parser_status")),
            "pair_eligible": bool(row.get("pair_eligible")),
            "pair_exclusion_reason": clean_text(row.get("pair_exclusion_reason")),
            "invalid_addition_rate": parse_float(row.get("invalid_addition_rate")),
            "valid_omission_rate": parse_float(row.get("valid_omission_rate")),
            "study_name": clean_text(row.get("study_name")),
        }
    return metadata


def load_candidate_bank_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for raw_row in load_json_records(path):
        row = dict(raw_row)
        response_id = clean_text(row.get("response_id"))
        if not response_id:
            continue
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        metadata[response_id] = {
            "question_id": clean_text(row.get("question_id")),
            "question_text": clean_text(row.get("question_text")),
            "prompt": clean_text(row.get("prompt")),
            "evidence": [str(item) for item in evidence if clean_text(item)],
            "generator_checkpoint": clean_text(row.get("generator_checkpoint")),
            "sample_id": parse_int(row.get("sample_id")),
        }
    return metadata


def normalized_to_surface_map(items: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        surface = clean_text(item)
        if not surface:
            continue
        normalized = normalize_answer_surface(surface)
        if normalized and normalized not in mapping:
            mapping[normalized] = surface
    return mapping


def surfaces_for_normalized_items(
    normalized_items: Iterable[str],
    *,
    preferred_items: Sequence[str],
    dispreferred_items: Sequence[str],
) -> list[str]:
    preferred_map = normalized_to_surface_map(preferred_items)
    dispreferred_map = normalized_to_surface_map(dispreferred_items)
    surfaces: list[str] = []
    for normalized in sorted({clean_text(item) for item in normalized_items if clean_text(item)}):
        surface = preferred_map.get(normalized) or dispreferred_map.get(normalized) or normalized
        surfaces.append(surface)
    return surfaces


def normalize_support_text(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def flatten_evidence_text(evidence_items: Sequence[str]) -> str:
    chunks: list[str] = []
    for item in evidence_items:
        normalized = normalize_support_text(item)
        if normalized:
            chunks.append(normalized)
    return " ".join(chunks)


def entity_support_details(
    entity_surfaces: Sequence[str],
    *,
    evidence_items: Sequence[str],
) -> list[dict[str, Any]]:
    evidence_text = flatten_evidence_text(evidence_items)
    details: list[dict[str, Any]] = []
    for surface in entity_surfaces:
        normalized_surface = normalize_support_text(surface)
        supported = bool(normalized_surface) and normalized_surface in evidence_text
        details.append(
            {
                "surface": clean_text(surface),
                "normalized_surface": normalized_surface,
                "evidence_supported": supported,
            }
        )
    return details


def summarize_support(details: Sequence[Mapping[str, Any]]) -> str:
    if not details:
        return "not_applicable"
    flags = [bool(item.get("evidence_supported")) for item in details]
    if all(flags):
        return "supported"
    if not any(flags):
        return "unsupported"
    return "mixed"


def operation_specific_changed_entity(
    *,
    semantic_operation: str,
    preferred_only_entities: Sequence[str],
    dispreferred_only_entities: Sequence[str],
) -> str | None:
    if semantic_operation == "fp_removal":
        return preferred_only_entities[0] if len(preferred_only_entities) == 1 else (
            dispreferred_only_entities[0] if len(dispreferred_only_entities) == 1 else None
        )
    if semantic_operation == "tp_restoration":
        return preferred_only_entities[0] if len(preferred_only_entities) == 1 else None
    if semantic_operation == "fp_to_tp_substitution":
        if len(preferred_only_entities) == 1 and len(dispreferred_only_entities) == 1:
            return f"remove:{dispreferred_only_entities[0]} -> add:{preferred_only_entities[0]}"
        return None
    return None


def expected_edit_deviation(row: Mapping[str, Any]) -> int:
    operation = clean_text(row.get("semantic_operation"))
    expected = OPERATION_TO_EXPECTED_EDIT_DISTANCE.get(operation)
    if expected is None:
        return 999999
    observed = parse_int(row.get("semantic_set_edit_distance"))
    return abs(observed - expected)


def probe_selection_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        expected_edit_deviation(row),
        parse_int(row.get("semantic_set_edit_distance")),
        parse_int(row.get("absolute_entity_gap")),
        parse_int(row.get("absolute_token_length_gap")),
        round(parse_float(row.get("delta_f1")), 6),
        -round(parse_float(row.get("rejected_f1")), 6),
        -round(parse_float(row.get("chosen_f1")), 6),
        clean_text(row.get("source_pair_id")),
    )


def dedupe_and_trim_probes(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_probes: int,
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=probe_selection_rank):
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
        if len(selected) >= max_probes:
            break
    return selected


def build_probe_row(
    *,
    pair_row: Mapping[str, Any],
    standardized_metadata_by_response: Mapping[str, Mapping[str, Any]],
    candidate_bank_by_response: Mapping[str, Mapping[str, Any]],
    split: str,
    probe_source: str,
) -> dict[str, Any]:
    preferred_response_id = clean_text(pair_row.get("chosen_response_id"))
    dispreferred_response_id = clean_text(pair_row.get("rejected_response_id"))
    preferred_bank = candidate_bank_by_response.get(preferred_response_id, {})
    dispreferred_bank = candidate_bank_by_response.get(dispreferred_response_id, {})
    preferred_meta = standardized_metadata_by_response.get(preferred_response_id, {})
    dispreferred_meta = standardized_metadata_by_response.get(dispreferred_response_id, {})

    evidence = preferred_bank.get("evidence") or dispreferred_bank.get("evidence") or []
    question_text = clean_text(
        preferred_bank.get("question_text")
        or dispreferred_bank.get("question_text")
        or pair_row.get("question_text")
    )
    prompt = clean_text(
        preferred_bank.get("prompt")
        or dispreferred_bank.get("prompt")
        or pair_row.get("prompt")
    )

    preferred_items = [clean_text(item) for item in pair_row.get("chosen_items", []) or [] if clean_text(item)]
    dispreferred_items = [clean_text(item) for item in pair_row.get("rejected_items", []) or [] if clean_text(item)]
    preferred_normalized = {
        normalized
        for item in preferred_items
        if (normalized := normalize_answer_surface(item))
    }
    dispreferred_normalized = {
        normalized
        for item in dispreferred_items
        if (normalized := normalize_answer_surface(item))
    }
    preferred_only_normalized = preferred_normalized - dispreferred_normalized
    dispreferred_only_normalized = dispreferred_normalized - preferred_normalized
    preferred_only_entities = surfaces_for_normalized_items(
        preferred_only_normalized,
        preferred_items=preferred_items,
        dispreferred_items=dispreferred_items,
    )
    dispreferred_only_entities = surfaces_for_normalized_items(
        dispreferred_only_normalized,
        preferred_items=preferred_items,
        dispreferred_items=dispreferred_items,
    )

    preferred_only_support_details = entity_support_details(preferred_only_entities, evidence_items=evidence)
    dispreferred_only_support_details = entity_support_details(dispreferred_only_entities, evidence_items=evidence)
    all_changed_details = [
        {**detail, "side": "preferred_only"}
        for detail in preferred_only_support_details
    ] + [
        {**detail, "side": "dispreferred_only"}
        for detail in dispreferred_only_support_details
    ]

    semantic_operation = clean_text(pair_row.get("semantic_operation"))
    changed_entity = operation_specific_changed_entity(
        semantic_operation=semantic_operation,
        preferred_only_entities=preferred_only_entities,
        dispreferred_only_entities=dispreferred_only_entities,
    )

    return {
        "probe_id": f"{clean_text(pair_row.get('pair_id'))}-natural-probe",
        "source_pair_id": clean_text(pair_row.get("pair_id")),
        "split": split,
        "probe_source": probe_source,
        "semantic_operation": semantic_operation,
        "direction_label": clean_text(pair_row.get("direction_label")),
        "question_id": clean_text(pair_row.get("question_id")),
        "question_group_id": clean_text(pair_row.get("question_group_id")),
        "question_text": question_text,
        "question_source_path": clean_text(pair_row.get("question_source_path")),
        "prompt": prompt,
        "evidence": list(evidence),
        "preferred_answer": clean_text(pair_row.get("chosen")),
        "dispreferred_answer": clean_text(pair_row.get("rejected")),
        "preferred_items": preferred_items,
        "dispreferred_items": dispreferred_items,
        "preferred_response_id": preferred_response_id,
        "dispreferred_response_id": dispreferred_response_id,
        "preferred_sample_id": parse_int(pair_row.get("chosen_sample_id")),
        "dispreferred_sample_id": parse_int(pair_row.get("rejected_sample_id")),
        "preferred_generator_checkpoint": clean_text(pair_row.get("chosen_generator_checkpoint")),
        "dispreferred_generator_checkpoint": clean_text(pair_row.get("rejected_generator_checkpoint")),
        "generator_checkpoint": clean_text(pair_row.get("generator_checkpoint")),
        "changed_entity": changed_entity,
        "preferred_only_entities": preferred_only_entities,
        "dispreferred_only_entities": dispreferred_only_entities,
        "changed_entity_details": all_changed_details,
        "changed_entity_count": len(all_changed_details),
        "changed_entity_support_status": summarize_support(all_changed_details),
        "preferred_only_support_status": summarize_support(preferred_only_support_details),
        "dispreferred_only_support_status": summarize_support(dispreferred_only_support_details),
        "preferred_only_support_details": preferred_only_support_details,
        "dispreferred_only_support_details": dispreferred_only_support_details,
        "preferred_f1": parse_float(pair_row.get("chosen_f1")),
        "dispreferred_f1": parse_float(pair_row.get("rejected_f1")),
        "f1_margin": parse_float(pair_row.get("delta_f1")),
        "preferred_precision": parse_float(pair_row.get("chosen_precision")),
        "preferred_recall": parse_float(pair_row.get("chosen_recall")),
        "dispreferred_precision": parse_float(pair_row.get("rejected_precision")),
        "dispreferred_recall": parse_float(pair_row.get("rejected_recall")),
        "preferred_tp": parse_int(pair_row.get("chosen_tp")),
        "preferred_fp": parse_int(pair_row.get("chosen_fp")),
        "preferred_fn": parse_int(pair_row.get("chosen_fn")),
        "dispreferred_tp": parse_int(pair_row.get("rejected_tp")),
        "dispreferred_fp": parse_int(pair_row.get("rejected_fp")),
        "dispreferred_fn": parse_int(pair_row.get("rejected_fn")),
        "preferred_entity_count": parse_int(pair_row.get("chosen_entity_count")),
        "dispreferred_entity_count": parse_int(pair_row.get("rejected_entity_count")),
        "entity_count_delta": parse_int(pair_row.get("entity_count_delta")),
        "absolute_entity_gap": parse_int(pair_row.get("absolute_entity_gap")),
        "entity_ratio": pair_row.get("entity_ratio"),
        "gold_entity_count": parse_int(pair_row.get("gold_entity_count")),
        "preferred_token_length_chars": parse_int(pair_row.get("chosen_token_length_chars")),
        "dispreferred_token_length_chars": parse_int(pair_row.get("rejected_token_length_chars")),
        "token_length_delta": parse_int(pair_row.get("token_length_delta")),
        "absolute_token_length_gap": parse_int(pair_row.get("absolute_token_length_gap")),
        "preferred_generated_token_count": parse_int(preferred_meta.get("generated_token_count")),
        "dispreferred_generated_token_count": parse_int(dispreferred_meta.get("generated_token_count")),
        "semantic_set_edit_distance": parse_int(pair_row.get("semantic_set_edit_distance")),
        "f1_margin_bin": clean_text(pair_row.get("f1_margin_bin")),
        "edit_size_bin": clean_text(pair_row.get("edit_size_bin")),
        "token_gap_bin": clean_text(pair_row.get("token_gap_bin")),
        "preferred_invalid_addition_rate": parse_float(preferred_meta.get("invalid_addition_rate")),
        "preferred_valid_omission_rate": parse_float(preferred_meta.get("valid_omission_rate")),
        "dispreferred_invalid_addition_rate": parse_float(dispreferred_meta.get("invalid_addition_rate")),
        "dispreferred_valid_omission_rate": parse_float(dispreferred_meta.get("valid_omission_rate")),
        "preferred_parser_status": clean_text(preferred_meta.get("parser_status")),
        "dispreferred_parser_status": clean_text(dispreferred_meta.get("parser_status")),
        "operation_edit_deviation": expected_edit_deviation(pair_row),
    }


def build_probe_candidates_for_group(
    *,
    rows: Sequence[CandidateRow],
    standardized_metadata_by_response: Mapping[str, Mapping[str, Any]],
    candidate_bank_by_response: Mapping[str, Mapping[str, Any]],
    min_delta_f1: float,
    min_response_f1: float,
    max_probes_per_question_per_operation: int,
    max_response_entities: int | None,
    max_semantic_set_edit_distance: int | None,
    max_entity_gap: int | None,
    max_entity_ratio: float | None,
    response_dedupe_mode: str,
    f1_margin_edges: Sequence[float],
    edit_size_edges: Sequence[int],
    token_gap_edges: Sequence[int],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_rows: list[CandidateRow] = []
    audit = Counter({"responses_total": len(rows)})
    for row in rows:
        if not row.pair_eligible or not row.predicted_entities:
            audit["filtered_pair_ineligible_responses"] += 1
            continue
        if row.f1 <= min_response_f1:
            audit["filtered_low_f1_responses"] += 1
            continue
        if max_response_entities is not None and row.entity_count > max_response_entities:
            audit["filtered_large_responses"] += 1
            continue
        eligible_rows.append(row)
    audit["responses_eligible"] = len(eligible_rows)

    unique_rows: list[CandidateRow] = []
    seen_response_sets: set[tuple[str, ...]] = set()
    for row in sorted(eligible_rows, key=response_rank_key, reverse=True):
        dedupe_key = response_dedupe_key(row, response_dedupe_mode=response_dedupe_mode)
        if dedupe_key in seen_response_sets:
            audit["duplicate_response_sets"] += 1
            continue
        seen_response_sets.add(dedupe_key)
        unique_rows.append(row)
    audit["unique_eligible_response_sets"] = len(unique_rows)

    pair_id_prefix = (
        f"{slugify(unique_rows[0].dataset)}-"
        f"{slugify(unique_rows[0].question_id)}-"
        f"{slugify(unique_rows[0].generator_checkpoint)}"
    ) if unique_rows else "empty"

    probe_candidates_by_operation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_counter = 0
    for left_index, left_row in enumerate(unique_rows):
        for right_row in unique_rows[left_index + 1 :]:
            audit["candidate_response_pairs_total"] += 1
            if response_rank_key(left_row) >= response_rank_key(right_row):
                preferred = left_row
                dispreferred = right_row
            else:
                preferred = right_row
                dispreferred = left_row

            if (preferred.f1 - dispreferred.f1) < min_delta_f1:
                audit["filtered_low_delta_pairs"] += 1
                continue

            pair_row = build_pair_row(
                chosen=preferred,
                rejected=dispreferred,
                pair_id=f"{pair_id_prefix}-probe-{pair_counter:04d}",
                f1_margin_edges=f1_margin_edges,
                edit_size_edges=edit_size_edges,
                token_gap_edges=token_gap_edges,
                source_dataset_label="cardinality_shortcut_natural_heldout_probes",
            )
            pair_counter += 1
            if pair_row is None:
                audit["filtered_invalid_pairs"] += 1
                continue

            edit_distance = parse_int(pair_row.get("semantic_set_edit_distance"))
            if max_semantic_set_edit_distance is not None and edit_distance > max_semantic_set_edit_distance:
                audit["filtered_edit_distance_pairs"] += 1
                continue

            entity_gap = parse_int(pair_row.get("absolute_entity_gap"))
            if max_entity_gap is not None and entity_gap > max_entity_gap:
                audit["filtered_entity_gap_pairs"] += 1
                continue

            entity_ratio = pair_row.get("entity_ratio")
            if max_entity_ratio is not None and isinstance(entity_ratio, (int, float)) and float(entity_ratio) > max_entity_ratio:
                audit["filtered_entity_ratio_pairs"] += 1
                continue

            operation = clean_text(pair_row.get("semantic_operation"))
            if operation not in TARGET_OPERATIONS:
                audit[f"filtered_non_target_{operation or 'unknown'}"] += 1
                continue

            probe_row = build_probe_row(
                pair_row=pair_row,
                standardized_metadata_by_response=standardized_metadata_by_response,
                candidate_bank_by_response=candidate_bank_by_response,
                split=split,
                probe_source="natural",
            )
            if not probe_row["evidence"]:
                audit["probes_missing_evidence"] += 1
            probe_candidates_by_operation[operation].append(probe_row)
            audit[f"probe_candidates_{operation}"] += 1

    selected: list[dict[str, Any]] = []
    for operation in TARGET_OPERATIONS:
        trimmed = dedupe_and_trim_probes(
            probe_candidates_by_operation.get(operation, []),
            max_probes=max_probes_per_question_per_operation,
        )
        selected.extend(trimmed)
        audit[f"retained_{operation}"] = len(trimmed)
    audit["retained_probe_total"] = len(selected)
    return selected, dict(sorted(audit.items()))


def summarize_probes(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    input_jsonl: str,
    candidate_bank_jsonl: str,
    output_dir: str,
    seed: int,
    max_probes_per_question_per_operation: int,
    group_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    probe_counts_by_operation = Counter(clean_text(row.get("semantic_operation")) for row in rows)
    probe_counts_by_direction = Counter(clean_text(row.get("direction_label")) for row in rows)
    probe_counts_by_support = Counter(clean_text(row.get("changed_entity_support_status")) for row in rows)
    questions_by_operation: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        questions_by_operation[clean_text(row.get("semantic_operation"))].add(clean_text(row.get("question_id")))

    return {
        "study_name": "Cardinality Shortcut Study",
        "artifact_type": "natural_probe_bank",
        "split": split,
        "input_jsonl": input_jsonl,
        "candidate_bank_jsonl": candidate_bank_jsonl,
        "output_dir": output_dir,
        "seed": seed,
        "probe_source": "natural",
        "selection_policy": {
            "max_probes_per_question_per_operation": max_probes_per_question_per_operation,
            "target_operations": list(TARGET_OPERATIONS),
            "ranking_order": [
                "operation_edit_deviation",
                "semantic_set_edit_distance",
                "absolute_entity_gap",
                "absolute_token_length_gap",
                "f1_margin",
                "rejected_f1",
                "chosen_f1",
            ],
        },
        "probe_count": len(rows),
        "question_count": len({clean_text(row.get("question_id")) for row in rows}),
        "probe_counts_by_operation": dict(sorted(probe_counts_by_operation.items())),
        "probe_counts_by_direction": dict(sorted(probe_counts_by_direction.items())),
        "probe_counts_by_changed_entity_support": dict(sorted(probe_counts_by_support.items())),
        "question_counts_by_operation": {
            operation: len(question_ids)
            for operation, question_ids in sorted(questions_by_operation.items())
        },
        "f1_margin_distribution": summarize_numeric(parse_float(row.get("f1_margin")) for row in rows),
        "semantic_set_edit_distance_distribution": summarize_numeric(
            parse_int(row.get("semantic_set_edit_distance")) for row in rows
        ),
        "absolute_token_length_gap_distribution": summarize_numeric(
            parse_int(row.get("absolute_token_length_gap")) for row in rows
        ),
        "gold_entity_count_distribution": summarize_numeric(
            parse_int(row.get("gold_entity_count")) for row in rows
        ),
        "group_audits": {
            question_group_id: dict(audit)
            for question_group_id, audit in sorted(group_audits.items())
        },
        "notes": [
            "This script constructs natural probes only. Constructed minimal-edit probes remain a separate later step.",
            "changed_entity_support_status is a surface-match evidence-support proxy computed from the saved candidate-bank evidence snippets.",
            "Validation and test banks should be interpreted separately from any later pooled summary.",
        ],
    }


def write_empty_outputs(
    *,
    output_dir: Path,
    split: str,
    input_jsonl: str,
    candidate_bank_jsonl: str,
    seed: int,
    max_probes_per_question_per_operation: int,
) -> None:
    combined_path = output_dir / f"natural_probes_{split}.jsonl"
    write_jsonl(combined_path, [])
    for operation in TARGET_OPERATIONS:
        write_jsonl(output_dir / f"natural_probes_{split}_{operation}.jsonl", [])
    write_json(
        output_dir / f"natural_probe_summary_{split}.json",
        {
            "study_name": "Cardinality Shortcut Study",
            "artifact_type": "natural_probe_bank",
            "split": split,
            "input_jsonl": input_jsonl,
            "candidate_bank_jsonl": candidate_bank_jsonl,
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

    input_path = resolve_project_path(str(args.input_jsonl))
    candidate_bank_path = resolve_project_path(str(args.candidate_bank_jsonl))
    output_dir = resolve_project_path(str(args.output_dir))

    candidate_rows = load_candidate_rows_allow_empty(input_path, expected_split=str(args.split))
    if not candidate_rows:
        write_empty_outputs(
            output_dir=output_dir,
            split=str(args.split),
            input_jsonl=str(input_path),
            candidate_bank_jsonl=str(candidate_bank_path),
            seed=int(args.seed),
            max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
        )
        print(f"No standardized candidate rows available for split='{args.split}'. Wrote empty probe-bank files to {output_dir}")
        return

    standardized_metadata_by_response = load_standardized_response_metadata(
        input_path,
        expected_split=str(args.split),
    )
    candidate_bank_by_response = load_candidate_bank_metadata(candidate_bank_path)

    rows_by_group: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_group[row.question_group_id].append(row)

    f1_margin_edges = parse_float_list(str(args.f1_margin_bin_edges))
    edit_size_edges = [int(value) for value in parse_float_list(str(args.edit_size_bin_edges))]
    token_gap_edges = [int(value) for value in parse_float_list(str(args.token_gap_bin_edges))]

    all_selected_rows: list[dict[str, Any]] = []
    group_audits: dict[str, dict[str, Any]] = {}
    for question_group_id, rows in sorted(rows_by_group.items()):
        selected_rows, audit = build_probe_candidates_for_group(
            rows=rows,
            standardized_metadata_by_response=standardized_metadata_by_response,
            candidate_bank_by_response=candidate_bank_by_response,
            min_delta_f1=float(args.min_delta_f1),
            min_response_f1=float(args.min_response_f1),
            max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
            max_response_entities=args.max_response_entities,
            max_semantic_set_edit_distance=args.max_semantic_set_edit_distance,
            max_entity_gap=args.max_entity_gap,
            max_entity_ratio=args.max_entity_ratio,
            response_dedupe_mode=str(args.response_dedupe_mode),
            f1_margin_edges=f1_margin_edges,
            edit_size_edges=edit_size_edges,
            token_gap_edges=token_gap_edges,
            split=str(args.split),
        )
        all_selected_rows.extend(selected_rows)
        group_audits[question_group_id] = audit

    all_selected_rows = sorted(
        all_selected_rows,
        key=lambda row: (
            clean_text(row.get("question_id")),
            clean_text(row.get("semantic_operation")),
            clean_text(row.get("probe_id")),
        ),
    )

    combined_path = output_dir / f"natural_probes_{args.split}.jsonl"
    write_jsonl(combined_path, all_selected_rows)
    for operation in TARGET_OPERATIONS:
        operation_rows = [
            row
            for row in all_selected_rows
            if clean_text(row.get("semantic_operation")) == operation
        ]
        write_jsonl(output_dir / f"natural_probes_{args.split}_{operation}.jsonl", operation_rows)

    summary = summarize_probes(
        all_selected_rows,
        split=str(args.split),
        input_jsonl=str(input_path),
        candidate_bank_jsonl=str(candidate_bank_path),
        output_dir=str(output_dir),
        seed=int(args.seed),
        max_probes_per_question_per_operation=int(args.max_probes_per_question_per_operation),
        group_audits=group_audits,
    )
    write_json(output_dir / f"natural_probe_summary_{args.split}.json", summary)

    print("Saved Cardinality Shortcut Study natural probe bank:")
    print(f"  split: {args.split}")
    print(f"  combined: {combined_path}")
    for operation in TARGET_OPERATIONS:
        print(f"  {operation}: {output_dir / f'natural_probes_{args.split}_{operation}.jsonl'}")
    print(f"  summary: {output_dir / f'natural_probe_summary_{args.split}.json'}")


if __name__ == "__main__":
    main()
