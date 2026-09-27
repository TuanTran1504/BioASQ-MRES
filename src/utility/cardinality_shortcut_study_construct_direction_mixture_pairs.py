from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.audit_preference_pairs import build_summary
from cse_dpo.common import load_json_records, summarize_numeric, write_json, write_jsonl
from cse_dpo.construct_whole_response_dpo_pairs import (
    RESPONSE_DEDUPE_MODE_EXACT_ORDERED,
    RESPONSE_DEDUPE_MODE_EXACT_UNORDERED,
    RESPONSE_DEDUPE_MODE_SEMANTIC,
)
from cse_dpo.normalize_set_answers import normalize_answer_surface, serialize_list_items
from src.model_registry import slugify
from src.utility.data import clean_text


PAIR_TYPE_WHOLE_RESPONSE_METRIC = "whole_response_metric"


@dataclass(frozen=True)
class CandidateRow:
    dataset: str
    question_id: str
    question_group_id: str
    question_text: str
    question_source_path: str
    split: str
    response_id: str
    sample_id: int
    prompt: str
    raw_output: str
    generator_checkpoint: str
    pair_eligible: bool
    pair_exclusion_reason: str
    predicted_entities: tuple[str, ...]
    normalized_entity_set: tuple[str, ...]
    matched_entities: tuple[str, ...]
    unmatched_entities: tuple[str, ...]
    missing_gold_entities: tuple[str, ...]
    matched_gold_group_ids: tuple[int, ...]
    missing_gold_group_ids: tuple[int, ...]
    gold_entity_count: int
    entity_count: int
    token_length_chars: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class PairCandidate:
    row: dict[str, Any]
    rank: tuple[Any, ...]


@dataclass(frozen=True)
class DirectionFamily:
    matched_family_id: str
    question_group_id: str
    question_id: str
    generator_checkpoint: str
    short_pair: dict[str, Any]
    long_pair: dict[str, Any]
    family_stratum: str
    same_f1_margin_bin: bool
    same_edit_size_bin: bool
    delta_f1_abs_diff: float
    edit_size_abs_diff: int
    token_gap_abs_diff: int
    mixed_edit_penalty: int
    match_score: tuple[Any, ...]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_float_list(value: str) -> list[float]:
    items: list[float] = []
    for chunk in value.split(","):
        text = clean_text(chunk)
        if not text:
            continue
        items.append(float(text))
    return items


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for chunk in value.split(","):
        text = clean_text(chunk)
        if not text:
            continue
        items.append(int(text))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct ordinary and direction-mixture preference-pair datasets for the "
            "Cardinality Shortcut Study from standardized candidate rows."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Standardized candidate JSONL, typically standardized_candidates_train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the constructed pair datasets and summaries will be written.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Expected split label in the standardized candidate file.",
    )
    parser.add_argument(
        "--min-delta-f1",
        type=float,
        default=0.05,
        help="Minimum chosen-minus-rejected F1 margin required for an eligible pair.",
    )
    parser.add_argument(
        "--min-response-f1",
        type=float,
        default=0.0,
        help="Minimum response F1 required before a candidate can participate in pairing.",
    )
    parser.add_argument(
        "--max-pairs-per-question",
        type=int,
        default=8,
        help="Maximum ordinary pairs retained per question/checkpoint group.",
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
        help="Optional cap on semantic set edit distance between paired responses.",
    )
    parser.add_argument(
        "--max-entity-gap",
        type=int,
        default=None,
        help="Optional cap on the absolute entity-count difference between paired responses.",
    )
    parser.add_argument(
        "--max-entity-ratio",
        type=float,
        default=None,
        help="Optional cap on the larger/smaller entity-count ratio between paired responses.",
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
        "--allow-multiple-generator-checkpoints",
        action="store_true",
        help="Allow multiple generator checkpoints. Pairing still stays within each checkpoint.",
    )
    parser.add_argument(
        "--f1-margin-bin-edges",
        default="0.05,0.10,0.20,0.30,0.50",
        help="Comma-separated bin edges for delta-F1 matching within direction families.",
    )
    parser.add_argument(
        "--edit-size-bin-edges",
        default="1,2,3,4,6",
        help="Comma-separated bin edges for semantic set edit distance matching within direction families.",
    )
    parser.add_argument(
        "--token-gap-bin-edges",
        default="20,50,100,200,400",
        help="Comma-separated bin edges for absolute token-length-gap summaries.",
    )
    parser.add_argument(
        "--longer-skewed-shorter-ratio",
        type=float,
        default=0.25,
        help="Shorter-preferred ratio for the longer-skewed condition.",
    )
    parser.add_argument(
        "--balanced-shorter-ratio",
        type=float,
        default=0.50,
        help="Shorter-preferred ratio for the direction-balanced condition.",
    )
    parser.add_argument(
        "--shorter-skewed-shorter-ratio",
        type=float,
        default=0.75,
        help="Shorter-preferred ratio for the shorter-skewed condition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Deterministic seed for all sampling and tie-breaking.",
    )
    return parser.parse_args()


def parse_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(clean_text(item) for item in value if clean_text(item))
    return ()


def parse_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = clean_text(value)
    return int(text) if text else 0


def parse_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_text(value)
    return float(text) if text else 0.0


def load_candidate_rows(path: Path, *, expected_split: str) -> list[CandidateRow]:
    rows = load_json_records(path)
    candidate_rows: list[CandidateRow] = []
    for raw_row in rows:
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
    if not candidate_rows:
        raise ValueError(f"No standardized candidate rows were loaded from {path}")
    return candidate_rows


def response_rank_key(row: CandidateRow) -> tuple[float, float, float, int, str]:
    return (
        row.f1,
        row.precision,
        row.recall,
        -row.entity_count,
        row.response_id,
    )


def response_dedupe_key(row: CandidateRow, *, response_dedupe_mode: str) -> tuple[str, ...]:
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_SEMANTIC:
        return tuple(sorted(set(row.normalized_entity_set)))
    normalized_items = [
        normalized
        for item in row.predicted_entities
        if (normalized := normalize_answer_surface(item))
    ]
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_EXACT_UNORDERED:
        return tuple(sorted(normalized_items))
    if response_dedupe_mode == RESPONSE_DEDUPE_MODE_EXACT_ORDERED:
        return tuple(normalized_items)
    raise ValueError(f"Unsupported response_dedupe_mode: {response_dedupe_mode}")


def semantic_set_edit_distance(left: CandidateRow, right: CandidateRow) -> int:
    return len(set(left.normalized_entity_set).symmetric_difference(set(right.normalized_entity_set)))


def bin_numeric(value: float, edges: Sequence[float], *, prefix: str) -> str:
    ordered_edges = sorted(float(edge) for edge in edges)
    for edge in ordered_edges:
        if value <= edge:
            return f"{prefix}_le_{edge:g}"
    if not ordered_edges:
        return f"{prefix}_all"
    return f"{prefix}_gt_{ordered_edges[-1]:g}"


def classify_direction(chosen: CandidateRow, rejected: CandidateRow) -> str:
    if chosen.entity_count < rejected.entity_count:
        return "shorter_preferred"
    if chosen.entity_count > rejected.entity_count:
        return "longer_preferred"
    return "equal_preferred"


def classify_semantic_operation(chosen: CandidateRow, rejected: CandidateRow, *, direction: str) -> str:
    delta_tp = chosen.tp - rejected.tp
    delta_fp = chosen.fp - rejected.fp
    delta_fn = chosen.fn - rejected.fn
    if direction == "shorter_preferred" and delta_fp < 0 and delta_tp == 0 and delta_fn == 0:
        return "fp_removal"
    if direction == "longer_preferred" and delta_tp > 0 and delta_fp == 0 and delta_fn < 0:
        return "tp_restoration"
    if direction == "equal_preferred" and delta_tp > 0 and delta_fp < 0 and delta_fn < 0:
        return "fp_to_tp_substitution"
    return "mixed_edit"


def generator_checkpoint_value(chosen: CandidateRow, rejected: CandidateRow) -> str:
    if chosen.generator_checkpoint == rejected.generator_checkpoint:
        return chosen.generator_checkpoint
    return f"{chosen.generator_checkpoint} || {rejected.generator_checkpoint}"


def build_pair_row(
    *,
    chosen: CandidateRow,
    rejected: CandidateRow,
    pair_id: str,
    f1_margin_edges: Sequence[float],
    edit_size_edges: Sequence[int],
    token_gap_edges: Sequence[int],
    source_dataset_label: str,
) -> dict[str, Any] | None:
    if not chosen.predicted_entities or not rejected.predicted_entities:
        return None
    if chosen.f1 <= rejected.f1:
        return None

    edit_distance = semantic_set_edit_distance(chosen, rejected)
    if edit_distance <= 0:
        return None

    direction = classify_direction(chosen, rejected)
    operation = classify_semantic_operation(chosen, rejected, direction=direction)
    entity_gap = abs(chosen.entity_count - rejected.entity_count)
    smaller_count = min(chosen.entity_count, rejected.entity_count)
    entity_ratio = (
        float("inf")
        if smaller_count == 0 and max(chosen.entity_count, rejected.entity_count) > 0
        else (max(chosen.entity_count, rejected.entity_count) / max(1, smaller_count))
    )
    token_delta = chosen.token_length_chars - rejected.token_length_chars
    f1_margin = chosen.f1 - rejected.f1
    delta_precision = chosen.precision - rejected.precision
    delta_recall = chosen.recall - rejected.recall
    delta_tp = chosen.tp - rejected.tp
    delta_fp = chosen.fp - rejected.fp
    delta_fn = chosen.fn - rejected.fn

    return {
        "pair_id": pair_id,
        "dataset": chosen.dataset,
        "question_id": chosen.question_id,
        "question_group_id": chosen.question_group_id,
        "question_text": chosen.question_text,
        "question_source_path": chosen.question_source_path,
        "prompt": chosen.prompt or rejected.prompt,
        "chosen": serialize_list_items(chosen.predicted_entities),
        "rejected": serialize_list_items(rejected.predicted_entities),
        "pair_type": PAIR_TYPE_WHOLE_RESPONSE_METRIC,
        "base_response_id": rejected.response_id,
        "edited_candidate": "",
        "edited_candidate_normalized": "",
        "edited_gold_group_id": None,
        "candidate_label": "whole_response_ranked_by_f1",
        "candidate_label_source": source_dataset_label,
        "positive_source": None,
        "semantic_set_edit_distance": edit_distance,
        "chosen_items": list(chosen.predicted_entities),
        "rejected_items": list(rejected.predicted_entities),
        "chosen_precision": chosen.precision,
        "chosen_recall": chosen.recall,
        "chosen_f1": chosen.f1,
        "rejected_precision": rejected.precision,
        "rejected_recall": rejected.recall,
        "rejected_f1": rejected.f1,
        "delta_precision": delta_precision,
        "delta_recall": delta_recall,
        "delta_f1": f1_margin,
        "generator_checkpoint": generator_checkpoint_value(chosen, rejected),
        "sample_id": chosen.sample_id,
        "chosen_response_id": chosen.response_id,
        "rejected_response_id": rejected.response_id,
        "chosen_sample_id": chosen.sample_id,
        "rejected_sample_id": rejected.sample_id,
        "chosen_generator_checkpoint": chosen.generator_checkpoint,
        "rejected_generator_checkpoint": rejected.generator_checkpoint,
        "chosen_entity_count": chosen.entity_count,
        "rejected_entity_count": rejected.entity_count,
        "chosen_token_length_chars": chosen.token_length_chars,
        "rejected_token_length_chars": rejected.token_length_chars,
        "chosen_tp": chosen.tp,
        "chosen_fp": chosen.fp,
        "chosen_fn": chosen.fn,
        "rejected_tp": rejected.tp,
        "rejected_fp": rejected.fp,
        "rejected_fn": rejected.fn,
        "gold_entity_count": chosen.gold_entity_count,
        "entity_count_delta": chosen.entity_count - rejected.entity_count,
        "absolute_entity_gap": entity_gap,
        "entity_ratio": entity_ratio,
        "token_length_delta": token_delta,
        "absolute_token_length_gap": abs(token_delta),
        "tp_delta": delta_tp,
        "fp_delta": delta_fp,
        "fn_delta": delta_fn,
        "direction_label": direction,
        "semantic_operation": operation,
        "f1_margin_bin": bin_numeric(f1_margin, f1_margin_edges, prefix="delta_f1"),
        "edit_size_bin": bin_numeric(float(edit_distance), edit_size_edges, prefix="edit"),
        "token_gap_bin": bin_numeric(float(abs(token_delta)), token_gap_edges, prefix="token_gap"),
    }


def pair_selection_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row.get("delta_f1") or 0.0),
        float(row.get("chosen_f1") or 0.0),
        float(row.get("chosen_precision") or 0.0),
        -float(row.get("rejected_f1") or 0.0),
        clean_text(row.get("pair_id")),
    )


def dedupe_and_trim_pairs(
    pairs: Sequence[dict[str, Any]],
    *,
    max_pairs: int,
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    selected: list[dict[str, Any]] = []
    for row in sorted(pairs, key=pair_selection_rank, reverse=True):
        key = (
            clean_text(row.get("question_group_id")),
            tuple(clean_text(item) for item in row.get("chosen_items", [])),
            tuple(clean_text(item) for item in row.get("rejected_items", [])),
            clean_text(row.get("pair_type")),
        )
        reverse_key = (key[0], key[2], key[1], key[3])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        selected.append(dict(row))
        if len(selected) >= max_pairs:
            break
    return selected


def build_pairs_for_group(
    *,
    rows: Sequence[CandidateRow],
    min_delta_f1: float,
    min_response_f1: float,
    max_pairs_per_question: int,
    max_response_entities: int | None,
    max_semantic_set_edit_distance: int | None,
    max_entity_gap: int | None,
    max_entity_ratio: float | None,
    response_dedupe_mode: str,
    f1_margin_edges: Sequence[float],
    edit_size_edges: Sequence[int],
    token_gap_edges: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eligible_rows: list[CandidateRow] = []
    audit = Counter(
        {
            "responses_total": len(rows),
        }
    )
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

    all_pairs: list[PairCandidate] = []
    pair_counter = 0
    for left_index, left_row in enumerate(unique_rows):
        for right_row in unique_rows[left_index + 1 :]:
            audit["candidate_response_pairs_total"] += 1
            if response_rank_key(left_row) >= response_rank_key(right_row):
                chosen = left_row
                rejected = right_row
            else:
                chosen = right_row
                rejected = left_row

            delta_f1 = chosen.f1 - rejected.f1
            if delta_f1 < min_delta_f1:
                audit["filtered_low_delta_pairs"] += 1
                continue

            pair = build_pair_row(
                chosen=chosen,
                rejected=rejected,
                pair_id=f"{pair_id_prefix}-whole-{pair_counter:04d}",
                f1_margin_edges=f1_margin_edges,
                edit_size_edges=edit_size_edges,
                token_gap_edges=token_gap_edges,
                source_dataset_label="cardinality_shortcut_standardized_candidates",
            )
            pair_counter += 1
            if pair is None:
                audit["filtered_invalid_pairs"] += 1
                continue

            edit_distance = int(pair["semantic_set_edit_distance"])
            if max_semantic_set_edit_distance is not None and edit_distance > max_semantic_set_edit_distance:
                audit["filtered_edit_distance_pairs"] += 1
                continue

            entity_gap = int(pair["absolute_entity_gap"])
            if max_entity_gap is not None and entity_gap > max_entity_gap:
                audit["filtered_entity_gap_pairs"] += 1
                continue

            entity_ratio = float(pair["entity_ratio"])
            if max_entity_ratio is not None and entity_ratio > max_entity_ratio:
                audit["filtered_entity_ratio_pairs"] += 1
                continue

            all_pairs.append(
                PairCandidate(
                    row=pair,
                    rank=pair_selection_rank(pair),
                )
            )

    all_pair_rows = [candidate.row for candidate in sorted(all_pairs, key=lambda item: item.rank, reverse=True)]
    ordinary_pairs = dedupe_and_trim_pairs(all_pair_rows, max_pairs=max_pairs_per_question)
    audit["emitted_whole_response_pairs"] = len(ordinary_pairs)
    return all_pair_rows, ordinary_pairs, dict(sorted(audit.items()))


def build_direction_family(
    *,
    question_group_id: str,
    short_pairs: Sequence[dict[str, Any]],
    long_pairs: Sequence[dict[str, Any]],
    family_index: int,
) -> DirectionFamily:
    best_family: DirectionFamily | None = None
    question_id = clean_text(short_pairs[0].get("question_id")) if short_pairs else clean_text(long_pairs[0].get("question_id"))
    generator_checkpoint = clean_text(short_pairs[0].get("generator_checkpoint")) if short_pairs else clean_text(long_pairs[0].get("generator_checkpoint"))

    for short_pair in short_pairs:
        for long_pair in long_pairs:
            same_f1_margin_bin = clean_text(short_pair.get("f1_margin_bin")) == clean_text(long_pair.get("f1_margin_bin"))
            same_edit_size_bin = clean_text(short_pair.get("edit_size_bin")) == clean_text(long_pair.get("edit_size_bin"))
            delta_f1_abs_diff = abs(float(short_pair.get("delta_f1") or 0.0) - float(long_pair.get("delta_f1") or 0.0))
            edit_size_abs_diff = abs(
                int(short_pair.get("semantic_set_edit_distance") or 0)
                - int(long_pair.get("semantic_set_edit_distance") or 0)
            )
            token_gap_abs_diff = abs(
                int(short_pair.get("absolute_token_length_gap") or 0)
                - int(long_pair.get("absolute_token_length_gap") or 0)
            )
            mixed_edit_penalty = int(short_pair.get("semantic_operation") == "mixed_edit") + int(
                long_pair.get("semantic_operation") == "mixed_edit"
            )
            family_stratum = (
                f"f1:{clean_text(short_pair.get('f1_margin_bin'))}|{clean_text(long_pair.get('f1_margin_bin'))}"
                f"__edit:{clean_text(short_pair.get('edit_size_bin'))}|{clean_text(long_pair.get('edit_size_bin'))}"
            )
            match_score = (
                0 if same_f1_margin_bin else 1,
                0 if same_edit_size_bin else 1,
                mixed_edit_penalty,
                round(delta_f1_abs_diff, 6),
                edit_size_abs_diff,
                token_gap_abs_diff,
                clean_text(short_pair.get("pair_id")),
                clean_text(long_pair.get("pair_id")),
            )
            family = DirectionFamily(
                matched_family_id=f"{slugify(question_group_id)}-family-{family_index:04d}",
                question_group_id=question_group_id,
                question_id=question_id,
                generator_checkpoint=generator_checkpoint,
                short_pair=dict(short_pair),
                long_pair=dict(long_pair),
                family_stratum=family_stratum,
                same_f1_margin_bin=same_f1_margin_bin,
                same_edit_size_bin=same_edit_size_bin,
                delta_f1_abs_diff=delta_f1_abs_diff,
                edit_size_abs_diff=edit_size_abs_diff,
                token_gap_abs_diff=token_gap_abs_diff,
                mixed_edit_penalty=mixed_edit_penalty,
                match_score=match_score,
            )
            if best_family is None or family.match_score < best_family.match_score:
                best_family = family

    if best_family is None:
        raise ValueError(f"Could not build a direction family for {question_group_id}")
    return best_family


def build_direction_families(all_pairs_by_group: Mapping[str, Sequence[dict[str, Any]]]) -> list[DirectionFamily]:
    families: list[DirectionFamily] = []
    for family_index, (question_group_id, pair_rows) in enumerate(sorted(all_pairs_by_group.items()), start=1):
        short_pairs = [dict(row) for row in pair_rows if clean_text(row.get("direction_label")) == "shorter_preferred"]
        long_pairs = [dict(row) for row in pair_rows if clean_text(row.get("direction_label")) == "longer_preferred"]
        if not short_pairs or not long_pairs:
            continue
        families.append(
            build_direction_family(
                question_group_id=question_group_id,
                short_pairs=short_pairs,
                long_pairs=long_pairs,
                family_index=family_index,
            )
        )
    return families


def exact_target_count(group_size: int, ratio: float) -> tuple[int, float]:
    exact = group_size * ratio
    base = math.floor(exact)
    return base, exact - base


def rounded_target_count(total_size: int, ratio: float) -> int:
    return int(math.floor((total_size * ratio) + 0.5))


def stratified_short_assignment(
    families: Sequence[DirectionFamily],
    *,
    shorter_ratio: float,
    seed: int,
) -> set[str]:
    if not families:
        return set()

    grouped: dict[str, list[DirectionFamily]] = defaultdict(list)
    for family in families:
        grouped[family.family_stratum].append(family)

    total_target = rounded_target_count(len(families), shorter_ratio)
    base_allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, items in grouped.items():
        base, remainder = exact_target_count(len(items), shorter_ratio)
        base_allocations[stratum] = base
        remainders.append((remainder, stratum))

    assigned = sum(base_allocations.values())
    remaining = max(0, total_target - assigned)
    for _remainder, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if base_allocations[stratum] >= len(grouped[stratum]):
            continue
        base_allocations[stratum] += 1
        remaining -= 1

    selected_short_family_ids: set[str] = set()
    for stratum, items in sorted(grouped.items()):
        target = min(base_allocations[stratum], len(items))
        shuffled = list(items)
        random.Random(f"{seed}:{stratum}").shuffle(shuffled)
        for family in shuffled[:target]:
            selected_short_family_ids.add(family.matched_family_id)
    return selected_short_family_ids


def family_to_manifest_row(family: DirectionFamily) -> dict[str, Any]:
    return {
        "matched_family_id": family.matched_family_id,
        "question_group_id": family.question_group_id,
        "question_id": family.question_id,
        "generator_checkpoint": family.generator_checkpoint,
        "family_stratum": family.family_stratum,
        "same_f1_margin_bin": family.same_f1_margin_bin,
        "same_edit_size_bin": family.same_edit_size_bin,
        "delta_f1_abs_diff": family.delta_f1_abs_diff,
        "edit_size_abs_diff": family.edit_size_abs_diff,
        "token_gap_abs_diff": family.token_gap_abs_diff,
        "mixed_edit_penalty": family.mixed_edit_penalty,
        "short_pair_id": clean_text(family.short_pair.get("pair_id")),
        "short_pair_direction": clean_text(family.short_pair.get("direction_label")),
        "short_pair_operation": clean_text(family.short_pair.get("semantic_operation")),
        "short_pair_delta_f1": float(family.short_pair.get("delta_f1") or 0.0),
        "short_pair_edit_size": int(family.short_pair.get("semantic_set_edit_distance") or 0),
        "long_pair_id": clean_text(family.long_pair.get("pair_id")),
        "long_pair_direction": clean_text(family.long_pair.get("direction_label")),
        "long_pair_operation": clean_text(family.long_pair.get("semantic_operation")),
        "long_pair_delta_f1": float(family.long_pair.get("delta_f1") or 0.0),
        "long_pair_edit_size": int(family.long_pair.get("semantic_set_edit_distance") or 0),
    }


def annotate_condition_pair(
    row: Mapping[str, Any],
    *,
    family: DirectionFamily,
    condition_name: str,
    target_shorter_ratio: float,
    assignment_seed: int,
) -> dict[str, Any]:
    annotated = dict(row)
    annotated["condition_name"] = condition_name
    annotated["condition_target_shorter_ratio"] = target_shorter_ratio
    annotated["matched_family_id"] = family.matched_family_id
    annotated["family_stratum"] = family.family_stratum
    annotated["family_same_f1_margin_bin"] = family.same_f1_margin_bin
    annotated["family_same_edit_size_bin"] = family.same_edit_size_bin
    annotated["family_delta_f1_abs_diff"] = family.delta_f1_abs_diff
    annotated["family_edit_size_abs_diff"] = family.edit_size_abs_diff
    annotated["family_token_gap_abs_diff"] = family.token_gap_abs_diff
    annotated["family_mixed_edit_penalty"] = family.mixed_edit_penalty
    annotated["direction_assignment_seed"] = assignment_seed
    return annotated


def condition_rows_from_families(
    families: Sequence[DirectionFamily],
    *,
    condition_name: str,
    shorter_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_short = stratified_short_assignment(families, shorter_ratio=shorter_ratio, seed=seed)
    rows: list[dict[str, Any]] = []
    for family in sorted(families, key=lambda item: item.matched_family_id):
        base_row = family.short_pair if family.matched_family_id in selected_short else family.long_pair
        rows.append(
            annotate_condition_pair(
                base_row,
                family=family,
                condition_name=condition_name,
                target_shorter_ratio=shorter_ratio,
                assignment_seed=seed,
            )
        )
    direction_counts = Counter(clean_text(row.get("direction_label")) for row in rows)
    stratum_counts = Counter(clean_text(row.get("family_stratum")) for row in rows)
    return rows, {
        "condition_name": condition_name,
        "target_shorter_ratio": shorter_ratio,
        "realized_shorter_ratio": (
            direction_counts.get("shorter_preferred", 0) / len(rows) if rows else None
        ),
        "pair_count": len(rows),
        "direction_counts": dict(sorted(direction_counts.items())),
        "family_stratum_counts": dict(sorted(stratum_counts.items())),
        "selected_pair_ids": [clean_text(row.get("pair_id")) for row in rows],
    }


def summarize_pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = build_summary(rows, question_gold_counts={clean_text(row.get("question_id")): int(row.get("gold_entity_count") or 0) for row in rows})
    summary["direction_label_counts"] = dict(
        sorted(Counter(clean_text(row.get("direction_label")) for row in rows).items())
    )
    summary["semantic_operation_counts"] = dict(
        sorted(Counter(clean_text(row.get("semantic_operation")) for row in rows).items())
    )
    summary["entity_count_delta_distribution"] = summarize_numeric(
        float(row["entity_count_delta"]) for row in rows if isinstance(row.get("entity_count_delta"), (int, float))
    )
    summary["token_length_delta_distribution"] = summarize_numeric(
        float(row["token_length_delta"]) for row in rows if isinstance(row.get("token_length_delta"), (int, float))
    )
    summary["fp_delta_distribution"] = summarize_numeric(
        float(row["fp_delta"]) for row in rows if isinstance(row.get("fp_delta"), (int, float))
    )
    summary["fn_delta_distribution"] = summarize_numeric(
        float(row["fn_delta"]) for row in rows if isinstance(row.get("fn_delta"), (int, float))
    )
    return summary


def update_condition_pair_ids_in_summary(summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    summary["pair_ids_by_question_group"] = {
        question_group_id: [clean_text(row.get("pair_id")) for row in group_rows]
        for question_group_id, group_rows in sorted(group_rows_by_question_group(rows).items())
    }


def group_rows_by_question_group(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get("question_group_id"))].append(dict(row))
    return grouped


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input_jsonl)
    output_dir = resolve_project_path(args.output_dir)

    f1_margin_edges = parse_float_list(args.f1_margin_bin_edges)
    edit_size_edges = parse_int_list(args.edit_size_bin_edges)
    token_gap_edges = parse_int_list(args.token_gap_bin_edges)

    candidate_rows = load_candidate_rows(input_path, expected_split=str(args.split))
    generator_checkpoints = sorted({row.generator_checkpoint for row in candidate_rows})
    if len(generator_checkpoints) > 1 and not args.allow_multiple_generator_checkpoints:
        raise ValueError(
            "Multiple generator checkpoints were found in the standardized candidates. "
            "Construct pairs from one frozen generator, or pass "
            "--allow-multiple-generator-checkpoints to partition by checkpoint. "
            f"Found: {generator_checkpoints}"
        )

    rows_by_group: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_group[row.question_group_id].append(row)

    all_pairs_by_group: dict[str, list[dict[str, Any]]] = {}
    ordinary_pairs: list[dict[str, Any]] = []
    ordinary_pair_audit = Counter()
    eligible_group_count = 0
    for question_group_id, rows in sorted(rows_by_group.items()):
        all_pairs, selected_pairs, audit = build_pairs_for_group(
            rows=rows,
            min_delta_f1=float(args.min_delta_f1),
            min_response_f1=float(args.min_response_f1),
            max_pairs_per_question=int(args.max_pairs_per_question),
            max_response_entities=(
                int(args.max_response_entities) if args.max_response_entities is not None else None
            ),
            max_semantic_set_edit_distance=(
                int(args.max_semantic_set_edit_distance)
                if args.max_semantic_set_edit_distance is not None
                else None
            ),
            max_entity_gap=(
                int(args.max_entity_gap) if args.max_entity_gap is not None else None
            ),
            max_entity_ratio=(
                float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
            ),
            response_dedupe_mode=str(args.response_dedupe_mode),
            f1_margin_edges=f1_margin_edges,
            edit_size_edges=edit_size_edges,
            token_gap_edges=token_gap_edges,
        )
        if all_pairs:
            eligible_group_count += 1
        ordinary_pairs.extend(selected_pairs)
        ordinary_pair_audit.update(audit)
        all_pairs_by_group[question_group_id] = all_pairs

    ordinary_pairs = sorted(ordinary_pairs, key=lambda row: (clean_text(row.get("question_id")), clean_text(row.get("pair_id"))))
    equal_cardinality_pairs = [
        dict(row)
        for row in ordinary_pairs
        if clean_text(row.get("direction_label")) == "equal_preferred"
    ]

    direction_families = build_direction_families(all_pairs_by_group)
    family_manifest_rows = [family_to_manifest_row(family) for family in direction_families]

    longer_skewed_rows, longer_skewed_manifest = condition_rows_from_families(
        direction_families,
        condition_name="longer_skewed",
        shorter_ratio=float(args.longer_skewed_shorter_ratio),
        seed=int(args.seed) + 1,
    )
    direction_balanced_rows, direction_balanced_manifest = condition_rows_from_families(
        direction_families,
        condition_name="direction_balanced",
        shorter_ratio=float(args.balanced_shorter_ratio),
        seed=int(args.seed) + 2,
    )
    shorter_skewed_rows, shorter_skewed_manifest = condition_rows_from_families(
        direction_families,
        condition_name="shorter_skewed",
        shorter_ratio=float(args.shorter_skewed_shorter_ratio),
        seed=int(args.seed) + 3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    direction_dir = output_dir / "direction_mixture"

    ordinary_pairs_path = output_dir / "ordinary_pairs.jsonl"
    equal_pairs_path = output_dir / "equal_cardinality_pairs.jsonl"
    family_manifest_path = direction_dir / "master_eligible_direction_families.jsonl"
    longer_skewed_path = direction_dir / "longer_skewed_pairs.jsonl"
    direction_balanced_path = direction_dir / "direction_balanced_pairs.jsonl"
    shorter_skewed_path = direction_dir / "shorter_skewed_pairs.jsonl"
    summary_path = output_dir / "pair_construction_summary.json"
    condition_manifest_path = direction_dir / "condition_assignment_manifest.json"

    write_jsonl(ordinary_pairs_path, ordinary_pairs)
    write_jsonl(equal_pairs_path, equal_cardinality_pairs)
    write_jsonl(family_manifest_path, family_manifest_rows)
    write_jsonl(longer_skewed_path, longer_skewed_rows)
    write_jsonl(direction_balanced_path, direction_balanced_rows)
    write_jsonl(shorter_skewed_path, shorter_skewed_rows)

    ordinary_summary = summarize_pair_rows(ordinary_pairs)
    equal_summary = summarize_pair_rows(equal_cardinality_pairs)
    longer_summary = summarize_pair_rows(longer_skewed_rows)
    direction_balanced_summary = summarize_pair_rows(direction_balanced_rows)
    shorter_summary = summarize_pair_rows(shorter_skewed_rows)
    update_condition_pair_ids_in_summary(longer_summary, longer_skewed_rows)
    update_condition_pair_ids_in_summary(direction_balanced_summary, direction_balanced_rows)
    update_condition_pair_ids_in_summary(shorter_summary, shorter_skewed_rows)

    family_stratum_counts = Counter(family.family_stratum for family in direction_families)
    summary = {
        "study_name": "Cardinality Shortcut Study",
        "step_name": "Construct ordinary and direction-mixture preference pairs",
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "split": str(args.split),
        "generator_checkpoints": generator_checkpoints,
        "pair_construction_policy": {
            "min_delta_f1": float(args.min_delta_f1),
            "min_response_f1": float(args.min_response_f1),
            "max_pairs_per_question": int(args.max_pairs_per_question),
            "max_response_entities": (
                int(args.max_response_entities) if args.max_response_entities is not None else None
            ),
            "max_semantic_set_edit_distance": (
                int(args.max_semantic_set_edit_distance)
                if args.max_semantic_set_edit_distance is not None
                else None
            ),
            "max_entity_gap": (
                int(args.max_entity_gap) if args.max_entity_gap is not None else None
            ),
            "max_entity_ratio": (
                float(args.max_entity_ratio) if args.max_entity_ratio is not None else None
            ),
            "response_dedupe_mode": str(args.response_dedupe_mode),
            "f1_margin_bin_edges": f1_margin_edges,
            "edit_size_bin_edges": edit_size_edges,
            "token_gap_bin_edges": token_gap_edges,
            "seed": int(args.seed),
        },
        "ordinary_pair_audit": dict(sorted(ordinary_pair_audit.items())),
        "coverage": {
            "question_group_count_in_input": len(rows_by_group),
            "question_group_count_with_any_eligible_pairs": eligible_group_count,
            "question_group_count_with_dual_direction_families": len(direction_families),
            "question_group_count_lost_before_dual_direction_matching": eligible_group_count - len(direction_families),
        },
        "direction_family_summary": {
            "family_count": len(direction_families),
            "family_stratum_counts": dict(sorted(family_stratum_counts.items())),
            "same_f1_margin_bin_count": sum(1 for family in direction_families if family.same_f1_margin_bin),
            "same_edit_size_bin_count": sum(1 for family in direction_families if family.same_edit_size_bin),
            "delta_f1_abs_diff_distribution": summarize_numeric(
                family.delta_f1_abs_diff for family in direction_families
            ),
            "edit_size_abs_diff_distribution": summarize_numeric(
                family.edit_size_abs_diff for family in direction_families
            ),
            "token_gap_abs_diff_distribution": summarize_numeric(
                family.token_gap_abs_diff for family in direction_families
            ),
            "mixed_edit_penalty_distribution": summarize_numeric(
                family.mixed_edit_penalty for family in direction_families
            ),
        },
        "files": {
            "ordinary_pairs": str(ordinary_pairs_path),
            "equal_cardinality_pairs": str(equal_pairs_path),
            "family_manifest": str(family_manifest_path),
            "longer_skewed_pairs": str(longer_skewed_path),
            "direction_balanced_pairs": str(direction_balanced_path),
            "shorter_skewed_pairs": str(shorter_skewed_path),
            "condition_assignment_manifest": str(condition_manifest_path),
        },
        "ordinary_pair_summary": ordinary_summary,
        "equal_cardinality_pair_summary": equal_summary,
        "direction_mixture_condition_summaries": {
            "longer_skewed": longer_summary,
            "direction_balanced": direction_balanced_summary,
            "shorter_skewed": shorter_summary,
        },
        "direction_mixture_assignment_policy": {
            "one_pair_per_question_group": True,
            "question_group_set_fixed_across_conditions": True,
            "resampled_component": "direction assignment only",
            "conditions": {
                "longer_skewed": longer_skewed_manifest,
                "direction_balanced": direction_balanced_manifest,
                "shorter_skewed": shorter_skewed_manifest,
            },
        },
    }
    write_json(
        condition_manifest_path,
        {
            "longer_skewed": longer_skewed_manifest,
            "direction_balanced": direction_balanced_manifest,
            "shorter_skewed": shorter_skewed_manifest,
        },
    )
    write_json(summary_path, summary)

    print("Saved Cardinality Shortcut Study pair datasets:")
    print(f"  ordinary:            {ordinary_pairs_path}")
    print(f"  equal-cardinality:   {equal_pairs_path}")
    print(f"  family manifest:     {family_manifest_path}")
    print(f"  longer-skewed:       {longer_skewed_path}")
    print(f"  direction-balanced:  {direction_balanced_path}")
    print(f"  shorter-skewed:      {shorter_skewed_path}")
    print(f"  summary:             {summary_path}")
    print(f"  ordinary pair rows:  {len(ordinary_pairs):,}")
    print(f"  dual-direction families: {len(direction_families):,}")


if __name__ == "__main__":
    main()
