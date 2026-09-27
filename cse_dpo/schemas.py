from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence


PAIR_TYPE_NEGATIVE_ADDITION = "negative_addition"
PAIR_TYPE_VALID_OMISSION = "valid_omission"
PAIR_TYPE_GOLD_VS_SAMPLED_RESPONSE = "gold_vs_sampled_response"

LABEL_GOLD_MATCHED = "gold_matched"
LABEL_METRIC_NEGATIVE = "metric_negative"
LABEL_GOLD_MISSING = "gold_missing"


@dataclass(frozen=True)
class GoldGroup:
    group_id: int
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]
    match_normalized_aliases: tuple[str, ...]
    canonical_alias: str


@dataclass(frozen=True)
class QuestionExample:
    dataset: str
    question_id: str
    question_type: str
    question_text: str
    instruction: str
    evidence: tuple[str, ...]
    gold_groups: tuple[GoldGroup, ...]
    source_path: str


@dataclass(frozen=True)
class CandidateBankRecord:
    dataset: str
    question_id: str
    sample_id: int
    prompt: str
    question_text: str
    evidence: tuple[str, ...]
    raw_output: str
    generated_token_count: int | None
    generator_checkpoint: str | None
    response_id: str
    source_path: str | None = None
    prompt_instruction: str | None = None


@dataclass(frozen=True)
class ParsedListOutput:
    raw_output: str
    items: tuple[str, ...]
    status: str
    warnings: tuple[str, ...]
    begin_tag_count: int
    end_tag_count: int
    empty_item_count: int
    dropped_placeholder_count: int
    used_fallback_split: bool


@dataclass(frozen=True)
class SemanticListItem:
    surface: str
    normalized: str
    first_index: int
    duplicate_count: int


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    match_type: str
    confidence: float


@dataclass(frozen=True)
class MatchedCandidate:
    surface: str
    normalized: str
    matched_gold_group_id: int | None
    label: str
    match_type: str | None
    match_confidence: float
    first_index: int
    duplicate_count: int


@dataclass(frozen=True)
class ResponseMetrics:
    precision: float
    recall: float
    f1: float
    prediction_count: int
    gold_count: int
    invalid_addition_rate: float
    valid_omission_rate: float


@dataclass(frozen=True)
class MatchedResponse:
    record: CandidateBankRecord
    parsed: ParsedListOutput
    candidates: tuple[MatchedCandidate, ...]
    metrics: ResponseMetrics
    matched_gold_group_ids: tuple[int, ...]
    missing_gold_group_ids: tuple[int, ...]
    has_duplicate_semantic_candidates: bool
    pair_eligible: bool
    pair_exclusion_reason: str | None


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    dataset: str
    question_id: str
    question_text: str
    question_source_path: str
    prompt: str
    chosen: str
    rejected: str
    pair_type: str
    base_response_id: str
    edited_candidate: str
    edited_candidate_normalized: str
    edited_gold_group_id: int | None
    candidate_label: str
    candidate_label_source: str
    positive_source: str | None
    semantic_set_edit_distance: int
    chosen_items: tuple[str, ...]
    rejected_items: tuple[str, ...]
    chosen_precision: float
    chosen_recall: float
    chosen_f1: float
    rejected_precision: float
    rejected_recall: float
    rejected_f1: float
    delta_precision: float
    delta_recall: float
    delta_f1: float
    generator_checkpoint: str | None
    sample_id: int


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
