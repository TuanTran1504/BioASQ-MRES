"""BioASQ answer parsing and submission formatting.

This module deliberately contains no metric implementation. Exact-answer
metrics are computed only by the official BioASQ Java evaluator through
``src.utility.bioasq_official``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from .data import clean_text
from .eval_types import EvalExample


def normalize_for_match(text: str) -> str:
    """Loose text key for deduplication only; never use it for scoring."""
    text = clean_text(text)
    text = re.sub(r"\[[A-Za-z]{2,3}\]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_bioasq_exact_match(text: str) -> str:
    """Official-compatible key used for data validation, never metric scoring."""
    return clean_text(text).lower()


def parse_tagged_items(text: str, begin_tag: str, end_tag: str) -> List[str]:
    pattern = re.compile(re.escape(begin_tag) + r"(.*?)" + re.escape(end_tag), flags=re.DOTALL)
    values = [clean_text(match.group(1)) for match in pattern.finditer(text or "")]
    return [value for value in values if value]


def parse_prediction_items(text: str, question_type: str) -> List[str]:
    if question_type == "factoid":
        values = parse_tagged_items(text, "[BE]", "[EE]")
    elif question_type == "list":
        values = parse_tagged_items(text, "[BI]", "[EI]")
    else:
        values = []
    if values:
        return values
    cleaned = clean_text(text)
    if not cleaned:
        return []
    if question_type == "list":
        return [item for item in [clean_text(part) for part in re.split(r"\n|;", cleaned)] if item]
    return [cleaned]


def normalize_yesno_prediction(text: str) -> str:
    normalized = clean_text(text).lower()
    if normalized.startswith("yes"):
        return "yes"
    if normalized.startswith("no"):
        return "no"
    return normalized


def exact_answer_groups(example: EvalExample, question_type: str) -> List[List[str]]:
    if example.raw_question is not None:
        exact_answer = example.raw_question.get("exact_answer")
        groups: List[List[str]] = []
        if isinstance(exact_answer, list):
            for item in exact_answer:
                if isinstance(item, list):
                    values = [clean_text(alias) for alias in item if clean_text(alias)]
                    if values:
                        groups.append(values)
                else:
                    value = clean_text(item)
                    if value:
                        groups.append([value])
        if groups:
            return groups
    return [[value] for value in parse_prediction_items(example.gold_output, question_type)]


def match_to_gold_group(candidate: str, gold_group: Sequence[str]) -> bool:
    """Data-audit helper matching the official scorer's case-insensitive surface rule."""
    normalized_candidate = normalize_for_bioasq_exact_match(candidate)
    normalized_gold = {
        normalize_for_bioasq_exact_match(value)
        for value in gold_group
        if clean_text(value)
    }
    return bool(normalized_candidate) and normalized_candidate in normalized_gold


def build_bioasq_prediction_entry(record: Dict[str, Any]) -> Dict[str, Any]:
    question_type = record["question_type"]
    prediction = record["prediction"]
    entry = {
        "id": record["question_id"],
        "type": question_type,
        "body": record["body"],
    }
    if question_type == "summary":
        entry["ideal_answer"] = [prediction] if prediction else []
        return entry
    if question_type == "yesno":
        entry["exact_answer"] = normalize_yesno_prediction(prediction)
        entry["ideal_answer"] = [prediction] if prediction else []
        return entry
    values = parse_prediction_items(prediction, question_type)
    if question_type == "factoid":
        values = values[:5]
    entry["exact_answer"] = [[value] for value in values]
    entry["ideal_answer"] = [prediction] if prediction else []
    return entry
