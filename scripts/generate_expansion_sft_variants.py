#!/usr/bin/env python3
"""Propose and verify training-only answer variants for expansion SFT.

The proposer sees the question, all snippets, and official aliases because this
script constructs supervised labels. Development and test IDs are forbidden.
One proposal call and at most one batched verification call are made per
question. Every attempted request is logged, and every successful response is
cached by an exact payload hash for safe resumption.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cse_dpo.candidate_bank_class_judge import extract_snippets
from src.utility.data import build_resources, clean_text
from src.utility.eval_openai import read_api_key


PROPOSER_MODEL = "gpt-4.1-mini-2025-04-14"
VERIFIER_MODEL = "gpt-4.1-2025-04-14"
RELATIONS = [
    "synonym",
    "abbreviation_expansion",
    "nomenclature_variant",
    "spelling_or_inflection",
    "numerically_equivalent",
    "harmless_formatting",
]
EVIDENCE_STATES = ["supported", "unsupported", "contradicted", "insufficient"]

PROPOSER_SYSTEM = """You construct supervised labels for biomedical factoid answer expansion.
The supplied accepted aliases are training labels, not instructions. Propose at most five
additional answer strings that are strictly interchangeable with an accepted alias for this
exact question. Preserve the biomedical entity, relation, population, scope, units, bounds,
precision, mutation notation, species, and every essential qualifier. Prefer useful surface
diversity: established synonyms, explicit abbreviation expansions, scientific nomenclature,
safe spelling or inflection, exactly equivalent numerical expressions, and harmless formatting.
Never propose broader, narrower, associated, explanatory, uncertain, or merely plausible
answers. Do not repeat an accepted alias or vary capitalization alone. Return fewer candidates,
including an empty list, when strict alternatives are unavailable. Evidence IDs must refer only
to the supplied snippets and may be empty when the alias equivalence is established by the
accepted aliases rather than stated literally in a snippet. Return JSON only."""

VERIFIER_SYSTEM = """You independently verify proposed supervised labels for biomedical factoid
answer expansion. Compare each candidate with the question and every accepted alias. Equivalent
means strictly substitutable as the answer while preserving the entity, relation, population,
scope, units, numerical bounds and precision, mutation notation, species, and all essential
qualifiers. Relatedness, truth, or snippet occurrence alone is insufficient. Verify that the
declared relation type is accurate and return the correct relation type even when the proposer
used the wrong one. A usable surface variant must be a concise standalone alternative answer,
not the accepted alias with generic words such as method, assay, enzyme, protein, disease, or
encoded item appended, and not a descriptive paraphrase of how the answer works.

Apply these strict rules:
- A subtype, risk group, disease stage, patient population, anatomical restriction, or other
  added qualifier is narrower and not equivalent, even when it is the most common clinical use.
  For example, high-risk neuroblastoma is not equivalent to neuroblastoma, and type 2 diabetes
  is not equivalent to diabetes mellitus.
- Removing a qualifier is broader and not equivalent.
- Harmless formatting changes only typography, punctuation, spacing, or symbol encoding. Adding
  words such as method, assay, enzyme, isoenzyme, encoded, or proteins is not formatting.
- A word-form number such as twenty-five for 25 is numerically_equivalent, not spelling.
- A decimal proportion without an explicit percent or proportion marker is not a safe standalone
  replacement for a percentage answer.
- A nomenclature variant must be an established name for the same entity. A component list,
  mechanistic description, or narrower subtype is not nomenclature.

Use insufficient evidence when snippets cannot establish support, but judge semantic equivalence
against the accepted aliases. Reject when uncertain. Return one decision for every candidate,
in the supplied order, as JSON only."""


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
PROPOSAL_SCHEMA = object_schema(
    {
        "candidates": {
            "type": "array",
            "maxItems": 5,
            "items": object_schema(
                {
                    "answer": STRING,
                    "relation_type": {"type": "string", "enum": RELATIONS},
                    "evidence_ids": {"type": "array", "items": STRING},
                    "basis": STRING,
                }
            ),
        }
    }
)
VERIFICATION_SCHEMA = object_schema(
    {
        "decisions": {
            "type": "array",
            "maxItems": 5,
            "items": object_schema(
                {
                    "candidate_index": {"type": "integer"},
                    "equivalent": {"type": "boolean"},
                    "relation_valid": {"type": "boolean"},
                    "standalone_surface_valid": {"type": "boolean"},
                    "relation_type": {"type": "string", "enum": RELATIONS},
                    "rejection_type": {
                        "type": "string",
                        "enum": [
                            "none",
                            "broader",
                            "narrower",
                            "part_whole",
                            "missing_qualifier",
                            "extra_qualifier",
                            "wrong_value",
                            "generic_addition",
                            "descriptive_paraphrase",
                            "unnatural_surface",
                            "uncertain",
                            "other",
                        ],
                    },
                    "evidence_support": {"type": "string", "enum": EVIDENCE_STATES},
                    "evidence_ids": {"type": "array", "items": STRING},
                    "basis": STRING,
                }
            ),
        }
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def flatten_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        value = clean_text(value)
        return [value] if value else []
    if isinstance(value, list):
        return [alias for item in value for alias in flatten_aliases(item)]
    return []


def dedupe_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def terminal_format_key(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"(?:™|®|\(\s*tm\s*\)|\(\s*r\s*\))", "", value, flags=re.I)
    value = value.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"[\s.,;:]+$", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def routing_format_key(value: str) -> str:
    """Stable v2 key: changing request routing would invalidate verifier caches."""
    value = html.unescape(value).replace("™", " TM").replace("®", " R")
    value = re.sub(r"[\s.,;:]+$", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "and", "percent", "percentage", "more", "greater", "less", "fewer",
    "than", "at", "least", "most", "no",
}


def numeric_only_surface(value: str) -> bool:
    value = html.unescape(value).casefold().replace("%", " percent ")
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[<>]=?", value)
    if not tokens or not any(token[0].isdigit() or token in NUMBER_WORDS - {"and"} for token in tokens):
        return False
    return all(
        token[0].isdigit() or token in NUMBER_WORDS or token in {">", "<", ">=", "<="}
        for token in tokens
    )


def embedded_parenthetical_abbreviation(candidate: str, aliases: list[str]) -> bool:
    for alias in aliases:
        if re.search(re.escape(alias) + r"\s*\(\s*[A-Za-z0-9-]{2,15}\s*\)", candidate, re.I):
            return True
        if re.search(r"^[A-Za-z0-9-]{2,15}\s*\(\s*" + re.escape(alias) + r"\s*\)$", candidate, re.I):
            return True
    return False


def orthographic_variant(candidate: str, aliases: list[str]) -> bool:
    candidate_key = re.sub(r"[^a-z0-9]+", "", candidate.casefold())
    if not candidate_key:
        return False
    return any(
        SequenceMatcher(
            None,
            candidate_key,
            re.sub(r"[^a-z0-9]+", "", alias.casefold()),
        ).ratio() >= 0.9
        for alias in aliases
    )


def numeric_expression_key(value: str) -> str | None:
    value = html.unescape(value).casefold().strip()
    replacements = [
        (r"\b(?:more|greater)\s+than\b", ">"),
        (r"\b(?:less|fewer)\s+than\b", "<"),
        (r"\b(?:at\s+least|no\s+less\s+than)\b", ">="),
        (r"\b(?:at\s+most|no\s+more\s+than)\b", "<="),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    value = value.replace("≥", ">=").replace("≤", "<=")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"\s+", "", value)
    return value if re.search(r"\d", value) else None


def explicit_parenthetical_pair(left: str, right: str, snippets: list[dict[str, str]]) -> bool:
    left_pattern = re.escape(left).replace(r"\ ", r"\s+")
    right_pattern = re.escape(right).replace(r"\ ", r"\s+")
    patterns = [
        re.compile(left_pattern + r"\s*\(\s*" + right_pattern + r"\s*\)", re.I),
        re.compile(right_pattern + r"\s*\(\s*" + left_pattern + r"\s*\)", re.I),
    ]
    return any(pattern.search(snippet["text"]) for pattern in patterns for snippet in snippets)


def verified_relation_or_none(
    candidate: str,
    aliases: list[str],
    snippets: list[dict[str, str]],
    verifier_relation: str,
) -> tuple[str | None, str]:
    """Enforce mechanically checkable boundaries for commonly abused labels."""
    if embedded_parenthetical_abbreviation(candidate, aliases) or any(
        explicit_parenthetical_pair(candidate, alias, snippets) for alias in aliases
    ):
        return "abbreviation_expansion", "deterministic_relation_abbreviation"
    if numeric_only_surface(candidate) and any(numeric_only_surface(alias) for alias in aliases):
        if any(terminal_format_key(candidate) == terminal_format_key(alias) for alias in aliases):
            return "harmless_formatting", "deterministic_relation_numeric_formatting"
        return "numerically_equivalent", "deterministic_relation_numeric_equivalence"
    if verifier_relation == "harmless_formatting":
        if any(terminal_format_key(candidate) == terminal_format_key(alias) for alias in aliases):
            return verifier_relation, "deterministic_relation_formatting"
        return None, "unproven_harmless_formatting_relation"
    if verifier_relation == "spelling_or_inflection":
        if orthographic_variant(candidate, aliases):
            return verifier_relation, "deterministic_relation_spelling_or_inflection"
        return None, "unproven_spelling_or_inflection_relation"
    return verifier_relation, "verifier_relation"


def local_decision(
    candidate: dict[str, Any], aliases: list[str], snippets: list[dict[str, str]]
) -> tuple[str, str]:
    answer = candidate["answer"]
    relation = candidate["relation_type"]
    if any(dedupe_key(answer) == dedupe_key(alias) for alias in aliases):
        return "reject", "duplicates_official_alias"
    if relation == "harmless_formatting" and any(
        routing_format_key(answer) == routing_format_key(alias) for alias in aliases
    ):
        return "accept", "deterministic_terminal_or_trademark_formatting"
    candidate_numeric = numeric_expression_key(answer)
    if relation == "numerically_equivalent" and candidate_numeric and any(
        candidate_numeric == numeric_expression_key(alias) for alias in aliases
    ):
        return "accept", "deterministic_numeric_expression"
    if relation == "abbreviation_expansion" and any(
        explicit_parenthetical_pair(answer, alias, snippets) for alias in aliases
    ):
        return "accept", "explicit_parenthetical_abbreviation_in_snippets"
    return "verify", "requires_semantic_verification"


def question_snippets(question: dict[str, Any]) -> list[dict[str, str]]:
    resources = build_resources(
        question,
        max_resources=0,
        max_resource_chars=0,
        question_text=clean_text(question.get("body")),
        resource_granularity="document",
        resource_selection="first",
    )
    snippets = extract_snippets(resources)
    if not snippets:
        raise ValueError(f"{question.get('id')}: no usable snippets")
    return snippets


def public_context(question: dict[str, Any], aliases: list[str], snippets: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "question_id": clean_text(question.get("id")),
        "question": clean_text(question.get("body")),
        "accepted_aliases": aliases,
        "snippets": [
            {"id": snippet["snippet_id"], "text": snippet["text"]}
            for snippet in snippets
        ],
    }


def proposal_payload(
    question: dict[str, Any], aliases: list[str], snippets: list[dict[str, str]], model: str
) -> dict[str, Any]:
    user = {
        **public_context(question, aliases, snippets),
        "maximum_additional_candidates": 5,
        "allowed_relation_types": RELATIONS,
    }
    return {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": PROPOSER_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "expansion_training_variants", "strict": True, "schema": PROPOSAL_SCHEMA},
        },
    }


def verification_payload(
    question: dict[str, Any],
    aliases: list[str],
    snippets: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    user = {
        **public_context(question, aliases, snippets),
        "candidates": [
            {
                "candidate_index": index,
                "answer": row["answer"],
                "declared_relation_type": row["relation_type"],
                "proposer_evidence_ids": row["evidence_ids"],
                "proposer_basis": row["basis"],
            }
            for index, row in enumerate(candidates)
        ],
    }
    return {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1800,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "verified_expansion_variants", "strict": True, "schema": VERIFICATION_SCHEMA},
        },
    }


def validate_proposal(value: Any, snippet_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("Proposal response has unexpected fields")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) > 5:
        raise ValueError("Proposal must contain at most five candidates")
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"answer", "relation_type", "evidence_ids", "basis"}:
            raise ValueError("Proposal candidate has unexpected fields")
        answer = clean_text(row["answer"])
        relation = row["relation_type"]
        evidence_ids = row["evidence_ids"]
        basis = clean_text(row["basis"])
        if not answer or len(answer) > 500 or relation not in RELATIONS or not basis:
            raise ValueError("Proposal candidate has invalid content")
        if not isinstance(evidence_ids, list) or any(not isinstance(item, str) for item in evidence_ids):
            raise ValueError("Proposal evidence_ids must be a list of strings")
        valid_evidence_ids = [str(item) for item in evidence_ids if str(item) in snippet_ids]
        invalid_evidence_ids = [str(item) for item in evidence_ids if str(item) not in snippet_ids]
        key = dedupe_key(answer)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "answer": answer,
                "relation_type": relation,
                "evidence_ids": valid_evidence_ids,
                "invalid_evidence_ids": invalid_evidence_ids,
                "basis": basis,
            }
        )
    return cleaned


def validate_verification(
    value: Any, candidates: list[dict[str, Any]], snippet_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"decisions"}:
        raise ValueError("Verification response has unexpected fields")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(candidates):
        raise ValueError("Verifier must return one decision per candidate")
    expected_fields = {
        "candidate_index",
        "equivalent",
        "relation_valid",
        "standalone_surface_valid",
        "relation_type",
        "rejection_type",
        "evidence_support",
        "evidence_ids",
        "basis",
    }
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict) or set(decision) != expected_fields:
            raise ValueError("Verification decision has unexpected fields")
        if decision["candidate_index"] != index:
            raise ValueError("Verification decisions are not in candidate order")
        if (
            type(decision["equivalent"]) is not bool
            or type(decision["relation_valid"]) is not bool
            or type(decision["standalone_surface_valid"]) is not bool
        ):
            raise ValueError("Verification booleans are invalid")
        if decision["relation_type"] not in RELATIONS or decision["evidence_support"] not in EVIDENCE_STATES:
            raise ValueError("Verification labels are invalid")
        if not isinstance(decision["evidence_ids"], list) or any(
            not isinstance(item, str) for item in decision["evidence_ids"]
        ):
            raise ValueError("Verification evidence_ids must be a list of strings")
        raw_evidence_ids = [str(item) for item in decision["evidence_ids"]]
        decision["evidence_ids"] = [item for item in raw_evidence_ids if item in snippet_ids]
        decision["invalid_evidence_ids"] = [item for item in raw_evidence_ids if item not in snippet_ids]
        if not clean_text(decision["basis"]):
            raise ValueError("Verification basis is empty")
        if decision["rejection_type"] not in {
            "none", "broader", "narrower", "part_whole", "missing_qualifier",
            "extra_qualifier", "wrong_value", "generic_addition",
            "descriptive_paraphrase", "unnatural_surface", "uncertain", "other",
        }:
            raise ValueError("Verification rejection type is invalid")
        if decision["equivalent"] and decision["standalone_surface_valid"]:
            if decision["rejection_type"] != "none":
                raise ValueError("Accepted semantic surfaces must use rejection_type=none")
    return decisions


class CachedOpenAIClient:
    def __init__(
        self,
        output_dir: Path,
        *,
        api_key_file: Path,
        allow_api: bool,
        max_new_calls: int,
        delay: float,
        previous_cache_dir: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.api_key_file = api_key_file
        self.allow_api = allow_api
        self.max_new_calls = max_new_calls
        self.delay = delay
        self.previous_cache_dir = previous_cache_dir
        self.new_calls = 0
        self.cache_hits = 0

    def call(self, stage: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        key = digest({"stage": stage, "payload": payload})
        cache_path = self.output_dir / "cache" / stage / f"{key}.json"
        attempt_path = self.output_dir / "attempts" / stage / f"{key}.json"
        previous_path = (
            self.previous_cache_dir / "cache" / stage / f"{key}.json"
            if self.previous_cache_dir is not None
            else None
        )
        existing_path = cache_path if cache_path.exists() else previous_path if previous_path and previous_path.exists() else None
        if existing_path is not None:
            envelope = json.loads(existing_path.read_text(encoding="utf-8"))
            if envelope.get("request") != payload:
                raise ValueError("Cache payload mismatch")
            if existing_path != cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.cache_hits += 1
            return envelope["response"], "cache" if existing_path == cache_path else "previous_cache"
        if not self.allow_api:
            raise RuntimeError(f"Cache miss for {stage} while API access is disabled")
        if self.new_calls >= self.max_new_calls:
            raise RuntimeError("Hard API request limit reached")

        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        attempt = {
            "stage": stage,
            "request_sha256": key,
            "request": payload,
            "attempted_at": utc_now(),
            "status": "pending",
        }
        attempt_path.write_text(json.dumps(attempt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if self.delay:
            time.sleep(self.delay)
        self.new_calls += 1
        try:
            import requests

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {read_api_key(self.api_key_file)}"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            raw = response.json()
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise ValueError("API response was incomplete or refused")
            value = json.loads(choice["message"]["content"])
            envelope = {
                "stage": stage,
                "request_sha256": key,
                "request": payload,
                "response": value,
                "response_metadata": {
                    "id": raw.get("id"),
                    "model": raw.get("model"),
                    "created": raw.get("created"),
                    "usage": raw.get("usage"),
                },
                "completed_at": utc_now(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            attempt["status"] = "complete"
            attempt["response_cache"] = str(cache_path)
            attempt_path.write_text(json.dumps(attempt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return value, "api"
        except Exception as exc:
            attempt["status"] = "error"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempt_path.write_text(json.dumps(attempt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def aggregate(output_dir: Path, expected_ids: list[str]) -> dict[str, Any]:
    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    route_counts: Counter[str] = Counter()
    for qid in expected_ids:
        path = output_dir / "question_results" / f"{qid}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        completed_ids.append(qid)
        validated.extend(row["validated_variants"])
        rejected.extend(row["rejected_variants"])
        route_counts.update(item["validation_route"] for item in row["validated_variants"])
        route_counts.update(item["rejection_reason"] for item in row["rejected_variants"])
    write_jsonl(output_dir / "validated_variants.jsonl", validated)
    write_jsonl(output_dir / "rejected_variants.jsonl", rejected)
    summary = {
        "expected_questions": len(expected_ids),
        "completed_questions": len(completed_ids),
        "validated_variants": len(validated),
        "rejected_variants": len(rejected),
        "questions_with_validated_variants": len({row["question_id"] for row in validated}),
        "route_counts": dict(sorted(route_counts.items())),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "run"], default="validate")
    parser.add_argument("--input", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=ROOT / "data/BioASQ_expansion_sft/seed3407_gold_aliases_v1/train_question_ids.json",
    )
    parser.add_argument(
        "--dev-ids",
        type=Path,
        default=ROOT / "data/BioASQ_expansion_sft/seed3407_gold_aliases_v1/dev_question_ids.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Artifacts/expansion_sft_teacher/seed3407-gpt41-v4",
    )
    parser.add_argument("--previous-cache-dir", type=Path)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "open_ai_api.txt")
    parser.add_argument("--proposer-model", default=PROPOSER_MODEL)
    parser.add_argument("--verifier-model", default=VERIFIER_MODEL)
    parser.add_argument("--selection", choices=["all", "single_alias_first"], default="all")
    parser.add_argument("--question-limit", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    factoids = {clean_text(q.get("id")): q for q in raw.get("questions", []) if q.get("type") == "factoid"}
    train_ids = json.loads(args.train_ids.read_text(encoding="utf-8"))
    dev_ids = set(json.loads(args.dev_ids.read_text(encoding="utf-8")))
    if len(train_ids) != 1440 or len(set(train_ids)) != 1440:
        raise ValueError("Expected exactly 1,440 unique training IDs")
    if set(train_ids) & dev_ids or any(qid not in factoids for qid in train_ids):
        raise ValueError("Training IDs overlap dev or are missing from the raw corpus")

    aliases_by_id = {
        qid: list(dict.fromkeys(flatten_aliases(factoids[qid].get("exact_answer"))))
        for qid in train_ids
    }
    if any(not aliases for aliases in aliases_by_id.values()):
        raise ValueError("Every training question must have at least one accepted alias")
    selected_ids = sorted(train_ids)
    if args.selection == "single_alias_first":
        selected_ids.sort(key=lambda qid: (len(aliases_by_id[qid]) != 1, digest([args.seed, qid])))
    if args.question_limit:
        if args.question_limit < 1:
            raise ValueError("question-limit must be nonnegative")
        selected_ids = selected_ids[: args.question_limit]

    preflight = {
        "mode": args.mode,
        "proposer_model": args.proposer_model,
        "verifier_model": args.verifier_model,
        "available_training_questions": len(train_ids),
        "selected_questions": len(selected_ids),
        "selection": args.selection,
        "maximum_proposals_per_question": 5,
        "maximum_api_calls": len(selected_ids) * 2,
        "hard_new_call_limit": args.max_new_calls,
        "gold_usage": "Accepted aliases are included only because this pipeline constructs training labels. Dev and test IDs are forbidden.",
        "cache_policy": "Exact request/response envelopes are content-addressed; every attempted request is logged separately.",
        "previous_cache_dir": str(args.previous_cache_dir) if args.previous_cache_dir else None,
    }
    write_json(args.output_dir / "preflight.json", preflight)
    if args.mode == "validate":
        print(json.dumps(preflight, indent=2))
        return
    if not args.allow_api:
        raise ValueError("Run mode requires --allow-api")
    if args.max_new_calls < 0:
        raise ValueError("max-new-calls must be nonnegative")

    client = CachedOpenAIClient(
        args.output_dir,
        api_key_file=args.api_key_file,
        allow_api=args.allow_api,
        max_new_calls=args.max_new_calls,
        delay=args.delay,
        previous_cache_dir=args.previous_cache_dir,
    )
    status = {"status": "running", **preflight, "started_at": utc_now()}
    write_json(args.output_dir / "status.json", status)

    try:
        for ordinal, qid in enumerate(selected_ids, 1):
            result_path = args.output_dir / "question_results" / f"{qid}.json"
            if result_path.exists():
                continue
            question = factoids[qid]
            aliases = aliases_by_id[qid]
            snippets = question_snippets(question)
            snippet_ids = {snippet["snippet_id"] for snippet in snippets}

            raw_proposal, proposal_origin = client.call(
                "proposal", proposal_payload(question, aliases, snippets, args.proposer_model)
            )
            proposals = validate_proposal(raw_proposal, snippet_ids)

            validated: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            needs_verification: list[dict[str, Any]] = []
            for proposal in proposals:
                decision, reason = local_decision(proposal, aliases, snippets)
                base = {
                    "question_id": qid,
                    "answer": proposal["answer"],
                    "relation_type": proposal["relation_type"],
                    "proposer_model": args.proposer_model,
                    "proposer_evidence_ids": proposal["evidence_ids"],
                    "proposer_invalid_evidence_ids": proposal.get("invalid_evidence_ids", []),
                    "proposer_basis": proposal["basis"],
                }
                if decision == "accept":
                    validated.append(
                        {
                            **base,
                            "validated": True,
                            "source": "teacher_generated",
                            "validation_route": reason,
                            "verifier_model": None,
                        }
                    )
                elif decision == "reject":
                    rejected.append({**base, "validated": False, "rejection_reason": reason})
                else:
                    needs_verification.append(proposal)

            verification_origin = None
            if needs_verification:
                raw_verification, verification_origin = client.call(
                    "verification",
                    verification_payload(
                        question,
                        aliases,
                        snippets,
                        needs_verification,
                        args.verifier_model,
                    ),
                )
                decisions = validate_verification(raw_verification, needs_verification, snippet_ids)
                for proposal, decision in zip(needs_verification, decisions):
                    base = {
                        "question_id": qid,
                        "answer": proposal["answer"],
                        "relation_type": proposal["relation_type"],
                        "proposer_model": args.proposer_model,
                        "proposer_evidence_ids": proposal["evidence_ids"],
                        "proposer_invalid_evidence_ids": proposal.get("invalid_evidence_ids", []),
                        "proposer_basis": proposal["basis"],
                        "verifier_model": args.verifier_model,
                        "verifier_relation_type": decision["relation_type"],
                        "verifier_declared_relation_valid": decision["relation_valid"],
                        "verifier_standalone_surface_valid": decision["standalone_surface_valid"],
                        "verifier_rejection_type": decision["rejection_type"],
                        "verifier_evidence_support": decision["evidence_support"],
                        "verifier_evidence_ids": decision["evidence_ids"],
                        "verifier_invalid_evidence_ids": decision.get("invalid_evidence_ids", []),
                        "verifier_basis": clean_text(decision["basis"]),
                    }
                    corrected_relation, relation_route = verified_relation_or_none(
                        proposal["answer"], aliases, snippets, decision["relation_type"]
                    )
                    valid = (
                        decision["equivalent"]
                        and decision["standalone_surface_valid"]
                        and decision["rejection_type"] == "none"
                        and decision["evidence_support"] != "contradicted"
                        and not (
                            decision["evidence_support"] == "supported"
                            and decision.get("invalid_evidence_ids")
                            and not decision["evidence_ids"]
                        )
                        and corrected_relation is not None
                    )
                    if valid:
                        validated.append(
                            {
                                **base,
                                "relation_type": corrected_relation,
                                "validated": True,
                                "source": "teacher_generated",
                                "validation_route": f"gpt41_semantic_verification+{relation_route}",
                            }
                        )
                    else:
                        rejected.append(
                            {
                                **base,
                                "validated": False,
                                "rejection_reason": (
                                    relation_route if corrected_relation is None
                                    else "semantic_or_relation_verification_failed"
                                ),
                            }
                        )

            write_json(
                result_path,
                {
                    "question_id": qid,
                    "proposal_origin": proposal_origin,
                    "verification_origin": verification_origin,
                    "official_alias_count": len(aliases),
                    "proposed_variants": len(proposals),
                    "validated_variants": validated,
                    "rejected_variants": rejected,
                    "completed_at": utc_now(),
                },
            )
            summary = aggregate(args.output_dir, selected_ids)
            print(
                f"{ordinal}/{len(selected_ids)} | proposed={len(proposals)} "
                f"validated={len(validated)} rejected={len(rejected)} "
                f"new_calls={client.new_calls} cache_hits={client.cache_hits}"
            )

        summary = aggregate(args.output_dir, selected_ids)
        write_json(
            args.output_dir / "status.json",
            {
                **status,
                "status": "complete",
                "completed_at": utc_now(),
                "new_api_calls": client.new_calls,
                "cache_hits": client.cache_hits,
                **summary,
            },
        )
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        summary = aggregate(args.output_dir, selected_ids)
        write_json(
            args.output_dir / "status.json",
            {
                **status,
                "status": "incomplete",
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "new_api_calls": client.new_calls,
                "cache_hits": client.cache_hits,
                **summary,
            },
        )
        raise


if __name__ == "__main__":
    main()
