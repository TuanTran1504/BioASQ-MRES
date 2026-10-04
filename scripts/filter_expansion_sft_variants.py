#!/usr/bin/env python3
"""Create a conservative, auditable subset of teacher-generated SFT variants.

The teacher verifier is useful but can occasionally accept a narrower answer,
an altered numeric range, a salt/prodrug form, or a descriptive expansion. This
script applies high-precision local risk rules after joining every variant back
to its official aliases. It never deletes input records and makes no API calls.

Outputs:
  validated_variants.jsonl   directly usable by build_expansion_sft_dataset.py
  quarantined_variants.jsonl excluded rows with explicit filter reasons
  review.tsv                 compact human-review view of quarantined rows
  summary.json               counts, rule frequencies, and file hashes
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "Artifacts/expansion_sft_teacher/seed3407-gpt41-full-v1"
DEFAULT_OUTPUT = DEFAULT_RUN / "strict_filter_v1"
FILTER_VERSION = "strict_filter_v1"

APPROX_RE = re.compile(
    r"(?:\b(?:about|approximately|approx\.?|around|roughly|nearly|circa)\b|[~≈])",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
RANGE_RE = re.compile(
    r"(?P<low>\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<high>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|percent(?:age)?|years?|months?|weeks?|days?|hours?|"
    r"mg|g|kg|µg|ug|ml|l|mm|cm|m)?(?=$|[\s,.;:)\]])",
    re.I,
)
SALT_RE = re.compile(
    r"\b(?:hydrochloride|hcl|hydrobromide|mesylate|maleate|fumarate|"
    r"succinate|tartrate|acetate|phosphate|sodium|potassium|calcium|"
    r"magnesium)\b",
    re.I,
)
SPECIES_RE = re.compile(r"\b(?:human|murine|mouse|rat|bovine|porcine)\b", re.I)
SCOPE_RE = re.compile(
    r"\b(?:high[- ]risk|drug[- ]resistant|treatment[- ]resistant|"
    r"medically refractory|refractory|intractable|relapsed|metastatic|"
    r"advanced|late[- ]stage|early[- ]stage|pediatric|paediatric|adult)\b",
    re.I,
)

# These phrases are admissions in an otherwise positive verifier explanation.
# Negated claims such as "not narrower" are handled separately as rationalized
# scope changes rather than by matching the single word "narrower".
ADMITTED_SCOPE_PATTERNS = [
    re.compile(r"\bslightly narrower\b", re.I),
    re.compile(r"\bis (?:a |the )?narrower (?:answer|term|range|concept|form)\b", re.I),
    re.compile(r"\bis broader than\b", re.I),
    re.compile(r"\baccepted alias is (?:a |the )?broader term\b", re.I),
    re.compile(r"\bcandidate is (?:a |the )?(?:subtype|subset)\b", re.I),
    # Strict answer equivalence does not permit a verifier to rescue a changed
    # surface by describing it as merely "more specific". This deliberately
    # sends even potentially defensible cases to human review.
    re.compile(r"\bmore specific\b", re.I),
    re.compile(r"\bmore precise and correct answer\b", re.I),
    re.compile(r"\baccepted alias .*?\bomits?\b", re.I),
]
SCOPE_RATIONALIZATION_RE = re.compile(
    r"\b(?:does not|doesn't|not) (?:make (?:it|the candidate) )?"
    r"(?:narrow|narrower|broaden|broader)|\bnot (?:a )?narrower or broader\b",
    re.I,
)
PRODRUG_RE = re.compile(r"\b(?:prodrug|pro-drug) (?:of|form of)\b", re.I)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
TOKEN_STOP = {"a", "an", "and", "of", "the", "to", "in", "for", "with"}


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def flatten_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        value = clean(value)
        return [value] if value else []
    if isinstance(value, list):
        return [alias for item in value for alias in flatten_aliases(item)]
    return []


def dedupe_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def content_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(value)
        if token.casefold() not in TOKEN_STOP
    }


def introduced(pattern: re.Pattern[str], candidate: str, aliases: list[str]) -> bool:
    return bool(pattern.search(candidate)) and not any(pattern.search(alias) for alias in aliases)


def range_values(value: str) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for match in RANGE_RE.finditer(value):
        low = match.group("low").replace(",", "")
        high = match.group("high").replace(",", "")
        unit = (match.group("unit") or "").casefold()
        if unit in {"percent", "percentage"}:
            unit = "%"
        values.append((low, high, unit))
    return values


def conflicting_range(candidate: str, aliases: list[str]) -> bool:
    candidate_ranges = range_values(candidate)
    if not candidate_ranges:
        return False
    alias_ranges = [item for alias in aliases for item in range_values(alias)]
    if not alias_ranges:
        return False
    # Compare only ranges with the same explicit unit. This avoids treating a
    # legitimate months/years conversion as a changed bound.
    comparable = [
        (candidate_range, alias_range)
        for candidate_range in candidate_ranges
        for alias_range in alias_ranges
        if candidate_range[2] == alias_range[2] and candidate_range[2]
    ]
    return bool(comparable) and not any(left == right for left, right in comparable)


def strict_token_superset(candidate: str, aliases: list[str]) -> bool:
    candidate_tokens = content_tokens(candidate)
    return any(
        alias_tokens and alias_tokens < candidate_tokens
        for alias in aliases
        if (alias_tokens := content_tokens(alias))
    )


def descriptive_expansion(candidate: str, aliases: list[str], relation: str) -> bool:
    if relation == "abbreviation_expansion":
        return False
    candidate_words = TOKEN_RE.findall(candidate)
    longest_alias = max((len(TOKEN_RE.findall(alias)) for alias in aliases), default=0)
    return len(candidate_words) >= 13 and len(candidate_words) > 2 * max(longest_alias, 1) + 4


def filter_reasons(row: dict[str, Any], aliases: list[str]) -> list[str]:
    answer = clean(row.get("answer"))
    relation = clean(row.get("relation_type"))
    verifier_basis = clean(row.get("verifier_basis"))
    route = clean(row.get("validation_route"))
    reasons: list[str] = []

    # Locally proven transformations are retained unless a structural problem
    # is detected. Their validity does not depend on the semantic verifier.
    locally_proven = route.startswith("deterministic_") or route == (
        "explicit_parenthetical_abbreviation_in_snippets"
    )

    if row.get("validated") is not True:
        reasons.append("input_not_validated")
    if not answer:
        reasons.append("empty_answer")
    if any(dedupe_key(answer) == dedupe_key(alias) for alias in aliases):
        reasons.append("duplicates_official_alias")

    if locally_proven:
        return list(dict.fromkeys(reasons))

    if introduced(APPROX_RE, answer, aliases) and NUMBER_RE.search(answer):
        reasons.append("introduced_numeric_approximation")
    if conflicting_range(answer, aliases):
        reasons.append("changed_numeric_range")

    # A salt form and its active/free form are not guaranteed to be strictly
    # interchangeable strings. Single-token element answers such as "sodium"
    # are exempted by requiring at least two words.
    if len(TOKEN_RE.findall(answer)) >= 2 and introduced(SALT_RE, answer, aliases):
        reasons.append("introduced_salt_form")

    normalized_surfaces = " ".join([answer, *aliases]).casefold().replace("-", "")
    if PRODRUG_RE.search(verifier_basis) and "prodrug" not in normalized_surfaces:
        reasons.append("prodrug_active_moiety_change")

    if relation != "abbreviation_expansion":
        if introduced(SPECIES_RE, answer, aliases):
            reasons.append("introduced_species_qualifier")
        if introduced(SCOPE_RE, answer, aliases):
            reasons.append("introduced_scope_qualifier")

    if any(pattern.search(verifier_basis) for pattern in ADMITTED_SCOPE_PATTERNS):
        reasons.append("verifier_admits_scope_or_specificity_change")
    if SCOPE_RATIONALIZATION_RE.search(verifier_basis) and strict_token_superset(answer, aliases):
        reasons.append("verifier_rationalizes_added_scope")
    if descriptive_expansion(answer, aliases, relation):
        reasons.append("descriptive_expansion_not_concise_surface")

    return list(dict.fromkeys(reasons))


def load_training_aliases(input_path: Path, seed: int, dev_ratio: float) -> dict[str, list[str]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    factoids = [row for row in payload.get("questions", []) if row.get("type") == "factoid"]
    by_id = {clean(row.get("id")): row for row in factoids}
    if len(factoids) != 1600 or len(by_id) != 1600 or "" in by_id:
        raise ValueError(f"Expected 1,600 unique factoid questions; found {len(factoids)}/{len(by_id)}")
    all_ids = sorted(by_id)
    dev_count = math.ceil(len(all_ids) * dev_ratio)
    dev_ids = set(random.Random(seed).sample(all_ids, dev_count))
    train_ids = set(all_ids) - dev_ids
    if len(train_ids) != 1440 or len(dev_ids) != 160:
        raise ValueError("Expected the fixed split to contain 1,440 train and 160 dev IDs")
    return {
        qid: list(dict.fromkeys(flatten_aliases(by_id[qid].get("exact_answer"))))
        for qid in train_ids
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument("--validated-variants", type=Path, default=DEFAULT_RUN / "validated_variants.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be between zero and one")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace filter outputs."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aliases_by_id = load_training_aliases(args.input, args.seed, args.dev_ratio)
    input_rows = load_rows(args.validated_variants)
    retained: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    relation_retained: Counter[str] = Counter()
    relation_quarantined: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for line_number, row in enumerate(input_rows, 1):
        qid = clean(row.get("question_id"))
        if qid not in aliases_by_id:
            raise ValueError(f"Input row {line_number} has a non-training or unknown question ID: {qid!r}")
        answer = clean(row.get("answer"))
        key = (qid, dedupe_key(answer))
        if key in seen:
            raise ValueError(f"Duplicate variant in input at row {line_number}: {qid} / {answer!r}")
        seen.add(key)
        aliases = aliases_by_id[qid]
        reasons = filter_reasons(row, aliases)
        if reasons:
            output_row = {
                **row,
                "filter_status": "quarantined",
                "filter_version": FILTER_VERSION,
                "filter_reasons": reasons,
                "official_aliases": aliases,
            }
            quarantined.append(output_row)
            reason_counts.update(reasons)
            relation_quarantined[clean(row.get("relation_type"))] += 1
        else:
            retained.append(
                {
                    **row,
                    "filter_status": "retained",
                    "filter_version": FILTER_VERSION,
                }
            )
            relation_retained[clean(row.get("relation_type"))] += 1

    retained_path = args.output_dir / "validated_variants.jsonl"
    quarantine_path = args.output_dir / "quarantined_variants.jsonl"
    write_jsonl(retained_path, retained)
    write_jsonl(quarantine_path, quarantined)

    review_path = args.output_dir / "review.tsv"
    with review_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("question_id\tanswer\trelation_type\treasons\tofficial_aliases\tverifier_basis\n")
        for row in quarantined:
            fields = [
                clean(row.get("question_id")),
                clean(row.get("answer")),
                clean(row.get("relation_type")),
                ";".join(row.get("filter_reasons", [])),
                " | ".join(row.get("official_aliases", [])),
                clean(row.get("verifier_basis")),
            ]
            handle.write("\t".join(value.replace("\t", " ").replace("\n", " ") for value in fields) + "\n")

    summary = {
        "filter_version": FILTER_VERSION,
        "input_variants": len(input_rows),
        "retained_variants": len(retained),
        "quarantined_variants": len(quarantined),
        "retained_fraction": len(retained) / len(input_rows) if input_rows else 0.0,
        "questions_with_retained_variants": len({row["question_id"] for row in retained}),
        "questions_with_quarantined_variants": len({row["question_id"] for row in quarantined}),
        "reason_counts": dict(sorted(reason_counts.items())),
        "retained_relation_counts": dict(sorted(relation_retained.items())),
        "quarantined_relation_counts": dict(sorted(relation_quarantined.items())),
        "configuration": {
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "api_calls": 0,
            "official_aliases_used_for_filtering_only": True,
        },
        "inputs": {
            "training_data": str(args.input.resolve()),
            "training_data_sha256": sha256_file(args.input),
            "validated_variants": str(args.validated_variants.resolve()),
            "validated_variants_sha256": sha256_file(args.validated_variants),
        },
        "outputs": {
            "validated_variants": str(retained_path.resolve()),
            "validated_variants_sha256": sha256_file(retained_path),
            "quarantined_variants": str(quarantine_path.resolve()),
            "quarantined_variants_sha256": sha256_file(quarantine_path),
            "review_tsv": str(review_path.resolve()),
            "review_tsv_sha256": sha256_file(review_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
