"""Create a transparent pre-audit ledger for literal span DPO pairs.

This tool validates source grounding and exact span relations. It deliberately
does not label a pair as medically correct: annotation-compatible answer spans
need expert review. Instead, it ranks pairs by formal validity and semantic
risk so reviewers can focus on the ambiguous cases first.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cse_dpo.normalize_set_answers import normalize_answer_surface


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[+./'-][A-Za-z0-9]+)*")
RISK_TOKENS = {
    "acute", "activating", "all", "at", "chronic", "familial", "human",
    "mouse", "only", "oral", "partial", "primary", "received", "secondary",
    "treated", "type", "wild", "with", "without",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", normalize_answer_surface(str(text or "")))


def tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(str(text or ""))]


def is_in_context(value: str, context: str) -> bool:
    return normalized(value) in normalized(context)


def strict_relation(chosen: str, rejected: str, direction: str) -> bool:
    chosen_key, rejected_key = normalized(chosen), normalized(rejected)
    if not chosen_key or not rejected_key or chosen_key == rejected_key:
        return False
    if direction == "too_short":
        return rejected_key in chosen_key
    if direction == "too_long":
        return chosen_key in rejected_key
    return False


def extension_tokens(chosen: str, rejected: str) -> list[str]:
    chosen_key = str(chosen).casefold().strip()
    rejected_key = str(rejected).casefold().strip()
    if rejected_key.endswith(chosen_key):
        return tokens(rejected[: len(rejected) - len(chosen)].strip())
    if rejected_key.startswith(chosen_key):
        return tokens(rejected[len(chosen) :].strip())
    chosen_tokens, rejected_tokens = tokens(chosen), tokens(rejected)
    return [token for token in rejected_tokens if token not in chosen_tokens]


def risk_flags(row: dict[str, Any]) -> list[str]:
    chosen = str(row.get("chosen_entity", ""))
    rejected = str(row.get("rejected_entity", ""))
    direction = str(row.get("span_direction", ""))
    flags: list[str] = []
    boundary_tokens = extension_tokens(chosen, rejected) if direction == "too_long" else list(
        set(tokens(chosen)) - set(tokens(rejected))
    )
    if any(token in RISK_TOKENS for token in boundary_tokens):
        flags.append("semantic_modifier_at_boundary")
    if any(token.isdigit() for token in boundary_tokens):
        flags.append("numeric_boundary")
    if "(" in chosen or ")" in chosen or "(" in rejected or ")" in rejected:
        flags.append("abbreviation_or_parenthetical_boundary")
    if len(tokens(chosen)) <= 2 or len(tokens(rejected)) <= 2:
        flags.append("very_short_answer")
    if int(row.get("span_delta_tokens", 0) or 0) > 1:
        flags.append("multi_token_boundary_change")
    return flags


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    chosen = str(row.get("chosen_entity", ""))
    rejected = str(row.get("rejected_entity", ""))
    context = str(row.get("source_context", ""))
    gold_entities = [str(value) for value in row.get("accepted_gold_entities", [])]
    formal_checks = {
        "chosen_in_source_context": is_in_context(chosen, context),
        "rejected_in_source_context": is_in_context(rejected, context),
        "strict_directional_span_relation": strict_relation(chosen, rejected, str(row.get("span_direction", ""))),
        "rejected_does_not_match_any_gold_alias": not any(
            normalized(rejected) == normalized(alias) for alias in gold_entities
        ),
    }
    failed = [name for name, value in formal_checks.items() if not value]
    flags = risk_flags(row)
    if failed:
        disposition = "reject_formal"
        rationale = ";".join(failed)
    elif flags:
        disposition = "manual_review_high_risk"
        rationale = ";".join(flags)
    else:
        disposition = "manual_review_standard"
        rationale = "formal_checks_passed"
    return {
        **row,
        **formal_checks,
        "risk_flags": ";".join(flags),
        "pre_audit_disposition": disposition,
        "pre_audit_rationale": rationale,
        "review_decision": "",
        "review_notes": "",
    }


def main() -> None:
    args = parse_args()
    rows = [audit_row(row) for row in read_jsonl(args.pairs_jsonl)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "span_pair_pre_audit.csv"
    fields = list(rows[0]) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["pre_audit_disposition"] for row in rows)
    summary = {
        "pair_count": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "review_instructions": {
            "approve": "Use only when the rejected span is source-grounded, non-gold, and a plausible wrong boundary for the same answer.",
            "reject": "Use when the contrast changes the entity, is another valid answer, or is not a useful extraction error.",
            "uncertain": "Use when BioASQ annotation specificity or biomedical semantics require adjudication.",
        },
        "output_csv": str(output_csv),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
