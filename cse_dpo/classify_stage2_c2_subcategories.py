"""Enrich Stage-2 C3>C2 pairs with semantic and surface-form subcategories."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/class_judgments/"
    "candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130/merged"
)

RELATION_TO_FAMILY = {
    "harmless_formatting": "surface_form",
    "spelling_or_inflection": "surface_form",
    "abbreviation_expansion": "name_equivalence",
    "nomenclature_variant": "name_equivalence",
    "synonym": "name_equivalence",
    "numerically_equivalent": "numeric_equivalence",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def weak_normalize(text: str) -> str:
    """Case-fold and collapse whitespace while preserving punctuation."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def literal_tokens(text: str) -> list[str]:
    # Punctuation remains attached to its token so hyphenated and unhyphenated
    # biomedical names are not silently treated as identical.
    return weak_normalize(text).split()


def compare_to_alias(candidate: str, alias: str) -> dict[str, Any]:
    candidate_norm = weak_normalize(candidate)
    alias_norm = weak_normalize(alias)
    candidate_tokens = literal_tokens(candidate)
    alias_tokens = literal_tokens(alias)

    if candidate_norm == alias_norm:
        relation = "same_weak_surface"
        priority = 5
    elif alias_norm and alias_norm in candidate_norm:
        relation = "gold_inside_candidate"
        priority = 4
    elif candidate_norm and candidate_norm in alias_norm:
        relation = "candidate_inside_gold"
        priority = 4
    elif set(candidate_tokens) & set(alias_tokens):
        relation = "shared_literal_tokens"
        priority = 3
    else:
        relation = "zero_literal_token_overlap"
        priority = 1

    if len(candidate_tokens) > len(alias_tokens):
        length_direction = "candidate_longer"
    elif len(candidate_tokens) < len(alias_tokens):
        length_direction = "candidate_shorter"
    elif len(candidate_norm) > len(alias_norm):
        length_direction = "candidate_longer_same_token_count"
    elif len(candidate_norm) < len(alias_norm):
        length_direction = "candidate_shorter_same_token_count"
    else:
        length_direction = "same_length"

    return {
        "matched_gold_alias": alias,
        "c2_span_relation": relation,
        "c2_length_direction": length_direction,
        "c2_candidate_token_count": len(candidate_tokens),
        "c2_gold_token_count": len(alias_tokens),
        "_match_score": (priority, SequenceMatcher(None, candidate_norm, alias_norm).ratio()),
    }


def best_alias_comparison(candidate: str, aliases: list[str]) -> dict[str, Any]:
    comparisons = [compare_to_alias(candidate, alias) for alias in aliases]
    best = max(comparisons, key=lambda item: item["_match_score"])
    best.pop("_match_score")
    return best


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def classify(judgments_path: Path, stage2_path: Path, output_dir: Path) -> dict[str, Any]:
    judgments = read_jsonl(judgments_path)
    stage2_rows = read_jsonl(stage2_path)

    by_response: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_candidate: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        if row.get("class") != "C2":
            continue
        by_response[(row["question_id"], row["source_model"], row["response_id"])] = row
        by_candidate[(row["question_id"], row["source_model"], row["candidate"])].append(row)

    enriched: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in stage2_rows:
        key = (pair["question_id"], pair["rejected_source_model"], pair["rejected_response_id"])
        judgment = by_response.get(key)
        if judgment is None:
            fallback = by_candidate.get(
                (pair["question_id"], pair["rejected_source_model"], pair["rejected_candidate"]), []
            )
            judgment = fallback[0] if fallback else None
        if judgment is None:
            missing.append(pair["pair_id"])
            continue

        relation = judgment["relation_type"]
        family = RELATION_TO_FAMILY.get(relation, "other_equivalence")
        comparison = best_alias_comparison(pair["rejected_candidate"], pair["gold_aliases"])
        prompt_norm = weak_normalize(pair["prompt"])
        candidate_norm = weak_normalize(pair["rejected_candidate"])
        enriched.append(
            {
                **pair,
                "c2_family": family,
                "c2_subcategory": relation,
                **comparison,
                "c2_extractive_from_prompt": bool(candidate_norm and candidate_norm in prompt_norm),
                "c2_judge_confidence": judgment.get("confidence"),
                "c2_judge_basis": judgment.get("basis"),
                "c2_judge_evidence_ids": judgment.get("evidence_ids", []),
                "c2_judge_rubric_version": judgment.get("judge_rubric_version"),
            }
        )

    if missing:
        raise ValueError(f"Could not match {len(missing)} Stage-2 pairs to C2 judgments: {missing[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_jsonl = output_dir / "dpo_stage2_format_alignment_c2_categorized.jsonl"
    combined_csv = output_dir / "dpo_stage2_format_alignment_c2_categorized.csv"
    write_jsonl(combined_jsonl, enriched)
    write_csv(combined_csv, enriched)

    by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_subcategory[row["c2_subcategory"]].append(row)
    category_dir = output_dir / "by_c2_subcategory"
    for subcategory, rows in sorted(by_subcategory.items()):
        write_jsonl(category_dir / f"{safe_filename(subcategory)}.jsonl", rows)

    summary = {
        "source_judgments": str(judgments_path.resolve()),
        "source_stage2_pairs": str(stage2_path.resolve()),
        "pair_count": len(enriched),
        "question_count": len({row["question_id"] for row in enriched}),
        "family_counts": dict(sorted(Counter(row["c2_family"] for row in enriched).items())),
        "subcategory_counts": dict(sorted(Counter(row["c2_subcategory"] for row in enriched).items())),
        "span_relation_counts": dict(sorted(Counter(row["c2_span_relation"] for row in enriched).items())),
        "length_direction_counts": dict(sorted(Counter(row["c2_length_direction"] for row in enriched).items())),
        "source_model_counts": dict(sorted(Counter(row["rejected_source_model"] for row in enriched).items())),
        "extractive_from_prompt_counts": {
            str(key).lower(): value
            for key, value in sorted(Counter(row["c2_extractive_from_prompt"] for row in enriched).items())
        },
        "combined_jsonl": str(combined_jsonl.resolve()),
        "combined_csv": str(combined_csv.resolve()),
        "category_dir": str(category_dir.resolve()),
        "normalization": "NFKC + casefold + whitespace collapse; punctuation preserved",
    }
    (output_dir / "c2_subcategory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments",
        type=Path,
        default=DEFAULT_MERGED_ROOT / "candidate_class_judgments.jsonl",
    )
    parser.add_argument(
        "--stage2-pairs",
        type=Path,
        default=DEFAULT_MERGED_ROOT / "staged_curriculum_pairs/dpo_stage2_format_alignment_all_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MERGED_ROOT / "staged_curriculum_pairs/c2_categorized",
    )
    args = parser.parse_args()
    print(json.dumps(classify(args.judgments, args.stage2_pairs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
