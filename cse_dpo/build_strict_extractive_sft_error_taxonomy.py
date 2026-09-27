#!/usr/bin/env python3
"""Create a reproducible error taxonomy for the strict-extractive SFT run.

The reviewed Cal-DPO full-test taxonomy supplies labels for identical SFT
failures.  The SFT-only failure is documented explicitly below rather than
silently inheriting a category from another model.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT
    / "Artifacts/Factoid_SFT/evaluations/factoid_test_qwen25_05b_full_test_20260907_210258"
)
SFT_DIR = EVALUATION_ROOT / "evidence_per_alias_sft_dropout_strict_extractive"
SEED_DIR = EVALUATION_ROOT / "error_analysis_cal_dpo_full_test_v1"
OUTPUT_DIR = EVALUATION_ROOT / "error_analysis_strict_extractive_sft_full_test_v1"

CATEGORY_FILES = {
    "unsupported_gold_absent_from_provided_resources": "unsupported_gold_absent_from_provided_resources.csv",
    "numerical_or_quantitative_selection_error": "numerical_or_quantitative_selection_error.csv",
    "strict_surface_or_alias_mismatch": "strict_surface_or_alias_mismatch.csv",
    "span_or_granularity_mismatch": "span_or_granularity_mismatch.csv",
    "wrong_information_or_question_target": "wrong_information_or_question_target.csv",
    "long_form_or_ambiguous_annotation": "long_form_or_ambiguous_annotation.csv",
}

CATEGORY_DESCRIPTIONS = {
    "unsupported_gold_absent_from_provided_resources": (
        "No normalized gold alias occurs in the supplied resources. This is an "
        "evidence-availability limitation, not an extractive selection failure."
    ),
    "numerical_or_quantitative_selection_error": (
        "The prediction selects an incorrect number, range, unit, threshold, or denominator."
    ),
    "strict_surface_or_alias_mismatch": (
        "The answer is semantically close but fails official matching because of spelling, "
        "punctuation, acronym, translation, or an unlisted alias."
    ),
    "span_or_granularity_mismatch": (
        "The model extracts a related span that is too broad, too narrow, incomplete, "
        "or includes unnecessary context."
    ),
    "wrong_information_or_question_target": (
        "The model selects a different entity, relation, outcome, or question target from "
        "the answer supported by the resources."
    ),
    "long_form_or_ambiguous_annotation": (
        "The official factoid target is sentence-like, multi-part, or otherwise ambiguous; "
        "a short supported extraction can still fail strict scoring."
    ),
}

# This record is wrong only for strict matching. Its short prediction is a
# supported portion of the long, sentence-style official answer.
MANUAL_CATEGORIES = {
    "67f8ea8318b1e36f2e000107": {
        "category": "long_form_or_ambiguous_annotation",
        "confidence": "high",
        "notes": (
            "The prediction 'rapid onset of action' is supported, but the official answers "
            "require a complete formoterol-versus-salmeterol comparison or agonist contrast."
        ),
    }
}

OUTPUT_COLUMNS = [
    "question_id",
    "current_status",
    "error_category",
    "evidence_status",
    "batch",
    "question",
    "prediction",
    "gold_output",
    "mrr",
    "strict_accuracy",
    "resource_count",
    "review_confidence",
    "review_notes",
    "classification_provenance",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    seed_rows: dict[str, dict[str, str]] = {}
    seed_categories: dict[str, str] = {}
    for category, filename in CATEGORY_FILES.items():
        for row in read_csv(SEED_DIR / filename):
            question_id = row["question_id"]
            if question_id in seed_categories:
                raise ValueError(f"Duplicate seed taxonomy label for {question_id}")
            seed_rows[question_id] = row
            seed_categories[question_id] = category

    metrics = json.loads((SFT_DIR / "per_question_metrics.json").read_text(encoding="utf-8"))
    wrong_metrics = [row for row in metrics if float(row["mrr"]) == 0.0]
    input_rows = {
        row["question_id"]: row
        for line in (SFT_DIR / "evaluation_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }

    results: list[dict[str, str]] = []
    for metric in wrong_metrics:
        question_id = metric["question_id"]
        input_row = input_rows[question_id]
        seed = seed_rows.get(question_id)
        manual = MANUAL_CATEGORIES.get(question_id)
        if seed is None and manual is None:
            raise ValueError(f"No taxonomy label for strict-SFT error {question_id}")

        category = manual["category"] if manual else seed_categories[question_id]
        results.append(
            {
                "question_id": question_id,
                "current_status": "wrong_after_strict_extractive_sft",
                "error_category": category,
                "evidence_status": "supported" if metric["is_evidence_supported"] else "unsupported",
                "batch": metric.get("batch", ""),
                "question": input_row["question"],
                "prediction": metric["prediction"],
                "gold_output": metric["gold_output"],
                "mrr": str(metric["mrr"]),
                "strict_accuracy": str(metric["strict_accuracy"]),
                "resource_count": str(input_row["resource_count"]),
                "review_confidence": manual["confidence"] if manual else seed["review_confidence"],
                "review_notes": manual["notes"] if manual else seed["review_notes"],
                "classification_provenance": (
                    "manual_sft_only_review" if manual else "transferred_from_reviewed_cal_dpo_taxonomy"
                ),
            }
        )

    results.sort(key=lambda row: (row["error_category"], row["question_id"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for category, filename in CATEGORY_FILES.items():
        category_rows = [row for row in results if row["error_category"] == category]
        with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(category_rows)

    with (OUTPUT_DIR / "all_strict_extractive_sft_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["error_category"] for row in results)
    supported_failures = sum(row["evidence_status"] == "supported" for row in results)
    summary = {
        "model_name": "evidence_per_alias_sft_dropout_strict_extractive",
        "evaluation_variant": "full_test",
        "official_error_count": len(results),
        "correct_count": len(metrics) - len(results),
        "supported_error_count": supported_failures,
        "unsupported_error_count": len(results) - supported_failures,
        "categories": [
            {
                "name": category,
                "count": counts[category],
                "description": CATEGORY_DESCRIPTIONS[category],
            }
            for category in CATEGORY_FILES
        ],
        "classification_method": (
            "54 labels were transferred from reviewed full-test Cal-DPO records with the same "
            "strict-SFT failure; one strict-SFT-only record was reviewed manually."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
