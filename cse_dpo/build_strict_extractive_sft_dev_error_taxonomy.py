#!/usr/bin/env python3
"""Write the reviewed full-development error taxonomy for strict-extractive SFT."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT
    / "Artifacts/Factoid_SFT/evaluations/factoid_dev_qwen25_05b_full_dev_20260908_110518"
)
SFT_DIR = EVALUATION_ROOT / "evidence_per_alias_sft_dropout_strict_extractive"
OUTPUT_DIR = EVALUATION_ROOT / "error_analysis_strict_extractive_sft_full_dev_v1"

CATEGORY_DESCRIPTIONS = {
    "unsupported_gold_absent_from_provided_resources": (
        "No normalized official gold alias occurs in the provided resources. This is an "
        "evidence-availability limitation, not an extractive selection failure."
    ),
    "numerical_or_quantitative_selection_error": (
        "The model selects an incorrect number, range, unit, threshold, or denominator."
    ),
    "strict_surface_or_alias_mismatch": (
        "A semantically equivalent or nearly equivalent answer fails official matching due to "
        "abbreviation, spelling, punctuation, hyphenation, or an unlisted alias."
    ),
    "span_or_granularity_mismatch": (
        "The prediction is related to the answer but is too broad, too narrow, incomplete, "
        "or over-specific for the required answer span."
    ),
    "wrong_information_or_question_target": (
        "The prediction selects a different entity, relation, outcome, or target from the "
        "evidence-supported answer."
    ),
    "long_form_or_ambiguous_annotation": (
        "The official factoid target is sentence-like, has multiple plausible answer forms, "
        "or is annotation-sensitive despite a defensible short extraction."
    ),
}

# These 59 evidence-supported failures were reviewed against their full-context
# prompts. Unsupported failures are labelled automatically from the evaluation.
SUPPORTED_CATEGORIES = {
    "51bdd9c2047fa84d1d000002": "strict_surface_or_alias_mismatch",
    "516e7fda298dcd4e51000081": "strict_surface_or_alias_mismatch",
    "5171438a8ed59a060a000007": "strict_surface_or_alias_mismatch",
    "52c7311903868f1b0600001d": "numerical_or_quantitative_selection_error",
    "52e8e93498d023950500001e": "strict_surface_or_alias_mismatch",
    "530cf4fe960c95ad0c00000b": "wrong_information_or_question_target",
    "5324a8ac9b2d7acc7e000018": "span_or_granularity_mismatch",
    "5324ce779b2d7acc7e00001e": "wrong_information_or_question_target",
    "5348307daeec6fbd07000011": "strict_surface_or_alias_mismatch",
    "54e0e902ae9738404b000001": "long_form_or_ambiguous_annotation",
    "54f2210164850a5854000001": "long_form_or_ambiguous_annotation",
    "54f5bc7d5f206a0c06000001": "long_form_or_ambiguous_annotation",
    "54f9cb34dd3fc62544000002": "strict_surface_or_alias_mismatch",
    "5505edac8e1671127b000005": "wrong_information_or_question_target",
    "552faababc4f83e828000005": "strict_surface_or_alias_mismatch",
    "56a8ee75a17756b72f000007": "long_form_or_ambiguous_annotation",
    "56c1f021ef6e394741000048": "wrong_information_or_question_target",
    "56c86aa95795f9a73e000018": "wrong_information_or_question_target",
    "56df03c751531f7e3300000a": "wrong_information_or_question_target",
    "56e2acfe51531f7e33000014": "wrong_information_or_question_target",
    "56e6ec49edfc094c1f000005": "strict_surface_or_alias_mismatch",
    "57136a7e1174fb1755000006": "strict_surface_or_alias_mismatch",
    "571e2beabb137a4b0c000006": "span_or_granularity_mismatch",
    "58af1cb3717cd3f655000003": "span_or_granularity_mismatch",
    "58bc8e7a02b8c60953000007": "span_or_granularity_mismatch",
    "58c6635f02b8c60953000023": "strict_surface_or_alias_mismatch",
    "5a6d1733b750ff4455000030": "strict_surface_or_alias_mismatch",
    "5a7373f63b9d13c708000008": "long_form_or_ambiguous_annotation",
    "5a8056a2faa1ab7d2e00001f": "strict_surface_or_alias_mismatch",
    "5aae6499fcf456587200000c": "strict_surface_or_alias_mismatch",
    "5abd5a62fcf4565872000031": "wrong_information_or_question_target",
    "5c0117fd133db5eb7800002a": "span_or_granularity_mismatch",
    "5c5f08ad1a4c55d80b00000b": "long_form_or_ambiguous_annotation",
    "5c5f1f371a4c55d80b00001a": "wrong_information_or_question_target",
    "5c6e05f37c78d69471000049": "span_or_granularity_mismatch",
    "5c72f5247c78d6947100007e": "strict_surface_or_alias_mismatch",
    "5c73acec7c78d69471000086": "span_or_granularity_mismatch",
    "5cb0d647ecadf2e73f000059": "span_or_granularity_mismatch",
    "5d35f1267bc3fee31f000004": "wrong_information_or_question_target",
    "5d36a9507bc3fee31f000005": "strict_surface_or_alias_mismatch",
    "5e4adb486d0a277941000015": "wrong_information_or_question_target",
    "5e48edb1f8b2df0d49000002": "strict_surface_or_alias_mismatch",
    "5e5b626fb761aafe0900000c": "long_form_or_ambiguous_annotation",
    "5e7f64d6835f4e477700001f": "span_or_granularity_mismatch",
    "602598201cb411341a0000af": "wrong_information_or_question_target",
    "6026ed981cb411341a0000d2": "wrong_information_or_question_target",
    "602828b11cb411341a0000fc": "strict_surface_or_alias_mismatch",
    "6048ff3e1cb411341a000160": "strict_surface_or_alias_mismatch",
    "61f80e22882a024a1000003d": "span_or_granularity_mismatch",
    "621b54f03a8413c65300003b": "long_form_or_ambiguous_annotation",
    "621fcaf93a8413c653000065": "span_or_granularity_mismatch",
    "6221209b3a8413c653000074": "span_or_granularity_mismatch",
    "62532ffee764a53204000023": "span_or_granularity_mismatch",
    "63f03ea0f36125a426000020": "span_or_granularity_mismatch",
    "64178ffb690f196b51000029": "span_or_granularity_mismatch",
    "643bc8f957b1c7a31500002b": "span_or_granularity_mismatch",
    "65cec1fb1930410b13000005": "span_or_granularity_mismatch",
    "66302487187cba990d000031": "span_or_granularity_mismatch",
    "6630390b187cba990d000035": "strict_surface_or_alias_mismatch",
}

OUTPUT_COLUMNS = [
    "question_id", "current_status", "error_category", "evidence_status", "batch",
    "question", "prediction", "gold_output", "mrr", "strict_accuracy", "resource_count",
    "review_confidence", "review_notes", "classification_provenance",
]


def main() -> None:
    metrics = json.loads((SFT_DIR / "per_question_metrics.json").read_text(encoding="utf-8"))
    inputs = {
        row["question_id"]: row
        for line in (SFT_DIR / "evaluation_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    wrong = [row for row in metrics if float(row["mrr"]) == 0.0]
    supported_ids = {row["question_id"] for row in wrong if row["is_evidence_supported"]}
    if supported_ids != set(SUPPORTED_CATEGORIES):
        missing = sorted(supported_ids - set(SUPPORTED_CATEGORIES))
        stale = sorted(set(SUPPORTED_CATEGORIES) - supported_ids)
        raise ValueError(f"Supported taxonomy mismatch; missing={missing}, stale={stale}")

    rows = []
    for metric in wrong:
        question_id = metric["question_id"]
        input_row = inputs[question_id]
        supported = bool(metric["is_evidence_supported"])
        category = (
            SUPPORTED_CATEGORIES[question_id]
            if supported
            else "unsupported_gold_absent_from_provided_resources"
        )
        rows.append({
            "question_id": question_id,
            "current_status": "wrong_after_strict_extractive_sft",
            "error_category": category,
            "evidence_status": "supported" if supported else "unsupported",
            "batch": metric.get("batch", ""),
            "question": input_row["question"],
            "prediction": metric["prediction"],
            "gold_output": metric["gold_output"],
            "mrr": str(metric["mrr"]),
            "strict_accuracy": str(metric["strict_accuracy"]),
            "resource_count": str(input_row["resource_count"]),
            "review_confidence": "medium" if supported else "high",
            "review_notes": CATEGORY_DESCRIPTIONS[category],
            "classification_provenance": "manual_full_context_review" if supported else "evidence_support_audit",
        })

    rows.sort(key=lambda row: (row["error_category"], row["question_id"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "all_strict_extractive_sft_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["error_category"] for row in rows)
    for category in CATEGORY_DESCRIPTIONS:
        with (OUTPUT_DIR / f"{category}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(row for row in rows if row["error_category"] == category)

    summary = {
        "model_name": "evidence_per_alias_sft_dropout_strict_extractive",
        "evaluation_variant": "full_dev",
        "official_error_count": len(rows),
        "correct_count": len(metrics) - len(rows),
        "supported_error_count": sum(row["evidence_status"] == "supported" for row in rows),
        "unsupported_error_count": sum(row["evidence_status"] == "unsupported" for row in rows),
        "categories": [
            {"name": category, "count": counts[category], "description": description}
            for category, description in CATEGORY_DESCRIPTIONS.items()
        ],
        "classification_method": (
            "All evidence-supported full-context errors were manually assigned a dominant cause. "
            "Unsupported errors are identified by the gold-evidence audit."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
