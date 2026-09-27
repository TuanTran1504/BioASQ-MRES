from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from src.model_registry import resolve_repo_path
from src.utility.bioasq_format import normalize_for_match
from src.utility.data import clean_text

from .common import load_json_records, read_json, write_json
from .normalize_set_answers import parse_list_output


def safe_mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return sum(items) / len(items) if items else None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not items:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": mean(items),
        "median": median(items),
        "min": min(items),
        "max": max(items),
    }


def counter_json(counter: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def item_count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return ">100"


def normalized_item_key(item: str) -> str:
    return normalize_for_match(item)


def parse_list_items(text: str) -> list[str]:
    return list(parse_list_output(text or "", allow_fallback_split=True).items)


def normalized_set(items: Sequence[str]) -> set[str]:
    return {key for key in (normalized_item_key(item) for item in items) if key}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def resolve_path(path_text: str, project_root: Path) -> Path:
    resolved = resolve_repo_path(path_text, project_root=project_root)
    return resolved if resolved is not None else Path(path_text)


def load_gold_groups(question_input: Sequence[str]) -> dict[str, list[list[str]]]:
    gold_by_question: dict[str, list[list[str]]] = {}
    for raw_path in question_input:
        path = Path(raw_path)
        payload = read_json(path)
        raw_questions = payload.get("questions") if isinstance(payload, Mapping) else None
        if not isinstance(raw_questions, list):
            continue
        for row in raw_questions:
            if not isinstance(row, Mapping):
                continue
            if clean_text(row.get("type", "")).lower() != "list":
                continue
            question_id = clean_text(row.get("id", ""))
            exact_answer = row.get("exact_answer")
            groups: list[list[str]] = []
            if isinstance(exact_answer, list):
                for item in exact_answer:
                    if isinstance(item, list):
                        aliases = [clean_text(alias) for alias in item if clean_text(alias)]
                    else:
                        aliases = [clean_text(item)] if clean_text(item) else []
                    if aliases:
                        groups.append(aliases)
            if question_id and groups:
                gold_by_question[question_id] = groups
    return gold_by_question


def match_items_to_gold(items: Sequence[str], gold_groups: Sequence[Sequence[str]]) -> dict[str, Any]:
    matched_gold: set[int] = set()
    matched_items: list[str] = []
    unsupported_items: list[str] = []
    normalized_gold_groups = [
        {normalized_item_key(alias) for alias in group if normalized_item_key(alias)}
        for group in gold_groups
    ]

    for item in items:
        key = normalized_item_key(item)
        matched = False
        if key:
            for gold_index, gold_keys in enumerate(normalized_gold_groups):
                if gold_index not in matched_gold and key in gold_keys:
                    matched_gold.add(gold_index)
                    matched_items.append(item)
                    matched = True
                    break
        if not matched:
            unsupported_items.append(item)

    prediction_count = len(items)
    gold_count = len(gold_groups)
    true_positives = len(matched_gold)
    precision = true_positives / prediction_count if prediction_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "matched_gold": matched_gold,
        "matched_items": matched_items,
        "unsupported_items": unsupported_items,
        "missing_gold": set(range(gold_count)) - matched_gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prediction_count": prediction_count,
        "gold_count": gold_count,
    }


def model_label_from_dir(path: Path) -> str:
    return path.name


def load_eval_models(eval_root: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for predictions_path in sorted(eval_root.glob("*/predictions.json")):
        model_dir = predictions_path.parent
        label = model_label_from_dir(model_dir)
        rows = json.loads(predictions_path.read_text(encoding="utf-8"))
        scores_path = model_dir / "scores.json"
        scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}
        models[label] = {
            "label": label,
            "model_dir": str(model_dir),
            "predictions": rows,
            "scores": scores,
        }
    return models


def aggregate_model_metrics(model: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(model.get("predictions", []))
    list_metrics = (
        model.get("scores", {})
        .get("aggregate", {})
        .get("by_type", {})
        .get("list", {})
        .get("metrics", {})
    )
    return {
        "mean_precision": list_metrics.get("mean_precision") if list_metrics else safe_mean(row["score"].get("precision") for row in rows),
        "mean_recall": list_metrics.get("mean_recall") if list_metrics else safe_mean(row["score"].get("recall") for row in rows),
        "mean_f1": list_metrics.get("mean_f1") if list_metrics else safe_mean(row["score"].get("f1") for row in rows),
        "avg_prediction_count": list_metrics.get("avg_prediction_count") if list_metrics else safe_mean(row["score"].get("prediction_count") for row in rows),
        "avg_gold_count": list_metrics.get("avg_gold_count") if list_metrics else safe_mean(row["score"].get("gold_count") for row in rows),
        "question_count": len(rows),
    }


def row_by_question(model: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {clean_text(row.get("question_id")): row for row in model.get("predictions", [])}


def compare_models(
    models: Mapping[str, Mapping[str, Any]],
    baseline_label: str | None,
    gold_by_question: Mapping[str, Sequence[Sequence[str]]],
    example_limit: int,
) -> dict[str, Any]:
    if not models:
        return {}
    if baseline_label is None or baseline_label not in models:
        baseline_label = next((label for label in models if "answer-gen-list-fullcontext" in label), None)
    if baseline_label is None or baseline_label not in models:
        return {"warning": "No baseline model was found for pairwise comparison."}

    baseline_rows = row_by_question(models[baseline_label])
    comparisons: dict[str, Any] = {}
    for label, model in models.items():
        if label == baseline_label:
            continue
        current_rows = row_by_question(model)
        shared_ids = sorted(set(baseline_rows) & set(current_rows))
        deltas = []
        item_deltas = []
        unsupported_deltas = []
        valid_removed_counts = []
        shorter_count = 0
        unsupported_reduced_count = 0
        valid_removed_question_count = 0
        improves = []
        worsens = []

        for question_id in shared_ids:
            base = baseline_rows[question_id]
            cur = current_rows[question_id]
            base_score = base.get("score", {})
            cur_score = cur.get("score", {})
            delta_f1 = float(cur_score.get("f1", 0.0)) - float(base_score.get("f1", 0.0))
            delta_precision = float(cur_score.get("precision", 0.0)) - float(base_score.get("precision", 0.0))
            delta_recall = float(cur_score.get("recall", 0.0)) - float(base_score.get("recall", 0.0))
            base_items = parse_list_items(base.get("prediction", ""))
            cur_items = parse_list_items(cur.get("prediction", ""))
            item_delta = len(cur_items) - len(base_items)
            item_deltas.append(item_delta)
            if item_delta < 0:
                shorter_count += 1

            gold_groups = gold_by_question.get(question_id, [])
            base_match = match_items_to_gold(base_items, gold_groups)
            cur_match = match_items_to_gold(cur_items, gold_groups)
            valid_removed = sorted(base_match["matched_gold"] - cur_match["matched_gold"])
            unsupported_delta = len(cur_match["unsupported_items"]) - len(base_match["unsupported_items"])
            unsupported_deltas.append(unsupported_delta)
            valid_removed_counts.append(len(valid_removed))
            if valid_removed:
                valid_removed_question_count += 1
            if unsupported_delta < 0:
                unsupported_reduced_count += 1

            record = {
                "question_id": question_id,
                "question": clean_text(cur.get("body", "")),
                "delta_f1": delta_f1,
                "delta_precision": delta_precision,
                "delta_recall": delta_recall,
                "baseline_f1": base_score.get("f1"),
                "model_f1": cur_score.get("f1"),
                "baseline_prediction_count": len(base_items),
                "model_prediction_count": len(cur_items),
                "baseline_prediction": base.get("prediction", ""),
                "model_prediction": cur.get("prediction", ""),
                "valid_gold_groups_removed_count": len(valid_removed),
                "unsupported_item_delta": unsupported_delta,
            }
            deltas.append(delta_f1)
            if delta_f1 > 0:
                improves.append(record)
            elif delta_f1 < 0:
                worsens.append(record)

        improves.sort(key=lambda row: row["delta_f1"], reverse=True)
        worsens.sort(key=lambda row: row["delta_f1"])
        comparisons[label] = {
            "baseline": baseline_label,
            "shared_question_count": len(shared_ids),
            "delta_f1": numeric_summary(deltas),
            "delta_item_count": numeric_summary(item_deltas),
            "delta_unsupported_item_count": numeric_summary(unsupported_deltas),
            "valid_gold_groups_removed_per_question": numeric_summary(valid_removed_counts),
            "questions_model_shorter_than_baseline": shorter_count,
            "questions_reducing_unsupported_additions": unsupported_reduced_count,
            "questions_removing_valid_gold_groups": valid_removed_question_count,
            "improved_question_count": len(improves),
            "worsened_question_count": len(worsens),
            "unchanged_question_count": len(shared_ids) - len(improves) - len(worsens),
            "top_improvements": improves[:example_limit],
            "top_regressions": worsens[:example_limit],
        }
    return comparisons


def load_candidate_manifest(candidate_bank: Path) -> dict[str, Any]:
    manifest_path = candidate_bank.parent / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}


def analyze_candidate_bank(
    candidate_bank: Path | None,
    gold_by_question: Mapping[str, Sequence[Sequence[str]]],
    pairwise_limit_per_question: int,
) -> dict[str, Any]:
    if candidate_bank is None or not candidate_bank.exists():
        return {"available": False}

    rows = list(load_json_records(candidate_bank))
    status_counts = Counter()
    generated_tokens = []
    lengths = []
    score_values = []
    precision_values = []
    recall_values = []
    false_positive_counts = []
    false_negative_counts = []
    normalized_sets_by_question: defaultdict[str, list[set[str]]] = defaultdict(list)
    exact_set_counts_by_question: defaultdict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    useful_difference_questions: set[str] = set()
    questions_with_valid_additions: set[str] = set()
    questions_with_omissions: set[str] = set()

    for row in rows:
        question_id = clean_text(row.get("question_id") or row.get("id"))
        parsed = parse_list_output(clean_text(row.get("raw_output") or row.get("prediction")), allow_fallback_split=False)
        status_counts[parsed.status] += 1
        token_count = row.get("generated_token_count")
        if isinstance(token_count, int):
            generated_tokens.append(token_count)
        if parsed.status != "ok":
            continue

        items = list(parsed.items)
        item_set = normalized_set(items)
        normalized_sets_by_question[question_id].append(item_set)
        exact_set_counts_by_question[question_id][tuple(sorted(item_set))] += 1
        lengths.append(len(items))

        gold_groups = gold_by_question.get(question_id, [])
        if gold_groups:
            match = match_items_to_gold(items, gold_groups)
            score_values.append(match["f1"])
            precision_values.append(match["precision"])
            recall_values.append(match["recall"])
            false_positive_counts.append(len(match["unsupported_items"]))
            false_negative_counts.append(len(match["missing_gold"]))
            if match["unsupported_items"]:
                questions_with_valid_additions.add(question_id)
            if match["missing_gold"]:
                questions_with_omissions.add(question_id)

    duplicate_response_count = 0
    ok_response_count = 0
    unique_response_counts = []
    pairwise_jaccards = []
    score_range_by_question = defaultdict(list)

    for question_id, set_list in normalized_sets_by_question.items():
        ok_response_count += len(set_list)
        counter = exact_set_counts_by_question[question_id]
        duplicate_response_count += sum(max(0, count - 1) for count in counter.values())
        unique_response_counts.append(len(counter))
        sampled_sets = set_list[:pairwise_limit_per_question]
        for left, right in itertools.combinations(sampled_sets, 2):
            pairwise_jaccards.append(jaccard(left, right))

    if gold_by_question:
        scores_by_question: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            question_id = clean_text(row.get("question_id") or row.get("id"))
            parsed = parse_list_output(clean_text(row.get("raw_output") or row.get("prediction")), allow_fallback_split=False)
            if parsed.status != "ok" or question_id not in gold_by_question:
                continue
            match = match_items_to_gold(parsed.items, gold_by_question[question_id])
            scores_by_question[question_id].append(match["f1"])
        for question_id, scores in scores_by_question.items():
            if len(scores) >= 2 and max(scores) - min(scores) >= 0.05:
                useful_difference_questions.add(question_id)
            score_range_by_question[question_id] = scores

    return {
        "available": True,
        "path": str(candidate_bank),
        "manifest": load_candidate_manifest(candidate_bank),
        "row_count": len(rows),
        "parser_status_counts": counter_json(status_counts),
        "generated_token_count": numeric_summary(generated_tokens),
        "answer_set_size": numeric_summary(lengths),
        "answer_set_size_buckets": counter_json(Counter(item_count_bucket(int(value)) for value in lengths)),
        "score_distribution_exact_alias": numeric_summary(score_values),
        "precision_distribution_exact_alias": numeric_summary(precision_values),
        "recall_distribution_exact_alias": numeric_summary(recall_values),
        "unsupported_addition_count_per_response": numeric_summary(false_positive_counts),
        "omission_count_per_response": numeric_summary(false_negative_counts),
        "question_count_with_valid_outputs": len(normalized_sets_by_question),
        "question_count_with_useful_f1_differences": len(useful_difference_questions),
        "question_count_with_any_unsupported_addition": len(questions_with_valid_additions),
        "question_count_with_any_omission": len(questions_with_omissions),
        "unique_response_sets_per_question": numeric_summary(unique_response_counts),
        "duplicate_response_count": duplicate_response_count,
        "duplicate_response_rate_among_ok": duplicate_response_count / ok_response_count if ok_response_count else 0.0,
        "candidate_diversity_unique_set_rate": (
            sum(unique_response_counts) / ok_response_count if ok_response_count else 0.0
        ),
        "pairwise_normalized_set_jaccard": numeric_summary(pairwise_jaccards),
        "note": (
            "Similarity is normalized answer-set Jaccard, not SapBERT. It is a fast proxy "
            "for candidate diversity and duplicate-like behavior."
        ),
    }


def analyze_pair_file(
    pair_path: Path,
    gold_by_question: Mapping[str, Sequence[Sequence[str]]],
) -> dict[str, Any]:
    rows = list(load_json_records(pair_path))
    pair_type_counts = Counter(clean_text(row.get("pair_type")) for row in rows)
    length_direction = Counter()
    conflicting_changes = 0
    valid_only_in_rejected = 0
    unsupported_only_in_chosen = 0
    chosen_shorter_delta_f1 = []
    chosen_longer_delta_f1 = []
    same_length_delta_f1 = []
    exact_member_diff_counts = []
    chosen_only_counts = []
    rejected_only_counts = []

    for row in rows:
        chosen_items = [clean_text(item) for item in row.get("chosen_items", []) if clean_text(item)]
        rejected_items = [clean_text(item) for item in row.get("rejected_items", []) if clean_text(item)]
        chosen_set = normalized_set(chosen_items)
        rejected_set = normalized_set(rejected_items)
        chosen_only = chosen_set - rejected_set
        rejected_only = rejected_set - chosen_set
        chosen_only_counts.append(len(chosen_only))
        rejected_only_counts.append(len(rejected_only))
        exact_member_diff_counts.append(len(chosen_only | rejected_only))
        if chosen_only and rejected_only:
            conflicting_changes += 1

        length_delta = len(chosen_items) - len(rejected_items)
        delta_f1 = row.get("delta_f1")
        if length_delta < 0:
            length_direction["chosen_shorter"] += 1
            if isinstance(delta_f1, (int, float)):
                chosen_shorter_delta_f1.append(float(delta_f1))
        elif length_delta > 0:
            length_direction["chosen_longer"] += 1
            if isinstance(delta_f1, (int, float)):
                chosen_longer_delta_f1.append(float(delta_f1))
        else:
            length_direction["same_length"] += 1
            if isinstance(delta_f1, (int, float)):
                same_length_delta_f1.append(float(delta_f1))

        gold_groups = gold_by_question.get(clean_text(row.get("question_id")), [])
        if gold_groups:
            chosen_match = match_items_to_gold(chosen_items, gold_groups)
            rejected_match = match_items_to_gold(rejected_items, gold_groups)
            if rejected_match["matched_gold"] - chosen_match["matched_gold"]:
                valid_only_in_rejected += 1
            unsupported_chosen_keys = normalized_set(chosen_match["unsupported_items"])
            unsupported_rejected_keys = normalized_set(rejected_match["unsupported_items"])
            if unsupported_chosen_keys - unsupported_rejected_keys:
                unsupported_only_in_chosen += 1

    semantic_distances = [
        row.get("semantic_set_edit_distance")
        for row in rows
        if isinstance(row.get("semantic_set_edit_distance"), (int, float))
    ]
    return {
        "path": str(pair_path),
        "pair_count": len(rows),
        "unique_questions": len({clean_text(row.get("question_id")) for row in rows}),
        "pair_type_counts": counter_json(pair_type_counts),
        "delta_f1": numeric_summary(row.get("delta_f1") for row in rows),
        "delta_precision": numeric_summary(row.get("delta_precision") for row in rows),
        "delta_recall": numeric_summary(row.get("delta_recall") for row in rows),
        "semantic_set_edit_distance": numeric_summary(semantic_distances),
        "exact_normalized_member_diff": numeric_summary(exact_member_diff_counts),
        "chosen_only_member_count": numeric_summary(chosen_only_counts),
        "rejected_only_member_count": numeric_summary(rejected_only_counts),
        "length_direction_counts": counter_json(length_direction),
        "delta_f1_by_length_direction": {
            "chosen_shorter": numeric_summary(chosen_shorter_delta_f1),
            "chosen_longer": numeric_summary(chosen_longer_delta_f1),
            "same_length": numeric_summary(same_length_delta_f1),
        },
        "pairs_with_conflicting_exact_changes": conflicting_changes,
        "pairs_where_valid_gold_member_only_in_rejected": valid_only_in_rejected,
        "pairs_where_unsupported_member_only_in_chosen": unsupported_only_in_chosen,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return ""
        text = str(value)
        return text.replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def short_prediction(text: str, max_chars: int = 450) -> str:
    cleaned = clean_text(text)
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 3] + "..."


def render_report(payload: Mapping[str, Any]) -> str:
    parts = ["# DPO Diagnostic Report", ""]

    model_metrics = payload.get("model_metrics", {})
    if model_metrics:
        parts.extend([
            "## 1. Model Performance",
            "",
            markdown_table(
                ["Model", "Mean F1", "Precision", "Recall", "Avg Items", "Avg Gold", "Questions"],
                [
                    [
                        label,
                        metrics.get("mean_f1"),
                        metrics.get("mean_precision"),
                        metrics.get("mean_recall"),
                        metrics.get("avg_prediction_count"),
                        metrics.get("avg_gold_count"),
                        metrics.get("question_count"),
                    ]
                    for label, metrics in model_metrics.items()
                ],
            ),
            "",
        ])

    comparisons = payload.get("model_comparisons", {})
    if comparisons:
        parts.extend(["## 2. SFT Comparison", ""])
        for label, comparison in comparisons.items():
            if "warning" in comparison:
                parts.append(comparison["warning"])
                parts.append("")
                continue
            parts.append(f"### {label} vs {comparison.get('baseline')}")
            parts.append("")
            parts.append(
                markdown_table(
                    ["Signal", "Value"],
                    [
                        ["Shared questions", comparison.get("shared_question_count")],
                        ["Improved / worsened / unchanged", f"{comparison.get('improved_question_count')} / {comparison.get('worsened_question_count')} / {comparison.get('unchanged_question_count')}"],
                        ["Mean delta F1", comparison.get("delta_f1", {}).get("mean")],
                        ["Mean delta item count", comparison.get("delta_item_count", {}).get("mean")],
                        ["Questions shorter than SFT", comparison.get("questions_model_shorter_than_baseline")],
                        ["Questions reducing unsupported additions", comparison.get("questions_reducing_unsupported_additions")],
                        ["Questions removing valid gold groups", comparison.get("questions_removing_valid_gold_groups")],
                    ],
                )
            )
            parts.append("")
            for title, examples in [
                ("Top Improvements", comparison.get("top_improvements", [])),
                ("Top Regressions", comparison.get("top_regressions", [])),
            ]:
                parts.append(f"#### {title}")
                parts.append("")
                if not examples:
                    parts.append("No examples.")
                    parts.append("")
                    continue
                for example in examples:
                    parts.append(f"- QID `{example['question_id']}` delta F1 `{example['delta_f1']:.4f}`")
                    parts.append(f"  Question: {example['question']}")
                    parts.append(f"  Baseline: {short_prediction(example['baseline_prediction'])}")
                    parts.append(f"  Model: {short_prediction(example['model_prediction'])}")
                parts.append("")

    candidate = payload.get("candidate_quality", {})
    if candidate and candidate.get("available"):
        generation = candidate.get("manifest", {}).get("generation", {})
        parts.extend([
            "## 3. Candidate Sample Quality",
            "",
            markdown_table(
                ["Signal", "Value"],
                [
                    ["Rows", candidate.get("row_count")],
                    ["Generation settings", json.dumps(generation, ensure_ascii=False)],
                    ["Parser status", json.dumps(candidate.get("parser_status_counts"), ensure_ascii=False)],
                    ["Answer-set size mean/median/max", f"{candidate['answer_set_size'].get('mean'):.2f} / {candidate['answer_set_size'].get('median'):.2f} / {candidate['answer_set_size'].get('max'):.0f}" if candidate.get("answer_set_size", {}).get("mean") is not None else ""],
                    ["Size buckets", json.dumps(candidate.get("answer_set_size_buckets"), ensure_ascii=False)],
                    ["Duplicate response rate among OK", candidate.get("duplicate_response_rate_among_ok")],
                    ["Unique set rate among OK", candidate.get("candidate_diversity_unique_set_rate")],
                    ["Mean pairwise normalized-set Jaccard", candidate.get("pairwise_normalized_set_jaccard", {}).get("mean")],
                    ["Mean F1 exact-alias proxy", candidate.get("score_distribution_exact_alias", {}).get("mean")],
                    ["Questions with useful F1 differences", candidate.get("question_count_with_useful_f1_differences")],
                    ["Questions with unsupported additions", candidate.get("question_count_with_any_unsupported_addition")],
                    ["Questions with omissions", candidate.get("question_count_with_any_omission")],
                ],
            ),
            "",
        ])

    pair_quality = payload.get("pair_quality", {})
    if pair_quality:
        parts.extend(["## 4. Preference-Pair Quality", ""])
        pair_rows = []
        for label, metrics in pair_quality.items():
            pair_rows.append([
                label,
                metrics.get("pair_count"),
                metrics.get("unique_questions"),
                json.dumps(metrics.get("pair_type_counts"), ensure_ascii=False),
                metrics.get("delta_f1", {}).get("mean"),
                metrics.get("semantic_set_edit_distance", {}).get("median"),
                json.dumps(metrics.get("length_direction_counts"), ensure_ascii=False),
                metrics.get("pairs_with_conflicting_exact_changes"),
                metrics.get("pairs_where_valid_gold_member_only_in_rejected"),
                metrics.get("pairs_where_unsupported_member_only_in_chosen"),
            ])
        parts.append(
            markdown_table(
                [
                    "Dataset",
                    "Pairs",
                    "Questions",
                    "Types",
                    "Mean Delta F1",
                    "Median Semantic Diff",
                    "Length Direction",
                    "Conflicting Changes",
                    "Valid Only Rejected",
                    "Unsupported Only Chosen",
                ],
                pair_rows,
            )
        )
        parts.append("")

    parts.extend([
        "## Interpretation",
        "",
        "- If DPO has lower average item count than SFT while precision rises and recall falls, it is learning a conservative shortening strategy.",
        "- Whole-response pairs are clearest when chosen and rejected differ by quality, not just length. Large semantic edit distances and many chosen-shorter pairs indicate broad anti-overgeneration supervision.",
        "- Edit pairs are easier to audit, but omission pairs built on very long rejected outputs can still be noisy.",
    ])
    return "\n".join(parts).rstrip() + "\n"


def parse_pair_arg(values: Sequence[str]) -> list[tuple[str, Path]]:
    pairs = []
    for value in values:
        if "=" in value:
            label, path_text = value.split("=", 1)
        else:
            path_text = value
            label = Path(path_text).stem
        pairs.append((clean_text(label) or Path(path_text).stem, Path(path_text)))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose DPO evaluation, candidate-bank, and preference-pair quality.")
    parser.add_argument("--eval-root", default=None, help="Evaluation root containing one subdirectory per model.")
    parser.add_argument("--question-input", nargs="+", default=["data/training13b.json"], help="Raw BioASQ files for gold aliases.")
    parser.add_argument("--candidate-bank", default=None, help="Candidate-bank JSONL to analyse.")
    parser.add_argument(
        "--pair-jsonl",
        action="append",
        default=[],
        help="Preference pair JSONL. Use label=path to control the report label. Can be repeated.",
    )
    parser.add_argument("--baseline-label", default=None, help="Model directory label to use as SFT baseline.")
    parser.add_argument("--output-json", required=True, help="Output diagnostic JSON path.")
    parser.add_argument("--output-md", required=True, help="Output diagnostic Markdown path.")
    parser.add_argument("--example-limit", type=int, default=5, help="Examples per improvement/regression section.")
    parser.add_argument("--pairwise-limit-per-question", type=int, default=16, help="Candidate sets per question for pairwise Jaccard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    question_paths = [resolve_path(path, project_root) for path in args.question_input]
    gold_by_question = load_gold_groups([str(path) for path in question_paths])

    models: dict[str, dict[str, Any]] = {}
    if args.eval_root:
        eval_root = resolve_path(args.eval_root, project_root)
        models = load_eval_models(eval_root)

    model_metrics = {label: aggregate_model_metrics(model) for label, model in models.items()}
    model_comparisons = compare_models(
        models=models,
        baseline_label=args.baseline_label,
        gold_by_question=gold_by_question,
        example_limit=int(args.example_limit),
    )

    candidate_bank = resolve_path(args.candidate_bank, project_root) if args.candidate_bank else None
    candidate_quality = analyze_candidate_bank(
        candidate_bank=candidate_bank,
        gold_by_question=gold_by_question,
        pairwise_limit_per_question=int(args.pairwise_limit_per_question),
    )

    pair_quality = {}
    for label, path in parse_pair_arg(args.pair_jsonl):
        resolved = resolve_path(str(path), project_root)
        if resolved.exists():
            pair_quality[label] = analyze_pair_file(resolved, gold_by_question=gold_by_question)
        else:
            pair_quality[label] = {"available": False, "path": str(resolved)}

    payload = {
        "model_metrics": model_metrics,
        "model_comparisons": model_comparisons,
        "candidate_quality": candidate_quality,
        "pair_quality": pair_quality,
    }
    output_json = resolve_path(args.output_json, project_root)
    output_md = resolve_path(args.output_md, project_root)
    write_json(output_json, payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_report(payload), encoding="utf-8")
    print(f"Wrote diagnostic JSON to {output_json}")
    print(f"Wrote diagnostic report to {output_md}")


if __name__ == "__main__":
    main()
