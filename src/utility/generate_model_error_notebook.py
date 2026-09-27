from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells = [
        md_cell(
            """
            # Model Evaluation Mistake Analysis

            This notebook helps inspect **list-question evaluation mistakes** for one or more model runs.

            It is designed for the evaluation folders already produced by this repo, for example:

            - `Artifacts/evaluations/bioasq-8b-visible-gold_mr5_test`
            - `Artifacts/evaluations/bioasq-unitor_mr5_test`
            - `Artifacts/evaluations/dpo_evals/dpo-unitor-8b-recall-balanced_mr5_test`

            What it does:

            - loads one or more evaluation runs
            - uses official BioASQ Java scores and parser-based error diagnostics
            - shows **precision / recall / F1 / average prediction count**
            - breaks mistakes into **perfect / under-only / over-only / both**
            - separates misses into **prompt-visible** vs **hidden by current evidence**
            - gives heuristics for likely causes:
              - undergeneration
              - spending slots on wrong extras
              - accepted-form / alias mismatch
            - lets you inspect a single question in detail across models

            Notes:

            - this notebook is currently **list-focused**
            - it uses the repo's own normalization and resource-building utilities
            - you can change `MODEL_RUNS` in the setup cell to compare any runs you want
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            import math
            import sys
            from collections import Counter
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Markdown, display


            def find_task_root(start: Path) -> Path:
                current = start.resolve()
                candidates = [current, *current.parents]
                for candidate in candidates:
                    if (candidate / "src").exists() and (candidate / "Artifacts").exists():
                        return candidate
                    nested = candidate / "Task-Structured Counterfactual Preference Mining"
                    if (nested / "src").exists() and (nested / "Artifacts").exists():
                        return nested
                raise FileNotFoundError("Could not locate the Task-Structured Counterfactual Preference Mining root.")


            TASK_ROOT = find_task_root(Path.cwd())
            if str(TASK_ROOT) not in sys.path:
                sys.path.insert(0, str(TASK_ROOT))

            from src.utility.bioasq_format import (
                exact_answer_groups,
                match_to_gold_group,
                normalize_for_match,
                parse_prediction_items,
            )
            from src.utility.data import build_resources, clean_text
            from src.utility.eval_types import EvalExample

            pd.options.display.max_colwidth = 160
            pd.options.display.max_rows = 200

            TEST_FILES = [
                TASK_ROOT / "data" / "Task13BTest" / "13B1_golden.json",
                TASK_ROOT / "data" / "Task13BTest" / "13B2_golden.json",
                TASK_ROOT / "data" / "Task13BTest" / "13B3_golden.json",
                TASK_ROOT / "data" / "Task13BTest" / "13B4_golden.json",
            ]

            DEFAULT_MODEL_RUNS = {
                "bioasq-8b-visible-gold": TASK_ROOT / "Artifacts" / "evaluations" / "bioasq-8b-visible-gold_mr5_test",
                "unitutor-8b": TASK_ROOT / "Artifacts" / "evaluations" / "bioasq-unitor_mr5_test",
                "bioasq-8b-first5-snippets": TASK_ROOT / "Artifacts" / "evaluations" / "bioasq-8b-first5-snippets_mr5_test",
                "bioasq-8b-baseline-full-snippets-mr5": TASK_ROOT / "Artifacts" / "evaluations" / "bioasq-8b-baseline-full-snippets_mr5_test",
                "dpo-unitutor-8b-recall-balanced": TASK_ROOT / "Artifacts" / "evaluations" / "dpo_evals" / "dpo-unitor-8b-recall-balanced_mr5_test",
            }

            MODEL_RUNS = {
                label: path
                for label, path in DEFAULT_MODEL_RUNS.items()
                if Path(path).exists()
            }

            print(f"TASK_ROOT = {TASK_ROOT}")
            print("Available default model runs:")
            for label, path in MODEL_RUNS.items():
                print(f"  - {label}: {path}")
            """
        ),
        code_cell(
            """
            def load_json(path: Path):
                with path.open(encoding="utf-8") as handle:
                    return json.load(handle)


            def resolve_eval_model_dir(path_like: str | Path) -> tuple[Path, Path]:
                path = Path(path_like)
                if (path / "predictions.json").exists():
                    eval_root = path.parent if (path.parent / "manifest.json").exists() else path
                    return path, eval_root

                candidates = sorted(pred.parent for pred in path.glob("*/predictions.json"))
                if not candidates:
                    raise FileNotFoundError(f"No predictions.json found under {path}")
                if len(candidates) > 1:
                    names = ", ".join(candidate.name for candidate in candidates)
                    raise ValueError(
                        f"{path} contains multiple model result folders ({names}). "
                        "Pass the specific subfolder instead."
                    )
                return candidates[0], path


            def make_eval_example(question: dict) -> EvalExample:
                return EvalExample(
                    question_id=clean_text(question.get("id", "")),
                    question_type="list",
                    body=clean_text(question.get("body", "")),
                    instruction="",
                    resources=(),
                    gold_output="",
                    source_path="",
                    raw_question=question,
                )


            def load_list_questions(paths: list[Path]) -> dict[str, dict]:
                questions: dict[str, dict] = {}
                for path in paths:
                    payload = load_json(path)
                    for row in payload.get("questions", []):
                        if clean_text(row.get("type", "")).lower() != "list":
                            continue
                        question_id = clean_text(row.get("id", ""))
                        if question_id:
                            questions[question_id] = row
                return questions


            def build_question_resources(question: dict, manifest: dict) -> list[str]:
                dataset_cfg = manifest.get("dataset", {}) if isinstance(manifest, dict) else {}
                return build_resources(
                    question,
                    max_resources=int(dataset_cfg.get("max_resources", 0) or 0),
                    max_resource_chars=int(dataset_cfg.get("max_resource_chars", 0) or 0),
                    question_text=clean_text(question.get("body", "")),
                    resource_granularity=clean_text(dataset_cfg.get("resource_granularity", "document")) or "document",
                    resource_selection=clean_text(dataset_cfg.get("resource_selection", "first")) or "first",
                )


            def normalized_tokens(text: str) -> set[str]:
                normalized = normalize_for_match(text)
                return set(normalized.split()) if normalized else set()


            def looks_like_alias_mismatch(prediction_item: str, aliases: list[str]) -> bool:
                prediction_norm = normalize_for_match(prediction_item)
                prediction_tokens = normalized_tokens(prediction_item)
                if not prediction_norm:
                    return False
                for alias in aliases:
                    alias_norm = normalize_for_match(alias)
                    if not alias_norm:
                        continue
                    if prediction_norm == alias_norm:
                        return True
                    if prediction_norm in alias_norm or alias_norm in prediction_norm:
                        return True
                    if prediction_tokens and prediction_tokens & normalized_tokens(alias):
                        return True
                return False


            def analyze_prediction(question: dict, prediction_text: str, resources: list[str]) -> dict:
                gold_groups = exact_answer_groups(make_eval_example(question), "list")
                predicted_items = parse_prediction_items(prediction_text or "", "list")

                matched_gold: set[int] = set()
                matched_pairs: list[tuple[str, int]] = []
                unsupported_items: list[str] = []

                for candidate in predicted_items:
                    matched = False
                    for gold_index, gold_group in enumerate(gold_groups):
                        if gold_index in matched_gold:
                            continue
                        if match_to_gold_group(candidate, gold_group):
                            matched_gold.add(gold_index)
                            matched_pairs.append((candidate, gold_index))
                            matched = True
                            break
                    if not matched:
                        unsupported_items.append(candidate)

                evidence_text = normalize_for_match("\\n".join(resources))
                group_rows: list[dict] = []
                missed_visible_count = 0
                missed_hidden_count = 0
                alias_like_visible_miss_count = 0

                for gold_index, gold_group in enumerate(gold_groups):
                    aliases = [clean_text(alias) for alias in gold_group if clean_text(alias)]
                    visible_aliases = []
                    for alias in aliases:
                        alias_norm = normalize_for_match(alias)
                        if alias_norm and alias_norm in evidence_text:
                            visible_aliases.append(alias)

                    matched_prediction = next(
                        (candidate for candidate, matched_index in matched_pairs if matched_index == gold_index),
                        None,
                    )
                    similar_unsupported_items = [
                        item for item in unsupported_items if looks_like_alias_mismatch(item, aliases)
                    ]

                    if matched_prediction is not None:
                        status = "matched"
                    elif visible_aliases:
                        status = "visible_missing"
                        missed_visible_count += 1
                        if similar_unsupported_items:
                            alias_like_visible_miss_count += 1
                    else:
                        status = "hidden"
                        missed_hidden_count += 1

                    group_rows.append(
                        {
                            "gold_index": gold_index,
                            "status": status,
                            "aliases": aliases,
                            "visible_aliases": visible_aliases,
                            "matched_prediction": matched_prediction,
                            "similar_unsupported_items": similar_unsupported_items,
                            "is_visible": bool(visible_aliases),
                        }
                    )

                matched_count = len(matched_gold)
                prediction_count = len(predicted_items)
                gold_count = len(gold_groups)
                unsupported_count = len(unsupported_items)
                precision = matched_count / prediction_count if prediction_count else 0.0
                recall = matched_count / gold_count if gold_count else 0.0
                f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

                under = matched_count < gold_count
                over = unsupported_count > 0
                if not under and not over:
                    question_category = "perfect"
                elif under and over:
                    question_category = "both"
                elif under:
                    question_category = "under_only"
                else:
                    question_category = "over_only"

                visible_gold_count = sum(1 for row in group_rows if row["is_visible"])
                likely_stopped_early = missed_visible_count > 0 and prediction_count < visible_gold_count
                likely_used_slots_on_extras = (
                    missed_visible_count > 0
                    and prediction_count >= visible_gold_count
                    and unsupported_count > 0
                )

                if missed_visible_count == 0 and missed_hidden_count > 0:
                    likely_reason = "evidence_limit"
                elif likely_stopped_early:
                    likely_reason = "undergeneration"
                elif alias_like_visible_miss_count > 0:
                    likely_reason = "accepted_form_mismatch"
                elif likely_used_slots_on_extras:
                    likely_reason = "extras_or_wrong_items"
                elif missed_visible_count > 0 and missed_hidden_count > 0:
                    likely_reason = "mixed_visible_and_hidden"
                elif missed_visible_count > 0:
                    likely_reason = "visible_but_missed"
                else:
                    likely_reason = "clean"

                return {
                    "predicted_items": predicted_items,
                    "unsupported_items": unsupported_items,
                    "matched_count": matched_count,
                    "prediction_count": prediction_count,
                    "gold_count": gold_count,
                    "unsupported_count": unsupported_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "question_category": question_category,
                    "visible_gold_count": visible_gold_count,
                    "missed_visible_count": missed_visible_count,
                    "missed_hidden_count": missed_hidden_count,
                    "alias_like_visible_miss_count": alias_like_visible_miss_count,
                    "likely_stopped_early": likely_stopped_early,
                    "likely_used_slots_on_extras": likely_used_slots_on_extras,
                    "likely_reason": likely_reason,
                    "gold_group_rows": group_rows,
                    "resources": resources,
                }


            def list_metrics_from_score_payload(payload: dict) -> dict:
                if not isinstance(payload, dict):
                    return {}
                return payload.get("by_type", {}).get("list", {}).get("metrics", {})


            def build_summary_row(model_bundle: dict) -> dict:
                frame = pd.DataFrame(model_bundle["question_records"])
                official_metrics = list_metrics_from_score_payload(model_bundle.get("official_scores", {}))
                if not official_metrics:
                    raise ValueError(
                        f"Missing official BioASQ Java scores for {model_bundle['label']}; "
                        "regenerate the evaluation before running this analysis."
                    )

                def category_count(name: str) -> int:
                    if frame.empty:
                        return 0
                    return int((frame["question_category"] == name).sum())

                avg_prediction_count = frame["prediction_count"].mean() if not frame.empty else None
                avg_gold_count = frame["gold_count"].mean() if not frame.empty else None

                return {
                    "model_label": model_bundle["label"],
                    "model_dir": str(model_bundle["model_dir"]),
                    "question_count": int(len(frame)),
                    "official_mean_f1": official_metrics.get("mean_f1"),
                    "official_mean_precision": official_metrics.get("mean_precision"),
                    "official_mean_recall": official_metrics.get("mean_recall"),
                    "avg_prediction_count": avg_prediction_count,
                    "avg_gold_count": avg_gold_count,
                    "perfect_q": category_count("perfect"),
                    "under_only_q": category_count("under_only"),
                    "over_only_q": category_count("over_only"),
                    "both_q": category_count("both"),
                    "questions_with_visible_miss": int((frame["missed_visible_count"] > 0).sum()) if not frame.empty else 0,
                    "questions_with_hidden_miss": int((frame["missed_hidden_count"] > 0).sum()) if not frame.empty else 0,
                    "total_visible_missing": int(frame["missed_visible_count"].sum()) if not frame.empty else 0,
                    "total_hidden_missing": int(frame["missed_hidden_count"].sum()) if not frame.empty else 0,
                    "likely_undergeneration_q": int(frame["likely_stopped_early"].sum()) if not frame.empty else 0,
                    "likely_extras_q": int(frame["likely_used_slots_on_extras"].sum()) if not frame.empty else 0,
                    "alias_like_visible_misses": int(frame["alias_like_visible_miss_count"].sum()) if not frame.empty else 0,
                }


            def load_eval_artifact(label: str, path_like: str | Path, questions: dict[str, dict]) -> dict:
                model_dir, eval_root = resolve_eval_model_dir(path_like)
                manifest_path = eval_root / "manifest.json"
                manifest = load_json(manifest_path) if manifest_path.exists() else {}

                predictions = load_json(model_dir / "predictions.json")
                scores_path = model_dir / "scores.json"
                official_scores_path = model_dir / "official_bioasq" / "official_scores.json"
                scores = load_json(scores_path) if scores_path.exists() else {}
                official_scores = load_json(official_scores_path) if official_scores_path.exists() else {}

                question_records: list[dict] = []
                gold_group_records: list[dict] = []
                unsupported_records: list[dict] = []
                analysis_by_question: dict[str, dict] = {}
                row_by_question: dict[str, dict] = {}

                for row in predictions:
                    question_id = clean_text(row.get("question_id", ""))
                    if question_id not in questions:
                        continue

                    question = questions[question_id]
                    resources = build_question_resources(question, manifest)
                    analysis = analyze_prediction(question, clean_text(row.get("prediction", "")), resources)

                    analysis_by_question[question_id] = analysis
                    row_by_question[question_id] = row

                    question_records.append(
                        {
                            "model_label": label,
                            "question_id": question_id,
                            "question": clean_text(question.get("body", "")),
                            "f1": float(row.get("score", {}).get("f1", analysis["f1"])),
                            "precision": float(row.get("score", {}).get("precision", analysis["precision"])),
                            "recall": float(row.get("score", {}).get("recall", analysis["recall"])),
                            "prediction_count": analysis["prediction_count"],
                            "gold_count": analysis["gold_count"],
                            "matched_count": analysis["matched_count"],
                            "unsupported_count": analysis["unsupported_count"],
                            "visible_gold_count": analysis["visible_gold_count"],
                            "missed_visible_count": analysis["missed_visible_count"],
                            "missed_hidden_count": analysis["missed_hidden_count"],
                            "alias_like_visible_miss_count": analysis["alias_like_visible_miss_count"],
                            "question_category": analysis["question_category"],
                            "likely_stopped_early": analysis["likely_stopped_early"],
                            "likely_used_slots_on_extras": analysis["likely_used_slots_on_extras"],
                            "likely_reason": analysis["likely_reason"],
                            "prediction": clean_text(row.get("prediction", "")),
                            "gold_output": clean_text(row.get("gold_output", "")),
                        }
                    )

                    for group_row in analysis["gold_group_rows"]:
                        gold_group_records.append(
                            {
                                "model_label": label,
                                "question_id": question_id,
                                "question": clean_text(question.get("body", "")),
                                **group_row,
                            }
                        )

                    for item in analysis["unsupported_items"]:
                        unsupported_records.append(
                            {
                                "model_label": label,
                                "question_id": question_id,
                                "question": clean_text(question.get("body", "")),
                                "unsupported_prediction": item,
                            }
                        )

                return {
                    "label": label,
                    "model_dir": model_dir,
                    "eval_root": eval_root,
                    "manifest": manifest,
                    "scores": scores,
                    "official_scores": official_scores,
                    "question_records": question_records,
                    "gold_group_records": gold_group_records,
                    "unsupported_records": unsupported_records,
                    "analysis_by_question": analysis_by_question,
                    "row_by_question": row_by_question,
                }
            """
        ),
        code_cell(
            """
            QUESTIONS = load_list_questions(TEST_FILES)
            print(f"Loaded {len(QUESTIONS)} list questions from the BioASQ test files.")

            MODEL_BUNDLES: dict[str, dict] = {}
            question_records = []
            gold_group_records = []
            unsupported_records = []

            for label, path in MODEL_RUNS.items():
                bundle = load_eval_artifact(label, path, QUESTIONS)
                MODEL_BUNDLES[label] = bundle
                question_records.extend(bundle["question_records"])
                gold_group_records.extend(bundle["gold_group_records"])
                unsupported_records.extend(bundle["unsupported_records"])

            QUESTION_DF = pd.DataFrame(question_records)
            GOLD_GROUP_DF = pd.DataFrame(gold_group_records)
            UNSUPPORTED_DF = pd.DataFrame(unsupported_records)
            SUMMARY_DF = pd.DataFrame([build_summary_row(bundle) for bundle in MODEL_BUNDLES.values()])

            numeric_summary_columns = [
                "question_count",
                "official_mean_f1",
                "official_mean_precision",
                "official_mean_recall",
                "official_mean_precision",
                "official_mean_recall",
                "avg_prediction_count",
                "avg_gold_count",
                "perfect_q",
                "under_only_q",
                "over_only_q",
                "both_q",
                "questions_with_visible_miss",
                "questions_with_hidden_miss",
                "total_visible_missing",
                "total_hidden_missing",
                "likely_undergeneration_q",
                "likely_extras_q",
                "alias_like_visible_misses",
            ]
            for column in numeric_summary_columns:
                if column in SUMMARY_DF.columns:
                    SUMMARY_DF[column] = pd.to_numeric(SUMMARY_DF[column], errors="coerce")

            summary_columns = [
                "model_label",
                "official_mean_f1",
                "official_mean_precision",
                "official_mean_recall",
                "avg_prediction_count",
                "avg_gold_count",
                "perfect_q",
                "under_only_q",
                "over_only_q",
                "both_q",
                "total_visible_missing",
                "total_hidden_missing",
                "likely_undergeneration_q",
                "likely_extras_q",
                "alias_like_visible_misses",
            ]

            display(SUMMARY_DF[summary_columns].sort_values("official_mean_f1", ascending=False))
            """
        ),
        code_cell(
            """
            if SUMMARY_DF.empty:
                raise ValueError("No model runs were loaded. Update MODEL_RUNS in the setup cell.")

            plot_columns = [
                "official_mean_f1",
                "official_mean_precision",
                "official_mean_recall",
                "avg_prediction_count",
                "avg_gold_count",
                "perfect_q",
                "under_only_q",
                "over_only_q",
                "both_q",
                "total_visible_missing",
                "total_hidden_missing",
            ]
            plot_df = SUMMARY_DF.copy()
            for column in plot_columns:
                if column in plot_df.columns:
                    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0)
            plot_df = plot_df.sort_values("official_mean_f1", ascending=False).reset_index(drop=True)
            labels = plot_df["model_label"].tolist()
            x = range(len(labels))

            fig, axes = plt.subplots(2, 2, figsize=(16, 11))

            axes[0, 0].bar(x, plot_df["official_mean_f1"], color="#4c78a8", label="official")
            axes[0, 0].bar(x, plot_df["official_mean_precision"], color="#72b7b2", alpha=0.45, label="official precision")
            axes[0, 0].bar(x, plot_df["official_mean_recall"], color="#f58518", alpha=0.45, label="official recall")
            axes[0, 0].set_title("Score Summary")
            axes[0, 0].set_xticks(list(x), labels, rotation=35, ha="right")
            axes[0, 0].set_ylim(0, 1.0)
            axes[0, 0].legend()

            axes[0, 1].bar(x, plot_df["avg_prediction_count"], color="#54a24b", label="avg prediction count")
            axes[0, 1].plot(x, plot_df["avg_gold_count"], color="#e45756", marker="o", linewidth=2, label="avg gold count")
            axes[0, 1].set_title("Average Answer Length")
            axes[0, 1].set_xticks(list(x), labels, rotation=35, ha="right")
            axes[0, 1].legend()

            category_columns = ["perfect_q", "under_only_q", "over_only_q", "both_q"]
            category_colors = {
                "perfect_q": "#4c78a8",
                "under_only_q": "#f58518",
                "over_only_q": "#e45756",
                "both_q": "#72b7b2",
            }
            bottom = [0] * len(plot_df)
            for column in category_columns:
                axes[1, 0].bar(
                    x,
                    plot_df[column],
                    bottom=bottom,
                    color=category_colors[column],
                    label=column.replace("_q", ""),
                )
                bottom = [current + value for current, value in zip(bottom, plot_df[column])]
            axes[1, 0].set_title("Question-Level Error Profile")
            axes[1, 0].set_xticks(list(x), labels, rotation=35, ha="right")
            axes[1, 0].legend()

            axes[1, 1].bar(x, plot_df["total_visible_missing"], color="#b279a2", label="visible missing")
            axes[1, 1].bar(
                x,
                plot_df["total_hidden_missing"],
                bottom=plot_df["total_visible_missing"],
                color="#9d755d",
                label="hidden missing",
            )
            axes[1, 1].set_title("Missed Gold Groups")
            axes[1, 1].set_xticks(list(x), labels, rotation=35, ha="right")
            axes[1, 1].legend()

            fig.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            def worst_questions(model_label: str, n: int = 12) -> pd.DataFrame:
                frame = QUESTION_DF[QUESTION_DF["model_label"] == model_label].copy()
                sort_columns = ["f1", "missed_visible_count", "unsupported_count"]
                ascending = [True, False, False]
                return frame.sort_values(sort_columns, ascending=ascending).head(n)


            def disagreement_table() -> pd.DataFrame:
                if QUESTION_DF["model_label"].nunique() < 2:
                    return pd.DataFrame()
                pivot = QUESTION_DF.pivot(index="question_id", columns="model_label", values="f1")
                if pivot.empty:
                    return pd.DataFrame()
                summary = pd.DataFrame(index=pivot.index)
                summary["best_f1"] = pivot.max(axis=1)
                summary["worst_f1"] = pivot.min(axis=1)
                summary["f1_gap"] = summary["best_f1"] - summary["worst_f1"]
                summary["best_model"] = pivot.idxmax(axis=1)
                summary["worst_model"] = pivot.idxmin(axis=1)
                summary["question"] = summary.index.map(lambda qid: clean_text(QUESTIONS[qid].get("body", "")))
                return summary.sort_values(["f1_gap", "best_f1"], ascending=[False, False]).reset_index()


            display(Markdown("## Worst Questions Per Model"))
            for label in SUMMARY_DF.sort_values("official_mean_f1", ascending=False)["model_label"]:
                display(Markdown(f"### {label}"))
                display(
                    worst_questions(label)[
                        [
                            "question_id",
                            "question",
                            "f1",
                            "precision",
                            "recall",
                            "prediction_count",
                            "gold_count",
                            "missed_visible_count",
                            "missed_hidden_count",
                            "unsupported_count",
                            "likely_reason",
                        ]
                    ]
                )

            DISAGREEMENT_DF = disagreement_table()
            if not DISAGREEMENT_DF.empty:
                display(Markdown("## Largest Cross-Model F1 Disagreements"))
                display(DISAGREEMENT_DF.head(20))
            """
        ),
        code_cell(
            """
            def inspect_question(
                question_id: str,
                model_labels: list[str] | None = None,
                show_resources: bool = True,
                resource_chars: int = 800,
            ) -> None:
                qid = clean_text(question_id)
                if qid not in QUESTIONS:
                    raise KeyError(f"Question id not found: {question_id}")

                question = QUESTIONS[qid]
                model_labels = model_labels or list(MODEL_BUNDLES.keys())

                display(Markdown(f"## {qid}"))
                display(Markdown(f"**Question:** {clean_text(question.get('body', ''))}"))

                gold_groups = exact_answer_groups(make_eval_example(question), "list")
                gold_table = pd.DataFrame(
                    {
                        "gold_index": list(range(len(gold_groups))),
                        "accepted_aliases": [[clean_text(alias) for alias in group if clean_text(alias)] for group in gold_groups],
                    }
                )
                display(Markdown("### Gold Accepted Answer Groups"))
                display(gold_table)

                for label in model_labels:
                    bundle = MODEL_BUNDLES[label]
                    analysis = bundle["analysis_by_question"].get(qid)
                    row = bundle["row_by_question"].get(qid)
                    if analysis is None or row is None:
                        continue

                    display(Markdown(f"### {label}"))
                    display(
                        pd.DataFrame(
                            {
                                "metric": [
                                    "f1",
                                    "precision",
                                    "recall",
                                    "prediction_count",
                                    "gold_count",
                                    "visible_gold_count",
                                    "missed_visible_count",
                                    "missed_hidden_count",
                                    "unsupported_count",
                                    "question_category",
                                    "likely_reason",
                                ],
                                "value": [
                                    analysis["f1"],
                                    analysis["precision"],
                                    analysis["recall"],
                                    analysis["prediction_count"],
                                    analysis["gold_count"],
                                    analysis["visible_gold_count"],
                                    analysis["missed_visible_count"],
                                    analysis["missed_hidden_count"],
                                    analysis["unsupported_count"],
                                    analysis["question_category"],
                                    analysis["likely_reason"],
                                ],
                            }
                        )
                    )

                    display(Markdown("**Prediction**"))
                    print(clean_text(row.get("prediction", "")) or "[empty]")

                    group_frame = pd.DataFrame(analysis["gold_group_rows"])[
                        [
                            "gold_index",
                            "status",
                            "aliases",
                            "visible_aliases",
                            "matched_prediction",
                            "similar_unsupported_items",
                        ]
                    ]
                    display(Markdown("**Gold-group status**"))
                    display(group_frame)

                    if analysis["unsupported_items"]:
                        display(Markdown("**Unsupported predicted items**"))
                        display(pd.DataFrame({"unsupported_prediction": analysis["unsupported_items"]}))

                    if show_resources:
                        display(Markdown("**Prompt resources used during evaluation**"))
                        for index, resource in enumerate(analysis["resources"], start=1):
                            preview = resource[:resource_chars]
                            if len(resource) > resource_chars:
                                preview += "..."
                            display(Markdown(f"Resource {index}"))
                            print(preview)


            if not QUESTION_DF.empty:
                if "DISAGREEMENT_DF" in globals() and not DISAGREEMENT_DF.empty:
                    EXAMPLE_QUESTION_ID = DISAGREEMENT_DF.iloc[0]["question_id"]
                else:
                    EXAMPLE_QUESTION_ID = QUESTION_DF.sort_values(
                        ["f1", "missed_visible_count", "unsupported_count"],
                        ascending=[True, False, False],
                    ).iloc[0]["question_id"]
                print(f"Example question id for inspection: {EXAMPLE_QUESTION_ID}")
                inspect_question(EXAMPLE_QUESTION_ID, show_resources=False)
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    task_root = Path(__file__).resolve().parents[2]
    notebook_path = task_root / "notebooks" / "model_evaluation_mistake_analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(build_notebook(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    main()
