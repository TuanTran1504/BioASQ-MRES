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
            # Cardinality Shortcut Study: Exposure And Pair-Quality Report

            This notebook reports the **exposure stage** of the Cardinality Shortcut Study for the current
            `ordinary_pairs.jsonl` dataset. The aim is to quantify whether the ordinary F1-ranked preference pairs
            expose the optimiser to a systematic direction cue in answer cardinality, and whether the two directional
            pair groups also differ in difficulty or construction profile.

            The analysis is intentionally split into two complementary views:

            - **Pair-micro exposure:** how often the chosen answer is shorter versus longer across retained directional pairs.
            - **Question-macro exposure:** whether the same directional pattern remains after each question contributes equally.
            - **Pair-quality audit:** whether shorter- and longer-preferred pairs also differ in F1 margin, edit size,
              token-length difference, gold cardinality, candidate quality, evidence support, or generation provenance.

            The notebook is designed to support a research-style write-up by:

            - computing the directional imbalance statistics already defined in the draft,
            - adding **question-clustered bootstrap confidence intervals**,
            - visualising whether the shorter-preferred dominance is broad or concentrated,
            - auditing whether directional exposure is entangled with pair-quality differences,
            - producing a concise narrative summary suitable for the current draft.

            Key outputs:

            - pair-micro shorter-preferred proportion and `D_pair`
            - question-macro shorter-preferred proportion and `D_question`
            - counts of shorter-dominant, longer-dominant, tied, and no-directional questions
            - pair-quality comparison tables and plots
            - publication-style figures for the exposure and pair-quality sections
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            import json
            import re
            import statistics
            import sys
            from collections import Counter, defaultdict
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
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

            NOTEBOOK_TITLE = "Cardinality Shortcut Study: Exposure And Pair-Quality Report"
            ORDINARY_PAIR_PATH = (
                TASK_ROOT
                / "Artifacts"
                / "Cardinality_Shortcut_Study"
                / "pairs"
                / "original_full_resources_all_resources"
                / "ordinary_pairs.jsonl"
            )
            SUMMARY_PATH = (
                TASK_ROOT
                / "Artifacts"
                / "Cardinality_Shortcut_Study"
                / "pairs"
                / "original_full_resources_all_resources"
                / "pair_construction_summary.json"
            )
            STANDARDIZED_CANDIDATE_PATH = (
                TASK_ROOT
                / "data"
                / "Cardinality_Shortcut_Study"
                / "standardized_candidates"
                / "original_full_resources_all_resources"
                / "standardized_candidates_all.jsonl"
            )
            CANDIDATE_BANK_PATH = (
                TASK_ROOT
                / "Artifacts"
                / "Cardinality_Shortcut_Study"
                / "candidate_banks"
                / "original_full_resources_all_resources"
                / "original-full-resources"
                / "candidate_bank.jsonl"
            )
            BOOTSTRAP_REPS = 10_000
            BOOTSTRAP_SEED = 3407

            SHORTER_COLOR = "#B54708"
            LONGER_COLOR = "#005F73"
            EQUAL_COLOR = "#6C757D"
            NEUTRAL_COLOR = "#8D99AE"

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.options.display.max_colwidth = 180
            pd.options.display.float_format = lambda value: f"{value:0.4f}"

            print(f"TASK_ROOT = {TASK_ROOT}")
            print(f"ORDINARY_PAIR_PATH exists = {ORDINARY_PAIR_PATH.exists()}")
            print(f"SUMMARY_PATH exists = {SUMMARY_PATH.exists()}")
            print(f"STANDARDIZED_CANDIDATE_PATH exists = {STANDARDIZED_CANDIDATE_PATH.exists()}")
            print(f"CANDIDATE_BANK_PATH exists = {CANDIDATE_BANK_PATH.exists()}")
            print(f"BOOTSTRAP_REPS = {BOOTSTRAP_REPS:,}")
            """
        ),
        code_cell(
            """
            def load_jsonl(path: Path) -> list[dict]:
                rows: list[dict] = []
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
                return rows


            ordinary_rows = load_jsonl(ORDINARY_PAIR_PATH)
            pair_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

            ordinary_df = pd.DataFrame(ordinary_rows)
            ordinary_df["question_id"] = ordinary_df["question_id"].astype(str)
            ordinary_df["direction_label"] = ordinary_df["direction_label"].astype(str)
            ordinary_df["delta_f1"] = ordinary_df["delta_f1"].astype(float)
            ordinary_df["semantic_set_edit_distance"] = ordinary_df["semantic_set_edit_distance"].astype(int)
            ordinary_df["absolute_entity_gap"] = ordinary_df["absolute_entity_gap"].astype(int)
            ordinary_df["absolute_token_length_gap"] = ordinary_df["absolute_token_length_gap"].astype(float)
            ordinary_df["gold_entity_count"] = ordinary_df["gold_entity_count"].astype(int)
            ordinary_df["chosen_f1"] = ordinary_df["chosen_f1"].astype(float)
            ordinary_df["rejected_f1"] = ordinary_df["rejected_f1"].astype(float)
            ordinary_df["chosen_sample_id"] = ordinary_df["chosen_sample_id"].astype(int)
            ordinary_df["rejected_sample_id"] = ordinary_df["rejected_sample_id"].astype(int)
            ordinary_df["candidate_label_source"] = ordinary_df["candidate_label_source"].astype(str)

            ordinary_df.head(3)
            """
        ),
        code_cell(
            """
            direction_counts = ordinary_df["direction_label"].value_counts().to_dict()
            directional_df = ordinary_df.loc[
                ordinary_df["direction_label"].isin(["shorter_preferred", "longer_preferred"])
            ].copy()

            question_rows = []
            for question_id, question_df in ordinary_df.groupby("question_id", sort=True):
                n_short = int((question_df["direction_label"] == "shorter_preferred").sum())
                n_long = int((question_df["direction_label"] == "longer_preferred").sum())
                n_equal = int((question_df["direction_label"] == "equal_preferred").sum())
                directional_count = n_short + n_long
                d_q = np.nan if directional_count == 0 else (n_short - n_long) / directional_count
                shorter_prop_q = np.nan if directional_count == 0 else n_short / directional_count
                if directional_count == 0:
                    dominance = "no_directional_pairs"
                elif d_q > 0:
                    dominance = "shorter_dominant"
                elif d_q < 0:
                    dominance = "longer_dominant"
                else:
                    dominance = "tied"

                question_rows.append(
                    {
                        "question_id": question_id,
                        "pair_count_total": int(len(question_df)),
                        "pair_count_directional": int(directional_count),
                        "pair_count_equal": int(n_equal),
                        "n_short": int(n_short),
                        "n_long": int(n_long),
                        "n_equal": int(n_equal),
                        "d_q": float(d_q) if not np.isnan(d_q) else np.nan,
                        "shorter_prop_q": float(shorter_prop_q) if not np.isnan(shorter_prop_q) else np.nan,
                        "dominance_label": dominance,
                        "mean_delta_f1": float(question_df["delta_f1"].mean()),
                    }
                )

            question_df = pd.DataFrame(question_rows)
            question_df_directional = question_df.loc[
                question_df["pair_count_directional"] > 0
            ].copy()

            short_count = int((ordinary_df["direction_label"] == "shorter_preferred").sum())
            long_count = int((ordinary_df["direction_label"] == "longer_preferred").sum())
            equal_count = int((ordinary_df["direction_label"] == "equal_preferred").sum())

            exposure_metrics = {
                "pair_micro_shorter_prop_directional": short_count / (short_count + long_count),
                "pair_micro_D_pair": (short_count - long_count) / (short_count + long_count),
                "pair_micro_shorter_prop_all_pairs": short_count / len(ordinary_df),
                "question_macro_shorter_prop": float(question_df_directional["shorter_prop_q"].mean()),
                "question_macro_D_question": float(question_df_directional["d_q"].mean()),
                "question_count_all": int(len(question_df)),
                "question_count_directional": int(len(question_df_directional)),
                "question_count_no_directional": int((question_df["pair_count_directional"] == 0).sum()),
                "shorter_dominant_questions": int((question_df["dominance_label"] == "shorter_dominant").sum()),
                "longer_dominant_questions": int((question_df["dominance_label"] == "longer_dominant").sum()),
                "tied_questions": int((question_df["dominance_label"] == "tied").sum()),
            }

            exposure_metrics
            """
        ),
        code_cell(
            """
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            question_records = question_df.to_dict(orient="records")


            def bootstrap_exposure(records: list[dict], reps: int) -> pd.DataFrame:
                rows = []
                for _ in range(reps):
                    sample_indices = rng.integers(0, len(records), len(records))
                    sample = [records[index] for index in sample_indices]

                    sample_short = sum(record["n_short"] for record in sample)
                    sample_long = sum(record["n_long"] for record in sample)
                    pair_micro_shorter = sample_short / (sample_short + sample_long)
                    d_pair = (sample_short - sample_long) / (sample_short + sample_long)

                    directional_records = [record for record in sample if record["pair_count_directional"] > 0]
                    shorter_prop_macro = float(np.mean([record["shorter_prop_q"] for record in directional_records]))
                    d_question = float(np.mean([record["d_q"] for record in directional_records]))

                    rows.append(
                        {
                            "pair_micro_shorter_prop_directional": pair_micro_shorter,
                            "pair_micro_D_pair": d_pair,
                            "question_macro_shorter_prop": shorter_prop_macro,
                            "question_macro_D_question": d_question,
                        }
                    )
                return pd.DataFrame(rows)


            bootstrap_df = bootstrap_exposure(question_records, BOOTSTRAP_REPS)


            def percentile_ci(series: pd.Series) -> tuple[float, float]:
                return tuple(np.quantile(series.to_numpy(), [0.025, 0.975]))


            ci_map = {
                metric: percentile_ci(bootstrap_df[metric])
                for metric in [
                    "pair_micro_shorter_prop_directional",
                    "pair_micro_D_pair",
                    "question_macro_shorter_prop",
                    "question_macro_D_question",
                ]
            }

            ci_map
            """
        ),
        code_cell(
            """
            def dominance_band(value: float) -> str:
                magnitude = abs(value)
                if magnitude < 0.10:
                    return "weak_or_none"
                if magnitude < 0.25:
                    return "moderate"
                return "strong"


            summary_table = pd.DataFrame(
                [
                    {
                        "metric": "Pair-micro shorter-preferred proportion",
                        "estimate": exposure_metrics["pair_micro_shorter_prop_directional"],
                        "ci_low": ci_map["pair_micro_shorter_prop_directional"][0],
                        "ci_high": ci_map["pair_micro_shorter_prop_directional"][1],
                        "interpretation": "Conditional on directional pairs only",
                    },
                    {
                        "metric": "Pair-micro directional imbalance (D_pair)",
                        "estimate": exposure_metrics["pair_micro_D_pair"],
                        "ci_low": ci_map["pair_micro_D_pair"][0],
                        "ci_high": ci_map["pair_micro_D_pair"][1],
                        "interpretation": f"{dominance_band(exposure_metrics['pair_micro_D_pair'])} dominance",
                    },
                    {
                        "metric": "Question-macro shorter-preferred proportion",
                        "estimate": exposure_metrics["question_macro_shorter_prop"],
                        "ci_low": ci_map["question_macro_shorter_prop"][0],
                        "ci_high": ci_map["question_macro_shorter_prop"][1],
                        "interpretation": "Each directional question contributes equally",
                    },
                    {
                        "metric": "Question-macro directional imbalance (D_question)",
                        "estimate": exposure_metrics["question_macro_D_question"],
                        "ci_low": ci_map["question_macro_D_question"][0],
                        "ci_high": ci_map["question_macro_D_question"][1],
                        "interpretation": f"{dominance_band(exposure_metrics['question_macro_D_question'])} dominance",
                    },
                ]
            )

            dominance_count_table = pd.DataFrame(
                [
                    {
                        "category": "Shorter-dominant questions",
                        "count": exposure_metrics["shorter_dominant_questions"],
                    },
                    {
                        "category": "Longer-dominant questions",
                        "count": exposure_metrics["longer_dominant_questions"],
                    },
                    {
                        "category": "Tied directional questions",
                        "count": exposure_metrics["tied_questions"],
                    },
                    {
                        "category": "No directional pairs",
                        "count": exposure_metrics["question_count_no_directional"],
                    },
                ]
            )

            display(Markdown("## Exposure Summary Table"))
            display(summary_table)

            display(Markdown("## Question-Level Dominance Counts"))
            display(dominance_count_table)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

            pair_view = pd.DataFrame(
                {
                    "direction": ["Shorter preferred", "Longer preferred", "Equal cardinality"],
                    "count": [short_count, long_count, equal_count],
                    "proportion": [short_count / len(ordinary_df), long_count / len(ordinary_df), equal_count / len(ordinary_df)],
                    "color": [SHORTER_COLOR, LONGER_COLOR, EQUAL_COLOR],
                }
            )

            axes[0].bar(pair_view["direction"], pair_view["count"], color=pair_view["color"], width=0.65)
            for idx, row in pair_view.iterrows():
                axes[0].text(
                    idx,
                    row["count"] + 8,
                    f"{row['count']}\\n({row['proportion']*100:0.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
            axes[0].set_title("Observed Direction Counts In The Ordinary Pair Set", fontsize=15, weight="bold")
            axes[0].set_ylabel("Retained pair count", fontsize=12)
            axes[0].tick_params(axis="x", rotation=10, labelsize=11)
            axes[0].tick_params(axis="y", labelsize=11)

            directional_view = pd.DataFrame(
                {
                    "metric": ["Pair-micro", "Question-macro"],
                    "shorter_prop": [
                        exposure_metrics["pair_micro_shorter_prop_directional"],
                        exposure_metrics["question_macro_shorter_prop"],
                    ],
                    "ci_low": [
                        ci_map["pair_micro_shorter_prop_directional"][0],
                        ci_map["question_macro_shorter_prop"][0],
                    ],
                    "ci_high": [
                        ci_map["pair_micro_shorter_prop_directional"][1],
                        ci_map["question_macro_shorter_prop"][1],
                    ],
                }
            )
            errors = np.vstack(
                [
                    directional_view["shorter_prop"] - directional_view["ci_low"],
                    directional_view["ci_high"] - directional_view["shorter_prop"],
                ]
            )
            axes[1].bar(
                directional_view["metric"],
                directional_view["shorter_prop"],
                color=[SHORTER_COLOR, "#9A3412"],
                width=0.58,
            )
            axes[1].errorbar(
                directional_view["metric"],
                directional_view["shorter_prop"],
                yerr=errors,
                fmt="none",
                ecolor="#111827",
                elinewidth=1.8,
                capsize=5,
            )
            axes[1].axhline(0.5, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.5, label="No directional preference")
            for idx, row in directional_view.iterrows():
                axes[1].text(
                    idx,
                    row["shorter_prop"] + 0.025,
                    f"{row['shorter_prop']*100:0.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_ylabel("Shorter-preferred proportion", fontsize=12)
            axes[1].set_title("Shorter-Preferred Exposure With Clustered 95% CIs", fontsize=15, weight="bold")
            axes[1].tick_params(axis="x", labelsize=11)
            axes[1].tick_params(axis="y", labelsize=11)
            axes[1].legend(frameon=False, loc="lower right", fontsize=11)

            fig.suptitle("Exposure Imbalance In The Ordinary Training Pairs", fontsize=18, weight="bold", y=0.98)
            fig.text(
                0.5,
                0.93,
                "Research question: Does the ordinary pair dataset expose DPO to a shorter-preferred directional cue?",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#374151",
            )
            fig.text(
                0.5,
                0.045,
                "Claim tested here: the shorter-preferred cue should remain visible both at the pair level and after equalising question contribution.",
                ha="center",
                va="center",
                fontsize=11.5,
                color="#4B5563",
            )
            fig.tight_layout(rect=[0.03, 0.08, 0.98, 0.88])
            plt.show()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

            axes[0].hist(
                question_df_directional["d_q"],
                bins=np.linspace(-1.0, 1.0, 17),
                color=SHORTER_COLOR,
                alpha=0.78,
                edgecolor="white",
            )
            axes[0].axvline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.5)
            axes[0].axvline(exposure_metrics["question_macro_D_question"], color="#111827", linewidth=2.0)
            axes[0].axvspan(-0.10, 0.10, color="#E5E7EB", alpha=0.4, label="Weak / no dominance band")
            axes[0].set_title("Distribution Of Question-Level Directional Imbalance (D_q)", fontsize=15, weight="bold")
            axes[0].set_xlabel("D_q = (N_short - N_long) / (N_short + N_long)", fontsize=12)
            axes[0].set_ylabel("Question count", fontsize=12)
            axes[0].tick_params(axis="both", labelsize=11)
            axes[0].legend(frameon=False, loc="upper left", fontsize=11)

            category_order = [
                "shorter_dominant",
                "longer_dominant",
                "tied",
                "no_directional_pairs",
            ]
            category_labels = [
                "Shorter dominant",
                "Longer dominant",
                "Tied",
                "No directional pairs",
            ]
            category_counts = [
                int((question_df["dominance_label"] == label).sum())
                for label in category_order
            ]
            category_colors = [SHORTER_COLOR, LONGER_COLOR, EQUAL_COLOR, NEUTRAL_COLOR]

            axes[1].bar(category_labels, category_counts, color=category_colors, width=0.62)
            for idx, count in enumerate(category_counts):
                axes[1].text(idx, count + 1.5, str(count), ha="center", va="bottom", fontsize=11)
            axes[1].set_title("Question-Level Dominance Categories", fontsize=15, weight="bold")
            axes[1].set_ylabel("Question count", fontsize=12)
            axes[1].tick_params(axis="x", rotation=10, labelsize=11)
            axes[1].tick_params(axis="y", labelsize=11)

            fig.suptitle("Breadth Of The Shorter-Preferred Exposure Pattern", fontsize=18, weight="bold", y=0.98)
            fig.text(
                0.5,
                0.93,
                "Research question: Is the shorter-preferred cue broad across many questions rather than driven by a small subset?",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#374151",
            )
            fig.text(
                0.5,
                0.045,
                "Claim tested here: if exposure is genuinely broad, many questions should lean shorter and the D_q distribution should sit mostly on the positive side.",
                ha="center",
                va="center",
                fontsize=11.5,
                color="#4B5563",
            )
            fig.tight_layout(rect=[0.03, 0.08, 0.98, 0.88])
            plt.show()
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(10.5, 7.4))

            scatter_df = question_df_directional.copy()
            scatter_df["dominance_color"] = np.where(
                scatter_df["d_q"] > 0,
                SHORTER_COLOR,
                np.where(scatter_df["d_q"] < 0, LONGER_COLOR, EQUAL_COLOR),
            )

            ax.scatter(
                scatter_df["pair_count_directional"],
                scatter_df["d_q"],
                c=scatter_df["dominance_color"],
                alpha=0.72,
                s=46,
                edgecolors="white",
                linewidths=0.6,
            )
            ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.4)
            ax.axhline(exposure_metrics["question_macro_D_question"], color="#111827", linewidth=1.8, label="Question-macro mean")
            ax.set_title("Question-Level Imbalance Versus Directional Pair Count", fontsize=16, weight="bold")
            ax.set_xlabel("Directional pairs retained for the question", fontsize=12)
            ax.set_ylabel("Question-level imbalance (D_q)", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.legend(frameon=False, loc="lower right", fontsize=11)

            fig.suptitle("Stability Of The Exposure Pattern Across Questions", fontsize=18, weight="bold", y=0.98)
            fig.text(
                0.5,
                0.93,
                "Research question: Does the shorter-preferred tendency persist even when questions have different numbers of retained directional pairs?",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#374151",
            )
            fig.text(
                0.5,
                0.045,
                "Claim tested here: the positive question-macro mean should not be an artifact of only the highest-combinatorics questions.",
                ha="center",
                va="center",
                fontsize=11.5,
                color="#4B5563",
            )

            plt.tight_layout(rect=[0.03, 0.08, 0.98, 0.88])
            plt.show()
            """
        ),
        md_cell(
            """
            ## Pair-Quality Distributions

            The exposure imbalance alone does not tell us whether the directional groups are otherwise comparable.
            This section compares **shorter-preferred** and **longer-preferred** directional pairs on the following
            axes:

            - F1 margin
            - entity edit size
            - token-length difference
            - gold cardinality
            - candidate F1
            - evidence support
            - candidate-generation provenance

            For evidence support, the current pair file does not carry a precomputed support label. The notebook
            therefore augments the retained responses with a **surface-match evidence-support proxy** by checking
            whether each predicted entity string appears in the original retrieved evidence snippets for that response.
            This proxy should be treated as descriptive rather than definitive.
            """
        ),
        code_cell(
            """
            relevant_response_ids = set(directional_df["chosen_response_id"]).union(
                set(directional_df["rejected_response_id"])
            )


            def normalize_support_text(text: object) -> str:
                return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


            def flatten_evidence_text(evidence_items: object) -> str:
                if not isinstance(evidence_items, list):
                    return ""
                chunks: list[str] = []
                for item in evidence_items:
                    if isinstance(item, dict):
                        raw_text = item.get("text") or item.get("snippet") or item.get("content") or ""
                    else:
                        raw_text = str(item)
                    normalized = normalize_support_text(raw_text)
                    if normalized:
                        chunks.append(normalized)
                return " ".join(chunks)


            def surface_support_stats(predicted_entities: object, evidence_items: object) -> tuple[float, int, int]:
                entities = predicted_entities if isinstance(predicted_entities, list) else []
                total = len(entities)
                if total == 0:
                    return (np.nan, 0, 0)

                evidence_text = flatten_evidence_text(evidence_items)
                supported = 0
                for entity in entities:
                    if isinstance(entity, dict):
                        surface = entity.get("surface") or entity.get("text") or ""
                    else:
                        surface = str(entity)
                    normalized_surface = normalize_support_text(surface)
                    if normalized_surface and normalized_surface in evidence_text:
                        supported += 1
                return (supported / total, supported, total)


            standardized_rows = [
                row
                for row in load_jsonl(STANDARDIZED_CANDIDATE_PATH)
                if row.get("response_id") in relevant_response_ids
            ]
            candidate_bank_rows = [
                row
                for row in load_jsonl(CANDIDATE_BANK_PATH)
                if row.get("response_id") in relevant_response_ids
            ]

            standardized_candidate_df = pd.DataFrame(standardized_rows)
            candidate_bank_df = pd.DataFrame(candidate_bank_rows)

            response_quality_df = standardized_candidate_df[
                [
                    "response_id",
                    "predicted_entities",
                    "invalid_addition_rate",
                    "valid_omission_rate",
                    "f1",
                    "generator_checkpoint",
                    "sample_id",
                ]
            ].rename(
                columns={
                    "f1": "standardized_candidate_f1",
                    "generator_checkpoint": "standardized_generator_checkpoint",
                    "sample_id": "standardized_sample_id",
                }
            )

            response_quality_df = response_quality_df.merge(
                candidate_bank_df[["response_id", "evidence"]],
                on="response_id",
                how="left",
            )

            support_triplets = response_quality_df.apply(
                lambda row: surface_support_stats(row["predicted_entities"], row["evidence"]),
                axis=1,
            )
            response_quality_df["surface_evidence_support_rate"] = [triplet[0] for triplet in support_triplets]
            response_quality_df["surface_supported_prediction_count"] = [triplet[1] for triplet in support_triplets]
            response_quality_df["surface_prediction_count"] = [triplet[2] for triplet in support_triplets]

            pair_quality_df = directional_df.copy()
            for prefix, key_column in [("chosen", "chosen_response_id"), ("rejected", "rejected_response_id")]:
                prefixed_response_quality_df = response_quality_df.rename(
                    columns={
                        column: f"{prefix}_{column}"
                        for column in response_quality_df.columns
                    }
                )
                pair_quality_df = pair_quality_df.merge(
                    prefixed_response_quality_df,
                    left_on=key_column,
                    right_on=f"{prefix}_response_id",
                    how="left",
                )

            pair_quality_df["direction_pretty"] = pair_quality_df["direction_label"].map(
                {
                    "shorter_preferred": "Shorter preferred",
                    "longer_preferred": "Longer preferred",
                }
            )
            pair_quality_df["mean_candidate_f1"] = (
                pair_quality_df["chosen_f1"] + pair_quality_df["rejected_f1"]
            ) / 2.0
            pair_quality_df["mean_surface_evidence_support_rate"] = (
                pair_quality_df["chosen_surface_evidence_support_rate"]
                + pair_quality_df["rejected_surface_evidence_support_rate"]
            ) / 2.0
            pair_quality_df["surface_support_rate_gap"] = (
                pair_quality_df["chosen_surface_evidence_support_rate"]
                - pair_quality_df["rejected_surface_evidence_support_rate"]
            )

            augmentation_coverage_table = pd.DataFrame(
                [
                    {
                        "artifact": "Directional pairs",
                        "rows_loaded": len(pair_quality_df),
                        "missing_rows_after_merge": 0,
                        "note": "Retained shorter/longer preferred pairs only",
                    },
                    {
                        "artifact": "Standardized candidate rows",
                        "rows_loaded": len(standardized_candidate_df),
                        "missing_rows_after_merge": int(
                            pair_quality_df["chosen_standardized_candidate_f1"].isna().sum()
                            + pair_quality_df["rejected_standardized_candidate_f1"].isna().sum()
                        ),
                        "note": "Response-level quality metadata joined by response_id",
                    },
                    {
                        "artifact": "Candidate-bank rows",
                        "rows_loaded": len(candidate_bank_df),
                        "missing_rows_after_merge": int(
                            pair_quality_df["chosen_surface_evidence_support_rate"].isna().sum()
                            + pair_quality_df["rejected_surface_evidence_support_rate"].isna().sum()
                        ),
                        "note": "Used for the surface-match evidence-support proxy",
                    },
                ]
            )

            metric_availability_table = pd.DataFrame(
                [
                    {
                        "requested_metric": "F1 margin",
                        "implementation": "delta_f1",
                        "status": "observed",
                        "note": "Exact pair-level winner-minus-loser F1 margin from the retained pair file",
                    },
                    {
                        "requested_metric": "Entity edit size",
                        "implementation": "semantic_set_edit_distance",
                        "status": "observed",
                        "note": "Full set-edit distance between the chosen and rejected candidates",
                    },
                    {
                        "requested_metric": "Token-length difference",
                        "implementation": "absolute_token_length_gap",
                        "status": "observed",
                        "note": "Absolute character-length gap between the two candidates",
                    },
                    {
                        "requested_metric": "Gold cardinality",
                        "implementation": "gold_entity_count",
                        "status": "observed",
                        "note": "Gold answer-set size for the underlying question",
                    },
                    {
                        "requested_metric": "Candidate F1",
                        "implementation": "chosen_f1, rejected_f1, mean_candidate_f1",
                        "status": "observed",
                        "note": "Both candidate F1 scores retained directly in the pair file",
                    },
                    {
                        "requested_metric": "Evidence support",
                        "implementation": "surface_evidence_support_rate",
                        "status": "proxy",
                        "note": "Surface-match proxy derived by checking predicted entity strings against the saved evidence snippets",
                    },
                    {
                        "requested_metric": "Candidate-generation provenance",
                        "implementation": "generator_checkpoint, candidate_label_source, sample_id",
                        "status": "partial",
                        "note": "Checkpoint and label source are exact; within-model provenance is audited with sample indices",
                    },
                ]
            )

            provenance_audit_table = pd.DataFrame(
                [
                    {
                        "provenance_field": "generator_checkpoint",
                        "unique_values": int(pair_quality_df["generator_checkpoint"].nunique()),
                        "dominant_value": pair_quality_df["generator_checkpoint"].mode().iloc[0],
                        "interpretation": "Constant model-level provenance if unique_values = 1",
                    },
                    {
                        "provenance_field": "candidate_label_source",
                        "unique_values": int(pair_quality_df["candidate_label_source"].nunique()),
                        "dominant_value": pair_quality_df["candidate_label_source"].mode().iloc[0],
                        "interpretation": "Construction-label provenance in the retained ordinary pairs",
                    },
                    {
                        "provenance_field": "pair_type",
                        "unique_values": int(pair_quality_df["pair_type"].nunique()),
                        "dominant_value": pair_quality_df["pair_type"].mode().iloc[0],
                        "interpretation": "Whole-response versus any alternative pair-construction mode",
                    },
                ]
            )

            display(Markdown("## Pair-Quality Augmentation Coverage"))
            display(augmentation_coverage_table)

            display(Markdown("## Requested Metric Availability"))
            display(metric_availability_table)

            display(Markdown("## Provenance Audit"))
            display(provenance_audit_table)
            """
        ),
        code_cell(
            """
            def summarize_metric_by_direction(frame: pd.DataFrame, column: str, metric_label: str) -> dict[str, object]:
                shorter_values = frame.loc[
                    frame["direction_label"] == "shorter_preferred",
                    column,
                ].dropna()
                longer_values = frame.loc[
                    frame["direction_label"] == "longer_preferred",
                    column,
                ].dropna()
                return {
                    "metric": metric_label,
                    "shorter_mean": float(shorter_values.mean()),
                    "longer_mean": float(longer_values.mean()),
                    "shorter_median": float(shorter_values.median()),
                    "longer_median": float(longer_values.median()),
                    "longer_minus_shorter_mean": float(longer_values.mean() - shorter_values.mean()),
                }


            quality_summary_df = pd.DataFrame(
                [
                    summarize_metric_by_direction(pair_quality_df, "delta_f1", "F1 margin"),
                    summarize_metric_by_direction(pair_quality_df, "semantic_set_edit_distance", "Set-edit distance"),
                    summarize_metric_by_direction(pair_quality_df, "absolute_entity_gap", "Absolute entity-count gap"),
                    summarize_metric_by_direction(pair_quality_df, "absolute_token_length_gap", "Absolute token-length gap"),
                    summarize_metric_by_direction(pair_quality_df, "gold_entity_count", "Gold cardinality"),
                    summarize_metric_by_direction(pair_quality_df, "chosen_f1", "Chosen candidate F1"),
                    summarize_metric_by_direction(pair_quality_df, "rejected_f1", "Rejected candidate F1"),
                    summarize_metric_by_direction(pair_quality_df, "mean_candidate_f1", "Mean candidate F1"),
                    summarize_metric_by_direction(
                        pair_quality_df,
                        "chosen_surface_evidence_support_rate",
                        "Chosen evidence-support proxy",
                    ),
                    summarize_metric_by_direction(
                        pair_quality_df,
                        "rejected_surface_evidence_support_rate",
                        "Rejected evidence-support proxy",
                    ),
                    summarize_metric_by_direction(
                        pair_quality_df,
                        "surface_support_rate_gap",
                        "Chosen-minus-rejected support gap",
                    ),
                ]
            )

            semantic_operation_mix = pd.crosstab(
                pair_quality_df["direction_pretty"],
                pair_quality_df["semantic_operation"],
                normalize="index",
            ).round(4)

            display(Markdown("## Pair-Quality Summary By Direction"))
            display(quality_summary_df)

            display(Markdown("## Semantic Operation Mix By Direction"))
            display(semantic_operation_mix)
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(2, 3, figsize=(18, 12.5))


            def draw_direction_boxplot(
                ax: plt.Axes,
                frame: pd.DataFrame,
                column: str,
                title: str,
                ylabel: str,
                *,
                ylim: tuple[float, float] | None = None,
            ) -> None:
                shorter_values = frame.loc[
                    frame["direction_label"] == "shorter_preferred",
                    column,
                ].dropna()
                longer_values = frame.loc[
                    frame["direction_label"] == "longer_preferred",
                    column,
                ].dropna()

                box = ax.boxplot(
                    [shorter_values, longer_values],
                    tick_labels=["Shorter preferred", "Longer preferred"],
                    patch_artist=True,
                    widths=0.58,
                    showfliers=False,
                    medianprops={"color": "#111827", "linewidth": 2.0},
                    whiskerprops={"color": "#4B5563"},
                    capprops={"color": "#4B5563"},
                )
                for patch, color in zip(box["boxes"], [SHORTER_COLOR, LONGER_COLOR]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.78)

                ax.set_title(title, fontsize=14, weight="bold")
                ax.set_ylabel(ylabel, fontsize=11)
                ax.tick_params(axis="x", rotation=10, labelsize=10.5)
                ax.tick_params(axis="y", labelsize=10.5)
                if ylim is not None:
                    ax.set_ylim(*ylim)

                medians = [float(shorter_values.median()), float(longer_values.median())]
                for xpos, median_value in enumerate(medians, start=1):
                    ax.text(
                        xpos,
                        median_value,
                        f"median={median_value:0.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9.5,
                        color="#111827",
                    )


            plot_specs = [
                ("delta_f1", "F1 Margin", "Chosen-minus-rejected F1"),
                ("semantic_set_edit_distance", "Set-Edit Distance", "Entity operations"),
                ("absolute_token_length_gap", "Token-Length Gap", "Absolute char-length gap"),
                ("gold_entity_count", "Gold Cardinality", "Gold entity count"),
                ("mean_candidate_f1", "Mean Candidate F1", "Mean candidate F1", (0.0, 1.02)),
                (
                    "mean_surface_evidence_support_rate",
                    "Mean Evidence-Support Proxy",
                    "Mean surface-match support rate",
                    (0.0, 1.02),
                ),
            ]

            for ax, spec in zip(axes.flatten(), plot_specs):
                if len(spec) == 3:
                    column, title, ylabel = spec
                    ylim = None
                else:
                    column, title, ylabel, ylim = spec
                draw_direction_boxplot(ax, pair_quality_df, column, title, ylabel, ylim=ylim)

            fig.suptitle("Pair-Quality Distributions Across Directional Pair Types", fontsize=18, weight="bold", y=0.985)
            fig.text(
                0.5,
                0.945,
                "Research question: Do shorter- and longer-preferred pairs differ only in direction, or also in pair quality and difficulty?",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#374151",
            )
            fig.text(
                0.5,
                0.03,
                "Claim tested here: if direction is confounded with difficulty, the directional groups should diverge on F1 margin, edit size, token gap, gold cardinality, candidate quality, or evidence support.",
                ha="center",
                va="center",
                fontsize=11.5,
                color="#4B5563",
            )
            fig.tight_layout(rect=[0.03, 0.06, 0.98, 0.90])
            plt.show()
            """
        ),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.6))

            chosen_sample_distribution = pd.crosstab(
                pair_quality_df["direction_pretty"],
                pair_quality_df["chosen_sample_id"],
                normalize="index",
            ).sort_index(axis=1)
            rejected_sample_distribution = pd.crosstab(
                pair_quality_df["direction_pretty"],
                pair_quality_df["rejected_sample_id"],
                normalize="index",
            ).sort_index(axis=1)


            def draw_sample_provenance(ax: plt.Axes, distribution: pd.DataFrame, title: str) -> None:
                sample_ids = distribution.columns.to_numpy()
                x_positions = np.arange(len(sample_ids))
                width = 0.38
                shorter_values = distribution.loc["Shorter preferred"].to_numpy()
                longer_values = distribution.loc["Longer preferred"].to_numpy()

                ax.bar(
                    x_positions - width / 2,
                    shorter_values,
                    width=width,
                    color=SHORTER_COLOR,
                    alpha=0.82,
                    label="Shorter preferred",
                )
                ax.bar(
                    x_positions + width / 2,
                    longer_values,
                    width=width,
                    color=LONGER_COLOR,
                    alpha=0.82,
                    label="Longer preferred",
                )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(sample_ids.astype(int), fontsize=10.5)
                ax.set_ylabel("Row-normalized share", fontsize=11)
                ax.set_xlabel("Generation sample index", fontsize=11)
                ax.set_title(title, fontsize=14, weight="bold")
                ax.tick_params(axis="y", labelsize=10.5)


            draw_sample_provenance(axes[0], chosen_sample_distribution, "Chosen Candidate Sample-Index Provenance")
            draw_sample_provenance(axes[1], rejected_sample_distribution, "Rejected Candidate Sample-Index Provenance")
            axes[1].legend(frameon=False, loc="upper right", fontsize=10.5)

            fig.suptitle("Candidate-Generation Provenance Audit", fontsize=18, weight="bold", y=0.985)
            fig.text(
                0.5,
                0.945,
                "Research question: Could directional imbalance be an artifact of candidate-generation provenance rather than pair content alone?",
                ha="center",
                va="center",
                fontsize=12.5,
                color="#374151",
            )
            fig.text(
                0.5,
                0.08,
                f"Model-level provenance is constant in this export: {pair_quality_df['generator_checkpoint'].nunique()} generator checkpoint and {pair_quality_df['candidate_label_source'].nunique()} candidate-label source. The figure therefore audits within-model sample-index provenance only.",
                ha="center",
                va="center",
                fontsize=11.3,
                color="#4B5563",
            )
            fig.tight_layout(rect=[0.03, 0.12, 0.98, 0.90])
            plt.show()
            """
        ),
        code_cell(
            """
            pair_quality_narrative = f'''
            ## Draft-Ready Pair-Quality Summary

            The ordinary directional pairs are **not matched on pair quality alone**. Longer-preferred pairs have a
            larger mean F1 margin (**{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "delta_f1"].mean():0.3f}**
            versus **{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "delta_f1"].mean():0.3f}**),
            a larger mean set-edit distance (**{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "semantic_set_edit_distance"].mean():0.2f}**
            versus **{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "semantic_set_edit_distance"].mean():0.2f}**),
            and a larger mean gold cardinality (**{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "gold_entity_count"].mean():0.2f}**
            versus **{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "gold_entity_count"].mean():0.2f}**).

            Shorter-preferred pairs, however, show a substantially larger mean token-length gap
            (**{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "absolute_token_length_gap"].mean():0.1f}**
            versus **{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "absolute_token_length_gap"].mean():0.1f}**),
            indicating that answer-length separation is itself asymmetric across directions.

            Candidate quality also differs by direction. The mean rejected-candidate F1 is
            **{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "rejected_f1"].mean():0.3f}**
            for shorter-preferred pairs versus
            **{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "rejected_f1"].mean():0.3f}**
            for longer-preferred pairs, suggesting that shorter-preferred pairs often retain harder negatives.
            Under the notebook's descriptive surface-match evidence-support proxy, the mean chosen-minus-rejected
            support gap is
            **{pair_quality_df.loc[pair_quality_df["direction_label"] == "shorter_preferred", "surface_support_rate_gap"].mean():0.3f}**
            for shorter-preferred pairs and
            **{pair_quality_df.loc[pair_quality_df["direction_label"] == "longer_preferred", "surface_support_rate_gap"].mean():0.3f}**
            for longer-preferred pairs. This support metric should be treated as a proxy, not as a definitive
            evidence-verification label.

            Model-level provenance is controlled in the current export: all retained ordinary pairs come from a single
            generator checkpoint and a single candidate-label source. Overall, the present ordinary pair set differs by
            more than cardinality direction alone, so downstream shortcut claims should report both **exposure
            imbalance** and **pair-quality imbalance**.
            '''

            display(Markdown(pair_quality_narrative))
            """
        ),
        code_cell(
            """
            narrative = f'''
            ## Draft-Ready Exposure Summary

            The current `ordinary_pairs.jsonl` dataset exhibits a **clear shorter-preferred exposure imbalance**.
            At the pair-micro level, the chosen answer is shorter in **{exposure_metrics['pair_micro_shorter_prop_directional']*100:0.1f}%**
            of directional pairs (`D_pair = {exposure_metrics['pair_micro_D_pair']:0.3f}`, clustered 95% CI
            [{ci_map['pair_micro_D_pair'][0]:0.3f}, {ci_map['pair_micro_D_pair'][1]:0.3f}]).

            This pattern remains visible after equalising questions. The question-macro shorter-preferred proportion is
            **{exposure_metrics['question_macro_shorter_prop']*100:0.1f}%**, with
            `D_question = {exposure_metrics['question_macro_D_question']:0.3f}` (clustered 95% CI
            [{ci_map['question_macro_D_question'][0]:0.3f}, {ci_map['question_macro_D_question'][1]:0.3f}]).
            Under the draft's descriptive dominance bands, both pair-micro and question-macro imbalance fall in the
            **strong dominance** regime.

            The breadth analysis also argues against the imbalance being driven by only a few high-combinatorics questions.
            Among the **{exposure_metrics['question_count_directional']}** questions with at least one directional pair,
            **{exposure_metrics['shorter_dominant_questions']}** are shorter-dominant,
            **{exposure_metrics['longer_dominant_questions']}** are longer-dominant, and
            **{exposure_metrics['tied_questions']}** are tied; **{exposure_metrics['question_count_no_directional']}**
            additional questions contribute only equal-cardinality pairs. The question-macro estimate therefore supports
            the same qualitative conclusion as the pair-micro estimate: the current ordinary preference data exposes DPO
            to a substantial shorter-preferred directional cue.
            '''

            display(Markdown(narrative))
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
                "version": "3.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    task_root = Path(__file__).resolve().parents[2]
    output_path = (
        task_root
        / "Artifacts"
        / "Cardinality_Shortcut_Study"
        / "cardinality_shortcut_exposure_imbalance_report.ipynb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote notebook to {output_path}")


if __name__ == "__main__":
    main()
