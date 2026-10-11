#!/usr/bin/env python3
"""Cross-validate logistic rerankers on matched three-model SFT and DPO pools.

Reuses archived official candidate labels. No generation, Java or paid judging.
Both policies use identical question folds, with every vectorizer fitted only
on training questions. This is an exploratory development-set experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
from sklearn.model_selection import KFold

from scripts.export_expansion_dpo_cases import load_scored
from scripts.analyze_expansion_dpo_evaluation import digest
from scripts.analyze_matched_qwen_expansion import paired_contrast, ranking_metrics
from scripts.train_candidate_reranker_pilot import (
    key, fit_components, score_components, write_json, write_jsonl,
)

MODELS = ("llama31", "qwen3", "ministral3")
POLICIES = ("sft", "dpo")


def prepare_pools(loaded, mode):
    sources = {item[0]["model_key"]: item for item in loaded}
    if len(loaded) != 3 or set(sources) != set(MODELS):
        raise ValueError("Supply exactly one scored experiment for each of the three models")
    examples = sources[MODELS[0]][1]
    for _, other, _, _ in loaded:
        if other != examples:
            raise ValueError("Questions, evidence or gold aliases differ across models")
    pools = {}
    for policy in POLICIES:
        merged = {}
        for model in MODELS:
            arm = sources[model][2][f"{policy}_{mode}"]
            if set(arm["scores"]) != set(examples):
                raise ValueError("Pool question membership differs")
            for qid in sorted(examples):
                for candidate in arm["candidates"][qid]:
                    identity = (qid, key(candidate["answer"]))
                    label = int(candidate["official_accepted"])
                    if identity not in merged:
                        merged[identity] = {
                            "question_id": qid, "answer": candidate["answer"],
                            "sources": [], "source_ranks": {}, "relation_types": [],
                            "surface_operations": [], "is_format_variant": False,
                            "label": label,
                        }
                    row = merged[identity]
                    # Never choose a surface form using gold. Investigate a
                    # conflicting official label rather than silently OR it.
                    if row["label"] != label:
                        raise ValueError(f"Deduplication changes official acceptance: {qid}, {identity[1]}")
                    if model not in row["sources"]:
                        row["sources"].append(model)
                    rank = candidate["rank"]
                    row["source_ranks"][model] = min(rank, row["source_ranks"].get(model, rank))
                    relation = candidate.get("relation_type", "unknown")
                    if relation not in row["relation_types"]:
                        row["relation_types"].append(relation)
        pools[policy] = list(merged.values())
    return examples, pools


def tie_key(row):
    # Fixed source order used only for exact score ties, never official labels.
    provenance = min((MODELS.index(m), r) for m, r in row["source_ranks"].items())
    return provenance, key(row["answer"])


def round_robin(rows):
    queues = {m: sorted((r for r in rows if m in r["source_ranks"]),
                       key=lambda r: (r["source_ranks"][m], key(r["answer"]))) for m in MODELS}
    result, seen = [], set()
    for i in range(max((len(q) for q in queues.values()), default=0)):
        for model in MODELS:
            if i < len(queues[model]):
                row = queues[model][i]
                if key(row["answer"]) not in seen:
                    result.append(row)
                    seen.add(key(row["answer"]))
    return result


def question_weights(rows, policies):
    """Equal weight per question/policy, then mean-one scaling for fixed C."""
    counts = defaultdict(int)
    for row, policy in zip(rows, policies):
        counts[(row["question_id"], policy)] += 1
    weights = np.asarray([1 / counts[(r["question_id"], p)] for r, p in zip(rows, policies)])
    return weights / weights.mean()


def evaluate_rankings(orderings):
    official_rows = [{"question_id": q, "answers": [r["answer"] for r in rows],
                      "matching_answers": [r["answer"] for r in rows if r["label"]]}
                     for q, rows in sorted(orderings.items())]
    metrics, scored = ranking_metrics(official_rows)
    for row, values in zip(official_rows, scored):
        values["coverage_full_pool"] = float(bool(row["matching_answers"]))
    metrics["coverage_full_pool"] = sum(r["coverage_full_pool"] for r in scored) / len(scored)
    return metrics, scored


def cross_validate(examples, pools, folds=5, resamples=10000):
    qids = sorted(examples)
    # The feature fitter receives no gold aliases, even though the archived
    # examples contain them for validation/scoring.
    feature_examples = {q: {"question": e.get("question", ""), "snippets": e.get("snippets", [])}
                        for q, e in examples.items()}
    if not 2 <= folds <= len(qids):
        raise ValueError("Need at least two folds and one test question per fold")
    scores = {(train, test): np.full(len(pools[test]), np.nan)
              for train in (*POLICIES, "mixed") for test in POLICIES}
    row_folds = {p: np.full(len(pools[p]), -1, dtype=int) for p in POLICIES}
    fold_manifest = []
    for fold, (train_indices, test_indices) in enumerate(KFold(
            n_splits=folds, shuffle=True, random_state=3407).split(qids)):
        train_qids = {qids[i] for i in train_indices}
        test_qids = {qids[i] for i in test_indices}
        details = {"fold": fold, "training_question_ids": sorted(train_qids),
                   "test_question_ids": sorted(test_qids), "training": {}}
        for test in POLICIES:
            for i, row in enumerate(pools[test]):
                if row["question_id"] in test_qids:
                    row_folds[test][i] = fold
        for train in (*POLICIES, "mixed"):
            rows, origins = [], []
            for policy in POLICIES if train == "mixed" else (train,):
                selected = [r for r in pools[policy] if r["question_id"] in train_qids]
                rows.extend(selected)
                origins.extend([policy] * len(selected))
            labels = np.asarray([r["label"] for r in rows])
            if not rows or set(labels.tolist()) != {0, 1}:
                raise ValueError(f"Fold {fold} {train} needs positive and negative training candidates")
            components = fit_components(rows, feature_examples, labels,
                                        sample_weight=question_weights(rows, origins))
            details["training"][train] = {"candidates": len(rows), "positives": int(labels.sum())}
            for test in POLICIES:
                indices = [i for i, r in enumerate(pools[test]) if r["question_id"] in test_qids]
                if indices:
                    predictions = score_components(components, [pools[test][i] for i in indices], feature_examples)
                    scores[(train, test)][indices] = predictions
        fold_manifest.append(details)
        print(f"Completed fold {fold + 1}/{folds}", flush=True)
    if any(not np.isfinite(s).all() for s in scores.values()):
        raise RuntimeError("Missing or nonfinite out-of-fold candidate scores")
    orderings = {}
    rankings = []
    for test in POLICIES:
        groups = {q: [] for q in qids}  # Includes empty/failed pools in denominator.
        for i, row in enumerate(pools[test]):
            groups[row["question_id"]].append({**row, "fold": int(row_folds[test][i])})
        orderings[f"round_robin_on_{test}"] = {q: round_robin(rs) for q, rs in groups.items()}
        for train in (*POLICIES, "mixed"):
            named = f"trained_{train}_on_{test}"
            enriched = {q: [] for q in qids}
            for i, row in enumerate(pools[test]):
                enriched[row["question_id"]].append({**row, "score": float(scores[(train, test)][i]),
                                                     "fold": int(row_folds[test][i])})
            orderings[named] = {q: sorted(rs, key=lambda r: (-r["score"], tie_key(r)))
                               for q, rs in enriched.items()}
            rankings.extend({"condition": named, "rank": i, **row}
                            for rs in orderings[named].values() for i, row in enumerate(rs, 1))
    evaluated = {name: evaluate_rankings(order) for name, order in orderings.items()}
    for test in POLICIES:
        expected_coverage = evaluated[f"round_robin_on_{test}"][0]["coverage_full_pool"]
        if any(evaluated[f"trained_{train}_on_{test}"][0]["coverage_full_pool"] != expected_coverage
               for train in (*POLICIES, "mixed")):
            raise RuntimeError("Reranking changed pool membership or official labels")
    contrasts = {}
    pairs = {
        "dpo_pipeline_minus_sft_pipeline": ("trained_sft_on_sft", "trained_dpo_on_dpo"),
        **{f"dpo_pool_minus_sft_pool_with_{train}_reranker":
           (f"trained_{train}_on_sft", f"trained_{train}_on_dpo") for train in (*POLICIES, "mixed")},
        **{f"dpo_training_minus_sft_training_on_{p}":
           (f"trained_sft_on_{p}", f"trained_dpo_on_{p}") for p in POLICIES},
        **{f"reranker_minus_round_robin_on_{p}":
           (f"round_robin_on_{p}", f"trained_mixed_on_{p}") for p in POLICIES},
    }
    for name, (before, after) in pairs.items():
        delta = paired_contrast(evaluated[before][1], evaluated[after][1], resamples=resamples,
                               metrics=("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10"))
        contrasts[name] = {"before": before, "after": after, "metrics": {
            metric: {"difference": values["sft_minus_base"],
                     "paired_bootstrap_95ci": values["paired_bootstrap_95ci"]}
            for metric, values in delta.items()}}
    return {"metrics": {k: v[0] for k, v in evaluated.items()}, "paired_contrasts": contrasts,
            "folds": fold_manifest}, {k: v[1] for k, v in evaluated.items()}, rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--mode", choices=("greedy", "sampling10", "both"), default="sampling10")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loaded = [load_scored(p.resolve()) for p in args.experiment_dirs]
    modes = ("greedy", "sampling10") if args.mode == "both" else (args.mode,)
    prepared = {mode: prepare_pools(loaded, mode) for mode in modes}
    for mode, (examples, pools) in prepared.items():
        if not 2 <= args.folds <= len(examples):
            raise ValueError("Invalid fold count")
        print(f"Validated {mode}: {len(examples)} questions; " + ", ".join(
            f"{p}={len(rs)} candidates" for p, rs in pools.items()), flush=True)
    if args.validate_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "method": "TF-IDF/metadata logistic reranker",
        "evaluation": "Paired question-level cross-validation on reused development data",
        "seed": 3407, "fold_count": args.folds, "modes": list(modes),
        "software": {"python": sys.version, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "script_sha256": digest(Path(__file__)),
                     "feature_pipeline_sha256": digest(ROOT / "scripts/train_candidate_reranker_pilot.py")},
        "sources": {data[0]["model_key"]: {"directory": str(path.resolve()), "arms": data[3],
            **{name: digest(path / name) for name in ("manifest.json", "comparison_summary.json", "comparison_per_question.json")}}
            for path, data in zip(args.experiment_dirs, loaded)},
        "settings": {"new_candidates_or_format_variants": False,
            "deduplication": "Case/whitespace; fixed Llama/Qwen/Ministral first occurrence",
            "training": "Same question folds; all feature fitting restricted to training questions",
            "weighting": "Equal total weight per nonempty question/policy, mean-one scaling",
            "hyperparameters": "Earlier logistic pilot defaults; no tuning on held-out folds",
            "policy_feature": False, "bootstrap_resamples": 10000},
        "limitations": ["Exploratory, reused development set and one generation seed.",
            "Bootstrap intervals condition on CV predictions; they exclude retraining uncertainty and have no multiplicity correction.",
            "Separate pipelines change both training data and generator pools; fixed-reranker contrasts isolate pool changes.",
            "Mixed reranker sees more training rows; treat it as a shared-selection diagnostic.",
            "No final model trained on all development questions is evaluated or exported."]}
    write_json(args.output / "manifest.json", manifest)
    try:
        summaries = {}
        for mode, (examples, pools) in prepared.items():
            summary, questions, rankings = cross_validate(examples, pools, args.folds)
            summaries[mode] = summary
            write_json(args.output / f"{mode}_per_question.json", questions)
            write_jsonl(args.output / f"{mode}_rankings.jsonl", rankings)
        write_json(args.output / "comparison_summary.json", summaries)
        manifest["status"] = "complete"
        for mode, summary in summaries.items():
            print(mode, flush=True)
            for name, metrics in summary["metrics"].items():
                print(f"  {name}: MRR@5={metrics['mrr_at5']:.6f}, "
                      f"coverage@5={metrics['coverage_at5']:.4f}, "
                      f"full pool={metrics['coverage_full_pool']:.4f}")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
