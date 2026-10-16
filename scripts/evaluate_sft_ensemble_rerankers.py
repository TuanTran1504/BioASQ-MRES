#!/usr/bin/env python3
"""Apply three frozen logistic selectors to existing original/max-five test pools.

No fitting, generation, new official matching, or paid judging. Reports every
selector on every identical pool, with paired comparisons against fixed voting.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn

from scripts.rescore_sft_ensemble_frequency import (
    ARMS, METHODS, MODES, MODELS, METRICS, load_comparison, summarize_pools,
    paired_contrast, read, records, digest, write, write_jsonl,
)
from scripts.compare_expansion_pool_rerankers import tie_key, evaluate_rankings
from scripts.train_sft_ensemble_reranker import SELECTORS, feature_rows
from scripts.train_candidate_reranker_pilot import score_components


def validate_sources(prepared, comparison):
    sources = read(comparison / "manifest.json")["sources"]
    for model in MODELS:
        archive = Path(sources[f"{model}_greedy"]["directory"])
        source_submission = Path(read(archive / "manifest.json")["submission_dir"])
        inference = read(source_submission / "manifest.json")["sources"][model]
        for arm in ARMS:
            bank = Path(prepared[f"{'original' if arm == 'original' else 'max5'}_banks"][model])
            trained = read(bank / "manifest.json")
            if trained.get("status") != "complete" or trained.get("model_key") != model:
                raise ValueError("Reranker source bank is incomplete or has changed identity")
            expected = inference[arm]["source"]
            for field in ("adapter_files_sha256", "base_snapshot", "base_revision", "system_prompt",
                          "model_loader", "chat_template_kwargs", "execution_mode"):
                if trained["source"].get(field) != expected.get(field):
                    raise ValueError(f"Reranker and test generator sources differ: {model}/{arm}/{field}")


def load_frozen(directory, examples, comparison):
    state = read(directory / "manifest.json")
    if (state.get("status") != "complete" or set(state.get("selectors", {})) != set(SELECTORS)
            or state.get("question_counts") != {"train": 1296, "validation": 144}
            or state.get("sklearn_version") != sklearn.__version__):
        raise ValueError("Need three complete fitting-only selectors and the same scikit-learn version")
    hashes = {"manifest.json": digest(directory / "manifest.json")}
    for name, value in state["artifacts_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != value:
            raise ValueError(f"Frozen reranker artifact changed: {name}")
        hashes[name] = value
    split_ids = {}
    for split, count in (("train", 1296), ("validation", 144)):
        filename = f"{split}_examples.jsonl"
        if filename not in hashes:
            raise ValueError("Missing hashed reranker question membership")
        rows = records(directory / filename)
        split_ids[split] = {r["question_id"] for r in rows}
        if len(rows) != count or len(split_ids[split]) != count or split_ids[split] & set(examples):
            raise ValueError("Duplicate, missing or test-overlapping reranker questions")
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Reranker fitting and validation overlap")
    submission = Path(state["submission"])
    if digest(submission / "prepared.json") != state["prepared_sha256"]:
        raise ValueError("Reranker preparation metadata changed")
    prepared = read(submission / "prepared.json")
    feature_code = "scripts/train_candidate_reranker_pilot.py"
    if digest(ROOT / feature_code) != prepared["code_hashes"][feature_code]:
        raise ValueError("Candidate feature/scoring code changed since reranker training")
    validate_sources(prepared, comparison)
    payloads = {}
    for selector, arms in SELECTORS.items():
        metadata = state["selectors"][selector]
        path = (directory / metadata["model_path"]).resolve()
        if (not path.is_relative_to(directory.resolve()) or digest(path) != metadata["model_sha256"]
                or metadata["model_path"] not in hashes):
            raise ValueError("Frozen selector weights changed")
        payload = joblib.load(path)
        expected_conditions = [f"{a}_{m}" for a in arms for m in ("greedy", "sampling4")]
        if (payload.get("schema") != "sft-ensemble-logistic-v1" or payload.get("selector") != selector
                or tuple(payload.get("models", ())) != MODELS
                or payload.get("training_arms") != list(arms)
                or payload.get("selection_conditions") != expected_conditions
                or payload.get("selected_C") != state["selected_C"][selector]):
            raise ValueError("Frozen selector schema or selection metadata differs")
        payloads[selector] = payload
    return state, payloads, hashes


def rank_frozen(payload, pools, examples):
    conditions, scores, rankings = {}, {}, []
    # Labels and gold aliases are excluded from model inputs. These defaults
    # match fitting metadata; no synthetic surface forms are introduced.
    contexts = {q: {"question": e["question"], "snippets": e["snippets"]} for q, e in examples.items()}
    for name, rows in pools.items():
        features = feature_rows([{**r, "surface_operations": [], "is_format_variant": False} for r in rows])
        values = score_components(payload["components"], features, contexts) if rows else np.array([])
        if not np.isfinite(values).all():
            raise ValueError("Non-finite frozen reranker predictions")
        groups = {q: [] for q in examples}
        for row, value in zip(rows, values):
            groups[row["question_id"]].append({**row, "score": float(value)})
        ordered = {q: sorted(rr, key=lambda r: (-r["score"], tie_key(r))) for q, rr in groups.items()}
        condition = f"{name}_{payload['selector']}_reranker"
        metrics, scored = evaluate_rankings(ordered)
        conditions[condition] = {"metrics": metrics, "mean_unique_candidates": len(rows) / len(examples)}
        scores[condition] = scored
        rankings.extend({"condition": condition, "rank": i, **r}
                        for rr in ordered.values() for i, r in enumerate(rr, 1))
    return conditions, scores, rankings


def evaluate(comparison, rerankers, output, resamples=10000):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if resamples < 2:
        raise ValueError("Need at least two bootstrap resamples")
    original, pools, examples, gold, source_hashes = load_comparison(comparison)
    trained, payloads, model_hashes = load_frozen(rerankers, examples, comparison)
    print("Validated test pools, frozen selectors, source checkpoints and disjoint question membership.", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "question_count": len(examples), "source_comparison": str(comparison),
             "source_file_hashes": source_hashes, "reranker_directory": str(rerankers),
             "reranker_file_hashes": model_hashes, "training": False, "generation": False,
             "scoring_code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                 Path(__file__), ROOT / "scripts/rescore_sft_ensemble_frequency.py",
                 ROOT / "scripts/train_candidate_reranker_pilot.py",
                 ROOT / "scripts/compare_expansion_pool_rerankers.py",
                 ROOT / "scripts/analyze_sft_ensemble_test.py",
                 ROOT / "scripts/analyze_matched_qwen_expansion.py")}}
    write(output / "manifest.json", state)
    try:
        conditions, scores, rankings = summarize_pools(pools, examples, gold)
        for selector, payload in payloads.items():
            cc, ss, rr = rank_frozen(payload, pools, examples)
            for name, condition in cc.items():
                condition["per_batch"] = {}
                for batch in sorted({r["batch"] for r in gold.values()}):
                    selected = [r for r in ss[name] if gold[r["question_id"]]["batch"] == batch]
                    condition["per_batch"][batch] = {"question_count": len(selected), **{
                        metric: sum(r[metric] for r in selected) / len(selected) for metric in condition["metrics"]}}
            conditions.update(cc); scores.update(ss); rankings.extend(rr)
            print(f"Scored frozen {selector} reranker on all four pools.", flush=True)
        for pool in pools:
            if len({c["metrics"]["coverage_full_pool"] for n, c in conditions.items() if n.startswith(pool + "_")}) != 1:
                raise ValueError("Selection changed full-pool coverage")
        comparisons = {f"{s}_reranker_minus_{method}_{arm}_{mode}":
            (f"{arm}_{mode}_{method}", f"{arm}_{mode}_{s}_reranker")
            for s in SELECTORS for method in METHODS for arm in ARMS for mode in MODES}
        comparisons.update({f"max5_minus_original_{mode}_{s}_reranker":
            (f"original_{mode}_{s}_reranker", f"max5_{mode}_{s}_reranker") for mode in MODES for s in SELECTORS})
        comparisons.update({f"max5_specialised_minus_original_specialised_{mode}":
            (f"original_{mode}_original_reranker", f"max5_{mode}_max5_reranker") for mode in MODES})
        contrasts = {}
        for i, (name, (before, after)) in enumerate(comparisons.items(), 1):
            delta = paired_contrast(scores[before], scores[after], resamples=resamples, metrics=METRICS)
            contrasts[name] = {"before": before, "after": after, "metrics": {
                k: {"difference": v["sft_minus_base"], "paired_bootstrap_95ci": v["paired_bootstrap_95ci"],
                    "resamples": resamples, "seed": 3407} for k, v in delta.items()}}
            if i % 8 == 0:
                print(f"Paired comparisons: {i}/{len(comparisons)}", flush=True)
        result = {"question_count": len(examples), "conditions": conditions, "paired_contrasts": contrasts,
                  "source_costs": original["source_costs"], "ensemble_costs": original["ensemble_costs"],
                  "settings": copy.deepcopy(original["settings"]),
                  "selectors": {s: {"selected_C": p["selected_C"], "training_arms": p["training_arms"]}
                                for s, p in payloads.items()},
                  "limitations": [*original["limitations"],
                      "Rerankers trained on greedy and four-sample pools; ten-sample application has a sampling-distribution shift.",
                      "Test scores are reported for all frozen selectors; no test-based model selection or refitting.",
                      "These are historically inspected project test batches, not a fresh confirmatory test."],
                  "training_question_counts": trained["question_counts"], "generation_reused": True}
        result["settings"]["fixed_selectors"] = list(METHODS)
        for directory, hashes in ((comparison, source_hashes), (rerankers, model_hashes)):
            if any(digest(directory / name) != value for name, value in hashes.items()):
                raise ValueError("Input artifacts changed during evaluation")
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scores)
        write_jsonl(output / "rankings.jsonl", rankings)
        for arm in ARMS:
            for mode in MODES:
                pool = f"{arm}_{mode}"
                print(f"\n{pool}", flush=True)
                for selector in (*METHODS, *(f"{s}_reranker" for s in SELECTORS)):
                    m = conditions[f"{pool}_{selector}"]["metrics"]
                    print(f"  {selector}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, "
                          f"full pool={m['coverage_full_pool']:.4f}", flush=True)
                for selector in SELECTORS:
                    c = contrasts[f"{selector}_reranker_minus_sample_frequency_{arm}_{mode}"]["metrics"]["mrr_at5"]
                    print(f"  {selector} reranker minus frequency: {c['difference']:+.6f}, "
                          f"95% CI={c['paired_bootstrap_95ci']}", flush=True)
        state["status"] = "complete"
        print(f"\nSaved: {output / 'comparison_summary.json'}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--rerankers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    evaluate(args.comparison.resolve(), args.rerankers.resolve(), args.output.resolve(), args.resamples)
