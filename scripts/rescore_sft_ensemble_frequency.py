#!/usr/bin/env python3
"""Evaluate sample-frequency voting on already scored, unchanged test pools.

Reconstructs pools from verified response archives, reuses official candidate
labels, and reproduces the previous three selectors before adding frequency.
No generation, training, Java invocation, or paid judging is performed.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_sft_ensemble_test import (
    ARMS, METHODS, MODELS, digest, load_archive, merge_pool, read, records,
    summarize_pools, write, write_jsonl,
)
from scripts.analyze_matched_qwen_expansion import paired_contrast

BASELINE_METHODS = ("round_robin", "model_vote", "rank_vote")
MODES = ("greedy", "sampling10")
METRICS = ("mrr_at5", "coverage_at5", "coverage_at10", "coverage_full_pool")


def load_comparison(directory):
    state = read(directory / "manifest.json")
    summary = read(directory / "comparison_summary.json")
    previous_scores = read(directory / "comparison_per_question.json")
    if state.get("status") != "complete" or summary.get("question_count") != 95:
        raise ValueError("Need a complete scored 95-question ensemble comparison")
    required_sources = {f"{model}_{mode}" for model in MODELS for mode in MODES}
    if set(state.get("sources", {})) != required_sources:
        raise ValueError("Need the same three models and both decoding modes")
    source_files = ["manifest.json", "comparison_summary.json", "comparison_per_question.json"]
    pools, examples, gold = {}, None, None
    for mode in MODES:
        by_model = {}
        for model in MODELS:
            source = state["sources"][f"{model}_{mode}"]
            archive = Path(source["directory"])
            if digest(archive / "manifest.json") != source["manifest_sha256"]:
                raise ValueError("Source archive manifest changed since scoring")
            saved, _, ee, gg, generations, _ = load_archive(archive, mode)
            if saved["model_key"] != model or saved["arm_hashes"] != source["arm_hashes"]:
                raise ValueError("Source identity or archived files changed since scoring")
            current_examples = {e["question_id"]: e for e in ee}
            if examples is None:
                examples, gold = current_examples, gg
            if examples != current_examples or gold != gg:
                raise ValueError("Question membership, evidence or gold differ")
            by_model[model] = generations
        for arm in ARMS:
            name = f"{arm}_{mode}"
            filename = f"{name}_candidates.jsonl"
            source_files.append(filename)
            rows = records(directory / filename)
            reconstructed = merge_pool(by_model, arm)
            if [{k: v for k, v in r.items() if k != "label"} for r in rows] != reconstructed:
                raise ValueError(f"Saved candidates or frequencies changed: {name}")
            if any(type(r.get("label")) is not int or r["label"] not in (0, 1) for r in rows):
                raise ValueError(f"Invalid cached official candidate labels: {name}")
            pools[name] = rows
    # Reproduce all old question-level metrics, including empty/failed outputs.
    baseline, scores, _ = summarize_pools(pools, examples, gold, BASELINE_METHODS)
    for name, condition in baseline.items():
        old = summary["conditions"][name]
        if (condition != old or scores[name] != previous_scores[name]):
            raise ValueError(f"Cached official labels do not reproduce previous results: {name}")
    hashes = {f: digest(directory / f) for f in source_files}
    return summary, pools, examples, gold, hashes


def rescore(directory, output, resamples=10000):
    if resamples < 2:
        raise ValueError("Need at least two bootstrap resamples")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    original, pools, examples, gold, hashes = load_comparison(directory)
    print("Validated unchanged pools, draw frequencies and all 12 previous conditions.", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "question_count": len(examples),
             "source_comparison": str(directory), "source_file_hashes": hashes,
             "scoring_code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                 Path(__file__), ROOT / "scripts/analyze_sft_ensemble_test.py",
                 ROOT / "scripts/analyze_matched_qwen_expansion.py",
                 ROOT / "scripts/compare_expansion_pool_rerankers.py")},
             "generation": False, "training": False, "paid_judging": False}
    write(output / "manifest.json", state)
    try:
        conditions, scores, rankings = summarize_pools(pools, examples, gold)
        result = copy.deepcopy(original)
        result["conditions"] = conditions
        comparisons = {
            f"sample_frequency_minus_{method}_{arm}_{mode}":
                (f"{arm}_{mode}_{method}", f"{arm}_{mode}_sample_frequency")
            for arm in ARMS for mode in MODES for method in BASELINE_METHODS}
        comparisons.update({f"max5_minus_original_{mode}_sample_frequency":
            (f"original_{mode}_sample_frequency", f"max5_{mode}_sample_frequency")
            for mode in MODES})
        comparisons.update({f"sampling10_minus_greedy_{arm}_sample_frequency":
            (f"{arm}_greedy_sample_frequency", f"{arm}_sampling10_sample_frequency")
            for arm in ARMS})
        for name, (before, after) in comparisons.items():
            contrast = paired_contrast(scores[before], scores[after], resamples=resamples, metrics=METRICS)
            result["paired_contrasts"][name] = {"before": before, "after": after, "metrics": {
                k: {"difference": v["sft_minus_base"],
                    "paired_bootstrap_95ci": v["paired_bootstrap_95ci"],
                    "resamples": resamples, "seed": 3407} for k, v in contrast.items()}}
        result["settings"]["selectors"] = list(METHODS)
        result["settings"]["sample_frequency"] = {
            "score": "sum over models of draws containing candidate / total draws",
            "within_draw": "each normalised candidate counts at most once",
            "failed_draws": "remain in each model's denominator",
            "ties": "fixed source order, source rank, then normalised string",
            "equal_draw_budgets": "equivalent to sorting by total occurrence count"}
        result["source_comparison"] = str(directory)
        result["generation_reused"] = True
        result["limitations"].append(
            "Frequency voting was added after inspecting test results; exploratory, not a held-out method selection.")
        # Recheck inputs before publishing a completed result.
        if any(digest(directory / name) != value for name, value in hashes.items()):
            raise ValueError("Source comparison changed while rescoring")
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scores)
        write_jsonl(output / "rankings.jsonl", rankings)
        for arm in ARMS:
            for mode in MODES:
                name = f"{arm}_{mode}"
                print(f"\n{name}", flush=True)
                for method in METHODS:
                    m = conditions[f"{name}_{method}"]["metrics"]
                    print(f"  {method}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, "
                          f"coverage@10={m['coverage_at10']:.4f}, full pool={m['coverage_full_pool']:.4f}", flush=True)
                c = result["paired_contrasts"][f"sample_frequency_minus_model_vote_{arm}_{mode}"]["metrics"]["mrr_at5"]
                print(f"  Frequency minus model voting: {c['difference']:+.6f}, "
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
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", default=10000, type=int)
    args = parser.parse_args()
    rescore(args.comparison.resolve(), args.output.resolve(), args.resamples)
