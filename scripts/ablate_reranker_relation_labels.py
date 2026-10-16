#!/usr/bin/env python3
"""Retrain frozen ensemble selectors without self-reported relation labels."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from scripts import evaluate_sft_ensemble_rerankers as ev
from scripts import train_sft_ensemble_reranker as rr
from scripts.train_candidate_reranker_pilot import fit_components


def without_relations(pools):
    """Copy pools; preserve answers, labels, ranks and frequencies exactly."""
    return {name: [{**copy.deepcopy(row), "relation_types": []} for row in rows]
            for name, rows in pools.items()}


def load_training(directory, hashes):
    examples, pools = {}, {}
    for split, count in (("train", 1296), ("validation", 144)):
        filename = f"{split}_examples.jsonl"
        if filename not in hashes:
            raise ValueError("Unhashed fitting/validation membership")
        rows = ev.records(directory / filename)
        examples[split] = {r["question_id"]: r for r in rows}
        if len(rows) != count or len(examples[split]) != count:
            raise ValueError("Incomplete or duplicate fitting/validation questions")
        pools[split] = {}
        for arm in rr.ARMS:
            for mode in rr.MODES:
                name = f"{arm}_{mode}"
                filename = f"{split}_{name}_candidates.jsonl"
                if filename not in hashes:
                    raise ValueError("Unhashed cached candidate pool")
                candidates = ev.records(directory / filename)
                identities = set()
                for r in candidates:
                    identity = (r["question_id"], rr.key(r["answer"]))
                    if (r["question_id"] not in examples[split] or identity in identities
                            or type(r["label"]) is not int or r["label"] not in (0, 1)):
                        raise ValueError("Invalid cached candidate membership, label or duplicate")
                    identities.add(identity)
                pools[split][name] = candidates
    if set(examples["train"]) & set(examples["validation"]):
        raise ValueError("Fitting/validation question overlap")
    return examples, pools


def prepare(repository, baseline, comparison, submission):
    # Validate every source before creating a queued job. No fitting here.
    _, _, test, _, source_hashes = ev.load_comparison(comparison)
    _, _, model_hashes = ev.load_frozen(baseline, test, comparison)
    load_training(baseline, model_hashes)
    submission.mkdir(parents=True, exist_ok=False)
    hashes = rr.snapshot_code(repository, submission / "code")
    job = repository / "gadi_sft_8b_starter/jobs/ablate_reranker_relation_labels.pbs"
    import shutil
    shutil.copy2(job, submission / "job.pbs")
    ev.write(submission / "prepared.json", {
        "baseline": str(baseline), "comparison": str(comparison),
        "baseline_hashes": model_hashes, "comparison_hashes": source_hashes,
        "code_hashes": hashes, "job_sha256": ev.digest(submission / "job.pbs"),
        "scope": "Relation-label ablation; cached pools; no generation or DPO",
    })
    print("Prepared one CPU ablation using verified cached fitting, validation and test pools.", flush=True)


def verify_files(directory, hashes):
    for name, value in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or ev.digest(path) != value:
            raise ValueError(f"Frozen input changed: {name}")


def fit_ablation(pools, examples, baseline_payloads, C_grid):
    """Return fixed-C and validation-selected models; test inputs are absent."""
    blind = {s: without_relations(p) for s, p in pools.items()}
    fitted, trials = {}, {}
    for selector, arms in rr.SELECTORS.items():
        names = [f"{a}_{m}" for a in arms for m in rr.MODES]
        rows = [r for n in names for r in blind["train"][n]]
        strata = [n for n in names for _ in blind["train"][n]]
        labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("Need accepted and unaccepted fitting candidates")
        weights = rr.question_weights(rows, strata)
        fixed_C = baseline_payloads[selector]["selected_C"]
        if fixed_C not in C_grid:
            raise ValueError("Archived selected C is outside its original search grid")
        best, fixed, trials[selector] = None, None, []
        validation = {n: blind["validation"][n] for n in names}
        for C in C_grid:
            print(f"Tag-blind {selector}: fitting C={C} on {len(rows)} candidates", flush=True)
            components = fit_components(rr.feature_rows(rows), examples["train"], labels,
                                        sample_weight=weights, C=C)
            if any(n.startswith("relation=") for n in components["meta"].get_feature_names_out()):
                raise ValueError("Relation feature survived ablation")
            metrics, _, _ = rr.rank_conditions(components, validation, examples["validation"])
            score = float(np.mean([metrics[f"{n}_logistic"]["mrr_at5"] for n in names]))
            payload = {**baseline_payloads[selector], "components": components, "selected_C": C,
                       "schema": "sft-ensemble-logistic-no-relations-v1",
                       "feature_policy": "ignore_relation_types"}
            trials[selector].append({"C": C, "mean_validation_mrr_at5": score})
            if C == fixed_C:
                fixed = payload
            if best is None or score > best[0]:
                best = (score, payload)
        fitted[f"{selector}_fixed_C"] = fixed
        fitted[f"{selector}_validation_selected"] = best[1]
    return fitted, trials


def compare(pools, examples, baseline, blind_models, resamples):
    conditions, questions, contrasts = {}, {}, {}
    for selector, payload in baseline.items():
        cc, ss, _ = ev.rank_frozen(payload, pools, examples)
        conditions.update(cc); questions.update(ss)
    blind = without_relations(pools)
    for name, payload in blind_models.items():
        cc, ss, _ = ev.rank_frozen({**payload, "selector": name}, blind, examples)
        conditions.update(cc); questions.update(ss)
        selector = name.split("_", 1)[0]
        for pool in pools:
            before = f"{pool}_{selector}_reranker"
            after = f"{pool}_{name}_reranker"
            if cc[after]["metrics"]["coverage_full_pool"] != conditions[before]["metrics"]["coverage_full_pool"]:
                raise ValueError("Ablation changed pool coverage")
            result = ev.paired_contrast(questions[before], questions[after], resamples=resamples, metrics=ev.METRICS)
            contrasts[f"{pool}_{name}_minus_with_relations"] = {
                "before": before, "after": after, "metrics": {
                    m: {"difference": v["sft_minus_base"], "paired_bootstrap_95ci": v["paired_bootstrap_95ci"]}
                    for m, v in result.items()}}
    return {"question_count": len(examples), "conditions": conditions,
            "paired_contrasts": contrasts, "resamples": resamples, "seed": 3407}, questions


def run(submission, output, resamples=10000):
    if output.exists():
        raise FileExistsError(output)
    if resamples < 2:
        raise ValueError("Need at least two bootstrap resamples")
    prepared = ev.read(submission / "prepared.json")
    verify_files(submission / "code", prepared["code_hashes"])
    baseline_dir, comparison = Path(prepared["baseline"]), Path(prepared["comparison"])
    verify_files(baseline_dir, prepared["baseline_hashes"])
    verify_files(comparison, prepared["comparison_hashes"])
    _, test_pools, test_examples, gold, _ = ev.load_comparison(comparison)
    state, baseline, hashes = ev.load_frozen(baseline_dir, test_examples, comparison)
    examples, pools = load_training(baseline_dir, hashes)
    original_prepared = ev.read(Path(state["submission"]) / "prepared.json")
    C_grid = original_prepared["C_grid"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "submission": str(submission),
                "prepared_sha256": ev.digest(submission / "prepared.json"),
                "feature_policy": "ignore_relation_types", "generation": False, "DPO_training": False}
    ev.write(output / "manifest.json", manifest)
    try:
        fitted, trials = fit_ablation(pools, examples, baseline, C_grid)
        # Freeze models before looking at test outcomes.
        models = {}
        for name, payload in fitted.items():
            path = output / f"{name}.joblib"
            joblib.dump(payload, path)
            models[name] = {"selected_C": payload["selected_C"], "sha256": ev.digest(path)}
        ev.write(output / "validation_selection.json", {"trials": trials, "models": models})
        for split, pp, ee in (("validation", pools["validation"], examples["validation"]),
                               ("test", test_pools, test_examples)):
            result, per_question = compare(pp, ee, baseline, fitted, resamples)
            if split == "test":
                for name, condition in result["conditions"].items():
                    condition["per_batch"] = {}
                    for batch in sorted({r["batch"] for r in gold.values()}):
                        selected = [r for r in per_question[name] if gold[r["question_id"]]["batch"] == batch]
                        condition["per_batch"][batch] = {"question_count": len(selected), **{
                            metric: sum(r[metric] for r in selected) / len(selected) for metric in ev.METRICS}}
            result["limitations"] = [
                "Validation selects C; its scores are selection diagnostics.",
                "Test95 has been inspected previously; this is an exploratory ablation.",
                "Pointwise paired bootstrap intervals are not adjusted for multiple comparisons.",
                "Training pools use four samples; test pools use ten samples.",
                "No new DPO preferences are produced by this experiment."]
            ev.write(output / f"{split}_summary.json", result)
            ev.write(output / f"{split}_per_question.json", per_question)
            print(f"\n{split}: no-relations minus with-relations MRR@5", flush=True)
            for name, contrast in result["paired_contrasts"].items():
                m = contrast["metrics"]["mrr_at5"]
                before = result["conditions"][contrast["before"]]["metrics"]["mrr_at5"]
                after = result["conditions"][contrast["after"]]["metrics"]["mrr_at5"]
                print(f"  {name}: {before:.6f} -> {after:.6f}; delta={m['difference']:+.6f}; "
                      f"95% CI={m['paired_bootstrap_95ci']}", flush=True)
        verify_files(baseline_dir, prepared["baseline_hashes"])
        verify_files(comparison, prepared["comparison_hashes"])
        manifest.update(status="complete", models=models, question_counts={"train": 1296, "validation": 144, "test": len(test_examples)})
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        ev.write(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "run"))
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.baseline is None or args.comparison is None:
            parser.error("Preparation needs --baseline and --comparison")
        prepare(args.repository.resolve(), args.baseline.resolve(), args.comparison.resolve(), args.submission.resolve())
    else:
        if args.output is None:
            parser.error("Execution needs --output")
        run(args.submission.resolve(), args.output.resolve(), args.resamples)
