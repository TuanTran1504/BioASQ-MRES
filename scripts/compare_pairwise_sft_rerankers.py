#!/usr/bin/env python3
"""Same-pool pairwise ranking and typography-redundancy selection experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scripts import evaluate_sft_ensemble_rerankers as ev
from scripts import train_sft_ensemble_reranker as rr
from scripts.ablate_reranker_relation_labels import load_training, verify_files
from scripts.train_candidate_reranker_pilot import candidate_features, make_text, score_components

C_GRID = (0.1, 1.0, 10.0)
PENALTIES = (0.0, 0.1, 0.3, 1.0)
PAIR_CAP = 128
SEED = 3407


def contexts(examples):
    return {q: {"question": e["question"], "snippets": e["snippets"]}
            for q, e in examples.items()}


def feature_matrix(components, rows, examples, fit=False):
    """Use the baseline feature definitions, with fitting vocabulary only."""
    clean = rr.feature_rows([{**r, "surface_operations": [], "is_format_variant": False} for r in rows])
    ee = contexts(examples)
    inputs = ([make_text(r, ee[r["question_id"]]) for r in clean],
              [r["answer"] for r in clean],
              [candidate_features(r, ee[r["question_id"]]) for r in clean])
    return hstack([getattr(components[n], "fit_transform" if fit else "transform")(values)
                   for n, values in zip(("word", "char", "meta"), inputs)], format="csr")


def make_pairs(rows, strata, cap=PAIR_CAP):
    """Accepted/nonaccepted pairs within one question AND one pool only.

    Uniform, label-defined pair sampling has no dependence on validation/test.
    Each eligible question/pool contributes equal total loss weight. Both
    orientations are included; nonacceptance is not a semantic-error label.
    """
    if cap < 1 or len(rows) != len(strata):
        raise ValueError("Invalid pair cap or strata")
    groups = defaultdict(list)
    for i, (r, pool) in enumerate(zip(rows, strata)):
        if type(r.get("label")) is not int or r["label"] not in (0, 1):
            raise ValueError("Need binary official acceptance labels")
        groups[(r["question_id"], pool)].append(i)
    positive, negative, weights, details = [], [], [], []
    for (qid, pool), ids in sorted(groups.items()):
        pp = sorted((i for i in ids if rows[i]["label"]), key=lambda i: rr.key(rows[i]["answer"]))
        nn = sorted((i for i in ids if not rows[i]["label"]), key=lambda i: rr.key(rows[i]["answer"]))
        total = len(pp) * len(nn)
        if not total:
            continue
        seed = int.from_bytes(hashlib.sha256(f"{SEED}/{qid}/{pool}".encode()).digest()[:8], "big")
        choices = (np.arange(total) if total <= cap else
                   np.sort(np.random.default_rng(seed).choice(total, cap, replace=False)))
        for k in choices:
            positive.append(pp[int(k) // len(nn)])
            negative.append(nn[int(k) % len(nn)])
            weights.append(1.0 / len(choices))
        details.append({"question_id": qid, "pool": pool, "available_pairs": total,
                        "sampled_pairs": len(choices)})
    if not positive:
        raise ValueError("No within-question accepted/nonaccepted pairs")
    return np.array(positive), np.array(negative), np.array(weights), {
        "eligible_question_pools": len(details), "skipped_question_pools": len(groups) - len(details),
        "unordered_pairs": len(positive), "oriented_pairs": 2 * len(positive),
        "capped_question_pools": sum(d["available_pairs"] > cap for d in details), "groups": details}


def fit_pairwise(pools, examples, arms, C_grid=C_GRID, cap=PAIR_CAP):
    names = [f"{a}_{m}" for a in arms for m in rr.MODES]
    rows = [r for n in names for r in pools["train"][n]]
    strata = [n for n in names for _ in pools["train"][n]]
    components = {
        "word": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        "char": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                               max_features=30000, sublinear_tf=True),
        "meta": DictVectorizer(sparse=True),
    }
    matrix = feature_matrix(components, rows, examples["train"], fit=True)
    pp, nn, weights, pair_stats = make_pairs(rows, strata, cap)
    differences = matrix[pp] - matrix[nn]
    design = vstack((differences, -differences), format="csr")
    labels = np.r_[np.ones(len(pp), dtype=int), np.zeros(len(pp), dtype=int)]
    weights = np.tile(weights, 2)
    weights /= weights.mean()
    validation = {n: pools["validation"][n] for n in names}
    best, trials = None, []
    for C in C_grid:
        print(f"Pairwise {','.join(arms)}: C={C}; {len(pp)} pairs", flush=True)
        model = LogisticRegression(C=C, fit_intercept=False, class_weight=None,
                                   solver="liblinear", max_iter=2000, random_state=SEED)
        model.fit(design, labels, sample_weight=weights)
        trial = {**components, "model": model}
        payload = {"components": trial, "selector": "trial"}
        metrics, _, _ = ev.rank_frozen(payload, validation, contexts(examples["validation"]))
        score = float(np.mean([v["metrics"]["mrr_at5"] for v in metrics.values()]))
        trials.append({"C": C, "mean_validation_mrr_at5": score})
        if best is None or score > best[0]:
            best = (score, C, trial)
    return {"schema": "sft-ensemble-pairwise-linear-v1", "components": best[2],
            "selected_C": best[1], "mean_validation_mrr_at5": best[0],
            "training_arms": list(arms), "selection_conditions": names,
            "trials": trials, "pair_statistics": pair_stats}


def typography_signature(answer):
    """Comparison only: no edited/new answer and no biomedical alias claim.

    Preserve all digits, decimal points, slashes, percent signs, Greek letters
    and word order. Only between-letter hyphens/spaces and terminal full stops
    can become redundant. This deliberately leaves most semantic aliases alone.
    """
    value = rr.key(answer)
    value = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", value)
    value = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", " ", value)
    return re.sub(r"\s+", " ", value.rstrip(".").strip())


def diversify(ordered, penalty):
    """Soft redundancy penalty on reciprocal base rank; retain the full pool."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Invalid redundancy penalty")
    if not penalty or len(ordered) < 2:
        return list(ordered)
    remaining = list(enumerate(ordered, 1))
    output, seen = [], set()
    # Only the first five are selected jointly; the remainder retains base order.
    # No candidate is removed, and the original top-ranked candidate stays first.
    while remaining and len(output) < 5:
        best = min(range(len(remaining)), key=lambda i: (
            -(1.0 / remaining[i][0] - penalty * int(typography_signature(remaining[i][1]["answer"]) in seen)),
            remaining[i][0]))
        _, row = remaining.pop(best)
        output.append(row)
        seen.add(typography_signature(row["answer"]))
    return output + [r for _, r in remaining]


def rank(payload, pools, examples, penalty=0.0):
    conditions, questions, rankings = {}, {}, []
    ee = contexts(examples)
    for name, rows in pools.items():
        values = score_components(payload["components"], rr.feature_rows([
            {**r, "surface_operations": [], "is_format_variant": False} for r in rows]), ee) if rows else []
        if not np.isfinite(values).all():
            raise ValueError("Non-finite candidate scores")
        groups = {q: [] for q in ee}
        for r, value in zip(rows, values):
            groups[r["question_id"]].append({**r, "score": float(value)})
        ordered = {q: diversify(sorted(group, key=lambda r: (-r["score"], rr.tie_key(r))), penalty)
                   for q, group in groups.items()}
        metrics, scored = rr.evaluate_rankings(ordered)
        conditions[name] = {"metrics": metrics, "mean_unique_candidates": len(rows) / len(ee)}
        questions[name] = scored
        rankings.extend({"pool": name, "rank": i, **r} for group in ordered.values()
                        for i, r in enumerate(group, 1))
    return conditions, questions, rankings


def select_penalty(payload, pools, examples, arms, grid=PENALTIES):
    names = [f"{a}_{m}" for a in arms for m in rr.MODES]
    trials, best = [], None
    for value in grid:
        metrics, _, _ = rank(payload, {n: pools[n] for n in names}, examples, value)
        score = float(np.mean([c["metrics"]["mrr_at5"] for c in metrics.values()]))
        trials.append({"penalty": value, "mean_validation_mrr_at5": score})
        if best is None or score > best[0]:
            best = (score, value)
    return {"selected_penalty": best[1], "mean_validation_mrr_at5": best[0], "trials": trials}


def prepare(repository, baseline, comparison, submission):
    _, _, test, _, source_hashes = ev.load_comparison(comparison)
    _, _, model_hashes = ev.load_frozen(baseline, test, comparison)
    load_training(baseline, model_hashes)
    submission.mkdir(parents=True, exist_ok=False)
    code_hashes = rr.snapshot_code(repository, submission / "code")
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/compare_pairwise_sft_rerankers.pbs", submission / "job.pbs")
    ev.write(submission / "prepared.json", {
        "baseline": str(baseline), "comparison": str(comparison), "baseline_hashes": model_hashes,
        "comparison_hashes": source_hashes, "code_hashes": code_hashes,
        "job_sha256": ev.digest(submission / "job.pbs"), "C_grid": C_GRID,
        "penalties": PENALTIES, "pair_cap": PAIR_CAP, "seed": SEED,
        "scope": "Same-pool pairwise ranking and typography redundancy; no generation, API or DPO"})
    print("Prepared one CPU comparison; verified frozen sources and question separation.", flush=True)


def compare(pools, examples, models, selections, gold, resamples):
    conditions, questions, rankings, contrasts = {}, {}, [], {}
    for method, payload in models.items():
        for policy, penalty in (("plain", 0), ("diverse", selections[method]["selected_penalty"])):
            cc, ss, ranked = rank(payload, pools, examples, penalty)
            for pool in pools:
                name = f"{pool}_{method}_{policy}"
                conditions[name], questions[name] = cc[pool], ss[pool]
                if gold is not None:
                    cc[pool]["per_batch"] = {}
                    for batch in sorted({r["batch"] for r in gold.values()}):
                        subset = [r for r in ss[pool] if gold[r["question_id"]]["batch"] == batch]
                        cc[pool]["per_batch"][batch] = {"question_count": len(subset), **{
                            m: sum(r[m] for r in subset) / len(subset) for m in ev.METRICS}}
            rankings.extend({"condition": f"{r['pool']}_{method}_{policy}", **r} for r in ranked)
    for pool in pools:
        if len({v["metrics"]["coverage_full_pool"] for n, v in conditions.items() if n.startswith(pool + "_")}) != 1:
            raise ValueError("Ranking changed full-pool coverage")
        for selector in rr.SELECTORS:
            base = f"{pool}_{selector}_logistic_plain"
            pairs = [(base, f"{pool}_{selector}_pairwise_plain"),
                     (base, f"{pool}_{selector}_logistic_diverse"),
                     (base, f"{pool}_{selector}_pairwise_diverse"),
                     (f"{pool}_{selector}_pairwise_plain", f"{pool}_{selector}_pairwise_diverse")]
            for before, after in pairs:
                delta = ev.paired_contrast(questions[before], questions[after], resamples=resamples, metrics=ev.METRICS)
                aa = {r["question_id"]: r for r in questions[after]}
                gained = [r["question_id"] for r in questions[before] if not r["coverage_at5"] and aa[r["question_id"]]["coverage_at5"]]
                lost = [r["question_id"] for r in questions[before] if r["coverage_at5"] and not aa[r["question_id"]]["coverage_at5"]]
                contrasts[f"{after}_minus_{before}"] = {"before": before, "after": after,
                    "coverage_at5_transitions": {"gained": len(gained), "lost": len(lost),
                                                  "gained_question_ids": gained, "lost_question_ids": lost},
                    "metrics": {m: {"difference": v["sft_minus_base"], "paired_bootstrap_95ci": v["paired_bootstrap_95ci"]}
                                for m, v in delta.items()}}
    return {"question_count": len(examples), "conditions": conditions, "paired_contrasts": contrasts,
            "resamples": resamples, "seed": SEED}, questions, rankings


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
    _, test_pools, test, gold, _ = ev.load_comparison(comparison)
    _, baseline, hashes = ev.load_frozen(baseline_dir, test, comparison)
    examples, pools = load_training(baseline_dir, hashes)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "submission": str(submission),
             "prepared_sha256": ev.digest(submission / "prepared.json"), "generation": False,
             "paid_judging": False, "DPO_training": False, "sklearn_version": ev.sklearn.__version__}
    ev.write(output / "manifest.json", state)
    try:
        models, selections, model_hashes = {}, {}, {}
        for selector, arms in rr.SELECTORS.items():
            pairwise = fit_pairwise(pools, examples, arms, prepared["C_grid"], prepared["pair_cap"])
            for objective, payload in (("logistic", baseline[selector]), ("pairwise", pairwise)):
                name = f"{selector}_{objective}"
                models[name] = payload
                selections[name] = select_penalty(payload, pools["validation"], examples["validation"],
                                                  arms, prepared["penalties"])
                if objective == "pairwise":
                    path = output / f"{name}.joblib"
                    joblib.dump(payload, path)
                    model_hashes[path.name] = ev.digest(path)
                    ev.write(output / f"{name}_training.json", {k: v for k, v in payload.items() if k != "components"})
        ev.write(output / "validation_selection.json", {"selections": selections, "pairwise_model_hashes": model_hashes,
                 "selection": "C first by plain validation MRR; redundancy penalty second; no test selection"})
        # Freeze every model and selection rule before producing test scores.
        for split, pp, ee, gg in (("validation", pools["validation"], examples["validation"], None),
                                  ("test", test_pools, test, gold)):
            result, questions, rankings = compare(pp, ee, models, selections, gg, resamples)
            result["limitations"] = ["Validation scores select C and redundancy penalties; they are not held-out gains.",
                "Test95 was inspected previously; this is an exploratory comparison.",
                "Fitting pools have four sampled draws per model; test pools have ten.",
                "Official nonacceptance is not a semantic incorrectness label.",
                "Typography redundancy does not identify biomedical aliases; no strings are edited or removed.",
                "Bootstrap intervals are pointwise and unadjusted for multiple comparisons."]
            ev.write(output / f"{split}_summary.json", result)
            ev.write(output / f"{split}_per_question.json", questions)
            rr.write_jsonl(output / f"{split}_rankings.jsonl", rankings)
            for pool in pp:
                selector = "max5" if pool.startswith("max5_") else "original"
                print(f"\n{split} {pool} (matching-pool selector)", flush=True)
                for objective in ("logistic", "pairwise"):
                    for policy in ("plain", "diverse"):
                        name = f"{pool}_{selector}_{objective}_{policy}"
                        m = result["conditions"][name]["metrics"]
                        print(f"  {objective}_{policy}: MRR@5={m['mrr_at5']:.6f}; "
                              f"coverage@5={m['coverage_at5']:.4f}; full pool={m['coverage_full_pool']:.4f}", flush=True)
        verify_files(baseline_dir, prepared["baseline_hashes"])
        verify_files(comparison, prepared["comparison_hashes"])
        verify_files(submission / "code", prepared["code_hashes"])
        if any(ev.digest(output / p) != value for p, value in model_hashes.items()):
            raise ValueError("Frozen pairwise model changed")
        state.update(status="complete", question_counts={s: len(ee) for s, ee in examples.items()},
                     test_question_count=len(test), pairwise_model_hashes=model_hashes)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        ev.write(output / "manifest.json", state)


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
            parser.error("Preparation requires --baseline and --comparison")
        prepare(args.repository.resolve(), args.baseline.resolve(), args.comparison.resolve(), args.submission.resolve())
    else:
        if args.output is None:
            parser.error("Execution requires --output")
        run(args.submission.resolve(), args.output.resolve(), args.resamples)
