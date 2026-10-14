#!/usr/bin/env python3
"""Fit original-only, max-five-only and shared frozen logistic selectors."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))

import joblib
import numpy as np
import sklearn

from scripts.train_candidate_reranker_pilot import (
    clean, key, fit_components, score_components, write_json, write_jsonl,
)
from scripts.compare_expansion_pool_rerankers import (
    MODELS, question_weights, evaluate_rankings, tie_key, round_robin,
)
from src.notebook_workflows.coverage_comparison import official_candidate_matches

ARMS = ("original", "max5")
MODES = ("greedy", "sampling4")
SELECTORS = {"original": ("original",), "max5": ("max5",), "shared": ARMS}
ORIGINAL_JOBS = dict(zip(MODELS, ("181032357", "181032359", "181032361")))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_max5_banks(starter):
    latest = {}
    records = sorted((starter / "outputs/max5_dpo_banks_8b").glob("submission-*.tsv"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for path in records:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("stage") == "full" and row.get("model") in MODELS:
                    latest.setdefault(row["model"], row["job_id"])
    if set(latest) != set(MODELS):
        raise ValueError("Need submitted full max-five banks for all three models")
    if any(not re.fullmatch(r"\d+\.gadi-pbs", job) for job in latest.values()):
        raise ValueError("Invalid recorded PBS job ID")
    return {m: starter / "outputs/max5_dpo_banks_8b" / f"{m}-generate-{latest[m]}"
            for m in MODELS}, latest


def pending_dependencies(paths, jobs):
    pending = []
    for model in MODELS:
        manifest = paths[model] / "manifest.json"
        if manifest.exists() and read(manifest).get("status") == "complete":
            continue  # Full bank integrity is checked by the training worker.
        result = subprocess.run(["qstat", "-fx", jobs[model]], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"Cannot confirm max-five bank {model}: {result.stderr.strip()}")
        state = re.search(r"job_state\s*=\s*(\w+)", result.stdout)
        exit_status = re.search(r"Exit_status\s*=\s*(-?\d+)", result.stdout)
        if not state or state[1] not in ("Q", "H", "R", "E"):
            detail = exit_status[1] if exit_status else "not recorded"
            raise ValueError(f"{model} bank unavailable; PBS state={state[1] if state else '?'}; exit={detail}")
        pending.append(jobs[model])
    return pending


def original_banks(repository):
    starter = repository / "gadi_sft_8b_starter"
    # Set data roots before importing the bank validators. Their code can be a
    # frozen submission snapshot while large data remain in the real checkout.
    import prepare_matched_8b_sft as shared
    shared.ROOT = starter
    frozen_starter = ROOT / "gadi_sft_8b_starter"
    for module in tuple(sys.modules.values()):
        if getattr(module, "ROOT", None) == frozen_starter:
            module.ROOT = starter
            default = getattr(module, "DEFAULT_CONFIG", None)
            if isinstance(default, Path) and default.is_relative_to(frozen_starter):
                module.DEFAULT_CONFIG = starter / default.relative_to(frozen_starter)
    import single_answer_dpo_data as original
    config = read(starter / "configs/original_sft_dpo_8b.json")
    paths = [starter / "outputs/original_sft_dpo_8b" / f"{m}-mixed-generate-{ORIGINAL_JOBS[m]}.gadi-pbs"
             for m in MODELS]
    banks, rows = original.pack_banks(paths, config)
    if any(not b["manifest"]["full_dataset"] for b in banks.values()):
        raise ValueError("Original banks must contain the full fitting and validation panels")
    return banks, rows, original.audit_aliases(config), paths


def snapshot_code(repository, destination):
    hashes = {}
    for folder in ("scripts", "src", "gadi_sft_8b_starter/scripts"):
        for path in sorted((repository / folder).rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".java", ".jar", ".json") or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(repository)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative.as_posix())] = digest(target)
    return hashes


def prepare(repository, submission):
    repository = repository.resolve()
    starter = repository / "gadi_sft_8b_starter"
    banks, _, _, paths = original_banks(repository)
    max5_paths, jobs = latest_max5_banks(starter)
    dependencies = pending_dependencies(max5_paths, jobs)
    submission.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_code(repository, submission / "code")
    state = {
        "repository": str(repository), "max5_banks": {m: str(p) for m, p in max5_paths.items()},
        "original_banks": {m: str(p) for m, p in zip(MODELS, paths)},
        "original_bank_hashes": {m: b["manifest"]["responses_sha256"] for m, b in banks.items()},
        "dependencies": dependencies, "code_hashes": hashes,
        "config_hashes": {name: digest(starter / "configs" / name) for name in
                          ("original_sft_dpo_8b.json", "max5_dpo_banks_8b.json")},
        "C_grid": [0.1, 1.0, 10.0], "seed": 3407,
        "selectors": SELECTORS,
        "scope": "1296 fitting / 144 validation; no dev160 or test labels",
    }
    write_json(submission / "prepared.json", state)
    (submission / "dependencies.txt").write_text(":".join(dependencies), encoding="utf-8")
    print(f"Prepared CPU reranker; waiting for {len(dependencies)} max-five bank jobs", flush=True)


def contexts_from_rows(rows):
    examples = {s: {} for s in ("train", "validation")}
    for row in rows:
        if row["split"] not in examples or len(row["prompt"]) != 2 or any(m["role"] == "assistant" for m in row["prompt"]):
            raise ValueError("Expected gold-blind fitting/validation prompts")
        context = json.loads(row["prompt"][1]["content"])
        qid = row["question_id"]
        example = {"question_id": qid, "question": context["question"], "snippets": context["snippets"]}
        if qid in examples[row["split"]] and examples[row["split"]][qid] != example:
            raise ValueError("Question or evidence differs between models/formulations")
        examples[row["split"]][qid] = example
    if set(examples["train"]) & set(examples["validation"]):
        raise ValueError("Training/validation question overlap")
    return examples


def build_pools(rows, examples):
    """Keep original strings; source votes count models, not repeated draws."""
    from run_single_answer_sampling import parse_single_answer
    from run_extractive_expansion_8b import parse_equivalent_response
    pools = {s: {f"{a}_{m}": [] for a in ARMS for m in MODES} for s in examples}
    diagnostics = Counter()
    for split in examples:
        for arm in ARMS:
            for mode in MODES:
                merged = {}
                source_lengths = defaultdict(int)
                draws = 1 if mode == "greedy" else 4
                for row in sorted(rows, key=lambda r: (MODELS.index(r["model_key"]), r["question_id"], r["draw"])):
                    if row["split"] != split or row["arm"] != arm or (row["draw"] == 0) != (mode == "greedy"):
                        continue
                    try:
                        aa = ([{"answer": parse_single_answer(row["raw_response"]), "relation_type": "original"}]
                              if arm == "original" else parse_equivalent_response(row["raw_response"], 5)[0])
                    except (ValueError, TypeError):
                        aa = []
                    diagnostics[f"{split}/{arm}/{mode}/responses"] += 1
                    diagnostics[f"{split}/{arm}/{mode}/empty_responses"] += int(not aa)
                    model, qid = row["model_key"], row["question_id"]
                    seen = set()
                    for rank, candidate in enumerate(aa, 1):
                        identity = (qid, key(candidate["answer"]))
                        if identity in seen:
                            continue
                        seen.add(identity)
                        target = merged.setdefault(identity, {
                            "question_id": qid, "answer": clean(candidate["answer"]), "sources": [],
                            "source_ranks": {}, "relation_types": [], "surface_operations": [],
                            "is_format_variant": False, "sample_frequencies": {},
                        })
                        if model not in target["sources"]:
                            target["sources"].append(model)
                            source_lengths[(qid, model)] += 1
                            target["source_ranks"][model] = source_lengths[(qid, model)]
                        target["sample_frequencies"][model] = target["sample_frequencies"].get(model, 0) + 1 / draws
                        relation = candidate.get("relation_type", "unknown")
                        if relation not in target["relation_types"]:
                            target["relation_types"].append(relation)
                pools[split][f"{arm}_{mode}"] = list(merged.values())
    return pools, dict(diagnostics)


def label_pools(pools, examples, aliases, output, jar):
    for split in examples:
        distinct = {}
        for rows in pools[split].values():
            for row in rows:
                distinct.setdefault((row["question_id"], row["answer"]),
                                    {"question_id": row["question_id"], "answer": row["answer"]})
        report = output / "official_labels" / split
        report.mkdir(parents=True)
        labelled = [{**e, "gold_aliases": aliases[q]} for q, e in sorted(examples[split].items())]
        matches = official_candidate_matches(labelled, list(distinct.values()), report, jar_path=jar) if distinct else {}
        for rows in pools[split].values():
            for row in rows:
                row["label"] = int(matches[(row["question_id"], row["answer"])])


def feature_rows(rows):
    # Official labels never enter vectorizers or classifier input features.
    return [{k: v for k, v in r.items() if k != "label"} for r in rows]


def rank_conditions(components, pools, examples):
    result, per_question, rankings = {}, {}, []
    for name, rows in pools.items():
        groups = {qid: [] for qid in examples}  # Keep empty/failed pools.
        if rows:
            values = score_components(components, feature_rows(rows), examples)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite reranker predictions")
            for row, score in zip(rows, values):
                groups[row["question_id"]].append({**row, "score": float(score)})
        for method in ("round_robin", "model_vote", "rank_vote", "logistic"):
            if method == "round_robin":
                ordered = {q: round_robin(rr) for q, rr in groups.items()}
            else:
                def order(row):
                    if method == "model_vote":
                        value = len(row["sources"])
                    elif method == "rank_vote":
                        value = sum(1 / r for r in row["source_ranks"].values())
                    else:
                        value = row["score"]
                    return -value, tie_key(row)
                ordered = {q: sorted(rr, key=order) for q, rr in groups.items()}
            condition = f"{name}_{method}"
            result[condition], per_question[condition] = evaluate_rankings(ordered)
            if method == "logistic":
                rankings.extend({"condition": condition, "rank": i, **r}
                                for rr in ordered.values() for i, r in enumerate(rr, 1))
    return result, per_question, rankings


def fit_selector(pools, examples, C_grid, training_arms=ARMS):
    if not training_arms or not set(training_arms) <= set(ARMS):
        raise ValueError("Unknown or empty reranker training arms")
    selected_names = {f"{a}_{m}" for a in training_arms for m in MODES}
    train_rows, strata = [], []
    for name, rows in pools["train"].items():
        if name not in selected_names:
            continue
        train_rows.extend(rows)
        strata.extend([name] * len(rows))
    labels = np.asarray([r["label"] for r in train_rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Need both accepted and unaccepted fitting candidates")
    weights = question_weights(train_rows, strata)
    trials, best = [], None
    for C in C_grid:
        print(f"Fitting logistic C={C} on {len(train_rows)} candidates", flush=True)
        components = fit_components(feature_rows(train_rows), examples["train"], labels,
                                    sample_weight=weights, C=C)
        validation = {name: rows for name, rows in pools["validation"].items() if name in selected_names}
        metrics, questions, rankings = rank_conditions(components, validation, examples["validation"])
        score = float(np.mean([v["mrr_at5"] for k, v in metrics.items() if k.endswith("_logistic")]))
        trials.append({"C": C, "mean_validation_mrr_at5": score, "metrics": metrics})
        if best is None or score > best[0]:
            best = (score, C, components, metrics, questions, rankings)
    return best, trials


def fit_three_selectors(pools, examples, C_grid):
    """Select each model on its declared validation arms, then compare all pools."""
    fitted, metrics, questions, rankings = {}, {}, {}, []
    for selector, arms in SELECTORS.items():
        print(f"Training {selector} reranker from {', '.join(arms)} pools", flush=True)
        best, trials = fit_selector(pools, examples, C_grid, training_arms=arms)
        score, C, components, _, _, _ = best
        all_metrics, all_questions, all_rankings = rank_conditions(
            components, pools["validation"], examples["validation"])
        fitted[selector] = {"components": components, "selected_C": C,
                            "mean_validation_mrr_at5": score, "trials": trials,
                            "training_arms": list(arms),
                            "selection_conditions": [f"{a}_{m}" for a in arms for m in MODES]}
        for condition, values in all_metrics.items():
            name = (condition.removesuffix("_logistic") + f"_{selector}_reranker"
                    if condition.endswith("_logistic") else condition)
            metrics[name] = values
            questions[name] = all_questions[condition]
        rankings.extend({**r, "condition": r["condition"].removesuffix("_logistic") + f"_{selector}_reranker"}
                        for r in all_rankings)
    return fitted, metrics, questions, rankings


def save_selectors(output, fitted):
    summaries = {}
    for name, result in fitted.items():
        directory = output / name
        directory.mkdir()
        payload = {"components": result["components"], "schema": "sft-ensemble-logistic-v1",
                   "models": MODELS, "sample_frequency": "unique draw support / requested draws per model",
                   "selector": name, "training_arms": result["training_arms"],
                   "selection_conditions": result["selection_conditions"], "selected_C": result["selected_C"]}
        model_file = directory / "reranker.joblib"
        joblib.dump(payload, model_file)
        summary = {k: v for k, v in result.items() if k != "components"}
        write_json(directory / "validation_selection.json", summary)
        summaries[name] = {**summary, "model_path": f"{name}/reranker.joblib",
                           "model_sha256": digest(model_file)}
    # Preserve the original shared-selector path for existing consumers.
    shutil.copy2(output / "shared/reranker.joblib", output / "reranker.joblib")
    return summaries


def run(submission, output):
    prepared = read(submission / "prepared.json")
    repository = Path(prepared["repository"])
    for relative, value in prepared["code_hashes"].items():
        if digest(submission / "code" / relative) != value:
            raise ValueError("Frozen training code changed")
    starter = repository / "gadi_sft_8b_starter"
    for name, value in prepared["config_hashes"].items():
        if digest(starter / "configs" / name) != value:
            raise ValueError("Generation config changed since preparation")
    _, original_rows, original_aliases, _ = original_banks(repository)
    import max5_dpo_data as expansion
    cfg = read(starter / "configs/max5_dpo_banks_8b.json")
    banks, max5_rows = expansion.pack_banks([Path(prepared["max5_banks"][m]) for m in MODELS], cfg)
    _, aliases = expansion.panels(cfg)
    rows = [{**r, "arm": arm} for arm, rr in (("original", original_rows), ("max5", max5_rows)) for r in rr]
    examples = contexts_from_rows(rows)
    if {s: len(e) for s, e in examples.items()} != {"train": 1296, "validation": 144}:
        raise ValueError("Need full 1296/144 panels")
    if any(original_aliases[q] != aliases[q] for e in examples.values() for q in e):
        raise ValueError("Aliases differ across formulations")
    for model in MODELS:
        path = Path(prepared["original_banks"][model]) / "responses.jsonl"
        if digest(path) != prepared["original_bank_hashes"][model]:
            raise ValueError("Original bank changed after preparation")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json")}
    write_json(output / "manifest.json", state)
    try:
        pools, diagnostics = build_pools(rows, examples)
        label_pools(pools, examples, aliases, output,
                    repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
        fitted, metrics, questions, rankings = fit_three_selectors(pools, examples, prepared["C_grid"])
        summaries = save_selectors(output, fitted)
        shared = fitted["shared"]
        model_file = output / "reranker.joblib"
        write_json(output / "validation_summary.json", {"question_count": 144, "selected_C": shared["selected_C"],
                    "mean_validation_mrr_at5": shared["mean_validation_mrr_at5"],
                    "trials": shared["trials"], "selectors": summaries, "metrics": metrics,
                    "interpretation": "Validation diagnostics used for C selection; not independent test results"})
        write_json(output / "validation_per_question.json", questions)
        write_jsonl(output / "validation_rankings.jsonl", rankings)
        for split in examples:
            write_jsonl(output / f"{split}_examples.jsonl", examples[split].values())
            for name, rr in pools[split].items():
                write_jsonl(output / f"{split}_{name}_candidates.jsonl", rr)
        state.update(status="complete", selected_C={name: r["selected_C"] for name, r in fitted.items()},
                     selectors={name: {"training_arms": r["training_arms"], "model_path": r["model_path"],
                                       "model_sha256": r["model_sha256"]} for name, r in summaries.items()},
                     question_counts={s: len(e) for s, e in examples.items()},
                     training_scope="Fitting only; validation selects C; neither dev160 nor test used",
                     labels="Official acceptance only; unmatched answers may be semantically correct",
                     sampling="one greedy and four sampled draws; not ten-sample generation",
                     generator_cross_fitting=False, model_sha256=digest(model_file),
                     diagnostics=diagnostics, sklearn_version=sklearn.__version__,
                     max5_bank_hashes={m: b["manifest"]["responses_sha256"] for m, b in banks.items()},
                     artifacts_sha256={p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*")
                                       if p.is_file() and p != output / "manifest.json"})
        for name, result in summaries.items():
            print(f"Frozen {name} reranker: {output / result['model_path']}; selected C={result['selected_C']}; "
                  f"validation selection MRR={result['mean_validation_mrr_at5']:.6f}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "manifest.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.repository, args.submission)
    else:
        if args.output is None:
            parser.error("--output is required for training")
        run(args.submission.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
