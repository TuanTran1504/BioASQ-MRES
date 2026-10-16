#!/usr/bin/env python3
"""Prepare natural generator preferences using frozen ensemble-ranking feedback."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from sklearn.model_selection import KFold

from scripts import prepare_reranker_guided_dpo as audit
from scripts.ablate_reranker_relation_labels import load_training, verify_files

MODELS = audit.MODELS
ARMS = ("ranking_gain", "compactness", "retain_needed_expansion")


def validate_folds(directory, examples, config):
    state = audit.read(directory / "manifest.json")
    if (state.get("status") not in ("review_required", "insufficient_primary_pairs")
            or state.get("configuration") != config or state.get("question_count") != len(examples)):
        raise ValueError("Need the completed matching out-of-fold audit")
    hashes = {"manifest.json": audit.digest(directory / "manifest.json")}
    declared = state["output_sha256"]
    folds = audit.read(directory / "folds.json")
    qids = sorted(examples)
    wanted = list(KFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(qids))
    if len(folds) != len(wanted):
        raise ValueError("Incomplete feedback folds")
    for number, (fold, (fit, held)) in enumerate(zip(folds, wanted), 1):
        name = f"fold-{number}.joblib"
        expected_fit, expected_held = sorted(qids[i] for i in fit), sorted(qids[i] for i in held)
        if (fold["fold"] != number or fold["fitting_question_ids"] != expected_fit
                or fold["held_out_question_ids"] != expected_held
                or fold["model_sha256"] != declared[name]):
            raise ValueError("Feedback fold membership or hashes differ")
        if audit.digest(directory / name) != declared[name]:
            raise ValueError("Feedback weights changed")
        payload = joblib.load(directory / name)
        if (payload["C"] != config["reranker_C"]
                or payload["fitting_question_ids"] != expected_fit
                or payload["held_out_question_ids"] != expected_held):
            raise ValueError("Feedback weights or their question membership changed")
        hashes[name] = declared[name]
    if audit.digest(directory / "folds.json") != declared["folds.json"]:
        raise ValueError("Fold manifest changed")
    hashes["folds.json"] = declared["folds.json"]
    return folds, hashes


def load_sources(repository, rerankers, previous):
    config = audit.read(repository / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json")
    fitting, _, _, _, hashes, bank_hashes, data = audit.load_inputs(repository, rerankers, config)
    previous_state = audit.read(previous / "manifest.json")
    previous_prepared = Path(previous_state["submission"]) / "prepared.json"
    if audit.digest(previous_prepared) != previous_state["prepared_sha256"]:
        raise ValueError("Original feedback preparation metadata changed")
    provenance = audit.read(previous_prepared)
    if provenance["reranker_hashes"] != hashes or provenance["banks"] != bank_hashes:
        raise ValueError("Feedback folds came from different reranker artifacts or banks")
    folds, fold_hashes = validate_folds(previous, fitting, config)
    examples, pools = load_training(rerankers, hashes)
    generation = audit.read(data.DEFAULT_CONFIG)
    _, raw = data.pack_banks([Path(bank_hashes[m]["path"]) for m in MODELS], generation)
    labels = {}
    for split, pp in pools.items():
        for name in ("max5_greedy", "max5_sampling4"):
            for r in pp[name]:
                identity = (split, r["question_id"], audit.key(r["answer"]))
                if identity in labels and labels[identity] != r["label"]:
                    raise ValueError("Conflicting official acceptance after normalization")
                labels[identity] = r["label"]
    baseline_state = audit.read(rerankers / "manifest.json")
    path = rerankers / baseline_state["selectors"]["max5"]["model_path"]
    if audit.digest(path) != baseline_state["selectors"]["max5"]["model_sha256"]:
        raise ValueError("Frozen validation feedback model changed")
    payload = joblib.load(path)
    if payload.get("selector") != "max5" or payload.get("schema") != "sft-ensemble-logistic-v1":
        raise ValueError("Need the archived max-five logistic feedback model")
    return config, examples, raw, labels, folds, hashes, bank_hashes, fold_hashes, payload, data


def make_options(rows, examples, labels, split, focal, config, data):
    """Change one saved focal response, keeping eleven sampled companions fixed."""
    groups = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            if r["question_id"] not in examples:
                raise ValueError("Response outside declared split")
            groups[r["question_id"]].append({**r, "arm": "max5"})
    output = {}
    split_labels = {(q, a): value for (s, q, a), value in labels.items() if s == split}
    for qid in sorted(examples):
        group = groups[qid]
        if len(group) != 15 or {(r["model_key"], r["draw"]) for r in group} != {(m, d) for m in MODELS for d in range(5)}:
            raise ValueError("Need all three models and all five saved draws")
        if any(r["prompt"] != group[0]["prompt"] for r in group):
            raise ValueError("Native prompts differ")
        slot = audit.replacement_slot(qid, config["seed"], 4)
        fixed = [r for r in group if r["draw"] > 0 and not (r["model_key"] == focal and r["draw"] == slot)]
        fixed_ids = [r["response_id"] for r in sorted(fixed, key=lambda r: (r["model_key"], r["draw"]))]
        if len(fixed_ids) != 11:
            raise ValueError("Expected eleven unchanged sampled responses")
        options = []
        for response in sorted((r for r in group if r["model_key"] == focal), key=lambda r: r["draw"]):
            projected = [*fixed, {**response, "draw": slot}]
            pools, _ = audit.rr.build_pools(projected, {split: {qid: examples[qid]}})
            pool = pools[split]["max5_sampling4"]
            for r in pool:
                r["label"] = split_labels[(qid, audit.key(r["answer"]))]
            options.append({"response": response, "summary": audit.response_summary(response, split_labels, data),
                            "pool": pool, "replacement_slot": slot, "fixed_response_ids": fixed_ids,
                            "baseline": response["draw"] == slot})
        output[qid] = options
    return output


def content(row, data):
    return tuple((audit.key(c["answer"]), c["relation_type"]) for c in data.answers(row["raw_response"]))


def ordered_subset(short, long):
    position = iter(long)
    return all(any(candidate == current for current in position) for candidate in short)


def build_preferences(scored, responses, labels, split, model, data, cap=2):
    """Natural, non-tag-only comparisons; each objective is exported separately."""
    groups, diagnostics = defaultdict(list), Counter()
    for r in scored:
        if responses[r["response_id"]]["split"] != split:
            raise ValueError("Feedback split differs from original response")
        groups[r["question_id"]].append(r)
    output = {arm: [] for arm in ARMS}
    for qid, group in sorted(groups.items()):
        choices, seen = {arm: [] for arm in ARMS}, {arm: set() for arm in ARMS}
        for chosen, rejected in itertools.permutations(group, 2):
            aa, bb = chosen["response_summary"], rejected["response_summary"]
            if not aa["eligible"] or not bb["eligible"]:
                diagnostics["invalid_or_ceiling"] += 1
                continue
            a, b = responses[chosen["response_id"]], responses[rejected["response_id"]]
            ac, bc = content(a, data), content(b, data)
            # Ignore relation labels when rejecting tag-only/case/space changes.
            if tuple(c[0] for c in ac) == tuple(c[0] for c in bc):
                diagnostics["tag_only_or_same_ordered_answers"] += 1
                continue
            if (a["prompt"] != b["prompt"] or a["model_key"] != model or b["model_key"] != model
                    or chosen["fixed_response_ids"] != rejected["fixed_response_ids"]
                    or chosen["fold"] != rejected["fold"]):
                raise ValueError("Mismatched preference context or model")
            gain = chosen["mrr_at5"] - rejected["mrr_at5"]
            if gain < -1e-12 or chosen["coverage_full_pool"] < rejected["coverage_full_pool"]:
                continue
            accepted_a = {text for text, _ in ac if labels[(split, qid, text)]}
            accepted_b = {text for text, _ in bc if labels[(split, qid, text)]}
            if not accepted_a or not accepted_b <= accepted_a:
                continue
            same_initial = ac[0] == bc[0]
            arms = []
            if gain > 1e-12:
                arms.append("ranking_gain")
            if same_initial and len(ac) < len(bc) and ordered_subset(ac, bc):
                arms.append("compactness")
            if same_initial and len(ac) > len(bc) and ordered_subset(bc, ac) and not accepted_b:
                arms.append("retain_needed_expansion")
            category = ("same_accepted_initial" if ac[0][0] == bc[0][0] and aa["initial_accepted"] and bb["initial_accepted"] else
                        "same_unaccepted_initial" if ac[0][0] == bc[0][0] else "changed_initial_expression")
            for arm in arms:
                identity = (ac, bc)
                if identity in seen[arm]:
                    continue
                seen[arm].add(identity)
                pair = {"question_id": qid, "split": split, "model_key": model, "objective": arm,
                        "category": category, "prompt": a["prompt"], "chosen": a["raw_response"],
                        "rejected": b["raw_response"], "chosen_id": a["response_id"], "rejected_id": b["response_id"],
                        "fold": chosen["fold"], "fixed_response_ids": chosen["fixed_response_ids"],
                        "replacement_slot": chosen["replacement_slot"], "reciprocal_rank_gain": gain,
                        "chosen_outcome": chosen, "rejected_outcome": rejected,
                        "synthetic_chosen": False, "semantic_review_status": "pending", "training_ready": False}
                choices[arm].append(pair)
                diagnostics[f"eligible_before_cap/{arm}"] += 1
        for arm in ARMS:
            ranked = sorted(choices[arm], key=lambda p: (
                -(p["chosen_outcome"]["coverage_at5"] - p["rejected_outcome"]["coverage_at5"]),
                -p["reciprocal_rank_gain"], p["chosen_id"], p["rejected_id"]))
            output[arm].extend(ranked[:cap])
    return output, dict(diagnostics)


def prepare(repository, rerankers, previous, submission, models):
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError("Unknown or duplicate focal models")
    config, _, _, _, _, hashes, banks, fold_hashes, _, _ = load_sources(repository, rerankers, previous)
    submission.mkdir(parents=True, exist_ok=False)
    code = audit.rr.snapshot_code(repository, submission / "code")
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/prepare_generator_feedback_preferences.pbs", submission / "job.pbs")
    audit.write(submission / "prepared.json", {"repository": str(repository), "rerankers": str(rerankers),
        "previous_audit": str(previous), "reranker_hashes": hashes, "banks": banks, "fold_hashes": fold_hashes,
        "code_hashes": code, "models": models, "audit_config": config,
        "job_sha256": audit.digest(submission / "job.pbs"), "cap_per_question_per_objective": 2})
    print(f"Prepared frozen generator-feedback preferences for {', '.join(models)}; review export only.", flush=True)


def run(submission, output):
    if output.exists():
        raise FileExistsError(output)
    prepared = audit.read(submission / "prepared.json")
    verify_files(submission / "code", prepared["code_hashes"])
    repository, rerankers, previous = (Path(prepared[k]) for k in ("repository", "rerankers", "previous_audit"))
    loaded = load_sources(repository, rerankers, previous)
    config, examples, raw, labels, folds, hashes, banks, fold_hashes, final, data = loaded
    if (hashes != prepared["reranker_hashes"] or banks != prepared["banks"]
            or fold_hashes != prepared["fold_hashes"] or config != prepared["audit_config"]):
        raise ValueError("Frozen sources changed after preparation")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "submission": str(submission), "prepared_sha256": audit.digest(submission / "prepared.json"),
             "models": prepared["models"], "training_ready": False, "DPO_training": False,
             "generation": False, "paid_judging": False}
    audit.write(output / "manifest.json", state)
    try:
        by_id = {r["response_id"]: r for r in raw}
        if len(by_id) != len(raw):
            raise ValueError("Duplicate raw response IDs")
        summaries, review_ids = {}, set()
        for model in prepared["models"]:
            directory = output / model
            directory.mkdir()
            summaries[model] = {}
            for split, ee in examples.items():
                options = make_options(raw, ee, labels, split, model, config, data)
                scores = []
                if split == "train":
                    for fold in folds:
                        held = fold["held_out_question_ids"]
                        payload = joblib.load(previous / f"fold-{fold['fold']}.joblib")
                        print(f"{model} fitting fold {fold['fold']}/5: reusing frozen feedback weights", flush=True)
                        scores.extend({**r, "fold": fold["fold"], "split": split, "model_key": model}
                                      for r in audit.score_options(payload["components"], {q: options[q] for q in held}, {q: ee[q] for q in held}))
                else:
                    scores = [{**r, "fold": "validation_frozen", "split": split, "model_key": model}
                              for r in audit.score_options(final["components"], options, ee)]
                if len(scores) != len(ee) * 5:
                    raise ValueError("Missing saved alternatives")
                pairs, diagnostics = build_preferences(scores, by_id, labels, split, model, data,
                                                       prepared["cap_per_question_per_objective"])
                audit.write_jsonl(directory / f"{split}_outcomes.jsonl", scores)
                counts = {}
                for arm, values in pairs.items():
                    audit.write_jsonl(directory / f"{split}_{arm}_review_candidates.jsonl", values)
                    counts[arm] = {"pairs": len(values), "questions": len({p["question_id"] for p in values}),
                                   "categories": dict(Counter(p["category"] for p in values))}
                    review_ids.update(p[k] for p in values for k in ("chosen_id", "rejected_id"))
                summaries[model][split] = {"questions": len(ee), "scored_alternatives": len(scores),
                                            "objectives": counts, "diagnostics": diagnostics}
                print(f"{model} {split}: {counts}", flush=True)
        reviews = []
        for identity in sorted(review_ids):
            row = data.review_template(by_id[identity])
            for c in row["candidates"]:
                c["official_accepted"] = bool(labels[(row["split"], row["question_id"], audit.key(c["answer"]))])
            reviews.append(row)
        audit.write_jsonl(output / "review_responses.jsonl", reviews)
        summary = {"models": summaries, "responses_requiring_review": len(reviews), "training_ready": False,
                   "feedback": "Fitting: frozen question-disjoint OOF models. Validation: frozen final max5 logistic model.",
                   "next_step": "Review candidate support/equivalence, then build separate validated DPO arms; no trainer accepts this provisional package.",
                   "limitations": ["Generator SFT saw fitting questions; only reranker feedback is out of fold.",
                       "Ranking preferences can change the initial answer; categories are explicit.",
                       "Compactness permits tied MRR and is a separate objective, not a demonstrated ranking improvement.",
                       "Official acceptance is not proof that every expansion is semantically valid.",
                       "Retain-needed-expansion controls must be considered before training compactness preferences.",
                       "No validation, dev160 or test questions contribute training preferences."]}
        audit.write(output / "summary.json", summary)
        verify_files(rerankers, prepared["reranker_hashes"])
        verify_files(previous, prepared["fold_hashes"])
        verify_files(submission / "code", prepared["code_hashes"])
        for bank in banks.values():
            verify_files(Path(bank["path"]), {"manifest.json": bank["manifest_sha256"], "responses.jsonl": bank["responses_sha256"]})
        state.update(status="review_required", summary=summary,
                     output_sha256={p.relative_to(output).as_posix(): audit.digest(p) for p in output.rglob("*")
                                    if p.is_file() and p.name != "manifest.json"})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        audit.write(output / "manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--rerankers", type=Path)
    parser.add_argument("--previous-audit", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["llama31"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.rerankers is None or args.previous_audit is None:
            parser.error("Preparation requires --rerankers and --previous-audit")
        prepare(args.repository.resolve(), args.rerankers.resolve(), args.previous_audit.resolve(), args.submission.resolve(), args.models)
    else:
        if args.output is None:
            parser.error("Execution requires --output")
        run(args.submission.resolve(), args.output.resolve())
