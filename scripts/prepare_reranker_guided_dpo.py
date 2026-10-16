#!/usr/bin/env python3
"""Audit out-of-fold reranker-guided max-five preferences; no DPO training.

One CPU workflow fits five selectors, changes one focal response at a time,
and exports provisional preferences requiring semantic/evidence review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import KFold

from scripts import train_sft_ensemble_reranker as rr
from scripts.train_candidate_reranker_pilot import key, fit_components, score_components
from prepare_matched_8b_sft import read, records, digest, write
from run_extractive_expansion_8b import write_jsonl

MODELS = rr.MODELS
DEFAULT_CONFIG = ROOT / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json"


def configure_data_root(repository):
    import prepare_matched_8b_sft as shared
    frozen = ROOT / "gadi_sft_8b_starter"
    starter = repository / "gadi_sft_8b_starter"
    shared.ROOT = starter
    for module in tuple(sys.modules.values()):
        if getattr(module, "ROOT", None) == frozen:
            module.ROOT = starter
            default = getattr(module, "DEFAULT_CONFIG", None)
            if isinstance(default, Path) and default.is_relative_to(frozen):
                module.DEFAULT_CONFIG = starter / default.relative_to(frozen)
    import max5_dpo_data
    return max5_dpo_data


def validate_config(config):
    if (config["focal_model"] != "llama31" or config["folds"] != 5
            or config["sampled_responses"] != 4 or config["max_candidates"] != 5
            or config["training_pools"] != ["max5_greedy", "max5_sampling4"]
            or config["stage"] != "audit_only" or config["reranker_C"] <= 0
            or config["max_pairs_per_question"] < 1
            or config["minimum_reciprocal_rank_gain"] < 0):
        raise ValueError("Expected the declared Llama/five-fold/four-sample audit-only pilot")


def load_inputs(repository, rerankers, config):
    validate_config(config)
    state = read(rerankers / "manifest.json")
    if (state["status"] != "complete" or state["question_counts"] != {"train": 1296, "validation": 144}
            or state["sklearn_version"] != sklearn.__version__ or "max5" not in state["selectors"]):
        raise ValueError("Need completed fitting-only max-five reranker outputs and matching scikit-learn")
    hashes = {"manifest.json": digest(rerankers / "manifest.json")}
    for name, value in state["artifacts_sha256"].items():
        path = (rerankers / name).resolve()
        if not path.is_relative_to(rerankers.resolve()) or digest(path) != value:
            raise ValueError(f"Reranker artifact changed: {name}")
        hashes[name] = value
    prepared_path = Path(state["submission"]) / "prepared.json"
    if digest(prepared_path) != state["prepared_sha256"]:
        raise ValueError("Reranker preparation metadata changed")
    prepared = read(prepared_path)
    for name in ("scripts/train_candidate_reranker_pilot.py", "scripts/train_sft_ensemble_reranker.py"):
        if digest(ROOT / name) != prepared["code_hashes"][name]:
            raise ValueError("Reranker feature/pooling implementation changed since bank labelling")
    data = configure_data_root(repository)
    generation_config = read(data.DEFAULT_CONFIG)
    if generation_config["sampled_responses"] != 4:
        raise ValueError("Expected four saved stochastic responses per model/question")
    paths = [Path(prepared["max5_banks"][m]) for m in MODELS]
    banks, raw = data.pack_banks(paths, generation_config)
    if any(banks[m]["manifest"]["responses_sha256"] != state["max5_bank_hashes"][m] for m in MODELS):
        raise ValueError("Max-five bank differs from the one used to train the reranker")
    examples = {split: {r["question_id"]: r for r in records(rerankers / f"{split}_examples.jsonl")}
                for split in ("train", "validation")}
    if ({s: len(e) for s, e in examples.items()} != {"train": 1296, "validation": 144}
            or set(examples["train"]) & set(examples["validation"])):
        raise ValueError("Fitting/validation membership changed")
    current = rr.contexts_from_rows([{**r, "arm": "max5"} for r in raw])
    if current != examples:
        raise ValueError("Saved fitting/validation evidence differs from generation banks")
    # pack_banks/panels also enforce the pinned fitting/validation split and
    # disjoint outer dev/test membership. Only fitting rows are used below.
    fitting = [{**r, "arm": "max5"} for r in raw if r["split"] == "train"]
    rebuilt, _ = rr.build_pools(fitting, {"train": examples["train"]})
    pools, labels = {}, {}
    for name in config["training_pools"]:
        filename = f"train_{name}_candidates.jsonl"
        if filename not in hashes:
            raise ValueError("Missing hashed fitting candidate labels")
        rows = records(rerankers / filename)
        if [{k: v for k, v in r.items() if k != "label"} for r in rows] != rebuilt["train"][name]:
            raise ValueError("Saved fitting candidate features differ from reconstructed banks")
        for row in rows:
            identity = (row["question_id"], key(row["answer"]))
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError("Invalid cached official label")
            if identity in labels and labels[identity] != row["label"]:
                raise ValueError("Normalisation changes official acceptance")
            labels[identity] = row["label"]
        pools[name] = rows
    bank_hashes = {m: {"path": str(p), "manifest_sha256": digest(p / "manifest.json"),
                     "responses_sha256": digest(p / "responses.jsonl")} for m, p in zip(MODELS, paths)}
    return examples["train"], pools, fitting, labels, hashes, bank_hashes, data


def replacement_slot(qid, seed, draws):
    value = hashlib.sha256(f"{seed}:{qid}:replacement-slot".encode()).digest()
    return 1 + int.from_bytes(value[:8], "big") % draws


def response_summary(row, labels, data, ceiling=512):
    try:
        aa = data.answers(row["raw_response"])
    except (ValueError, TypeError):
        return {"eligible": False, "reason": "invalid_strict_max5_schema"}
    if row["output_tokens"] >= ceiling:
        return {"eligible": False, "reason": "output_token_ceiling"}
    accepted = [bool(labels[(row["question_id"], key(c["answer"]))]) for c in aa]
    return {"eligible": True, "candidate_count": len(aa), "accepted_form_count": sum(accepted),
            "initial_answer": aa[0]["answer"], "initial_accepted": accepted[0],
            "canonical_response": json.dumps(aa, ensure_ascii=False, sort_keys=True)}


def alternatives(rows, examples, labels, config, data):
    by_question = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or row["question_id"] not in examples:
            raise ValueError("Only fitting responses may construct preference alternatives")
        by_question[row["question_id"]].append(row)
    output = {}
    for qid in sorted(examples):
        group = by_question[qid]
        expected = {(m, d) for m in MODELS for d in range(5)}
        if len(group) != 15 or {(r["model_key"], r["draw"]) for r in group} != expected:
            raise ValueError("Every fitting question needs one greedy plus four samples from each model")
        focal = sorted((r for r in group if r["model_key"] == config["focal_model"]), key=lambda r: r["draw"])
        if any(r["prompt"] != focal[0]["prompt"] for r in group):
            raise ValueError("Different native prompts across max-five banks")
        slot = replacement_slot(qid, config["seed"], 4)
        fixed = [r for r in group if r["draw"] > 0 and not
                 (r["model_key"] == config["focal_model"] and r["draw"] == slot)]
        if len(fixed) != 11:
            raise ValueError("Expected exactly 11 fixed responses")
        options = []
        for response in focal:
            # This projection changes draw position only inside a counterfactual
            # pool. Original raw records/response IDs are never rewritten.
            projected = [*fixed, {**response, "draw": slot}]
            pools, _ = rr.build_pools(projected, {"train": {qid: examples[qid]}})
            pool = pools["train"]["max5_sampling4"]
            for candidate in pool:
                candidate["label"] = labels[(qid, key(candidate["answer"]))]
            options.append({"response": response, "summary": response_summary(response, labels, data),
                            "pool": pool, "replacement_slot": slot,
                            "fixed_response_ids": [r["response_id"] for r in sorted(fixed, key=lambda r: (r["model_key"], r["draw"]))],
                            "baseline": response["draw"] == slot})
        output[qid] = options
    return output


def score_options(components, options, examples):
    flattened = [r for oo in options.values() for option in oo for r in option["pool"]]
    values = []
    # Batching avoids multiplying sparse-feature memory by all alternatives.
    for offset in range(0, len(flattened), 1024):
        batch = flattened[offset:offset + 1024]
        values.extend(score_components(components, rr.feature_rows(batch), examples).tolist())
    if not np.isfinite(values).all():
        raise ValueError("Non-finite out-of-fold predictions")
    offset = 0
    scored = []
    for qid, oo in options.items():
        for option in oo:
            pool = option["pool"]
            ranked = sorted(zip(pool, values[offset:offset + len(pool)]),
                            key=lambda r: (-r[1], rr.tie_key(r[0])))
            offset += len(pool)
            rank = next((i for i, (r, _) in enumerate(ranked, 1) if r["label"]), None)
            row = option["response"]
            scored.append({"question_id": qid, "response_id": row["response_id"], "draw": row["draw"],
                "replacement_slot": option["replacement_slot"], "fixed_response_ids": option["fixed_response_ids"],
                "baseline": option["baseline"], "response_summary": option["summary"],
                "mrr_at5": 1 / rank if rank and rank <= 5 else 0.0,
                "first_accepted_rank": rank, "coverage_at5": int(rank is not None and rank <= 5),
                "coverage_full_pool": int(rank is not None), "pool_candidates": len(pool),
                "top_five": [{"answer": r["answer"], "score": float(score), "official_accepted": bool(r["label"])}
                             for r, score in ranked[:5]]})
    return scored


def preference_candidates(scored, responses, config):
    groups = defaultdict(list)
    for row in scored:
        groups[row["question_id"]].append(row)
    primary, other, diagnostics = [], [], Counter()
    for qid, group in sorted(groups.items()):
        choices = {"primary": [], "other": []}
        seen = set()
        for a, b in itertools.combinations(group, 2):
            if b["mrr_at5"] > a["mrr_at5"]:
                a, b = b, a
            gain = a["mrr_at5"] - b["mrr_at5"]
            if gain <= config["minimum_reciprocal_rank_gain"] + 1e-12:
                diagnostics["tied_or_below_minimum_gain"] += 1
                continue
            aa, bb = a["response_summary"], b["response_summary"]
            if not aa["eligible"] or not bb["eligible"]:
                diagnostics["invalid_or_ceiling_response_pair"] += 1
                continue
            identity = (aa["canonical_response"], bb["canonical_response"])
            if identity in seen:
                continue
            seen.add(identity)
            if aa["accepted_form_count"] < bb["accepted_form_count"] or a["coverage_full_pool"] < b["coverage_full_pool"]:
                diagnostics["accepted_form_or_pool_coverage_loss"] += 1
                continue
            same_initial = key(aa["initial_answer"]) == key(bb["initial_answer"])
            category = ("same_accepted_initial" if same_initial and aa["initial_accepted"] and bb["initial_accepted"] else
                        "different_accepted_initial" if aa["initial_accepted"] and bb["initial_accepted"] else
                        "initial_acceptance_changed" if aa["initial_accepted"] != bb["initial_accepted"] else
                        "neither_initial_accepted")
            chosen, rejected = responses[a["response_id"]], responses[b["response_id"]]
            if chosen["prompt"] != rejected["prompt"] or a["fixed_response_ids"] != b["fixed_response_ids"]:
                raise ValueError("Pair prompt or fixed context differs")
            pair = {"question_id": qid, "split": "train", "model_key": config["focal_model"],
                "prompt": chosen["prompt"], "chosen": chosen["raw_response"], "rejected": rejected["raw_response"],
                "chosen_id": chosen["response_id"], "rejected_id": rejected["response_id"],
                "fold": a["fold"], "category": category, "reciprocal_rank_gain": gain,
                "replacement_slot": a["replacement_slot"], "fixed_response_ids": a["fixed_response_ids"],
                "chosen_outcome": a, "rejected_outcome": b, "semantic_review_status": "pending",
                "training_ready": False, "synthetic_chosen": False}
            choices["primary" if category == "same_accepted_initial" else "other"].append(pair)
            diagnostics[f"eligible_before_cap/{category}"] += 1
        for kind, destination in (("primary", primary), ("other", other)):
            ranked = sorted(choices[kind], key=lambda r: (-r["reciprocal_rank_gain"], r["chosen_id"], r["rejected_id"]))
            destination.extend(ranked[:config["max_pairs_per_question"]])
    return primary, other, dict(diagnostics)


def prepare(repository, rerankers, config_path, submission):
    config = read(config_path)
    _, _, _, _, hashes, banks, _ = load_inputs(repository, rerankers, config)
    submission.mkdir(parents=True, exist_ok=False)
    code = rr.snapshot_code(repository, submission / "code")
    write(submission / "prepared.json", {"repository": str(repository), "rerankers": str(rerankers),
        "reranker_hashes": hashes, "banks": banks, "configuration": config,
        "config_path": str(config_path), "config_sha256": digest(config_path), "code_hashes": code,
        "scope": "Fitting questions only; validation, dev160 and test excluded from OOF feedback"})
    print("Prepared one CPU audit: five out-of-fold fits; no new generation, paid judging or DPO training.", flush=True)


def run(submission, output):
    prepared = read(submission / "prepared.json")
    for name, value in prepared["code_hashes"].items():
        if digest(submission / "code" / name) != value:
            raise ValueError("Frozen pilot code changed")
    if digest(Path(prepared["config_path"])) != prepared["config_sha256"]:
        raise ValueError("Pilot configuration changed")
    config = prepared["configuration"]
    examples, pools, raw, labels, hashes, banks, data = load_inputs(
        Path(prepared["repository"]), Path(prepared["rerankers"]), config)
    if hashes != prepared["reranker_hashes"] or banks != prepared["banks"]:
        raise ValueError("Reranker or response banks changed since preparation")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "stage": "preference_audit_only", "training_ready": False,
             "configuration": config, "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
             "question_count": len(examples), "dpo_training": False, "generation": False, "paid_judging": False}
    write(output / "manifest.json", state)
    try:
        options = alternatives(raw, examples, labels, config, data)
        qids, all_scores, folds = sorted(examples), [], []
        for fold, (train_index, held_index) in enumerate(KFold(
                n_splits=config["folds"], shuffle=True, random_state=config["seed"]).split(qids), 1):
            fitting, held = {qids[i] for i in train_index}, {qids[i] for i in held_index}
            if fitting & held:
                raise ValueError("Out-of-fold question overlap")
            train_rows, strata = [], []
            for name, rows in pools.items():
                selected = [r for r in rows if r["question_id"] in fitting]
                train_rows.extend(selected); strata.extend([name] * len(selected))
            target = np.asarray([r["label"] for r in train_rows])
            if set(target.tolist()) != {0, 1}:
                raise ValueError("Each fold needs positive and negative fitting candidates")
            fitting_examples = {q: examples[q] for q in fitting}
            print(f"Fold {fold}/5: fitting on {len(fitting)} questions; judging {len(held)} unseen questions.", flush=True)
            components = fit_components(rr.feature_rows(train_rows), fitting_examples, target,
                sample_weight=rr.question_weights(train_rows, strata), C=config["reranker_C"])
            model_path = output / f"fold-{fold}.joblib"
            joblib.dump({"components": components, "fitting_question_ids": sorted(fitting),
                         "held_out_question_ids": sorted(held), "C": config["reranker_C"]}, model_path)
            scored = score_options(components, {q: options[q] for q in sorted(held)}, {q: examples[q] for q in held})
            all_scores.extend({**r, "fold": fold} for r in scored)
            folds.append({"fold": fold, "fitting_question_ids": sorted(fitting), "held_out_question_ids": sorted(held),
                          "model_sha256": digest(model_path), "scored_alternatives": len(scored)})
        if len(all_scores) != 5 * len(examples):
            raise ValueError("Every fitting question must have all five focal alternatives scored out of fold")
        by_id = {r["response_id"]: r for r in raw}
        primary, other, diagnostics = preference_candidates(all_scores, by_id, config)
        review_ids = {p[k] for p in primary for k in ("chosen_id", "rejected_id")}
        reviews = []
        for identity in sorted(review_ids):
            row = data.review_template(by_id[identity])
            for c in row["candidates"]:
                c["official_accepted"] = bool(labels[(row["question_id"], key(c["answer"]))])
            reviews.append(row)
        groups = defaultdict(list)
        for row in all_scores:
            groups[row["question_id"]].append(row)
        baseline = [next(r for r in values if r["baseline"]) for values in groups.values()]
        best = [max(r["mrr_at5"] for r in values) for values in groups.values()]
        gains = [max(r["mrr_at5"] for r in values) - next(r["mrr_at5"] for r in values if r["baseline"])
                 for values in groups.values()]
        summary = {"fitting_questions": len(examples), "folds": len(folds),
            "scored_alternatives": len(all_scores), "primary_preference_candidates": len(primary),
            "primary_questions": len({p["question_id"] for p in primary}),
            "other_diagnostic_candidates": len(other), "responses_requiring_review": len(reviews),
            "questions_with_oracle_rank_improvement": sum(g > 1e-12 for g in gains),
            "baseline_oof_mrr_at5": sum(r["mrr_at5"] for r in baseline) / len(baseline),
            "best_saved_alternative_oracle_mrr_at5": sum(best) / len(best),
            "diagnostics": diagnostics,
            "primary_length_counts": dict(Counter("chosen_shorter" if by_id[p["chosen_id"]]["output_tokens"] < by_id[p["rejected_id"]]["output_tokens"]
                else "chosen_longer" if by_id[p["chosen_id"]]["output_tokens"] > by_id[p["rejected_id"]]["output_tokens"]
                else "same_token_count" for p in primary)),
            "semantic_labels": "Unresolved; official nonacceptance never implies semantic incorrectness",
            "interpretation": "OOF fitting diagnostics and oracle over saved alternatives, not a deployable system or held-out test improvement",
            "training_ready": False, "next_step": "Review primary chosen/rejected responses before constructing any DPO training files"}
        for filename, values in (("alternative_outcomes.jsonl", all_scores), ("preference_candidates.jsonl", primary),
                                 ("other_diagnostic_candidates.jsonl", other), ("review_responses.jsonl", reviews)):
            write_jsonl(output / filename, values)
        write(output / "folds.json", folds)
        write(output / "summary.json", summary)
        # Detect concurrent edits before marking the audit complete.
        for model, bank in banks.items():
            path = Path(bank["path"])
            if digest(path / "responses.jsonl") != bank["responses_sha256"] or digest(path / "manifest.json") != bank["manifest_sha256"]:
                raise ValueError(f"Source bank changed during audit: {model}")
        state.update(status="review_required" if primary else "insufficient_primary_pairs", summary=summary,
            output_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Audit saved: {output}\nNo DPO training has been submitted.", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "audit"), required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--rerankers", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.rerankers is None:
            parser.error("--rerankers is required for preparation")
        prepare(args.repository.resolve(), args.rerankers.resolve(), args.config.resolve(), args.submission.resolve())
    else:
        if args.output is None:
            parser.error("--output is required for audit")
        run(args.submission.resolve(), args.output.resolve())
