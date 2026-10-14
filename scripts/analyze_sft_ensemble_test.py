#!/usr/bin/env python3
"""Score original/max-five ensembles with three fixed, untrained selectors."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))

from prepare_matched_8b_sft import read, records, write, digest
from run_single_answer_sampling import (
    parse_single_answer, unique_candidates, expansion_candidates, sample_seed,
)
from run_extractive_expansion_8b import parse_equivalent_response, write_jsonl
from scripts.compare_expansion_pool_rerankers import MODELS, round_robin, tie_key, evaluate_rankings
from scripts.analyze_matched_qwen_expansion import paired_contrast
from scripts.train_candidate_reranker_pilot import key, clean
from src.notebook_workflows.coverage_comparison import official_candidate_matches

ARMS = ("original", "max5")
METHODS = ("round_robin", "model_vote", "rank_vote")


def load_archive(directory, mode):
    """Verify saved raw responses, reconstruct candidates, retain all failures."""
    directory = directory.resolve()
    state = read(directory / "manifest.json")
    if (state.get("status") != "complete" or state.get("smoke_test")
            or state.get("expected_questions") != 95 or set(state["runs"]) != set(ARMS)):
        raise ValueError("Need complete original/max-five test archives, not smoke runs")
    submission = Path(state["submission_dir"])
    if digest(submission / "manifest.json") != state["submission_manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    prepared = read(submission / "manifest.json")
    for name, value in prepared["file_hashes"].items():
        if digest(submission / name) != value:
            raise ValueError("Prepared test data changed")
    examples = records(submission / "test_inputs.jsonl")
    gold = {r["question_id"]: r for r in records(submission / "test_gold.jsonl")}
    expected = {r["question_id"]: r for r in examples}
    if len(examples) != 95 or len(expected) != 95 or set(expected) != set(gold):
        raise ValueError("Wrong or duplicate test membership")
    if any(set(e) != {"question_id", "question", "snippets"} for e in examples):
        raise ValueError("Gold fields in inference inputs")
    draws = 1 if mode == "greedy" else 10
    generations, costs = {}, {}
    for arm in ARMS:
        run = (directory / state["runs"][arm]).resolve()
        if not run.is_relative_to(directory):
            raise ValueError("Archive path escaped evaluation")
        for name, value in state["arm_hashes"][arm].items():
            if digest(run / name) != value:
                raise ValueError("Archived inference changed")
        cfg = read(run / "config.json")
        if any(cfg.get(k) != v for k, v in state["settings"][arm].items()):
            raise ValueError("Archived settings differ")
        response_mode = ("equivalent_" if arm == "max5" else "single_answer_") + (
            "greedy" if mode == "greedy" else "sampling")
        if (cfg["num_generations"] != draws or cfg["response_mode"] != response_mode
                or cfg["max_candidates"] != (5 if arm == "max5" else 1)
                or cfg["max_new_tokens"] != 512 or cfg["seed"] != 3407
                or cfg["temperature"] != (0 if mode == "greedy" else 0.8)
                or (draws == 10 and cfg["top_p"] != 0.95)
                or not cfg["require_all_snippets"] or cfg["mark_snippets"]):
            raise ValueError("Inference budget, prompt format or decoding differs")
        prompt = Path(state["settings"][arm]["prompt"])
        source = prepared["sources"][state["model_key"]][arm]["source"]
        if (digest(prompt) != cfg["prompt_sha256"]
                or prompt.read_text(encoding="utf-8").strip() != source["system_prompt"].strip()):
            raise ValueError("Native SFT prompt changed")
        saved_examples = records(run / "examples.jsonl")
        if len(saved_examples) != 95 or {e["question_id"] for e in saved_examples} != set(expected):
            raise ValueError("Missing/duplicate input questions")
        if any(e != expected[e["question_id"]] for e in saved_examples):
            raise ValueError("Input text/evidence changed")
        rows = records(run / "generations.jsonl")
        if (read(run / "status.json")["status"] != "complete" or len(rows) != 95
                or {r["question_id"] for r in rows} != set(expected)):
            raise ValueError("Incomplete/duplicate generations")
        candidates = []
        for row in rows:
            qid = row["question_id"]
            if (row["snippets_truncated"] or row["included_snippets"] != len(expected[qid]["snippets"])
                    or row["total_snippets"] != row["included_snippets"]
                    or len(row["samples"]) != draws
                    or [s["draw"] for s in row["samples"]] != list(range(1, draws + 1))):
                raise ValueError("Evidence or draw membership differs")
            for sample in row["samples"]:
                if sample["seed"] != sample_seed(qid, sample["draw"], 3407):
                    raise ValueError("Sample seed differs")
                if arm == "original":
                    try:
                        answer = parse_single_answer(sample["raw_response"])
                    except ValueError:
                        answer = None
                    if sample["answer"] != answer:
                        raise ValueError("Tagged answer changed")
                else:
                    try:
                        aa = parse_equivalent_response(sample["raw_response"], 5)[0]
                    except ValueError:
                        aa = []
                    if sample["answers"] != aa:
                        raise ValueError("Expansion parsing changed")
            cc = unique_candidates(row["samples"]) if arm == "original" else expansion_candidates(row["samples"])
            candidates.extend({"question_id": qid, **c} for c in cc)
            if (row["request_count"] != draws
                    or row["input_tokens"] != row["prompt_tokens"] * draws
                    or row["output_tokens"] != sum(s["output_tokens"] for s in row["samples"])
                    or abs(row["generation_seconds"] - sum(s["generation_seconds"] for s in row["samples"])) > 1e-6
                    or any(s["generation_seconds"] < 0 or s["output_tokens"] < 0 for s in row["samples"])):
                raise ValueError("Generation cost counters differ")
        if candidates != records(run / "candidates.jsonl"):
            raise ValueError("Candidate membership/order changed")
        generations[arm] = rows
        costs[arm] = {k: sum(r[k] for r in rows) for k in
                      ("request_count", "input_tokens", "output_tokens", "generation_seconds")}
        costs[arm].update(parse_success_count=sum(not r.get("parse_error") for r in rows),
            failed_draws=sum(bool(s.get("parse_error")) for r in rows for s in r["samples"]),
            output_token_ceiling_count=sum(s["output_tokens"] >= 512 for r in rows for s in r["samples"]),
            candidate_cap_exceeded_count=sum(bool(s.get("candidate_limit_applied")) for r in rows for s in r["samples"]))
    return state, prepared, examples, gold, generations, costs


def merge_pool(by_model, arm):
    """Unique model votes; ranks are first unique occurrence in source draw order."""
    merged = {}
    for model in MODELS:
        for row in by_model[model][arm]:
            seen_source = set()
            for sample in row["samples"]:
                aa = ([{"answer": sample["answer"], "relation_type": "original"}]
                      if arm == "original" and sample.get("answer") else
                      sample.get("answers", []) if arm == "max5" else [])
                seen_draw = set()
                for candidate in aa:
                    identity = (row["question_id"], key(candidate["answer"]))
                    if identity in seen_draw:
                        continue
                    seen_draw.add(identity)
                    target = merged.setdefault(identity, {"question_id": row["question_id"],
                        "answer": clean(candidate["answer"]), "sources": [], "source_ranks": {},
                        "sample_frequencies": {}, "relation_types": []})
                    if identity not in seen_source:
                        seen_source.add(identity)
                        target["sources"].append(model)
                        target["source_ranks"][model] = len(seen_source)
                    target["sample_frequencies"][model] = target["sample_frequencies"].get(model, 0) + 1 / len(row["samples"])
                    relation = candidate.get("relation_type", "unknown")
                    if relation not in target["relation_types"]:
                        target["relation_types"].append(relation)
    return list(merged.values())


def order_pool(rows, method):
    if method == "round_robin":
        return round_robin(rows)
    if method == "model_vote":
        return sorted(rows, key=lambda r: (-len(r["sources"]), tie_key(r)))
    if method == "rank_vote":
        return sorted(rows, key=lambda r: (-sum(1 / rank for rank in r["source_ranks"].values()), tie_key(r)))
    raise ValueError("Unknown fixed selector")


def summarize_pools(pools, examples, gold):
    conditions, per_question, rankings = {}, {}, []
    for name, rows in pools.items():
        groups = {q: [] for q in examples}
        for row in rows:
            groups[row["question_id"]].append(row)
        for method in METHODS:
            ordered = {q: order_pool(rr, method) for q, rr in groups.items()}
            condition = f"{name}_{method}"
            metrics, scored = evaluate_rankings(ordered)
            per_question[condition] = scored
            batches = {}
            for batch in sorted({r["batch"] for r in gold.values()}):
                rr = [r for r in scored if gold[r["question_id"]]["batch"] == batch]
                batches[batch] = {"question_count": len(rr), **{k: sum(r[k] for r in rr) / len(rr)
                                   for k in metrics}}
            conditions[condition] = {"metrics": metrics, "per_batch": batches,
                "mean_unique_candidates": len(rows) / len(examples)}
            rankings.extend({"condition": condition, "rank": rank, **r}
                            for rr in ordered.values() for rank, r in enumerate(rr, 1))
    return conditions, per_question, rankings


def analyze(submission, output, jar, resamples=10000):
    prepared = read(submission / "manifest.json")
    jobs = read(submission / "sampling_jobs.json")
    if set(jobs) != set(MODELS) or any(not re.fullmatch(r"\d+\.gadi-pbs", job) for job in jobs.values()):
        raise ValueError("Need exactly three recorded sampling jobs")
    for relative, value in prepared["analysis_code_hashes"].items():
        if digest(submission / "code" / relative) != value:
            raise ValueError("Frozen scoring code changed")
    ensembles, source_costs, provenance = {}, {}, {}
    examples = gold = None
    for mode in ("greedy", "sampling10"):
        by_model = {}
        for model in MODELS:
            directory = Path(prepared["greedy_archives"][model]) if mode == "greedy" else (
                Path(prepared["repository"]) / "gadi_sft_8b_starter/outputs/sft_ensemble_test" /
                f"{model}-sampling10-{jobs[model]}")
            if mode == "greedy" and digest(directory / "manifest.json") != prepared["greedy_manifest_hashes"][model]:
                raise ValueError("Greedy archive changed since preparation")
            state, source, ee, gg, generations, costs = load_archive(directory, mode)
            if state["model_key"] != model:
                raise ValueError("Archive model differs")
            if examples is None:
                examples, gold = {e["question_id"]: e for e in ee}, gg
            if examples != {e["question_id"]: e for e in ee} or gold != gg:
                raise ValueError("Questions, snippets or gold differ across archives")
            for arm in ARMS:
                current = prepared["sources"][model][arm]["source"]
                saved = source["sources"][model][arm]["source"]
                for field in ("adapter_files_sha256", "base_snapshot", "base_revision", "system_prompt",
                              "model_loader", "chat_template_kwargs", "execution_mode"):
                    if current.get(field) != saved.get(field):
                        raise ValueError(f"Different frozen source: {model}/{arm}/{field}")
                cfg = read(directory / state["runs"][arm] / "config.json")
                prompt = Path(state["settings"][arm]["prompt"])
                if prompt.read_text(encoding="utf-8").strip() != current["system_prompt"].strip() or digest(prompt) != cfg["prompt_sha256"]:
                    raise ValueError("Native SFT prompt changed")
            by_model[model] = generations
            source_costs[f"{model}_{mode}"] = costs
            provenance[f"{model}_{mode}"] = {"directory": str(directory),
                "manifest_sha256": digest(directory / "manifest.json"), "arm_hashes": state["arm_hashes"]}
        for arm in ARMS:
            ensembles[f"{arm}_{mode}"] = merge_pool(by_model, arm)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "question_count": 95, "training": False,
             "submission_manifest_sha256": digest(submission / "manifest.json"),
             "sampling_jobs_sha256": digest(submission / "sampling_jobs.json"), "sources": provenance}
    write(output / "manifest.json", state)
    try:
        distinct = {(r["question_id"], r["answer"]): {"question_id": r["question_id"], "answer": r["answer"]}
                    for rows in ensembles.values() for r in rows}
        report = output / "official_scores"; report.mkdir()
        matches = official_candidate_matches(
            [{**e, "gold_aliases": gold[q]["gold_aliases"]} for q, e in examples.items()],
            list(distinct.values()), report, jar_path=jar) if distinct else {}
        for rows in ensembles.values():
            for row in rows:
                row["label"] = int(matches[(row["question_id"], row["answer"])])
        conditions, scored, rankings = summarize_pools(ensembles, examples, gold)
        contrasts = {}
        comparisons = {**{f"max5_minus_original_{mode}_{method}":
            (f"original_{mode}_{method}", f"max5_{mode}_{method}")
            for mode in ("greedy", "sampling10") for method in METHODS},
            **{f"sampling10_minus_greedy_{arm}_{method}":
            (f"{arm}_greedy_{method}", f"{arm}_sampling10_{method}") for arm in ARMS for method in METHODS}}
        for name, (before, after) in comparisons.items():
            delta = paired_contrast(scored[before], scored[after], resamples=resamples,
                metrics=("mrr_at5", "coverage_at5", "coverage_at10", "coverage_full_pool"))
            contrasts[name] = {"before": before, "after": after, "metrics": {
                k: {"difference": v["sft_minus_base"], "paired_bootstrap_95ci": v["paired_bootstrap_95ci"]}
                for k, v in delta.items()}}
        result = {"question_count": 95, "conditions": conditions, "paired_contrasts": contrasts,
            "source_costs": source_costs,
            "ensemble_costs": {f"{arm}_{mode}": {k: sum(source_costs[f"{m}_{mode}"][arm][k] for m in MODELS)
                for k in ("request_count", "input_tokens", "output_tokens", "generation_seconds")}
                for arm in ARMS for mode in ("greedy", "sampling10")},
            "settings": {"temperature": 0.8, "top_p": 0.95, "seed": 3407, "final_answers": 5,
                         "source_order": MODELS, "sampling_draws": 10},
            "limitations": ["Historically inspected project test batches, not certified unseen.",
                "Equal requests do not imply equal output-token, runtime or candidate budgets.",
                "Coverage@10 is a ten-candidate limit, not ten draws; full-pool coverage uses all unique candidates.",
                "Paired intervals describe one sampling seed and have no multiple-comparison correction."]}
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scored)
        write_jsonl(output / "rankings.jsonl", rankings)
        for name, rows in ensembles.items():
            write_jsonl(output / f"{name}_candidates.jsonl", rows)
        for name, values in conditions.items():
            m = values["metrics"]
            print(f"{name}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, "
                  f"coverage@10={m['coverage_at10']:.4f}, full pool={m['coverage_full_pool']:.4f}", flush=True)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    analyze(args.submission.resolve(), args.output.resolve(), args.jar.resolve())
