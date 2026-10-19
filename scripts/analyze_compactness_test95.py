#!/usr/bin/env python3
"""Evaluate fixed SFT/strict/all-epoch1 ensembles on the same test95 pools."""
import argparse
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))

from prepare_matched_8b_sft import read, records, write, digest
from run_compactness_test95 import (
    CONFIGURATIONS, MODELS, MODES, OUTPUT, validate_prepared, RUN_FILES,
)
from scripts.analyze_generator_feedback_evaluation import (
    load_generation, compactness, contrast, coverage_transitions, METRICS,
)
from scripts.analyze_sft_ensemble_test import merge_pool, summarize_pools
from scripts.evaluate_sft_ensemble_rerankers import rank_frozen, load_frozen
from scripts.compare_expansion_pool_rerankers import evaluate_rankings
from scripts.train_candidate_reranker_pilot import key
from scripts.ablate_reranker_relation_labels import verify_files
from run_extractive_expansion_8b import write_jsonl
from src.notebook_workflows.coverage_comparison import official_candidate_matches


def make_pools(baseline, dpo):
    pools = {}
    for mode in MODES:
        for name, settings in CONFIGURATIONS.items():
            responses = {m: {"max5": (dpo if settings[m] else baseline)[f"{m}_{mode}"]} for m in MODELS}
            pools[f"{name}_{mode}"] = merge_pool(responses, "max5")
        for model in MODELS:
            for policy, bank in (("sft", baseline), ("dpo", dpo)):
                solo = {m: {"max5": bank[f"{m}_{mode}"] if m == model else []} for m in MODELS}
                pools[f"solo_{model}_{policy}_{mode}"] = merge_pool(solo, "max5")
    return pools


def add_batches(conditions, scores, gold):
    for name, condition in conditions.items():
        per_batch = {}
        for batch in sorted({r["batch"] for r in gold.values()}):
            rows = [r for r in scores[name] if gold[r["question_id"]]["batch"] == batch]
            per_batch[batch] = {"question_count": len(rows), **{
                k: sum(r[k] for r in rows) / len(rows) for k in condition["metrics"]}}
        condition["per_batch"] = per_batch


def score_pools(pools, examples, gold, payload, output, jar, matcher=official_candidate_matches):
    distinct = {(r["question_id"], r["answer"]): r for rr in pools.values() for r in rr}
    official = output / "official"; official.mkdir(parents=True, exist_ok=True)
    matches = matcher(list(gold.values()), list(distinct.values()), official, jar_path=jar) if distinct else {}
    normalized = {}
    for (qid, answer), label in matches.items():
        identity = (qid, key(answer))
        if identity in normalized and normalized[identity] != label:
            raise ValueError("Pool deduplication changes official acceptance")
        normalized[identity] = label
    for rr in pools.values():
        for r in rr:
            r["label"] = int(matches[(r["question_id"], r["answer"])])
    ensemble = {n: rr for n, rr in pools.items() if not n.startswith("solo_")}
    conditions, scores, rankings = rank_frozen(payload, ensemble, examples)
    cc, ss, rr = summarize_pools(ensemble, examples, gold, methods=("sample_frequency",))
    conditions.update(cc); scores.update(ss); rankings.extend(rr)
    for name, rows in pools.items():
        if not name.startswith("solo_"):
            continue
        groups = {q: [] for q in examples}
        for row in rows:
            groups[row["question_id"]].append(row)
        metrics, perq = evaluate_rankings(groups)
        conditions[name + "_generated_order"] = {"metrics": metrics, "mean_unique_candidates": len(rows) / len(examples)}
        scores[name + "_generated_order"] = perq
        rankings.extend({"condition": name + "_generated_order", "rank": i, **r}
                        for rr in groups.values() for i, r in enumerate(rr, 1))
    add_batches(conditions, scores, gold)
    return conditions, scores, rankings


def check_baseline(pools, comparison, expected, examples, gold):
    """New official labels must exactly reproduce the historical SFT pools."""
    for mode in MODES:
        old = records(comparison / f"max5_{mode}_candidates.jsonl")
        if pools[f"baseline_{mode}"] != old:
            raise ValueError("Baseline candidates, frequencies or official labels differ from historical test95")
        conditions, _, _ = summarize_pools({"baseline": old}, examples, gold, methods=("round_robin",))
        if conditions["baseline_round_robin"]["metrics"] != expected[f"max5_{mode}"]:
            raise ValueError("Historical baseline metrics differ")


def analyze(submission, output, resamples=10000, matcher=official_candidate_matches):
    if output.exists() or resamples < 2:
        raise ValueError("Need a new output directory and at least two bootstrap resamples")
    prepared = validate_prepared(submission)
    jobs_file = submission / "generation_jobs.json"
    jobs_hash = digest(jobs_file); jobs = read(jobs_file)
    import re
    if set(jobs) != set(MODELS) or any(not re.fullmatch(r"\d+\.gadi-pbs", j) for j in jobs.values()):
        raise ValueError("Need all three full epoch-one generation jobs")
    panel = records(submission / "test_inputs.jsonl")
    examples = {e["question_id"]: e for e in panel}
    gold = {e["question_id"]: e for e in records(submission / "test_gold.jsonl")}
    if len(panel) != 95 or len(examples) != 95 or set(examples) != set(gold):
        raise ValueError("Wrong test question membership")
    baseline, dpo, generation_hashes = {}, {}, {}
    for model in MODELS:
        generation = Path(prepared["repository"]) / "gadi_sft_8b_starter" / OUTPUT / f"{model}-generation-{jobs[model]}"
        state = read(generation / "manifest.json")
        if (state.get("status") != "complete" or state.get("smoke_test") is not False
                or state.get("expected_questions") != 95 or state.get("epoch") != 1
                or state.get("model_key") != model or set(state.get("runs", {})) != set(MODES)
                or Path(state["submission"]).resolve() != submission.resolve()
                or state["prepared_sha256"] != digest(submission / "prepared.json")
                or state["source"] != prepared["dpo_sources"][model]):
            raise ValueError("Need a complete epoch-one test95 generation for " + model)
        generation_hashes[str(generation / "manifest.json")] = digest(generation / "manifest.json")
        for mode in MODES:
            baseline[f"{model}_{mode}"] = load_generation(Path(prepared["baseline_runs"][f"{model}_{mode}"]),
                prepared["sources"][model], panel, mode)
            run = (generation / state["runs"][mode]).resolve()
            if not run.is_relative_to(generation.resolve()):
                raise ValueError("DPO generation escaped its output directory")
            verify_files(run, state["arm_hashes"][mode])
            if set(state["arm_hashes"][mode]) != set(RUN_FILES):
                raise ValueError("Missing generation artifact hashes")
            cfg = read(run / "config.json")
            if (cfg.get("sample_seed_offset", 0) != 0 or any(cfg[k] != prepared["input_plans"][model][k] for k in (
                    "max_seq_length", "expected_prompt_tokens", "expected_prompt_ids_sha256"))):
                raise ValueError("Test seed indices or input allocation differ from SFT")
            dpo[f"{model}_{mode}"] = load_generation(run, prepared["dpo_sources"][model], panel, mode)
            generation_hashes.update({str(run / n): v for n, v in state["arm_hashes"][mode].items()})
    _, selectors, hashes = load_frozen(Path(prepared["rerankers"]), examples, Path(prepared["comparison"]))
    if hashes != prepared["reranker_hashes"]:
        raise ValueError("Frozen reranker changed")
    pools = make_pools(baseline, dpo)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
             "generation_jobs": jobs, "generation_hashes": generation_hashes,
             "configurations": CONFIGURATIONS, "training": False, "paid_judging": False}
    write(output / "manifest.json", state)
    try:
        jar = Path(prepared["repository"]) / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
        conditions, scores, rankings = score_pools(pools, examples, gold, selectors["max5"], output, jar, matcher)
        check_baseline(pools, Path(prepared["comparison"]), prepared["baseline_expected_metrics"], examples, gold)
        comparisons, transitions, mrr_changes = {}, {}, {}
        for mode in MODES:
            for method in ("max5_reranker", "sample_frequency"):
                for before, after in (("baseline", "strict"), ("baseline", "alternative"), ("strict", "alternative")):
                    a, b = f"{before}_{mode}_{method}", f"{after}_{mode}_{method}"
                    name = f"{after}_minus_{before}_{mode}_{method}"
                    comparisons[name] = {"before": a, "after": b, "metrics": contrast(scores[a], scores[b], METRICS, resamples)}
                    transitions[name] = {k: coverage_transitions(scores[a], scores[b], k) for k in METRICS if k != "mrr_at5"}
                    aa, bb = ({r["question_id"]: r["mrr_at5"] for r in scores[n]} for n in (a, b))
                    improved = sorted(q for q in aa if bb[q] > aa[q])
                    worsened = sorted(q for q in aa if bb[q] < aa[q])
                    mrr_changes[name] = {"improved": len(improved), "worsened": len(worsened),
                        "unchanged": len(aa)-len(improved)-len(worsened), "improved_question_ids": improved, "worsened_question_ids": worsened}
            for model in MODELS:
                a, b = (f"solo_{model}_{p}_{mode}_generated_order" for p in ("sft", "dpo"))
                name = f"solo_{model}_dpo_minus_sft_{mode}"
                comparisons[name] = {"before": a, "after": b, "metrics": contrast(scores[a], scores[b], METRICS, resamples)}
                transitions[name] = {k: coverage_transitions(scores[a], scores[b], k) for k in METRICS if k != "mrr_at5"}
        generation, size_rows, warnings = {}, {}, []
        for bank_name, bank in (("sft", baseline), ("dpo", dpo)):
            for model in MODELS:
                for mode in MODES:
                    name = f"{model}_{bank_name}_{mode}"
                    rows = bank[f"{model}_{mode}"]
                    generation[name], size_rows[name] = compactness(rows)
                    samples = [s for r in rows for s in r["samples"]]
                    generation[name].update(total_output_tokens=sum(s["output_tokens"] for s in samples),
                        generation_seconds=sum(r["generation_seconds"] for r in rows),
                        failed_draws=sum(bool(s.get("parse_error")) for s in samples),
                        ceiling_draws=sum(s["output_tokens"] >= 512 for s in samples),
                        capped_draws=sum(bool(s.get("candidate_limit_applied")) for s in samples))
                    if any(generation[name][k] for k in ("failed_draws", "ceiling_draws", "capped_draws")):
                        warnings.append(name + ": failed/capped/ceiling draws retained; inspect counters")
        costs = {}
        for name, settings in CONFIGURATIONS.items():
            for mode in MODES:
                members = [generation[f"{m}_{'dpo' if settings[m] else 'sft'}_{mode}"] for m in MODELS]
                costs[f"{name}_{mode}"] = {
                    "mean_output_tokens_per_response": sum(v["mean_output_tokens_per_response"] for v in members)/3,
                    "mean_candidates_per_response": sum(v["mean_candidates_per_response"] for v in members)/3,
                    "total_output_tokens": sum(v["total_output_tokens"] for v in members),
                    "ensemble_mean_unique_candidates": len(pools[f"{name}_{mode}"])/95}
        result = {"question_count": 95, "configurations": CONFIGURATIONS, "conditions": conditions,
            "paired_contrasts": comparisons, "coverage_transitions": transitions, "mrr_changes": mrr_changes,
            "model_generation": generation, "ensemble_generation": costs, "quality_warnings": warnings,
            "primary_selector": "unchanged frozen max5 logistic with relation labels", "selected_C": prepared["selected_C"],
            "limitations": ["Historically inspected test95; exploratory comparison, no checkpoint selection on these scores.",
                "Strict selection used validation144; all-epoch1 alternative was fixed before this test evaluation.",
                "Four training/validation samples versus ten test samples; one decoding seed.",
                "Official nonacceptance is not a semantic incorrectness label.",
                "Historical pooling, including whitespace normalization, is unchanged to isolate checkpoint effects.",
                "Archived generation times differ in hardware/run conditions and are descriptive only.",
                "Shorter outputs do not establish a proportional walltime reduction."]}
        validate_prepared(submission)
        if digest(jobs_file) != jobs_hash or any(digest(Path(p)) != v for p, v in generation_hashes.items()):
            raise ValueError("Generation changed while scoring")
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scores)
        write(output / "compactness_per_question.json", size_rows)
        write_jsonl(output / "rankings.jsonl", rankings)
        for name, rows in pools.items():
            write_jsonl(output / f"{name}_candidates.jsonl", rows)
        for mode in MODES:
            print("\n" + mode, flush=True)
            for config in CONFIGURATIONS:
                name = f"{config}_{mode}"
                m = conditions[name + "_max5_reranker"]["metrics"]
                z = costs[name]
                print(f"  {config}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, "
                      f"full pool={m['coverage_full_pool']:.4f}, tokens/response={z['mean_output_tokens_per_response']:.2f}, "
                      f"pool size={z['ensemble_mean_unique_candidates']:.2f}", flush=True)
            for config in ("strict", "alternative"):
                name = f"{config}_minus_baseline_{mode}_max5_reranker"
                c = comparisons[name]["metrics"]["mrr_at5"]; t = transitions[name]["coverage_full_pool"]
                print(f"  {config} minus baseline: MRR delta={c['difference']:+.6f}, CI={c['paired_bootstrap_95ci']}; "
                      f"full-pool gained={t['gained']}, lost={t['lost']}", flush=True)
        state.update(status="complete", output_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    print("Saved: " + str(output / "comparison_summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resamples", type=int, default=10000)
    a = p.parse_args()
    analyze(a.submission.resolve(), a.output.resolve(), a.resamples)
