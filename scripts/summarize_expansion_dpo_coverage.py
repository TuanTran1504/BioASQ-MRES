#!/usr/bin/env python3
"""Compare saved full-pool coverage, coverage@10 and MRR@5 without new scoring.

Reads complete officially scored experiments, verifies archived matches against
candidate pools, and reports gains AND losses across all 160 questions. No GPU,
generation, Java or judge requests are needed.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_expansion_dpo_cases import by_question, load_scored
from scripts.analyze_expansion_dpo_evaluation import digest
from scripts.analyze_matched_qwen_expansion import paired_contrast

COVERAGE = ("coverage_full_pool", "coverage_at10")
BACKBONES = {"llama31", "qwen3", "ministral3"}


def coverage_transitions(sft, dpo, metric):
    before, after = by_question(sft), by_question(dpo)
    if not before or set(before) != set(after):
        raise ValueError("Expected nonempty, identical question sets")
    if any(row[metric] not in (0, 1) for row in sft + dpo):
        raise ValueError("Coverage must be binary at question level")
    groups = {name: [] for name in ("both_covered", "neither_covered", "gained", "lost")}
    for qid in sorted(before):
        a, b = bool(before[qid][metric]), bool(after[qid][metric])
        group = ("both_covered" if a else "gained") if b else ("lost" if a else "neither_covered")
        groups[group].append(qid)
    n = len(before)
    both, neither, gained, lost = [len(groups[k]) for k in groups]
    discordant = gained + lost
    # Exact two-sided McNemar test, conditional on discordant pairs.
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
                 / 2 ** discordant) if discordant else 1.0
    return {"question_count": n, "sft_covered": both + lost, "dpo_covered": both + gained,
            "sft_rate": (both + lost) / n, "dpo_rate": (both + gained) / n,
            "gained": gained, "lost": lost, "both_covered": both, "neither_covered": neither,
            "net_questions": gained - lost, "dpo_minus_sft": (gained - lost) / n,
            "gained_question_ids": groups["gained"], "lost_question_ids": groups["lost"],
            "mcnemar_exact_two_sided_p": pvalue,
            "observed_gains_without_losses": gained > 0 and lost == 0}


def summarize_scored(directory, loaded, resamples=10000):
    manifest, examples, arms, hashes = loaded
    result = {"model_key": manifest["model_key"], "question_count": len(examples),
              "comparisons": {}, "source_hashes": {"arms": hashes,
                  **{name: digest(directory / name) for name in ("manifest.json", "comparison_summary.json", "comparison_per_question.json")}}}
    for mode in ("sampling10", "greedy"):
        sft, dpo = [list(arms[p + "_" + mode]["scores"].values()) for p in ("sft", "dpo")]
        contrasts = paired_contrast(sft, dpo, resamples=resamples, metrics=(*COVERAGE, "mrr_at5"))
        compared = {}
        for metric in COVERAGE:
            compared[metric] = {**coverage_transitions(sft, dpo, metric),
                "paired_bootstrap_95ci": contrasts[metric]["paired_bootstrap_95ci"]}
        mrr = contrasts["mrr_at5"]
        compared["mrr_at5"] = {"sft": sum(r["mrr_at5"] for r in sft) / len(sft),
            "dpo": sum(r["mrr_at5"] for r in dpo) / len(dpo), "dpo_minus_sft": mrr["sft_minus_base"],
            "paired_bootstrap_95ci": mrr["paired_bootstrap_95ci"]}
        result["comparisons"][mode] = compared
    result["paired_bootstrap"] = {"resamples": resamples, "seed": 3407}
    return result


def combined_full_pool(loaded, resamples=10000):
    """Question-level union of accepted candidates; no ordering is implied."""
    sources = {data[0]["model_key"]: data for _, data in loaded}
    if len(loaded) != 3 or set(sources) != BACKBONES:
        raise ValueError("Combined comparison requires Llama, Qwen3 and Ministral exactly once")
    qids = set(next(iter(sources.values()))[1])
    if not qids or any(set(data[1]) != qids for data in sources.values()):
        raise ValueError("Combined pool questions differ")
    result = {"sources": sorted(sources), "question_count": len(qids),
        "pool": "All candidates from the same three backbones per policy; a question is covered if any backbone has an accepted candidate",
        "ranking": "Full-pool oracle coverage only; no combined coverage@10 or MRR@5 ordering is defined",
        "comparisons": {}, "paired_bootstrap": {"resamples": resamples, "seed": 3407}}
    for mode in ("sampling10", "greedy"):
        scores = {p: {model: data[2][p + "_" + mode]["scores"] for model, data in sources.items()}
                  for p in ("sft", "dpo")}
        if any(set(rows) != qids for policy in scores.values() for rows in policy.values()):
            raise ValueError("Combined scoring question membership differs")
        if any(row["coverage_full_pool"] not in (0, 1)
               for policy in scores.values() for rows in policy.values() for row in rows.values()):
            raise ValueError("Combined source coverage must be binary")
        pooled = {p: [{"question_id": q, "coverage_full_pool": float(any(
            scores[p][m][q]["coverage_full_pool"] for m in sources))} for q in sorted(qids)]
            for p in ("sft", "dpo")}
        transitions = coverage_transitions(pooled["sft"], pooled["dpo"], "coverage_full_pool")
        contrast = paired_contrast(pooled["sft"], pooled["dpo"], resamples=resamples,
                                   metrics=("coverage_full_pool",))["coverage_full_pool"]
        gained = set(transitions["gained_question_ids"])
        original_union = {r["question_id"] for r in pooled["sft"] if r["coverage_full_pool"]}
        individual = {}
        for model in sorted(sources):
            own_gains = {q for q in qids if scores["dpo"][model][q]["coverage_full_pool"]
                         and not scores["sft"][model][q]["coverage_full_pool"]}
            individual[model] = {"gained_question_ids": sorted(own_gains),
                "new_to_combined_pool_question_ids": sorted(own_gains & gained),
                "already_covered_by_another_sft_model_question_ids": sorted(own_gains & original_union)}
        result["comparisons"][mode] = {"coverage_full_pool": {
                **transitions, "paired_bootstrap_95ci": contrast["paired_bootstrap_95ci"]},
            "requests_per_question_per_policy": 30 if mode == "sampling10" else 3,
            "max_candidate_slots_before_deduplication": 300 if mode == "sampling10" else 30,
            "individual_gain_overlap": individual,
            "per_question": pooled}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New combined summary; source scoring files are preserved")
    parser.add_argument("--combine-models", action="store_true", help="Also compare the complete three-backbone SFT and DPO unions")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Summary destination already exists; choose a new filename: " + str(args.output))
    loaded = [(p.resolve(), load_scored(p.resolve())) for p in args.experiment_dirs]
    models = [data[0]["model_key"] for _, data in loaded]
    if len(set(models)) != len(models):
        raise ValueError("Duplicate backbone experiment")
    if args.combine_models and set(models) != BACKBONES:
        raise ValueError("Combined comparison requires Llama, Qwen3 and Ministral exactly once")
    if any(data[1] != loaded[0][1][1] for _, data in loaded[1:]):
        raise ValueError("Backbones have different questions or evidence")
    result = {"scope": "Exploratory development-only SFT/DPO comparison; one training seed",
        "primary_generation_measure": "coverage_full_pool: all unique candidates from ten complete expansion draws (up to 100)",
        "fixed_candidate_budget_measure": "coverage_at10: first ten unique candidates in recorded order",
        "submission_measure": "mrr_at5: first five unique candidates in recorded order",
        "interpretation": ["Net improvement can contain losses; read gained and lost together.",
            "An answer moving from rank 7 to 11 loses coverage@10 but retains full-pool coverage.",
            "Accepted means matched by the archived official scorer, not an audited semantic correctness judgment.",
            "Failed parses remain in the 160-question denominator.",
            "Intervals represent question-sampling uncertainty, not training-seed variation; p-values are unadjusted exploratory diagnostics.",
            "No new official scoring or generation is performed."],
        "models": {data[0]["model_key"]: summarize_scored(p, data) for p, data in loaded}}
    if args.combine_models:
        result["combined_full_pool"] = combined_full_pool(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    for model, report in result["models"].items():
        print(f"\n{model}: {report['question_count']} questions")
        for mode, metrics in report["comparisons"].items():
            for metric in COVERAGE:
                r = metrics[metric]
                print(f"  {mode} {metric}: SFT={r['sft_covered']}, DPO={r['dpo_covered']}, "
                      f"gained={r['gained']}, lost={r['lost']}, net={r['net_questions']:+d}, "
                      f"95% CI={r['paired_bootstrap_95ci']}, McNemar p={r['mcnemar_exact_two_sided_p']:.4f}")
            r = metrics["mrr_at5"]
            print(f"  {mode} MRR@5: SFT={r['sft']:.6f}, DPO={r['dpo']:.6f}, "
                  f"delta={r['dpo_minus_sft']:+.6f}, 95% CI={r['paired_bootstrap_95ci']}")
    if args.combine_models:
        print("\nCOMBINED THREE-MODEL FULL POOL (same source composition for SFT and DPO)")
        for mode, compared in result["combined_full_pool"]["comparisons"].items():
            r = compared["coverage_full_pool"]
            print(f"  {mode}: SFT={r['sft_covered']}/{r['question_count']}, DPO={r['dpo_covered']}/{r['question_count']}, "
                  f"gained={r['gained']}, lost={r['lost']}, net={r['net_questions']:+d}, "
                  f"95% CI={r['paired_bootstrap_95ci']}, McNemar p={r['mcnemar_exact_two_sided_p']:.4f}")
            for model, overlap in compared["individual_gain_overlap"].items():
                print(f"    {model}: individual gains={len(overlap['gained_question_ids'])}, "
                      f"new to union={len(overlap['new_to_combined_pool_question_ids'])}, "
                      f"already in another SFT pool={len(overlap['already_covered_by_another_sft_model_question_ids'])}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
