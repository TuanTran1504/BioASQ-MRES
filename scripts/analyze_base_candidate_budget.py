#!/usr/bin/env python3
"""Official paired coverage gains/losses for base greedy max-five/max-ten prompts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import digest, read, records, write
from run_base_candidate_budget import five_prompt
from run_extractive_expansion_8b import parse_equivalent_response
from scripts.analyze_matched_qwen_expansion import ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency
from scripts.summarize_expansion_dpo_coverage import coverage_transitions

JAR = ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"


def validate_experiment(directory, allow_smoke=False):
    manifest = read(directory / "manifest.json")
    expected = 4 if manifest["smoke_test"] else 160
    if (manifest["status"] != "complete" or manifest["expected_questions"] != expected
            or manifest["smoke_test"] and not allow_smoke or set(manifest["runs"]) != {"max10", "max5"}):
        raise ValueError("Expected a complete paired evaluation; smoke scores are not final results")
    source = manifest["provenance"]
    prepared_path = Path(manifest["prepared_manifest"])
    if digest(prepared_path) != manifest["prepared_manifest_sha256"]:
        raise ValueError("Prepared protocol changed")
    prepared = read(prepared_path)
    if source != prepared["models"][manifest["model_key"]]:
        raise ValueError("Backbone provenance differs")
    runs, examples, prompts = {}, [], {}
    for limit in (10, 5):
        condition = f"max{limit}"
        path = (directory / manifest["runs"][condition]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Arm escaped experiment directory")
        for name, value in manifest["arm_hashes"][condition].items():
            if digest(path / name) != value:
                raise ValueError(f"Archived arm changed: {condition}/{name}")
        config = read(path / "config.json")
        expected_settings = {"model_name": source["base_snapshot"], "max_candidates": limit,
            "temperature": 0, "response_mode": "equivalent", "require_all_snippets": True,
            "local_files_only": True, "gold_blind_generation": True, "question_count": expected,
            "model_loader": source["model_loader"], "execution_mode": source["execution_mode"],
            "chat_template_kwargs": source["chat_template_kwargs"], "max_seq_length": 6144,
            "max_new_tokens": 512, "input_sha256": prepared["input_sha256"],
            "prompt_sha256": prepared["prompt_sha256"][str(limit)]}
        if any(config.get(k) != v for k, v in expected_settings.items()):
            raise ValueError("Base model or matched greedy settings differ")
        prompt_path = Path(config["prompt"])
        if digest(prompt_path) != config["prompt_sha256"]:
            raise ValueError("Archived prompt changed")
        prompts[limit] = prompt_path.read_text(encoding="utf-8")
        panel = records(path / "examples.jsonl")
        generations = records(path / "generations.jsonl")
        ids = {r["question_id"] for r in panel}
        if (len(panel) != expected or len(ids) != expected or len(generations) != expected
                or {r["question_id"] for r in generations} != ids or read(path / "status.json")["status"] != "complete"):
            raise ValueError("Missing or duplicate questions; failed parses must stay in denominators")
        if any(r["snippets_truncated"] or r["included_snippets"] != r["total_snippets"]
               or r["request_count"] != 1 for r in generations):
            raise ValueError("Evidence or request budgets differ")
        replay, rejected_replay = [], []
        for row in generations:
            try:
                candidates, rejected, _, _ = parse_equivalent_response(row["raw_response"], limit)
            except ValueError:
                candidates, rejected = [], []
            replay.extend({"question_id": row["question_id"], "position": c["raw_position"], **c} for c in candidates)
            rejected_replay.extend({"question_id": row["question_id"], **c} for c in rejected)
        if replay != records(path / "candidates.jsonl") or rejected_replay != records(path / "invalid_candidates.jsonl"):
            raise ValueError("Saved candidates differ from raw-response parsing and candidate cap")
        runs[condition] = path
        examples.append(panel)
    if examples[0] != examples[1] or prompts[5] != five_prompt(prompts[10]):
        raise ValueError("Conditions must differ only in prompt candidate limit")
    return manifest, runs


def compare(before, after, baseline_metric, resamples=10000):
    # Both sides use a common binary key; the max-ten baseline may use @5 or @10.
    a = [{"question_id": r["question_id"], "coverage": r[baseline_metric]} for r in before]
    b = [{"question_id": r["question_id"], "coverage": r["coverage_at5"]} for r in after]
    transition = coverage_transitions(a, b, "coverage")
    renamed = {}
    for key, value in transition.items():
        renamed[key.replace("sft", "max10").replace("dpo", "max5")] = value
    interval = paired_contrast(a, b, resamples, metrics=("coverage",))["coverage"]
    return {**renamed, "paired_bootstrap_95ci": interval["paired_bootstrap_95ci"],
            "baseline_metric": baseline_metric, "new_metric": "coverage_at5"}


def analyze(directory, jar=JAR, allow_smoke=False):
    manifest, runs = validate_experiment(directory, allow_smoke)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_key": manifest["model_key"], "smoke_test": manifest["smoke_test"],
        "question_count": manifest["expected_questions"], "evaluation": manifest["evaluation"],
        "scoring": "Official BioASQ Java matcher; original generated order; failures retained",
        "conditions": {}, "coverage_comparisons": {}, "quality_warnings": [],
        "source_manifest_sha256": digest(directory / "manifest.json")}
    scored = {}
    for condition, path in runs.items():
        report, summary, rows = analyze_local_expansion(path, jar_path=jar)
        metrics, scored[condition] = ranking_metrics(rows)
        rejected = records(path / "invalid_candidates.jsonl")
        clipped_ids = {r["question_id"] for r in rejected if r.get("reason") == "unique_candidate_limit_exceeded"}
        result["conditions"][condition] = {"metrics": metrics, "candidate_diagnostics": summary,
            "questions_exceeding_unique_candidate_cap": len(clipped_ids),
            "efficiency": efficiency(path), "official_report": str(report)}
        if clipped_ids:
            result["quality_warnings"].append(f"{condition}: {len(clipped_ids)} responses exceeded the candidate cap; surplus candidates excluded and raw outputs retained")
        if not summary["parse_success_count"]:
            result["quality_warnings"].append(f"{condition}: no parseable outputs; smoke/model-format investigation required")
    result["coverage_comparisons"] = {
        "same_five_candidate_budget": compare(scored["max10"], scored["max5"], "coverage_at5"),
        "five_prompt_vs_ten_full_pool": compare(scored["max10"], scored["max5"], "coverage_at10")}
    mrr = paired_contrast(scored["max10"], scored["max5"], metrics=("mrr_at5",))["mrr_at5"]
    result["mrr_at5_contrast"] = {"max5_minus_max10": mrr["sft_minus_base"],
        "paired_bootstrap_95ci": mrr["paired_bootstrap_95ci"]}
    result["interpretation"] = ["Same-five comparison isolates the prompt change at equal submitted candidate budgets.",
        "Five-versus-ten-full-pool comparison measures total coverage gained/lost with a smaller generation budget.",
        "Official acceptance does not establish semantic correctness or evidence support.",
        "Exploratory development-only results; unadjusted intervals/test p-values; no training or paid judge."]
    write(directory / "comparison_per_question.json", scored)
    write(directory / "comparison_summary.json", result)
    print(f"\n{manifest['model_key']}: {'SMOKE ONLY' if manifest['smoke_test'] else '160 development questions'}")
    for condition, report in result["conditions"].items():
        metrics, cost = report["metrics"], report["efficiency"]
        print(f"  {condition}: coverage@5={metrics['coverage_at5']:.4f}, "
              f"coverage@10={metrics['coverage_at10']:.4f}, MRR@5={metrics['mrr_at5']:.6f}, "
              f"output tokens={cost['output_tokens']}, generation seconds={cost['generation_seconds']}")
    for key, comparison in result["coverage_comparisons"].items():
        print(f"  {key}: max10={comparison['max10_covered']}, max5={comparison['max5_covered']}, "
              f"gained={comparison['gained']}, lost={comparison['lost']}, net={comparison['net_questions']:+d}, "
              f"95% CI={comparison['paired_bootstrap_95ci']}")
    print(f"  Saved {directory / 'comparison_summary.json'}", flush=True)
    if result["quality_warnings"]:
        print("  Warnings:", result["quality_warnings"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--jar", type=Path, default=JAR)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    for path in args.experiment_dirs:
        analyze(path.resolve(), args.jar, args.allow_smoke)


if __name__ == "__main__":
    main()
