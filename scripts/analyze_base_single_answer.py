#!/usr/bin/env python3
"""Score base one-answer greedy outputs and pair them with saved five/ten arms."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import digest, read, records, write
from run_base_single_answer import validate_submission
from run_extractive_expansion_8b import parse_equivalent_response
from scripts.analyze_matched_qwen_expansion import ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency
from scripts.summarize_expansion_dpo_coverage import coverage_transitions

JAR = ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"


def validate_single(directory, allow_smoke=False):
    manifest = read(directory / "manifest.json")
    expected = 4 if manifest["smoke_test"] else 160
    if (manifest["status"] != "complete" or manifest["expected_questions"] != expected
            or manifest["smoke_test"] and not allow_smoke or set(manifest["runs"]) != {"max1"}):
        raise ValueError("Expected complete one-answer generation; smoke scores are diagnostic")
    prepared_path = Path(manifest["prepared_manifest"])
    if digest(prepared_path) != manifest["prepared_manifest_sha256"]:
        raise ValueError("Prepared one-answer protocol changed")
    _, source, baselines = validate_submission(prepared_path.parent, manifest["model_key"])
    if (source["provenance"] != manifest["provenance"] or source["baseline"] != manifest["baseline"]
            or source["baseline_manifest_sha256"] != manifest["baseline_manifest_sha256"]):
        raise ValueError("One-answer model differs from the frozen baseline")
    path = (directory / manifest["runs"]["max1"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Single-answer arm escaped experiment directory")
    for name, value in manifest["arm_hashes"]["max1"].items():
        if digest(path / name) != value:
            raise ValueError("Archived single-answer arm changed")
    config = read(path / "config.json")
    provenance = source["provenance"]
    settings = {"model_name": provenance["base_snapshot"], "model_loader": provenance["model_loader"],
        "chat_template_kwargs": provenance["chat_template_kwargs"], "execution_mode": provenance["execution_mode"],
        "max_seq_length": source["max_seq_length"], "max_new_tokens": source["max_new_tokens"],
        "temperature": 0, "max_candidates": 1, "response_mode": "equivalent", "require_all_snippets": True,
        "local_files_only": True, "gold_blind_generation": True, "question_count": expected,
        "input_sha256": source["input_sha256"], "prompt_sha256": source["prompt_sha256"], "seed": source["seed"]}
    if any(config.get(k) != v for k, v in settings.items()):
        raise ValueError("One-answer backbone or matched inference settings differ")
    if digest(Path(config["prompt"])) != source["prompt_sha256"]:
        raise ValueError("One-answer prompt changed")
    panel = records(path / "examples.jsonl")
    baseline_panel = records(baselines["max10"] / "examples.jsonl")
    ids = {r["question_id"] for r in panel}
    generations = records(path / "generations.jsonl")
    if (panel != baseline_panel[:expected] or len(ids) != expected or len(generations) != expected
            or {r["question_id"] for r in generations} != ids or read(path / "status.json")["status"] != "complete"):
        raise ValueError("One-answer questions/evidence differ; failed parses cannot be dropped")
    if any(r["request_count"] != 1 or r["snippets_truncated"] or r["included_snippets"] != r["total_snippets"] for r in generations):
        raise ValueError("One-answer request or evidence budget differs")
    replay, rejected_replay = [], []
    for row in generations:
        try:
            candidates, rejected, _, _ = parse_equivalent_response(row["raw_response"], 1)
        except ValueError:
            candidates, rejected = [], []
        replay.extend({"question_id": row["question_id"], "position": c["raw_position"], **c} for c in candidates)
        rejected_replay.extend({"question_id": row["question_id"], **c} for c in rejected)
    if replay != records(path / "candidates.jsonl") or rejected_replay != records(path / "invalid_candidates.jsonl"):
        raise ValueError("Single-answer candidates differ from raw parsing")
    return manifest, {"max1": path, **baselines}, ids


def paired_results(one, expanded, label, resamples=10000):
    metrics = ("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10")
    contrasts = paired_contrast(one, expanded, resamples=resamples, metrics=metrics)
    output = {m: {f"{label}_minus_max1": c["sft_minus_base"],
                   "paired_bootstrap_95ci": c["paired_bootstrap_95ci"]} for m, c in contrasts.items()}
    for metric in ("coverage_at5", "coverage_at10"):
        transition = coverage_transitions(one, expanded, metric)
        output[metric].update({key.replace("sft", "max1").replace("dpo", label): value
                               for key, value in transition.items()})
    return output


def analyze(directory, jar=JAR, allow_smoke=False):
    manifest, runs, ids = validate_single(directory, allow_smoke)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_key": manifest["model_key"], "smoke_test": manifest["smoke_test"],
        "question_count": len(ids), "evaluation": manifest["evaluation"], "conditions": {}, "paired_contrasts": {},
        "source_manifest_sha256": digest(directory / "manifest.json"), "quality_warnings": [],
        "interpretation": ["One answer uses the same grounding instructions, JSON schema, base revision and greedy settings.",
            "For max1, MRR equals exact accuracy; extra-answer MRR accounts for rank up to five.",
            "Five/ten generations are reused; fresh official scoring reports are added without replacing source files.",
            "Failed parses remain in denominators. Dev-only comparisons are exploratory; intervals and p-values are unadjusted."]}
    scored = {}
    for condition, path in runs.items():
        report, diagnostics, rows = analyze_local_expansion(path, jar_path=jar)
        rows = [row for row in rows if row["question_id"] in ids]
        metrics, scored[condition] = ranking_metrics(rows)
        if condition == "max1" and metrics["mrr_at5"] != metrics["strict_accuracy"]:
            raise ValueError("One-answer MRR must equal exact accuracy")
        costs = efficiency(path) if not manifest["smoke_test"] or condition == "max1" else None
        clipped = {r["question_id"] for r in records(path / "invalid_candidates.jsonl")
                   if r["question_id"] in ids and r.get("reason") == "unique_candidate_limit_exceeded"}
        # Smoke compares the first four saved baseline questions; do not attach
        # 160-question generation totals to a four-question comparison.
        result["conditions"][condition] = {"metrics": metrics, "candidate_diagnostics": diagnostics,
            "diagnostics_question_count": diagnostics["question_count"], "efficiency": costs,
            "questions_exceeding_candidate_cap": len(clipped), "official_report": str(report)}
        if clipped:
            result["quality_warnings"].append(f"{condition}: {len(clipped)} responses exceeded the candidate cap; surplus candidates excluded")
        if not metrics["strict_accuracy"] and not diagnostics["parse_success_count"]:
            result["quality_warnings"].append(f"{condition}: no parseable answers")
    for condition in ("max5", "max10"):
        result["paired_contrasts"][f"{condition}_minus_max1"] = paired_results(scored["max1"], scored[condition], condition)
    write(directory / "comparison_per_question.json", scored)
    write(directory / "comparison_summary.json", result)
    print(f"\n{manifest['model_key']}: {'SMOKE ONLY' if manifest['smoke_test'] else '160 development questions'}")
    for condition, report in result["conditions"].items():
        m = report["metrics"]
        print(f"  {condition}: MRR@5={m['mrr_at5']:.6f}, accuracy@1={m['strict_accuracy']:.4f}, "
              f"coverage@5={m['coverage_at5']:.4f}, coverage@10={m['coverage_at10']:.4f}")
    for name, compared in result["paired_contrasts"].items():
        mrr, coverage = compared["mrr_at5"], compared["coverage_at5"]
        condition = name.split("_minus_")[0]
        print(f"  {name}: MRR delta={mrr[condition + '_minus_max1']:+.6f}, 95% CI={mrr['paired_bootstrap_95ci']}; "
              f"top-five coverage gained={coverage['gained']}, lost={coverage['lost']}, net={coverage['net_questions']:+d}")
    if result["quality_warnings"]:
        print("  Warnings:", result["quality_warnings"])
    print(f"  Saved {directory / 'comparison_summary.json'}", flush=True)
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
