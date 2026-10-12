#!/usr/bin/env python3
"""Official original-SFT versus DPO one-answer scores, preserving failed questions."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import read, write, digest, records
from run_single_answer_sampling import parse_single_answer, unique_candidates
from scripts.analyze_matched_qwen_expansion import validate_pair, ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency
from scripts.summarize_expansion_dpo_coverage import coverage_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    jar = ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    for directory in args.experiment_dirs:
        manifest = read(directory / "manifest.json")
        if manifest["status"] != "complete" or manifest["expected_questions"] != 160 or manifest["smoke_test"]:
            raise ValueError("Expected completed 160-question paired evaluation")
        runs = {name: (directory / manifest["runs"][name]).resolve() for name in ("sft", "dpo")}
        validate_pair(list(runs.values()), 160)
        result = {"model_key": manifest["model_key"], "pair_strategy": manifest["pair_strategy"], "conditions": {},
                  "scope": "Exploratory dev160 paired one-answer SFT/DPO comparison", "source_manifest_sha256": digest(directory / "manifest.json")}
        scored = {}
        for name, run in runs.items():
            for filename, value in manifest["arm_hashes"][name].items():
                if digest(run / filename) != value:
                    raise ValueError("Archived inference changed")
            expected_candidates = []
            for row in records(run / "generations.jsonl"):
                if len(row["samples"]) != 1 or row["snippets_truncated"]:
                    raise ValueError("Expected one untruncated answer attempt")
                sample = row["samples"][0]
                try:
                    answer = parse_single_answer(sample["raw_response"])
                except ValueError:
                    answer = None
                if sample["answer"] != answer:
                    raise ValueError("Answer differs from raw parse")
                expected_candidates.extend({"question_id": row["question_id"], **r} for r in unique_candidates([sample]))
            if records(run / "candidates.jsonl") != expected_candidates:
                raise ValueError("Archived single-answer candidates changed")
            report, diagnostic, rows = analyze_local_expansion(run, jar_path=jar)
            metrics, scored[name] = ranking_metrics(rows)
            if metrics["mrr_at5"] != metrics["strict_accuracy"]:
                raise ValueError("Single-answer MRR must equal exact accuracy")
            result["conditions"][name] = {"metrics": metrics, "candidate_diagnostics": diagnostic, "efficiency": efficiency(run), "report": str(report)}
        result["paired_contrasts"] = {metric: {"dpo_minus_sft": contrast["sft_minus_base"],
            **{key: value for key, value in contrast.items() if key != "sft_minus_base"}}
            for metric, contrast in paired_contrast(scored["sft"], scored["dpo"]).items()}
        result["coverage_transitions"] = coverage_transitions(scored["sft"], scored["dpo"], "coverage_at1")
        write(directory / "comparison_summary.json", result)
        write(directory / "comparison_per_question.json", scored)
        a, b = [result["conditions"][name]["metrics"]["mrr_at5"] for name in ("sft", "dpo")]
        contrast = result["paired_contrasts"]["mrr_at5"]
        print(f"{manifest['model_key']} {manifest['pair_strategy']}: SFT MRR={a:.6f}, DPO MRR={b:.6f}, "
              f"delta={contrast['dpo_minus_sft']:+.6f}, 95% CI={contrast['paired_bootstrap_95ci']}")


if __name__ == "__main__":
    main()
