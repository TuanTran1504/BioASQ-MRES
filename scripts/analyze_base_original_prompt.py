#!/usr/bin/env python3
"""Official paired before/after original SFT scoring with one identical prompt."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import read, write, digest, records
from run_single_answer_sampling import parse_single_answer, unique_candidates, sample_seed, PARSER_VERSION
from scripts.analyze_matched_qwen_expansion import validate_pair, ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency
from scripts.summarize_expansion_dpo_coverage import coverage_transitions


def validate_experiment(directory, allow_smoke=False):
    manifest = read(directory / "manifest.json")
    count = 4 if manifest["smoke_test"] else 160
    if (manifest["status"] != "complete" or manifest["expected_questions"] != count
            or manifest["smoke_test"] and not allow_smoke or set(manifest["runs"]) != {"base", "sft"}):
        raise ValueError("Expected complete paired generation; smoke is diagnostic only")
    prompt = directory / "original_prompt.txt"
    if digest(prompt) != manifest["prompt_sha256"] or prompt.read_text(encoding="utf-8") != manifest["provenance"]["system_prompt"]:
        raise ValueError("Prompt differs from original SFT training")
    runs = {name: (directory / value).resolve() for name, value in manifest["runs"].items()}
    if any(not run.is_relative_to(directory.resolve()) for run in runs.values()):
        raise ValueError("Inference arm escaped experiment")
    validate_pair([runs["base"], runs["sft"]], count)
    source, settings, plan = [manifest[k] for k in ("provenance", "settings", "input_plan")]
    if (settings["max_seq_length"] != plan["max_prompt_tokens"] + settings["max_new_tokens"]
            or settings["max_seq_length"] > plan["native_context_tokens"]
            or settings["expected_prompt_tokens"] != plan["expected_prompt_tokens"]
            or settings["expected_prompt_ids_sha256"] != plan["expected_prompt_ids_sha256"]):
        raise ValueError("Input allocation differs from complete-input plan")
    counts_by_arm = {}
    for name, run in runs.items():
        for filename, expected in manifest["arm_hashes"][name].items():
            if digest(run / filename) != expected:
                raise ValueError("Archived inference changed")
        config = read(run / "config.json")
        target = source["base_snapshot"] if name == "base" else source["adapter"]
        expected = {**settings, "model_name": target, "question_count": count,
            "input_sha256": manifest["input_sha256"], "prompt_sha256": manifest["prompt_sha256"],
            "parser_version": PARSER_VERSION, "gold_blind_generation": True, "local_files_only": True}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Inference configuration differs from matched protocol")
        candidates, counts_by_arm[name] = [], {}
        for row in records(run / "generations.jsonl"):
            if (len(row["samples"]) != 1 or row["request_count"] != 1 or row["snippets_truncated"]
                    or row["included_snippets"] != row["total_snippets"]
                    or row["prompt_tokens"] != plan["expected_prompt_tokens"][row["question_id"]]):
                raise ValueError("Single attempt or complete evidence invariant failed")
            sample = row["samples"][0]
            if sample["draw"] != 1 or sample["seed"] != sample_seed(row["question_id"], 1, settings["seed"]):
                raise ValueError("Generation seed changed")
            try:
                answer = parse_single_answer(sample["raw_response"])
            except ValueError:
                answer = None
            if sample["answer"] != answer:
                raise ValueError("Parsed answer differs from raw response")
            candidates.extend({"question_id": row["question_id"], **r} for r in unique_candidates([sample]))
            counts_by_arm[name][row["question_id"]] = row["prompt_tokens"]
        if records(run / "candidates.jsonl") != candidates:
            raise ValueError("Candidates differ from raw parsing")
    if counts_by_arm["base"] != counts_by_arm["sft"]:
        raise ValueError("Base/SFT input token counts differ")
    return manifest, runs


def analyze(directory, jar, allow_smoke=False):
    manifest, runs = validate_experiment(directory, allow_smoke)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_key": manifest["model_key"], "question_count": manifest["expected_questions"],
        "smoke_test": manifest["smoke_test"], "conditions": {}, "quality_warnings": [],
        "evaluation": manifest["evaluation"], "input_plan": manifest["input_plan"],
        "source_manifest_sha256": digest(directory / "manifest.json"),
        "interpretation": "Same original tagged prompt, complete snippets, native backbone revision, parser and greedy settings; exploratory dev results from one training seed."}
    scored = {}
    for name, run in runs.items():
        report, diagnostic, rows = analyze_local_expansion(run, jar_path=jar)
        metrics, scored[name] = ranking_metrics(rows)
        if metrics["mrr_at5"] != metrics["strict_accuracy"]:
            raise ValueError("Single-answer MRR must equal exact accuracy")
        result["conditions"][name] = {"metrics": metrics, "candidate_diagnostics": diagnostic,
            "efficiency": efficiency(run), "official_report": str(report)}
        if diagnostic["parse_success_count"] < manifest["expected_questions"]:
            result["quality_warnings"].append(f"{name}: format failures retained in denominators; inspect parse rate")
    result["paired_contrasts"] = paired_contrast(scored["base"], scored["sft"])
    transitions = coverage_transitions(scored["base"], scored["sft"], "coverage_at1")
    result["coverage_transitions"] = {key.replace("sft", "base").replace("dpo", "sft"): value for key, value in transitions.items()}
    write(directory / "comparison_summary.json", result)
    write(directory / "comparison_per_question.json", scored)
    a, b = [result["conditions"][name]["metrics"]["mrr_at5"] for name in ("base", "sft")]
    contrast = result["paired_contrasts"]["mrr_at5"]
    print(f"{manifest['model_key']}{' SMOKE ONLY' if manifest['smoke_test'] else ''}: base MRR={a:.6f}, SFT MRR={b:.6f}, "
          f"delta={contrast['sft_minus_base']:+.6f}, 95% CI={contrast['paired_bootstrap_95ci']}")
    print(f"Saved: {directory / 'comparison_summary.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    for directory in args.experiment_dirs:
        analyze(directory.resolve(), args.jar, args.allow_smoke)


if __name__ == "__main__":
    main()
