#!/usr/bin/env python3
"""Official paired before/after original SFT scoring with one identical prompt."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import read, write, digest, records
from run_single_answer_sampling import parse_single_answer, unique_candidates, sample_seed, PARSER_VERSION
from scripts.analyze_matched_qwen_expansion import validate_pair, ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency
from scripts.summarize_expansion_dpo_coverage import coverage_transitions
from single_answer_format_recovery import recover_single_answer, VERSION as RECOVERY_VERSION


def score_format_recovered(run, jar):
    from src.notebook_workflows.local_expansion import official_candidate_matches
    examples = records(run / "examples.jsonl")
    generations = records(run / "generations.jsonl")
    report = run / f"format-recovery-{uuid.uuid4().hex[:8]}"
    report.mkdir()
    candidates, decisions = [], []
    for row in generations:
        answer, method = recover_single_answer(row["samples"][0]["raw_response"])
        decisions.append({"question_id": row["question_id"], "answer": answer, "recovery_method": method})
        if answer:
            candidates.append({"question_id": row["question_id"], "answer": answer, "position": 1})
    (report / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidates), encoding="utf-8")
    matches = official_candidate_matches(examples, candidates, report, jar_path=jar) if candidates else {}
    rows = [{"question_id": row["question_id"], "answers": [row["answer"]] if row["answer"] else [],
        "matching_answers": [row["answer"]] if matches.get((row["question_id"], row["answer"]), False) else []}
        for row in decisions]
    metrics, scored = ranking_metrics(rows)
    write(report / "decisions.json", decisions)
    return {"metrics": metrics, "official_report": str(report), "recovery_version": RECOVERY_VERSION,
        "strict_parse_count": sum(r["recovery_method"] == "strict_tagged" for r in decisions),
        "format_recovered_count": sum(r["recovery_method"] == "single_answer_line" for r in decisions),
        "unrecoverable_count": sum(r["answer"] is None for r in decisions)}, scored


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
    recovered_scores = {}
    result["format_recovered_conditions"] = {}
    for name, run in runs.items():
        result["format_recovered_conditions"][name], recovered_scores[name] = score_format_recovered(run, jar)
    result["format_recovered_paired_contrasts"] = paired_contrast(recovered_scores["base"], recovered_scores["sft"])
    result["format_recovery_interpretation"] = (
        "Secondary post-hoc diagnostic: identical gold-blind recovery for both arms. Accept one Answer: line with missing boundary tags; "
        "keep all remaining expression text. Strict tagged scores remain primary. Answer acceptance uses the official matcher; recovery alone does not establish semantic correctness.")
    write(directory / "comparison_summary.json", result)
    write(directory / "comparison_per_question.json", {**scored, **{f"format_recovered_{k}": v for k, v in recovered_scores.items()}})
    a, b = [result["conditions"][name]["metrics"]["mrr_at5"] for name in ("base", "sft")]
    contrast = result["paired_contrasts"]["mrr_at5"]
    print(f"{manifest['model_key']}{' SMOKE ONLY' if manifest['smoke_test'] else ''}: base MRR={a:.6f}, SFT MRR={b:.6f}, "
          f"delta={contrast['sft_minus_base']:+.6f}, 95% CI={contrast['paired_bootstrap_95ci']}")
    a, b = [result["format_recovered_conditions"][name]["metrics"]["mrr_at5"] for name in ("base", "sft")]
    print(f"Format-recovered diagnostic: base MRR={a:.6f}, SFT MRR={b:.6f}, "
          f"delta={result['format_recovered_paired_contrasts']['mrr_at5']['sft_minus_base']:+.6f}")
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
