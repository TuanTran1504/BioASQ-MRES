#!/usr/bin/env python3
"""Score original SFT expansion/sampling and optionally compare expansion SFT."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_matched_qwen_expansion import paired_contrast, ranking_metrics, validate_pair


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_inference_pair(expansion, sampling):
    for path in (expansion, sampling):
        validate_pair([path, path], 160)
    configs = [read(path / "config.json") for path in (expansion, sampling)]
    for key in ("model_name", "input_sha256", "max_seq_length", "max_new_tokens", "require_all_snippets", "chat_template_kwargs"):
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Original SFT inference arms differ in {key}")
    if records(expansion / "examples.jsonl") != records(sampling / "examples.jsonl"):
        raise ValueError("Inference arms must use identical questions, evidence and gold")
    if configs[0].get("temperature") != 0 or configs[0].get("response_mode") != "equivalent":
        raise ValueError("Expected greedy equivalent expansion")
    if any(configs[1].get(k) != v for k, v in
           {"num_generations": 10, "temperature": 0.8, "top_p": 0.95, "seed": 3407,
            "response_mode": "single_answer_sampling"}.items()):
        raise ValueError("Unexpected sampling protocol")
    for row in records(sampling / "generations.jsonl"):
        if len(row["samples"]) != 10 or [s["draw"] for s in row["samples"]] != list(range(1, 11)):
            raise ValueError("Every question must retain exactly ten attempted draws")


def efficiency(path):
    generations = records(path / "generations.jsonl")
    result = {key: (sum(row[key] for row in generations) if all(key in row for row in generations) else None)
              for key in ("request_count", "input_tokens", "output_tokens", "generation_seconds")}
    result["note"] = "Generation time excludes model load/preflight; null means older run did not record this field"
    return result


def sampling_diagnostics(path, rows):
    accepted = {row["question_id"]: set(row["matching_answers"]) for row in rows}
    generations = records(path / "generations.jsonl")
    samples = [s for row in generations for s in row["samples"]]
    result = {"attempted_draws": len(samples), "parse_successful_draws": sum(not s["parse_error"] for s in samples),
              "parse_success_rate_per_draw": sum(not s["parse_error"] for s in samples) / len(samples)}
    if any("answers" in s for s in samples):
        result.update(schema_compliant_draws=sum(s.get("schema_compliant", False) for s in samples),
                      recovered_draws=sum("incomplete_top_level_json_recovered" in s.get("validation_issues", []) for s in samples),
                      draws_with_candidate_limit_applied=sum(s.get("candidate_limit_applied", False) for s in samples),
                      invalid_candidate_count=sum(s.get("invalid_candidate_count", 0) for s in samples))
    for k in (1, 5, 10):
        result[f"coverage_within_first_{k}_draws"] = sum(
            any(any(answer in accepted[row["question_id"]] for answer in
                    ([s["answer"]] if "answer" in s else [a["answer"] for a in s["answers"]]))
                for s in row["samples"][:k])
            for row in generations) / len(generations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--expansion-sft-experiment", type=Path)
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    directory = args.experiment_dir.resolve()
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "complete" or manifest["smoke_test"] or manifest["expected_questions"] != 160:
        raise ValueError("Expected a complete full original SFT experiment")
    runs = {name: directory / manifest["runs"][name] for name in
            ("original_sft_expansion", "original_sft_sampling10")}
    validate_inference_pair(*runs.values())
    if args.expansion_sft_experiment:
        other = args.expansion_sft_experiment.resolve()
        other_manifest = read(other / "manifest.json")
        if (other_manifest["status"] != "complete" or other_manifest["smoke_test"]
                or other_manifest["expected_questions"] != 160 or other_manifest["model_size"] != manifest["model_size"]):
            raise ValueError("Expected a completed same-size expansion SFT experiment")
        runs["expansion_sft_expansion"] = other / other_manifest["runs"]["sft"]
        validate_pair([runs["original_sft_expansion"], runs["expansion_sft_expansion"]], 160)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_size": manifest["model_size"], "historical_run": manifest["historical_run"],
              "question_count": 160, "evaluation": manifest["evaluation"],
              "scoring": "official BioASQ Java candidate matcher", "selection": manifest["selection"],
              "conditions": {}, "paired_contrasts": {}, "quality_warnings": [],
              "interpretation": ["Sampling uses ten requests per question; expansion uses one.",
                                 "Comparison with expansion SFT concerns checkpoints: training data and backbone revisions differ.",
                                 "Coverage at ten is an offline diagnostic; ranked submissions use at most five."]}
    scored = {}
    for name, path in runs.items():
        report, summary, rows = analyze_local_expansion(path, jar_path=args.jar)
        metrics, scored[name] = ranking_metrics(rows)
        result["conditions"][name] = {"metrics": metrics, "candidate_diagnostics": summary,
                                      "efficiency": efficiency(path), "report": str(report),
                                      "parser_version": read(path / "config.json").get("parser_version", "original-run-parser")}
        if not summary["parse_success_count"]:
            result["quality_warnings"].append(
                f"{name}: no response produced parseable candidates; scores measure output/parse failure. "
                "Inspect raw responses before drawing conclusions about biomedical accuracy.")
        if name == "original_sft_sampling10":
            result["conditions"][name]["sampling_diagnostics"] = sampling_diagnostics(path, rows)
    contrasts = [("original_sft_expansion", "original_sft_sampling10")]
    if "expansion_sft_expansion" in runs:
        contrasts.append(("original_sft_expansion", "expansion_sft_expansion"))
    for first, second in contrasts:
        contrast = paired_contrast(scored[first], scored[second])
        for values in contrast.values():
            values["second_minus_first"] = values.pop("sft_minus_base")
        result["paired_contrasts"][f"{second}_minus_{first}"] = contrast
    (directory / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (directory / "comparison_per_question.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
