#!/usr/bin/env python3
"""Score the three matched 8B inference conditions and every paired contrast."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_matched_qwen_expansion import paired_contrast, ranking_metrics, validate_pair
from scripts.analyze_original_qwen25_inference import efficiency, read, records, sampling_diagnostics

CONDITIONS = ("original_greedy", "original_sampling10", "expansion_greedy")


def validate_experiment(directory, conditions=CONDITIONS):
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "complete" or manifest["smoke_test"] or manifest["expected_questions"] != 160:
        raise ValueError("Expected a complete full 160-question matched evaluation")
    if set(manifest["runs"]) != set(conditions):
        raise ValueError("Expected all declared inference arms")
    runs = {}
    configs = {}
    examples = None
    for name in conditions:
        path = (directory / manifest["runs"][name]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Inference arm must stay within the experiment directory")
        validate_pair([path, path], 160)
        saved_examples = records(path / "examples.jsonl")
        if examples is not None and examples != saved_examples:
            raise ValueError("Inference arms have different questions, evidence or gold")
        examples = saved_examples
        config = read(path / "config.json")
        formulation = "expansion" if name.startswith("expansion_") else "original"
        provenance = manifest["provenance"][formulation]
        # Use the archived prompt so complete experiments can be copied locally.
        prompt = directory / (formulation + "_prompt.txt")
        if (hashlib.sha256(prompt.read_bytes()).hexdigest() != config["prompt_sha256"]
                or prompt.read_text(encoding="utf-8") != provenance["system_prompt"]):
            raise ValueError("Inference prompt differs from its trained formulation")
        if (config["model_name"] != provenance["adapter"]
                or config["model_loader"] != provenance["model_loader"]
                or config["chat_template_kwargs"] != provenance["chat_template_kwargs"]):
            raise ValueError("Inference uses the wrong adapter or native template settings")
        if config["input_sha256"] != manifest["input_sha256"] or not config["gold_blind_generation"]:
            raise ValueError("Unpinned input or generation not marked gold blind")
        if not config["require_all_snippets"] or config["mark_snippets"]:
            raise ValueError("Matched evaluation must retain the exact unmarked evidence")
        expected = {"num_generations": 1, "temperature": 0,
                    "response_mode": "equivalent" if formulation == "expansion" else "single_answer_greedy"}
        if name == "original_sampling10":
            expected.update(num_generations=10, temperature=0.8, top_p=0.95, seed=3407,
                            response_mode="single_answer_sampling")
        if name == "expansion_sampling10":
            expected.update(num_generations=10, temperature=0.8, top_p=0.95, top_k=0, seed=3407,
                            response_mode="equivalent_sampling")
        if any(config.get(field) != value for field, value in expected.items()):
            raise ValueError("Unexpected inference protocol")
        for row in records(path / "generations.jsonl"):
            if row["snippets_truncated"] or row["included_snippets"] != row["total_snippets"]:
                raise ValueError("Inference truncated evidence")
            if name != "expansion_greedy":
                count = expected["num_generations"]
                if len(row["samples"]) != count or [s["draw"] for s in row["samples"]] != list(range(1, count + 1)):
                    raise ValueError("Missing attempted single-answer draws")
                for sample in row["samples"]:
                    seed = int.from_bytes(hashlib.sha256(
                        f"{config['seed']}:{row['question_id']}:{sample['draw']}".encode()).digest()[:4], "big")
                    if sample["seed"] != seed:
                        raise ValueError("Sample seed differs from the declared protocol")
        runs[name], configs[name] = path, config
    for field in ("max_seq_length", "max_new_tokens", "chat_template_kwargs", "model_loader", "seed"):
        if len({json.dumps(config[field], sort_keys=True) for config in configs.values()}) != 1:
            raise ValueError(f"Inference arms differ in {field}")
    original, expansion = (manifest["provenance"][key] for key in ("original", "expansion"))
    for field in ("base_snapshot", "base_revision", "matrix_sha256"):
        if original[field] != expansion[field]:
            raise ValueError(f"Training pair differs in {field}")
    if ("original_greedy" in configs and "original_sampling10" in configs and
            configs["original_greedy"]["prompt_sha256"] != configs["original_sampling10"]["prompt_sha256"]):
        raise ValueError("Original greedy and sampled prompts differ")
    return manifest, runs


def score_experiment(directory, jar):
    manifest, runs = validate_experiment(directory)
    return score_validated_runs(directory, manifest, runs, jar)


def score_validated_runs(directory, manifest, runs, jar):
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_key": manifest["model_key"], "question_count": 160,
              "evaluation": manifest["evaluation"], "selection": manifest["selection"],
              "scoring": "official BioASQ Java candidate matcher", "conditions": {},
              "paired_contrasts": {}, "quality_warnings": [],
              "interpretation": ["Greedy original and greedy expansion each use one request per question.",
                                 "Ten-sample original uses ten requests; efficiency is reported separately.",
                                 "Both SFT formulations share fitting/validation questions and a pinned backbone.",
                                 "Development-only results from one training seed; coverage at ten is diagnostic."]}
    scored = {}
    modes = {key: source.get("execution_mode", "default") for key, source in manifest["provenance"].items()}
    result["training_execution_modes"] = modes
    if len(set(modes.values())) > 1:
        result["quality_warnings"].append("Training execution modes differ between SFT formulations; interpret this as a backend-adjusted comparison.")
    for name, path in runs.items():
        report, summary, rows = analyze_local_expansion(path, jar_path=jar)
        metrics, scored[name] = ranking_metrics(rows)
        if "expansion_sampling10" in runs:
            for row, scored_row in zip(rows, scored[name]):
                scored_row["coverage_full_pool"] = float(bool(row["matching_answers"]))
            metrics["coverage_full_pool"] = sum(r["coverage_full_pool"] for r in scored[name]) / len(rows)
        result["conditions"][name] = {"metrics": metrics, "candidate_diagnostics": summary,
                                      "efficiency": efficiency(path), "report": str(report)}
        if not summary["parse_success_count"]:
            result["quality_warnings"].append(f"{name}: zero parseable responses; inspect raw outputs before interpreting accuracy.")
        if name in ("original_sampling10", "expansion_sampling10"):
            result["conditions"][name]["sampling_diagnostics"] = sampling_diagnostics(path, rows)
    contrasts = [("original_greedy", "original_sampling10"),
                          ("original_greedy", "expansion_greedy"),
                          ("original_sampling10", "expansion_greedy")]
    if "expansion_sampling10" in runs:
        contrasts.extend((name, "expansion_sampling10") for name in CONDITIONS)
        result["interpretation"].append("Ten-sample expansion uses ten requests and at most 100 candidates before deduplication; report fixed top-ten and full-pool coverage separately.")
    for first, second in contrasts:
        metrics = ("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10")
        if "expansion_sampling10" in runs:
            metrics += ("coverage_full_pool",)
        contrast = paired_contrast(scored[first], scored[second], metrics=metrics)
        for values in contrast.values():
            values["second_minus_first"] = values.pop("sft_minus_base")
        result["paired_contrasts"][f"{second}_minus_{first}"] = contrast
    for filename, value in (("comparison_summary.json", result), ("comparison_per_question.json", scored)):
        (directory / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    for directory in args.experiment_dirs:
        print(json.dumps(score_experiment(directory.resolve(), args.jar), indent=2))


if __name__ == "__main__":
    main()
