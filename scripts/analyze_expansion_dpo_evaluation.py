#!/usr/bin/env python3
"""Score matched expansion SFT/DPO generation with official BioASQ matching."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from expansion_dpo_data import strict_answers
from matched_8b_provenance import validate_provenance
from run_extractive_expansion_8b import parse_equivalent_response
from run_single_answer_sampling import expansion_candidates, sample_seed
from scripts.analyze_matched_qwen_expansion import paired_contrast, ranking_metrics, validate_pair
from scripts.analyze_original_qwen25_inference import efficiency, read, records, sampling_diagnostics

CONDITIONS = ("sft_greedy", "sft_sampling10", "dpo_greedy", "dpo_sampling10")
INPUT_SHA = "0b4b342f62391e0d6d21577957a454ffbfe8a2619978462c03bbdf36161379b8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_experiment(directory):
    directory = directory.resolve()
    manifest = read(directory / "manifest.json")
    if (manifest["status"] != "complete" or manifest["smoke_test"]
            or manifest["expected_questions"] != 160 or set(manifest["runs"]) != set(CONDITIONS)
            or manifest["input_sha256"] != INPUT_SHA):
        raise ValueError("Expected a complete pinned 160-question SFT/DPO experiment")
    sources = manifest["provenance"]
    if set(sources) != {"sft", "dpo"}:
        raise ValueError("Expected SFT and DPO provenance")
    training = manifest["dpo_training_manifest"]
    if (training["status"] != "complete" or training["smoke_test"]
            or training["model_key"] != manifest["model_key"]
            or training["adapter_files_sha256"] != sources["dpo"]["adapter_files_sha256"]):
        raise ValueError("DPO training provenance differs")
    validate_provenance({"expansion": training["source"]}, {"expansion": sources["sft"]})
    for field in ("base_snapshot", "base_revision", "matrix_sha256", "system_prompt", "model_loader",
                  "chat_template_kwargs", "execution_mode"):
        if sources["sft"].get(field, "default" if field == "execution_mode" else None) != sources["dpo"].get(field, "default" if field == "execution_mode" else None):
            raise ValueError("SFT/DPO sources differ in " + field)
    if sources["sft"]["adapter"] == sources["dpo"]["adapter"]:
        raise ValueError("SFT and DPO must use distinct adapters")
    prompt = directory / "expansion_prompt.txt"
    if prompt.read_text(encoding="utf-8") != sources["sft"]["system_prompt"]:
        raise ValueError("Archived expansion prompt differs")
    examples, runs = None, {}
    for name in CONDITIONS:
        path = (directory / manifest["runs"][name]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Inference arm escaped experiment directory")
        validate_pair([path, path], 160)
        saved = records(path / "examples.jsonl")
        if examples is not None and examples != saved:
            raise ValueError("Questions, evidence or gold differ across arms")
        examples = saved
        config = read(path / "config.json")
        sampled = name.endswith("sampling10")
        source = sources[name.split("_", 1)[0]]
        expected = {"model_name": source["adapter"], "model_loader": source["model_loader"],
                    "chat_template_kwargs": source["chat_template_kwargs"],
                    "input_sha256": INPUT_SHA, "prompt_sha256": digest(prompt),
                    "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407,
                    "require_all_snippets": True, "mark_snippets": False, "gold_blind_generation": True,
                    "num_generations": 10 if sampled else 1, "temperature": 0.8 if sampled else 0,
                    "prompt_version": "matched-8b-expansion-native-v1",
                    "response_mode": "equivalent_sampling" if sampled else "equivalent"}
        if sampled:
            expected.update(top_p=0.95, top_k=0)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Inference protocol differs for " + name)
        if config.get("execution_mode", "default") != source.get("execution_mode", "default"):
            raise ValueError("Wrong inference backend")
        reconstructed = []
        for row in records(path / "generations.jsonl"):
            if row["snippets_truncated"] or row["included_snippets"] != row["total_snippets"]:
                raise ValueError("Truncated evidence")
            if row["request_count"] != (10 if sampled else 1):
                raise ValueError("Wrong request budget")
            if sampled:
                samples = row["samples"]
                if [s["draw"] for s in samples] != list(range(1, 11)):
                    raise ValueError("Missing or reordered draws")
                for sample in samples:
                    if sample["seed"] != sample_seed(row["question_id"], sample["draw"], 3407):
                        raise ValueError("Wrong sample seed")
                    try:
                        answers, _, _, _ = parse_equivalent_response(sample["raw_response"])
                    except ValueError:
                        answers = []
                    if answers != sample["answers"]:
                        raise ValueError("Saved sampled candidates differ from raw completion")
                reconstructed.extend({"question_id": row["question_id"], **c}
                                     for c in expansion_candidates(samples))
            else:
                try:
                    answers, _, _, _ = parse_equivalent_response(row["raw_response"])
                except ValueError:
                    answers = []
                reconstructed.extend({"question_id": row["question_id"], "position": c["raw_position"], **c}
                                     for c in answers)
        if records(path / "candidates.jsonl") != reconstructed:
            raise ValueError("Pool does not preserve parser and draw/response ordering")
        runs[name] = path
    return manifest, runs


def score_experiment(directory, jar):
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    manifest, runs = validate_experiment(directory)
    result = {"model_key": manifest["model_key"], "question_count": 160,
              "evaluation": manifest["evaluation"], "selection": manifest["selection"],
              "scoring": "official BioASQ Java candidate matcher", "conditions": {},
              "paired_contrasts": {}, "quality_warnings": [],
              "dpo_global_steps": manifest["dpo_training_manifest"]["global_steps"],
              "interpretation": ["Only compare SFT and DPO at the same inference budget.",
                  "All failed parses remain in denominators; no gold-guided reranking.",
                  "Sampled expansion may contain up to 100 candidates before deduplication.",
                  "Coverage at ten and full-pool coverage are generation diagnostics, not five-answer submission scores.",
                  "Schema validity and exact matching do not measure scientific correctness or evidence support.",
                  "Single-seed, development-only pilot using automatically labelled preferences."]}
    scored = {}
    for name, path in runs.items():
        report, summary, rows = analyze_local_expansion(path, jar_path=jar)
        metrics, scored[name] = ranking_metrics(rows)
        for row, values in zip(rows, scored[name]):
            values["coverage_full_pool"] = float(bool(row["matching_answers"]))
        metrics["coverage_full_pool"] = sum(r["coverage_full_pool"] for r in scored[name]) / len(rows)
        generations = records(path / "generations.jsonl")
        raw = [s["raw_response"] for r in generations for s in r["samples"]] if name.endswith("sampling10") else [r["raw_response"] for r in generations]
        strict = 0
        for text in raw:
            try:
                strict_answers(text)
                strict += 1
            except ValueError:
                pass
        result["conditions"][name] = {"metrics": metrics, "candidate_diagnostics": summary,
                "efficiency": efficiency(path), "report": str(report),
                "strict_training_schema": {"valid_draws": strict, "attempted_draws": len(raw),
                                           "rate": strict / len(raw)}}
        if name.endswith("sampling10"):
            result["conditions"][name]["sampling_diagnostics"] = sampling_diagnostics(path, rows)
        if not summary["parse_success_count"]:
            result["quality_warnings"].append(name + ": zero parseable responses; inspect raw completions")
    for inference in ("greedy", "sampling10"):
        contrast = paired_contrast(scored["sft_" + inference], scored["dpo_" + inference],
                                   metrics=("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10", "coverage_full_pool"))
        for values in contrast.values():
            values["dpo_minus_sft"] = values.pop("sft_minus_base")
        result["paired_contrasts"]["dpo_minus_sft_" + inference] = contrast
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
