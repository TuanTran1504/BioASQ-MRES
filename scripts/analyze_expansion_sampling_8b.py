#!/usr/bin/env python3
"""Compare ten expansion draws with the archived matched 8B evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_matched_8b_evaluation import validate_experiment, score_validated_runs
from scripts.analyze_original_qwen25_inference import read, records


def validate_sampling_comparison(directory, baseline):
    manifest, sampled = validate_experiment(directory, ("expansion_sampling10",))
    previous, runs = validate_experiment(baseline)
    if (hashlib.sha256((baseline / "manifest.json").read_bytes()).hexdigest() != manifest["baseline_manifest_sha256"]
            or manifest["model_key"] != previous["model_key"]
            or manifest["provenance"] != previous["provenance"]
            or manifest["input_sha256"] != previous["input_sha256"]):
        raise ValueError("Baseline adapter provenance, data or manifest differs")
    path = sampled["expansion_sampling10"]
    if records(path / "examples.jsonl") != records(runs["expansion_greedy"] / "examples.jsonl"):
        raise ValueError("Baseline and sampled evaluation questions/evidence differ")
    new, old = read(path / "config.json"), read(runs["expansion_greedy"] / "config.json")
    for field in ("max_seq_length", "max_new_tokens", "seed", "prompt_sha256", "model_name"):
        if new[field] != old[field]:
            raise ValueError(f"Expansion inference differs in {field}")
    actual = records(path / "candidates.jsonl")
    expected = []
    for row in records(path / "generations.jsonl"):
        seen = set()
        position = 0
        for sample in row["samples"]:
            if len(sample["answers"]) > 10:
                raise ValueError("More than ten candidates in one draw")
            for answer in sample["answers"]:
                key = " ".join(answer["answer"].casefold().split())
                if key in seen:
                    continue
                seen.add(key)
                position += 1
                expected.append({"question_id": row["question_id"], **answer,
                                 "position": position, "draw": sample["draw"],
                                 "within_draw_position": answer["raw_position"]})
        if bool(row.get("parse_error")) != (position == 0) or row["request_count"] != 10:
            raise ValueError("Pool status or request count differs from saved draws")
    if actual != expected:
        raise ValueError("Candidates do not preserve deduplicated draw/within-response order")
    return manifest, {**runs, **sampled}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path, help="Override archived baseline path after copying experiments")
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    if args.baseline and len(args.experiment_dirs) != 1:
        parser.error("--baseline requires one experiment directory")
    for directory in args.experiment_dirs:
        directory = directory.resolve()
        baseline = args.baseline or Path(read(directory / "manifest.json")["baseline_experiment"])
        manifest, runs = validate_sampling_comparison(directory, baseline.resolve())
        result = score_validated_runs(directory, manifest, runs, args.jar)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
