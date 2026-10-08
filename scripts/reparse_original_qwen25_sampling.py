#!/usr/bin/env python3
"""Reparse saved historical SFT samples, preserving original files and generation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
_original_import_path = sys.path[:]
try:
    sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
    from run_single_answer_sampling import PARSER_VERSION, parse_single_answer, unique_candidates
finally:
    sys.path[:] = _original_import_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reparse_samples(rows):
    output, candidates = [], []
    for original in rows:
        row = {**original, "samples": []}
        for original_sample in original["samples"]:
            sample = {**original_sample, "previous_answer": original_sample.get("answer"),
                      "previous_parse_error": original_sample.get("parse_error"),
                      "parser_version": PARSER_VERSION}
            try:
                sample.update(answer=parse_single_answer(sample["raw_response"]), parse_error=None)
            except ValueError as exc:
                sample.update(answer=None, parse_error=str(exc))
            row["samples"].append(sample)
        unique = unique_candidates(row["samples"])
        candidates.extend({"question_id": row["question_id"], **candidate} for candidate in unique)
        row.update(previous_parse_error=original.get("parse_error"), parser_version=PARSER_VERSION,
                   parse_error=None if unique else "All ten samples failed")
        output.append(row)
    return output, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    original = args.experiment_dir.resolve()
    manifest = read(original / "manifest.json")
    if manifest["status"] != "complete" or manifest["smoke_test"] or manifest["expected_questions"] != 160:
        raise ValueError("Expected a completed 160-question original SFT experiment")
    source = original / manifest["runs"]["original_sft_sampling10"]
    rows = records(source / "generations.jsonl")
    examples = records(source / "examples.jsonl")
    ids = {row["question_id"] for row in examples}
    if (read(source / "status.json")["status"] != "complete" or len(rows) != 160
            or len(ids) != 160 or {row["question_id"] for row in rows} != ids):
        raise ValueError("Incomplete sampling run")
    if any(len(row["samples"]) != 10 or [s["draw"] for s in row["samples"]] != list(range(1, 11)) for row in rows):
        raise ValueError("Every question must retain all ten attempted draws")
    output = (args.output_dir or original.with_name(original.name + "-sampling-parser-v2")).resolve()
    generations, candidates = reparse_samples(rows)
    sampling = output / "sampling10"
    output.mkdir(parents=True, exist_ok=False)
    sampling.mkdir()
    shutil.copy2(source / "examples.jsonl", sampling / "examples.jsonl")
    config = {**read(source / "config.json"), "parser_version": PARSER_VERSION,
              "reparsed_from": str(source), "generation_unchanged": True,
              "source_generations_sha256": hashlib.sha256((source / "generations.jsonl").read_bytes()).hexdigest()}
    write(sampling / "config.json", config)
    for name, values in (("generations", generations), ("candidates", candidates)):
        (sampling / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    write(sampling / "status.json", {"status": "complete", "expected_questions": 160,
                                     "completed_questions": 160, "generation_unchanged": True,
                                     "parser_version": PARSER_VERSION})
    expansion = (original / manifest["runs"]["original_sft_expansion"]).resolve()
    revised = {**manifest, "generation_unchanged": True, "reparsed_from": str(original),
               "parser_version": PARSER_VERSION, "runs": {
                   "original_sft_expansion": Path(os.path.relpath(expansion, output)).as_posix(),
                   "original_sft_sampling10": "sampling10"}}
    write(output / "manifest.json", revised)
    draws = [sample for row in generations for sample in row["samples"]]
    print(json.dumps({"experiment_dir": str(output), "parser_version": PARSER_VERSION,
                      "generation_unchanged": True, "attempted_draws": len(draws),
                      "parse_successful_draws": sum(not sample["parse_error"] for sample in draws),
                      "questions_with_answers": sum(not row["parse_error"] for row in generations)}, indent=2))


if __name__ == "__main__":
    main()
