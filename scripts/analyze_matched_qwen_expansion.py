#!/usr/bin/env python3
"""Score a complete paired Gadi base/SFT experiment with the BioASQ matcher."""

import argparse
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ranking_metrics(rows):
    """Score the first five candidates in their generated order, retaining failures."""
    if not rows:
        raise ValueError("No evaluation questions")
    output = []
    for row in rows:
        accepted = set(row["matching_answers"])
        rank = next((i for i, answer in enumerate(row["answers"], 1) if answer in accepted), None)
        output.append({"question_id": row["question_id"],
                       "mrr_at5": 1.0 / rank if rank and rank <= 5 else 0.0,
                       "strict_accuracy": float(rank == 1),
                       "lenient_accuracy": float(rank is not None and rank <= 5),
                       **{f"coverage_at{k}": float(rank is not None and rank <= k) for k in (1, 5, 10)}})
    return {key: sum(r[key] for r in output) / len(output) for key in output[0] if key != "question_id"}, output


def paired_contrast(base, sft, resamples=10000):
    by_id = {row["question_id"]: row for row in base}
    if len(by_id) != len(base) or len({r["question_id"] for r in sft}) != len(sft) or set(by_id) != {r["question_id"] for r in sft}:
        raise ValueError("Paired results have different or duplicate questions")
    result = {}
    for metric in ("mrr_at5", "coverage_at1", "coverage_at5", "coverage_at10"):
        differences = [r[metric] - by_id[r["question_id"]][metric] for r in sft]
        rng = random.Random(3407)
        samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(resamples))
        result[metric] = {"sft_minus_base": sum(differences) / len(differences),
                          "paired_bootstrap_95ci": [samples[int(0.025 * resamples)], samples[min(int(0.975 * resamples), resamples - 1)]],
                          "resamples": resamples, "seed": 3407}
    return result


def validate_pair(run_dirs, expected):
    configs = [json.loads((path / "config.json").read_text()) for path in run_dirs]
    keys = ("input_sha256", "prompt_sha256", "max_seq_length", "max_new_tokens", "temperature",
            "require_all_snippets", "prompt_version", "response_mode", "chat_template_kwargs", "question_count")
    if any(configs[0].get(k) != configs[1].get(k) for k in keys):
        raise ValueError("Base and SFT evaluation settings differ")
    examples = [[json.loads(line) for line in (path / "examples.jsonl").read_text().splitlines() if line.strip()] for path in run_dirs]
    if examples[0] != examples[1] or len(examples[0]) != expected:
        raise ValueError("Base and SFT must use identical questions, evidence and gold labels")
    ids = {row["question_id"] for row in examples[0]}
    if len(ids) != expected:
        raise ValueError("Duplicate evaluation IDs")
    for path in run_dirs:
        state = json.loads((path / "status.json").read_text())
        generations = [json.loads(line) for line in (path / "generations.jsonl").read_text().splitlines() if line.strip()]
        if state["status"] != "complete" or len(generations) != expected or {r["question_id"] for r in generations} != ids:
            raise ValueError("Incomplete generation; failed parses must remain in the denominator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    manifest = json.loads((args.experiment_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["smoke_test"] or manifest["expected_questions"] != 160:
        raise ValueError("Expected a complete full 160-question experiment")
    run_dirs = [(args.experiment_dir / manifest["runs"][condition]).resolve() for condition in ("base", "sft")]
    validate_pair(run_dirs, 160)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_size": manifest["model_size"], "question_count": 160,
              "selection": manifest["selection"], "scoring": "official BioASQ Java candidate matcher",
              "evaluation": "development-only matched generator comparison", "conditions": {}}
    scored = {}
    for condition, path in zip(("base", "sft"), run_dirs):
        report, summary, rows = analyze_local_expansion(path, jar_path=args.jar)
        metrics, scored[condition] = ranking_metrics(rows)
        result["conditions"][condition] = {"metrics": metrics, "candidate_diagnostics": summary,
                                            "report": report.relative_to(args.experiment_dir.resolve()).as_posix()}
    result["paired_contrasts"] = paired_contrast(scored["base"], scored["sft"])
    (args.experiment_dir / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.experiment_dir / "comparison_per_question.json").write_text(json.dumps(scored, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
