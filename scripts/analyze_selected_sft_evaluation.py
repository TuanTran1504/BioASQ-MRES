#!/usr/bin/env python3
"""Compare frozen coverage/loss-selected SFT checkpoints on dev160."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
from prepare_matched_8b_sft import read, records, write, digest
from run_selected_sft_evaluation import CRITERIA, MODES
from run_single_answer_sampling import sample_seed, expansion_candidates
from run_extractive_expansion_8b import parse_equivalent_response
from scripts.analyze_matched_qwen_expansion import validate_pair, ranking_metrics, paired_contrast
from scripts.analyze_original_qwen25_inference import efficiency


def validate_experiment(directory):
    state = read(directory / "manifest.json")
    names = {f"{criterion}_{mode}" for criterion in CRITERIA for mode in MODES}
    if state["status"] != "complete" or state["expected_questions"] != 160 or set(state["runs"]) != names:
        raise ValueError("Expected complete four-arm dev160 evaluation")
    if digest(directory / "expansion_prompt.txt") != state["prompt_sha256"]:
        raise ValueError("Archived prompt changed")
    runs, examples = {}, None
    for name, relative in state["runs"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Inference escaped evaluation directory")
        validate_pair([path, path], 160)
        for filename, expected in state["arm_hashes"][name].items():
            if digest(path / filename) != expected:
                raise ValueError("Archived inference changed")
        panel = records(path / "examples.jsonl")
        if examples is not None and examples != panel:
            raise ValueError("Checkpoint evaluation questions/evidence differ")
        examples = panel
        canonical = state["reused_from"].get(name, name)
        criterion, mode = canonical.rsplit("_", 1)
        if canonical != name:
            own = name.rsplit("_", 1)[0]
            if (canonical not in names or state["runs"][canonical] != relative
                    or state["sources"][own]["generation_files_sha256"] != state["sources"][criterion]["generation_files_sha256"]):
                raise ValueError("Reused generations are not the identical checkpoint/tokenizer")
        config = read(path / "config.json")
        count = 1 if mode == "greedy" else 10
        expected = {**state["common_settings"], "model_name": state["sources"][criterion]["adapter"],
            "response_mode": "equivalent_greedy" if count == 1 else "equivalent_sampling",
            "temperature": 0 if count == 1 else .8, "num_generations": count, "question_count": 160,
            "input_sha256": state["input_sha256"], "prompt_sha256": state["prompt_sha256"],
            "gold_blind_generation": True, "local_files_only": True}
        if count == 10:
            expected.update(top_p=.95, top_k=0)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint evaluation settings differ")
        candidates, rejected = [], []
        for row in records(path / "generations.jsonl"):
            if (row["request_count"] != count or len(row["samples"]) != count or row["snippets_truncated"]
                    or row["included_snippets"] != row["total_snippets"]
                    or row["prompt_tokens"] != state["input_plan"]["expected_prompt_tokens"][row["question_id"]]):
                raise ValueError("Incomplete evidence or sample budget")
            for draw, sample in enumerate(row["samples"], 1):
                if sample["draw"] != draw or sample["seed"] != sample_seed(row["question_id"], draw, config["seed"]):
                    raise ValueError("Draw order or seed changed")
                try:
                    answers, invalid, _, _ = parse_equivalent_response(sample["raw_response"], config["max_candidates"])
                except ValueError:
                    answers, invalid = [], []
                if sample["answers"] != answers:
                    raise ValueError("Candidates differ from raw parsing")
                rejected.extend({"question_id": row["question_id"], "draw": draw, **r} for r in invalid)
            candidates.extend({"question_id": row["question_id"], **r} for r in expansion_candidates(row["samples"]))
        if records(path / "candidates.jsonl") != candidates or records(path / "invalid_candidates.jsonl") != rejected:
            raise ValueError("Archived pools differ from raw parsing")
        runs[name] = path
    return state, runs


def analyze(directory, jar):
    state, runs = validate_experiment(directory)
    from src.notebook_workflows.local_expansion import analyze_local_expansion
    result = {"model_key": state["model_key"], "question_count": 160, "conditions": {}, "paired_contrasts": {},
        "evaluation": state["evaluation"], "quality_warnings": [], "reused_from": state["reused_from"],
        "selected_epochs": {k: source["epoch"] for k, source in state["sources"].items()},
        "interpretation": "Development scores evaluate previously frozen checkpoints; they do not select a new checkpoint. One training seed; paired intervals measure question-sampling uncertainty."}
    scored, cached = {}, {}
    for name, run in runs.items():
        if run not in cached:
            report, diagnostics, rows = analyze_local_expansion(run, jar_path=jar)
            metrics, scores = ranking_metrics(rows)
            for raw, row in zip(rows, scores):
                row["coverage_full_pool"] = float(bool(raw["matching_answers"]))
            metrics["coverage_full_pool"] = sum(r["coverage_full_pool"] for r in scores) / 160
            cached[run] = ({"metrics": metrics, "candidate_diagnostics": diagnostics,
                            "efficiency": efficiency(run), "official_report": str(report)}, scores)
        result["conditions"][name], scored[name] = cached[run]
        if result["conditions"][name]["candidate_diagnostics"]["parse_success_count"] < 160:
            result["quality_warnings"].append(f"{name}: parse failures retained in denominator")
        m = result["conditions"][name]["metrics"]
        print(f"{state['model_key']} {name}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, "
              f"coverage@10={m['coverage_at10']:.4f}, full pool={m['coverage_full_pool']:.4f}", flush=True)
    for mode in MODES:
        contrast = paired_contrast(scored[f"best_eval_loss_{mode}"], scored[f"best_coverage_{mode}"],
                                  metrics=("mrr_at5", "coverage_at5", "coverage_at10", "coverage_full_pool"))
        result["paired_contrasts"][f"coverage_selected_minus_loss_selected_{mode}"] = {
            k: {"coverage_selected_minus_loss_selected": v["sft_minus_base"],
                **{field: value for field, value in v.items() if field != "sft_minus_base"}}
            for k, v in contrast.items()}
    write(directory / "comparison_summary.json", result)
    write(directory / "comparison_per_question.json", scored)
    print(f"Saved: {directory / 'comparison_summary.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", type=Path, nargs="+")
    parser.add_argument("--jar", type=Path, default=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    args = parser.parse_args()
    for directory in args.experiment_dirs:
        analyze(directory.resolve(), args.jar)


if __name__ == "__main__":
    main()
