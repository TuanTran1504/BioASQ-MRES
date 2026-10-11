#!/usr/bin/env python3
"""Export changed outcomes and sampled selection misses from existing official scores.

No generation, Java scoring, paid judging or training is performed. Gold labels
are used only to select offline diagnostic cases, never inference candidates.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_expansion_dpo_evaluation import CONDITIONS, digest, validate_experiment
from scripts.analyze_matched_qwen_expansion import ranking_metrics
from scripts.analyze_original_qwen25_inference import read, records


def by_question(rows):
    result = {r["question_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate scored question")
    return result


def load_scored(directory):
    manifest, runs = validate_experiment(directory)
    summary = read(directory / "comparison_summary.json")
    scored = read(directory / "comparison_per_question.json")
    if summary["model_key"] != manifest["model_key"] or set(scored) != set(CONDITIONS):
        raise ValueError("Scored comparison differs from the evaluated model/arms")
    examples = by_question(records(runs["sft_greedy"] / "examples.jsonl"))
    arms, hashes = {}, {}
    for name, path in runs.items():
        # Resolve the recorded analysis basename inside its own run. This works
        # after copying an experiment to another machine and never follows an
        # old absolute host path or chooses a different, more recent analysis.
        basename = str(summary["conditions"][name]["report"]).replace("\\", "/").rstrip("/").split("/")[-1]
        if not basename.startswith("analysis-") or basename in (".", ".."):
            raise ValueError("Unexpected analysis report name")
        report = path / basename / "per_question.jsonl"
        rows = by_question(records(report))
        scores = by_question(scored[name])
        generations = by_question(records(path / "generations.jsonl"))
        if set(rows) != set(examples) or set(scores) != set(examples) or set(generations) != set(examples):
            raise ValueError("Report question membership differs")
        if any(rows[q][k] != examples[q][k] for q in examples for k in ("question", "gold_aliases")):
            raise ValueError("Scoring report question or gold labels differ from saved evaluation data")
        candidates = defaultdict(list)
        for candidate in records(path / "candidates.jsonl"):
            candidates[candidate["question_id"]].append(candidate)
        _, recomputed = ranking_metrics(list(rows.values()))
        for values in recomputed:
            qid = values["question_id"]
            values["coverage_full_pool"] = float(bool(rows[qid]["matching_answers"]))
            if any(abs(scores[qid][k] - v) > 1e-12 for k, v in values.items() if k != "question_id"):
                raise ValueError("Saved metrics differ from archived official candidate matches")
            ordered = sorted(candidates[qid], key=lambda c: c["position"])
            if rows[qid]["answers"] != [c["answer"] for c in ordered]:
                raise ValueError("Scoring report candidates differ from the inference pool")
            if not set(rows[qid]["matching_answers"]) <= set(rows[qid]["answers"]):
                raise ValueError("Accepted answer absent from candidate pool")
            accepted = set(rows[qid]["matching_answers"])
            candidates[qid] = [{**c, "rank": i, "official_accepted": c["answer"] in accepted}
                               for i, c in enumerate(ordered, 1)]
        arms[name] = {"scores": scores, "reports": rows, "candidates": candidates,
                      "generations": generations}
        hashes[name] = {"official_report": digest(report),
                       **{f: digest(path / f) for f in ("examples.jsonl", "generations.jsonl", "candidates.jsonl", "config.json")}}
    return manifest, examples, arms, hashes


def diagnostics(arms):
    result = {"mrr_changes": {}, "sampled_selection_misses": {}}
    for mode in ("greedy", "sampling10"):
        sft, dpo = arms["sft_" + mode], arms["dpo_" + mode]
        differences = [dpo["scores"][q]["mrr_at5"] - s["mrr_at5"] for q, s in sft["scores"].items()]
        def completions(g):
            return [s["raw_response"] for s in g["samples"]] if "samples" in g else [g["raw_response"]]
        result["mrr_changes"][mode] = {
            "improved": sum(d > 1e-12 for d in differences),
            "worsened": sum(d < -1e-12 for d in differences),
            "unchanged": sum(abs(d) <= 1e-12 for d in differences),
            "raw_completions_changed": sum(completions(sft["generations"][q]) != completions(dpo["generations"][q])
                                           for q in sft["scores"])}
    for policy in ("sft", "dpo"):
        rows = list(arms[policy + "_sampling10"]["scores"].values())
        result["sampled_selection_misses"][policy] = {
            "within_first_ten": sum(bool(r["coverage_at10"]) and not r["coverage_at5"] for r in rows),
            "anywhere_in_pool": sum(bool(r["coverage_full_pool"]) and not r["coverage_at5"] for r in rows)}
    return result


def choose_cases(arms, model, limit):
    reasons = defaultdict(set)
    for mode in ("greedy", "sampling10"):
        for qid, before in arms["sft_" + mode]["scores"].items():
            delta = arms["dpo_" + mode]["scores"][qid]["mrr_at5"] - before["mrr_at5"]
            if abs(delta) > 1e-12:
                reasons[qid].add(mode + ("_mrr_improved" if delta > 0 else "_mrr_worsened"))
    for policy in ("sft", "dpo"):
        strata = defaultdict(list)
        for qid, row in arms[policy + "_sampling10"]["scores"].items():
            if row["coverage_full_pool"] and not row["coverage_at5"]:
                strata["accepted_at_rank_6_to_10" if row["coverage_at10"] else "accepted_beyond_rank_10"].append(qid)
        for stratum, qids in strata.items():
            order = sorted(qids, key=lambda q: hashlib.sha256(f"3407:{model}:{policy}:{stratum}:{q}".encode()).hexdigest())
            for qid in order[:limit]:
                reasons[qid].add(policy + "_sampling10_" + stratum)
    return reasons


def export_cases(directory, limit=2):
    directory = directory.resolve()
    manifest, examples, arms, hashes = load_scored(directory)
    selected = choose_cases(arms, manifest["model_key"], limit)
    cases = []
    for qid in sorted(selected):
        case = {"model_key": manifest["model_key"], "question_id": qid,
                "case_reasons": sorted(selected[qid]), **{k: examples[qid][k] for k in ("question", "snippets", "gold_aliases")},
                "arms": {}}
        for name, arm in arms.items():
            accepted = [c["rank"] for c in arm["candidates"][qid] if c["official_accepted"]]
            case["arms"][name] = {"metrics": arm["scores"][qid], "first_accepted_rank": min(accepted) if accepted else None,
                                 "candidates": arm["candidates"][qid], "generation": arm["generations"][qid]}
        cases.append(case)
    summary = {"model_key": manifest["model_key"], "question_count": len(examples), **diagnostics(arms),
               "exported_cases": len(cases), "selection_miss_examples_per_stratum": limit,
               "sampling": "All MRR changes plus deterministic seed-3407 samples per policy and selection-miss stratum; cases merged by question ID",
               "interpretation": "Outcome-selected diagnostic cases, not a representative semantic-quality audit; official acceptance is not a C1/C2 semantic label"}
    output = directory / "case_audit"
    output.mkdir(exist_ok=False)
    (output / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    for name, value in (("audit_summary.json", summary), ("source_hashes.json", {"arms": hashes,
            "comparison_summary": digest(directory / "comparison_summary.json"),
            "comparison_per_question": digest(directory / "comparison_per_question.json"),
            "evaluation_manifest": digest(directory / "manifest.json")})):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--selection-examples", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.selection_examples <= 10:
        parser.error("Use 0 to 10 examples per selection-miss stratum")
    # Validate all inputs and output destinations before creating any exports.
    for directory in args.experiment_dirs:
        load_scored(directory.resolve())
        if (directory / "case_audit").exists():
            raise FileExistsError(directory / "case_audit")
    for directory in args.experiment_dirs:
        output, summary = export_cases(directory, args.selection_examples)
        print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
