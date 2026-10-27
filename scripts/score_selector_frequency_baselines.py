#!/usr/bin/env python3
"""Score four untrained ranking rules on the exact selector validation pools."""
from collections import Counter
from fractions import Fraction
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_answer_selector_data as data
from scripts import answer_selector_core as core
from scripts.audit_generator_coverage_dpo import frozen_audit

METHODS = ("round_robin", "model_vote", "rank_vote", "sample_frequency")
POOLS = ("max5_greedy", "max5_sampling4")


def validate_support(view, response_audit):
    """Reconstruct every source rank and per-draw frequency, including failures."""
    qid, pool = view["question_id"], view["pool"]
    draws = [0] if pool == "max5_greedy" else list(range(1, 5))
    requested = {(r["model_key"], r["draw"]): r for r in response_audit if r["question_id"] == qid and r["draw"] in draws}
    if len(requested) != 3 * len(draws) or set(requested) != {(m, d) for m in data.MODELS for d in draws}:
        raise ValueError("Incomplete requested-draw denominator")
    candidates = view["candidates"]
    if len({c["answer"] for c in candidates}) != len(candidates) or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate exact surface or source ID")
    first = {m: {} for m in data.MODELS}
    positions = set()
    for c in candidates:
        support = {m: set() for m in data.MODELS}
        for occurrence in c["occurrences"]:
            m, d = occurrence["model"], occurrence["draw"]
            r = requested[m, d]
            pos = occurrence["position"]
            if (r["response_id"] != occurrence["response_id"] or pos not in r["retained_positions"]
                    or (m, d, pos) in positions):
                raise ValueError("Wrong or duplicate occurrence")
            positions.add((m, d, pos))
            support[m].add(d)
            first[m][c["id"]] = min(first[m].get(c["id"], (d, pos)), (d, pos))
        counts = {m: len(dd) for m, dd in support.items() if dd}
        if (c["draw_support"] != counts or set(c["sources"]) != set(counts)
                or c["sample_frequencies"] != {m: n / len(draws) for m, n in counts.items()}):
            raise ValueError("Draw/model support counters differ")
    expected_positions = {(m, d, pos) for (m, d), r in requested.items() for pos in r["retained_positions"]}
    if positions != expected_positions:
        raise ValueError("Candidate pool lost response surfaces")
    ranks = {m: {identity: i for i, identity in enumerate(sorted(first[m], key=first[m].get), 1)} for m in data.MODELS}
    for c in candidates:
        if c["source_ranks"] != {m: ranks[m][c["id"]] for m in data.MODELS if c["id"] in ranks[m]}:
            raise ValueError("Source-order ranks differ")


def order(candidates, method):
    if method not in METHODS:
        raise ValueError("Unknown fixed ranking rule")
    # Same source-priority tie policy as historical heuristics. Exact surfaces
    # remain distinct; never reuse the historical normalized-string deduplicator.
    def tie(c):
        return min((data.MODELS.index(m), r) for m, r in c["source_ranks"].items()), c["answer"]
    if method == "round_robin":
        queues = {m: sorted([c for c in candidates if m in c["source_ranks"]], key=lambda c: c["source_ranks"][m]) for m in data.MODELS}
        result, seen = [], set()
        for i in range(max((len(q) for q in queues.values()), default=0)):
            for model in data.MODELS:
                if i < len(queues[model]):
                    c = queues[model][i]
                    if c["id"] not in seen:
                        result.append(c)
                        seen.add(c["id"])
        return result
    score = {
        "model_vote": lambda c: len(c["sources"]),
        "rank_vote": lambda c: sum((Fraction(1, r) for r in c["source_ranks"].values()), Fraction()),
        "sample_frequency": lambda c: sum(c["draw_support"].values()),
    }[method]
    return sorted(candidates, key=lambda c: (-score(c), tie(c)))


def evaluate(labels):
    conditions, per_question, predictions = {}, {}, {}
    for method in METHODS:
        predicted = []
        for row in labels:
            full = order(row["candidates"], method)
            if {c["id"] for c in full} != {c["id"] for c in row["candidates"]} or len(full) != len(row["candidates"]):
                raise ValueError("Ranking altered source pool")
            predicted.append({"question_id": row["question_id"], "pool": row["pool"], "selected_source_ids": [c["id"] for c in full[:5]]})
        summary, per_question[method] = core.score_predictions(predicted, labels)
        conditions[method] = summary["conditions"]
        predictions[method] = predicted
    return conditions, per_question, predictions


def run(audit, output):
    original = frozen_audit(audit)
    if output.exists():
        raise FileExistsError(output)
    inputs = data.records(audit / "validation_inputs.jsonl")
    labels = data.records(audit / "validation_supervision.jsonl")
    response_audit = [r for r in data.records(audit / "response_audit.jsonl") if r["split"] == "validation"]
    fitting_ids = {r["question_id"] for r in data.records(audit / "train_inputs.jsonl")}
    heldout = data.read(audit / "heldout_ids.json")
    expected = {(r["question_id"], r["pool"]) for r in inputs}
    if (len(inputs) != 2 * data.COUNTS["validation"] or len(expected) != len(inputs) or len(labels) != len(inputs)
            or {(r["question_id"], r["pool"]) for r in labels} != expected
            or {r["pool"] for r in labels} != set(POOLS)
            or any(r["split"] != "validation" for r in labels + inputs)
            or len(response_audit) != 15 * data.COUNTS["validation"]):
        raise ValueError("Wrong validation panel")
    by_input = {(r["question_id"], r["pool"]): r for r in inputs}
    panels = [{r["question_id"] for r in labels if r["pool"] == p} for p in POOLS]
    if panels[0] != panels[1] or panels[0] & (fitting_ids | set(heldout["dev160"] + heldout["test95"])):
        raise ValueError("Wrong or overlapping question membership")
    for row in labels:
        payload = json.loads(by_input[row["question_id"], row["pool"]]["messages"][1]["content"])
        if payload["candidates"] != [{"id": c["id"], "answer": c["answer"]} for c in row["candidates"]]:
            raise ValueError("Input and scoring surfaces differ")
        validate_support(row, response_audit)
    conditions, scores, predictions = evaluate(labels)
    frozen_audit(audit)
    result = {"question_count": data.COUNTS["validation"], "split": "validation", "conditions": conditions,
        "source_audit_manifest_sha256": data.digest(audit / "manifest.json"), "source_audit": str(audit),
        "validation_inputs_sha256": original["output_sha256"]["validation_inputs.jsonl"],
        "validation_supervision_sha256": original["output_sha256"]["validation_supervision.jsonl"],
        "policy": {"pool": "unchanged exact surfaces; no normalized deduplication", "ties": "fixed model/source-rank priority, then exact answer",
            "model_vote": "number of generators supporting candidate, not selector-list votes",
            "rank_vote": "sum(1/source unique-candidate rank), not selector-list rank fusion",
            "sample_frequency": "sum of distinct supporting draws; same draw budget per generator; failures stay in denominator"},
        "training": False, "new_generation": False, "paid_judging": False, "test_evaluation": False}
    output.mkdir(parents=True)
    data.write(output / "comparison_summary.json", result)
    data.write(output / "comparison_per_question.json", scores)
    for method, rows in predictions.items():
        data.jsonl(output / f"{method}_predictions.jsonl", rows)
    lines = ["# Frequency baselines on the selector validation pools", "", "Same 144 validation questions, candidate surfaces and official labels. No training or generation.", "",
        "| Method | Greedy MRR@5 | Greedy top-five hits | Sampling4 MRR@5 | Sampling4 top-five hits |", "|---|---:|---:|---:|---:|"]
    for name, pp in conditions.items():
        a, b = (pp[p] for p in POOLS)
        n = result["question_count"]
        lines.append(f"| {name} | {a['mrr_at5']:.6f} | {round(a['coverage_at5']*n)}/{n} | {b['mrr_at5']:.6f} | {round(b['coverage_at5']*n)}/{n} |")
    lines += ["", "Source full-pool coverage remains " + ", ".join(f"{p}: {round(conditions['round_robin'][p]['coverage_full_pool']*result['question_count'])}/{result['question_count']}" for p in POOLS) + ".",
        "These rules vote over generator candidates. They differ from voting over selectors' top-five lists.",
        "All exact surfaces, including case/spacing variants, remain separate. Historical test95 scores used a different pooling policy and are not copied here.",
        "Ties follow fixed Llama/Qwen/Ministral source priority, never accepted labels. No tie parameters are tuned on validation.",
        "No statistical superiority claim is made. Same-pool logistic/selector predictions are needed for paired comparisons against those systems.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    data.write(output / "manifest.json", {"status": "complete", "source_audit_manifest_sha256": result["source_audit_manifest_sha256"],
        "training": False, "artifacts_sha256": {p.name: data.digest(p) for p in output.iterdir() if p.is_file()}})
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.audit.resolve(), args.output.resolve())
