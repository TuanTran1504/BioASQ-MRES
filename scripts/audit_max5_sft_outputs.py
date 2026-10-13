#!/usr/bin/env python3
"""Diagnose saved max-five outputs; official nonmatches are not semantic errors."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def indexed(values):
    result = {v["question_id"]: v for v in values}
    if len(result) != len(values):
        raise ValueError("Duplicate question IDs")
    return result


def audit(directory):
    state = read(directory / "manifest.json")
    summary = read(directory / "comparison_summary.json")
    scores = read(directory / "comparison_per_question.json")
    if state["status"] != "complete" or state["expected_questions"] != 160:
        raise ValueError("Expected complete dev160 evaluation")
    counts, cases, panel = {}, [], None
    for name, relative in state["runs"].items():
        run = (directory / relative).resolve()
        if not run.is_relative_to(directory.resolve()):
            raise ValueError("Run path outside evaluation")
        for filename, expected in state["arm_hashes"][name].items():
            if hashlib.sha256((run / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed archive: {name}/{filename}")
        recorded = summary["conditions"][name]["official_report"].replace("\\", "/").rstrip("/").split("/")[-1]
        report = run / recorded / "per_question.jsonl"
        if not report.exists():
            raise ValueError(f"Missing recorded official report: {report}")
        official, examples = indexed(rows(report)), indexed(rows(run / "examples.jsonl"))
        scored, generations = indexed(scores[name]), indexed(rows(run / "generations.jsonl"))
        if len(official) != 160 or any(set(v) != set(official) for v in (examples, scored, generations)):
            raise ValueError("Question panels differ")
        current = {q: (v["question"], v["gold_aliases"], v["snippets"]) for q, v in examples.items()}
        if panel is not None and current != panel:
            raise ValueError("Evidence/questions differ between arms")
        panel = current
        c = Counter()
        ranks = Counter()
        candidate_total = 0
        for q, r in official.items():
            answers = r["answers"]
            accepted = set(r["matching_answers"])
            rank = next((i for i, a in enumerate(answers, 1) if a in accepted), None)
            expected_mrr = 1 / rank if rank and rank <= 5 else 0
            if abs(scored[q]["mrr_at5"] - expected_mrr) > 1e-10:
                raise ValueError("Scores differ from archived matching answers")
            if bool(scored[q]["coverage_full_pool"]) != bool(accepted):
                raise ValueError("Coverage differs from archived matching answers")
            category = ("parse_failure" if not r["parse_success"] else
                        "accepted_first" if rank == 1 else
                        "accepted_ranks_2_5" if rank and rank <= 5 else
                        "accepted_after_5" if rank else "no_accepted_answer")
            c[category] += 1
            if rank:
                ranks[str(rank)] += 1
            candidate_total += len(answers)
            c["questions_with_unaccepted_candidates"] += any(a not in accepted for a in answers)
            c["questions_with_accepted_and_unaccepted_candidates"] += bool(accepted) and any(a not in accepted for a in answers)
            if "greedy" in name and len(answers) > 5:
                raise ValueError("Greedy response exceeds max-five budget")
            if name.endswith("greedy"):
                cases.append({"model": state["model_key"], "condition": name, "category": category,
                              "question_id": q, "question": r["question"], "gold_aliases": r["gold_aliases"],
                              "answers": answers, "matching_answers": r["matching_answers"],
                              "first_accepted_rank": rank, "snippets": examples[q]["snippets"],
                              "raw_samples": generations[q]["samples"],
                              "semantic_label": "unreviewed; official nonmatches may be correct"})
        counts[name] = {**dict(c), "first_accepted_rank_counts": dict(ranks),
                       "candidate_count": candidate_total, "metrics": summary["conditions"][name]["metrics"]}
    for criterion in ("best_eval_loss", "best_coverage"):
        greedy = indexed([r for r in cases if r["condition"] == criterion + "_greedy"])
        sampled = indexed(scores[criterion + "_sampling10"])
        counts[criterion + "_greedy"]["greedy_misses_recovered_by_sampling_full_pool"] = sum(
            r["first_accepted_rank"] is None and bool(sampled[q]["coverage_full_pool"])
            for q, r in greedy.items())
    return {"model": state["model_key"], "question_count": 160,
            "selected_epochs": summary["selected_epochs"], "conditions": counts,
            "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summaries, cases = [], []
    for path in sorted(args.root.rglob("comparison_summary.json")):
        result, items = audit(path.parent)
        summaries.append(result)
        cases.extend(items)
    if len(summaries) != 3 or len({r["model"] for r in summaries}) != 3:
        raise ValueError("Expected exactly three model evaluations")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    for r in summaries:
        print(r["model"], json.dumps(r["conditions"]["best_eval_loss_greedy"]))


if __name__ == "__main__":
    main()
