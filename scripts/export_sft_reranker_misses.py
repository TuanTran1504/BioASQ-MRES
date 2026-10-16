#!/usr/bin/env python3
"""Export all covered-but-outside-top-five cases from frozen SFT rerankers.

Uses existing official labels and frozen weights. No fitting, generation,
Java matching, paid judging or DPO preference construction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.sparse import hstack

from scripts import evaluate_sft_ensemble_rerankers as ev
from scripts.train_candidate_reranker_pilot import candidate_features, make_text


def check_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or ev.digest(path) != expected:
            raise ValueError(f"Changed or unsafe source file: {name}")


def validate_rankings(rows, pools, examples, summary, saved_scores):
    """Verify all 28 rankings against unchanged candidates and per-question scores."""
    names = {f"{p}_{s}_reranker": p for p in pools for s in ev.SELECTORS}
    names.update({f"{p}_{m}": p for p in pools for m in ev.METHODS})
    if set(summary["conditions"]) != set(names) or set(saved_scores) != set(names):
        raise ValueError("Expected all 28 frozen test conditions")
    groups = {n: {q: [] for q in examples} for n in names}
    lookup = {p: {(r["question_id"], r["answer"]): r for r in rr} for p, rr in pools.items()}
    for r in rows:
        name, qid = r["condition"], r["question_id"]
        if name not in names or qid not in examples:
            raise ValueError("Unknown ranking condition or question")
        candidate = lookup[names[name]].get((qid, r["answer"]))
        if candidate is None or any(r.get(k) != v for k, v in candidate.items()):
            raise ValueError("Ranked candidate metadata or official label changed")
        if type(r["rank"]) is not int or r["rank"] < 1:
            raise ValueError("Invalid candidate rank")
        if name.endswith("_reranker") and not math.isfinite(r["score"]):
            raise ValueError("Non-finite stored logistic score")
        groups[name][qid].append(r)
    for name, qq in groups.items():
        for q, rr in qq.items():
            rr.sort(key=lambda r: r["rank"])
            expected = {a for (qid, a) in lookup[names[name]] if qid == q}
            if ([r["rank"] for r in rr] != list(range(1, len(rr) + 1))
                    or len(rr) != len(expected) or {r["answer"] for r in rr} != expected):
                raise ValueError("Missing or duplicate ranked candidates")
            if name.endswith("_reranker") and rr != sorted(rr, key=lambda r: (-r["score"], ev.tie_key(r))):
                raise ValueError("Stored ranks disagree with logistic score ordering")
        metrics, scores = ev.evaluate_rankings(qq)
        if len(saved_scores[name]) != len(examples):
            raise ValueError("Missing saved per-question scores")
        old = {r["question_id"]: r for r in saved_scores[name]}
        if len(old) != len(examples) or set(old) != set(examples):
            raise ValueError("Duplicate or changed scored membership")
        if any(abs(metrics[k] - summary["conditions"][name]["metrics"][k]) > 1e-12 for k in metrics):
            raise ValueError("Rankings do not reproduce summary metrics")
        if any(abs(r[k] - old[r["question_id"]][k]) > 1e-12
               for r in scores for k in r if k != "question_id"):
            raise ValueError("Rankings do not reproduce per-question metrics")
    return groups, names


def describe(rows):
    accepted = [r for r in rows if r["label"]]
    return {"pool_size": len(rows), "first_accepted_rank": accepted[0]["rank"] if accepted else None,
            "coverage_at5": bool(accepted and accepted[0]["rank"] <= 5),
            "coverage_full_pool": bool(accepted), "accepted_candidates": accepted,
            "top_five": rows[:5]}


def explain(components, row, example):
    """Exact additive logistic contributions, not a causal semantic diagnosis."""
    feature_row = {k: v for k, v in row.items() if k not in ("label", "condition", "rank", "score")}
    feature_row.update(surface_operations=[], is_format_variant=False)
    parts = (components["word"].transform([make_text(feature_row, example)]),
             components["char"].transform([feature_row["answer"]]),
             components["meta"].transform([candidate_features(feature_row, example)]))
    matrix = hstack(parts, format="csr")
    coef = components["model"].coef_[0]
    bias = float(components["model"].intercept_[0])
    score = bias + float((matrix @ coef)[0])
    if not math.isclose(score, row["score"], abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError("Frozen model no longer reproduces a diagnostic candidate score")
    names = (components["word"].get_feature_names_out(), components["char"].get_feature_names_out(),
             components["meta"].get_feature_names_out())
    totals, strongest, offset = {}, {}, 0
    for block, part, nn in zip(("question_answer_words", "answer_characters", "metadata"), parts, names):
        values = [{"feature": str(nn[i]), "value": float(value), "coefficient": float(coef[offset+i]),
                   "contribution": float(value * coef[offset+i])} for i, value in zip(part.indices, part.data)]
        totals[block] = sum(v["contribution"] for v in values)
        strongest[block] = sorted(values, key=lambda v: -abs(v["contribution"]))[:15]
        offset += part.shape[1]
    return {"score": score, "intercept": bias, "block_totals": totals, "largest_contributions": strongest,
            "input_metadata": candidate_features(feature_row, example)}


def collect_cases(groups, names, examples, gold, payloads):
    misses, counts = defaultdict(list), {}
    for name in groups:
        if not name.endswith("_reranker"):
            continue
        descriptions = {q: describe(rr) for q, rr in groups[name].items()}
        failed = [q for q, d in descriptions.items() if d["coverage_full_pool"] and not d["coverage_at5"]]
        counts[name] = {"questions": len(examples),
            "covered_in_pool": sum(d["coverage_full_pool"] for d in descriptions.values()),
            "covered_in_top5": sum(d["coverage_at5"] for d in descriptions.values()),
            "selection_misses": len(failed), "question_ids": sorted(failed)}
        for q in failed:
            misses[q].append(name)
    cases = []
    for q in sorted(misses):
        case = {**examples[q], "question_id": q, "gold_aliases": gold[q]["gold_aliases"], "batch": gold[q]["batch"],
                "missed_conditions": sorted(misses[q]), "conditions": {},
                "review": {"status": "unreviewed", "top5_semantically_valid_but_unaccepted": None,
                    "accepted_answer_evidence_support": None, "suspected_ranking_factors": [], "notes": ""}}
        for name, qq in groups.items():
            dd = describe(qq[q])
            if name in misses[q]:
                selector = name.removeprefix(names[name] + "_").removesuffix("_reranker")
                accepted, fifth = dd["accepted_candidates"][0], dd["top_five"][-1]
                c = payloads[selector]["components"]
                context = {"question": examples[q]["question"], "snippets": examples[q]["snippets"]}
                explanations = {r["answer"]: explain(c, r, context) for r in [*dd["top_five"], accepted]}
                dd["score_gap_fifth_minus_best_accepted"] = fifth["score"] - accepted["score"]
                dd["score_explanations"] = explanations
                dd["gap_by_feature_block"] = {b: explanations[fifth["answer"]]["block_totals"][b] -
                    explanations[accepted["answer"]]["block_totals"][b]
                    for b in explanations[fifth["answer"]]["block_totals"]}
            case["conditions"][name] = dd
        cases.append(case)
    return cases, counts


def export(evaluation, output, archive=None):
    if output.exists() or (archive is not None and archive.exists()):
        raise FileExistsError("Preserving an existing export or archive")
    state = ev.read(evaluation / "manifest.json")
    if state.get("status") != "complete" or state.get("question_count") != 95:
        raise ValueError("Need the complete frozen-reranker test95 evaluation")
    files = ("manifest.json", "comparison_summary.json", "comparison_per_question.json", "rankings.jsonl")
    input_hashes = {n: ev.digest(evaluation / n) for n in files}
    comparison, rerankers = Path(state["source_comparison"]), Path(state["reranker_directory"])
    check_hashes(comparison, state["source_file_hashes"])
    check_hashes(rerankers, state["reranker_file_hashes"])
    _, pools, examples, gold, _ = ev.load_comparison(comparison)
    _, payloads, _ = ev.load_frozen(rerankers, examples, comparison)
    summary = ev.read(evaluation / "comparison_summary.json")
    scores = ev.read(evaluation / "comparison_per_question.json")
    rankings = ev.records(evaluation / "rankings.jsonl")
    groups, names = validate_rankings(rankings, pools, examples, summary, scores)
    cases, counts = collect_cases(groups, names, examples, gold, payloads)
    selected = {c["question_id"] for c in cases}
    source_generations, raw_hashes = [], {}
    sources = ev.read(comparison / "manifest.json")["sources"]
    for source, metadata in sources.items():
        directory = Path(metadata["directory"])
        saved = ev.read(directory / "manifest.json")
        for arm, relative in saved["runs"].items():
            path = directory / relative / "generations.jsonl"
            raw_hashes[f"{source}/{arm}"] = {"path": str(path), "sha256": ev.digest(path)}
            source_generations.extend({"source": source, "arm": arm, **r} for r in ev.records(path)
                                      if r["question_id"] in selected)
    output.mkdir(parents=True, exist_ok=False)
    ev.write_jsonl(output / "cases.jsonl", cases)
    ev.write_jsonl(output / "source_generations.jsonl", source_generations)
    result = {"question_count": len(examples), "exported_unique_questions": len(cases), "conditions": counts,
        "priority_condition": "max5_sampling10_max5_reranker", "training_ready": False,
        "interpretation": "All official-acceptance selection misses from the 12 logistic conditions, merged by question. Outcome-selected test diagnostics; not representative semantic error rates or training examples.",
        "semantic_caution": "Official nonacceptance does not mean semantic incorrectness. Feature contributions explain this frozen linear score, not biomedical truth or causal effects."}
    ev.write(output / "summary.json", result)
    lines = ["# Accepted answers missed by frozen rerankers", "",
        "All 12 logistic conditions are included. Each case contains full evidence and accepted aliases, top-five competitors, accepted candidate ranks, frequencies, source ranks and relation tags. Raw saved responses are in source_generations.jsonl.", "",
        "The first review priority is max5_sampling10_max5_reranker. Score explanations decompose the exact frozen logistic score into word, character and metadata contributions. Treat these as model explanations, not semantic labels.", "",
        "| Condition | Covered in pool | Covered in top five | Missed |", "|---|---:|---:|---:|"]
    for n, d in counts.items():
        lines.append(f"| {n} | {d['covered_in_pool']} | {d['covered_in_top5']} | {d['selection_misses']} |")
    lines.extend(["", "## Priority cases", "", "| Question | First accepted rank | Accepted expression | Fifth-ranked competitor |", "|---|---:|---|---|"])
    priority = result["priority_condition"]
    for c in cases:
        if priority in c["missed_conditions"]:
            dd = c["conditions"][priority]
            cells = [c["question"], str(dd["first_accepted_rank"]), dd["accepted_candidates"][0]["answer"], dd["top_five"][-1]["answer"]]
            lines.append("| " + " | ".join(s.replace("|", "\\|").replace("\n", " ") for s in cells) + " |")
    lines.extend(["", "Review whether top-five answers are semantically correct unaccepted forms; whether accepted answers have evidence support; and whether frequency, source position, relation metadata or lexical features explain the score gap. Do not automatically call every top-five competitor wrong.", "",
        "These inspected test cases are for diagnosis. Construct future preference/training examples from fitting questions and select settings on validation, keeping test cases out of training.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    check_hashes(evaluation, input_hashes)
    check_hashes(comparison, state["source_file_hashes"])
    check_hashes(rerankers, state["reranker_file_hashes"])
    ev.write(output / "manifest.json", {"status": "complete", "training": False, "generation": False,
        "paid_judging": False, "evaluation": str(evaluation), "evaluation_sha256": input_hashes,
        "comparison_sha256": state["source_file_hashes"], "reranker_sha256": state["reranker_file_hashes"],
        "source_generation_sha256": raw_hashes, "export_code_sha256": ev.digest(Path(__file__)),
        "output_sha256": {p.name: ev.digest(p) for p in output.iterdir() if p.is_file()}})
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(output.iterdir()):
                handle.add(path, arcname=path.name)
    for name, d in counts.items():
        print(f"{name}: {d['covered_in_pool']} covered; {d['covered_in_top5']} top five; {d['selection_misses']} missed", flush=True)
    print(f"Exported {len(cases)} unique questions to {output}", flush=True)
    if archive is not None:
        print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    export(args.evaluation.resolve(), args.output.resolve(), args.archive.resolve() if args.archive else None)
