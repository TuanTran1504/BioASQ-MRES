#!/usr/bin/env python3
"""Export changed dev160 rankings and exact score/input differences; no new model calls."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))

# Cache the repository's evaluation package before portable inference helpers
# put their own starter/src directory ahead of it on sys.path.
import src

from scripts import analyze_generator_feedback_evaluation as ev
from scripts.export_sft_reranker_misses import explain
from scripts.analyze_sft_ensemble_test import order_pool
from scripts.compare_expansion_pool_rerankers import tie_key
from run_generator_feedback_evaluation import validate_prepared


def condition_pools():
    names = {}
    for policy in ("sft", "dpo"):
        for mode in ev.MODES:
            pool = f"{policy}_{mode}"
            for method in ("max5_reranker", "sample_frequency"):
                names[f"{pool}_{method}"] = pool
            names[f"llama_{pool}_generated_order"] = f"llama_{pool}"
    return names


def check_rankings(rows, pools, examples, summary, per_question):
    """All saved scores must be reproducible from the exported ordering and labels."""
    names = condition_pools()
    if set(summary["conditions"]) != set(names) or set(per_question) != set(names):
        raise ValueError("Expected all twelve completed comparison conditions")
    lookup = {p: {(r["question_id"], r["answer"]): r for r in rr} for p, rr in pools.items()}
    groups = {n: {q: [] for q in examples} for n in names}
    for row in rows:
        name, qid = row["condition"], row["question_id"]
        if name not in names or qid not in examples:
            raise ValueError("Unknown scored condition or question")
        original = lookup[names[name]].get((qid, row["answer"]))
        if original is None or any(row.get(k) != v for k, v in original.items()):
            raise ValueError("Ranked candidate metadata or official label differs")
        if type(row["rank"]) is not int or row["rank"] < 1:
            raise ValueError("Invalid rank")
        if name.endswith("_reranker") and not math.isfinite(row["score"]):
            raise ValueError("Invalid reranker score")
        groups[name][qid].append(row)
    for name, qq in groups.items():
        for qid, rr in qq.items():
            rr.sort(key=lambda r: r["rank"])
            expected = {a for q, a in lookup[names[name]] if q == qid}
            if ([r["rank"] for r in rr] != list(range(1, len(rr)+1))
                    or len(rr) != len(expected) or {r["answer"] for r in rr} != expected):
                raise ValueError("Missing or duplicate ranked candidate")
            if name.endswith("_reranker"):
                ordered = sorted(rr, key=lambda r: (-r["score"], tie_key(r)))
            elif name.endswith("_sample_frequency"):
                ordered = order_pool(rr, "sample_frequency")
            else:
                ordered = [lookup[names[name]][(qid, r["answer"])]
                           for r in pools[names[name]] if r["question_id"] == qid]
                if [r["answer"] for r in rr] != [r["answer"] for r in ordered]:
                    raise ValueError("Generated occurrence order differs")
                ordered = rr
            if rr != ordered:
                raise ValueError("Stored ordering differs from scores or fixed tie breaking")
        metrics, scored = ev.evaluate_rankings(qq)
        saved = {r["question_id"]: r for r in per_question[name]}
        if len(saved) != len(per_question[name]) or set(saved) != set(examples):
            raise ValueError("Incomplete/duplicate per-question metrics")
        if (any(abs(metrics[k] - summary["conditions"][name]["metrics"][k]) > 1e-12 for k in metrics)
                or any(abs(r[k] - saved[r["question_id"]][k]) > 1e-12
                       for r in scored for k in r if k != "question_id")):
            raise ValueError("Rankings do not reproduce reported metrics")
    return groups


def occurrences(generation, answer):
    result = []
    for sample in generation["samples"]:
        for position, candidate in enumerate(sample["answers"], 1):
            if ev.key(candidate["answer"]) == ev.key(answer):
                result.append({"draw": sample["draw"], "position": position,
                               "answer": candidate["answer"], "relation_type": candidate["relation_type"]})
    return {"draw_count": len({r["draw"] for r in result}),
            "initial_draw_count": len({r["draw"] for r in result if r["position"] == 1}),
            "occurrences": result}


def describe(rows):
    accepted = [r for r in rows if r["label"]]
    return {"pool_size": len(rows), "first_accepted_rank": accepted[0]["rank"] if accepted else None,
            "top_five": rows[:5], "accepted_candidates": accepted, "all_rankings": rows}


def build_cases(groups, examples, raw, components):
    changed = defaultdict(list)
    for mode in ev.MODES:
        for qid in examples:
            before = groups[f"sft_{mode}_max5_reranker"][qid]
            after = groups[f"dpo_{mode}_max5_reranker"][qid]
            def rr(rows):
                rank = next((r["rank"] for r in rows if r["label"]), None)
                return 1/rank if rank and rank <= 5 else 0.
            delta = rr(after) - rr(before)
            if abs(delta) > 1e-12:
                changed[qid].append({"mode": mode, "sft_mrr_at5": rr(before), "dpo_mrr_at5": rr(after), "difference": delta})
    cases = []
    for qid in sorted(changed, key=lambda q: (min(r["difference"] for r in changed[q]), q)):
        case = {**examples[qid], "changes": changed[qid], "conditions": {}, "candidate_changes": {}}
        for mode in ev.MODES:
            a, b = (groups[f"{p}_{mode}_max5_reranker"][qid] for p in ("sft", "dpo"))
            aa, bb = ({ev.key(r["answer"]): r for r in rows} for rows in (a, b))
            case["candidate_changes"][mode] = {
                "removed_expressions": [r["answer"] for k, r in aa.items() if k not in bb],
                "added_expressions": [r["answer"] for k, r in bb.items() if k not in aa],
                "focus_candidates": []}
            focus = {ev.key(r["answer"]) for rows in (a, b) for r in rows if r["label"] or r["rank"] <= 5}
            for identity in sorted(focus):
                entry = {"answer_key": identity}
                for policy, candidates in (("sft", aa), ("dpo", bb)):
                    if identity not in candidates:
                        entry[policy] = None
                        continue
                    row = candidates[identity]
                    context = {k: examples[qid][k] for k in ("question", "snippets")}
                    entry[policy] = {**row, "score_explanation": explain(components, row, context),
                        "llama_occurrences": occurrences(raw[("llama31", policy, mode, qid)], row["answer"])}
                if entry["sft"] is not None and entry["dpo"] is not None:
                    x, y = (entry[p]["score_explanation"] for p in ("sft", "dpo"))
                    entry["score_difference"] = entry["dpo"]["score"] - entry["sft"]["score"]
                    entry["score_block_differences"] = {k: y["block_totals"][k] - x["block_totals"][k] for k in x["block_totals"]}
                    entry["changed_input_metadata"] = {k: {"sft": x["input_metadata"].get(k, 0), "dpo": y["input_metadata"].get(k, 0)}
                        for k in sorted(x["input_metadata"].keys() | y["input_metadata"].keys())
                        if x["input_metadata"].get(k, 0) != y["input_metadata"].get(k, 0)}
                case["candidate_changes"][mode]["focus_candidates"].append(entry)
            for policy in ("sft", "dpo"):
                for scope, method in (("", "max5_reranker"), ("llama_", "generated_order")):
                    name = f"{scope}{policy}_{mode}_{method}"
                    case["conditions"][name] = describe(groups[name][qid])
        cases.append(case)
    return cases


def render_report(cases):
    def cell(text):
        return str(text).replace("|", "\\|").replace("\n", " ")
    lines = ["# Changes after compactness DPO", "",
        "Outcome-selected development diagnostics, not representative error rates or training examples.",
        "Official acceptance is kept separate from semantic correctness. Score decompositions explain the frozen linear model; they do not prove biomedical correctness or that retraining would fix a case.", ""]
    for case in cases:
        lines.extend([f"## {case['question_id']}", "", case["question"], "",
                      "Gold aliases: " + "; ".join(case["gold_aliases"]), ""])
        for change in case["changes"]:
            mode = change["mode"]
            lines.extend([f"### {mode}: MRR {change['sft_mrr_at5']:.4f} -> {change['dpo_mrr_at5']:.4f}", "",
                "| Policy | Rank | Answer | Officially accepted | Score | Llama draws / initial draws |",
                "|---|---:|---|---|---:|---|"])
            for policy in ("sft", "dpo"):
                d = case["conditions"][f"{policy}_{mode}_max5_reranker"]
                focus = case["candidate_changes"][mode]["focus_candidates"]
                lookup = {ev.key(r[policy]["answer"]): r[policy] for r in focus if r[policy] is not None}
                for row in d["top_five"]:
                    occurrence = lookup[ev.key(row["answer"])]["llama_occurrences"]
                    lines.append("| " + " | ".join(map(cell, (policy, row["rank"], row["answer"], bool(row["label"]),
                        f"{row['score']:.6f}", f"{occurrence['draw_count']} / {occurrence['initial_draw_count']}"))) + " |")
                lines.extend(["", f"{policy}: first accepted ensemble rank={d['first_accepted_rank']}; "
                    f"Llama generated-order rank={case['conditions'][f'llama_{policy}_{mode}_generated_order']['first_accepted_rank']}; pool size={d['pool_size']}.", ""])
            changes = case["candidate_changes"][mode]
            lines.extend(["Removed expressions: " + repr(changes["removed_expressions"]),
                          "", "Added expressions: " + repr(changes["added_expressions"]), ""])
            for r in changes["focus_candidates"]:
                if "score_difference" in r and abs(r["score_difference"]) > 1e-10:
                    lines.extend([f"- {r['answer_key']}: score delta={r['score_difference']:+.6f}; blocks={r['score_block_differences']}; changed metadata={r['changed_input_metadata']}"])
            lines.append("")
    return "\n".join(lines)


def export(comparison, output, archive=None):
    if output.exists() or (archive is not None and archive.exists()):
        raise FileExistsError("Choose new export/archive paths")
    state = ev.read(comparison / "manifest.json")
    if state.get("status") != "complete":
        raise ValueError("Need completed comparison scoring")
    ev.verify_files(comparison, state["output_sha256"])
    comparison_hash = ev.digest(comparison / "manifest.json")
    summary = ev.read(comparison / "comparison_summary.json")
    scores = ev.read(comparison / "comparison_per_question.json")
    if summary["question_count"] != 160:
        raise ValueError("Expected dev160")
    submission, generation = Path(state["submission"]), Path(state["generation"])
    if (ev.digest(submission / "prepared.json") != state["prepared_sha256"]
            or ev.digest(generation / "manifest.json") != state["generation_manifest_sha256"]):
        raise ValueError("Comparison provenance changed")
    prepared = validate_prepared(submission)
    generated = ev.read(generation / "manifest.json")
    if generated.get("status") != "complete" or generated.get("smoke_test") is not False:
        raise ValueError("Need completed full DPO generation")
    panel = ev.records(submission / "dev_gold.jsonl")
    examples = {r["question_id"]: r for r in panel}
    if len(panel) != 160 or len(examples) != 160:
        raise ValueError("Expected 160 distinct development examples")
    baseline, dpo, raw = {}, {}, {}
    for model in ev.MODELS:
        for mode in ev.MODES:
            rows = ev.load_generation(Path(prepared["baseline_runs"][f"{model}_{mode}"]), prepared["sources"][model], panel, mode)
            baseline[f"{model}_{mode}"] = rows
            raw.update({(model, "sft", mode, r["question_id"]): r for r in rows})
    for mode in ev.MODES:
        path = (generation / generated["runs"][mode]).resolve()
        if not path.is_relative_to(generation):
            raise ValueError("Unsafe generation path")
        ev.verify_files(path, generated["arm_hashes"][mode])
        dpo[mode] = ev.load_generation(path, prepared["dpo_source"], panel, mode)
        raw.update({("llama31", "dpo", mode, r["question_id"]): r for r in dpo[mode]})
    pools = ev.make_pools(baseline, dpo)
    for name, rows in pools.items():
        labeled = ev.records(comparison / f"{name}_candidates.jsonl")
        if ([{k: v for k, v in r.items() if k != "label"} for r in labeled] != rows
                or any(type(r.get("label")) is not int or r["label"] not in (0, 1) for r in labeled)):
            raise ValueError("Stored pool differs from original responses")
        pools[name] = labeled
    groups = check_rankings(ev.records(comparison / "rankings.jsonl"), pools, examples, summary, scores)
    payload, _ = ev.load_selector(Path(prepared["rerankers"]), prepared["reranker_hashes"], panel, prepared["sources"])
    cases = build_cases(groups, examples, raw, payload["components"])
    selected = {r["question_id"] for r in cases}
    output.mkdir(parents=True, exist_ok=False)
    ev.write_jsonl(output / "cases.jsonl", cases)
    ev.write_jsonl(output / "source_generations.jsonl", [{"model_key": model, "policy": policy, "mode": mode, **row}
        for (model, policy, mode, qid), row in raw.items() if qid in selected])
    (output / "report.md").write_text(render_report(cases), encoding="utf-8")
    ev.write(output / "summary.json", {"exported_questions": len(cases), "question_ids": [r["question_id"] for r in cases],
        "selection": "All ensemble max-five logistic MRR changes in greedy and ten-sample dev160 conditions; regressions first",
        "training_ready": False, "new_generation": False, "new_judging": False})
    validate_prepared(submission)
    ev.verify_files(comparison, state["output_sha256"])
    if ev.digest(comparison / "manifest.json") != comparison_hash:
        raise ValueError("Comparison changed during export")
    for mode in ev.MODES:
        ev.verify_files(generation / generated["runs"][mode], generated["arm_hashes"][mode])
    ev.write(output / "manifest.json", {"status": "complete", "comparison": str(comparison),
        "comparison_manifest_sha256": comparison_hash, "export_code_sha256": ev.digest(Path(__file__)),
        "output_sha256": {p.name: ev.digest(p) for p in output.iterdir() if p.is_file()}})
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(output.iterdir()):
                handle.add(path, arcname=path.name)
    print((output / "report.md").read_text(encoding="utf-8"), flush=True)
    print(f"Exported {len(cases)} cases to {output}", flush=True)
    if archive is not None:
        print(f"Download: {archive}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    export(args.comparison.resolve(), args.output.resolve(), args.archive.resolve() if args.archive else None)
