#!/usr/bin/env python3
"""Export every question with no officially accepted answer in the sampled pool.

Reuses verified saved generations and candidate labels. No generation, training,
reranker fitting, Java scoring or paid semantic review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rescore_sft_ensemble_frequency as source
from scripts.analyze_sft_ensemble_test import order_pool

TARGET = "max5_sampling10"


def group_pools(pools, examples):
    groups = {}
    expected = {f"{arm}_{mode}" for arm in source.ARMS for mode in source.MODES}
    if set(pools) != expected:
        raise ValueError("Need all four original/max-five greedy/sampling pools")
    for name, rows in pools.items():
        groups[name] = {q: [] for q in examples}
        seen = set()
        for row in rows:
            q = row["question_id"]
            identity = (q, row["answer"])
            if q not in examples or identity in seen or type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("Unknown, duplicate or invalidly labeled pool candidate")
            seen.add(identity)
            groups[name][q].append(row)
    return groups


def collect_cases(pools, examples, gold, expected_misses=20):
    if set(examples) != set(gold):
        raise ValueError("Gold and inference question membership differ")
    groups = group_pools(pools, examples)
    missing = sorted(q for q, rows in groups[TARGET].items() if not any(r["label"] for r in rows))
    if len(missing) != expected_misses:
        raise ValueError(f"Expected {expected_misses} full-pool absences; found {len(missing)}")
    cases = []
    for q in missing:
        conditions = {}
        for name, questions in groups.items():
            rows = questions[q]
            ordered = order_pool(rows, "sample_frequency")
            accepted = [r for r in rows if r["label"]]
            conditions[name] = {
                "pool_size": len(rows), "coverage_full_pool": bool(accepted),
                "accepted_candidates": accepted, "candidates": rows,
                "sample_frequency_top5": ordered[:5],
                "model_unique_candidates": {m: sum(m in r["sources"] for r in rows) for m in source.MODELS},
            }
        cases.append({"question_id": q, "question": examples[q]["question"],
            "batch": gold[q]["batch"], "gold_aliases": gold[q]["gold_aliases"],
            "snippets": examples[q]["snippets"], "conditions": conditions,
            "review": {"status": "unreviewed", "diagnostic_category": None,
                "supported_but_officially_unaccepted_answer_present": None,
                "gold_answer_supported_in_supplied_evidence": None,
                "initial_answer_semantics": dict.fromkeys(source.MODELS), "notes": ""}})
    summary = {"question_count": len(examples), "target_pool": TARGET,
        "covered_in_full_pool": len(examples) - len(missing), "absent_from_full_pool": len(missing),
        "question_ids": missing, "absences_per_batch": dict(sorted(Counter(c["batch"] for c in cases).items())),
        "among_target_absences_covered_by": {
            name: sum(c["conditions"][name]["coverage_full_pool"] for c in cases) for name in groups},
        "training_ready": False,
        "interpretation": "All full-pool official-acceptance absences, not covered-but-outside-top-five reranker misses. Test diagnostics only.",
        "semantic_caution": "No accepted string anywhere in the parsed capped pool does not establish that every answer is semantically wrong. Semantic review is unresolved."}
    return cases, summary


def export(comparison, output, archive=None, expected_misses=20):
    comparison, output = comparison.resolve(), output.resolve()
    archive = archive.resolve() if archive is not None else None
    if output.exists() or (archive is not None and (archive.exists() or archive.is_relative_to(output))):
        raise ValueError("Choose new output and archive paths; keep the archive outside the output")
    if expected_misses < 1:
        raise ValueError("Need a positive expected absence count")
    _, pools, examples, gold, hashes = source.load_comparison(comparison)
    if len(examples) != 95:
        raise ValueError("Expected the 95-question test comparison, not dev or validation")
    cases, summary = collect_cases(pools, examples, gold, expected_misses)
    selected = set(summary["question_ids"])
    sources = source.read(comparison / "manifest.json")["sources"]
    raw, raw_hashes, source_hashes = [], {}, {}
    for name, metadata in sources.items():
        directory = Path(metadata["directory"])
        state = source.read(directory / "manifest.json")
        if source.digest(directory / "manifest.json") != metadata["manifest_sha256"]:
            raise ValueError("Source archive changed after verification")
        source_hashes[str(directory / "manifest.json")] = metadata["manifest_sha256"]
        for arm, relative in state["runs"].items():
            path = (directory / relative / "generations.jsonl").resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Generation path escaped its source archive")
            expected = state["arm_hashes"][arm]["generations.jsonl"]
            if source.digest(path) != expected:
                raise ValueError("Saved generation changed after verification")
            raw_hashes[f"{name}/{arm}"] = {"path": str(path), "sha256": expected}
            rows = source.records(path)
            chosen = [r for r in rows if r["question_id"] in selected]
            if len(chosen) != len(selected) or {r["question_id"] for r in chosen} != selected:
                raise ValueError("Missing or duplicate raw responses for an exported question")
            raw.extend({"source": name, "arm": arm, **r} for r in chosen)
    if len(raw_hashes) != 12:
        raise ValueError("Need three models, both prompts and both generation modes")
    output.mkdir(parents=True)
    source.write_jsonl(output / "cases.jsonl", cases)
    source.write_jsonl(output / "source_generations.jsonl", raw)
    summary["exported_generation_rows"] = len(raw)
    source.write(output / "summary.json", summary)
    lines = ["# Max-five ten-sample ensemble: accepted answers absent from the pool", "",
        f"{summary['absent_from_full_pool']} of {summary['question_count']} test questions have no officially accepted candidate anywhere in the parsed capped pool.", "",
        "These cases are independent of ranking: a reranker cannot promote an accepted expression that is absent. Semantic quality and evidence adequacy still require inspection.", "",
        "The export preserves every pool candidate, full evidence and gold aliases, all saved responses for both prompts and decoding modes, and a blank semantic-review template. Frequency top-five lists are descriptive baselines, not the logistic reranker's output.", "",
        "| Batch | Question | Accepted aliases | Max-five sampled pool size | Covered by original ten-sample pool? |",
        "|---|---|---|---:|---|"]
    for c in cases:
        cells = [c["batch"], c["question"], "; ".join(c["gold_aliases"]),
                 str(c["conditions"][TARGET]["pool_size"]), str(c["conditions"]["original_sampling10"]["coverage_full_pool"])]
        lines.append("| " + " | ".join(s.replace("|", "\\|").replace("\n", " ") for s in cells) + " |")
    lines.extend(["", "Inspect accepted wording versus semantically valid unaccepted forms; initial-answer errors; unsupported/missing evidence; parse/output-cap failures; and coverage by the original prompt. These outcome-selected test cases must not become training preferences.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for path, expected in [*((comparison / n, h) for n, h in hashes.items()),
                           *((Path(n), h) for n, h in source_hashes.items()),
                           *((Path(x["path"]), x["sha256"]) for x in raw_hashes.values())]:
        if source.digest(path) != expected:
            raise ValueError("A source changed during export")
    source.write(output / "manifest.json", {"status": "complete", "training": False, "generation": False,
        "paid_judging": False, "new_official_scoring": False, "comparison": str(comparison),
        "comparison_sha256": hashes, "source_generation_sha256": raw_hashes,
        "source_manifest_sha256": source_hashes, "export_code_sha256": source.digest(Path(__file__)),
        "output_sha256": {p.name: source.digest(p) for p in output.iterdir() if p.is_file()}})
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "x:gz") as handle:
            for path in sorted(output.iterdir()):
                handle.add(path, arcname=path.name, recursive=False)
        with tarfile.open(archive, "r:gz") as handle:
            files = {p.name: source.digest(p) for p in output.iterdir() if p.is_file()}
            if {m.name for m in handle.getmembers()} != set(files):
                raise ValueError("Archive membership differs from the export")
            import hashlib
            for name, expected in files.items():
                if hashlib.sha256(handle.extractfile(name).read()).hexdigest() != expected:
                    raise ValueError("Archived bytes differ from the verified export")
        print(f"Verified archive: {archive}; SHA256={source.digest(archive)}", flush=True)
    print(f"Exported all {len(cases)} full-pool absences and {len(raw)} saved generation rows to {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--expected-misses", type=int, default=20)
    args = p.parse_args()
    export(args.comparison, args.output, args.archive, args.expected_misses)
