#!/usr/bin/env python3
"""Blinded, cached semantic review of the compactness/retention shortlist."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import judge_expansion_dpo_annotations as shared
from scripts.judge_single_answer_dpo import paid_session

MODEL = "gpt-4.1-mini-2025-04-14"
INPUT_RATE, OUTPUT_RATE = .4 / 1_000_000, 1.6 / 1_000_000
SYSTEM = shared.SYSTEM.replace("and accepted training aliases", "without gold-answer aliases").replace(
    "Accepted aliases are incomplete reference labels,\nnot an exhaustive definition of scientific correctness.",
    "No accepted aliases are provided; judge using the question and supplied evidence.")


def load_shortlist(directory):
    manifest = shared.read(directory / "manifest.json")
    if manifest.get("status") != "review_required" or manifest.get("models") != ["llama31"]:
        raise ValueError("Expected the completed Llama generator-feedback package")
    for name, digest in manifest["output_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or shared.digest(path) != digest:
            raise ValueError("Source preference package changed")
    pairs = []
    for split in ("train", "validation"):
        for objective in ("compactness", "retain_needed_expansion"):
            rows = shared.records(directory / f"llama31/{split}_{objective}_review_candidates.jsonl")
            if objective == "compactness":
                rows = [p for p in rows if p["category"] == "same_accepted_initial" and
                        p["chosen_outcome"]["pool_candidates"] < p["rejected_outcome"]["pool_candidates"]]
            pairs.extend(rows)
    wanted = {p[k] for p in pairs for k in ("chosen_id", "rejected_id")}
    rows = [r for r in shared.records(directory / "review_responses.jsonl") if r["response_id"] in wanted]
    if len(rows) != len(wanted) or {r["response_id"] for r in rows} != wanted:
        raise ValueError("Missing or duplicate shortlisted raw responses")
    return manifest, pairs, rows


def prepare(rows):
    aliases = {r["question_id"]: {"exact_answer": []} for r in rows}
    jobs, units = shared.prepare(rows, aliases)
    split = {r["question_id"]: r["split"] for r in rows}
    for job in jobs:
        payload = job["payload"]
        payload["model"] = MODEL
        payload["messages"][0]["content"] = SYSTEM
        public = json.loads(payload["messages"][1]["content"])
        del public["accepted_training_aliases"]
        payload["messages"][1]["content"] = json.dumps(public, ensure_ascii=False)
        payload["max_tokens"] = min(32768, 256 + 320 * len(job["unit_ids"]))
        job["request_sha256"] = shared.sha(payload)
        job["maximum_cost_reserve_usd"] = ((len(json.dumps(payload, ensure_ascii=False).encode()) + 2048)
                                             * INPUT_RATE + payload["max_tokens"] * OUTPUT_RATE)
    random.Random(3407).shuffle(jobs)
    jobs.sort(key=lambda j: split[units[j["unit_ids"][0]]["question_id"]] != "validation")
    return jobs, units


def run(source, output, execute=False, max_calls=10000, maximum=10.0, workers=4, key_file=ROOT / "open_ai_api.txt"):
    if not 0 < maximum <= 10 or not 1 <= workers <= 8 or max_calls < 1:
        raise ValueError("Use a positive call limit, 1-8 workers and at most US$10")
    manifest, pairs, rows = load_shortlist(source)
    jobs, units = prepare(rows)
    state = {"status": "prepared", "source": str(source.resolve()), "source_manifest_sha256": shared.digest(source / "manifest.json"),
             "model": MODEL, "system_sha256": shared.sha(SYSTEM), "shortlisted_pairs": len(pairs),
             "response_count": len(rows), "unit_count": len(units), "planned_requests": len(jobs),
             "max_usd": maximum, "full_plan_maximum_reserve_usd": sum(j["maximum_cost_reserve_usd"] for j in jobs),
             "annotation_source": "LLM-only; assistant vetoes recorded separately; not independent expert review",
             "pricing_source": "https://developers.openai.com/api/docs/models/gpt-4.1-mini"}
    if (output / "manifest.json").exists():
        old = shared.read(output / "manifest.json")
        if any(old[k] != state[k] for k in ("source_manifest_sha256", "model", "system_sha256", "unit_count", "max_usd")):
            raise ValueError("Cannot resume a changed judging plan or budget")
    shared.save(output / "manifest.json", state)
    shared.save(output / "plan.json", jobs)
    (output / "shortlisted_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in pairs), encoding="utf-8")
    if not execute:
        print(json.dumps(state, indent=2)); return
    decisions = {}
    with paid_session(output):
        client = shared.BudgetClient(output, maximum, key_file, input_rate=INPUT_RATE,
                                    output_rate=OUTPUT_RATE, validator=shared.validate_decisions)
        pending = []
        for job in jobs:
            if (output / "cache" / (job["request_sha256"] + ".json")).exists():
                decisions.update(client.call(job))
            else:
                pending.append(job)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tasks = [pool.submit(client.call, job) for job in pending[:max_calls]]
            for number, future in enumerate(as_completed(tasks), 1):
                decisions.update(future.result())
                if number % 10 == 0:
                    print(f"Reviewed requests {number}/{len(tasks)}; charged/reserved US${client.spent + client.reserved:.4f}", flush=True)
    annotated, exclusions = shared.apply_labels(rows, decisions)
    for row in annotated:
        row["reviewer"] = "LLM:" + MODEL
    (output / "reviewed.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in annotated), encoding="utf-8")
    state.update(status="complete" if len(decisions) == len(units) else "partial", judged_units=len(decisions),
                 exclusions=exclusions, charged_or_reserved_usd=client.spent, blocking_reason=client.blocked_reason,
                 reviewed_sha256=shared.digest(output / "reviewed.jsonl"))
    shared.save(output / "manifest.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--max-new-calls", type=int, default=10000)
    p.add_argument("--max-usd", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.run, args.max_new_calls, args.max_usd, args.workers)
