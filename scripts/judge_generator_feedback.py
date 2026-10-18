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


def load_shortlist(directory, scope="pool_reducing", model=None):
    if scope not in ("pool_reducing", "all_compactness"):
        raise ValueError("Unknown compactness review scope")
    manifest = shared.read(directory / "manifest.json")
    models = manifest.get("models", [])
    if (manifest.get("status") != "review_required" or not models or len(set(models)) != len(models)
            or not set(models) <= {"llama31", "qwen3", "ministral3"}):
        raise ValueError("Expected completed model-specific generator-feedback preferences")
    if model is not None and model not in models:
        raise ValueError("Requested model is absent from the feedback package")
    for name, digest in manifest["output_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or shared.digest(path) != digest:
            raise ValueError("Source preference package changed")
    pairs = []
    for focal in ([model] if model is not None else models):
        for split in ("train", "validation"):
            for objective in ("compactness", "retain_needed_expansion"):
                rows = shared.records(directory / f"{focal}/{split}_{objective}_review_candidates.jsonl")
                if any(p["model_key"] != focal or p["split"] != split or p["objective"] != objective for p in rows):
                    raise ValueError("Feedback file contains a different model, split or objective")
                if objective == "compactness":
                    rows = [p for p in rows if p["category"] == "same_accepted_initial" and
                            (scope == "all_compactness" or
                             p["chosen_outcome"]["pool_candidates"] < p["rejected_outcome"]["pool_candidates"])]
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


def reuse_decisions(source, seed):
    """Reuse only validated cached units from the same evidence, model and rubric."""
    state = shared.read(seed / "manifest.json")
    if (state["status"] != "complete" or state["source_manifest_sha256"] != shared.digest(source / "manifest.json")
            or state["model"] != MODEL or state["system_sha256"] != shared.sha(SYSTEM)
            or state["reviewed_sha256"] != shared.digest(seed / "reviewed.jsonl")):
        raise ValueError("Seed judgments have different provenance, model or rubric")
    decisions = {}
    for job in shared.read(seed / "plan.json"):
        cached = shared.read(seed / "cache" / (job["request_sha256"] + ".json"))
        if cached["payload"] != job["payload"]:
            raise ValueError("Seed cached request changed")
        decisions.update(shared.validate_decisions(cached["decisions"], job))
    if len(decisions) != state["judged_units"]:
        raise ValueError("Incomplete seed decisions")
    return decisions


def missing_jobs(jobs, decisions):
    result = []
    for job in jobs:
        missing = [i for i, uid in enumerate(job["unit_ids"]) if uid not in decisions]
        if not missing:
            continue
        payload = json.loads(json.dumps(job["payload"]))
        public = json.loads(payload["messages"][1]["content"])
        public["candidates"] = [{**public["candidates"][old], "index": new} for new, old in enumerate(missing)]
        payload["messages"][1]["content"] = json.dumps(public, ensure_ascii=False)
        payload["max_tokens"] = min(32768, 256 + 320*len(missing))
        result.append({"request_sha256": shared.sha(payload), "payload": payload,
            "unit_ids": [job["unit_ids"][i] for i in missing],
            "maximum_cost_reserve_usd": (len(json.dumps(payload, ensure_ascii=False).encode()) + 2048)*INPUT_RATE + payload["max_tokens"]*OUTPUT_RATE})
    return result


def recover_truncated_jobs(jobs, budget):
    """Increase output allowance only for saved length-truncated judgments."""
    result = []
    for job in jobs:
        key = job["request_sha256"]
        if not (budget / "cache" / (key + ".json")).exists():
            responses = list((budget / "responses").glob(key + "-*.json"))
            if any(shared.read(p)["choices"][0]["finish_reason"] == "length" for p in responses):
                payload = json.loads(json.dumps(job["payload"]))
                payload["max_tokens"] = min(32768, payload["max_tokens"] * 2)
                job = {**job, "payload": payload, "request_sha256": shared.sha(payload),
                       "maximum_cost_reserve_usd": (len(json.dumps(payload, ensure_ascii=False).encode()) + 2048)*INPUT_RATE + payload["max_tokens"]*OUTPUT_RATE}
        result.append(job)
    return result


def run(source, output, execute=False, max_calls=10000, maximum=10.0, workers=4, key_file=ROOT / "open_ai_api.txt",
        scope="pool_reducing", seed_judge=None, budget_directory=None):
    if not 0 < maximum <= 10 or not 1 <= workers <= 8 or max_calls < 1:
        raise ValueError("Use a positive call limit, 1-8 workers and at most US$10")
    manifest, pairs, rows = load_shortlist(source, scope)
    jobs, units = prepare(rows)
    reused = reuse_decisions(source, seed_judge) if seed_judge is not None else {}
    if seed_judge is not None:
        reused = {k: v for k, v in reused.items() if k in units}
        jobs = missing_jobs(jobs, reused)
    if budget_directory is not None and seed_judge is not None and budget_directory.resolve() != seed_judge.resolve():
        raise ValueError("Seed judgments must retain their original shared spending ledger")
    budget = budget_directory or (seed_judge if seed_judge is not None else output)
    if budget_directory is not None and not (budget / "spend-ledger.json").is_file():
        raise ValueError("Explicit shared budget must already have a spending ledger")
    jobs = recover_truncated_jobs(jobs, budget)
    state = {"status": "prepared", "source": str(source.resolve()), "source_manifest_sha256": shared.digest(source / "manifest.json"),
             "model": MODEL, "system_sha256": shared.sha(SYSTEM), "shortlisted_pairs": len(pairs),
             "response_count": len(rows), "unit_count": len(units), "planned_requests": len(jobs),
             "max_usd": maximum, "review_scope": scope, "reused_units": len(reused),
             "budget_directory": str(budget.resolve()),
             "seed_judging_manifest_sha256": shared.digest(seed_judge / "manifest.json") if seed_judge else None,
             "full_plan_maximum_reserve_usd": sum(j["maximum_cost_reserve_usd"] for j in jobs),
             "annotation_source": "LLM-only; assistant vetoes recorded separately; not independent expert review",
             "pricing_source": "https://developers.openai.com/api/docs/models/gpt-4.1-mini"}
    if (output / "manifest.json").exists():
        old = shared.read(output / "manifest.json")
        old.setdefault("review_scope", "pool_reducing")
        old.setdefault("budget_directory", str(output.resolve()))
        if any(old.get(k) != state.get(k) for k in ("source_manifest_sha256", "model", "system_sha256", "unit_count", "max_usd", "review_scope", "seed_judging_manifest_sha256", "budget_directory")):
            raise ValueError("Cannot resume a changed judging plan or budget")
    shared.save(output / "manifest.json", state)
    shared.save(output / "plan.json", jobs)
    (output / "shortlisted_pairs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False)+"\n" for p in pairs), encoding="utf-8")
    if not execute:
        print(json.dumps(state, indent=2)); return
    decisions = dict(reused)
    with paid_session(budget):
        client = shared.BudgetClient(budget, maximum, key_file, input_rate=INPUT_RATE,
                                    output_rate=OUTPUT_RATE, validator=shared.validate_decisions)
        pending = []
        for job in jobs:
            if (budget / "cache" / (job["request_sha256"] + ".json")).exists():
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
    p.add_argument("--scope", choices=("pool_reducing", "all_compactness"), default="pool_reducing")
    p.add_argument("--seed-judge", type=Path)
    p.add_argument("--budget-directory", type=Path, help="Reuse an existing cumulative spending ledger for this new review batch")
    args = p.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.run, args.max_new_calls, args.max_usd, args.workers,
        scope=args.scope, seed_judge=args.seed_judge.resolve() if args.seed_judge else None,
        budget_directory=args.budget_directory.resolve() if args.budget_directory else None)
