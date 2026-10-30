#!/usr/bin/env python3
"""CPU-only rescore of complete pool-aware generations; originals remain frozen."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCORER = "scripts/pool_aware_fitting_pilot.py"
OLD = '            if c["relation_type"] not in t["relation_types"]: t["relation_types"].append(c["relation_type"])'
NEW = ('            relation=str(c.get("relation_type") or "unrecognized")\n'
       '            if relation not in t["relation_types"]: t["relation_types"].append(relation)')


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify(root, hashes):
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Frozen recovery input changed: " + name)


def corrected(text):
    if text.count(OLD) != 1 or NEW in text:
        raise ValueError("Recovery requires the original missing-relation pool construction")
    return text.replace(OLD, NEW)


def frozen_inputs(submission):
    p = read(submission / "prepared.json")
    if p["schema"] != "pool-aware-fitting-pilot-v1":
        raise ValueError("Wrong original pilot submission")
    verify(submission / "code", p["code_hashes"])
    verify(submission, p["generation_files_sha256"])
    verify(submission, p["files_sha256"])
    verify(submission, p["scorer_files_sha256"])
    with (submission / "jobs.tsv").open() as f:
        jobs = [r for r in csv.DictReader(f, delimiter="\t") if r["stage"] == "generation"]
    models = ("llama31", "qwen3", "ministral3")
    if len(jobs) != 3 or {r["model"] for r in jobs} != set(models):
        raise ValueError("Need exactly three full generation records")
    hashes = {"prepared.json": digest(submission / "prepared.json"), "jobs.tsv": digest(submission / "jobs.tsv")}
    hashes.update(p["generation_files_sha256"])
    hashes.update(p["files_sha256"])
    hashes.update(p["scorer_files_sha256"])
    for r in jobs:
        path = Path(r["output"]).resolve()
        if not path.is_relative_to((submission / r["model"]).resolve()):
            raise ValueError("Generation directory escaped original model directory")
        state = read(path / "manifest.json")
        if (state["status"] != "complete" or state["smoke_test"] is not False
                or state["prepared_sha256"] != hashes["prepared.json"]
                or state["model_key"] != r["model"] or state["configuration"] != p["configuration"]
                or state["source"] != p["sources"][r["model"]]):
            raise ValueError("Need all three complete original full generations")
        verify(path, state["output_sha256"])
        for file in [path / "manifest.json", *(path / n for n in state["output_sha256"])]:
            hashes[file.relative_to(submission).as_posix()] = digest(file)
    return p, hashes


def prepare(submission, recovery, repository):
    if recovery.exists(): raise FileExistsError(recovery)
    p, pins = frozen_inputs(submission)
    original = (submission / "code" / SCORER).read_text(encoding="utf-8")
    patch = corrected(original)
    if (repository / SCORER).read_text(encoding="utf-8") != patch:
        raise ValueError("Checkout must contain exactly the declared scorer correction")
    recovery.mkdir(parents=True)
    hashes = dict(p["code_hashes"])
    for name in hashes:
        target = recovery / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(submission / "code" / name, target)
    target = recovery / "code" / SCORER
    target.write_text(patch, encoding="utf-8", newline="\n")
    hashes[SCORER] = digest(target)
    shutil.copy2(Path(__file__), recovery / "runner.py")
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/recover_pool_aware_scoring.pbs", recovery / "job.pbs")
    write(recovery / "prepared.json", {
        "submission": str(submission), "original_files_sha256": pins,
        "code_hashes": hashes, "original_code_sha256": p["code_hashes"],
        "runner_sha256": digest(recovery / "runner.py"), "job_sha256": digest(recovery / "job.pbs"),
        "correction": "Use parser-compatible unrecognized for missing/empty relation metadata; preserve all candidate surfaces and draws",
        "generation_reused": True, "GPU_jobs": 0, "training": False, "paid_judging": False,
    })
    # A fresh process uses the frozen scorer's imports, not modules from the checkout.
    subprocess.run([sys.executable, str(recovery / "runner.py"), "--mode", "check", "--recovery", str(recovery)], check=True)
    print("Prepared one CPU-only rescore: all saved requests and pools verified; real Java smoke passed. Original submission unchanged.", flush=True)


def validate(recovery):
    p = read(recovery / "prepared.json")
    verify(recovery, {"runner.py": p["runner_sha256"], "job.pbs": p["job_sha256"]})
    original = Path(p["submission"])
    verify(original, p["original_files_sha256"])
    verify(original / "code", p["original_code_sha256"])
    original_text = (original / "code" / SCORER).read_text(encoding="utf-8")
    if (recovery / "code" / SCORER).read_text(encoding="utf-8") != corrected(original_text):
        raise ValueError("Recovery exceeds the declared missing-relation correction")
    expected = {**p["original_code_sha256"], SCORER: digest(recovery / "code" / SCORER)}
    if p["code_hashes"] != expected:
        raise ValueError("Other frozen scoring code changed")
    verify(recovery / "code", p["code_hashes"])
    return p


def check(recovery):
    p = validate(recovery)
    sys.path.insert(0, str(recovery / "code"))
    sys.path.insert(1, str(recovery / "code/gadi_sft_8b_starter/scripts"))
    from scripts import pool_aware_fitting_pilot as pilot
    if Path(pilot.__file__).resolve() != (recovery / "code" / SCORER).resolve():
        raise ValueError("Recovery must use a fresh process and its frozen scorer")
    original = Path(p["submission"])
    state = pilot.validate_submission(original)
    with (original / "jobs.tsv").open() as f:
        jobs = [r for r in csv.DictReader(f, delimiter="\t") if r["stage"] == "generation"]
    generated = {}
    for job in jobs:
        model = job["model"]
        requests = {r["request_key"]: r for r in pilot.records(original / f"inputs/{model}.jsonl")}
        plans = pilot.read(original / f"plans/{model}.json")
        rows = pilot.records(Path(job["output"]) / "generations.jsonl")
        if len(rows) != len(requests) or {r["request_key"] for r in rows} != set(requests):
            raise ValueError("Incomplete or duplicate saved requests")
        for r in rows:
            pilot.verify_result(r, requests[r["request_key"]], plans[r["request_key"]], model, state["configuration"])
            generated[model, r["question_id"], r["arm"]] = r
    ids = set(state["question_ids"])
    baseline = pilot.records(original / "baseline_responses.jsonl")
    pilot.check_baseline(baseline, ids, state["configuration"])
    arms = pilot.assemble_arms(baseline, generated, state["question_ids"])
    for rows in arms.values(): pilot.pool_rows(rows, ids)
    pilot.scorer.smoke(original / "scorer/BioASQEvaluation.jar")
    print(f"Verified {len(generated)} saved fresh responses, all condition pools and frozen Java runtime.", flush=True)
    return p, pilot


def run(recovery, output):
    if output.exists(): raise FileExistsError(output)
    p, pilot = check(recovery)
    prepared_sha = digest(recovery / "prepared.json")
    pilot.score(Path(p["submission"]), output)
    validate(recovery)
    if digest(recovery / "prepared.json") != prepared_sha: raise ValueError("Recovery metadata changed")
    manifest = read(output / "manifest.json")
    manifest["scoring_recovery"] = {"directory": str(recovery), "prepared_sha256": prepared_sha,
        "generation_reused": True, "correction": p["correction"]}
    write(output / "manifest.json", manifest)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--mode", choices=("prepare", "check", "run"), required=True)
    a.add_argument("--submission", type=Path)
    a.add_argument("--recovery", type=Path, required=True)
    a.add_argument("--output", type=Path)
    x = a.parse_args()
    if x.mode == "prepare":
        if not x.submission: a.error("--submission is required")
        prepare(x.submission.resolve(), x.recovery.resolve(), Path(__file__).resolve().parents[1])
    elif x.mode == "check": check(x.recovery.resolve())
    else:
        run(x.recovery.resolve(), (x.output or x.recovery / f"comparison-{os.environ.get('PBS_JOBID', 'manual')}").resolve())
