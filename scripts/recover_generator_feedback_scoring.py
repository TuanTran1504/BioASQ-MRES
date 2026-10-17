#!/usr/bin/env python3
"""Rescore existing responses with one recorded output-directory correction."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys

SCORER = "scripts/analyze_generator_feedback_evaluation.py"
ANCHOR = '    matches = matcher(list(examples.values()), list(distinct.values()), output / "official", jar_path=jar) if distinct else {}'
CORRECTION = '    (output / "official").mkdir(parents=True, exist_ok=True)\n'


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def verify(directory, hashes):
    for relative, value in hashes.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != value:
            raise ValueError("Pinned scoring input changed: " + str(path))


def validate_original(submission, generation):
    prepared = read(submission / "prepared.json")
    verify(submission / "code", prepared["code_hashes"])
    verify(submission, prepared["file_hashes"])
    for path, value in prepared["external_hashes"].items():
        if digest(Path(path)) != value:
            raise ValueError("Pinned evaluation source changed: " + path)
    state = read(generation / "manifest.json")
    if (state.get("status") != "complete" or state.get("smoke_test") is not False
            or state.get("expected_questions") != 160 or set(state["runs"]) != {"greedy", "sampling10"}
            or Path(state["submission"]).resolve() != submission
            or state["prepared_sha256"] != digest(submission / "prepared.json")):
        raise ValueError("Need the completed full generation from this original submission")
    for mode, relative in state["runs"].items():
        path = (generation / relative).resolve()
        if not path.is_relative_to(generation):
            raise ValueError("Generation run escaped its directory")
        verify(path, state["arm_hashes"][mode])
    return prepared


def prepare(submission, generation, recovery):
    prepared = validate_original(submission, generation)
    original = submission / "code" / SCORER
    text = original.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1 or CORRECTION.strip() in text:
        raise ValueError("Recovery only applies to the known missing official-folder failure")
    # Copy the submitted code, not a newer checkout. Only this one explicit
    # filesystem fix is allowed; feature extraction and all scoring rules stay.
    recovery.mkdir(parents=True, exist_ok=False)
    hashes = dict(prepared["code_hashes"])
    for relative in hashes:
        target = recovery / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(submission / "code" / relative, target)
    corrected = recovery / "code" / SCORER
    corrected.write_text(text.replace(ANCHOR, CORRECTION + ANCHOR), encoding="utf-8", newline="\n")
    hashes[SCORER] = digest(corrected)
    shutil.copy2(Path(__file__), recovery / "runner.py")
    job = Path(__file__).resolve().parents[1] / "gadi_sft_8b_starter/jobs/rescore_generator_feedback_evaluation.pbs"
    shutil.copy2(job, recovery / "job.pbs")
    write(recovery / "prepared.json", {"submission": str(submission), "generation": str(generation),
        "original_prepared_sha256": digest(submission / "prepared.json"),
        "generation_manifest_sha256": digest(generation / "manifest.json"),
        "code_hashes": hashes, "original_scorer_sha256": prepared["code_hashes"][SCORER],
        "runner_sha256": digest(recovery / "runner.py"), "job_sha256": digest(recovery / "job.pbs"),
        "correction": "Create official report directory before writing Java scorer inputs",
        "generation_reused": True, "GPU_jobs": 0, "training": False, "paid_judging": False})
    print("Validated saved generation; prepared one CPU-only rescore with the directory fix. Original submission unchanged.", flush=True)


def validate_recovery(recovery):
    state = read(recovery / "prepared.json")
    submission, generation = Path(state["submission"]), Path(state["generation"])
    if (digest(submission / "prepared.json") != state["original_prepared_sha256"]
            or digest(generation / "manifest.json") != state["generation_manifest_sha256"]):
        raise ValueError("Original submission or generation metadata changed")
    original = validate_original(submission, generation)
    expected = dict(original["code_hashes"])
    text = (submission / "code" / SCORER).read_text(encoding="utf-8")
    fixed = text.replace(ANCHOR, CORRECTION + ANCHOR)
    if (text.count(ANCHOR) != 1 or CORRECTION.strip() in text
            or (recovery / "code" / SCORER).read_text(encoding="utf-8") != fixed
            or original["code_hashes"][SCORER] != state["original_scorer_sha256"]):
        raise ValueError("Recovery contains more than the declared directory correction")
    expected[SCORER] = digest(recovery / "code" / SCORER)
    if state["code_hashes"] != expected:
        raise ValueError("Recovery changed other submitted code")
    verify(recovery / "code", state["code_hashes"])
    verify(recovery, {"runner.py": state["runner_sha256"], "job.pbs": state["job_sha256"]})
    return state


def run(recovery, output):
    state = validate_recovery(recovery)
    metadata_hash = digest(recovery / "prepared.json")
    sys.path.insert(0, str(recovery / "code"))
    sys.path.insert(1, str(recovery / "code/gadi_sft_8b_starter/scripts"))
    sys.argv = [str(recovery / "code" / SCORER), "--submission", state["submission"],
                "--generation", state["generation"], "--output", str(output)]
    runpy.run_path(str(recovery / "code" / SCORER), run_name="__main__")
    validate_recovery(recovery)
    if digest(recovery / "prepared.json") != metadata_hash:
        raise ValueError("Recovery preparation changed while scoring")
    manifest = read(output / "manifest.json")
    manifest["scoring_recovery"] = {"directory": str(recovery), "prepared_sha256": metadata_hash,
                                     "correction": state["correction"], "generation_reused": True}
    write(output / "manifest.json", manifest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("prepare", "run"))
    p.add_argument("--submission", type=Path)
    p.add_argument("--generation", type=Path)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.mode == "prepare":
        if args.submission is None or args.generation is None:
            p.error("Preparation requires --submission and --generation")
        prepare(args.submission.resolve(), args.generation.resolve(), args.recovery.resolve())
    else:
        if args.output is None:
            p.error("Run requires --output")
        run(args.recovery.resolve(), args.output.resolve())
