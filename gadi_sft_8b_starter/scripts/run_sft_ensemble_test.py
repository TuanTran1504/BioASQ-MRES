#!/usr/bin/env python3
"""Generate ten samples for both SFT formulations; reuse completed greedy tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, read, records, write, digest
from run_sft_test_evaluation import prepare as prepare_test
from run_base_original_prompt import inference_worker

sys.path.insert(0, str(ROOT.parent))
from scripts.analyze_sft_ensemble_test import MODELS, ARMS, load_archive
from scripts.train_sft_ensemble_reranker import snapshot_code

GREEDY_JOBS = dict(zip(MODELS, ("181236678", "181236680", "181236682")))


def prepare(destination):
    # Reuse the full-input, disjoint-test, frozen-checkpoint and Java preflights.
    prepare_test(destination, list(MODELS))
    state = read(destination / "manifest.json")
    archives = {}
    expected = {r["question_id"]: r for r in records(destination / "test_inputs.jsonl")}
    expected_gold = {r["question_id"]: r for r in records(destination / "test_gold.jsonl")}
    for model in MODELS:
        directory = ROOT / "outputs/sft_test_evaluation" / f"{model}-sft-test-{GREEDY_JOBS[model]}.gadi-pbs"
        saved, old, examples, gold, _, _ = load_archive(directory, "greedy")
        if saved["model_key"] != model or {e["question_id"]: e for e in examples} != expected or gold != expected_gold:
            raise ValueError("Greedy archive test panel differs")
        for arm in ARMS:
            for field in ("adapter_files_sha256", "base_snapshot", "base_revision", "system_prompt",
                          "model_loader", "chat_template_kwargs", "execution_mode"):
                if old["sources"][model][arm]["source"].get(field) != state["sources"][model][arm]["source"].get(field):
                    raise ValueError(f"Greedy archive checkpoint or prompt differs: {model}/{arm}/{field}")
        archives[model] = str(directory)
    state.update(repository=str(ROOT.parent), greedy_archives=archives,
                 greedy_manifest_hashes={m: digest(Path(p) / "manifest.json") for m, p in archives.items()},
                 sampling={"num_generations": 10, "temperature": 0.8, "top_p": 0.95, "seed": 3407},
                 analysis_code_hashes=snapshot_code(ROOT.parent, destination / "code"))
    write(destination / "manifest.json", state)
    print("Prepared ten-sample original/max-five evaluation; completed greedy archives will be reused", flush=True)


def validate_code(submission, prepared):
    for relative, value in prepared["analysis_code_hashes"].items():
        if digest(submission / "code" / relative) != value:
            raise ValueError("Frozen orchestration/scoring code changed")


def run(submission, model, smoke):
    prepared = read(submission / "manifest.json")
    if not prepared["ready"] or model not in MODELS:
        raise ValueError("Submission not prepared for model")
    validate_code(submission, prepared)
    for name, value in prepared["file_hashes"].items():
        if digest(submission / name) != value:
            raise ValueError("Prepared input/gold data changed")
    worker = inference_worker(submission, prepared)
    starter = Path(prepared["repository"]) / "gadi_sft_8b_starter"
    output = starter / "outputs/sft_ensemble_test" / (
        f"{model}-sampling10-{os.environ.get('PBS_JOBID', 'manual')}" + ("-smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "model_key": model, "smoke_test": smoke,
             "expected_questions": 4 if smoke else 95, "submission_dir": str(submission),
             "submission_manifest_sha256": digest(submission / "manifest.json"),
             "runs": {}, "arm_hashes": {}, "settings": {}, "training": False}
    write(output / "manifest.json", state)
    try:
        for arm in ARMS:
            entry = prepared["sources"][model][arm]
            source = entry["source"]
            for name, value in source["adapter_files_sha256"].items():
                if digest(Path(source["adapter"]) / name) != value:
                    raise ValueError("Frozen adapter changed")
            prompt = output / f"{arm}_prompt.txt"
            prompt.write_text(source["system_prompt"], encoding="utf-8")
            plan = entry["smoke_plan" if smoke else "test_plan"]
            settings = {"input": str(submission / ("smoke_inputs.jsonl" if smoke else "test_inputs.jsonl")),
                "prompt": str(prompt), **prepared["sampling"], "max_new_tokens": 512,
                "max_candidates": 5 if arm == "max5" else 1,
                "response_mode": "equivalent_sampling" if arm == "max5" else "single_answer_sampling",
                "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
                "execution_mode": source.get("execution_mode", "default"), "mark_snippets": False,
                "require_all_snippets": True, "max_seq_length": plan["max_seq_length"],
                "expected_prompt_tokens": plan["expected_prompt_tokens"],
                "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
                "input_token_policy": "complete_evidence_auto_size", "prompt_version": "sft-ensemble-test-" + arm + "-v1"}
            cfg = output / f"{arm}_config.json"; write(cfg, settings)
            arm_dir = output / arm
            subprocess.run([sys.executable, str(worker), "--config", str(cfg), "--model-name", source["adapter"],
                "--output-dir", str(arm_dir), "--limit", str(state["expected_questions"]), "--preflight-all",
                *(["--longest-first"] if smoke else [])], cwd=starter, check=True)
            if (read(arm_dir / "status.json")["status"] != "complete"
                    or len(records(arm_dir / "generations.jsonl")) != state["expected_questions"]):
                raise ValueError("Incomplete generation")
            generated = records(arm_dir / "generations.jsonl")
            if smoke and not any(not s["parse_error"] for row in generated for s in row["samples"]):
                raise ValueError(f"{arm}: no parseable smoke responses; inspect raw completions")
            state["runs"][arm] = arm
            state["settings"][arm] = settings
            state["arm_hashes"][arm] = {f: digest(arm_dir / f) for f in
                ("config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "status.json")}
            write(output / "manifest.json", state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    print(f"Completed ten-sample generation: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "run", "record"))
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.submission.resolve())
    elif args.mode == "record":
        import csv
        with (args.submission / "jobs.tsv").open(encoding="utf-8") as handle:
            jobs = {r["model"]: r["job_id"] for r in csv.DictReader(handle, delimiter="\t") if r["stage"] == "full"}
        if set(jobs) != set(MODELS):
            raise ValueError("Need three full generation jobs")
        write(args.submission / "sampling_jobs.json", jobs)
    else:
        if args.model is None:
            parser.error("--model required for run")
        run(args.submission.resolve(), args.model, args.smoke_test)
