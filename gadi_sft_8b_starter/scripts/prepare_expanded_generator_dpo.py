#!/usr/bin/env python3
"""Prepare expanded compactness training and full validation144 epoch selection."""
import argparse
from pathlib import Path
import shutil

from run_generator_feedback_dpo import REPOSITORY, prepare
from run_base_original_prompt import snapshot_inference_code
from prepare_matched_8b_sft import read, write, digest


def prepare_expanded(repository, preferences, submission):
    config = repository / "gadi_sft_8b_starter/configs/generator_feedback_dpo_expanded.json"
    prepare(repository, preferences, submission, config)
    prepared = read(submission / "prepared.json")
    snapshot = snapshot_inference_code(submission)
    prepared.update(code_snapshot=snapshot, worker_sha256=snapshot["files_sha256"]["scripts/run_single_answer_sampling.py"])
    for kind in ("train", "select"):
        filename = f"{kind}_expanded_generator_dpo.pbs"
        shutil.copy2(repository / "gadi_sft_8b_starter/jobs" / filename, submission / filename)
    prepared["expanded_job_sha256"] = {f"{k}_expanded_generator_dpo.pbs": digest(submission / f"{k}_expanded_generator_dpo.pbs") for k in ("train", "select")}
    write(submission / "prepared.json", prepared)
    print("Prepared expanded Llama DPO, all retained epochs and starting-SFT fallback; selection uses validation144 only.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=REPOSITORY)
    p.add_argument("--preferences", type=Path, required=True)
    p.add_argument("--submission", type=Path, required=True)
    a = p.parse_args()
    prepare_expanded(a.repository.resolve(), a.preferences.resolve(), a.submission.resolve())
