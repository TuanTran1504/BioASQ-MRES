#!/usr/bin/env python3
"""Paired greedy five/ten candidate prompts on pinned instruction backbones."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, digest, read, records, write
from run_matched_8b_evaluation import DEFAULT_CONFIG, validate_training_pair
from run_extractive_expansion_8b import parse_equivalent_response
from expansion_prompt_budget import five_prompt

sys.path.insert(0, str(ROOT.parent))


def prepare(destination, models):
    config = read(DEFAULT_CONFIG)
    if not models or len(set(models)) != len(models) or not set(models) <= set(config["models"]):
        raise ValueError("Use unique configured backbone names")
    if destination.exists():
        raise FileExistsError(destination)
    # Existing matched-run provenance pins the original instruction backbone,
    # native tokenizer settings and stable Ministral eager loader, not adapters.
    sources = {m: validate_training_pair(config, m)["expansion"] for m in models}
    ten = (ROOT / "prompts/equivalent_expansion_v1.txt").read_text(encoding="utf-8")
    five = five_prompt(ten)
    from src.utility.bioasq_official import _conda_executable, _official_adapter_classes
    subprocess.run([*_conda_executable("java"), "-version"], check=True)
    _official_adapter_classes(ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    destination.mkdir(parents=True, exist_ok=False)
    for limit, prompt in ((10, ten), (5, five)):
        (destination / f"prompt_max{limit}.txt").write_text(prompt, encoding="utf-8", newline="\n")
    write(destination / "manifest.json", {"status": "ready", "models": sources,
        "input": str((ROOT / config["input"]).resolve()), "input_sha256": config["input_sha256"],
        "expected_questions": 160, "max_seq_length": 6144, "max_new_tokens": 512,
        "temperature": 0, "seed": 3407,
        "prompt_sha256": {str(k): digest(destination / f"prompt_max{k}.txt") for k in (10, 5)},
        "base_config_sha256": {m: digest(Path(s["base_snapshot"]) / "config.json") for m, s in sources.items()},
        "evaluation": "Development-only base instruction models; matched greedy prompt budget comparison"})
    print(f"Prepared base-only five/ten candidate evaluation: {destination}", flush=True)


def validate_prepared(submission, model):
    prepared = read(submission / "manifest.json")
    if prepared["status"] != "ready" or model not in prepared["models"]:
        raise ValueError("Unprepared model")
    if digest(Path(prepared["input"])) != prepared["input_sha256"]:
        raise ValueError("Prepared development data changed")
    examples = records(Path(prepared["input"]))
    if len(examples) != 160 or len({r["question_id"] for r in examples}) != 160:
        raise ValueError("Expected 160 unique questions")
    for k in (10, 5):
        if digest(submission / f"prompt_max{k}.txt") != prepared["prompt_sha256"][str(k)]:
            raise ValueError("Prepared prompt changed")
    source = prepared["models"][model]
    snapshot = Path(source["base_snapshot"])
    if (snapshot.name != source["base_revision"] or
            digest(snapshot / "config.json") != prepared["base_config_sha256"][model] or
            (snapshot / "adapter_config.json").exists()):
        raise ValueError("Expected the pinned base backbone, without an SFT or DPO adapter")
    return prepared, source


def run(submission, model, output, smoke):
    prepared, source = validate_prepared(submission, model)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "model_key": model, "smoke_test": smoke,
        "expected_questions": 4 if smoke else 160, "provenance": source,
        "prepared_manifest": str(submission / "manifest.json"),
        "prepared_manifest_sha256": digest(submission / "manifest.json"),
        "worker_sha256": digest(ROOT / "scripts/run_extractive_expansion_8b.py"),
        "evaluation": prepared["evaluation"], "runs": {}, "arm_hashes": {}}
    write(output / "manifest.json", state)
    try:
        for limit in (10, 5):
            condition = f"max{limit}"
            config = {"input": prepared["input"], "prompt": str(submission / f"prompt_max{limit}.txt"),
                "max_seq_length": prepared["max_seq_length"], "max_new_tokens": prepared["max_new_tokens"],
                "temperature": 0, "seed": prepared["seed"], "response_mode": "equivalent",
                "max_candidates": limit, "require_all_snippets": True,
                "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
                "execution_mode": source["execution_mode"], "prompt_version": f"equivalent-expansion-max{limit}-v1"}
            config_path = output / f"{condition}_config.json"
            write(config_path, config)
            parent = output / condition
            parent.mkdir()
            subprocess.run([sys.executable, str(ROOT / "scripts/run_extractive_expansion_8b.py"),
                "--config", str(config_path), "--model-name", source["base_snapshot"],
                "--output-parent", str(parent), "--run-name", condition,
                "--limit", str(state["expected_questions"])], check=True, cwd=ROOT)
            children = [p for p in parent.iterdir() if p.is_dir()]
            if len(children) != 1:
                raise ValueError("Expected one generated arm")
            path = children[0]
            generation = records(path / "generations.jsonl")
            if read(path / "status.json")["status"] != "complete" or len(generation) != state["expected_questions"]:
                raise ValueError("Incomplete generation")
            if smoke and not any(parseable(r["raw_response"], limit) for r in generation):
                raise ValueError(f"Smoke produced no usable {condition} candidates")
            state["runs"][condition] = path.relative_to(output).as_posix()
            state["arm_hashes"][condition] = {n: digest(path / n) for n in (
                "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
            write(output / "manifest.json", state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/analyze_base_candidate_budget.py"), str(output),
                    *( ["--allow-smoke"] if smoke else [])], check=True, cwd=ROOT)
    print(f"Completed: {output / 'comparison_summary.json'}", flush=True)


def parseable(raw, limit):
    try:
        return bool(parse_equivalent_response(raw, limit)[0])
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen3", "ministral3"])
    parser.add_argument("--model", choices=("llama31", "qwen3", "ministral3"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    submission = args.submission_dir.resolve()
    if args.mode == "prepare":
        prepare(submission, args.models)
    else:
        if not args.model:
            parser.error("--model is required for run")
        suffix = "-smoke" if args.smoke_test else ""
        output = ROOT / "outputs/base_candidate_budget" / f"{args.model}-base-budget-{os.environ.get('PBS_JOBID', 'manual')}{suffix}"
        run(submission, args.model, output, args.smoke_test)


if __name__ == "__main__":
    main()
