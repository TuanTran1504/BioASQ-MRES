#!/usr/bin/env python3
"""Generate one greedy answer on the same base checkpoints as saved five/ten runs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, digest, read, records, write
from run_base_candidate_budget import validate_prepared, parseable
sys.path.insert(0, str(ROOT.parent))
from scripts.analyze_base_candidate_budget import validate_experiment

DEFAULT_CONFIG = ROOT / "configs/base_single_answer_evaluation.json"


def one_prompt(ten):
    marker = "First choose your single best answer concept."
    if ten.count(marker) != 1:
        raise ValueError("Expected established equivalent-expansion prompt")
    prefix = ten.split(marker)[0]
    return prefix + ('Choose your single best answer concept. Return exactly ONE concise answer expression, '
        'marked original. Do not provide alternatives, equivalent variants, explanations or a list of answers.\n'
        'Return {"answers": [{"answer": "...", "relation_type": "original"}]}.\n')


def checked_baseline(directory, model):
    manifest, runs = validate_experiment(directory)
    if manifest["model_key"] != model:
        raise ValueError("Baseline belongs to another backbone")
    prepared_path = Path(manifest["prepared_manifest"])
    prepared, source = validate_prepared(prepared_path.parent, model)
    if source != manifest["provenance"]:
        raise ValueError("Baseline base checkpoint provenance changed")
    return manifest, runs, prepared, source


def prepare(config_path, destination, models):
    config = read(config_path)
    if not models or len(set(models)) != len(models) or not set(models) <= set(config["baselines"]):
        raise ValueError("Use unique configured model names")
    if destination.exists():
        raise FileExistsError(destination)
    sources, prompts, panels = {}, {}, []
    for model in models:
        directory = (ROOT / config["baselines"][model]).resolve()
        manifest, runs, prepared, source = checked_baseline(directory, model)
        ten = Path(read(runs["max10"] / "config.json")["prompt"]).read_text(encoding="utf-8")
        prompts[model] = one_prompt(ten)
        panels.append(records(runs["max10"] / "examples.jsonl"))
        sources[model] = {"baseline": str(directory), "baseline_manifest_sha256": digest(directory / "manifest.json"),
            "provenance": source, "input": prepared["input"], "input_sha256": prepared["input_sha256"],
            "max_seq_length": prepared["max_seq_length"], "max_new_tokens": prepared["max_new_tokens"],
            "seed": prepared["seed"], "base_config_sha256": digest(Path(source["base_snapshot"]) / "config.json")}
    if any(panel != panels[0] for panel in panels[1:]):
        raise ValueError("Backbone question/evidence panels differ")
    from src.utility.bioasq_official import _conda_executable, _official_adapter_classes
    subprocess.run([*_conda_executable("java"), "-version"], check=True)
    _official_adapter_classes(ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    destination.mkdir(parents=True, exist_ok=False)
    for model, prompt in prompts.items():
        path = destination / f"{model}_one_answer_prompt.txt"
        path.write_text(prompt, encoding="utf-8", newline="\n")
        sources[model]["prompt_sha256"] = digest(path)
    write(destination / "manifest.json", {"status": "ready", "models": sources,
        "config_sha256": digest(config_path), "expected_questions": 160,
        "evaluation": "Exploratory dev160 base instruction models; one vs five vs ten greedy prompt comparison"})
    print(f"Prepared one-answer evaluation: {destination}", flush=True)


def validate_submission(submission, model):
    manifest = read(submission / "manifest.json")
    if manifest["status"] != "ready" or model not in manifest["models"]:
        raise ValueError("Unprepared model")
    source = manifest["models"][model]
    baseline = Path(source["baseline"])
    old, runs, prepared, provenance = checked_baseline(baseline, model)
    if (digest(baseline / "manifest.json") != source["baseline_manifest_sha256"] or
            provenance != source["provenance"] or digest(Path(source["input"])) != source["input_sha256"] or
            digest(Path(provenance["base_snapshot"]) / "config.json") != source["base_config_sha256"]):
        raise ValueError("Frozen baseline, input or base checkpoint changed")
    prompt = submission / f"{model}_one_answer_prompt.txt"
    ten = Path(read(runs["max10"] / "config.json")["prompt"]).read_text(encoding="utf-8")
    if digest(prompt) != source["prompt_sha256"] or prompt.read_text(encoding="utf-8") != one_prompt(ten):
        raise ValueError("Prepared one-answer prompt changed")
    return manifest, source, runs


def run(submission, model, output, smoke):
    prepared, source, baseline_runs = validate_submission(submission, model)
    output.mkdir(parents=True, exist_ok=False)
    provenance = source["provenance"]
    state = {"status": "running", "model_key": model, "smoke_test": smoke,
        "expected_questions": 4 if smoke else 160, "prepared_manifest": str(submission / "manifest.json"),
        "prepared_manifest_sha256": digest(submission / "manifest.json"), "baseline": source["baseline"],
        "baseline_manifest_sha256": source["baseline_manifest_sha256"], "provenance": provenance,
        "evaluation": prepared["evaluation"], "runs": {}, "arm_hashes": {},
        "worker_sha256": digest(ROOT / "scripts/run_extractive_expansion_8b.py")}
    write(output / "manifest.json", state)
    try:
        config = {"input": source["input"], "prompt": str(submission / f"{model}_one_answer_prompt.txt"),
            "model_loader": provenance["model_loader"], "chat_template_kwargs": provenance["chat_template_kwargs"],
            "execution_mode": provenance["execution_mode"], "max_seq_length": source["max_seq_length"],
            "max_new_tokens": source["max_new_tokens"], "temperature": 0, "seed": source["seed"],
            "response_mode": "equivalent", "max_candidates": 1, "require_all_snippets": True,
            "prompt_version": "base-one-answer-equivalent-json-v1"}
        config_path = output / "max1_config.json"
        write(config_path, config)
        parent = output / "max1"
        parent.mkdir()
        subprocess.run([sys.executable, str(ROOT / "scripts/run_extractive_expansion_8b.py"),
            "--config", str(config_path), "--model-name", provenance["base_snapshot"],
            "--output-parent", str(parent), "--run-name", "max1", "--limit", str(state["expected_questions"])],
            check=True, cwd=ROOT)
        children = [p for p in parent.iterdir() if p.is_dir()]
        if len(children) != 1:
            raise ValueError("Expected one generated one-answer arm")
        path = children[0]
        rows = records(path / "generations.jsonl")
        if read(path / "status.json")["status"] != "complete" or len(rows) != state["expected_questions"]:
            raise ValueError("Incomplete one-answer generation")
        if smoke and not any(parseable(r["raw_response"], 1) for r in rows):
            raise ValueError("Smoke produced no usable single answers")
        state["runs"]["max1"] = path.relative_to(output).as_posix()
        state["arm_hashes"]["max1"] = {n: digest(path / n) for n in (
            "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/analyze_base_single_answer.py"), str(output),
                    *(["--allow-smoke"] if smoke else [])], check=True, cwd=ROOT)
    print(f"Completed: {output / 'comparison_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["llama31", "qwen3", "ministral3"])
    parser.add_argument("--model", choices=("llama31", "qwen3", "ministral3"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.config.resolve(), args.submission_dir.resolve(), args.models)
    else:
        if not args.model:
            parser.error("--model is required")
        suffix = "-smoke" if args.smoke_test else ""
        output = ROOT / "outputs/base_single_answer" / f"{args.model}-base-single-{os.environ.get('PBS_JOBID', 'manual')}{suffix}"
        run(args.submission_dir.resolve(), args.model, output, args.smoke_test)


if __name__ == "__main__":
    main()
