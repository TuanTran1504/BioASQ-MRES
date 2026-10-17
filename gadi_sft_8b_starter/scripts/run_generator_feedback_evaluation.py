#!/usr/bin/env python3
"""Freeze and generate the Llama compactness DPO versus starting-SFT dev160 comparison."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))

from prepare_matched_8b_sft import read, records, write, digest
from run_base_original_prompt import full_input_plan, snapshot_inference_code, inference_worker
from run_generator_feedback_dpo import validate_native
from scripts.prepare_reranker_guided_dpo import configure_data_root, load_inputs
from scripts.train_sft_ensemble_reranker import snapshot_code
from scripts.ablate_reranker_relation_labels import verify_files
from scripts.analyze_selected_sft_evaluation import validate_experiment
from scripts.analyze_generator_feedback_evaluation import (
    MODELS, MODES, load_generation, load_selector, validate_membership,
)

OUTPUT = "outputs/generator_feedback_dpo_evaluation"


def validate_training(training, repository):
    state = read(training / "manifest.json")
    if (state.get("status") != "complete" or state.get("smoke_test") is not False
            or state.get("model_key") != "llama31" or state.get("phase") != "save_adapter"
            or state.get("global_steps", 0) <= 0):
        raise ValueError("Need completed full Llama compactness DPO training")
    submission = Path(state["submission"])
    if digest(submission / "prepared.json") != state["prepared_sha256"]:
        raise ValueError("DPO training preparation changed")
    prepared = read(submission / "prepared.json")
    if state["configuration"] != prepared["configuration"]:
        raise ValueError("DPO training configuration changed")
    manifest, splits, source, feedback = validate_native(
        Path(prepared["preferences"]), state["configuration"], repository)
    counts = {s: len(rr) for s, rr in splits.items()}
    if (state["source"] != source or prepared["source"] != source
            or state["trained_pair_counts"] != counts or state["preference_pair_counts"] != counts
            or prepared["native_banks"] != feedback["banks"]
            or state["preferences_manifest_sha256"] != digest(Path(prepared["preferences"]) / "manifest.json")
            or prepared["preferences_manifest_sha256"] != state["preferences_manifest_sha256"]):
        raise ValueError("DPO run differs from its exact source and reviewed preferences")
    if digest(training / "reference_logps.jsonl") != state["reference_logps_sha256"]:
        raise ValueError("Frozen SFT reference log probabilities changed")
    adapter = training / "adapter"
    files = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
    if files != state["adapter_files_sha256"]:
        raise ValueError("Saved DPO adapter changed")
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if not (adapter / name).is_file() or not (adapter / name).stat().st_size:
            raise ValueError("Missing or empty DPO adapter: " + name)
    if read(adapter / "adapter_config.json")["base_model_name_or_path"] != source["base_snapshot"]:
        raise ValueError("DPO backbone differs from the starting SFT")
    return state, prepared, source, feedback, splits


def prepare(repository, training, submission):
    state, trained, source, feedback, splits = validate_training(training, repository)
    data = configure_data_root(repository)
    config = read(data.DEFAULT_CONFIG)
    sources = {m: data.selected_source(config, m) for m in MODELS}
    if sources["llama31"] != source:
        raise ValueError("Starting Llama differs from DPO reference")
    rerankers = Path(feedback["rerankers"])
    audit_config = read(repository / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json")
    _, _, _, _, hashes, banks, _ = load_inputs(repository, rerankers, audit_config)
    if hashes != feedback["reranker_hashes"] or banks != feedback["banks"]:
        raise ValueError("Reranker or banks differ from the feedback used to build preferences")
    archives, pins, baseline, examples = {}, {}, {}, None
    for model in MODELS:
        archive = repository / "gadi_sft_8b_starter" / config["models"][model]
        saved, paths = validate_experiment(archive)
        if saved["model_key"] != model or saved["common_settings"]["max_candidates"] != 5:
            raise ValueError("Need exact max-five SFT dev archives")
        archives[model] = str(archive)
        pins[str(archive / "manifest.json")] = digest(archive / "manifest.json")
        for mode in MODES:
            path = paths[f"best_eval_loss_{mode}"]
            if read(path / "config.json")["model_name"] != sources[model]["adapter"]:
                raise ValueError("Archive adapter differs from loss-selected SFT")
            panel = records(path / "examples.jsonl")
            if examples is None:
                examples = panel
            if panel != examples:
                raise ValueError("Starting SFT questions, evidence or aliases differ")
            baseline[f"{model}_{mode}"] = str(path)
            for name, value in saved["arm_hashes"][f"best_eval_loss_{mode}"].items():
                pins[str(path / name)] = value
            load_generation(path, sources[model], examples, mode)
    # Validate against the actual pinned outer development file as well as the
    # archived inference panels. Neither fitting nor validation may overlap it.
    matrix = read(repository / "gadi_sft_8b_starter" / config["training_matrix"])
    dev = repository / "gadi_sft_8b_starter" / matrix["outer_dev"]
    if records(dev) != examples:
        raise ValueError("Archives differ from the pinned outer dev160 input")
    membership = {s: records(rerankers / f"{s}_examples.jsonl") for s in ("train", "validation")}
    validate_membership(examples, membership, splits)
    selector, _ = load_selector(rerankers, hashes, examples, sources)
    dpo_source = {**source, "adapter": str(training / "adapter"),
                  "adapter_files_sha256": state["adapter_files_sha256"]}
    plan = full_input_plan(dpo_source, examples, 512)
    llama_config = read(Path(baseline["llama31_greedy"]) / "config.json")
    for field in ("expected_prompt_tokens", "expected_prompt_ids_sha256", "max_seq_length"):
        if plan[field] != llama_config[field]:
            raise ValueError("DPO tokenization or context allocation differs from starting SFT: " + field)
    # Java matching is preflighted on the login node; inference gets a separate
    # gold-blind file with exactly the same question and snippet payloads.
    from src.utility.bioasq_official import _official_adapter_classes
    jar = repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    _official_adapter_classes(jar)
    submission.mkdir(parents=True, exist_ok=False)
    from run_extractive_expansion_8b import write_jsonl
    write_jsonl(submission / "dev_inputs.jsonl", [{k: r[k] for k in ("question_id", "question", "snippets")} for r in examples])
    write_jsonl(submission / "dev_gold.jsonl", examples)
    (submission / "prompt.txt").write_text(source["system_prompt"], encoding="utf-8")
    code = snapshot_code(repository, submission / "code")
    inference = snapshot_inference_code(submission)
    for kind in ("generate", "score"):
        shutil.copy2(repository / f"gadi_sft_8b_starter/jobs/{kind}_generator_feedback_evaluation.pbs", submission / f"{kind}.pbs")
    for path in (training / "manifest.json", training / "reference_logps.jsonl", dev, jar):
        pins[str(path)] = digest(path)
    for model, ss in {**sources, "dpo": dpo_source}.items():
        for name, value in ss["adapter_files_sha256"].items():
            pins[str(Path(ss["adapter"]) / name)] = value
        pins[str(Path(ss["base_snapshot"]) / "config.json")] = digest(Path(ss["base_snapshot"]) / "config.json")
    pins[str(rerankers / "manifest.json")] = hashes["manifest.json"]
    for name, value in hashes.items():
        pins[str(rerankers / name)] = value
    write(submission / "prepared.json", {"repository": str(repository), "training": str(training),
        "sources": sources, "dpo_source": dpo_source, "baseline_runs": baseline, "archives": archives,
        "rerankers": str(rerankers), "reranker_hashes": hashes, "reranker_selected_C": selector["selected_C"],
        "input_plan": plan, "external_hashes": pins, "code_hashes": code,
        "code_snapshot": inference, "worker_sha256": inference["files_sha256"]["scripts/run_single_answer_sampling.py"],
        "file_hashes": {n: digest(submission / n) for n in ("dev_inputs.jsonl", "dev_gold.jsonl", "prompt.txt", "generate.pbs", "score.pbs")},
        "evaluation": "Exploratory dev160; frozen loss-selected DPO; only Llama changes; no fitting or test evaluation"})
    print("Validated exact SFT/DPO lineage, unchanged companions and max-five reranker; reusing six dev archives.", flush=True)


def validate_prepared(submission):
    prepared = read(submission / "prepared.json")
    verify_files(submission, prepared["file_hashes"])
    verify_files(submission / "code", prepared["code_hashes"])
    for name, value in prepared["external_hashes"].items():
        if digest(Path(name)) != value:
            raise ValueError("Pinned evaluation input changed: " + name)
    inference_worker(submission, prepared)
    return prepared


def generate(submission, output, smoke=False):
    prepared = validate_prepared(submission)
    source = prepared["dpo_source"]
    examples = records(submission / "dev_inputs.jsonl")
    plan = prepared["input_plan"]
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "smoke_test": smoke, "expected_questions": 4 if smoke else 160,
             "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
             "source": source, "runs": {}, "arm_hashes": {}}
    write(output / "manifest.json", state)
    try:
        for mode in MODES:
            cfg = {"input": str(submission / "dev_inputs.jsonl"), "prompt": str(submission / "prompt.txt"),
                "response_mode": "equivalent_greedy" if mode == "greedy" else "equivalent_sampling",
                "num_generations": 1 if mode == "greedy" else 10, "temperature": 0 if mode == "greedy" else .8,
                "top_p": .95, "top_k": 0, "seed": 3407, "max_new_tokens": 512, "max_candidates": 5,
                "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
                "execution_mode": source.get("execution_mode", "default"), "mark_snippets": False,
                "require_all_snippets": True, "max_seq_length": plan["max_seq_length"],
                "expected_prompt_tokens": plan["expected_prompt_tokens"],
                "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
                "input_token_policy": "complete_evidence_auto_size", "prompt_version": "generator-feedback-dpo-dev160-v1"}
            path = output / f"{mode}_config.json"
            write(path, cfg)
            subprocess.run([sys.executable, str(inference_worker(submission, prepared)), "--config", str(path),
                "--model-name", source["adapter"], "--output-dir", str(output / mode), "--limit",
                str(state["expected_questions"]), "--preflight-all", *(["--longest-first"] if smoke else [])], check=True)
            expected = sorted(examples, key=lambda e: (-plan["expected_prompt_tokens"][e["question_id"]], e["question_id"]))[:4] if smoke else examples
            rows = load_generation(output / mode, source, expected, mode)
            if smoke and not any(s["answers"] for r in rows for s in r["samples"]):
                raise ValueError("No parseable DPO smoke responses")
            state["runs"][mode] = mode
            state["arm_hashes"][mode] = {n: digest(output / mode / n) for n in (
                "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
            write(output / "manifest.json", state)
        validate_prepared(submission)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "generate"), required=True)
    p.add_argument("--repository", type=Path, default=REPOSITORY)
    p.add_argument("--training", type=Path)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.mode == "prepare":
        if args.training is None:
            p.error("--training required")
        prepare(args.repository.resolve(), args.training.resolve(), args.submission.resolve())
    else:
        prepared = read(args.submission / "prepared.json")
        suffix = "smoke" if args.smoke_test else "generation"
        output = Path(prepared["repository"]) / "gadi_sft_8b_starter" / OUTPUT / f"{suffix}-{os.environ.get('PBS_JOBID', 'manual')}"
        generate(args.submission.resolve(), output, args.smoke_test)
