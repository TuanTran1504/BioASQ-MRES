#!/usr/bin/env python3
"""Freeze three predeclared compactness ensembles; generate epoch-one test draws."""
import argparse
import copy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))

from prepare_matched_8b_sft import read, records, write, digest
from run_base_original_prompt import full_input_plan, snapshot_inference_code, inference_worker
from run_generator_feedback_dpo import validate_native
from select_generator_feedback_dpo import choose
from select_sft_epoch_checkpoints import epoch_checkpoints
from scripts.ablate_reranker_relation_labels import verify_files
from scripts.rescore_sft_ensemble_frequency import load_comparison, MODES, MODELS
from scripts.analyze_sft_ensemble_test import load_archive
from scripts.analyze_generator_feedback_evaluation import load_generation, input_view
from scripts.evaluate_sft_ensemble_rerankers import load_frozen
from scripts.train_sft_ensemble_reranker import snapshot_code
from run_extractive_expansion_8b import write_jsonl

OUTPUT = "outputs/compactness_test95"
CONFIGURATIONS = {
    "baseline": dict.fromkeys(MODELS, 0),
    "strict": {"llama31": 0, "qwen3": 1, "ministral3": 0},
    "alternative": dict.fromkeys(MODELS, 1),
}
SELECTION_JOBS = {
    "llama31": ("181507870", "181507871"),
    "qwen3": ("181529747", "181529748"),
    "ministral3": ("181529764", "181529765"),
}
RUN_FILES = ("config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl",
             "invalid_candidates.jsonl", "status.json")


def pin_tree(directory, hashes, pins):
    verify_files(directory, hashes)
    pins.update({str(directory / name): value for name, value in hashes.items()})


def validate_epoch_one(selection, model, repository):
    """Use retained epoch 1, never the final/loss-selected training adapter."""
    selected = read(selection / "manifest.json")
    if (selected.get("status") != "complete" or selected.get("smoke_test") is not False
            or selected.get("validation_question_count") != 144 or selected.get("focal_model") != model):
        raise ValueError("Need a complete validation144 selection for " + model)
    training = Path(selected["training"])
    state = read(training / "manifest.json")
    submission = Path(state["submission"])
    prepared = read(submission / "prepared.json")
    if (state.get("status") != "complete" or state.get("smoke_test") is not False
            or state.get("phase") != "save_adapter" or state.get("model_key") != model
            or state.get("global_steps", 0) <= 0
            or state.get("configuration") != prepared["configuration"]
            or state["configuration"].get("model_key") != model
            or not state["configuration"].get("retain_epoch_checkpoints")
            or Path(selected["submission"]).resolve() != submission.resolve()
            or selected["training_manifest_sha256"] != digest(training / "manifest.json")
            or state["prepared_sha256"] != digest(submission / "prepared.json")
            or selected["prepared_sha256"] != state["prepared_sha256"]):
        raise ValueError("Training and validation-selection provenance differ: " + model)
    preferences = Path(prepared["preferences"])
    _, splits, source, feedback = validate_native(preferences, state["configuration"], repository)
    counts = {s: len(rr) for s, rr in splits.items()}
    if (state["source"] != source or prepared["source"] != source
            or state["trained_pair_counts"] != counts or state["preference_pair_counts"] != counts
            or prepared["native_banks"] != feedback["banks"]
            or state["preferences_manifest_sha256"] != digest(preferences / "manifest.json")
            or prepared["preferences_manifest_sha256"] != state["preferences_manifest_sha256"]
            or selected["reranker_hashes"] != feedback["reranker_hashes"]):
        raise ValueError("DPO source/preferences differ: " + model)
    pins = {}
    for path in (selection / "manifest.json", training / "manifest.json", submission / "prepared.json",
                 preferences / "manifest.json", training / "reference_logps.jsonl"):
        pins[str(path)] = digest(path)
    if pins[str(training / "reference_logps.jsonl")] != state["reference_logps_sha256"]:
        raise ValueError("SFT reference probabilities changed")
    pin_tree(selection, selected["output_sha256"], pins)
    pin_tree(training / "adapter", state["adapter_files_sha256"], pins)
    summary = read(selection / "comparison_summary.json")
    winner = choose(copy.deepcopy(summary["checkpoint_scores"]), model)
    if (summary["question_count"] != 144 or summary["selected_epoch"] != winner["epoch"]
            or selected["selected_epoch"] != winner["epoch"]
            or winner["epoch"] != CONFIGURATIONS["strict"][model]):
        raise ValueError("Strict selection differs from predeclared test configuration: " + model)
    checkpoints = epoch_checkpoints(training, int(state["configuration"]["num_train_epochs"]))
    first = next(c for c in checkpoints if c["epoch"] == 1)
    adapter = selection / "epoch-1/adapter"
    for filename, value in first["adapter_sha256"].items():
        if digest(adapter / filename) != value:
            raise ValueError("Staged epoch-one adapter differs from retained training checkpoint")
    files = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if not (adapter / name).is_file() or not (adapter / name).stat().st_size:
            raise ValueError("Epoch-one adapter lacks " + name)
    if read(adapter / "adapter_config.json")["base_model_name_or_path"] != source["base_snapshot"]:
        raise ValueError("Epoch-one backbone differs from starting SFT")
    pin_tree(adapter, files, pins)
    return source, {**source, "adapter": str(adapter), "adapter_files_sha256": files, "dpo_epoch": 1}, feedback, splits, pins


def validate_partition(examples, membership, splits):
    ids = {e["question_id"] for e in examples}
    if len(examples) != 95 or len(ids) != 95:
        raise ValueError("Need exactly 95 distinct test questions")
    groups = {s: {r["question_id"] for r in rr} for s, rr in membership.items()}
    if ({s: len(v) for s, v in groups.items()} != {"train": 1296, "validation": 144}
            or any(len(membership[s]) != len(v) for s, v in groups.items())
            or groups["train"] & groups["validation"] or any(ids & v for v in groups.values())):
        raise ValueError("Test questions overlap fitting/validation or invalid membership")
    if any(r["question_id"] not in groups[s] for s, rr in splits.items() for r in rr):
        raise ValueError("DPO preferences outside fitting/validation partition")


def baseline_archives(comparison, state, sources, examples, pins):
    result = {}
    for model in MODELS:
        for mode in MODES:
            archive = Path(state["sources"][f"{model}_{mode}"]["directory"])
            saved, prepared, panel, _, _, _ = load_archive(archive, mode)
            native = prepared["sources"][model]["max5"]["source"]
            if native != sources[model] or {e["question_id"]: e for e in panel} != examples:
                raise ValueError("Test SFT archive differs from DPO starting source or panel")
            run = (archive / saved["runs"]["max5"]).resolve()
            result[f"{model}_{mode}"] = str(run)
            pins[str(archive / "manifest.json")] = digest(archive / "manifest.json")
            pin_tree(run, {n: digest(run / n) for n in RUN_FILES}, pins)
            load_generation(run, sources[model], panel, mode)
    return result


def prepare(repository, comparison, rerankers, selections, submission):
    original, _, examples, gold, hashes = load_comparison(comparison)
    state = read(comparison / "manifest.json")
    if len(examples) != 95 or set(selections) != set(MODELS):
        raise ValueError("Need test95 and all three completed selections")
    pins = {str(comparison / n): v for n, v in hashes.items()}
    sources, dpo, feedbacks, preference_splits = {}, {}, {}, {}
    for model in MODELS:
        sources[model], dpo[model], feedbacks[model], preference_splits[model], extra = validate_epoch_one(
            selections[model], model, repository)
        pins.update(extra)
    _, payloads, reranker_hashes = load_frozen(rerankers, examples, comparison)
    pin_tree(rerankers, reranker_hashes, pins)
    membership = {s: records(rerankers / f"{s}_examples.jsonl") for s in ("train", "validation")}
    for model in MODELS:
        feedback = feedbacks[model]
        if (Path(feedback["rerankers"]).resolve() != rerankers.resolve()
                or feedback["reranker_hashes"] != reranker_hashes):
            raise ValueError("Feedback used a different frozen reranker")
        validate_partition(list(examples.values()), membership, preference_splits[model])
    baseline = baseline_archives(comparison, state, sources, examples, pins)
    # Match archived row order as well as payloads; worker hashes are order-sensitive.
    panel = records(Path(baseline["llama31_greedy"]) / "examples.jsonl")
    plans = {}
    for model in MODELS:
        plans[model] = full_input_plan(dpo[model], panel, 512)
        for mode in MODES:
            cfg = read(Path(baseline[f"{model}_{mode}"]) / "config.json")
            if any(plans[model][k] != cfg[k] for k in (
                    "expected_prompt_tokens", "expected_prompt_ids_sha256", "max_seq_length")):
                raise ValueError("Epoch-one tokenization/allocation differs from SFT test archive: " + model)
        for ss in (sources[model], dpo[model]):
            pin_tree(Path(ss["adapter"]), ss["adapter_files_sha256"], pins)
            base = Path(ss["base_snapshot"]) / "config.json"
            pins[str(base)] = digest(base)
    jar = repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    from src.utility.bioasq_official import _official_adapter_classes
    _official_adapter_classes(jar)
    pins[str(jar)] = digest(jar)
    submission.mkdir(parents=True, exist_ok=False)
    write_jsonl(submission / "test_inputs.jsonl", input_view(panel))
    write_jsonl(submission / "test_gold.jsonl", [gold[e["question_id"]] for e in panel])
    for model in MODELS:
        (submission / f"{model}_prompt.txt").write_text(sources[model]["system_prompt"], encoding="utf-8")
    for kind in ("generate", "score"):
        shutil.copy2(repository / f"gadi_sft_8b_starter/jobs/{kind}_compactness_test95.pbs", submission / f"{kind}.pbs")
    files = ["test_inputs.jsonl", "test_gold.jsonl", "generate.pbs", "score.pbs"]
    files += [f"{m}_prompt.txt" for m in MODELS]
    inference = snapshot_inference_code(submission)
    write(submission / "prepared.json", {
        "repository": str(repository), "comparison": str(comparison), "rerankers": str(rerankers),
        "selections": {m: str(p) for m, p in selections.items()}, "configurations": CONFIGURATIONS,
        "sources": sources, "dpo_sources": dpo, "baseline_runs": baseline, "input_plans": plans,
        "reranker_hashes": reranker_hashes, "selected_C": payloads["max5"]["selected_C"],
        "code_hashes": snapshot_code(repository, submission / "code"),
        "code_snapshot": inference, "worker_sha256": inference["files_sha256"]["scripts/run_single_answer_sampling.py"],
        "external_hashes": pins,
        "file_hashes": {n: digest(submission / n) for n in files},
        "question_count": 95, "generation_epochs": dict.fromkeys(MODELS, 1),
        "baseline_expected_metrics": {f"max5_{mode}": original["conditions"][f"max5_{mode}_round_robin"]["metrics"]
                                      for mode in MODES},
        "policy": "Fixed baseline/strict/all-epoch1 alternatives; no selection on test95; unchanged historical pooling and frozen max5 reranker",
    })
    print("Validated test95, all three epoch-one adapters, strict choices 0/1/0, exact SFT lineage and frozen reranker.", flush=True)
    print("Prepared three GPU smoke/full chains and one CPU scorer; six SFT archives reused.", flush=True)


def validate_prepared(submission):
    prepared = read(submission / "prepared.json")
    if prepared["configurations"] != CONFIGURATIONS or prepared["question_count"] != 95:
        raise ValueError("Test configuration changed")
    verify_files(submission, prepared["file_hashes"])
    verify_files(submission / "code", prepared["code_hashes"])
    for name, value in prepared["external_hashes"].items():
        if digest(Path(name)) != value:
            raise ValueError("Pinned test input changed: " + name)
    inference_worker(submission, prepared)
    return prepared


def generate(submission, model, output, smoke=False):
    prepared = validate_prepared(submission)
    if model not in MODELS:
        raise ValueError("Unknown model")
    source, plan = prepared["dpo_sources"][model], prepared["input_plans"][model]
    examples = records(submission / "test_inputs.jsonl")
    expected = (sorted(examples, key=lambda e: (-plan["expected_prompt_tokens"][e["question_id"]], e["question_id"]))[:4]
                if smoke else examples)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "model_key": model, "epoch": 1, "smoke_test": smoke,
             "expected_questions": len(expected), "submission": str(submission),
             "prepared_sha256": digest(submission / "prepared.json"), "source": source,
             "runs": {}, "arm_hashes": {}}
    write(output / "manifest.json", state)
    try:
        for mode in MODES:
            cfg = {"input": str(submission / "test_inputs.jsonl"), "prompt": str(submission / f"{model}_prompt.txt"),
                "response_mode": "equivalent_greedy" if mode == "greedy" else "equivalent_sampling",
                "num_generations": 1 if mode == "greedy" else 10,
                "temperature": 0 if mode == "greedy" else .8, "top_p": .95, "top_k": 0,
                "seed": 3407, "sample_seed_offset": 0, "max_new_tokens": 512, "max_candidates": 5,
                "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
                "execution_mode": source.get("execution_mode", "default"), "mark_snippets": False,
                "require_all_snippets": True, "max_seq_length": plan["max_seq_length"],
                "expected_prompt_tokens": plan["expected_prompt_tokens"],
                "expected_prompt_ids_sha256": plan["expected_prompt_ids_sha256"],
                "input_token_policy": "complete_evidence_auto_size", "prompt_version": "compactness-test95-epoch1-v1"}
            config = output / f"{mode}_config.json"; write(config, cfg)
            subprocess.run([sys.executable, str(inference_worker(submission, prepared)), "--config", str(config),
                "--model-name", source["adapter"], "--output-dir", str(output / mode),
                "--limit", str(len(expected)), "--preflight-all", *(["--longest-first"] if smoke else [])], check=True)
            rows = load_generation(output / mode, source, expected, mode)
            if smoke and not any(s["answers"] for r in rows for s in r["samples"]):
                raise ValueError("No parseable epoch-one smoke responses")
            state["runs"][mode] = mode
            state["arm_hashes"][mode] = {n: digest(output / mode / n) for n in RUN_FILES}
            write(output / "manifest.json", state)
        validate_prepared(submission)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


def record_jobs(submission):
    import csv
    with (submission / "jobs.tsv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["stage"] == "generation"]
    jobs = {r["model"]: r["job_id"] for r in rows}
    if len(rows) != 3 or set(jobs) != set(MODELS) or any(not re.fullmatch(r"\d+\.gadi-pbs", j) for j in jobs.values()):
        raise ValueError("Need exactly one full generation job for each model")
    write(submission / "generation_jobs.json", jobs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("prepare", "generate", "record"))
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--repository", type=Path, default=REPOSITORY)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--rerankers", type=Path)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--smoke-test", action="store_true")
    for m in MODELS:
        p.add_argument("--" + m + "-selection", type=Path)
    a = p.parse_args()
    sub = a.submission.resolve()
    if a.mode == "prepare":
        repository = a.repository.resolve(); starter = repository / "gadi_sft_8b_starter"
        selections = {m: (getattr(a, m + "_selection") or starter / "outputs/generator_feedback_dpo_expanded" /
            f"{m}-compactness-train-{SELECTION_JOBS[m][0]}.gadi-pbs" /
            f"validation-selection-{SELECTION_JOBS[m][1]}.gadi-pbs").resolve() for m in MODELS}
        prepare(repository,
            (a.comparison or starter / "outputs/sft_ensemble_test/comparison-181287698.gadi-pbs").resolve(),
            (a.rerankers or starter / "outputs/sft_ensemble_reranker/train-181350943.gadi-pbs").resolve(), selections, sub)
    elif a.mode == "record":
        record_jobs(sub)
    else:
        if not a.model:
            p.error("--model required")
        prepared = read(sub / "prepared.json")
        suffix = "smoke" if a.smoke_test else "generation"
        output = Path(prepared["repository"]) / "gadi_sft_8b_starter" / OUTPUT / f"{a.model}-{suffix}-{os.environ.get('PBS_JOBID', 'manual')}"
        generate(sub, a.model, output, a.smoke_test)
