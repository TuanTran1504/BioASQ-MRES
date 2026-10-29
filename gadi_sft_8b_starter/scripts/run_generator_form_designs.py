#!/usr/bin/env python3
"""Prepare/run native-source DPO pilots for two size-matched data designs."""
import argparse
from importlib.metadata import version
import math
import os
from pathlib import Path
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))
from scripts import prepare_generator_form_designs as designs
from scripts.prepare_reranker_guided_dpo import configure_data_root, load_inputs
from scripts.train_sft_ensemble_reranker import snapshot_code
from scripts.prepare_answer_selector_data import read, records, write, digest, verify_files
from scripts import generator_form_scorer_runtime as scorer_runtime


def preference_rows(bundle, model, design):
    if model not in designs.data.MODELS or design not in designs.DESIGNS:
        raise ValueError("Unknown target model or data design")
    train = bundle / (f"model_specific/{model}/train.jsonl" if design == "model_specific" else "shared/train.jsonl")
    return {"train": records(train), "validation": records(bundle / "inner_validation.jsonl")}


def verify_preferences_native(splits, native, fitting_ids):
    """Response authorship is preserved even when the target model differs."""
    seen = {}
    for split, rows in splits.items():
        seen[split] = {r["question_id"] for r in rows}
        if not seen[split] <= set(fitting_ids):
            raise ValueError("Preference outside fitting panel")
        for row in rows:
            for origin in row["origins"]:
                a, b = (native[origin[k]] for k in ("chosen_id", "rejected_id"))
                if (a["split"] != "train" or b["split"] != "train" or
                        any(r["model_key"] != origin["model_key"] or r["question_id"] != row["question_id"] for r in (a, b)) or
                        a["raw_response"] != row["chosen"] or b["raw_response"] != row["rejected"] or
                        a["prompt"] != row["prompt"] or b["prompt"] != row["prompt"]):
                    raise ValueError("Shared/native response provenance changed")
    if seen["train"] & seen["validation"]:
        raise ValueError("Preference question leakage")


def native_inputs(repository, bundle, model, design):
    designs.validate(bundle)
    provenance = read(bundle / "source_provenance.json")
    rerankers = Path(provenance["rerankers"])
    if digest(rerankers / "manifest.json") != provenance["reranker_manifest_sha256"]:
        raise ValueError("Frozen SFT reranker changed")
    audit_config = read(repository / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json")
    fitting, _, raw, _, reranker_hashes, pins, data = load_inputs(repository, rerankers, audit_config)
    for m, pin in pins.items():
        original = provenance["native_banks"][m]
        if (pin["path"] != original["directory"] or any(pin[k] != original[k] for k in ("manifest_sha256", "responses_sha256"))):
            raise ValueError("Native source bank differs from inspected origin")
    generation = read(data.DEFAULT_CONFIG)
    if generation != provenance["generation_config"]:
        raise ValueError("Original max-five generation configuration changed")
    splits = preference_rows(bundle, model, design)
    verify_preferences_native(splits, {r["response_id"]: r for r in raw}, fitting)
    source = data.selected_source(generation, model)
    banks = {m: {"manifest": read(Path(p["path"]) / "manifest.json")} for m, p in pins.items()}
    from matched_8b_provenance import validate_provenance
    validate_provenance({"expansion": banks[model]["manifest"]["source"]}, {"expansion": source})
    feedback = {"banks": pins, "rerankers": str(rerankers), "reranker_hashes": reranker_hashes}
    manifest = {"banks": banks, "pair_counts": {s: len(r) for s, r in splits.items()}}
    return manifest, splits, source, feedback


def preflight(tokenizer, splits, source, cap=16384):
    from run_expansion_dpo_8b import render_preference
    maxima = {}
    for name, rows in splits.items():
        maxima[name] = max(max(len(encoded[k + "_input_ids"]) for k in ("chosen", "rejected"))
                           for encoded in (render_preference(tokenizer, r, source["chat_template_kwargs"], cap) for r in rows))
    return maxima


def scorer_source(bundle):
    """Bind inference labels to the exact JAR used for the audited preferences."""
    return scorer_runtime.source(bundle)


def snapshot(repository, destination):
    hashes = snapshot_code(repository, destination)
    for path in sorted((repository / "gadi_sft_8b_starter/src").rglob("*.py")):
        relative = path.relative_to(repository)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[relative.as_posix()] = digest(target)
    return hashes


def prepare(repository, bundle, submission):
    if submission.exists():
        raise FileExistsError(submission)
    state = designs.validate(bundle)
    expected = read(repository / "results/generator_form_full_inspection_20261029.json")["full_export_manifest_sha256"]
    if state["inspection_manifest_sha256"] != expected:
        raise ValueError("Wrong inspected preference source")
    expected_package = read(repository / "results/generator_form_designs_20261029.json")["bundle_manifest_sha256"]
    if digest(bundle / "manifest.json") != expected_package:
        raise ValueError("Wrong size-matched reviewed design package")
    protocol = read(bundle / "training_protocol.json")
    jar = scorer_source(bundle)
    scorer_runtime.smoke(jar)
    if (protocol["training_pairs"] != 132 or protocol["num_train_epochs"] != 3 or
            protocol["gradient_accumulation_steps"] != 4 or protocol["review_scope"] != "official_form"):
        raise ValueError("Wrong size-matched six-run protocol")
    from run_base_original_prompt import load_text_tokenizer, native_context, snapshot_inference_code
    from run_expansion_dpo_8b import training_schedule
    prepared_runs, sources, maximum = {}, {}, protocol["max_seq_length"]
    for model in designs.data.MODELS:
        print(f"Validating native sources and complete target-tokenizer inputs: {model}", flush=True)
        for design in designs.DESIGNS:
            manifest, splits, source, feedback = native_inputs(repository, bundle, model, design)
            if model not in sources:
                sources[model] = source
                tokenizer = load_text_tokenizer(source["adapter"])
            if source != sources[model]:
                raise ValueError("Starting SFT differs between designs")
            lengths = preflight(tokenizer, splits, source, min(16384, native_context(source["base_snapshot"])))
            maximum = max(maximum, *lengths.values())
            if len(splits["train"]) != 132 or len(splits["validation"]) != 96:
                raise ValueError("Size/common preference holdout mismatch")
            if len({r["question_id"] for r in splits["train"]}) < 64:
                raise ValueError("Insufficient distinct fitting questions")
            prepared_runs[model, design] = (manifest, source, feedback, lengths)
    maximum = math.ceil(maximum / 256) * 256
    if any(maximum > native_context(s["base_snapshot"]) for s in sources.values()):
        raise ValueError("Common complete-evidence allocation exceeds a backbone's native context")
    submission.mkdir(parents=True)
    children = []
    for (model, design), (manifest, source, feedback, lengths) in prepared_runs.items():
        child = submission / model / design
        child.mkdir(parents=True)
        code = snapshot(repository, child / "code")
        inference = snapshot_inference_code(child)
        config = {**protocol, "model_key": model, "design": design, "max_seq_length": maximum}
        for stage in ("train", "validate"):
            shutil.copy2(repository / f"gadi_sft_8b_starter/jobs/{stage}_generator_form_designs.pbs", child / f"{stage}.pbs")
        runtime_hashes = scorer_runtime.freeze(jar, child)
        write(child / "prepared.json", {"repository": str(repository), "bundle": str(bundle), "preferences": str(bundle),
            "bundle_manifest_sha256": digest(bundle / "manifest.json"), "configuration": config,
            "source": source, "feedback": feedback, "pair_counts": manifest["pair_counts"],
            "native_banks": feedback["banks"],
            "preflight_max_tokens": lengths, "code_hashes": code, "code_snapshot": inference,
            "worker_sha256": digest(child / inference["directory"] / "scripts/run_single_answer_sampling.py"),
            "scorer_relative": "scorer/BioASQEvaluation.jar",
            "job_hashes": {**{name: digest(child / name) for name in ("train.pbs", "validate.pbs")}, **runtime_hashes}})
        children.append(str(child))
    code = snapshot(repository, submission / "code")
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/compare_generator_form_designs.pbs", submission / "compare.pbs")
    runtime_hashes = scorer_runtime.freeze(jar, submission)
    write(submission / "prepared.json", {"repository": str(repository), "bundle": str(bundle),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"), "children": children,
        "child_prepared_sha256": {p: digest(Path(p) / "prepared.json") for p in children},
        "code_hashes": code, "common_context_allocation": maximum,
        "scorer_relative": "scorer/BioASQEvaluation.jar", "scorer_files_sha256": runtime_hashes,
        "scorer_sha256": digest(submission / "scorer/BioASQEvaluation.jar"),
        "planned_steps_per_run": training_schedule(protocol, 132)[0],
        "comparison_job_sha256": digest(submission / "compare.pbs")})
    print(f"Prepared six DPO pilots: 132 train / 96 common preference holdout; 99 updates each; common context {maximum}.", flush=True)


def validate_child(submission):
    p = read(submission / "prepared.json")
    verify_files(submission / "code", p["code_hashes"])
    verify_files(submission, p["job_hashes"])
    if digest(Path(p["bundle"]) / "manifest.json") != p["bundle_manifest_sha256"]:
        raise ValueError("Design package changed")
    manifest, splits, source, feedback = native_inputs(Path(p["repository"]), Path(p["bundle"]),
        p["configuration"]["model_key"], p["configuration"]["design"])
    if source != p["source"] or feedback != p["feedback"] or manifest["pair_counts"] != p["pair_counts"]:
        raise ValueError("Native source or fixed reranker changed after preparation")
    return p, manifest, splits, source


def run(submission, output, smoke=False):
    if output.exists():
        raise FileExistsError(output)
    p, manifest, splits, source = validate_child(submission)
    config = p["configuration"]
    output.mkdir(parents=True)
    state = {"status": "running", "phase": "train", "model_key": config["model_key"], "design": config["design"],
        "source": source, "configuration": config, "smoke_test": smoke, "full_dataset": True,
        "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
        "objective": config["objective"], "package_versions": {n: version(n) for n in ("torch", "transformers", "peft", "unsloth")}}
    write(output / "manifest.json", state)
    try:
        from run_generator_feedback_dpo import activate_portable_helpers
        import run_expansion_dpo_8b as worker
        activate_portable_helpers()
        args = argparse.Namespace(preferences=Path(p["bundle"]), model=config["model_key"], smoke_test=smoke)
        worker.train(args, config, source, output, state, preference_validator=lambda *_: (manifest, splits))
        validate_child(submission)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    a = parser.parse_args()
    if a.mode == "prepare":
        if not a.bundle: parser.error("--bundle is required")
        prepare(a.repository.resolve(), a.bundle.resolve(), a.submission.resolve())
    else:
        run(a.submission.resolve(), a.submission.resolve() / f"{'smoke' if a.smoke_test else 'train'}-{os.environ.get('PBS_JOBID','manual')}", a.smoke_test)
