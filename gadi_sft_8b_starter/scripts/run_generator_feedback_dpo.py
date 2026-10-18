#!/usr/bin/env python3
"""Continue exact max-five Llama SFT on reviewed compactness preferences."""
import argparse
from importlib.metadata import version
from pathlib import Path
import os
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(1, str(REPOSITORY / "gadi_sft_8b_starter/scripts"))

from prepare_matched_8b_sft import read, records, digest, write
from scripts.prepare_reranker_guided_dpo import configure_data_root
from scripts.train_sft_ensemble_reranker import snapshot_code
from scripts.ablate_reranker_relation_labels import verify_files
import generator_feedback_dpo_data as preferences


def validate_native(directory, config, repository):
    """Link reviewed strings back to every pinned native response bank."""
    manifest, splits = preferences.validate(directory, config)
    data = configure_data_root(repository)
    feedback = read(directory / "source/manifest.json")
    prepared_path = Path(feedback["submission"]) / "prepared.json"
    if digest(prepared_path) != feedback["prepared_sha256"]:
        raise ValueError("Original feedback preparation changed")
    prepared = read(prepared_path)
    generation = read(data.DEFAULT_CONFIG)
    paths = [Path(prepared["banks"][m]["path"]) for m in data.MODELS]
    banks, raw = data.pack_banks(paths, generation)
    for model, path in zip(data.MODELS, paths):
        saved = prepared["banks"][model]
        if digest(path / "manifest.json") != saved["manifest_sha256"] or digest(path / "responses.jsonl") != saved["responses_sha256"]:
            raise ValueError("Original max-five bank changed")
    native = {r["response_id"]: r for r in raw}
    for row in records(directory / "annotations.jsonl"):
        if row["response_id"] not in native or any(row.get(k) != v for k, v in native[row["response_id"]].items()):
            raise ValueError("Reviewed completion differs from its original native bank")
    source = data.selected_source(generation, "llama31")
    from matched_8b_provenance import validate_provenance
    validate_provenance({"expansion": banks["llama31"]["manifest"]["source"]}, {"expansion": source})
    from run_expansion_dpo_8b import training_schedule
    training_schedule(config, len(splits["train"]))
    # Existing trainer requires this explicit source provenance in its callback.
    manifest = {**manifest, "banks": banks}
    return manifest, splits, source, prepared


def prepare(repository, directory, submission, config_path=None):
    config_path = config_path or repository / "gadi_sft_8b_starter/configs/generator_feedback_dpo.json"
    config = read(config_path)
    _, splits, source, feedback_prepared = validate_native(directory, config, repository)
    submission.mkdir(parents=True, exist_ok=False)
    code = snapshot_code(repository, submission / "code")
    # The GPU trainer uses the starter's portable src helpers, rather than the
    # repository's evaluation src package. Pin those helpers too.
    for path in sorted((repository / "gadi_sft_8b_starter/src").rglob("*.py")):
        relative = path.relative_to(repository)
        target = submission / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        code[relative.as_posix()] = digest(target)
    shutil.copy2(repository / "gadi_sft_8b_starter/jobs/train_generator_feedback_dpo.pbs", submission / "job.pbs")
    write(submission / "prepared.json", {"repository": str(repository), "preferences": str(directory),
          "preferences_manifest_sha256": digest(directory / "manifest.json"), "configuration": config,
          "source": source, "code_hashes": code, "pair_counts": {s: len(r) for s, r in splits.items()},
          "native_banks": feedback_prepared["banks"], "job_sha256": digest(submission / "job.pbs")})
    print(f"Validated native max-five Llama source and reviewed preferences: { {s: len(r) for s, r in splits.items()} }", flush=True)


def activate_portable_helpers():
    # Validation imports the evaluation src namespace. Extend its package paths
    # before training so Python can resolve the pinned portable CUDA helpers.
    import src
    import src.utility
    portable = REPOSITORY / "gadi_sft_8b_starter/src"
    src.__path__ = [str(portable), *src.__path__]
    src.utility.__path__ = [str(portable / "utility"), *src.utility.__path__]
    for name in ("training_execution", "text_tokenizer", "adapter_save"):
        sys.modules.pop("src.utility." + name, None)


def run(submission, output, smoke=False):
    if output.exists():
        raise FileExistsError(output)
    prepared = read(submission / "prepared.json")
    verify_files(submission / "code", prepared["code_hashes"])
    directory, repository = Path(prepared["preferences"]), Path(prepared["repository"])
    if digest(directory / "manifest.json") != prepared["preferences_manifest_sha256"]:
        raise ValueError("Reviewed preference manifest changed")
    config = prepared["configuration"]
    manifest, splits, source, feedback = validate_native(directory, config, repository)
    if source != prepared["source"] or feedback["banks"] != prepared["native_banks"]:
        raise ValueError("Training source changed since submission")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "phase": "train", "model_key": "llama31", "formulation": "expansion",
             "source": source, "smoke_test": smoke, "full_dataset": True,
             "configuration": config, "submission": str(submission),
             "prepared_sha256": digest(submission / "prepared.json"),
             "package_versions": {n: version(n) for n in ("torch", "transformers", "peft", "unsloth")},
             "objective": "compactness with retention controls; frozen-reranker feedback"}
    write(output / "manifest.json", state)
    try:
        import run_expansion_dpo_8b as worker
        activate_portable_helpers()
        args = argparse.Namespace(preferences=directory, model="llama31", smoke_test=smoke)
        worker.train(args, config, source, output, state,
                     preference_validator=lambda path, cfg: (manifest, splits))
        # Detect edits while the GPU job was running.
        if digest(directory / "manifest.json") != prepared["preferences_manifest_sha256"]:
            raise ValueError("Reviewed preferences changed during training")
        preferences.validate(directory, config)
        verify_files(submission / "code", prepared["code_hashes"])
        verify_files(Path(source["adapter"]), source["adapter_files_sha256"])
        for bank in prepared["native_banks"].values():
            verify_files(Path(bank["path"]), {"manifest.json": bank["manifest_sha256"], "responses.jsonl": bank["responses_sha256"]})
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "run"), required=True)
    p.add_argument("--repository", type=Path, default=REPOSITORY)
    p.add_argument("--preferences", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.mode == "prepare":
        if args.preferences is None:
            p.error("Preparation requires --preferences")
        prepare(args.repository.resolve(), args.preferences.resolve(), args.submission.resolve(), args.config.resolve() if args.config else None)
    else:
        prepared = read(args.submission / "prepared.json")
        directory = Path(prepared["repository"]) / "gadi_sft_8b_starter" / prepared["configuration"]["output_root"]
        name = f"llama31-compactness-{'smoke' if args.smoke_test else 'train'}-{os.environ.get('PBS_JOBID', 'manual')}"
        run(args.submission.resolve(), directory / name, args.smoke_test)
