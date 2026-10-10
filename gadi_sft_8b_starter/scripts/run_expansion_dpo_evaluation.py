#!/usr/bin/env python3
"""Evaluate each expansion SFT policy and its DPO continuation at equal budgets."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from expansion_dpo_data import DEFAULT_CONFIG as DPO_CONFIG, validate_pairs
from matched_8b_provenance import validate_provenance
from prepare_matched_8b_sft import ROOT, digest, read, records, write
from run_matched_8b_evaluation import arm_config, validate_training_pair
from run_original_qwen25_inference import validate_smoke_outputs

DEFAULT_CONFIG = ROOT / "configs/expansion_dpo_evaluation.json"
CONDITIONS = ("sft_greedy", "sft_sampling10", "dpo_greedy", "dpo_sampling10")


def validate_sources(config, key):
    evaluation = read(ROOT / config["sft_evaluation_config"])
    sft = validate_training_pair(evaluation, key)["expansion"]
    if (ROOT / config["dpo_config"]).resolve() != DPO_CONFIG.resolve():
        raise ValueError("Use the pinned DPO training configuration")
    preferences = ROOT / config["preferences"]
    preference_manifest, pairs = validate_pairs(preferences, read(DPO_CONFIG))
    dev_ids = {r["question_id"] for r in records(ROOT / evaluation["input"])}
    if any(r["question_id"] in dev_ids for rows in pairs.values() for r in rows):
        raise ValueError("DPO preferences overlap evaluation questions")
    validate_provenance({"expansion": preference_manifest["banks"][key]["manifest"]["source"]},
                        {"expansion": sft})
    directory = (ROOT / config["models"][key]).resolve()
    state = read(directory / "manifest.json")
    expected = {"status": "complete", "phase": "save_adapter", "model_key": key,
                "smoke_test": False, "dpo_config_sha256": digest(DPO_CONFIG),
                "preferences_manifest_sha256": digest(preferences / "manifest.json"),
                "preference_pair_counts": preference_manifest["pair_counts"],
                "trained_pair_counts": preference_manifest["pair_counts"],
                "reference": "precomputed initial expansion SFT completion log probabilities"}
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("DPO run is incomplete or differs from the reviewed preferences/configuration")
    validate_provenance({"expansion": state["source"]}, {"expansion": sft})
    if not state.get("global_steps", 0) > 0:
        raise ValueError("DPO completed no optimisation updates")
    if digest(directory / "reference_logps.jsonl") != state["reference_logps_sha256"]:
        raise ValueError("DPO reference likelihood archive changed")
    adapter = directory / "adapter"
    files = {p.name: digest(p) for p in adapter.iterdir() if p.is_file()}
    if files != state["adapter_files_sha256"]:
        raise ValueError("Saved DPO adapter files changed")
    for filename in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if not (adapter / filename).is_file() or not (adapter / filename).stat().st_size:
            raise ValueError("Missing or empty DPO adapter file: " + filename)
    if read(adapter / "adapter_config.json")["base_model_name_or_path"] != sft["base_snapshot"]:
        raise ValueError("DPO adapter backbone differs from its SFT reference")
    dpo = {**sft, "training_run": directory.name, "adapter": str(adapter),
           "adapter_files_sha256": files, "training_manifest_sha256": digest(directory / "manifest.json")}
    dpo.pop("status_sha256", None)
    print(f"Validated {key}: complete SFT/DPO adapters; shared reviewed preferences; disjoint dev", flush=True)
    return evaluation, {"sft": sft, "dpo": dpo}, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, choices=("llama31", "qwen3", "ministral3"))
    parser.add_argument("--mode", choices=("validate", "run"), default="validate")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config = read(args.config)
    evaluation, sources, training = validate_sources(config, args.model)
    if args.mode == "validate":
        return
    name = args.run_name or f"{args.model}-dpo-evaluation-{os.environ.get('PBS_JOBID', 'manual')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a directory name")
    output = ROOT / config["output_root"] / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "model_key": args.model, "smoke_test": args.smoke_test,
                "expected_questions": 4 if args.smoke_test else 160, "runs": {}, "provenance": sources,
                "input_sha256": evaluation["input_sha256"], "evaluation_config_sha256": digest(args.config),
                "evaluation": "development-only expansion SFT versus DPO pilot",
                "selection": "first five unique candidates in draw then within-response order",
                "dpo_training_manifest": training}
    write(output / "manifest.json", manifest)
    prompt = output / "expansion_prompt.txt"
    prompt.write_text(sources["sft"]["system_prompt"], encoding="utf-8")
    try:
        for condition in CONDITIONS:
            policy, inference = condition.split("_", 1)
            source = sources[policy]
            arm = arm_config(evaluation, {"expansion": source}, "expansion_" + inference, prompt)
            path = output / (condition + "_config.json")
            write(path, arm)
            greedy = inference == "greedy"
            command = [sys.executable, str(ROOT / "scripts" / (
                "run_extractive_expansion_8b.py" if greedy else "run_single_answer_sampling.py")),
                "--config", str(path), "--model-name", source["adapter"],
                "--limit", str(manifest["expected_questions"])]
            target = output / condition
            if greedy:
                target.mkdir()
                command.extend(["--output-parent", str(target), "--run-name", name + "-" + condition])
            else:
                command.extend(["--output-dir", str(target)])
            subprocess.run(command, check=True)
            if greedy:
                children = list(target.iterdir())
                if len(children) != 1:
                    raise ValueError("Expected one greedy run")
                target = children[0]
            status = read(target / "status.json")
            if (status["status"] != "complete" or status["completed_questions"] != manifest["expected_questions"]
                    or len(records(target / "generations.jsonl")) != manifest["expected_questions"]):
                raise ValueError("Incomplete inference arm: " + condition)
            manifest["runs"][condition] = target.relative_to(output).as_posix()
            write(output / "manifest.json", manifest)
        if args.smoke_test:
            validate_smoke_outputs({k: output / v for k, v in manifest["runs"].items()})
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", manifest)
    print(f"Expansion DPO evaluation complete: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
