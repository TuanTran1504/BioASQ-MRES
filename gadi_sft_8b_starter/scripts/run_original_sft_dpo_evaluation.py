#!/usr/bin/env python3
"""Paired original-SFT/DPO greedy one-answer development evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, read, write, digest, records
from run_expansion_dpo_8b import source_for
from matched_8b_provenance import validate_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("llama31", "qwen3", "ministral3"))
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    directory = args.training_run.resolve()
    training = read(directory / "manifest.json")
    default_path = ROOT / "configs" / ("original_sft_dpo_selection_8b.json"
        if training["configuration"].get("epoch_validation_mrr") else "original_sft_dpo_8b.json")
    config_path = (args.config or Path(training.get("dpo_config_path", default_path))).resolve()
    config = read(config_path)
    if (training["status"] != "complete" or training["smoke_test"] or training["phase"] != "save_adapter"
            or training["model_key"] != args.model or training["formulation"] != "original"
            or training["dpo_config_sha256"] != digest(config_path)):
        raise ValueError("Expected completed original-SFT DPO training")
    source = source_for(config, args.model)
    validate_provenance({"original": training["source"]}, {"original": source})
    for name, value in training["adapter_files_sha256"].items():
        if digest(directory / "adapter" / name) != value:
            raise ValueError("DPO adapter changed")
    strategy = training["pair_strategy"]
    arms = {"sft": source["adapter"], "dpo": str(directory / "adapter")}
    if config.get("epoch_validation_mrr"):
        from original_dpo_checkpoint_selection import choose_winners
        scores_path = directory / "epoch_selection_scores.json"
        if digest(scores_path) != training["epoch_selection_scores_sha256"]:
            raise ValueError("Epoch selection scores changed")
        winners = choose_winners(read(scores_path), int(config["num_train_epochs"]))
        arms = {"sft": source["adapter"]}
        for name, winner in winners.items():
            selected = training["selected_adapters"][name]
            adapter = Path(selected["path"]).resolve()
            if (not adapter.is_relative_to(directory) or selected["epoch"] != winner["epoch"]
                    or selected["global_step"] != winner["global_step"]):
                raise ValueError("Selected adapter does not match internal-validation scores")
            for filename, expected in selected["files_sha256"].items():
                if digest(adapter / filename) != expected:
                    raise ValueError("Selected adapter changed")
            arms["dpo_" + name] = str(adapter)
    output = ROOT / config.get("evaluation_output_root", "outputs/original_sft_dpo_evaluation") / f"{args.model}-{strategy}-evaluation-{os.environ.get('PBS_JOBID', 'manual')}"
    output.mkdir(parents=True, exist_ok=False)
    evaluation = read(ROOT / config["evaluation_config"])
    prompt_path = output / "original_prompt.txt"
    prompt_path.write_text(source["system_prompt"], encoding="utf-8")
    state = {"status": "running", "model_key": args.model, "pair_strategy": strategy, "smoke_test": False,
        "expected_questions": 160, "training_run": str(directory), "training_manifest_sha256": digest(directory / "manifest.json"),
        "provenance": source, "runs": {}, "arm_hashes": {},
        "checkpoint_selection": training["checkpoint_selection"], "selected_adapters": training.get("selected_adapters", {})}
    write(output / "manifest.json", state)
    try:
        generated = {}
        for name, adapter in arms.items():
            key = tuple(sorted((p.name, digest(p)) for p in Path(adapter).iterdir()
                               if p.is_file() and p.name != "training_complete.json"))
            if key in generated:
                prior = generated[key]
                state["runs"][name] = state["runs"][prior]
                state["arm_hashes"][name] = state["arm_hashes"][prior]
                state.setdefault("reused_generation_from", {})[name] = prior
                write(output / "manifest.json", state)
                continue
            settings = {"input": evaluation["input"], "prompt": str(prompt_path), "model_loader": source["model_loader"],
                "chat_template_kwargs": source["chat_template_kwargs"], "execution_mode": source["execution_mode"],
                "max_seq_length": config["max_seq_length"], "max_new_tokens": config["max_new_tokens"],
                "response_mode": "single_answer_greedy", "temperature": 0, "num_generations": 1,
                "require_all_snippets": True, "mark_snippets": False, "seed": config["seed"],
                "prompt_version": "original-sft-dpo-paired-v1"}
            path = output / f"{name}_config.json"
            write(path, settings)
            run = output / name
            subprocess.run([sys.executable, str(ROOT / "scripts/run_single_answer_sampling.py"),
                "--config", str(path), "--model-name", adapter, "--output-dir", str(run), "--limit", "160"], check=True, cwd=ROOT)
            if read(run / "status.json")["status"] != "complete" or len(records(run / "generations.jsonl")) != 160:
                raise ValueError("Incomplete paired generation")
            state["runs"][name] = name
            state["arm_hashes"][name] = {n: digest(run / n) for n in ("config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "status.json")}
            generated[key] = name
            write(output / "manifest.json", state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/analyze_original_sft_dpo.py"), str(output)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
