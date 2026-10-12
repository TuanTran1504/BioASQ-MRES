#!/usr/bin/env python3
"""Evaluate base and original SFT with identical tagged prompts and full evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT, read, write, digest, records
from run_matched_8b_evaluation import DEFAULT_CONFIG, arm_config, validate_training_pair
from matched_8b_provenance import validate_provenance
from run_extractive_expansion_8b import render_prompt, tokenize_text, input_token_count


def native_context(snapshot):
    saved = read(Path(snapshot) / "config.json")
    text = saved.get("text_config", saved)
    size = text.get("max_position_embeddings")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Cannot establish the backbone's native context window")
    return size


def full_input_plan(source, examples, output_tokens, tokenizer_loader=None):
    """Size allocation to actual complete prompts, not a user-imposed input cap."""
    if tokenizer_loader is None:
        tokenizer_loader = load_text_tokenizer
    counts, encoded_by_arm = {}, {}
    for name, path in (("base", source["base_snapshot"]), ("sft", source["adapter"])):
        tokenizer = tokenizer_loader(path)
        encoded_by_arm[name] = []
        counts[name] = {}
        for row in examples:
            prompt = render_prompt(tokenizer, source["system_prompt"], row, source["chat_template_kwargs"])
            encoded = tokenize_text(tokenizer, prompt, add_special_tokens=True)
            encoded_by_arm[name].append(encoded["input_ids"])
            counts[name][row["question_id"]] = input_token_count(encoded)
    if encoded_by_arm["base"] != encoded_by_arm["sft"]:
        raise ValueError("Base and SFT native templates/tokenizers produce different input tokens")
    longest = max(counts["base"].values())
    required = longest + output_tokens
    native = native_context(source["base_snapshot"])
    if required > native:
        raise ValueError(f"Complete evidence needs {required} tokens but native context is {native}; no truncation allowed")
    return {"policy": "complete evidence; allocation sized to longest actual prompt plus output allowance",
            "native_context_tokens": native, "max_prompt_tokens": longest,
            "max_seq_length": required, "expected_prompt_tokens": counts["base"],
            "expected_prompt_ids_sha256": {row["question_id"]: hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                for row, ids in zip(examples, encoded_by_arm["base"])}}


def load_text_tokenizer(path):
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        # Ministral's base snapshot can keep its native template on a processor.
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        template = getattr(processor, "chat_template", None)
        if not template:
            raise ValueError("Checkpoint has no native text chat template")
        tokenizer.chat_template = template
    return tokenizer


def prepare(config_path, destination, models):
    if not models or len(set(models)) != len(models):
        raise ValueError("Specify unique model keys")
    config = read(config_path)
    examples = records(ROOT / config["input"])
    sources = {}
    for model in models:
        source = validate_training_pair(config, model)["original"]
        plan = full_input_plan(source, examples, config["max_new_tokens"])
        sources[model] = {"provenance": source, "input_plan": plan}
        print(f"{model}: complete inputs, longest={plan['max_prompt_tokens']} tokens; no evidence truncation", flush=True)
    sys.path.insert(0, str(ROOT.parent))
    from src.utility.bioasq_official import _conda_executable, _official_adapter_classes
    subprocess.run([*_conda_executable("java"), "-version"], check=True)
    _official_adapter_classes(ROOT.parent / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "manifest.json", {"ready": True, "config": str(config_path.resolve()),
        "config_sha256": digest(config_path), "sources": sources,
        "worker_sha256": digest(ROOT / "scripts/run_single_answer_sampling.py")})


def validate_submission(submission, model):
    prepared = read(submission / "manifest.json")
    if not prepared["ready"] or digest(Path(prepared["config"])) != prepared["config_sha256"]:
        raise ValueError("Evaluation configuration changed since preparation")
    if digest(ROOT / "scripts/run_single_answer_sampling.py") != prepared["worker_sha256"]:
        raise ValueError("Inference worker changed since preparation")
    config = read(Path(prepared["config"]))
    saved = prepared["sources"][model]
    current = validate_training_pair(config, model)["original"]
    validate_provenance({"original": saved["provenance"]}, {"original": current})
    return prepared, config, saved


def run(submission, model, output, smoke):
    prepared, evaluation, saved = validate_submission(submission, model)
    source, plan = saved["provenance"], saved["input_plan"]
    output.mkdir(parents=True, exist_ok=False)
    prompt = output / "original_prompt.txt"
    prompt.write_bytes(source["system_prompt"].encode("utf-8"))
    settings = arm_config(evaluation, {"original": source}, "original_greedy", prompt)
    settings.update(max_seq_length=plan["max_seq_length"], expected_prompt_tokens=plan["expected_prompt_tokens"],
                    expected_prompt_ids_sha256=plan["expected_prompt_ids_sha256"],
                    input_token_policy="complete_evidence_auto_size")
    count = 4 if smoke else 160
    state = {"status": "running", "model_key": model, "smoke_test": smoke, "expected_questions": count,
        "provenance": source, "input_plan": plan, "settings": settings,
        "input_sha256": evaluation["input_sha256"], "prompt_sha256": digest(prompt),
        "prepared_manifest_sha256": digest(submission / "manifest.json"),
        "runs": {}, "arm_hashes": {}, "evaluation": "Exploratory paired base/original SFT dev comparison; full evidence"}
    write(output / "manifest.json", state)
    try:
        for name, target in (("base", source["base_snapshot"]), ("sft", source["adapter"])):
            path = output / f"{name}_config.json"
            write(path, settings)
            directory = output / name
            subprocess.run([sys.executable, str(ROOT / "scripts/run_single_answer_sampling.py"),
                "--config", str(path), "--model-name", target, "--output-dir", str(directory),
                "--limit", str(count), "--preflight-all"], check=True, cwd=ROOT)
            rows = records(directory / "generations.jsonl")
            if read(directory / "status.json")["status"] != "complete" or len(rows) != count:
                raise ValueError("Incomplete generation")
            if smoke and not any(row["samples"][0]["answer"] for row in rows):
                raise ValueError(f"{name}: no parseable tagged answers in smoke; inspect raw responses")
            state["runs"][name] = name
            state["arm_hashes"][name] = {n: digest(directory / n) for n in (
                "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "status.json")}
            write(output / "manifest.json", state)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)
    subprocess.run([sys.executable, str(ROOT.parent / "scripts/analyze_base_original_prompt.py"), str(output),
                    *(["--allow-smoke"] if smoke else [])], check=True, cwd=ROOT)


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
        prepare(args.config, args.submission_dir.resolve(), args.models)
    else:
        if not args.model:
            parser.error("--model is required for run")
        suffix = "-smoke" if args.smoke_test else ""
        output = ROOT / "outputs/base_original_prompt" / f"{args.model}-base-original-{os.environ.get('PBS_JOBID', 'manual')}{suffix}"
        run(args.submission_dir.resolve(), args.model, output, args.smoke_test)


if __name__ == "__main__":
    main()
