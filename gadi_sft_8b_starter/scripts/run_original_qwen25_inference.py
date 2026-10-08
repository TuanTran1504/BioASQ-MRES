#!/usr/bin/env python3
"""Validate original SFT identity, then run expansion and ten single-answer draws."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_adapter(size, adapter):
    pinned = read(ROOT / "configs/original_qwen25_adapters.json")[size]
    if not (adapter / "identity.json").is_file():
        raise FileNotFoundError(f"Transfer the historical adapter package first: {adapter}")
    if digest(adapter / "identity.json") != pinned["identity_sha256"]:
        raise ValueError("Historical adapter identity differs from the pinned export")
    identity = read(adapter / "identity.json")
    for name, expected in identity["files"].items():
        if Path(name).name != name or digest(adapter / name) != expected:
            raise ValueError(f"Historical adapter file differs: {name}")
    if identity["model_size"] != size or identity["historical_run"] != pinned["historical_run"]:
        raise ValueError("Wrong historical SFT run")
    config = read(adapter / "adapter_config.json")
    if config["base_model_name_or_path"] != pinned["base_model"]:
        raise ValueError("Historical backbone differs")
    if read(adapter / "training_complete.json")["status"] != "completed":
        raise ValueError("Historical training is incomplete")
    examples = [json.loads(line) for line in (ROOT / "data/expansion_dev160.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {row["question_id"] for row in examples}
    if len(ids) != 160 or len(examples) != 160 or ids & set(identity["train_question_ids"]):
        raise ValueError("Expected 160 unique dev questions disjoint from historical fitting data")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_expansion_bundle.py"),
                    "--config", str(ROOT / f"configs/equivalent_expansion_qwen25_{size}.json")], check=True)
    print(f"Validated {size}: exact historical adapter, no fitting/dev overlap; dev was used for checkpoint selection", flush=True)
    return pinned


def check_cache(base_model, download=False):
    if download:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(base_model, local_files_only=not download)
    except Exception as exc:
        raise RuntimeError(f"Historical backbone cache unavailable: {base_model}. "
                           "Run this helper with --mode prepare-cache on the login node before submission.") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("05b", "3b"), required=True)
    parser.add_argument("--mode", choices=("validate", "prepare-cache", "run"), default="validate")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    adapter = (args.adapter or ROOT / "outputs/original_sft" / args.model_size / "adapter").resolve()
    pinned = validate_adapter(args.model_size, adapter)
    check_cache(pinned["base_model"], download=args.mode == "prepare-cache")
    if args.mode != "run":
        return
    name = args.run_name or f"qwen25-{args.model_size}-original-sft-{os.environ.get('PBS_JOBID', 'manual')}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a directory name")
    output = ROOT / "outputs/original_qwen25" / name
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "model_size": args.model_size,
                "historical_run": pinned["historical_run"], "adapter": str(adapter),
                "identity_sha256": pinned["identity_sha256"], "smoke_test": args.smoke_test,
                "expected_questions": 4 if args.smoke_test else 160,
                "evaluation": "development-only; historical dev-selected checkpoint",
                "selection": "first five unique candidates in response/draw order", "runs": {}}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        parent = output / "expansion"
        parent.mkdir()
        command = [sys.executable, str(ROOT / "scripts/run_extractive_expansion_8b.py"),
                   "--config", str(ROOT / f"configs/equivalent_expansion_qwen25_{args.model_size}.json"),
                   "--model-name", str(adapter), "--output-parent", str(parent), "--run-name", name]
        if args.smoke_test:
            command.extend(["--limit", "4"])
        subprocess.run(command, check=True)
        children = list(parent.iterdir())
        if len(children) != 1:
            raise ValueError("Expected exactly one expansion run")
        manifest["runs"]["original_sft_expansion"] = children[0].relative_to(output).as_posix()
        save()
        sampling = output / "sampling10"
        subprocess.run([sys.executable, str(ROOT / "scripts/run_single_answer_sampling.py"),
                        "--config", str(ROOT / "configs/original_sft_sampling10.json"),
                        "--model-name", str(adapter), "--output-dir", str(sampling),
                        "--limit", str(manifest["expected_questions"])], check=True)
        manifest["runs"]["original_sft_sampling10"] = sampling.relative_to(output).as_posix()
        for relative in manifest["runs"].values():
            state = read(output / relative / "status.json")
            if state["status"] != "complete" or state["completed_questions"] != manifest["expected_questions"]:
                raise ValueError("Incomplete inference arm")
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    print(f"Original SFT inference complete: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
