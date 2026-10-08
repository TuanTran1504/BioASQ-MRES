#!/usr/bin/env python3
"""Build and audit paired 8B SFT datasets without loading a model."""

import argparse
import hashlib
import json
from pathlib import Path

from run_expansion_sft_qwen3 import validate_rows

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/matched_8b_sft.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def original_rows(rows, prompt):
    result = []
    for row in rows:
        primary = json.loads(row["messages"][-1]["content"])["answers"][0]["answer"]
        if "[BE]" in primary or "[EE]" in primary:
            raise ValueError(f"{row['question_id']}: primary answer contains reserved delimiters")
        result.append({"question_id": row["question_id"], "messages": [
            {"role": "system", "content": prompt}, dict(row["messages"][1]),
            {"role": "assistant", "content": f"Answer: [BE]{primary}[EE]"}]})
    return result


def load_source(matrix):
    source = (ROOT / matrix["source_dataset"]).resolve()
    if digest(source / "manifest.json") != matrix["source_manifest_sha256"]:
        raise ValueError("Source manifest differs from the pinned teacher-validated dataset")
    manifest = read(source / "manifest.json")
    if manifest.get("official_test_overlap") != 0 or manifest.get("train_dev_overlap") != 0:
        raise ValueError("Source dataset must have audited zero dev/test overlap")
    rows = {}
    for split, count in (("train", 1296), ("validation", 144)):
        path = source / f"{split}.jsonl"
        if digest(path) != manifest["output_sha256"][path.name]:
            raise ValueError(f"Source hash mismatch: {path}")
        rows[split] = records(path)
        validate_rows(rows[split], split)
        if len(rows[split]) != count:
            raise ValueError(f"Expected {count} {split} questions")
    dev_path = ROOT / matrix["outer_dev"]
    dev = records(dev_path)
    ids = {name: {row["question_id"] for row in values} for name, values in {**rows, "dev": dev}.items()}
    if len(dev) != 160 or len(ids["dev"]) != 160 or any(
        ids[a] & ids[b] for a, b in (("train", "validation"), ("train", "dev"), ("validation", "dev"))):
        raise ValueError("Fitting, validation and dev must be disjoint with 160 unique dev questions")
    if digest(source / "dev_question_ids.json") != manifest["output_sha256"]["dev_question_ids.json"]:
        raise ValueError("Source dev ID manifest hash mismatch")
    if ids["dev"] != set(read(source / "dev_question_ids.json")):
        raise ValueError("Outer dev export differs from the source split")
    return source, rows, dev_path


def build(matrix):
    source, expanded, dev_path = load_source(matrix)
    destination = (ROOT / matrix["output_dataset"]).resolve()
    prompt_path = ROOT / matrix["single_answer_prompt"]
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    singles = {split: original_rows(rows, prompt) for split, rows in expanded.items()}
    for formulation, splits in (("original", singles), ("expansion", expanded)):
        path = destination / formulation
        path.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for split, rows in splits.items():
            validate_rows(rows, split, formulation)
            output = path / f"{split}.jsonl"
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
            if output.exists() and output.read_bytes() != payload:
                raise FileExistsError(f"Existing matched dataset differs: {output}")
            output.write_bytes(payload)
            hashes[output.name] = digest(output)
        write(path / "manifest.json", {
            "format_version": "bioasq-matched-8b-sft-v1", "formulation": formulation,
            "source_manifest_sha256": digest(source / "manifest.json"),
            "single_answer_prompt_sha256": digest(prompt_path), "outer_dev_sha256": digest(dev_path),
            "internal_train_questions": 1296, "internal_validation_questions": 144,
            "outer_dev_questions": 160, "official_test_overlap": 0, "train_dev_overlap": 0,
            "output_sha256": hashes,
            "target_policy": "one first official alias" if formulation == "original" else "unchanged validated expansion targets",
            "input_policy": "identical question/snippet JSON; formulation-specific system instructions"})
    validate(matrix)


def validate(matrix):
    source, expected, dev_path = load_source(matrix)
    destination = (ROOT / matrix["output_dataset"]).resolve()
    prompt_path = ROOT / matrix["single_answer_prompt"]
    expected_original = {split: original_rows(rows, prompt_path.read_text(encoding="utf-8").strip())
                         for split, rows in expected.items()}
    for formulation in ("original", "expansion"):
        path = destination / formulation
        manifest = read(path / "manifest.json")
        for field, actual in (("source_manifest_sha256", digest(source / "manifest.json")),
                              ("outer_dev_sha256", digest(dev_path)),
                              ("single_answer_prompt_sha256", digest(prompt_path))):
            if manifest[field] != actual:
                raise ValueError(f"Matched dataset provenance differs: {field}")
        for split in expected:
            data_path = path / f"{split}.jsonl"
            if digest(data_path) != manifest["output_sha256"][data_path.name]:
                raise ValueError(f"Matched data hash mismatch: {data_path}")
            rows = records(data_path)
            validate_rows(rows, split, formulation)
            reference = expected_original[split] if formulation == "original" else expected[split]
            if rows != reference:
                raise ValueError(f"Matched records differ from source: {formulation}/{split}")
    print("Validated paired 8B data: 1296 fitting / 144 validation / 160 disjoint dev; same evidence and primary answers")


def resolved_config(matrix, model_key, formulation):
    dataset = str(Path(matrix["output_dataset"]) / formulation)
    return {**matrix["common"], **matrix["models"][model_key], "formulation": formulation,
            "train_input": dataset + "/train.jsonl", "eval_input": dataset + "/validation.jsonl",
            "dataset_manifest": dataset + "/manifest.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("build", "validate", "pin-cache"), default="build")
    parser.add_argument("--pin-output", type=Path)
    parser.add_argument("--models", nargs="+", choices=("llama31", "qwen3", "ministral3"))
    args = parser.parse_args()
    matrix = read(args.config)
    if args.mode == "build":
        build(matrix)
    else:
        validate(matrix)
    if args.mode == "pin-cache":
        if not args.pin_output or args.pin_output.exists():
            raise ValueError("Choose a new --pin-output file for this submission")
        from huggingface_hub import snapshot_download
        pins = {}
        for key in args.models or matrix["models"]:
            model = matrix["models"][key]["model_name"]
            snapshot = Path(snapshot_download(model, local_files_only=True)).resolve()
            if not (snapshot / "config.json").is_file() or not list(snapshot.rglob("*.safetensors")):
                raise ValueError(f"Cached backbone files are incomplete: {model}; cache it on the login node first")
            for index in snapshot.rglob("*.safetensors.index.json"):
                missing = [name for name in set(read(index)["weight_map"].values())
                           if not (index.parent / name).is_file()]
                if missing:
                    raise ValueError(f"Cached backbone is missing weight shards: {model}: {missing}")
            pins[key] = {"model_name": model, "snapshot": str(snapshot), "revision": snapshot.name}
        args.pin_output.parent.mkdir(parents=True, exist_ok=True)
        write(args.pin_output, {"matrix_sha256": digest(args.config), "models": pins})
        print(f"Pinned shared base snapshots: {args.pin_output}")


if __name__ == "__main__":
    main()
