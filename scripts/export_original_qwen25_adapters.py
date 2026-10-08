#!/usr/bin/env python3
"""Package the exact historical SFT adapters for a separate transfer to Gadi."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"05b": "20260927-143545-67f7a338", "3b": "20260927-143545-3fca1c3d"}
FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja", "training_complete.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pinned = {}
    for size, run in RUNS.items():
        source = ROOT / "Artifacts/notebook_runs/sft/answer_sft" / run
        completion = json.loads((source / "adapter/training_complete.json").read_text(encoding="utf-8"))
        if completion["status"] != "completed":
            raise ValueError(f"Incomplete historical run: {run}")
        prepared = json.loads((source / "trainer_prepared/train_prepared.json").read_text(encoding="utf-8"))
        train_ids = sorted({row["id"].split("__supported_alias_", 1)[0] for row in prepared})
        target = ROOT / "Artifacts/gadi_transfer/original_sft" / size / "adapter"
        target.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            shutil.copy2(source / "adapter" / filename, target / filename)
        metadata = {"model_size": size, "historical_run": run,
                    "selection": completion["selection_metric"],
                    "train_question_ids": train_ids,
                    "development_used_for_checkpoint_selection": True,
                    "files": {name: digest(target / name) for name in FILES}}
        (target / "identity.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        pinned[size] = {"historical_run": run, "identity_sha256": digest(target / "identity.json"),
                        "base_model": json.loads((target / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]}
        print(f"{size}: {target}; {len(train_ids)} fitting question IDs")
    (ROOT / "gadi_sft_8b_starter/configs/original_qwen25_adapters.json").write_text(
        json.dumps(pinned, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
