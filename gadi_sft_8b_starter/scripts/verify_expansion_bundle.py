#!/usr/bin/env python3
"""Validate the 8B expansion package without loading the model."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/extractive_expansion_8b.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["input"]
    prompt_path = ROOT / config["prompt"]
    runner_path = ROOT / "scripts/run_extractive_expansion_8b.py"
    for path in (data_path, prompt_path, runner_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_jsonl(data_path)
    expected = int(config["expected_questions"])
    if len(rows) != expected or len({row["question_id"] for row in rows}) != expected:
        raise ValueError(f"Expected {expected} unique questions; found {len(rows)} rows")
    required = {"question_id", "question", "snippets", "gold_aliases"}
    if any(set(row) != required or not row["snippets"] or not row["gold_aliases"] for row in rows):
        raise ValueError("Expansion rows do not have the expected nonempty fields")
    prompt = prompt_path.read_text(encoding="utf-8")
    response_mode = str(config.get("response_mode", "extractive"))
    if response_mode == "extractive":
        if "exact text spans" not in prompt or "up to TEN" not in prompt:
            raise ValueError("Unexpected extractive expansion prompt")
    elif response_mode == "equivalent":
        if "equivalent answer expressions" not in prompt or "relation_type" not in prompt:
            raise ValueError("Unexpected equivalent expansion prompt")
    else:
        raise ValueError("Unknown response_mode")
    print("Expansion questions:", len(rows))
    print("Total snippets:", sum(len(row["snippets"]) for row in rows))
    print("Data SHA256:", hashlib.sha256(data_path.read_bytes()).hexdigest())
    print("Prompt SHA256:", hashlib.sha256(prompt_path.read_bytes()).hexdigest())
    print("Response mode:", response_mode)
    print("Expansion bundle is valid.")


if __name__ == "__main__":
    main()
