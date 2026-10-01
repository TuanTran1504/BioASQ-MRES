#!/usr/bin/env python3
"""Export the fixed 160-question expansion set into the portable Gadi bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_workflows.coverage_comparison import load_dev


DEFAULT_DATA_DIR = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "split_first_train90_dev10_supported_train_seed3407"
)
DEFAULT_OUTPUT = ROOT / "gadi_sft_8b_starter/data/expansion_dev160.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/training13b.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    examples = load_dev(
        args.data_dir / "dev_prepared.json",
        args.raw,
        train_path=args.data_dir / "train_questions.json",
        expected_count=160,
    )
    write_jsonl(args.output, examples)
    payload = args.output.read_bytes()
    manifest = {
        "question_count": len(examples),
        "unique_question_count": len({row["question_id"] for row in examples}),
        "snippet_count": sum(len(row["snippets"]) for row in examples),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "gold_usage": "Gold aliases are retained for scoring only and are never rendered into model prompts.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
