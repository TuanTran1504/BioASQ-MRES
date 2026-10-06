#!/usr/bin/env python3
"""Export lightweight, source-text-free Gadi generation results for Git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "Artifacts/gadi_runs/model_comparison"
OUTPUT_ROOT = ROOT / "results/expansion_generations"

RUNS = {
    "qwen3_8b_base": "20261001-131348-qwen3-8b-equivalent-dev160-180287465-gadi-pbs",
    "gemma3_27b_base": "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs",
    "qwen3_8b_expansion_sft": "20261005-094031-qwen3-8b-expansion-sft-equivalent-dev160-180530756-gadi-pbs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_runs: dict[str, dict] = {}

    for label, run_id in RUNS.items():
        source = SOURCE_ROOT / run_id
        target = OUTPUT_ROOT / label
        required = [
            source / "status.json",
            source / "generations.jsonl",
            source / "candidates.jsonl",
            source / "invalid_candidates.jsonl",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing source files for {label}: {missing}")

        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("completed_questions") != 160:
            raise ValueError(f"Run is not a complete 160-question result: {run_id}")

        generations = read_jsonl(source / "generations.jsonl")
        if len(generations) != 160:
            raise ValueError(f"Expected 160 generations for {run_id}, found {len(generations)}")
        sanitized_generations = []
        for row in generations:
            sanitized = dict(row)
            sanitized.pop("question", None)
            sanitized_generations.append(sanitized)
        if any("question" in row for row in sanitized_generations):
            raise AssertionError(f"Question text survived sanitization for {run_id}")

        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "status.json", target / "status.json")
        shutil.copyfile(source / "candidates.jsonl", target / "candidates.jsonl")
        shutil.copyfile(
            source / "invalid_candidates.jsonl",
            target / "invalid_candidates.jsonl",
        )
        write_jsonl(target / "generations.jsonl", sanitized_generations)

        output_files = [
            target / "status.json",
            target / "generations.jsonl",
            target / "candidates.jsonl",
            target / "invalid_candidates.jsonl",
        ]
        manifest_runs[label] = {
            "source_run_id": run_id,
            "status": status,
            "source_sha256": {path.name: sha256(path) for path in required},
            "export_sha256": {path.name: sha256(path) for path in output_files},
            "generation_rows": len(sanitized_generations),
            "question_text_removed": True,
            "examples_file_distributed": False,
        }

    write_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": 1,
            "description": (
                "Portable Gadi equivalent-expansion generations. BioASQ question text, "
                "snippets, gold answers, model weights and checkpoints are excluded."
            ),
            "runs": manifest_runs,
        },
    )
    print(f"Exported {len(manifest_runs)} runs to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
