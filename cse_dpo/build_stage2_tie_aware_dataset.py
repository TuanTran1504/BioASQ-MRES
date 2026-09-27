"""Build a Stage-2 dataset with explicit Davidson-DPO win/tie labels.

The labeling rule is intentionally conservative about normalization: it only
case-folds and collapses whitespace. Punctuation and hyphens remain meaningful.
Pairs sharing a literal token or a contained surface span remain clear boundary
preferences. Zero-token-overlap C3/C2 pairs are marked as semantic ties.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


STAGE_FILES = (
    "dpo_stage1_concept_learning_all_pairs.jsonl",
    "dpo_stage2_format_alignment_all_pairs.jsonl",
    "dpo_stage3_hierarchical_ranking_all_pairs.jsonl",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _unwrap_answer(value: str) -> str:
    value = str(value).strip()
    if value.casefold().startswith("[be]"):
        value = value[4:]
    if value.casefold().endswith("[ee]"):
        value = value[:-4]
    return " ".join(value.casefold().split())


def classify_pair(row: dict[str, Any]) -> tuple[str, str]:
    chosen = _unwrap_answer(row.get("chosen_candidate", row["chosen"]))
    rejected = _unwrap_answer(row.get("rejected_candidate", row["rejected"]))
    chosen_tokens = set(chosen.split())
    rejected_tokens = set(rejected.split())
    if chosen in rejected or rejected in chosen:
        return "win", "surface_containment"
    if chosen_tokens & rejected_tokens:
        return "win", "literal_token_overlap"
    return "tie", "zero_literal_token_overlap_semantic_equivalence"


def build(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("Output root must differ from input root")
    output_root.mkdir(parents=True, exist_ok=False)

    for filename in (STAGE_FILES[0], STAGE_FILES[2]):
        source = input_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)

    stage2_source = input_root / STAGE_FILES[1]
    rows = _jsonl(stage2_source)
    labeled: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for row in rows:
        label, relation = classify_pair(row)
        out = dict(row)
        out["preference_label"] = label
        out["tie_label_source"] = "weak_surface_relation_v1"
        out["tie_label_relation"] = relation
        labeled.append(out)
        counts[label] += 1
        relation_counts[relation] += 1

    stage2_output = output_root / STAGE_FILES[1]
    stage2_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labeled),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in labeled for key in row if key != "prompt"}) + ["prompt"]
    with (output_root / "dpo_stage2_format_alignment_all_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "stage2_pairs": len(labeled),
        "stage2_questions": len({row["question_id"] for row in labeled}),
        "preference_label_counts": dict(counts),
        "relation_counts": dict(relation_counts),
        "normalization": "Unicode case-fold plus whitespace collapse only; punctuation and hyphens preserved",
        "rule": {
            "win": "chosen/rejected surface containment or at least one identical whitespace token",
            "tie": "zero identical whitespace-token overlap between judge-confirmed semantic C3/C2 answers",
        },
        "review_csv": str((output_root / "dpo_stage2_format_alignment_all_pairs.csv").resolve()),
    }
    (output_root / "tie_aware_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
