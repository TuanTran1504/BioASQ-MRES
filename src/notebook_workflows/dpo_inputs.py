"""Deterministic input builders used by the DPO workflow notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


STAGE1_FILE = "dpo_stage1_concept_learning_all_pairs.jsonl"
STAGE2_FILE = "dpo_stage2_format_alignment_all_pairs.jsonl"
STAGE3_FILE = "dpo_stage3_hierarchical_ranking_all_pairs.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in ("question_id", "prompt", "chosen", "rejected"))


def build_c3_c1_union(
    staged_roots: Sequence[str | Path],
    output_root: str | Path,
    *,
    source_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Combine stage-1 C3>C1 pairs and create an explicitly stage-1-only root.

    Duplicate prompt/chosen/rejected triples are retained once. Pair identifiers
    are rebuilt from content so colliding IDs from separate candidate banks remain
    unambiguous. Empty stage-2 and stage-3 files satisfy the staged trainer's input
    contract without allowing C2 pairs into the experiment.
    """
    roots = [Path(root).resolve() for root in staged_roots]
    labels = list(source_labels or [root.parent.name for root in roots])
    if not roots:
        raise ValueError("At least one staged root is required")
    if len(labels) != len(roots) or len(set(labels)) != len(labels):
        raise ValueError("source_labels must be unique and match staged_roots")

    combined: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    source_details: list[dict[str, Any]] = []
    input_rows = 0
    for label, root in zip(labels, roots):
        source_path = root / STAGE1_FILE
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _read_jsonl(source_path)
        source_details.append({
            "label": label,
            "path": str(source_path),
            "rows": len(rows),
            "sha256": _file_sha256(source_path),
        })
        input_rows += len(rows)
        for row in rows:
            if row.get("stage") != "concept_learning":
                raise ValueError(f"{source_path} contains a non-stage-1 row")
            if (row.get("chosen_class"), row.get("rejected_class")) != ("C3", "C1"):
                raise ValueError(f"{source_path} contains a pair other than C3>C1")
            key = _pair_key(row)
            if not all(key):
                raise ValueError(f"{source_path} contains an incomplete preference row")
            if key not in combined:
                item = dict(row)
                item["source_pair_ids"] = [str(row.get("pair_id", ""))]
                item["combined_sources"] = [label]
                combined[key] = item
            else:
                item = combined[key]
                item["source_pair_ids"] = sorted(set(item["source_pair_ids"] + [str(row.get("pair_id", ""))]))
                item["combined_sources"] = sorted(set(item["combined_sources"] + [label]))

    rows = []
    for key in sorted(combined):
        row = combined[key]
        identity = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        row["pair_id"] = f"{row['question_id']}::combined-c3c1-{hashlib.sha256(identity).hexdigest()[:16]}"
        rows.append(row)

    destination = Path(output_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stage1_path = destination / STAGE1_FILE
    stage1_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (destination / STAGE2_FILE).write_text("", encoding="utf-8")
    (destination / STAGE3_FILE).write_text("", encoding="utf-8")

    summary = {
        "description": "Deduplicated union of C3>C1 stage-1 pairs for shared 0.5B/3B DPO training",
        "sources": source_details,
        "input_rows": input_rows,
        "deduplicated_rows": len(rows),
        "duplicates_removed": input_rows - len(rows),
        "questions": len({str(row["question_id"]) for row in rows}),
        "stage1_file": str(stage1_path),
        "stage1_sha256": _file_sha256(stage1_path),
        "stage2_rows": 0,
        "stage3_rows": 0,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
