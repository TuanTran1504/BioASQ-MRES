from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cse_dpo.common import load_json_records, write_json, write_jsonl
from src.utility.data import clean_text, truncate_text


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a stratified manual-review export from scored Cardinality Shortcut "
            "Study probe rows."
        )
    )
    parser.add_argument(
        "--scored-input",
        required=True,
        help="Scored probe JSONL produced by cardinality_shortcut_study_score_probe_bank.py.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where the audit sample JSONL will be written.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where the audit sample CSV will be written.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Where the audit selection summary JSON will be written.",
    )
    parser.add_argument(
        "--per-cell",
        type=int,
        default=4,
        help="Maximum audit records sampled from each probe_source x semantic_operation cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--max-evidence-items",
        type=int,
        default=2,
        help="How many evidence snippets to include in the preview field.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=320,
        help="Per-evidence-item truncation length for the preview field.",
    )
    return parser.parse_args()


def build_evidence_preview(
    evidence_items: Sequence[Any],
    *,
    max_items: int,
    max_chars: int,
) -> str:
    preview_items: list[str] = []
    for raw_value in evidence_items[: max(0, int(max_items))]:
        preview = truncate_text(clean_text(raw_value), max_chars=max_chars)
        if preview:
            preview_items.append(preview)
    return "\n\n".join(preview_items)


def choose_cell_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_cell: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_cell <= 0 or not rows:
        return []

    normalized_rows = [dict(row) for row in rows]
    sorted_by_boundary = sorted(
        normalized_rows,
        key=lambda row: (
            abs(float(row.get("avg_logprob_margin", 0.0) or 0.0)),
            clean_text(row.get("probe_id")),
        ),
    )
    boundary_count = min(len(sorted_by_boundary), max(1, per_cell // 2))
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    for row in sorted_by_boundary[:boundary_count]:
        record = dict(row)
        record["_audit_selection_reason"] = "closest_to_decision_boundary"
        selected.append(record)
        selected_ids.add(clean_text(record.get("probe_id")))

    remaining = [
        dict(row)
        for row in normalized_rows
        if clean_text(row.get("probe_id")) not in selected_ids
    ]
    rng = random.Random(int(seed))
    rng.shuffle(remaining)

    for row in remaining[: max(0, per_cell - len(selected))]:
        row["_audit_selection_reason"] = "random_within_cell"
        selected.append(row)

    return sorted(
        selected,
        key=lambda row: (
            clean_text(row.get("probe_source")),
            clean_text(row.get("semantic_operation")),
            clean_text(row.get("_audit_selection_reason")),
            abs(float(row.get("avg_logprob_margin", 0.0) or 0.0)),
            clean_text(row.get("probe_id")),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def main() -> None:
    args = parse_args()

    scored_input_path = resolve_project_path(str(args.scored_input))
    output_jsonl_path = resolve_project_path(str(args.output_jsonl))
    output_csv_path = resolve_project_path(str(args.output_csv))
    summary_json_path = resolve_project_path(str(args.summary_json))

    raw_rows = [dict(row) for row in load_json_records(scored_input_path)]
    scored_rows = [
        row
        for row in raw_rows
        if clean_text(row.get("scoring_status")) == "scored"
    ]

    rows_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        rows_by_cell[
            (
                clean_text(row.get("probe_source")) or "unknown",
                clean_text(row.get("semantic_operation")) or "unknown",
            )
        ].append(row)

    audit_rows: list[dict[str, Any]] = []
    selection_counts = Counter()
    for cell_index, ((probe_source, semantic_operation), group_rows) in enumerate(sorted(rows_by_cell.items()), start=1):
        selected_rows = choose_cell_rows(
            group_rows,
            per_cell=int(args.per_cell),
            seed=int(args.seed) + cell_index,
        )
        selection_counts[f"{probe_source}::{semantic_operation}"] = len(selected_rows)
        for row in selected_rows:
            evidence_items = row.get("evidence") if isinstance(row.get("evidence"), list) else []
            audit_rows.append(
                {
                    "audit_id": f"audit-{len(audit_rows) + 1:03d}",
                    "probe_id": clean_text(row.get("probe_id")),
                    "probe_source": probe_source,
                    "semantic_operation": semantic_operation,
                    "direction_label": clean_text(row.get("direction_label")),
                    "question_id": clean_text(row.get("question_id")),
                    "question_text": clean_text(row.get("question_text")),
                    "scoring_eos_treatment": clean_text(row.get("scoring_eos_treatment")),
                    "avg_logprob_margin": row.get("avg_logprob_margin"),
                    "avg_logprob_ranked_preferred": row.get("avg_logprob_ranked_preferred"),
                    "preferred_logp_mean": row.get("preferred_logp_mean"),
                    "dispreferred_logp_mean": row.get("dispreferred_logp_mean"),
                    "preferred_score_token_count": row.get("preferred_score_token_count"),
                    "dispreferred_score_token_count": row.get("dispreferred_score_token_count"),
                    "preferred_prompt_truncated": row.get("preferred_prompt_truncated"),
                    "dispreferred_prompt_truncated": row.get("dispreferred_prompt_truncated"),
                    "f1_margin": row.get("f1_margin"),
                    "changed_entity": clean_text(row.get("changed_entity")),
                    "preferred_answer": clean_text(row.get("preferred_answer")),
                    "dispreferred_answer": clean_text(row.get("dispreferred_answer")),
                    "preferred_items_json": json.dumps(row.get("preferred_items", []), ensure_ascii=False),
                    "dispreferred_items_json": json.dumps(row.get("dispreferred_items", []), ensure_ascii=False),
                    "evidence_preview": build_evidence_preview(
                        evidence_items,
                        max_items=int(args.max_evidence_items),
                        max_chars=int(args.max_evidence_chars),
                    ),
                    "audit_selection_reason": clean_text(row.get("_audit_selection_reason")),
                }
            )

    write_jsonl(output_jsonl_path, audit_rows)
    write_csv(output_csv_path, audit_rows)
    write_json(
        summary_json_path,
        {
            "study_name": "Cardinality Shortcut Study",
            "artifact_type": "probe_score_manual_audit_sample",
            "scored_input": str(scored_input_path),
            "output_jsonl": str(output_jsonl_path),
            "output_csv": str(output_csv_path),
            "total_scored_rows_available": len(scored_rows),
            "audit_row_count": len(audit_rows),
            "per_cell_target": int(args.per_cell),
            "seed": int(args.seed),
            "selection_counts_by_cell": dict(sorted(selection_counts.items())),
        },
    )

    print(f"Prepared {len(audit_rows)} manual-audit rows from {len(scored_rows)} scored probes.")
    print(f"Wrote audit JSONL to {output_jsonl_path}")
    print(f"Wrote audit CSV to {output_csv_path}")
    print(f"Wrote audit summary to {summary_json_path}")


if __name__ == "__main__":
    main()
