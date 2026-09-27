#!/usr/bin/env python
"""Annotate the remaining train candidate-bank questions and merge with first-400 labels.

This script continues the candidate-bank LLM judge workflow after the existing first
400 train questions. It annotates candidates from both 0.5B and 3B banks in
question-level chunks, then rebuilds merged class slates, DPO pairs, and the three
curriculum stage files.

C3 handling is deterministic: generated exact matches are not used as the C3 anchor
when building pairs. Instead, each question gets its C3 directly from its BioASQ gold
aliases. The LLM judge is used only for non-exact candidate answers, which are
classified as C2 or C1, with legacy C0 collapsed into C1 by the shared judge module.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cse_dpo.candidate_bank_class_judge import (
    CandidateBankClassJudge,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SYSTEM,
    build_pairs,
    digest,
    extract_snippets,
    extractive_evidence_ids,
    first_extractive_alias,
    format_answer,
    prepare_records,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK_ROOT = ROOT / "Artifacts/cse_dpo/candidate_banks/strict_extractive_qwen25_05b_3b_all_splits_s20_v2_seed3407"
DEFAULT_BANK_FILES = {
    "05b": DEFAULT_BANK_ROOT / "qwen25_05b/train/adapter-best-evidence-mrr/candidate_bank.jsonl",
    "3b": DEFAULT_BANK_ROOT / "qwen25_3b/train/adapter-best-evidence-mrr/candidate_bank.jsonl",
}
DEFAULT_BIOASQ_FILE = ROOT / "data/training13b.json"
DEFAULT_SUPPORTED_QUESTION_SOURCE = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/evidence_grounded_single_answer_qwen25_05b/train_prepared.json"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"
DEFAULT_FIRST_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_first400_c3_c1_v1_gold_c3"
DEFAULT_REST_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_rest_after400_c3_c1_v1_gold_c3_gold_supported"
DEFAULT_FULL_RUN_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported"
DEFAULT_MERGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_all_c3_c1_v1_gold_c3_gold_supported"

STAGE_SPECS = [
    (
        "concept_learning",
        "dpo_stage1_concept_learning_all_pairs.jsonl",
        "Stage 1 — Concept Learning: C3 > C1",
        lambda row: row.get("chosen_class") == "C3" and row.get("rejected_class") == "C1",
    ),
    (
        "format_alignment",
        "dpo_stage2_format_alignment_all_pairs.jsonl",
        "Stage 2 — Format Alignment: C3 > C2",
        lambda row: row.get("chosen_class") == "C3" and row.get("rejected_class") == "C2",
    ),
    (
        "hierarchical_ranking",
        "dpo_stage3_hierarchical_ranking_all_pairs.jsonl",
        "Stage 3 — Hierarchical Ranking: C2 > C1",
        lambda row: row.get("chosen_class") == "C2" and row.get("rejected_class") == "C1",
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key, "")) for key in fieldnames})


def ordered_question_ids(bank_path: Path) -> list[str]:
    seen: set[str] = set()
    qids: list[str] = []
    with bank_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    return qids


def flatten_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        aliases: list[str] = []
        for item in value:
            aliases.extend(flatten_aliases(item))
        return aliases
    cleaned = str(value).strip()
    return [cleaned] if cleaned else []


def load_factoid_questions(bioasq_file: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(bioasq_file).read_text(encoding="utf-8"))
    return {str(q["id"]): q for q in payload.get("questions", []) if q.get("type") == "factoid"}


def load_eligible_question_ids(path: Path) -> set[str]:
    """Load an authoritative question pool from a prepared JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("questions")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list or an object with a questions list")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} is not an object")
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if not qid:
            raise ValueError(f"{path}: row {index} has no id or question_id")
        ids.append(qid)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: question IDs are not unique")
    return set(ids)


def first_bank_row_by_question(bank_path: Path) -> dict[str, dict[str, Any]]:
    first_rows: dict[str, dict[str, Any]] = {}
    with Path(bank_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            first_rows.setdefault(str(row["question_id"]), row)
    return first_rows


def build_gold_support_index(
    bank_path: Path,
    bioasq_file: Path,
    ordered_qids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return whether each question has a gold alias textually present in snippets.

    The candidate banks store the rendered evidence resources for each question.
    We use the first bank row for each question because all candidates for a
    question share the same evidence context.
    """
    questions = load_factoid_questions(bioasq_file)
    first_rows = first_bank_row_by_question(bank_path)
    index: dict[str, dict[str, Any]] = {}
    for ordinal, qid in enumerate(ordered_qids):
        question = questions.get(qid)
        row = first_rows.get(qid)
        aliases = flatten_aliases(question.get("exact_answer") if question else [])
        snippets = extract_snippets((row or {}).get("evidence") or [])
        supported_alias = first_extractive_alias(aliases, snippets)
        evidence_ids = extractive_evidence_ids(supported_alias, snippets) if supported_alias else []
        index[qid] = {
            "question_id": qid,
            "original_ordinal": ordinal,
            "has_gold_alias": bool(aliases),
            "gold_aliases": aliases,
            "has_supported_gold_alias": bool(supported_alias),
            "supported_gold_alias": supported_alias,
            "supported_gold_evidence_ids": evidence_ids,
            "snippet_count": len(snippets),
        }
    return index


def select_questions_by_ids(path: Path, question_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    wanted = set(question_ids)
    selected_order = {qid: index for index, qid in enumerate(question_ids)}
    grouped: dict[str, list[dict[str, Any]]] = {qid: [] for qid in question_ids}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            if qid in wanted:
                grouped[qid].append(row)
    missing = [qid for qid in question_ids if not grouped.get(qid)]
    if missing:
        raise ValueError(f"{path}: missing selected question IDs: {missing[:5]}")
    rows: list[dict[str, Any]] = []
    for qid in sorted(question_ids, key=lambda item: selected_order[item]):
        rows.extend(grouped[qid])
    return rows, question_ids


def prepare_records_for_question_ids(
    bank_files: dict[str, Path],
    bioasq_file: Path,
    question_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    questions = load_factoid_questions(bioasq_file)
    records: list[dict[str, Any]] = []
    selected_ids: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}

    for model, bank_path in bank_files.items():
        rows, qids = select_questions_by_ids(Path(bank_path), question_ids)
        selected_ids[model] = qids
        row_counts[model] = len(rows)
        missing = [qid for qid in qids if qid not in questions]
        if missing:
            raise ValueError(f"{model}: missing gold factoid questions: {missing[:5]}")
        for row in rows:
            qid = str(row["question_id"])
            parsed = row.get("parsed_items") or []
            if row.get("parser_status") != "ok" or len(parsed) != 1:
                raise ValueError(f"{model}/{row.get('response_id')}: invalid parsed candidate {parsed!r}")
            question = questions[qid]
            aliases = flatten_aliases(question.get("exact_answer"))
            if not aliases:
                raise ValueError(f"{qid}: no exact_answer aliases")
            candidate = str(parsed[0]).strip()
            evidence = row.get("evidence") or []
            records.append({
                "question_id": qid,
                "question": str(row.get("question_text") or question.get("body") or ""),
                "gold_aliases": aliases,
                "candidate": candidate,
                "candidate_output": format_answer(candidate),
                "source_model": model,
                "response_id": str(row.get("response_id")),
                "sample_id": row.get("sample_id"),
                "bank_path": str(bank_path),
                "bank_prompt": str(row.get("prompt") or ""),
                "snippets": extract_snippets(evidence),
            })

    expected = len(question_ids) * len(bank_files) * 20
    if len(records) != expected:
        raise ValueError(f"Expected {expected} candidate rows ({len(question_ids)} x 20 x {len(bank_files)}), got {len(records)}")
    return records, selected_ids, row_counts


def judgment_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["question_id"]), str(row["source_model"]), str(row["response_id"]))


def chunk_name(offset: int, limit: int) -> str:
    return f"questions_{offset + 1:04d}_{offset + limit:04d}"


def filtered_chunk_name(chunk_index: int, question_ids: list[str], support_index: dict[str, dict[str, Any]]) -> str:
    ordinals = [int(support_index[qid]["original_ordinal"]) for qid in question_ids]
    return f"gold_supported_chunk_{chunk_index:04d}_orig_{min(ordinals) + 1:04d}_{max(ordinals) + 1:04d}"


def load_existing_judgment_map(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            key = judgment_key(row)
            if key in merged:
                duplicate_count += 1
            merged[key] = row
    if duplicate_count:
        print(f"Warning: replaced {duplicate_count} duplicate judgment keys while merging.")
    return merged


def split_curriculum_pairs(pair_file: Path, output_root: Path) -> dict[str, Any]:
    rows = read_jsonl(pair_file)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "source_pairs": str(pair_file),
        "class_scheme": "C3_C2_C1_collapsed_from_C3_C2_C1_C0",
        "curriculum": {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for stage_name, filename, stage_rule, predicate in STAGE_SPECS:
        selected = []
        for row in rows:
            if predicate(row):
                out = dict(row)
                out["stage"] = stage_name
                out["stage_rule"] = stage_rule
                selected.append(out)
        jsonl_path = output_root / filename
        csv_path = jsonl_path.with_suffix(".csv")
        write_jsonl(jsonl_path, selected)
        write_csv(csv_path, selected)
        summary["curriculum"][stage_name] = {
            "stage_rule": stage_rule,
            "pair_count": len(selected),
            "question_count": len({row.get("question_id") for row in selected}),
            "pair_class_counts": dict(Counter(f"{row.get('chosen_class')}>{row.get('rejected_class')}" for row in selected)),
            "jsonl_file": str(jsonl_path),
            "csv_file": str(csv_path),
        }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bioasq-file", type=Path, default=DEFAULT_BIOASQ_FILE)
    parser.add_argument(
        "--eligible-question-source",
        type=Path,
        default=None,
        help=(
            "Optional prepared JSON whose question IDs define the exact eligible pool. "
            "Use the 1,130-question supported SFT source to align annotation with SFT."
        ),
    )
    parser.add_argument("--bank-05b", type=Path, default=DEFAULT_BANK_FILES["05b"])
    parser.add_argument("--bank-3b", type=Path, default=DEFAULT_BANK_FILES["3b"])
    parser.add_argument("--first-root", type=Path, default=DEFAULT_FIRST_ROOT)
    parser.add_argument("--rest-root", type=Path, default=DEFAULT_REST_ROOT)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    parser.add_argument("--fresh-full-run", action="store_true", help="Annotate the full filtered train bank from scratch and merge only fresh chunk judgments. This ignores first-root and defaults to a separate full-run output root.")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--judge-endpoint", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--start-offset", type=int, default=400, help="Question offset where new annotation starts. Default: 400, after the first 400 questions.")
    parser.add_argument("--end-offset", type=int, default=None, help="Exclusive question offset to stop at. Default: all remaining questions.")
    parser.add_argument("--chunk-size", type=int, default=25, help="Smaller chunks recover cleanly from API limits. Default: 25 questions.")
    parser.add_argument("--filter-gold-supported", dest="filter_gold_supported", action="store_true", default=True, help="Only annotate and merge questions where at least one gold alias appears in the supplied snippets. Default: on.")
    parser.add_argument("--no-filter-gold-supported", dest="filter_gold_supported", action="store_false", help="Disable the supported-gold question filter and keep the legacy behavior.")
    parser.add_argument("--max-new-judge-calls", type=int, default=500, help="Optional cap for new API calls per chunk. Cached and deterministic exact cases do not count. Default: 500.")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0, help="Sleep between new API requests. Default: 1s.")
    parser.add_argument("--max-retries", type=int, default=2, help="Validation retry count for malformed judge JSON.")
    parser.add_argument("--rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--rate-limit-initial-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--rate-limit-max-sleep-seconds", type=float, default=600.0)
    parser.add_argument("--no-abort-on-rate-limit", action="store_true", help="Convert persistent 429s to UNCERTAIN instead of aborting the chunk. Not recommended.")
    parser.add_argument("--archive-incomplete-chunks", action="store_true", help="Move incomplete chunk directories aside before rerunning them.")
    parser.add_argument("--policy", default="all_ordered_class_pairs", choices=["all_ordered_class_pairs", "semantic_positive_vs_negative"])
    parser.add_argument("--gold-c3-policy", default="always_first_extractive_alias", choices=["inject_if_missing", "always_first_alias", "always_first_extractive_alias"])
    parser.add_argument("--require-c3-extractive", dest="require_c3_extractive", action="store_true", default=True, help="Only keep/inject C3 aliases that appear in snippets. Default: on.")
    parser.add_argument("--no-require-c3-extractive", dest="require_c3_extractive", action="store_false", help="Disable extractive C3 filtering.")
    parser.add_argument("--merge-only", action="store_true", help="Skip annotation and merge the existing first-root plus rest-root chunk judgments.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned chunks and exit without API calls or writes.")
    parser.add_argument("--overwrite-merged", action="store_true", help="Allow rewriting merged output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fresh_full_run:
        if args.start_offset == 400:
            args.start_offset = 0
        if args.rest_root == DEFAULT_REST_ROOT:
            args.rest_root = DEFAULT_FULL_RUN_ROOT / "chunks"
        if args.merged_root == DEFAULT_MERGED_ROOT:
            args.merged_root = DEFAULT_FULL_RUN_ROOT / "merged"

    bank_files = {"05b": args.bank_05b, "3b": args.bank_3b}
    qids_by_model = {model: ordered_question_ids(path) for model, path in bank_files.items()}
    reference_qids = qids_by_model["05b"]
    total_questions = min(len(qids) for qids in qids_by_model.values())
    for model, qids in qids_by_model.items():
        if qids[:total_questions] != reference_qids[:total_questions]:
            raise ValueError(
                f"Question order mismatch between 05b and {model}. "
                "This script expects aligned candidate-bank question order."
            )
    start = args.start_offset
    end = total_questions if args.end_offset is None else min(args.end_offset, total_questions)
    if start < 0 or end < start:
        raise ValueError(f"Invalid offsets: start={start}, end={end}")
    bank_scope_qids = reference_qids[:end]
    eligible_ids = None
    if args.eligible_question_source is not None:
        eligible_ids = load_eligible_question_ids(args.eligible_question_source)
        missing_eligible = sorted(eligible_ids - set(reference_qids))
        if missing_eligible:
            raise ValueError(
                f"{args.eligible_question_source}: {len(missing_eligible)} eligible IDs are absent "
                f"from the candidate banks; first IDs: {missing_eligible[:5]}"
            )
        bank_scope_qids = [qid for qid in bank_scope_qids if qid in eligible_ids]

    support_index = build_gold_support_index(args.bank_05b, args.bioasq_file, reference_qids[:end])
    if args.filter_gold_supported:
        merged_question_ids = [
            qid for qid in bank_scope_qids
            if support_index[qid]["has_supported_gold_alias"]
        ]
        annotation_question_ids = [
            qid for qid in reference_qids[start:end]
            if (eligible_ids is None or qid in eligible_ids)
            and support_index[qid]["has_supported_gold_alias"]
        ]
    else:
        merged_question_ids = bank_scope_qids
        annotation_question_ids = [
            qid for qid in reference_qids[start:end]
            if eligible_ids is None or qid in eligible_ids
        ]

    chunk_specs: list[dict[str, Any]] = []
    for chunk_index, chunk_start in enumerate(range(0, len(annotation_question_ids), args.chunk_size), 1):
        qids = annotation_question_ids[chunk_start: chunk_start + args.chunk_size]
        if not qids:
            continue
        if args.filter_gold_supported:
            name = filtered_chunk_name(chunk_index, qids, support_index)
            offset = min(int(support_index[qid]["original_ordinal"]) for qid in qids)
        else:
            offset = start + chunk_start
            name = chunk_name(offset, len(qids))
        chunk_specs.append({"name": name, "offset": offset, "limit": len(qids), "question_ids": qids})

    first_judgments_path = args.first_root / "candidate_class_judgments.jsonl"
    unsupported_in_merge_scope = [
        qid for qid in bank_scope_qids
        if not support_index[qid]["has_supported_gold_alias"]
    ]
    unsupported_in_annotation_scope = [
        qid for qid in reference_qids[start:end]
        if (eligible_ids is None or qid in eligible_ids)
        if not support_index[qid]["has_supported_gold_alias"]
    ]

    plan = {
        "bank_files": {model: str(path) for model, path in bank_files.items()},
        "total_train_questions_in_banks": total_questions,
        "first_root": str(args.first_root),
        "first_judgments_file": str(first_judgments_path),
        "rest_root": str(args.rest_root),
        "merged_root": str(args.merged_root),
        "fresh_full_run": args.fresh_full_run,
        "start_offset": start,
        "end_offset": end,
        "eligible_question_source": (
            None if args.eligible_question_source is None else str(args.eligible_question_source)
        ),
        "eligible_question_source_count": None if eligible_ids is None else len(eligible_ids),
        "eligible_questions_in_bank_scope": len(bank_scope_qids),
        "filter_gold_supported": args.filter_gold_supported,
        "unfiltered_merge_scope_questions": len(bank_scope_qids),
        "merged_question_count_after_filter": len(merged_question_ids),
        "unsupported_questions_removed_from_merge": len(unsupported_in_merge_scope),
        "unfiltered_annotation_scope_questions": end - start,
        "question_count_to_annotate": len(annotation_question_ids),
        "unsupported_questions_skipped_before_annotation": len(unsupported_in_annotation_scope),
        "chunk_size": args.chunk_size,
        "chunks": [
            {
                "name": spec["name"],
                "offset": spec["offset"],
                "limit": spec["limit"],
                "first_question_id": spec["question_ids"][0],
                "last_question_id": spec["question_ids"][-1],
            }
            for spec in chunk_specs
        ],
        "gold_c3_policy": args.gold_c3_policy,
        "require_c3_extractive": args.require_c3_extractive,
        "merge_only": args.merge_only,
        "max_new_judge_calls": args.max_new_judge_calls,
        "request_delay_seconds": args.request_delay_seconds,
        "rate_limit_max_retries": args.rate_limit_max_retries,
        "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        "judge_rubric_sha256": digest(JUDGE_SYSTEM),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    if not args.fresh_full_run and not first_judgments_path.exists():
        raise FileNotFoundError(first_judgments_path)

    chunk_judgment_paths: list[Path] = []
    args.rest_root.mkdir(parents=True, exist_ok=True)
    write_json(args.rest_root / "gold_support_filter_plan.json", {
        **plan,
        "unsupported_question_ids_removed_from_merge": unsupported_in_merge_scope,
        "unsupported_question_ids_skipped_before_annotation": unsupported_in_annotation_scope,
        "support_index": support_index,
    })
    if not args.merge_only:
        for spec in chunk_specs:
            name = str(spec["name"])
            question_ids = list(spec["question_ids"])
            chunk_root = args.rest_root / name
            judgments_path = chunk_root / "candidate_class_judgments.jsonl"
            summary_path = chunk_root / "judgment_summary.json"
            if judgments_path.exists() and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rubric_matches = (
                    summary.get("judge_rubric_version") == JUDGE_RUBRIC_VERSION
                    and summary.get("rubric_sha256") == digest(JUDGE_SYSTEM)
                )
                if summary.get("status") == "complete" and rubric_matches:
                    print(f"[{name}] already complete; reusing {judgments_path}")
                    chunk_judgment_paths.append(judgments_path)
                    continue
                if args.archive_incomplete_chunks:
                    archive_path = chunk_root.with_name(
                        f"{chunk_root.name}_incomplete_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                    )
                    chunk_root.rename(archive_path)
                    reason = "incomplete" if summary.get("status") != "complete" else "rubric-mismatched"
                    print(f"[{name}] archived {reason} chunk to {archive_path}")
                else:
                    print(
                        f"[{name}] existing judgments are incomplete or use another rubric; "
                        "rerunning with compatible cache reuse only. "
                        "Pass --archive-incomplete-chunks to move old failed outputs aside."
                    )
            elif chunk_root.exists() and args.archive_incomplete_chunks:
                archive_path = chunk_root.with_name(
                    f"{chunk_root.name}_incomplete_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )
                chunk_root.rename(archive_path)
                print(f"[{name}] archived interrupted chunk to {archive_path}")

            print(f"[{name}] preparing records")
            records, selected_ids, row_counts = prepare_records_for_question_ids(
                bank_files,
                args.bioasq_file,
                question_ids=question_ids,
            )
            write_json(chunk_root / "record_selection.json", {
                "chunk": name,
                "question_offset": spec["offset"],
                "question_limit": len(question_ids),
                "selected_ids": selected_ids,
                "row_counts": row_counts,
                "record_count": len(records),
                "filter_gold_supported": args.filter_gold_supported,
                "gold_support": {qid: support_index[qid] for qid in question_ids},
            })
            print(f"[{name}] annotating {len(records):,} candidate records")
            judge = CandidateBankClassJudge(
                records=records,
                output_root=chunk_root,
                api_key_file=args.api_key_file,
                judge_model=args.judge_model,
                judge_endpoint=args.judge_endpoint,
                max_new_judge_calls=args.max_new_judge_calls,
                max_retries=args.max_retries,
                request_delay_seconds=args.request_delay_seconds,
                rate_limit_max_retries=args.rate_limit_max_retries,
                rate_limit_initial_sleep_seconds=args.rate_limit_initial_sleep_seconds,
                rate_limit_max_sleep_seconds=args.rate_limit_max_sleep_seconds,
                abort_on_rate_limit=not args.no_abort_on_rate_limit,
            )
            _judgments, summary = judge.run()
            chunk_judgment_paths.append(judgments_path)
            print(
                f"[{name}] {summary.get('status')} | new API calls: {summary.get('new_api_calls')} "
                f"| rate limit hits: {summary.get('rate_limit_hits')}"
            )
    else:
        for spec in chunk_specs:
            judgments_path = args.rest_root / str(spec["name"]) / "candidate_class_judgments.jsonl"
            if judgments_path.exists():
                chunk_judgment_paths.append(judgments_path)
            else:
                raise FileNotFoundError(judgments_path)

    if args.merged_root.exists() and not args.overwrite_merged:
        existing_pairs = args.merged_root / "dpo_pairs.jsonl"
        existing_summary = args.merged_root / "summary.json"
        if existing_pairs.exists() or existing_summary.exists():
            raise FileExistsError(
                f"Merged output already exists: {args.merged_root}. "
                "Pass --overwrite-merged to regenerate it."
            )
    args.merged_root.mkdir(parents=True, exist_ok=True)

    merged_records, selected_ids, row_counts = prepare_records_for_question_ids(
        bank_files,
        args.bioasq_file,
        question_ids=merged_question_ids,
    )
    judgment_sources = chunk_judgment_paths if args.fresh_full_run else [first_judgments_path, *chunk_judgment_paths]
    judgment_map = load_existing_judgment_map(judgment_sources)
    missing = [judgment_key(record) for record in merged_records if judgment_key(record) not in judgment_map]
    if missing:
        preview = "; ".join(str(item) for item in missing[:10])
        raise ValueError(f"Missing {len(missing)} judgments for merged records. First missing: {preview}")
    merged_judgments = [judgment_map[judgment_key(record)] for record in merged_records]
    write_jsonl(args.merged_root / "candidate_class_judgments.jsonl", merged_judgments)

    pair_summary = build_pairs(
        records=merged_records,
        judgments=merged_judgments,
        output_root=args.merged_root,
        policy=args.policy,
        inject_gold_c3=True,
        require_c3_extractive=args.require_c3_extractive,
        gold_c3_policy=args.gold_c3_policy,
    )
    stage_summary = split_curriculum_pairs(args.merged_root / "dpo_pairs.jsonl", args.merged_root / "staged_curriculum_pairs")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_scheme": "C3_C2_C1_collapsed_from_C3_C2_C1_C0",
        "annotation_plan": plan,
        "selected_ids": selected_ids,
        "row_counts": row_counts,
        "chunk_judgment_files": [str(path) for path in chunk_judgment_paths],
        "judgment_sources": [str(path) for path in judgment_sources],
        "merged_candidate_records": len(merged_records),
        "merged_questions": len(merged_question_ids),
        "filtered_out_question_count": len(unsupported_in_merge_scope),
        "filtered_out_question_ids": unsupported_in_merge_scope,
        "pair_summary": pair_summary,
        "stage_summary": stage_summary,
    }
    write_json(args.merged_root / "summary.json", pair_summary)
    write_json(args.merged_root / "manifest.json", manifest)
    print(json.dumps({
        "merged_root": str(args.merged_root),
        "pair_summary": pair_summary,
        "stage_summary": stage_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
