#!/usr/bin/env python
"""Annotate the held-out SFT/DPO-split candidate bank and build curriculum pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cse_dpo.annotate_remaining_candidate_bank_questions import split_curriculum_pairs
from cse_dpo.candidate_bank_class_judge import (
    CandidateBankClassJudge,
    JUDGE_SYSTEM,
    build_pairs,
    digest,
    exact_norm,
    extract_snippets,
    first_extractive_alias,
    format_answer,
)
from src.utility.bioasq_format import parse_prediction_items


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    ROOT
    / "Artifacts/cse_dpo/candidate_banks/qwen25_05b_sft80_dpo20_r32_step175_dpo226_s10_t07_seed3407"
    / "adapter-best-evidence-mrr/candidate_bank.jsonl"
)
DEFAULT_QUESTIONS = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/gold_supported_question_split_sft80_dpo20_seed3407"
    / "dpo_questions_for_candidate_generation.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/class_judgments/qwen25_05b_sft80_dpo20_r32_step175_dpo226_s10_t07_seed3407_gold_c3"
)
DEFAULT_API_KEY = ROOT / "open_ai_api.txt"
SOURCE_MODEL = "qwen25_05b_sft80_dpo20_r32_step175"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field, "")
                    for field in fields
                }
            )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_records(
    bank_path: Path,
    questions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_rows = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = {str(row["id"]): row for row in question_rows}
    if len(questions) != len(question_rows):
        raise ValueError("Prepared DPO questions contain duplicate IDs")

    rows = read_jsonl(bank_path)
    bank_question_order = list(dict.fromkeys(str(row["question_id"]) for row in rows))
    if bank_question_order != list(questions):
        raise ValueError("Candidate-bank question order does not match the held-out DPO question file")
    sample_counts = Counter(str(row["question_id"]) for row in rows)
    if set(sample_counts.values()) != {10}:
        raise ValueError(f"Expected exactly 10 candidates per question, got {dict(Counter(sample_counts.values()))}")

    records: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row["question_id"])
        question = questions[question_id]
        aliases = parse_prediction_items(str(question.get("output", "")), "factoid")
        if not aliases:
            raise ValueError(f"{question_id}: no supported gold aliases")
        parsed = row.get("parsed_items") or []
        if row.get("parser_status") != "ok" or len(parsed) != 1:
            raise ValueError(f"{question_id}/{row.get('response_id')}: invalid parsed candidate {parsed!r}")
        snippets = extract_snippets(row.get("evidence") or [])
        supported_gold = first_extractive_alias(aliases, snippets)
        if not supported_gold:
            raise ValueError(f"{question_id}: none of its prepared gold aliases occurs in the bank snippets")
        candidate = str(parsed[0]).strip()
        records.append(
            {
                "question_id": question_id,
                "question": str(row.get("question_text") or question.get("input_1") or ""),
                "gold_aliases": aliases,
                "candidate": candidate,
                "candidate_output": format_answer(candidate),
                "source_model": SOURCE_MODEL,
                "response_id": str(row.get("response_id")),
                "sample_id": row.get("sample_id"),
                "bank_path": str(bank_path.resolve()),
                "bank_prompt": str(row.get("prompt") or ""),
                "snippets": snippets,
            }
        )

    exact_count = sum(
        any(exact_norm(record["candidate"]) == exact_norm(alias) for alias in record["gold_aliases"])
        for record in records
    )
    unique_nonexact_normalized = {
        (record["question_id"], exact_norm(record["candidate"]))
        for record in records
        if not any(exact_norm(record["candidate"]) == exact_norm(alias) for alias in record["gold_aliases"])
    }
    unique_nonexact_raw = {
        (record["question_id"], record["candidate"].strip())
        for record in records
        if not any(exact_norm(record["candidate"]) == exact_norm(alias) for alias in record["gold_aliases"])
    }
    summary = {
        "question_count": len(questions),
        "candidate_count": len(records),
        "samples_per_question": 10,
        "deterministic_exact_c3_records": exact_count,
        "nonexact_records": len(records) - exact_count,
        "unique_nonexact_normalized_question_candidates": len(unique_nonexact_normalized),
        "unique_nonexact_raw_question_candidates": len(unique_nonexact_raw),
        "maximum_uncached_judge_calls": len(unique_nonexact_raw),
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY)
    parser.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--max-new-judge-calls", type=int, default=500)
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank_path = args.bank.resolve()
    questions_path = args.questions.resolve()
    output_root = args.output_root.resolve()
    for path in (bank_path, questions_path, args.api_key_file.resolve()):
        if not path.exists():
            raise FileNotFoundError(path)

    records, selection_summary = prepare_records(bank_path, questions_path)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned" if args.dry_run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bank": str(bank_path),
        "bank_sha256": file_hash(bank_path),
        "questions": str(questions_path),
        "questions_sha256": file_hash(questions_path),
        "output_root": str(output_root),
        "source_model": SOURCE_MODEL,
        "judge_model": args.judge_model,
        "judge_rubric_sha256": digest(JUDGE_SYSTEM),
        "gold_c3_policy": "always_first_extractive_alias",
        "require_c3_extractive": True,
        "selection": selection_summary,
        "maximum_new_judge_calls": args.max_new_judge_calls,
    }
    write_json(output_root / "annotation_plan.json", plan)
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    judgments_path = output_root / "candidate_class_judgments.jsonl"
    judgment_summary_path = output_root / "judgment_summary.json"
    reuse_complete = False
    if judgments_path.exists() and judgment_summary_path.exists():
        existing_summary = json.loads(judgment_summary_path.read_text(encoding="utf-8"))
        existing_judgments = read_jsonl(judgments_path)
        reuse_complete = (
            existing_summary.get("status") == "complete"
            and len(existing_judgments) == len(records)
            and existing_summary.get("judge_model") == args.judge_model
            and existing_summary.get("rubric_sha256") == digest(JUDGE_SYSTEM)
        )
    if reuse_complete:
        judgments = existing_judgments
        judgment_summary = existing_summary
        print("Reusing complete judgments:", judgments_path)
    else:
        judge = CandidateBankClassJudge(
            records=records,
            output_root=output_root,
            api_key_file=args.api_key_file.resolve(),
            judge_model=args.judge_model,
            max_new_judge_calls=args.max_new_judge_calls,
            request_delay_seconds=args.request_delay_seconds,
        )
        judgments, judgment_summary = judge.run()
        if judgment_summary["status"] != "complete":
            raise RuntimeError(
                "Annotation is incomplete. Rerun the same command to reuse cached successful calls and retry failures."
            )

    write_csv(output_root / "candidate_class_judgments.csv", judgments)
    pair_summary = build_pairs(
        records=records,
        judgments=judgments,
        output_root=output_root,
        policy="all_ordered_class_pairs",
        inject_gold_c3=True,
        require_c3_extractive=True,
        gold_c3_policy="always_first_extractive_alias",
    )
    pair_rows = read_jsonl(output_root / "dpo_pairs.jsonl")
    write_csv(output_root / "dpo_pairs.csv", pair_rows)
    stage_summary = split_curriculum_pairs(
        output_root / "dpo_pairs.jsonl",
        output_root / "staged_curriculum_pairs",
    )
    final_manifest = {
        **plan,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "judgment_summary": judgment_summary,
        "pair_summary": pair_summary,
        "stage_summary": stage_summary,
    }
    write_json(output_root / "manifest.json", final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
