#!/usr/bin/env python
"""Run and compare a paired GPT-4.1 Mini versus GPT-4.1 generation pilot.

Both arms use the same deterministic 250-source manifest and evaluate its first
20 sources (10 clean and 10 distractor).  Only the generator differs:

- mini_mini: GPT-4.1 Mini generator, GPT-4.1 Mini verifier
- gpt41_mini: GPT-4.1 generator, GPT-4.1 Mini verifier

API phases are explicit and resumable.  ``prepare`` and ``compare`` make no API
calls. Successful responses remain cached inside each arm's output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "cse_dpo/build_synthetic_factoid_qa_pilot.py"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/synthetic_factoid_qa/"
    "gpt41mini_vs_gpt41_generation_20sources_v2"
)
DEFAULT_API_KEY_FILE = ROOT / "open_ai_api.txt"
SOURCE_LIMIT = 20
SOURCE_COUNT = 250
VALIDATION_SOURCE_COUNT = 50
SEED = 3407
VERIFIER_MODEL = "gpt-4.1-mini-2025-04-14"
ARMS = {
    "mini_mini": {
        "label": "GPT-4.1 Mini generator + GPT-4.1 Mini verifier",
        "generator_model": "gpt-4.1-mini-2025-04-14",
        "verifier_model": VERIFIER_MODEL,
    },
    "gpt41_mini": {
        "label": "GPT-4.1 generator + GPT-4.1 Mini verifier",
        "generator_model": "gpt-4.1-2025-04-14",
        "verifier_model": VERIFIER_MODEL,
    },
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weak_surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def arm_root(output_root: Path, arm: str) -> Path:
    return output_root / arm


def pipeline_command(args: argparse.Namespace, arm: str, phase: str) -> list[str]:
    config = ARMS[arm]
    cmd = [
        args.python,
        str(PIPELINE),
        "--phase", phase,
        "--output-root", str(arm_root(args.output_root, arm)),
        "--api-key-file", str(args.api_key_file),
        "--generator-model", config["generator_model"],
        "--verifier-model", config["verifier_model"],
        "--source-count", str(SOURCE_COUNT),
        "--validation-source-count", str(VALIDATION_SOURCE_COUNT),
        "--retrieval-candidate-count", "100",
        "--seed", str(SEED),
        "--source-limit", str(args.source_limit),
        "--progress-every", str(args.progress_every),
    ]
    if args.max_new_calls is not None:
        cmd.extend(["--max-new-calls", str(args.max_new_calls)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_pipeline(args: argparse.Namespace, phase: str) -> None:
    for arm in ARMS:
        cmd = pipeline_command(args, arm, phase)
        print(f"\n[{arm}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    if phase == "prepare":
        assert_identical_manifests(args.output_root)


def assert_identical_manifests(output_root: Path) -> dict[str, str]:
    checked: dict[str, str] = {}
    for filename in ("source_manifest.jsonl", "eligible_source_pool.jsonl"):
        hashes = {
            arm: file_sha256(arm_root(output_root, arm) / filename)
            for arm in ARMS
        }
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"A/B inputs differ for {filename}: {hashes}")
        checked[filename] = next(iter(hashes.values()))

    manifests = {
        arm: read_jsonl(arm_root(output_root, arm) / "source_manifest.jsonl")[:SOURCE_LIMIT]
        for arm in ARMS
    }
    reference = manifests["mini_mini"]
    expected_ids = [row["synthetic_source_id"] for row in reference]
    for arm, rows in manifests.items():
        if [row["synthetic_source_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"A/B source order differs in {arm}")
    balance = Counter((row["split"], row["evidence_arm"]) for row in reference)
    expected_balance = Counter({("train", "clean"): 10, ("train", "distractor"): 10})
    if SOURCE_LIMIT == 20 and balance != expected_balance:
        raise RuntimeError(f"Unexpected 20-source pilot balance: {dict(balance)}")
    checked["selected_source_ids_sha256"] = hashlib.sha256(
        json.dumps(expected_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return checked


def rejection_reasons(row: dict[str, Any]) -> list[str]:
    verification = row.get("verification", {})
    reasons: list[str] = []
    if row.get("generation_validation_passed") is False:
        reasons.append("generation_validation_failed")
    if verification.get("status") != "accepted":
        reasons.append("verifier_review")
    if not verification.get("unique_answer", False):
        reasons.append("not_unique")
    if not verification.get("explicit_relation", False):
        reasons.append("relation_not_explicit")
    if not row.get("verification_answer_match", False):
        reasons.append("answer_mismatch")
    if not row.get("verification_target_resource_match", False):
        reasons.append("wrong_resource")
    return reasons or ["accepted"]


def load_final_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / "synthetic_accepted_all.jsonl") + read_jsonl(root / "synthetic_review.jsonl")
    return {row["synthetic_question_id"]: row for row in rows}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_hashes = assert_identical_manifests(args.output_root)
    arm_rows = {arm: load_final_rows(arm_root(args.output_root, arm)) for arm in ARMS}
    generated_rows = {
        arm: read_jsonl(arm_root(args.output_root, arm) / "generated_sources.jsonl")
        for arm in ARMS
    }
    generated_source_ids = {
        arm: {row["synthetic_source_id"] for row in rows}
        for arm, rows in generated_rows.items()
    }
    ineligible_source_ids = {
        arm: {
            row["synthetic_source_id"]
            for row in rows
            if row.get("generation_eligible") is False or not row.get("generated_items")
        }
        for arm, rows in generated_rows.items()
    }
    verified_source_ids = {
        arm: {row["synthetic_source_id"] for row in read_jsonl(arm_root(args.output_root, arm) / "verified_sources.jsonl")}
        for arm in ARMS
    }
    missing_outputs = [arm for arm, rows in arm_rows.items() if not rows]
    if missing_outputs:
        raise FileNotFoundError(f"Missing finalized records for: {', '.join(missing_outputs)}")

    manifest = read_jsonl(arm_root(args.output_root, "mini_mini") / "source_manifest.jsonl")[
        : args.source_limit
    ]
    source_by_id = {row["synthetic_source_id"]: row for row in manifest}
    expected_ids = [
        f"{source['synthetic_source_id']}::q{question_index}"
        for source in manifest
        for question_index in (1, 2)
    ]

    detail_rows: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    answer_agreement = 0
    paired_generated_count = 0
    for question_id in expected_ids:
        mini = arm_rows["mini_mini"].get(question_id)
        full = arm_rows["gpt41_mini"].get(question_id)
        source_id = question_id.rsplit("::", 1)[0]
        source = source_by_id[source_id]
        mini_ok = bool(mini and mini.get("accepted"))
        full_ok = bool(full and full.get("accepted"))
        if mini_ok and full_ok:
            outcome = "both_accepted"
        elif mini_ok:
            outcome = "mini_only_accepted"
        elif full_ok:
            outcome = "gpt41_only_accepted"
        else:
            outcome = "neither_accepted"
        outcome_counts[outcome] += 1

        answers_match = False
        if mini is not None and full is not None:
            paired_generated_count += 1
            answers_match = weak_surface(mini["answer"]) == weak_surface(full["answer"])
            answer_agreement += int(answers_match)

        def field(row: dict[str, Any] | None, key: str, default: Any = "") -> Any:
            return row.get(key, default) if row is not None else default

        def verification_field(row: dict[str, Any] | None, key: str) -> Any:
            return (row.get("verification") or {}).get(key, "") if row is not None else ""

        def missing_reason(arm: str, row: dict[str, Any] | None) -> str:
            if row is not None:
                return "|".join(rejection_reasons(row))
            if source_id not in generated_source_ids[arm]:
                return "generation_failed"
            if source_id in ineligible_source_ids[arm]:
                return "source_ineligible"
            if source_id not in verified_source_ids[arm]:
                return "verification_failed"
            return "finalization_missing"

        detail_rows.append({
            "synthetic_question_id": question_id,
            "synthetic_source_id": source_id,
            "split": source["split"],
            "evidence_arm": source["evidence_arm"],
            "paired_outcome": outcome,
            "generated_answers_match": answers_match if mini is not None and full is not None else "",
            "mini_generated": mini is not None,
            "mini_question": field(mini, "question"),
            "mini_answer": field(mini, "answer"),
            "mini_accepted": mini_ok,
            "mini_verifier_answer": verification_field(mini, "extracted_answer"),
            "mini_verifier_basis": verification_field(mini, "basis"),
            "mini_rejection_reasons": missing_reason("mini_mini", mini),
            "gpt41_generated": full is not None,
            "gpt41_question": field(full, "question"),
            "gpt41_answer": field(full, "answer"),
            "gpt41_accepted": full_ok,
            "gpt41_verifier_answer": verification_field(full, "extracted_answer"),
            "gpt41_verifier_basis": verification_field(full, "basis"),
            "gpt41_rejection_reasons": missing_reason("gpt41_mini", full),
            "manual_preference": "",
            "manual_notes": "",
        })

    arm_metrics: dict[str, Any] = {}
    for arm, rows_by_id in arm_rows.items():
        rows = [rows_by_id[qid] for qid in expected_ids if qid in rows_by_id]
        accepted = [row for row in rows if row["accepted"]]
        missing_question_ids = [qid for qid in expected_ids if qid not in rows_by_id]
        generation_failed_question_ids = [
            qid for qid in missing_question_ids
            if qid.rsplit("::", 1)[0] not in generated_source_ids[arm]
        ]
        ineligible_question_ids = [
            qid for qid in missing_question_ids
            if qid.rsplit("::", 1)[0] in ineligible_source_ids[arm]
        ]
        verification_failed_question_ids = [
            qid for qid in missing_question_ids
            if qid.rsplit("::", 1)[0] in generated_source_ids[arm]
            and qid.rsplit("::", 1)[0] not in ineligible_source_ids[arm]
            and qid.rsplit("::", 1)[0] not in verified_source_ids[arm]
        ]
        finalization_missing_question_ids = [
            qid for qid in missing_question_ids
            if qid not in generation_failed_question_ids
            and qid not in ineligible_question_ids
            and qid not in verification_failed_question_ids
        ]
        rejection_counts = Counter(
            reason
            for row in rows
            for reason in rejection_reasons(row)
            if reason != "accepted"
        )
        rejection_counts["generation_failed"] += len(generation_failed_question_ids)
        rejection_counts["source_ineligible"] += len(ineligible_question_ids)
        rejection_counts["verification_failed"] += len(verification_failed_question_ids)
        rejection_counts["finalization_missing"] += len(finalization_missing_question_ids)
        rejection_counts += Counter()
        arm_metrics[arm] = {
            **ARMS[arm],
            "expected_question_count": len(expected_ids),
            "generated_and_finalized_question_count": len(rows),
            "generation_failed_question_count": len(generation_failed_question_ids),
            "generation_failed_question_ids": generation_failed_question_ids,
            "source_ineligible_question_count": len(ineligible_question_ids),
            "source_ineligible_question_ids": ineligible_question_ids,
            "verification_failed_question_count": len(verification_failed_question_ids),
            "verification_failed_question_ids": verification_failed_question_ids,
            "finalization_missing_question_count": len(finalization_missing_question_ids),
            "finalization_missing_question_ids": finalization_missing_question_ids,
            "accepted_count": len(accepted),
            "acceptance_rate_over_expected_slots": len(accepted) / len(expected_ids),
            "accepted_by_evidence_arm": dict(Counter(row["evidence_arm"] for row in accepted)),
            "expected_by_evidence_arm": dict(Counter(
                source["evidence_arm"] for source in manifest for _ in (1, 2)
            )),
            "rejection_reason_counts": dict(rejection_counts),
            "generation_summary": read_json(arm_root(args.output_root, arm) / "generation_summary.json", {}),
            "verification_summary": read_json(arm_root(args.output_root, arm) / "verification_summary.json", {}),
        }

    csv_path = args.output_root / "paired_manual_review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = {
        "experiment": "paired-generator-model-comparison-v2",
        "controlled_difference": "generator model only",
        "source_limit": args.source_limit,
        "expected_question_slot_count": len(expected_ids),
        "manifest_hashes": manifest_hashes,
        "paired_acceptance_outcomes": dict(outcome_counts),
        "paired_generated_question_count": paired_generated_count,
        "generated_answer_agreement_count": answer_agreement,
        "generated_answer_agreement_rate": (
            answer_agreement / paired_generated_count if paired_generated_count else 0.0
        ),
        "arms": arm_metrics,
        "manual_review_csv": str(csv_path.resolve()),
        "interpretation_warning": (
            "Verifier acceptance measures constraint compliance, not complete biomedical quality. "
            "Generation failures count against the arm. Blindly audit the paired CSV before selecting a generator."
        ),
    }
    write_json(args.output_root / "comparison_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["prepare", "generate", "verify", "finalize", "compare", "all"],
        default="prepare",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-limit", type=int, default=SOURCE_LIMIT)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    if args.source_limit != SOURCE_LIMIT:
        raise ValueError("This comparison is fixed to the paired 20-source pilot")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "comparison_config.json", {
        "source_limit": args.source_limit,
        "source_count": SOURCE_COUNT,
        "validation_source_count": VALIDATION_SOURCE_COUNT,
        "seed": SEED,
        "arms": ARMS,
    })
    if args.phase == "compare":
        compare(args)
        return
    phases = [args.phase] if args.phase != "all" else ["prepare", "generate", "verify", "finalize"]
    for phase in phases:
        run_pipeline(args, phase)
    if args.phase == "all" and not args.dry_run:
        compare(args)


if __name__ == "__main__":
    main()
