#!/usr/bin/env python
"""Prepare and optionally train error-aware answer-only DPO for Qwen2.5-0.5B.

The deployed task stays unchanged:
    answer-only prompt -> [BE] short extractive answer [EE]

Each C3>C1 preference row also carries one auxiliary diagnostic LM example:
    diagnostic prompt -> error label + grounded correction

The same LoRA adapter receives both gradients in one optimizer update:
    L = L_DPO(answer-only) + lambda_aux * L_CE(diagnostic)

The deliberately mistaken C1 rationale is supplied only as text to critique.
It is never used as a target.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = Path("/home/dinh-tuan/miniconda3/envs/bioasq/bin/python")

BASE_MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"
INITIAL_ADAPTER = (
    ROOT
    / "Artifacts/Factoid_SFT/models/"
    "evidence_grounded_per_supported_alias_qwen25_05b_lora_dropout_005_strict_extractive/"
    "adapter_best_evidence_mrr"
)
STRICT_STAGE1 = (
    ROOT
    / "Artifacts/cse_dpo/class_judgments/"
    "candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130/"
    "merged/staged_curriculum_pairs/dpo_stage1_concept_learning_all_pairs.jsonl"
)
C1_RATIONALES = (
    ROOT
    / "Artifacts/cse_dpo/c1_answer_rationales/"
    "gpt41mini_strict_equivalence_v2_positive1111_rubric_v3/"
    "c1_answer_rationales.jsonl"
)
GOLD_RATIONALES = (
    ROOT
    / "Artifacts/cse_dpo/gold_answer_rationales/"
    "gpt41mini_gold_supported_1130_v2/gold_answer_rationales.jsonl"
)
DIAGNOSTIC_PROMPT_SPEC = ROOT / "prompts/factoid_error_diagnostic_auxiliary.json"
DATASET_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/error_aware_multitask_pairs/"
    "strict_equivalence_v2_accepted466_v1"
)
OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/"
    "error_aware_multitask_dpo_qwen25_05b_strict_equivalence_v2_accepted466_lambda002"
)

AUXILIARY_WEIGHT = 0.02
MAX_LENGTH = 4096

# Fine-grained annotations remain in every row. This coarse target asks the
# model to learn a reusable decision rule rather than memorize sparse labels.
ERROR_GROUPS = {
    "SCOPE": {
        "broader",
        "narrower",
        "missing_qualifier",
        "extra_qualifier",
        "narrower_location",
        "missing_definition",
    },
    "ENTITY": {
        "wrong_entity",
        "wrong_target",
        "wrong_target_entity",
        "wrong_location",
        "wrong_answer",
    },
    "ENTITY_TYPE": {"wrong_answer_type", "part_whole"},
    "VALUE": {"wrong_value", "wrong_population"},
    "RELATION": {"wrong_relation"},
    "FORM": {"non_answer", "unsupported_explanation", "extra_information"},
}
ERROR_TO_GROUP = {
    fine: coarse for coarse, fine_values in ERROR_GROUPS.items() for fine in fine_values
}

DIAGNOSTIC_SYSTEM = json.loads(
    DIAGNOSTIC_PROMPT_SPEC.read_text(encoding="utf-8")
)["system"]


def ensure_project_python() -> None:
    current = Path(sys.executable).resolve()
    expected = PROJECT_PYTHON.resolve()
    if current == expected:
        return
    if not expected.exists():
        raise RuntimeError(
            f"Required environment interpreter is missing: {expected}. "
            "Activate the bioasq Conda environment before running this script."
        )
    print(f"[environment] Restarting with {expected}", flush=True)
    os.execv(str(expected), [str(expected), *sys.argv])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qwen_prompt(system: str, user: str) -> str:
    return (
        "<|im_start|>system\n"
        + system.strip()
        + "<|im_end|>\n<|im_start|>user\n"
        + user.strip()
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def snippet_lines(*records: dict[str, Any]) -> list[str]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        for snippet in record.get("supporting_snippets") or []:
            snippet_id = str(snippet.get("snippet_id") or "").strip()
            if snippet_id and snippet_id not in by_id:
                by_id[snippet_id] = snippet
    lines = []
    for snippet_id, snippet in sorted(
        by_id.items(),
        key=lambda item: tuple(
            int(part) if part.isdigit() else part for part in item[0].split(".")
        ),
    ):
        pubmed = str(snippet.get("pubmed_id") or "unknown")
        text = " ".join(str(snippet.get("text") or "").split())
        lines.append(f"Snippet {snippet_id} (PubMed {pubmed}): {text}")
    return lines


def auxiliary_prompt(c1: dict[str, Any], gold: dict[str, Any]) -> str:
    evidence = snippet_lines(gold, c1)
    if not evidence:
        raise ValueError(f"{c1['pair_id']}: no supporting snippets for diagnostic prompt")
    user = (
        f"Question: {c1['question']}\n\n"
        "Evidence:\n"
        + "\n".join(evidence)
        + f"\n\nPreferred answer: {c1['positive_answer']}"
        + f"\nRejected answer: {c1['c1_answer']}"
        + f"\nRejected reasoning to critique: {c1['reason']}"
    )
    return qwen_prompt(DIAGNOSTIC_SYSTEM, user)


def auxiliary_target(c1: dict[str, Any], gold: dict[str, Any], coarse_error: str) -> str:
    evidence_claim = " ".join(str(gold.get("evidence_claim") or "").split())
    positive_reason = " ".join(str(c1.get("positive_reason") or gold.get("reason") or "").split())
    c1_basis = " ".join(str(c1.get("c1_basis") or c1.get("basis") or "").split())
    if not evidence_claim or not positive_reason or not c1_basis:
        raise ValueError(f"{c1['pair_id']}: incomplete accepted rationale fields")
    return (
        f"Error: {coarse_error}\n"
        f"Evidence: {evidence_claim}\n"
        f"Why preferred: {positive_reason}\n"
        f"Why rejected: {c1_basis}"
    )


def build_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strict_rows = read_jsonl(STRICT_STAGE1)
    strict_by_pair = {row["pair_id"]: row for row in strict_rows}
    if len(strict_by_pair) != len(strict_rows):
        raise ValueError("Duplicate pair_id in strict Stage-1 pairs")

    c1_rows = read_jsonl(C1_RATIONALES)
    accepted_c1 = [row for row in c1_rows if row.get("status") == "accepted"]
    gold_rows = read_jsonl(GOLD_RATIONALES)
    accepted_gold_by_qid = {
        row["question_id"]: row for row in gold_rows if row.get("status") == "accepted"
    }

    built: list[dict[str, Any]] = []
    mismatch_count = 0
    for c1 in accepted_c1:
        pair_id = c1["pair_id"]
        qid = c1["question_id"]
        if pair_id not in strict_by_pair:
            raise ValueError(f"Accepted annotation has no source pair: {pair_id}")
        if qid not in accepted_gold_by_qid:
            raise ValueError(f"Accepted C1 annotation has no accepted gold rationale: {qid}")
        source_pair = strict_by_pair[pair_id]
        fine_error = str(c1.get("error_type") or "").strip()
        coarse_error = ERROR_TO_GROUP.get(fine_error)
        if coarse_error is None:
            raise ValueError(f"{pair_id}: unmapped error_type={fine_error!r}")

        chosen = str(c1["positive_output"]).strip()
        rejected = str(c1["c1_output"]).strip()
        if rejected != source_pair["rejected"]:
            raise ValueError(f"{pair_id}: rejected output changed since pair construction")
        if chosen != source_pair["chosen"]:
            mismatch_count += 1

        gold = accepted_gold_by_qid[qid]
        built.append(
            {
                **source_pair,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_candidate": c1["positive_answer"],
                "rejected_candidate": c1["c1_answer"],
                "source_pair_chosen": source_pair["chosen"],
                "auxiliary_prompt": auxiliary_prompt(c1, gold),
                "auxiliary_target": auxiliary_target(c1, gold, coarse_error),
                "auxiliary_error_group": coarse_error,
                "auxiliary_error_type": fine_error,
                "auxiliary_relation_type": c1.get("relation_type"),
                "auxiliary_positive_evidence_ids": c1.get("positive_evidence_ids") or [],
                "auxiliary_rejected_evidence_ids": c1.get("evidence_ids") or [],
                "auxiliary_rejected_reason_input_only": c1.get("reason"),
                "auxiliary_annotation_status": c1.get("status"),
                "auxiliary_rubric_version": c1.get("rubric_version"),
            }
        )

    pair_ids = [row["pair_id"] for row in built]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate accepted pair IDs in the built dataset")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    stage1_path = DATASET_ROOT / "dpo_stage1_concept_learning_all_pairs.jsonl"
    write_jsonl(stage1_path, built)
    # The common trainer expects all stage filenames to exist and skips empty ones.
    for filename in (
        "dpo_stage2_format_alignment_all_pairs.jsonl",
        "dpo_stage3_hierarchical_ranking_all_pairs.jsonl",
    ):
        (DATASET_ROOT / filename).write_text("", encoding="utf-8")

    summary = {
        "method": "answer_only_dpo_plus_auxiliary_error_diagnostic_lm",
        "loss": "L_DPO_answer + 0.02 * token_mean_CE_auxiliary",
        "pair_count": len(built),
        "question_count": len({row["question_id"] for row in built}),
        "accepted_c1_annotation_count": len(accepted_c1),
        "excluded_review_annotation_count": len(c1_rows) - len(accepted_c1),
        "chosen_alias_mismatch_with_source_pair_count": mismatch_count,
        "coarse_error_counts": dict(
            sorted(collections.Counter(row["auxiliary_error_group"] for row in built).items())
        ),
        "fine_error_counts": dict(
            sorted(collections.Counter(row["auxiliary_error_type"] for row in built).items())
        ),
        "source_files": {
            "strict_stage1": str(STRICT_STAGE1.resolve()),
            "c1_rationales": str(C1_RATIONALES.resolve()),
            "gold_rationales": str(GOLD_RATIONALES.resolve()),
            "diagnostic_prompt": str(DIAGNOSTIC_PROMPT_SPEC.resolve()),
        },
        "source_sha256": {
            "strict_stage1": sha256_file(STRICT_STAGE1),
            "c1_rationales": sha256_file(C1_RATIONALES),
            "gold_rationales": sha256_file(GOLD_RATIONALES),
            "diagnostic_prompt": sha256_file(DIAGNOSTIC_PROMPT_SPEC),
        },
        "stage1_file": str(stage1_path.resolve()),
        "diagnostic_labels": sorted(ERROR_GROUPS),
        "rejected_rationale_usage": "input_only_never_target",
    }
    write_json(DATASET_ROOT / "summary.json", summary)
    return built, summary


def token_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from cse_dpo.train_factoid_three_stage_dpo import _encode_pair

    tokenizer = AutoTokenizer.from_pretrained(
        str(INITIAL_ADAPTER), trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = []
    too_long = []
    for row in rows:
        try:
            item = _encode_pair(
                tokenizer,
                row,
                MAX_LENGTH,
                append_eos=True,
                allow_truncation=False,
            )
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
            too_long.append(
                {
                    "pair_id": row["pair_id"],
                    "question_id": row["question_id"],
                    "reason": str(exc),
                }
            )
            continue
        lengths.append(
            {
                "pair_id": row["pair_id"],
                "answer_max": max(len(item["chosen_ids"]), len(item["rejected_ids"])),
                "auxiliary": len(item["auxiliary_ids"]),
                "overall": max(
                    len(item["chosen_ids"]),
                    len(item["rejected_ids"]),
                    len(item["auxiliary_ids"]),
                ),
            }
        )
    summary = {
        "max_length": MAX_LENGTH,
        "rows": len(rows),
        "rows_within_limit": len(lengths),
        "rows_over_limit": len(too_long),
        "answer_sequence_max": max((row["answer_max"] for row in lengths), default=None),
        "auxiliary_sequence_max": max((row["auxiliary"] for row in lengths), default=None),
        "overall_sequence_max": max((row["overall"] for row in lengths), default=None),
        "over_limit_rows_file": (
            str((DATASET_ROOT / "over_length_rows.json").resolve()) if too_long else None
        ),
    }
    if too_long:
        write_json(DATASET_ROOT / "over_length_rows.json", too_long)
    write_json(DATASET_ROOT / "token_preflight.json", summary)
    return summary


def training_config():
    from cse_dpo.train_factoid_three_stage_dpo import Config

    return Config(
        model_preset="qwen25_05b",
        dataset_preset="gold_supported_strict_equivalence_v2_1130",
        staged_root=str(DATASET_ROOT),
        base_model=str(BASE_MODEL),
        initial_adapter=str(INITIAL_ADAPTER),
        output_root=str(OUTPUT_ROOT),
        seed=3407,
        beta=0.1,
        objective="dpo",
        auxiliary_diagnostic_weight=AUXILIARY_WEIGHT,
        learning_rate=5e-6,
        optimizer="AdamW",
        weight_decay=0.01,
        epochs=8,
        batch_size=1,
        gradient_accumulation=8,
        max_length=MAX_LENGTH,
        train_backprop_max_length=MAX_LENGTH,
        eval_fraction=0.2,
        eval_seed=3407,
        eval_every_updates=25,
        early_stopping_patience=10,
        selection_metric="dev_mrr",
        evaluate_generated_dev=True,
        generated_eval_max_seq_length=MAX_LENGTH,
        generated_eval_max_new_tokens=64,
        allow_truncated_examples=False,
        drop_truncated_examples=True,
        stop_after_stage="concept_learning",
        smoke_test=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start GPU training after building and validating the frozen dataset.",
    )
    parser.add_argument(
        "--skip-token-preflight",
        action="store_true",
        help="Skip tokenizer length validation (dataset validation still runs).",
    )
    args = parser.parse_args()
    ensure_project_python()

    rows, dataset_summary = build_dataset()
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
    if not args.skip_token_preflight:
        preflight = token_preflight(rows)
        print(json.dumps({"token_preflight": preflight}, indent=2, ensure_ascii=False))

    cfg = training_config()
    write_json(
        DATASET_ROOT / "planned_training_config.json",
        {
            "config": cfg.__dict__,
            "output_root": str(OUTPUT_ROOT.resolve()),
            "training_requested": args.run,
        },
    )
    if not args.run:
        print("\nPrepared successfully. Training was not started.")
        print(
            "Run:\n"
            f'PYTHONPATH=. "{PROJECT_PYTHON}" '
            "cse_dpo/run_factoid_error_aware_multitask_dpo_qwen25_05b.py --run"
        )
        return

    from cse_dpo.train_factoid_three_stage_dpo import run_three_stage

    run_three_stage(cfg)


if __name__ == "__main__":
    main()

