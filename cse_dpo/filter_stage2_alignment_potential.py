#!/usr/bin/env python
"""Build a MAP/alignment-potential filtered Stage-2 curriculum dataset.

The script scores Stage-2 C3>C2 pairs with the best Stage-1 adapter, ranks by
an alignment-potential proxy, then applies two practical filters:

1. reject overly long C2 answers;
2. cap selected Stage-2 pairs per question.

Stage 1 and Stage 3 are copied unchanged into the output staged root.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cse_dpo import train_factoid_three_stage_dpo as three_stage

DEFAULT_SOURCE_STAGED_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs"
DEFAULT_RUN_ROOT = ROOT / "Artifacts/cse_dpo/three_stage_dpo_qwen25_05b_c3_c1_curriculum_gold_supported_full_bp4096"
DEFAULT_OUTPUT_ROOT = ROOT / "Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v1_gold_c3_gold_supported/merged/staged_curriculum_pairs_map_stage2_top30_cap2_lenfilter_05b"

STAGE1_FILE = "dpo_stage1_concept_learning_all_pairs.jsonl"
STAGE2_FILE = "dpo_stage2_format_alignment_all_pairs.jsonl"
STAGE3_FILE = "dpo_stage3_hierarchical_ranking_all_pairs.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"\[/?b?e\]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def text_similarity(a: Any, b: Any) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def word_count(value: Any) -> int:
    return len(norm_text(value).split())


def zscore(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    if not np.isfinite(sigma) or sigma == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sigma


def stage_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "pairs": len(rows),
        "questions": len({row.get("question_id") for row in rows}),
        "pair_classes": dict(Counter(f"{row.get('chosen_class')}>{row.get('rejected_class')}" for row in rows)),
        "sha256": sha256(path),
        "path": str(path),
    }


def load_stage1_adapter(run_root: Path) -> Path:
    manifest_path = run_root / "stage_1_concept_learning" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter = Path(manifest.get("best_selected_adapter") or manifest.get("best_mrr_adapter") or manifest.get("adapter") or "")
    if not adapter.exists():
        raise FileNotFoundError(f"Stage-1 adapter not found from {manifest_path}: {adapter}")
    return adapter


def build_base_records(rows: list[dict[str, Any]]) -> pd.DataFrame:
    class_rank = {"C1": 0, "C2": 1, "C3": 2}
    records = []
    for row in rows:
        chosen_text = row.get("chosen_candidate") or row.get("chosen")
        rejected_text = row.get("rejected_candidate") or row.get("rejected")
        chosen_rank = class_rank.get(row.get("chosen_class"), math.nan)
        rejected_rank = class_rank.get(row.get("rejected_class"), math.nan)
        records.append({
            "pair_id": row.get("pair_id"),
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "chosen_candidate": chosen_text,
            "rejected_candidate": rejected_text,
            "chosen_class": row.get("chosen_class"),
            "rejected_class": row.get("rejected_class"),
            "chosen_source_model": row.get("chosen_source_model"),
            "rejected_source_model": row.get("rejected_source_model"),
            "chosen_evidence_support": row.get("chosen_evidence_support"),
            "rejected_evidence_support": row.get("rejected_evidence_support"),
            "explicit_class_margin": chosen_rank - rejected_rank,
            "string_similarity": text_similarity(chosen_text, rejected_text),
            "chosen_len_chars": len(str(chosen_text or "")),
            "rejected_len_chars": len(str(rejected_text or "")),
            "chosen_words": word_count(chosen_text),
            "rejected_words": word_count(rejected_text),
        })
    df = pd.DataFrame(records)
    df["rejected_to_chosen_word_ratio"] = df["rejected_words"] / df["chosen_words"].clip(lower=1)
    return df


def compute_logprob_scores(args: argparse.Namespace, rows: list[dict[str, Any]], adapter: Path, output_root: Path) -> dict[str, dict[str, Any]]:
    cfg = three_stage.Config(
        model_preset=args.model_preset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        append_eos_to_completions=args.append_eos_to_completions,
        allow_truncated_examples=args.allow_truncation,
        drop_truncated_examples=not args.allow_truncation,
        lora_dropout=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = []
    dropped = []
    for row in tqdm(rows, desc="Tokenizing Stage-2 pairs"):
        try:
            encoded.append(three_stage._encode_pair(
                tokenizer,
                row,
                args.max_length,
                append_eos=args.append_eos_to_completions,
                allow_truncation=args.allow_truncation,
            ))
        except Exception as exc:
            dropped.append({"pair_id": row.get("pair_id"), "question_id": row.get("question_id"), "error": str(exc)})
    if dropped:
        write_jsonl(output_root / "dropped_stage2_pairs_for_margin.jsonl", dropped)
        write_csv(output_root / "dropped_stage2_pairs_for_margin.csv", dropped)
    print(f"Encoded {len(encoded):,}; dropped {len(dropped):,}")

    model = three_stage._load_policy(cfg, adapter)
    scores = three_stage._reference_scores(model, encoded, tokenizer, cfg)
    scored: dict[str, dict[str, Any]] = {}
    for item, (chosen_logp, rejected_logp) in zip(encoded, scores):
        pair_id = item["row"]["pair_id"]
        chosen_tokens = int(item["chosen_tokens"])
        rejected_tokens = int(item["rejected_tokens"])
        chosen_logp = float(chosen_logp)
        rejected_logp = float(rejected_logp)
        scored[pair_id] = {
            "chosen_logp_sum": chosen_logp,
            "rejected_logp_sum": rejected_logp,
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "prompt_tokens": int(item["prompt_tokens"]),
            "chosen_logp_norm": chosen_logp / max(1, chosen_tokens),
            "rejected_logp_norm": rejected_logp / max(1, rejected_tokens),
            "implicit_margin_sum": chosen_logp - rejected_logp,
            "implicit_margin_norm": (chosen_logp / max(1, chosen_tokens)) - (rejected_logp / max(1, rejected_tokens)),
        }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scored


def make_filtered_dataset(args: argparse.Namespace) -> None:
    source_root = Path(args.source_staged_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_src = source_root / STAGE1_FILE
    stage2_src = source_root / STAGE2_FILE
    stage3_src = source_root / STAGE3_FILE
    for path in [stage1_src, stage2_src, stage3_src]:
        if not path.exists():
            raise FileNotFoundError(path)

    rows = read_jsonl(stage2_src)
    df = build_base_records(rows)
    adapter = load_stage1_adapter(Path(args.run_root))
    scored = compute_logprob_scores(args, rows, adapter, output_root) if args.compute_model_logprobs else {}
    if scored:
        score_df = pd.DataFrame.from_dict(scored, orient="index").reset_index(names="pair_id")
        df = df.merge(score_df, on="pair_id", how="left")

    if "implicit_margin_norm" in df.columns and df["implicit_margin_norm"].notna().any():
        df["abs_implicit_margin_norm"] = df["implicit_margin_norm"].abs()
        df["explicit_margin_z"] = zscore(df["explicit_class_margin"].fillna(df["explicit_class_margin"].median()))
        df["abs_implicit_margin_z"] = zscore(df["abs_implicit_margin_norm"].fillna(df["abs_implicit_margin_norm"].median()))
        df["alignment_potential_score"] = df["explicit_margin_z"] - df["abs_implicit_margin_z"]
        df["signed_gap_score"] = df["explicit_class_margin"] - df["implicit_margin_norm"].fillna(0.0)
        df["model_prefers_chosen"] = df["implicit_margin_norm"] > 0
    else:
        df["abs_implicit_margin_norm"] = np.nan
        df["explicit_margin_z"] = zscore(df["explicit_class_margin"].fillna(df["explicit_class_margin"].median()))
        df["abs_implicit_margin_z"] = np.nan
        df["alignment_potential_score"] = df["explicit_margin_z"]
        df["signed_gap_score"] = np.nan
        df["model_prefers_chosen"] = np.nan

    df["near_duplicate_text"] = df["string_similarity"] >= args.near_duplicate_similarity
    df["rejected_too_long_words"] = df["rejected_words"] > args.max_rejected_words
    df["rejected_too_long_ratio"] = df["rejected_to_chosen_word_ratio"] > args.max_rejected_to_chosen_word_ratio
    df["rejected_too_long"] = df["rejected_too_long_words"] | df["rejected_too_long_ratio"]
    df["eligible_length"] = ~df["rejected_too_long"]

    ranked_all = df.sort_values(
        ["alignment_potential_score", "string_similarity"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked_all["alignment_potential_rank_all"] = np.arange(1, len(ranked_all) + 1)
    eligible = ranked_all[ranked_all["eligible_length"]].copy().reset_index(drop=True)
    eligible["alignment_potential_rank_eligible"] = np.arange(1, len(eligible) + 1)

    top_n = max(1, int(round(args.select_ratio * len(rows))))
    selected_records = []
    per_question_count: dict[str, int] = defaultdict(int)
    for record in eligible.to_dict(orient="records"):
        qid = str(record["question_id"])
        if per_question_count[qid] >= args.max_pairs_per_question:
            continue
        selected_records.append(record)
        per_question_count[qid] += 1
        if len(selected_records) >= top_n:
            break

    selected_ids = {record["pair_id"] for record in selected_records}
    selected_rows = []
    by_pair = {row.get("pair_id"): row for row in rows}
    for record in selected_records:
        row = dict(by_pair[record["pair_id"]])
        row["alignment_potential_score"] = None if pd.isna(record.get("alignment_potential_score")) else float(record["alignment_potential_score"])
        row["implicit_margin_norm"] = None if pd.isna(record.get("implicit_margin_norm")) else float(record["implicit_margin_norm"])
        row["abs_implicit_margin_norm"] = None if pd.isna(record.get("abs_implicit_margin_norm")) else float(record["abs_implicit_margin_norm"])
        row["alignment_potential_rank_eligible"] = int(record["alignment_potential_rank_eligible"])
        row["stage2_filter"] = {
            "select_ratio": args.select_ratio,
            "max_pairs_per_question": args.max_pairs_per_question,
            "max_rejected_words": args.max_rejected_words,
            "max_rejected_to_chosen_word_ratio": args.max_rejected_to_chosen_word_ratio,
        }
        selected_rows.append(row)

    # Copy Stage 1 and 3 unchanged, replace Stage 2.
    shutil.copy2(stage1_src, output_root / STAGE1_FILE)
    shutil.copy2(stage3_src, output_root / STAGE3_FILE)
    write_jsonl(output_root / STAGE2_FILE, selected_rows)
    write_csv((output_root / STAGE2_FILE).with_suffix(".csv"), selected_rows)

    ranked_all.to_csv(output_root / "stage2_alignment_potential_scored_all.csv", index=False)
    eligible.to_csv(output_root / "stage2_alignment_potential_eligible_after_length_filter.csv", index=False)
    write_jsonl(output_root / "stage2_alignment_potential_selected_records.jsonl", selected_records)
    pd.DataFrame(selected_records).to_csv(output_root / "stage2_alignment_potential_selected_records.csv", index=False)

    # CSVs for unchanged copied stages.
    write_csv((output_root / STAGE1_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE1_FILE))
    write_csv((output_root / STAGE3_FILE).with_suffix(".csv"), read_jsonl(output_root / STAGE3_FILE))

    summary = {
        "description": "Staged curriculum with Stage 2 filtered by MAP/alignment-potential, a per-question cap, and a long-C2 rejection filter. Stage 1 and Stage 3 are unchanged.",
        "source_staged_root": str(source_root),
        "run_root": str(args.run_root),
        "stage1_adapter_for_margin": str(adapter),
        "model_preset": args.model_preset,
        "select_ratio": args.select_ratio,
        "target_top_n_before_cap": top_n,
        "max_pairs_per_question": args.max_pairs_per_question,
        "max_rejected_words": args.max_rejected_words,
        "max_rejected_to_chosen_word_ratio": args.max_rejected_to_chosen_word_ratio,
        "near_duplicate_similarity": args.near_duplicate_similarity,
        "source_stage2_pairs": len(rows),
        "length_eligible_stage2_pairs": int(len(eligible)),
        "selected_stage2_pairs": len(selected_rows),
        "selected_stage2_questions": len({row.get("question_id") for row in selected_rows}),
        "removed_by_length_filter": int((~df["eligible_length"]).sum()),
        "stages": {
            STAGE1_FILE: stage_summary(output_root / STAGE1_FILE),
            STAGE2_FILE: stage_summary(output_root / STAGE2_FILE),
            STAGE3_FILE: stage_summary(output_root / STAGE3_FILE),
        },
        "score_files": {
            "all_scored_csv": str(output_root / "stage2_alignment_potential_scored_all.csv"),
            "eligible_csv": str(output_root / "stage2_alignment_potential_eligible_after_length_filter.csv"),
            "selected_csv": str(output_root / "stage2_alignment_potential_selected_records.csv"),
            "selected_jsonl": str(output_root / "stage2_alignment_potential_selected_records.jsonl"),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-staged-root", type=Path, default=DEFAULT_SOURCE_STAGED_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-preset", default="qwen25_05b")
    parser.add_argument("--select-ratio", type=float, default=0.30)
    parser.add_argument("--max-pairs-per-question", type=int, default=2)
    parser.add_argument("--max-rejected-words", type=int, default=12)
    parser.add_argument("--max-rejected-to-chosen-word-ratio", type=float, default=4.0)
    parser.add_argument("--near-duplicate-similarity", type=float, default=0.85)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--append-eos-to-completions", action="store_true", default=True)
    parser.add_argument("--no-append-eos-to-completions", dest="append_eos_to_completions", action="store_false")
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--compute-model-logprobs", dest="compute_model_logprobs", action="store_true", default=True)
    parser.add_argument("--no-compute-model-logprobs", dest="compute_model_logprobs", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    make_filtered_dataset(parse_args())
