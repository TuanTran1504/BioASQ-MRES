#!/usr/bin/env python
"""Compare independent frequency voting with history-conditioned candidate search.

Arms
----
1. independent_sample10_frequency_top5:
   reuse the frozen ten-sample bank and rank unique candidates by frequency.
2. independent_sample5_natural_top5:
   use the first five frozen independent samples in generation order.
3. conditioned_sample5_natural_top5:
   reuse independent sample 1, then generate turns 2--5 while showing the
   model all previously discovered valid candidates and asking for a different
   evidence-supported answer.

All arms use the same original 0.5B SFT adapter and official BioASQ Java scorer.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = Path("/home/dinh-tuan/miniconda3/envs/bioasq/bin/python")
BASE_MODEL = ROOT / "models/Qwen2.5-0.5B-Instruct"
ADAPTER = (
    ROOT
    / "Artifacts/Factoid_SFT/models/"
    "evidence_grounded_per_supported_alias_qwen25_05b_lora_dropout_005_strict_extractive/"
    "adapter_best_evidence_mrr"
)
DEV_INPUT = (
    ROOT
    / "data/BioASQ_factoid_sft_prepared/"
    "single_answer_full_resources_qwen25_05b/eval_prepared.json"
)
PROMPT_REGISTRY = ROOT / "prompts/factoid_single_answer_aligned.json"
PROMPT_REF = "factoid-single-answer-extractive-v1"
FROZEN_SAMPLE10_BANK = (
    ROOT
    / "Artifacts/cse_dpo/inference_strategy_comparisons/"
    "qwen25_05b_stage1_dpo_vs_original_sft_dev_sample10_t07_seed3407/"
    "original_sft/raw_generations.jsonl"
)
OUTPUT_ROOT = (
    ROOT
    / "Artifacts/cse_dpo/inference_strategy_comparisons/"
    "qwen25_05b_sft_frequency10_vs_conditioned5_t07_seed3407"
)

TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 3407
CONDITIONED_SEED_OFFSET = 100_000
MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 64
MAX_ANSWERS = 5
USE_CACHE = False

HISTORY_INSTRUCTION = """Candidate-search history:
{history}

Generate one additional biomedical answer candidate that is explicitly supported
by the PubMed resources. Search for a different relevant target and do not repeat
a candidate listed above. Return exactly one expression in the required
[BE] extracted expression [EE] format."""


def ensure_project_python() -> None:
    current = Path(sys.executable).resolve()
    expected = PROJECT_PYTHON.resolve()
    if current == expected:
        return
    if not expected.exists():
        raise RuntimeError(f"Missing required interpreter: {expected}")
    print(f"[environment] Restarting with {expected}", flush=True)
    os.execv(str(expected), [str(expected), *sys.argv])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_jsonl_latest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["question_id"])] = row
    return rows


def weak_vote_key(candidate: str) -> str:
    # Match the earlier frequency experiment: punctuation and hyphens matter.
    return " ".join(unicodedata.normalize("NFKC", candidate).casefold().split())


def source_args(limit: int | None):
    return type(
        "SourceArgs",
        (),
        {
            "question_types": ["factoid"],
            "max_summary_answers": 5,
            "max_factoid_answers": 5,
            "max_list_items": 100,
            "max_resources": 0,
            "max_resource_chars": 0,
            "resource_selection": "first",
            "resource_granularity": "document",
            "resource_window_mode": "single",
            "resource_reranker_model": None,
            "resource_reranker_article_model": None,
            "resource_reranker_device": None,
            "resource_reranker_batch_size": 1,
            "local_files_only": True,
            "limit": limit,
        },
    )()


def load_examples(limit: int | None):
    from cse_dpo.generated_bioasq_eval import load_gold_examples
    from src.prompt_registry import resolve_prompt_bundle
    from src.utility.config import QUESTION_INSTRUCTIONS
    from src.utility.eval_dataset import load_eval_examples

    bundle = resolve_prompt_bundle(PROMPT_REGISTRY, PROMPT_REF, QUESTION_INSTRUCTIONS)
    examples = load_eval_examples([DEV_INPUT], source_args(limit), bundle["instructions"])
    gold = load_gold_examples([DEV_INPUT])
    examples = [example for example in examples if example.question_id in gold]
    if limit is not None:
        examples = examples[:limit]
    expected = limit if limit is not None else 160
    if len(examples) != expected:
        raise ValueError(f"Expected {expected} examples, found {len(examples)}")
    return bundle, examples, gold


def extract_single_candidate(sample: str | None) -> str | None:
    from src.utility.bioasq_format import parse_prediction_items
    from src.utility.data import clean_text

    if not sample:
        return None
    items = parse_prediction_items(sample, "factoid")
    if len(items) != 1:
        return None
    candidate = clean_text(items[0])
    return candidate or None


def unique_in_order(samples: list[str], n: int = 5) -> tuple[list[str], dict[str, Any]]:
    ranked: list[str] = []
    seen: set[str] = set()
    invalid: list[dict[str, Any]] = []
    duplicates = 0
    for index, sample in enumerate(samples[:n]):
        candidate = extract_single_candidate(sample)
        if candidate is None:
            invalid.append({"sample_index": index, "sample": sample})
            continue
        key = weak_vote_key(candidate)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        ranked.append(candidate)
    return ranked[:MAX_ANSWERS], {
        "samples_considered": min(n, len(samples)),
        "valid_unique_candidates": len(ranked),
        "duplicate_count": duplicates,
        "invalid_samples": invalid,
    }


def rank_by_frequency(samples: list[str]) -> tuple[list[str], dict[str, Any]]:
    votes: collections.Counter[str] = collections.Counter()
    representative: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    invalid: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        candidate = extract_single_candidate(sample)
        if candidate is None:
            invalid.append({"sample_index": index, "sample": sample})
            continue
        key = weak_vote_key(candidate)
        votes[key] += 1
        representative.setdefault(key, candidate)
        first_seen.setdefault(key, index)
    keys = sorted(votes, key=lambda key: (-votes[key], first_seen[key]))
    return [representative[key] for key in keys[:MAX_ANSWERS]], {
        "samples_considered": len(samples),
        "valid_sample_count": int(sum(votes.values())),
        "unique_candidate_count": len(votes),
        "duplicate_count": int(sum(votes.values()) - len(votes)),
        "modal_frequency": max(votes.values(), default=0),
        "votes": {representative[key]: int(votes[key]) for key in keys},
        "invalid_samples": invalid,
    }


def format_prediction(candidates: list[str]) -> str:
    from src.utility.data import clean_text

    return " ".join(
        f"[BE] {clean_text(candidate)} [EE]"
        for candidate in candidates[:MAX_ANSWERS]
        if clean_text(candidate)
    )


def add_history_to_prompt(prompt: str, previous: list[str]) -> str:
    if not previous:
        return prompt
    history = "\n".join(f"{index}. {candidate}" for index, candidate in enumerate(previous, 1))
    instruction = HISTORY_INSTRUCTION.format(history=history)
    marker = "<|im_end|>\n<|im_start|>assistant"
    position = prompt.rfind(marker)
    if position < 0:
        raise ValueError("Could not locate the final Qwen user/assistant boundary")
    return prompt[:position].rstrip() + "\n\n" + instruction + "\n" + prompt[position:]


def load_model_and_tokenizer():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for conditioned sampling")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER), local_files_only=True)
    model.config.use_cache = USE_CACHE
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, seed: int) -> str:
    import torch
    from src.utility.data import clean_text

    old_side = tokenizer.truncation_side
    try:
        tokenizer.truncation_side = "left"
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
    finally:
        tokenizer.truncation_side = old_side
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=USE_CACHE,
        )
    prompt_length = encoded["input_ids"].shape[-1]
    text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return re.sub(r"^answer\s*:\s*", "", clean_text(text), flags=re.I).strip()


def generate_conditioned_bank(bundle, examples, frozen, output_dir: Path):
    from src.utility.eval_dataset import render_prompt
    from tqdm.auto import tqdm

    raw_path = output_dir / "conditioned_sample5_raw.jsonl"
    completed = read_jsonl_latest(raw_path)
    missing = [example for example in examples if example.question_id not in completed]
    print(f"Conditioned generation: {len(completed)} cached, {len(missing)} remaining")
    if not missing:
        return completed

    model, tokenizer = load_model_and_tokenizer()
    index_by_id = {example.question_id: index for index, example in enumerate(examples)}
    try:
        with raw_path.open("a", encoding="utf-8") as handle:
            for example in tqdm(missing, desc="Conditioned sample-5"):
                qid = example.question_id
                base_prompt = render_prompt(
                    tokenizer,
                    example,
                    chat_template=bundle.get("chat_template", "qwen-2.5"),
                    prompt_format="chat",
                )
                samples = [str(frozen[qid]["samples"][0])]
                turn_records = [
                    {
                        "turn": 1,
                        "seed": frozen[qid].get("sample_seeds", [None])[0],
                        "previous_candidates": [],
                        "generation": samples[0],
                        "reused_frozen_independent_sample": True,
                    }
                ]
                for turn in range(2, 6):
                    previous, _ = unique_in_order(samples, len(samples))
                    prompt = add_history_to_prompt(base_prompt, previous)
                    seed = (
                        CONDITIONED_SEED_OFFSET
                        + SEED
                        + index_by_id[qid] * 5
                        + (turn - 1)
                    )
                    generated = generate_one(model, tokenizer, prompt, seed)
                    samples.append(generated)
                    turn_records.append(
                        {
                            "turn": turn,
                            "seed": seed,
                            "previous_candidates": previous,
                            "generation": generated,
                            "reused_frozen_independent_sample": False,
                        }
                    )
                row = {
                    "question_id": qid,
                    "question": example.body,
                    "samples": samples,
                    "turns": turn_records,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[qid] = row
    finally:
        del model, tokenizer
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    return completed


def official_score_args():
    return type(
        "OfficialArgs",
        (),
        {
            "bioasq_java_jar": str(
                ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
            ),
            "bioasq_java_heap": "512m",
            "bioasq_java_version": 5,
        },
    )()


def evaluate_strategy(
    examples,
    gold,
    strategy: str,
    frozen: dict[str, dict[str, Any]],
    conditioned: dict[str, dict[str, Any]],
    output_dir: Path,
):
    import pandas as pd
    from src.utility.bioasq_official import evaluate_with_bioasq_java

    strategy_dir = output_dir / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for example in examples:
        qid = example.question_id
        if strategy == "independent_sample10_frequency_top5":
            ranked, diagnostics = rank_by_frequency(frozen[qid]["samples"])
        elif strategy == "independent_sample5_natural_top5":
            ranked, diagnostics = unique_in_order(frozen[qid]["samples"], 5)
        elif strategy == "conditioned_sample5_natural_top5":
            ranked, diagnostics = unique_in_order(conditioned[qid]["samples"], 5)
        else:
            raise ValueError(strategy)
        rows.append(
            {
                "question_id": qid,
                "question_type": example.question_type,
                "body": example.body,
                "prediction": format_prediction(ranked),
                "ranked_candidates": ranked,
                "gold_output": example.gold_output,
                **diagnostics,
            }
        )
    official = evaluate_with_bioasq_java(
        prediction_rows=rows,
        examples_by_key={
            (row["question_id"], row["question_type"]): gold[row["question_id"]]
            for row in rows
        },
        model_label=strategy,
        model_dir=strategy_dir,
        args=official_score_args(),
        include_per_question=True,
    )
    per_question = {row["question_id"]: row for row in official["per_question"]}
    enriched = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                key: value
                for key, value in per_question[row["question_id"]].items()
                if key not in {"question_id", "question_type"}
            }
        )
        enriched.append(item)
    metrics = official["aggregate"]["by_type"]["factoid"]["metrics"]
    summary = {
        "strategy": strategy,
        "question_count": len(rows),
        "mrr": metrics["mrr"],
        "strict_accuracy": metrics["strict_accuracy"],
        "lenient_accuracy": metrics["lenient_accuracy"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "generation_calls_per_question": (
            10 if strategy == "independent_sample10_frequency_top5" else 5
        ),
        "mean_unique_candidates": sum(len(row["ranked_candidates"]) for row in rows) / len(rows),
        "mean_duplicate_count": sum(row["duplicate_count"] for row in rows) / len(rows),
        "questions_with_five_unique_candidates": sum(
            len(row["ranked_candidates"]) == 5 for row in rows
        ),
        "official_scores": str(
            (strategy_dir / "official_bioasq/official_scores.json").resolve()
        ),
    }
    write_json(strategy_dir / "summary.json", summary)
    write_json(strategy_dir / "per_question_metrics.json", enriched)
    pd.DataFrame(enriched).to_csv(strategy_dir / "per_question_metrics.csv", index=False)
    return summary, enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Do not generate; require a complete conditioned cache.",
    )
    args = parser.parse_args()
    ensure_project_python()

    for path in (BASE_MODEL, ADAPTER, DEV_INPUT, PROMPT_REGISTRY, FROZEN_SAMPLE10_BANK):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = (
        OUTPUT_ROOT if args.limit is None else OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}_smoke{args.limit}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, examples, gold = load_examples(args.limit)
    frozen_all = read_jsonl_latest(FROZEN_SAMPLE10_BANK)
    frozen = {example.question_id: frozen_all[example.question_id] for example in examples}
    for qid, row in frozen.items():
        if len(row.get("samples") or []) != 10:
            raise ValueError(f"{qid}: frozen bank does not contain exactly ten samples")

    config = {
        "base_model": str(BASE_MODEL.resolve()),
        "adapter": str(ADAPTER.resolve()),
        "dev_input": str(DEV_INPUT.resolve()),
        "prompt_ref": PROMPT_REF,
        "frozen_sample10_bank": str(FROZEN_SAMPLE10_BANK.resolve()),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "conditioned_seed_offset": CONDITIONED_SEED_OFFSET,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "limit": args.limit,
        "conditioned_turns": 5,
        "conditioned_turn1": "reused frozen independent sample 1",
        "conditioned_turns_2_to_5": "history-conditioned new generations",
        "history_instruction": HISTORY_INSTRUCTION,
        "frequency_normalization": "NFKC + casefold + whitespace collapse; punctuation preserved",
    }
    config_path = output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"Existing output has a different configuration: {output_dir}")
    write_json(config_path, config)

    conditioned_cache = read_jsonl_latest(output_dir / "conditioned_sample5_raw.jsonl")
    if args.score_only:
        missing = [
            example.question_id
            for example in examples
            if example.question_id not in conditioned_cache
        ]
        if missing:
            raise RuntimeError(f"Conditioned cache is incomplete; first missing IDs: {missing[:5]}")
        conditioned = conditioned_cache
    else:
        conditioned = generate_conditioned_bank(bundle, examples, frozen, output_dir)

    strategies = [
        "independent_sample10_frequency_top5",
        "independent_sample5_natural_top5",
        "conditioned_sample5_natural_top5",
    ]
    summaries = []
    results = {}
    for strategy in strategies:
        summary, rows = evaluate_strategy(
            examples, gold, strategy, frozen, conditioned, output_dir
        )
        summaries.append(summary)
        results[strategy] = {row["question_id"]: row for row in rows}
        print(json.dumps(summary, indent=2))

    baseline = results["independent_sample10_frequency_top5"]
    conditioned_rows = results["conditioned_sample5_natural_top5"]
    gains, losses, ties = [], [], 0
    comparison = []
    for example in examples:
        qid = example.question_id
        left = baseline[qid]
        right = conditioned_rows[qid]
        delta = float(right.get("mrr", 0.0)) - float(left.get("mrr", 0.0))
        if delta > 0:
            gains.append(qid)
        elif delta < 0:
            losses.append(qid)
        else:
            ties += 1
        comparison.append(
            {
                "question_id": qid,
                "body": example.body,
                "frequency10_prediction": left["prediction"],
                "frequency10_ranked_candidates": left["ranked_candidates"],
                "frequency10_mrr": left.get("mrr"),
                "conditioned5_prediction": right["prediction"],
                "conditioned5_ranked_candidates": right["ranked_candidates"],
                "conditioned5_mrr": right.get("mrr"),
                "conditioned5_minus_frequency10_mrr": delta,
            }
        )
    import pandas as pd

    pd.DataFrame(comparison).to_csv(output_dir / "question_level_comparison.csv", index=False)
    final = {
        "status": "complete",
        "summaries": summaries,
        "conditioned5_vs_frequency10": {
            "questions_improved": len(gains),
            "questions_worsened": len(losses),
            "questions_tied": ties,
            "improved_question_ids": gains,
            "worsened_question_ids": losses,
            "mrr_difference": next(
                x["mrr"] for x in summaries
                if x["strategy"] == "conditioned_sample5_natural_top5"
            )
            - next(
                x["mrr"] for x in summaries
                if x["strategy"] == "independent_sample10_frequency_top5"
            ),
        },
    }
    write_json(output_dir / "summary.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

