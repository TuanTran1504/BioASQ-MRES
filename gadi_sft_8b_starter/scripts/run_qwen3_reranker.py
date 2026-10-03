#!/usr/bin/env python3
"""Preflight, evaluate, or LoRA-fine-tune Qwen3-Reranker on BioASQ pools.

Every neural input contains the question, one candidate answer, and every
supplied snippet in original order. Inputs that exceed the configured context
limit fail preflight; evidence is never selected, dropped, or truncated.

Gold-derived exact-match labels define training pairs and evaluation only.
Gold aliases are never encoded. Source, rank, relation, format, and literal
evidence-occurrence metadata can be encoded in an explicit ablation because
these fields are available at inference. Questions without an accepted
candidate are excluded from the pairwise training loss but are still ranked
during evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold


STARTER_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = ROOT / (
    "Artifacts/reranker_pilot/20261002-023517-tfidf-logistic/"
    "candidate_pool_labeled.jsonl"
)
DEFAULT_EXAMPLES = STARTER_ROOT / (
    "outputs/model_comparison/"
    "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/"
    "examples.jsonl"
)
DEFAULT_OUTPUT_PARENT = STARTER_ROOT / "outputs/qwen3_reranker"
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_REVISION = "fd9fb1d26c07223ced488065909faf522e29cc7d"
DEFAULT_INSTRUCTION = (
    "Given a biomedical factoid question, a candidate answer, and supplied evidence "
    "snippets, determine whether the candidate is the correct concise answer supported "
    "by the evidence. Treat synonymous biomedical names and numerically equivalent "
    "forms as correct, but reject related, broader, narrower, or contradictory concepts."
)
INFERENCE_METADATA_FIELDS = [
    "generator sources",
    "source ranks",
    "independent generator count",
    "best source rank",
    "reciprocal-rank sum",
    "relation types",
    "surface operations",
    "format-variant indicator",
    "literal evidence occurrence",
    "supporting snippet count",
    "candidate word count",
    "candidate character count",
]
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def key(value: Any) -> str:
    return clean(value).casefold()


def validate_data(
    pool_rows: list[dict[str, Any]],
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {"question_id", "answer", "sources", "source_ranks", "label"}
    for index, row in enumerate(pool_rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Candidate row {index} is missing {sorted(missing)}")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"Candidate row {index} has a non-binary label")
    pool_ids = {row["question_id"] for row in pool_rows}
    if pool_ids != set(examples):
        raise ValueError(
            f"Question mismatch: pool={len(pool_ids)}, examples={len(examples)}, "
            f"pool_only={len(pool_ids-set(examples))}, examples_only={len(set(examples)-pool_ids)}"
        )
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_question[row["question_id"]].append(row)
    answerable = sum(any(int(row["label"]) for row in rows) for rows in by_question.values())
    return {
        "question_count": len(by_question),
        "candidate_count": len(pool_rows),
        "positive_candidate_count": sum(int(row["label"]) for row in pool_rows),
        "answerable_pool_count": answerable,
        "unanswerable_pool_count": len(by_question) - answerable,
    }


def format_document(example: dict[str, Any]) -> str:
    """Serialize every supplied snippet in original order without selection."""
    parts = []
    for index, snippet in enumerate(example["snippets"], 1):
        identifier = snippet.get("id") or snippet.get("snippet_id") or str(index)
        parts.append(f"[Snippet {identifier}] {clean(snippet['text'])}")
    return "Supplied evidence snippets:\n" + "\n".join(parts)


def format_candidate_metadata(
    example: dict[str, Any],
    row: dict[str, Any],
) -> str:
    """Serialize only candidate attributes available during inference."""
    answer = clean(row["answer"])
    sources = sorted(clean(source) for source in row.get("sources", []) if clean(source))
    base_sources = sorted({source.removeprefix("format::") for source in sources})
    source_ranks = {
        clean(source): int(rank)
        for source, rank in row.get("source_ranks", {}).items()
        if clean(source)
    }
    ordered_ranks = sorted(source_ranks.items())
    positive_ranks = [rank for rank in source_ranks.values() if rank > 0]
    relation_types = sorted({
        clean(value) for value in row.get("relation_types", []) if clean(value)
    })
    surface_operations = sorted({
        clean(value) for value in row.get("surface_operations", []) if clean(value)
    })
    snippet_hits = sum(
        bool(answer) and key(answer) in key(snippet["text"])
        for snippet in example["snippets"]
    )
    reciprocal_rank_sum = sum(1.0 / rank for rank in positive_ranks)
    rank_text = ", ".join(f"{source}={rank}" for source, rank in ordered_ranks) or "none"
    return "\n".join([
        "Candidate provenance and inference-time features:",
        f"- Generator sources: {', '.join(sources) or 'none'}",
        f"- Source ranks: {rank_text}",
        f"- Independent generator count: {len(base_sources)}",
        f"- Best source rank: {min(positive_ranks) if positive_ranks else 'none'}",
        f"- Reciprocal-rank sum: {reciprocal_rank_sum:.6f}",
        f"- Relation types: {', '.join(relation_types) or 'none'}",
        f"- Surface operations: {', '.join(surface_operations) or 'none'}",
        f"- Automatically generated format variant: {'yes' if row.get('is_format_variant') else 'no'}",
        f"- Literal occurrence in supplied evidence: {'yes' if snippet_hits else 'no'}",
        f"- Supporting snippet count: {snippet_hits}",
        f"- Candidate word count: {len(answer.split())}",
        f"- Candidate character count: {len(answer)}",
    ])


def format_reranker_body(
    example: dict[str, Any],
    candidate: str,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    candidate_row: dict[str, Any] | None = None,
    encode_source_metadata: bool = False,
) -> str:
    query = (
        f"Biomedical factoid question: {clean(example['question'])}\n"
        f"Candidate answer: {clean(candidate)}"
    )
    metadata = ""
    if encode_source_metadata:
        if candidate_row is None:
            raise ValueError("candidate_row is required when source metadata is encoded")
        metadata = "\n" + format_candidate_metadata(example, candidate_row)
    return (
        f"<Instruct>: {instruction}\n<Query>: {query}{metadata}\n"
        f"<Document>: {format_document(example)}"
    )


def hard_negative_key(row: dict[str, Any], example: dict[str, Any]) -> tuple[Any, ...]:
    """Prioritize plausible inference-time negatives; metadata is not encoded."""
    answer = clean(row["answer"])
    evidence = " ".join(clean(snippet["text"]) for snippet in example["snippets"])
    literal = int(bool(answer) and key(answer) in key(evidence))
    base_sources = {source.removeprefix("format::") for source in row["sources"]}
    reciprocal_rank = sum(
        1.0 / float(rank) for rank in row["source_ranks"].values() if float(rank) > 0
    )
    return (-literal, -len(base_sources), -reciprocal_rank, len(answer), key(answer))


def build_training_pairs(
    question_ids: Iterable[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
    *,
    max_negatives_per_positive: int,
) -> list[dict[str, Any]]:
    """Build within-question positive/negative pairs from authentic slates."""
    pairs = []
    for qid in question_ids:
        rows = by_question[qid]
        positives = [row for row in rows if int(row["label"]) == 1]
        if not positives:
            continue
        negatives = sorted(
            (row for row in rows if int(row["label"]) == 0),
            key=lambda row: hard_negative_key(row, examples[qid]),
        )[:max_negatives_per_positive]
        for positive in positives:
            for negative in negatives:
                pairs.append({
                    "question_id": qid,
                    "positive": positive,
                    "negative": negative,
                })
    return pairs


def make_question_folds(
    question_ids: list[str],
    by_question: dict[str, list[dict[str, Any]]],
    *,
    folds: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    labels = np.asarray([
        int(any(int(row["label"]) for row in by_question[qid])) for qid in question_ids
    ])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros(len(question_ids), dtype=np.int8)
    return [
        (
            [question_ids[index] for index in train_indices],
            [question_ids[index] for index in test_indices],
        )
        for train_indices, test_indices in splitter.split(dummy, labels)
    ]


def fixed_source_key(row: dict[str, Any]) -> tuple[Any, ...]:
    source_order = {
        "gpt_equivalent": 0,
        "gpt_sampling": 1,
        "llama31_8b": 2,
        "qwen3_8b": 3,
        "gemma3_27b": 4,
        "extractive_v1": 5,
        "extractive_v2": 6,
    }
    choices = []
    for source, rank in row["source_ranks"].items():
        base = source.removeprefix("format::")
        offset = len(source_order) if source.startswith("format::") else 0
        choices.append((offset + source_order.get(base, len(source_order)), int(rank)))
    return min(choices), len(clean(row["answer"])), key(row["answer"])


def consensus_key(row: dict[str, Any], example: dict[str, Any]) -> tuple[Any, ...]:
    answer = clean(row["answer"])
    evidence = " ".join(clean(snippet["text"]) for snippet in example["snippets"])
    literal = int(bool(answer) and key(answer) in key(evidence))
    base_sources = {source.removeprefix("format::") for source in row["sources"]}
    mean_rr = sum(1.0 / float(rank) for rank in row["source_ranks"].values()) / len(row["source_ranks"])
    return (-len(base_sources), -literal, -mean_rr, len(answer), key(answer))


def evaluate_ordering(orderings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = [
        next((index for index, row in enumerate(rows, 1) if int(row["label"])), None)
        for rows in orderings.values()
    ]
    output: dict[str, Any] = {"question_count": len(ranks)}
    for cutoff in (1, 5, 10):
        covered = sum(rank is not None and rank <= cutoff for rank in ranks)
        output[f"covered_at{cutoff}"] = covered
        output[f"coverage_at{cutoff}"] = covered / len(ranks)
    output["mrr_at5"] = sum(
        1.0 / rank for rank in ranks if rank is not None and rank <= 5
    ) / len(ranks)
    output["oracle_covered"] = sum(rank is not None for rank in ranks)
    output["oracle_coverage"] = output["oracle_covered"] / len(ranks)
    return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptEncoder:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        instruction: str,
        *,
        encode_source_metadata: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.encode_source_metadata = encode_source_metadata
        self.prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no_token_id = self._single_token_id("no")
        self.yes_token_id = self._single_token_id("yes")

    def _single_token_id(self, text: str) -> int:
        values = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(values) != 1:
            raise ValueError(f"Expected {text!r} to be one token, got {values}")
        return int(values[0])

    def encode(self, example: dict[str, Any], row: dict[str, Any]) -> list[int]:
        body = format_reranker_body(
            example,
            row["answer"],
            instruction=self.instruction,
            candidate_row=row,
            encode_source_metadata=self.encode_source_metadata,
        )
        body_tokens = self.tokenizer.encode(body, add_special_tokens=False)
        values = self.prefix_tokens + body_tokens + self.suffix_tokens
        if len(values) > self.max_length:
            raise ValueError(
                f"Input has {len(values)} tokens, exceeding max_length={self.max_length}; "
                "evidence truncation is disabled"
            )
        return values

    def batch(self, encoded: list[list[int]], device: Any) -> dict[str, Any]:
        result = self.tokenizer.pad(
            {"input_ids": encoded},
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {name: value.to(device) for name, value in result.items()}


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    source = resolve_model_source(args)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        padding_side="left",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_model_source(args: argparse.Namespace) -> str:
    """Use a concrete snapshot path offline to prevent hidden Hub lookups."""
    if not args.local_files_only:
        return args.model
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    config_path = try_to_load_from_cache(
        repo_id=args.model,
        filename="config.json",
        revision=args.revision,
    )
    if config_path is None or config_path is _CACHED_NO_EXIST:
        raise FileNotFoundError(
            f"Pinned model snapshot is not cached: {args.model}@{args.revision}"
        )
    return str(Path(config_path).parent)


def preflight_lengths(
    encoder: PromptEncoder,
    pool_rows: list[dict[str, Any]],
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lengths = []
    maximum = None
    overflow = []
    for row in pool_rows:
        body = format_reranker_body(
            examples[row["question_id"]],
            row["answer"],
            instruction=encoder.instruction,
            candidate_row=row,
            encode_source_metadata=encoder.encode_source_metadata,
        )
        length = (
            len(encoder.prefix_tokens)
            + len(encoder.tokenizer.encode(body, add_special_tokens=False))
            + len(encoder.suffix_tokens)
        )
        lengths.append(length)
        record = {
            "question_id": row["question_id"],
            "answer": row["answer"],
            "tokens": length,
        }
        if maximum is None or length > maximum["tokens"]:
            maximum = record
        if length > encoder.max_length:
            overflow.append(record)
    ordered = np.asarray(sorted(lengths), dtype=np.int64)
    return {
        "candidate_inputs": len(lengths),
        "max_length": encoder.max_length,
        "minimum_tokens": int(ordered[0]),
        "median_tokens": float(np.median(ordered)),
        "p90_tokens": int(np.percentile(ordered, 90, method="higher")),
        "p95_tokens": int(np.percentile(ordered, 95, method="higher")),
        "maximum": maximum,
        "overflow_count": len(overflow),
        "overflow_examples": overflow[:20],
        "all_inputs_fit_without_truncation": not overflow,
    }


def load_model(args: argparse.Namespace, *, train_lora: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        resolve_model_source(args),
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    if train_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    return model


def model_scores(model: Any, batch: dict[str, Any], encoder: PromptEncoder) -> Any:
    outputs = model(**batch, use_cache=False, logits_to_keep=1)
    final_logits = outputs.logits[:, -1, :]
    return final_logits[:, encoder.yes_token_id] - final_logits[:, encoder.no_token_id]


def score_questions(
    model: Any,
    encoder: PromptEncoder,
    question_ids: list[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    fold: int | None,
    score_name: str,
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    ranked_rows = []
    with torch.no_grad():
        for question_number, qid in enumerate(question_ids, 1):
            rows = by_question[qid]
            scores = []
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start : start + batch_size]
                encoded = [encoder.encode(examples[qid], row) for row in batch_rows]
                batch = encoder.batch(encoded, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    values = model_scores(model, batch, encoder)
                scores.extend(float(value) for value in values.float().cpu().tolist())
            ordered = sorted(
                ({**row, score_name: score, "fold": fold} for row, score in zip(rows, scores)),
                key=lambda row: (-row[score_name], fixed_source_key(row)),
            )
            for rank, row in enumerate(ordered, 1):
                ranked_rows.append({**row, "reranker_rank": rank})
            if question_number % 10 == 0 or question_number == len(question_ids):
                print(f"evaluated={question_number}/{len(question_ids)}", flush=True)
    return ranked_rows


def longest_questions(question_ids: list[str], examples: dict[str, dict[str, Any]], count: int) -> list[str]:
    return sorted(
        question_ids,
        key=lambda qid: sum(len(clean(row["text"])) for row in examples[qid]["snippets"]),
        reverse=True,
    )[:count]


def train_fold(
    *,
    fold: int,
    train_ids: list[str],
    test_ids: list[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
    encoder: PromptEncoder,
    args: argparse.Namespace,
    fold_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils import clip_grad_norm_
    from transformers.optimization import get_linear_schedule_with_warmup

    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    if args.smoke_test:
        answerable_train = [
            qid for qid in train_ids if any(int(row["label"]) for row in by_question[qid])
        ]
        train_ids = longest_questions(answerable_train, examples, 4)
        test_ids = longest_questions(test_ids, examples, 2)
    pairs = build_training_pairs(
        train_ids,
        by_question,
        examples,
        max_negatives_per_positive=args.max_negatives_per_positive,
    )
    if not pairs:
        raise ValueError(f"Fold {fold} has no positive/negative training pairs")

    model = load_model(args, train_lora=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use Gadi or pass --allow-cpu for a deliberate smoke test.")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(pairs) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(pairs)))
        random.Random(fold_seed + epoch).shuffle(order)
        losses = []
        updates = 0
        for step, pair_index in enumerate(order, 1):
            pair = pairs[pair_index]
            qid = pair["question_id"]
            encoded = [
                encoder.encode(examples[qid], pair["positive"]),
                encoder.encode(examples[qid], pair["negative"]),
            ]
            batch = encoder.batch(encoded, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                scores = model_scores(model, batch, encoder)
                loss = functional.softplus(-(scores[0] - scores[1]))
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            if step % args.gradient_accumulation_steps == 0 or step == len(order):
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        history.append({
            "epoch": epoch + 1,
            "mean_pairwise_loss": float(np.mean(losses)),
            "optimizer_updates": updates,
        })
        print(f"fold={fold} epoch={epoch+1}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)

    ranked = score_questions(
        model,
        encoder,
        test_ids,
        by_question,
        examples,
        device=device,
        batch_size=args.eval_batch_size,
        fold=fold,
        score_name="lora_score",
    )
    if args.save_fold_adapters:
        model.save_pretrained(fold_dir / "adapter")
    summary = {
        "fold": fold,
        "seed": fold_seed,
        "train_questions": len(train_ids),
        "test_questions": len(test_ids),
        "training_pairs_per_epoch": len(pairs),
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "history": history,
    }
    write_jsonl(fold_dir / "rankings.jsonl", ranked)
    write_json(fold_dir / "summary.json", summary)
    del optimizer, scheduler, scaler, trainable, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ranked, summary


def ordered_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["reranker_rank"])
    return grouped


def baseline_metrics(
    question_ids: Iterable[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = list(question_ids)
    return {
        "fixed_source_order": evaluate_ordering({
            qid: sorted(by_question[qid], key=fixed_source_key) for qid in ids
        }),
        "consensus_heuristic": evaluate_ordering({
            qid: sorted(by_question[qid], key=lambda row: consensus_key(row, examples[qid]))
            for qid in ids
        }),
    }


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("validate", "preflight", "zero-shot", "train-cv"),
        default="validate",
    )
    parser.add_argument("--labeled-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--run-name")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-negatives-per-positive", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-fold-adapters", action="store_true")
    parser.add_argument("--encode-source-metadata", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    pool_rows = read_jsonl(args.labeled_pool)
    examples = {row["question_id"]: row for row in read_jsonl(args.examples)}
    validation = validate_data(pool_rows, examples)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_question[row["question_id"]].append(row)
    question_ids = sorted(by_question)
    folds = make_question_folds(question_ids, by_question, folds=args.folds, seed=args.seed)
    split_summary = [
        {
            "fold": index,
            "train_questions": len(train_ids),
            "test_questions": len(test_ids),
            "overlap": len(set(train_ids) & set(test_ids)),
            "test_answerable": sum(
                any(int(row["label"]) for row in by_question[qid]) for qid in test_ids
            ),
        }
        for index, (train_ids, test_ids) in enumerate(folds)
    ]
    base_preflight = {
        "validation": validation,
        "splits": split_summary,
        "labeled_pool_sha256": sha256(args.labeled_pool),
        "examples_sha256": sha256(args.examples),
        "model": args.model,
        "revision": args.revision,
        "all_snippets_encoded_in_original_order": True,
        "evidence_truncation": False,
        "gold_blind_input": True,
        "source_metadata_encoded": args.encode_source_metadata,
        "inference_metadata_fields": (
            INFERENCE_METADATA_FIELDS if args.encode_source_metadata else []
        ),
    }
    if args.mode == "validate":
        print(json.dumps(base_preflight, indent=2))
        return

    tokenizer = load_tokenizer(args)
    encoder = PromptEncoder(
        tokenizer,
        args.max_length,
        args.instruction,
        encode_source_metadata=args.encode_source_metadata,
    )
    length_preflight = preflight_lengths(encoder, pool_rows, examples)
    full_preflight = {**base_preflight, "token_lengths": length_preflight}
    if not length_preflight["all_inputs_fit_without_truncation"]:
        raise ValueError(
            f"{length_preflight['overflow_count']} inputs exceed max_length={args.max_length}; "
            "refusing to truncate evidence"
        )
    if args.mode == "preflight":
        print(json.dumps(full_preflight, indent=2))
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{timestamp}-qwen3-reranker-{args.mode}"
    output = args.output_parent / run_name
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", jsonable_config(args))
    write_json(output / "preflight.json", full_preflight)
    write_json(output / "status.json", {"status": "running", "mode": args.mode})

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable. Run this experiment on Gadi.")

        if args.mode == "zero-shot":
            seed_everything(args.seed)
            model = load_model(args, train_lora=False)
            model.to(device)
            selected_ids = (
                longest_questions(question_ids, examples, 2)
                if args.smoke_test else question_ids
            )
            started = time.monotonic()
            ranked = score_questions(
                model,
                encoder,
                selected_ids,
                by_question,
                examples,
                device=device,
                batch_size=args.eval_batch_size,
                fold=None,
                score_name="zero_shot_score",
            )
            grouped = ordered_by_question(ranked)
            summary = {
                "status": "complete",
                "mode": "zero-shot",
                "scope": "exploratory development-set evaluation",
                "model": args.model,
                "revision": args.revision,
                "question_count": len(selected_ids),
                "elapsed_seconds": time.monotonic() - started,
                "metrics": {
                    "qwen3_reranker_zero_shot": evaluate_ordering(grouped),
                    **baseline_metrics(selected_ids, by_question, examples),
                },
            }
            write_jsonl(output / "rankings.jsonl", ranked)
        else:
            selected_folds = args.fold if args.fold is not None else list(range(args.folds))
            if args.smoke_test and args.fold is None:
                selected_folds = [0]
            invalid = [fold for fold in selected_folds if fold < 0 or fold >= args.folds]
            if invalid:
                raise ValueError(f"Invalid folds: {invalid}")
            all_ranked = []
            fold_summaries = []
            for fold in selected_folds:
                fold_dir = output / f"fold-{fold}"
                fold_dir.mkdir()
                ranked, fold_summary = train_fold(
                    fold=fold,
                    train_ids=folds[fold][0],
                    test_ids=folds[fold][1],
                    by_question=by_question,
                    examples=examples,
                    encoder=encoder,
                    args=args,
                    fold_dir=fold_dir,
                )
                all_ranked.extend(ranked)
                fold_summaries.append(fold_summary)
            grouped = ordered_by_question(all_ranked)
            evaluated_ids = sorted(grouped)
            summary = {
                "status": "complete",
                "mode": "LoRA pairwise cross-validation",
                "scope": "exploratory development-set cross-validation",
                "model": args.model,
                "revision": args.revision,
                "objective": "within-question pairwise logistic ranking loss",
                "input": (
                    "question + candidate + inference-time metadata + every supplied snippet "
                    "in original order"
                    if args.encode_source_metadata
                    else "question + candidate + every supplied snippet in original order"
                ),
                "selected_folds": selected_folds,
                "folds": fold_summaries,
                "metrics": {
                    "qwen3_reranker_lora": evaluate_ordering(grouped),
                    **baseline_metrics(evaluated_ids, by_question, examples),
                },
            }
            write_jsonl(output / "cross_validated_rankings.jsonl", all_ranked)

        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete", "mode": args.mode})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Output: {output}", flush=True)
    except Exception as exc:
        write_json(output / "status.json", {
            "status": "incomplete",
            "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
