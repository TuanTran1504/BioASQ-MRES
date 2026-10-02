#!/usr/bin/env python3
"""Train and cross-validate a biomedical neural candidate reranker.

Gold-derived exact-match labels define the training objective only. Model
inputs contain a question, one candidate answer, and candidate/question-
relevant snippets. Gold aliases, source identities, and source ranks are never
included in the encoded input.
"""

from __future__ import annotations

import argparse
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
DEFAULT_EXAMPLES = ROOT / (
    "gadi_sft_8b_starter/outputs/model_comparison/"
    "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/"
    "examples.jsonl"
)
DEFAULT_OUTPUT_PARENT = STARTER_ROOT / "outputs/neural_reranker"
DEFAULT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_REVISION = "b857e516dbf8a3a8bd9d03888e54d0618cd36eab"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.UNICODE)


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


def tokens(value: Any) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(clean(value))}


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
    example_ids = set(examples)
    if pool_ids != example_ids:
        raise ValueError(
            f"Question mismatch: pool={len(pool_ids)}, examples={len(example_ids)}, "
            f"pool_only={len(pool_ids-example_ids)}, examples_only={len(example_ids-pool_ids)}"
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


def select_evidence(
    example: dict[str, Any],
    candidate: str,
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    """Select evidence using inference-time lexical relevance only."""
    candidate_key = key(candidate)
    candidate_tokens = tokens(candidate)
    question_tokens = tokens(example["question"])
    scored = []
    for index, snippet in enumerate(example["snippets"]):
        text = clean(snippet["text"])
        snippet_tokens = tokens(text)
        candidate_coverage = (
            len(candidate_tokens & snippet_tokens) / len(candidate_tokens)
            if candidate_tokens else 0.0
        )
        question_coverage = (
            len(question_tokens & snippet_tokens) / len(question_tokens)
            if question_tokens else 0.0
        )
        exact = float(bool(candidate_key) and candidate_key in key(text))
        score = 8.0 * exact + 3.0 * candidate_coverage + question_coverage
        scored.append((score, -index, snippet))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max_snippets]]


def make_model_input(
    example: dict[str, Any],
    candidate: str,
    *,
    max_snippets: int,
) -> tuple[str, str]:
    first = f"Question: {clean(example['question'])}\nCandidate answer: {clean(candidate)}"
    selected = select_evidence(example, candidate, max_snippets=max_snippets)
    evidence_parts = []
    for index, snippet in enumerate(selected, 1):
        identifier = snippet.get("id") or snippet.get("snippet_id") or index
        evidence_parts.append(f"Snippet {identifier}: {clean(snippet['text'])}")
    second = "Evidence:\n" + "\n".join(evidence_parts)
    return first, second


def hard_negative_key(row: dict[str, Any], example: dict[str, Any]) -> tuple[Any, ...]:
    answer = clean(row["answer"])
    literal = int(bool(answer) and key(answer) in key(" ".join(s["text"] for s in example["snippets"])))
    base_sources = {source.removeprefix("format::") for source in row["sources"]}
    reciprocal_rank = sum(
        1.0 / float(rank) for rank in row["source_ranks"].values() if float(rank) > 0
    )
    return (-literal, -len(base_sources), -reciprocal_rank, len(answer), key(answer))


def build_training_slates(
    question_ids: Iterable[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
    *,
    max_negatives: int,
) -> list[dict[str, Any]]:
    """Return authentic positive slates; questions with no target are excluded."""
    slates = []
    for qid in question_ids:
        rows = by_question[qid]
        positives = [row for row in rows if int(row["label"]) == 1]
        if not positives:
            continue
        negatives = sorted(
            (row for row in rows if int(row["label"]) == 0),
            key=lambda row: hard_negative_key(row, examples[qid]),
        )[:max_negatives]
        selected = positives + negatives
        slates.append({
            "question_id": qid,
            "rows": selected,
            "positive_count": len(positives),
            "negative_count": len(negatives),
        })
    return slates


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
    output = []
    for train_indices, test_indices in splitter.split(dummy, labels):
        output.append((
            [question_ids[index] for index in train_indices],
            [question_ids[index] for index in test_indices],
        ))
    return output


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
    literal = int(bool(answer) and key(answer) in key(" ".join(s["text"] for s in example["snippets"])))
    base_sources = {source.removeprefix("format::") for source in row["sources"]}
    mean_rr = sum(1.0 / float(rank) for rank in row["source_ranks"].values()) / len(row["source_ranks"])
    return (-len(base_sources), -literal, -mean_rr, len(answer), key(answer))


def evaluate_ordering(orderings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ranks = []
    for rows in orderings.values():
        ranks.append(next((index for index, row in enumerate(rows, 1) if int(row["label"])), None))
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


def encode_rows(tokenizer: Any, pairs: list[tuple[str, str]], max_length: int, device: Any) -> dict[str, Any]:
    encoded = tokenizer(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    )
    return {name: value.to(device) for name, value in encoded.items()}


def train_fold(
    *,
    fold: int,
    train_ids: list[str],
    test_ids: list[str],
    by_question: dict[str, list[dict[str, Any]]],
    examples: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    fold_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    fold_seed = args.seed + fold
    seed_everything(fold_seed)
    train_slates = build_training_slates(
        train_ids,
        by_question,
        examples,
        max_negatives=args.max_negatives,
    )
    if args.smoke_test:
        train_slates = train_slates[: min(8, len(train_slates))]
        test_ids = test_ids[: min(4, len(test_ids))]
    if not train_slates:
        raise ValueError(f"Fold {fold} has no answerable training slates")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Pass --allow-cpu only for a deliberate tiny smoke test.")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_slates) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = round(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_slates)))
        random.Random(fold_seed + epoch).shuffle(order)
        losses = []
        update_count = 0
        for step, slate_index in enumerate(order, 1):
            slate = train_slates[slate_index]
            pairs = [
                make_model_input(
                    examples[slate["question_id"]],
                    row["answer"],
                    max_snippets=args.max_snippets,
                )
                for row in slate["rows"]
            ]
            inputs = encode_rows(tokenizer, pairs, args.max_length, device)
            positive_mask = torch.tensor(
                [bool(int(row["label"])) for row in slate["rows"]],
                dtype=torch.bool,
                device=device,
            )
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**inputs).logits.reshape(-1)
                loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[positive_mask], dim=0)
                scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            if step % args.gradient_accumulation_steps == 0 or step == len(order):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
        history.append({
            "epoch": epoch + 1,
            "mean_listwise_loss": float(np.mean(losses)),
            "optimizer_updates": update_count,
        })
        print(f"fold={fold} epoch={epoch+1}/{args.epochs} loss={np.mean(losses):.6f}", flush=True)

    model.eval()
    ranked_rows = []
    with torch.no_grad():
        for question_number, qid in enumerate(test_ids, 1):
            rows = by_question[qid]
            scores = []
            for start in range(0, len(rows), args.eval_batch_size):
                batch = rows[start : start + args.eval_batch_size]
                pairs = [
                    make_model_input(examples[qid], row["answer"], max_snippets=args.max_snippets)
                    for row in batch
                ]
                inputs = encode_rows(tokenizer, pairs, args.max_length, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(**inputs).logits.reshape(-1)
                scores.extend(float(value) for value in logits.float().cpu().tolist())
            ordered = sorted(
                ({**row, "neural_score": score, "fold": fold} for row, score in zip(rows, scores)),
                key=lambda row: (-row["neural_score"], fixed_source_key(row)),
            )
            for rank, row in enumerate(ordered, 1):
                ranked_rows.append({**row, "neural_rank": rank})
            if question_number % 10 == 0 or question_number == len(test_ids):
                print(f"fold={fold} evaluated={question_number}/{len(test_ids)}", flush=True)

    if args.save_fold_models:
        model.save_pretrained(fold_dir / "model")
        tokenizer.save_pretrained(fold_dir / "model")
    elapsed = time.monotonic() - started
    fold_summary = {
        "fold": fold,
        "seed": fold_seed,
        "train_question_count": len(train_ids),
        "train_answerable_slate_count": len(train_slates),
        "test_question_count": len(test_ids),
        "train_sequence_count_per_epoch": sum(len(slate["rows"]) for slate in train_slates),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "history": history,
    }
    write_jsonl(fold_dir / "rankings.jsonl", ranked_rows)
    write_json(fold_dir / "summary.json", fold_summary)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ranked_rows, fold_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--run-name")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-snippets", type=int, default=4)
    parser.add_argument("--max-negatives", type=int, default=15)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    pool_rows = read_jsonl(args.labeled_pool)
    examples = {row["question_id"]: row for row in read_jsonl(args.examples)}
    validation = validate_data(pool_rows, examples)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        by_question[row["question_id"]].append(row)
    question_ids = sorted(by_question)
    folds = make_question_folds(question_ids, by_question, folds=args.folds, seed=args.seed)
    split_validation = {
        "folds": [
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
    }
    preflight = {
        "validation": validation,
        "splits": split_validation,
        "labeled_pool_sha256": sha256(args.labeled_pool),
        "examples_sha256": sha256(args.examples),
        "model": args.model,
        "revision": args.revision,
        "gold_blind_input": True,
        "source_metadata_encoded": False,
    }
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return

    selected_folds = args.fold if args.fold is not None else list(range(args.folds))
    if args.smoke_test and args.fold is None:
        selected_folds = [0]
    invalid = [fold for fold in selected_folds if fold < 0 or fold >= args.folds]
    if invalid:
        raise ValueError(f"Invalid folds: {invalid}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or (
        f"{timestamp}-biomedbert-cross-encoder-" + ("smoke" if args.smoke_test else "cv")
    )
    output = args.output_parent / run_name
    output.mkdir(parents=True, exist_ok=False)
    resolved_config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    write_json(output / "config.json", resolved_config)
    write_json(output / "preflight.json", preflight)
    write_json(output / "status.json", {"status": "running", "selected_folds": selected_folds})

    all_ranked = []
    fold_summaries = []
    try:
        for fold in selected_folds:
            fold_dir = output / f"fold-{fold}"
            fold_dir.mkdir()
            ranked, summary = train_fold(
                fold=fold,
                train_ids=folds[fold][0],
                test_ids=folds[fold][1],
                by_question=by_question,
                examples=examples,
                args=args,
                fold_dir=fold_dir,
            )
            all_ranked.extend(ranked)
            fold_summaries.append(summary)

        ranked_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_ranked:
            ranked_by_question[row["question_id"]].append(row)
        for rows in ranked_by_question.values():
            rows.sort(key=lambda row: row["neural_rank"])
        neural_metrics = evaluate_ordering(ranked_by_question)
        fixed_metrics = evaluate_ordering({
            qid: sorted(by_question[qid], key=fixed_source_key) for qid in ranked_by_question
        })
        consensus_metrics = evaluate_ordering({
            qid: sorted(by_question[qid], key=lambda row: consensus_key(row, examples[qid]))
            for qid in ranked_by_question
        })
        write_jsonl(output / "cross_validated_rankings.jsonl", all_ranked)
        summary = {
            "status": "complete",
            "scope": "exploratory development-set cross-validation",
            "model": args.model,
            "revision": args.revision,
            "objective": "question-listwise softmax over authentic generated candidates",
            "input": "question + candidate + selected snippets; no source/rank metadata",
            "selected_folds": selected_folds,
            "folds": fold_summaries,
            "metrics": {
                "neural_cross_encoder": neural_metrics,
                "fixed_source_order": fixed_metrics,
                "consensus_heuristic": consensus_metrics,
            },
        }
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", {"status": "complete", "selected_folds": selected_folds})
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Output: {output}", flush=True)
    except Exception as exc:
        write_json(output / "status.json", {
            "status": "incomplete",
            "selected_folds": selected_folds,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    main()
