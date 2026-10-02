#!/usr/bin/env python3
"""Train and cross-validate a gold-blind candidate reranker pilot.

Gold aliases are used only by the official BioASQ matcher to create labels.
Reranker features contain question text, snippets, candidate text and candidate
generation metadata available at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_workflows.coverage_comparison import official_candidate_matches


DEFAULT_EXAMPLES = ROOT / (
    "gadi_sft_8b_starter/outputs/model_comparison/"
    "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/examples.jsonl"
)
DEFAULT_SOURCES = {
    "gpt_equivalent": (
        ROOT / "Artifacts/notebook_runs/coverage_comparison/20260929-070458-a6267ca4/candidates.jsonl",
        "expansion",
    ),
    "gpt_sampling": (
        ROOT / "Artifacts/notebook_runs/coverage_comparison/20260929-070458-a6267ca4/candidates.jsonl",
        "sampling",
    ),
    "llama31_8b": (ROOT / "Artifacts/gadi_runs/llama31_8b_equivalent_dev160/candidates.jsonl", None),
    "qwen3_8b": (ROOT / "Artifacts/gadi_runs/qwen3_8b_equivalent_dev160/candidates.jsonl", None),
    "gemma3_27b": (
        ROOT / (
            "gadi_sft_8b_starter/outputs/model_comparison/"
            "20261002-000810-gemma3-27b-equivalent-dev160-180316747-gadi-pbs/candidates.jsonl"
        ),
        None,
    ),
    "extractive_v1": (
        ROOT / "Artifacts/notebook_runs/coverage_comparison/20261001-010109-2d745f42/candidates.jsonl",
        "expansion",
    ),
    "extractive_v2": (
        ROOT / "Artifacts/notebook_runs/coverage_comparison/20261001-081220-082b3f96/candidates.jsonl",
        "expansion",
    ),
}
SOURCE_ORDER = list(DEFAULT_SOURCES)
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


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def key(text: Any) -> str:
    return clean(text).casefold()


def tokens(text: Any) -> set[str]:
    return {item.casefold() for item in TOKEN_RE.findall(clean(text))}


def conservative_surface_variants(answer: str) -> list[tuple[str, str]]:
    """Produce meaning-preserving typography variants without consulting gold."""
    original = clean(answer)
    variants: list[tuple[str, str]] = []

    def add(value: str, operation: str) -> None:
        value = clean(value)
        if value and key(value) != key(original) and key(value) not in {key(row[0]) for row in variants}:
            variants.append((value, operation))

    if original.endswith("."):
        add(original[:-1], "remove_terminal_period")
    else:
        add(original + ".", "add_terminal_period")
    ascii_hyphens = re.sub(r"[‐‑‒–—−]", "-", original)
    add(ascii_hyphens, "normalize_hyphen")
    add(re.sub(r"(?<=\w)-(?=\w)", " ", ascii_hyphens), "hyphen_to_space")
    if "™" in original:
        add(original.replace("™", "TM"), "trademark_symbol_to_letters")
    if "â„¢" in original:
        add(original.replace("â„¢", "TM"), "repair_trademark_mojibake")
        add(original.replace("â„¢", "™"), "repair_trademark_symbol")
    return variants


def load_source_rows() -> list[dict[str, Any]]:
    rows = []
    for source, (path, arm) in DEFAULT_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing candidate source {source}: {path}")
        for row in read_jsonl(path):
            if arm is not None and row.get("arm") != arm:
                continue
            rows.append({
                "question_id": row["question_id"],
                "answer": clean(row["answer"]),
                "source": source,
                "source_rank": int(row["position"]),
                "relation_type": row.get("relation_type") or row.get("candidate_type") or "unknown",
                "is_format_variant": False,
                "surface_operation": None,
            })
    originals = list(rows)
    for row in originals:
        for variant, operation in conservative_surface_variants(row["answer"]):
            rows.append({
                **row,
                "answer": variant,
                "source": f"format::{row['source']}",
                "is_format_variant": True,
                "surface_operation": operation,
            })
    return rows


def merge_candidate_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        pool_key = (row["question_id"], key(row["answer"]))
        if pool_key not in merged:
            merged[pool_key] = {
                "question_id": row["question_id"],
                "answer": row["answer"],
                "sources": [],
                "source_ranks": {},
                "relation_types": [],
                "surface_operations": [],
                "is_format_variant": bool(row["is_format_variant"]),
            }
        target = merged[pool_key]
        if row["source"] not in target["sources"]:
            target["sources"].append(row["source"])
        target["source_ranks"][row["source"]] = min(
            int(row["source_rank"]),
            int(target["source_ranks"].get(row["source"], row["source_rank"])),
        )
        if row["relation_type"] not in target["relation_types"]:
            target["relation_types"].append(row["relation_type"])
        if row["surface_operation"] and row["surface_operation"] not in target["surface_operations"]:
            target["surface_operations"].append(row["surface_operation"])
        target["is_format_variant"] = target["is_format_variant"] and bool(row["is_format_variant"])
    return list(merged.values())


def candidate_features(row: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    answer = row["answer"]
    question = example["question"]
    snippet_texts = [clean(snippet["text"]) for snippet in example["snippets"]]
    evidence = " ".join(snippet_texts)
    answer_tokens = tokens(answer)
    question_tokens = tokens(question)
    evidence_tokens = tokens(evidence)
    lower_answer = key(answer)
    lower_snippets = [key(value) for value in snippet_texts]
    literal_occurrences = sum(value.count(lower_answer) for value in lower_snippets if lower_answer)
    source_count = len([source for source in row["sources"] if not source.startswith("format::")])
    all_ranks = list(row["source_ranks"].values())
    features: dict[str, Any] = {
        "bias": 1.0,
        "source_count": source_count,
        "all_provenance_count": len(row["sources"]),
        "min_source_rank": min(all_ranks),
        "reciprocal_min_rank": 1.0 / min(all_ranks),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in all_ranks) / len(all_ranks),
        "answer_chars_log": math.log1p(len(answer)),
        "answer_tokens_log": math.log1p(len(answer_tokens)),
        "literal_in_evidence": int(literal_occurrences > 0),
        "literal_occurrences_log": math.log1p(literal_occurrences),
        "candidate_token_evidence_coverage": (
            len(answer_tokens & evidence_tokens) / len(answer_tokens) if answer_tokens else 0.0
        ),
        "question_candidate_jaccard": (
            len(answer_tokens & question_tokens) / len(answer_tokens | question_tokens)
            if answer_tokens | question_tokens else 0.0
        ),
        "has_parentheses": int("(" in answer and ")" in answer),
        "has_number": int(bool(re.search(r"\d", answer))),
        "has_percent": int("%" in answer),
        "has_range_marker": int(bool(re.search(r"\bto\b|[-–—]", answer, flags=re.I))),
        "terminal_period": int(answer.endswith(".")),
        "is_format_variant": int(row["is_format_variant"]),
    }
    for source in row["sources"]:
        features[f"source={source}"] = 1
        features[f"source_rr={source}"] = 1.0 / row["source_ranks"][source]
    for relation in row["relation_types"]:
        features[f"relation={relation}"] = 1
    for operation in row["surface_operations"]:
        features[f"surface_operation={operation}"] = 1
    return features


def make_text(row: dict[str, Any], example: dict[str, Any]) -> str:
    return f"question {example['question']} candidate {row['answer']}"


def fit_components(train_rows, examples, labels):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=30_000, sublinear_tf=True)
    meta = DictVectorizer(sparse=True)
    texts = [make_text(row, examples[row["question_id"]]) for row in train_rows]
    answers = [row["answer"] for row in train_rows]
    dicts = [candidate_features(row, examples[row["question_id"]]) for row in train_rows]
    matrix = hstack((word.fit_transform(texts), char.fit_transform(answers), meta.fit_transform(dicts)), format="csr")
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        solver="liblinear",
        random_state=3407,
    )
    model.fit(matrix, labels)
    return {"word": word, "char": char, "meta": meta, "model": model}


def score_components(components, rows, examples) -> np.ndarray:
    texts = [make_text(row, examples[row["question_id"]]) for row in rows]
    answers = [row["answer"] for row in rows]
    dicts = [candidate_features(row, examples[row["question_id"]]) for row in rows]
    matrix = hstack((
        components["word"].transform(texts),
        components["char"].transform(answers),
        components["meta"].transform(dicts),
    ), format="csr")
    return components["model"].decision_function(matrix)


def evaluate_orderings(orderings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {"question_count": len(orderings)}
    ranks = []
    for rows in orderings.values():
        rank = next((index for index, row in enumerate(rows, 1) if row["label"]), None)
        ranks.append(rank)
    for cutoff in (1, 5, 10):
        covered = sum(rank is not None and rank <= cutoff for rank in ranks)
        result[f"covered_at{cutoff}"] = covered
        result[f"coverage_at{cutoff}"] = covered / len(ranks)
    result["mrr_at5"] = sum(1.0 / rank for rank in ranks if rank is not None and rank <= 5) / len(ranks)
    result["oracle_covered"] = sum(rank is not None for rank in ranks)
    result["oracle_coverage"] = result["oracle_covered"] / len(ranks)
    return result


def fixed_source_key(row: dict[str, Any]):
    choices = []
    for source, rank in row["source_ranks"].items():
        base_source = source.removeprefix("format::")
        source_index = SOURCE_ORDER.index(base_source) if base_source in SOURCE_ORDER else len(SOURCE_ORDER)
        format_offset = len(SOURCE_ORDER) if source.startswith("format::") else 0
        choices.append((format_offset + source_index, rank))
    return min(choices), len(row["answer"]), key(row["answer"])


def consensus_key(row: dict[str, Any], example: dict[str, Any]):
    features = candidate_features(row, example)
    return (
        -features["source_count"],
        -features["literal_in_evidence"],
        -features["mean_reciprocal_rank"],
        features["answer_chars_log"],
        key(row["answer"]),
    )


def round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_ORDER:
        queues[source] = sorted(
            [row for row in rows if source in row["source_ranks"]],
            key=lambda row: (row["source_ranks"][source], len(row["answer"]), key(row["answer"])),
        )
    output, seen = [], set()
    for rank_index in range(max((len(queue) for queue in queues.values()), default=0)):
        for source in SOURCE_ORDER:
            queue = queues[source]
            if rank_index >= len(queue):
                continue
            row = queue[rank_index]
            identity = (row["question_id"], key(row["answer"]))
            if identity not in seen:
                seen.add(identity)
                output.append(row)
    remaining = sorted((row for row in rows if (row["question_id"], key(row["answer"])) not in seen), key=fixed_source_key)
    return output + remaining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-root", type=Path, default=ROOT / "Artifacts/reranker_pilot")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    examples_list = read_jsonl(args.examples)
    examples = {row["question_id"]: row for row in examples_list}
    rows = merge_candidate_pool(load_source_rows())
    if set(examples) != {row["question_id"] for row in rows}:
        raise ValueError("Candidate pool and example question IDs do not match")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output_root / f"{timestamp}-tfidf-logistic"
    output.mkdir(parents=True, exist_ok=False)
    scorer_dir = output / "official_labels"
    scorer_dir.mkdir()
    matches = official_candidate_matches(examples_list, rows, scorer_dir, jar_path=ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    for row in rows:
        row["label"] = int(matches[(row["question_id"], row["answer"])])

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["question_id"] for row in rows])
    strata = labels.copy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=3407)
    cv_scores = np.full(len(rows), np.nan, dtype=np.float64)
    fold_ids = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries = []
    dummy_x = np.zeros(len(rows), dtype=np.int8)
    for fold, (train_index, test_index) in enumerate(splitter.split(dummy_x, strata, groups)):
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        components = fit_components(train_rows, examples, labels[train_index])
        cv_scores[test_index] = score_components(components, test_rows, examples)
        fold_ids[test_index] = fold
        fold_summaries.append({
            "fold": fold,
            "train_questions": len(set(groups[train_index])),
            "test_questions": len(set(groups[test_index])),
            "train_candidates": len(train_index),
            "test_candidates": len(test_index),
            "train_positives": int(labels[train_index].sum()),
            "test_positives": int(labels[test_index].sum()),
        })
    if np.isnan(cv_scores).any() or (fold_ids < 0).any():
        raise RuntimeError("Cross-validation did not score every candidate")

    by_question = defaultdict(list)
    for index, row in enumerate(rows):
        enriched = {**row, "cv_score": float(cv_scores[index]), "fold": int(fold_ids[index])}
        by_question[row["question_id"]].append(enriched)
    orderings = {
        "cross_validated_reranker": {
            qid: sorted(group, key=lambda row: (-row["cv_score"], fixed_source_key(row)))
            for qid, group in by_question.items()
        },
        "fixed_source_order": {
            qid: sorted(group, key=fixed_source_key) for qid, group in by_question.items()
        },
        "source_balanced_round_robin": {
            qid: round_robin(group) for qid, group in by_question.items()
        },
        "consensus_heuristic": {
            qid: sorted(group, key=lambda row: consensus_key(row, examples[qid]))
            for qid, group in by_question.items()
        },
    }
    metrics = {name: evaluate_orderings(value) for name, value in orderings.items()}

    final_components = fit_components(rows, examples, labels)
    joblib.dump(final_components, output / "final_model.joblib")
    ranked_rows = []
    for qid in sorted(orderings["cross_validated_reranker"]):
        for rank, row in enumerate(orderings["cross_validated_reranker"][qid], 1):
            ranked_rows.append({**row, "cv_rank": rank})
    write_jsonl(output / "cross_validated_rankings.jsonl", ranked_rows)
    write_jsonl(output / "candidate_pool_labeled.jsonl", rows)
    summary = {
        "status": "complete",
        "method": "TF-IDF plus inference-available metadata logistic reranker",
        "evaluation": f"{args.folds}-fold question-grouped cross-validation",
        "warning": "The final_model is trained on all 160 development questions and has no independent evaluation score.",
        "question_count": len(examples),
        "candidate_count": len(rows),
        "positive_candidate_count": int(labels.sum()),
        "questions_with_positive_candidate": len({row["question_id"] for row in rows if row["label"]}),
        "mean_candidates_per_question": len(rows) / len(examples),
        "sources": {name: str(path) for name, (path, _) in DEFAULT_SOURCES.items()},
        "format_variants": True,
        "folds": fold_summaries,
        "metrics": metrics,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()
