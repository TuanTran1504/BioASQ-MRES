from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .common import write_json, write_jsonl
from .hybrid_semantic_labeler import HybridSemanticEquivalenceLabeler
from .match_gold_groups import load_candidate_bank_records, load_question_examples
from .semantic_member_clusters import cluster_rows_by_question, summarize_cluster_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build proposal-style semantic answer-member clusters for BioASQ list candidate banks."
    )
    parser.add_argument("--question-input", nargs="+", required=True)
    parser.add_argument("--candidate-input", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--dataset-name", default="bioasq")
    parser.add_argument("--max-resources", type=int, default=3)
    parser.add_argument("--max-resource-chars", type=int, default=1200)
    parser.add_argument("--gold-support-policy", choices=["all", "snippet"], default="all")
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of questions for smoke tests.")
    parser.add_argument(
        "--semantic-embedding-model",
        default=None,
        help="Optional SapBERT/SentenceTransformer model used to identify equivalent generated surfaces and uncertain gold-overlap cases.",
    )
    parser.add_argument("--semantic-embedding-match-threshold", type=float, default=0.90)
    parser.add_argument("--semantic-embedding-uncertain-threshold", type=float, default=0.82)
    parser.add_argument("--semantic-embedding-no-match-threshold", type=float, default=0.50)
    parser.add_argument(
        "--no-lexical-overlap-uncertain",
        action="store_true",
        help="Do not mark phrase/alias overlap cases as uncertain during cluster construction.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    questions_by_id = load_question_examples(
        paths=args.question_input,
        dataset_name=args.dataset_name,
        max_resources=int(args.max_resources),
        max_resource_chars=int(args.max_resource_chars),
        gold_support_policy=str(args.gold_support_policy),
    )
    if args.limit is not None:
        questions_by_id = dict(list(sorted(questions_by_id.items()))[: int(args.limit)])
    records = load_candidate_bank_records(
        paths=args.candidate_input,
        questions_by_id=questions_by_id,
        dataset_name=args.dataset_name,
    )
    records_by_question = defaultdict(list)
    for record in records:
        records_by_question[record.question_id].append(record)

    labeler = HybridSemanticEquivalenceLabeler(
        embedding_model_name=args.semantic_embedding_model,
        embedding_match_threshold=float(args.semantic_embedding_match_threshold),
        embedding_uncertain_threshold=float(args.semantic_embedding_uncertain_threshold),
        embedding_no_match_threshold=float(args.semantic_embedding_no_match_threshold),
    )
    rows = cluster_rows_by_question(
        questions_by_id=questions_by_id,
        records_by_question=records_by_question,
        labeler=labeler,
        allow_fallback_split=bool(args.allow_fallback_split),
        mark_lexical_overlap_uncertain=not bool(args.no_lexical_overlap_uncertain),
    )
    write_jsonl(Path(args.output_jsonl), rows)
    summary = summarize_cluster_rows(rows)
    summary["cluster_policy"] = {
        "max_resources": int(args.max_resources),
        "max_resource_chars": int(args.max_resource_chars),
        "gold_support_policy": str(args.gold_support_policy),
        "allow_fallback_split": bool(args.allow_fallback_split),
        "semantic_embedding_model": args.semantic_embedding_model,
        "semantic_embedding_match_threshold": float(args.semantic_embedding_match_threshold),
        "semantic_embedding_uncertain_threshold": float(args.semantic_embedding_uncertain_threshold),
        "semantic_embedding_no_match_threshold": float(args.semantic_embedding_no_match_threshold),
        "lexical_overlap_marks_uncertain": not bool(args.no_lexical_overlap_uncertain),
        "gold_alignment_policy": "exact_bioasq_alias_only",
        "sapbert_role": "generated-surface clustering and uncertain gold-overlap filtering; not gold expansion",
    }
    write_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
