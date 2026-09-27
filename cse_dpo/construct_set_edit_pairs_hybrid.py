from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import construct_set_edit_pairs as base
from .hybrid_semantic_labeler import HybridSemanticEquivalenceLabeler
from .match_gold_groups import best_surface_semantic_match
from .schemas import QuestionExample

_LABELER: HybridSemanticEquivalenceLabeler | None = None


def _hybrid_equivalence_reason(
    question: QuestionExample,
    left: str,
    right: str,
) -> str | None:
    if _LABELER is None:
        return None
    decision = _LABELER.answer_equivalent(left, right, question=question)
    if decision.matched or decision.uncertain:
        return decision.match_type
    return None


def _surface_represented_in_items_hybrid(
    question: QuestionExample,
    surface: str,
    items: Sequence[str],
) -> str | None:
    best_match = best_surface_semantic_match(question, surface, items)
    if best_match is not None:
        return best_match.match_type
    for item in items:
        hybrid_reason = _hybrid_equivalence_reason(question, surface, item)
        if hybrid_reason is not None:
            return hybrid_reason
    for item in items:
        overlap_reason = base.surface_overlap_reason(question, surface, item)
        if overlap_reason is not None:
            return overlap_reason
    return None


def _surface_overlaps_gold_alias_hybrid(
    question: QuestionExample,
    surface: str,
) -> str | None:
    for gold_group in question.gold_groups:
        best_match = best_surface_semantic_match(question, surface, gold_group.aliases)
        if best_match is not None:
            return best_match.match_type
    for gold_group in question.gold_groups:
        for alias in gold_group.aliases:
            hybrid_reason = _hybrid_equivalence_reason(question, surface, alias)
            if hybrid_reason is not None:
                return hybrid_reason
    for gold_group in question.gold_groups:
        for alias in gold_group.aliases:
            overlap_reason = base.surface_overlap_reason(question, surface, alias)
            if overlap_reason is not None:
                return overlap_reason
    return None


def _parse_hybrid_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--semantic-embedding-model",
        default=None,
        help="Optional sentence-transformer model or local path used as the second-stage biomedical semantic matcher.",
    )
    parser.add_argument(
        "--semantic-embedding-match-threshold",
        type=float,
        default=0.90,
        help="Cosine similarity threshold above which the embedding matcher treats two answers as equivalent.",
    )
    parser.add_argument(
        "--semantic-embedding-uncertain-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold above which the embedding matcher marks a pair as ambiguous and drops it conservatively.",
    )
    parser.add_argument(
        "--semantic-embedding-no-match-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold at or below which the embedding matcher can safely stop before NLI as a clear non-match.",
    )
    parser.add_argument(
        "--semantic-nli-model",
        default=None,
        help="Optional local or Hugging Face NLI model used to judge ambiguous biomedical answer pairs.",
    )
    parser.add_argument(
        "--semantic-nli-match-threshold",
        type=float,
        default=0.80,
        help="Bidirectional entailment threshold above which the NLI judge treats two answers as equivalent.",
    )
    parser.add_argument(
        "--semantic-nli-uncertain-threshold",
        type=float,
        default=0.60,
        help="Bidirectional entailment threshold above which the NLI judge marks a pair as ambiguous and drops it conservatively.",
    )
    return parser.parse_known_args(list(argv))


def _parse_summary_json_arg(argv: Sequence[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--summary-json", default=None)
    args, _ = parser.parse_known_args(list(argv))
    return args.summary_json


def _patch_base_module() -> None:
    base.surface_represented_in_items = _surface_represented_in_items_hybrid
    base.surface_overlaps_gold_alias = _surface_overlaps_gold_alias_hybrid


def _annotate_summary(summary_json_path: str | None, hybrid_args: argparse.Namespace) -> None:
    if not summary_json_path:
        return
    path = Path(summary_json_path)
    if not path.exists():
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    policy = dict(summary.get("pair_construction_policy") or {})
    policy.update(
        {
            "semantic_labeler_enabled": _LABELER is not None,
            "semantic_embedding_model": hybrid_args.semantic_embedding_model,
            "semantic_embedding_match_threshold": float(hybrid_args.semantic_embedding_match_threshold),
            "semantic_embedding_uncertain_threshold": float(hybrid_args.semantic_embedding_uncertain_threshold),
            "semantic_embedding_no_match_threshold": float(hybrid_args.semantic_embedding_no_match_threshold),
            "semantic_nli_model": hybrid_args.semantic_nli_model,
            "semantic_nli_match_threshold": float(hybrid_args.semantic_nli_match_threshold),
            "semantic_nli_uncertain_threshold": float(hybrid_args.semantic_nli_uncertain_threshold),
            "uncertain_semantic_pairs_are_dropped": True,
            "labeler_scope": (
                "Hybrid labeler is applied as a conservative post-filter for omission and negative-addition pair construction. "
                "Primary candidate labels and delta-F1 scoring still come from the base matcher."
            ),
        }
    )
    summary["pair_construction_policy"] = policy
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    global _LABELER

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    hybrid_args, base_argv = _parse_hybrid_args(raw_argv)

    _LABELER = HybridSemanticEquivalenceLabeler(
        embedding_model_name=hybrid_args.semantic_embedding_model,
        embedding_match_threshold=float(hybrid_args.semantic_embedding_match_threshold),
        embedding_uncertain_threshold=float(hybrid_args.semantic_embedding_uncertain_threshold),
        embedding_no_match_threshold=float(hybrid_args.semantic_embedding_no_match_threshold),
        nli_model_name=hybrid_args.semantic_nli_model,
        nli_match_threshold=float(hybrid_args.semantic_nli_match_threshold),
        nli_uncertain_threshold=float(hybrid_args.semantic_nli_uncertain_threshold),
    )
    _patch_base_module()

    summary_json_path = _parse_summary_json_arg(base_argv)
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + base_argv
        base.main()
    finally:
        sys.argv = original_argv

    _annotate_summary(summary_json_path, hybrid_args)


if __name__ == "__main__":
    main()
