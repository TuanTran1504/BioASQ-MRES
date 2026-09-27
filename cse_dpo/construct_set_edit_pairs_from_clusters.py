from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.model_registry import slugify
from src.utility.data import clean_text

from .audit_preference_pairs import build_manual_audit_markdown, build_summary
from .common import load_json_records, safe_ratio, summarize_numeric, write_json, write_jsonl
from .construct_set_edit_pairs import candidate_is_evidence_supported, group_responses_by_question
from .normalize_set_answers import normalize_answer_surface, serialize_list_items
from .schemas import (
    LABEL_GOLD_MISSING,
    LABEL_METRIC_NEGATIVE,
    PAIR_TYPE_NEGATIVE_ADDITION,
    PAIR_TYPE_VALID_OMISSION,
    MatchedResponse,
    PreferencePair,
    QuestionExample,
    ResponseMetrics,
    to_jsonable,
)


@dataclass(frozen=True)
class ClusterInfo:
    cluster_id: str
    representative: str
    surfaces: tuple[str, ...]
    normalized_surfaces: tuple[str, ...]
    aligned_gold_group_id: int | None
    uncertain: bool
    uncertain_gold_group_ids: tuple[int, ...]
    generated_count: int
    source_types: tuple[str, ...]


@dataclass(frozen=True)
class ClusteredItem:
    surface: str
    normalized: str
    cluster_id: str
    first_index: int


@dataclass(frozen=True)
class ClusteredResponse:
    matched_response: MatchedResponse
    items: tuple[ClusteredItem, ...]
    cluster_ids: tuple[str, ...]
    metrics: ResponseMetrics
    matched_gold_group_ids: tuple[int, ...]
    missing_gold_group_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct proposal-style DPO set-edit pairs from explicit semantic-member clusters."
    )
    parser.add_argument("--question-input", nargs="+", required=True)
    parser.add_argument("--candidate-input", nargs="+", required=True)
    parser.add_argument("--cluster-input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--manual-audit-md", required=True)
    parser.add_argument("--dataset-name", default="bioasq")
    parser.add_argument("--max-resources", type=int, default=3)
    parser.add_argument("--max-resource-chars", type=int, default=1200)
    parser.add_argument("--gold-support-policy", choices=["all", "snippet"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of questions for smoke tests.")
    parser.add_argument("--max-negative-addition-pairs-per-question", type=int, default=8)
    parser.add_argument("--max-valid-omission-pairs-per-question", type=int, default=8)
    parser.add_argument("--allow-fallback-split", action="store_true")
    parser.add_argument("--allow-fallback-pair-construction", action="store_true")
    parser.add_argument("--allow-gold-alias-fallback", action="store_true")
    parser.add_argument("--allow-evidence-supported-negative-pairs", action="store_true")
    parser.add_argument("--allow-evidence-unsupported-omission-pairs", action="store_true")
    parser.add_argument("--allow-uncertain-cluster-edits", action="store_true")
    parser.add_argument("--min-negative-addition-delta-f1", type=float, default=0.0)
    parser.add_argument("--manual-audit-sample-per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_clusters(path: Path) -> tuple[dict[str, dict[str, ClusterInfo]], dict[str, dict[str, str]]]:
    clusters_by_question: dict[str, dict[str, ClusterInfo]] = {}
    surface_to_cluster_by_question: dict[str, dict[str, str]] = {}
    for row in load_json_records(path):
        question_id = clean_text(row.get("question_id"))
        if not question_id:
            continue
        clusters: dict[str, ClusterInfo] = {}
        surface_map: dict[str, str] = {}
        for cluster_row in row.get("clusters", []):
            cluster_id = clean_text(cluster_row.get("cluster_id"))
            if not cluster_id:
                continue
            aligned_value = cluster_row.get("aligned_gold_group_id")
            aligned_gold_group_id = int(aligned_value) if aligned_value is not None else None
            cluster = ClusterInfo(
                cluster_id=cluster_id,
                representative=clean_text(cluster_row.get("representative")),
                surfaces=tuple(clean_text(surface) for surface in cluster_row.get("surfaces", []) if clean_text(surface)),
                normalized_surfaces=tuple(
                    clean_text(surface) for surface in cluster_row.get("normalized_surfaces", []) if clean_text(surface)
                ),
                aligned_gold_group_id=aligned_gold_group_id,
                uncertain=bool(cluster_row.get("uncertain")),
                uncertain_gold_group_ids=tuple(int(item) for item in cluster_row.get("uncertain_gold_group_ids", [])),
                generated_count=int(cluster_row.get("generated_count") or 0),
                source_types=tuple(clean_text(item) for item in cluster_row.get("source_types", []) if clean_text(item)),
            )
            clusters[cluster_id] = cluster
            for surface in cluster.surfaces:
                key = normalize_answer_surface(surface) or surface.lower()
                if key:
                    surface_map[key] = cluster_id
            for normalized in cluster.normalized_surfaces:
                if normalized:
                    surface_map[normalized] = cluster_id
        clusters_by_question[question_id] = clusters
        surface_to_cluster_by_question[question_id] = surface_map
    return clusters_by_question, surface_to_cluster_by_question


def cluster_metrics(question: QuestionExample, cluster_ids: Sequence[str], clusters: Mapping[str, ClusterInfo]) -> tuple[ResponseMetrics, tuple[int, ...]]:
    predicted_cluster_ids = tuple(dict.fromkeys(cluster_id for cluster_id in cluster_ids if cluster_id in clusters))
    matched_gold_group_ids: list[int] = []
    for cluster_id in predicted_cluster_ids:
        cluster = clusters[cluster_id]
        if cluster.aligned_gold_group_id is not None:
            matched_gold_group_ids.append(cluster.aligned_gold_group_id)
    matched_gold_group_ids = sorted(set(matched_gold_group_ids))
    prediction_count = len(predicted_cluster_ids)
    gold_count = len(question.gold_groups)
    true_positives = len(matched_gold_group_ids)
    precision = safe_ratio(true_positives, prediction_count)
    recall = safe_ratio(true_positives, gold_count)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return (
        ResponseMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            prediction_count=prediction_count,
            gold_count=gold_count,
            invalid_addition_rate=1.0 - precision if prediction_count else 0.0,
            valid_omission_rate=1.0 - recall if gold_count else 0.0,
        ),
        tuple(matched_gold_group_ids),
    )


def clustered_response_from_matched(
    response: MatchedResponse,
    question: QuestionExample,
    clusters: Mapping[str, ClusterInfo],
    surface_to_cluster: Mapping[str, str],
) -> ClusteredResponse:
    items: list[ClusteredItem] = []
    seen_cluster_ids: set[str] = set()
    cluster_ids: list[str] = []
    for candidate in response.candidates:
        key = normalize_answer_surface(candidate.surface) or candidate.surface.lower()
        cluster_id = surface_to_cluster.get(key)
        if cluster_id is None:
            continue
        items.append(
            ClusteredItem(
                surface=candidate.surface,
                normalized=candidate.normalized,
                cluster_id=cluster_id,
                first_index=candidate.first_index,
            )
        )
        if cluster_id not in seen_cluster_ids:
            seen_cluster_ids.add(cluster_id)
            cluster_ids.append(cluster_id)
    metrics, matched_gold_group_ids = cluster_metrics(question, cluster_ids, clusters)
    missing_gold_group_ids = tuple(
        group.group_id
        for group in question.gold_groups
        if group.group_id not in set(matched_gold_group_ids)
    )
    return ClusteredResponse(
        matched_response=response,
        items=tuple(items),
        cluster_ids=tuple(cluster_ids),
        metrics=metrics,
        matched_gold_group_ids=matched_gold_group_ids,
        missing_gold_group_ids=missing_gold_group_ids,
    )


def build_question_statistics(
    clustered_responses: Sequence[ClusteredResponse],
    clusters: Mapping[str, ClusterInfo],
    question: QuestionExample,
) -> dict[str, Any]:
    negative_frequency = Counter()
    positive_surface_by_gold_group = defaultdict(Counter)
    evidence_text = " ".join(question.evidence)
    evidence_text_normalized = normalize_answer_surface(evidence_text)
    for response in clustered_responses:
        if not response.matched_response.pair_eligible:
            continue
        for cluster_id in response.cluster_ids:
            cluster = clusters[cluster_id]
            if cluster.uncertain:
                continue
            if cluster.aligned_gold_group_id is None:
                negative_frequency[cluster_id] += 1
            else:
                positive_surface_by_gold_group[cluster.aligned_gold_group_id][cluster.representative] += max(1, cluster.generated_count)
    return {
        "negative_frequency": negative_frequency,
        "positive_surface_by_gold_group": positive_surface_by_gold_group,
        "evidence_text": evidence_text_normalized,
        "evidence_text_compact": evidence_text_normalized.replace(" ", ""),
    }


def response_items_without_cluster(response: ClusteredResponse, cluster_id: str) -> list[str]:
    return [item.surface for item in response.items if item.cluster_id != cluster_id]


def response_items(response: ClusteredResponse) -> list[str]:
    return [item.surface for item in response.items]


def cluster_ids_without(cluster_ids: Sequence[str], cluster_id: str) -> list[str]:
    return [item for item in cluster_ids if item != cluster_id]


def build_pair(
    *,
    pair_id: str,
    pair_type: str,
    question: QuestionExample,
    response: ClusteredResponse,
    clusters: Mapping[str, ClusterInfo],
    chosen_items: Sequence[str],
    rejected_items: Sequence[str],
    chosen_cluster_ids: Sequence[str],
    rejected_cluster_ids: Sequence[str],
    edited_candidate: str,
    edited_candidate_normalized: str,
    edited_gold_group_id: int | None,
    candidate_label: str,
    positive_source: str | None,
) -> PreferencePair | None:
    chosen_metrics, _chosen_matched = cluster_metrics(question, chosen_cluster_ids, clusters)
    rejected_metrics, _rejected_matched = cluster_metrics(question, rejected_cluster_ids, clusters)
    if chosen_metrics.f1 <= rejected_metrics.f1:
        return None
    return PreferencePair(
        pair_id=pair_id,
        dataset=question.dataset,
        question_id=question.question_id,
        question_text=question.question_text,
        question_source_path=question.source_path,
        prompt=response.matched_response.record.prompt,
        chosen=serialize_list_items(chosen_items),
        rejected=serialize_list_items(rejected_items),
        pair_type=pair_type,
        base_response_id=response.matched_response.record.response_id,
        edited_candidate=edited_candidate,
        edited_candidate_normalized=edited_candidate_normalized,
        edited_gold_group_id=edited_gold_group_id,
        candidate_label=candidate_label,
        candidate_label_source="proposal_semantic_cluster_alignment",
        positive_source=positive_source,
        semantic_set_edit_distance=1,
        chosen_items=tuple(chosen_items),
        rejected_items=tuple(rejected_items),
        chosen_precision=chosen_metrics.precision,
        chosen_recall=chosen_metrics.recall,
        chosen_f1=chosen_metrics.f1,
        rejected_precision=rejected_metrics.precision,
        rejected_recall=rejected_metrics.recall,
        rejected_f1=rejected_metrics.f1,
        delta_precision=chosen_metrics.precision - rejected_metrics.precision,
        delta_recall=chosen_metrics.recall - rejected_metrics.recall,
        delta_f1=chosen_metrics.f1 - rejected_metrics.f1,
        generator_checkpoint=response.matched_response.record.generator_checkpoint,
        sample_id=response.matched_response.record.sample_id,
    )


def dedupe_and_trim_pairs(pairs: Sequence[tuple[tuple[Any, ...], PreferencePair]], max_pairs: int) -> list[PreferencePair]:
    seen = set()
    selected: list[PreferencePair] = []
    for _rank, pair in sorted(pairs, key=lambda item: item[0], reverse=True):
        key = (pair.question_id, tuple(pair.chosen_items), tuple(pair.rejected_items), pair.pair_type)
        reverse_key = (pair.question_id, tuple(pair.rejected_items), tuple(pair.chosen_items), pair.pair_type)
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def build_negative_addition_pairs(
    *,
    question: QuestionExample,
    clustered_responses: Sequence[ClusteredResponse],
    clusters: Mapping[str, ClusterInfo],
    question_stats: Mapping[str, Any],
    max_pairs_per_question: int,
    pair_id_prefix: str,
    allow_evidence_supported_negative_pairs: bool,
    allow_uncertain_cluster_edits: bool,
    min_negative_addition_delta_f1: float,
) -> tuple[list[PreferencePair], Counter[str]]:
    audit = Counter()
    ranked_pairs: list[tuple[tuple[Any, ...], PreferencePair]] = []
    counter = 0
    for response in clustered_responses:
        if not response.matched_response.pair_eligible:
            continue
        rejected_items = response_items(response)
        for cluster_id in response.cluster_ids:
            cluster = clusters[cluster_id]
            if cluster.aligned_gold_group_id is not None:
                continue
            audit["candidate_metric_negative_clusters_total"] += 1
            if cluster.uncertain and not allow_uncertain_cluster_edits:
                audit["filtered_uncertain_negative_clusters"] += 1
                continue
            evidence_supported = candidate_is_evidence_supported(question, cluster.representative, question_stats)
            if evidence_supported:
                audit["evidence_supported_negative_clusters"] += 1
            if evidence_supported and not allow_evidence_supported_negative_pairs:
                audit["filtered_evidence_supported_negative_clusters"] += 1
                continue

            chosen_items = response_items_without_cluster(response, cluster_id)
            chosen_cluster_ids = cluster_ids_without(response.cluster_ids, cluster_id)
            pair = build_pair(
                pair_id=f"{pair_id_prefix}-cluster-addneg-{counter:04d}",
                pair_type=PAIR_TYPE_NEGATIVE_ADDITION,
                question=question,
                response=response,
                clusters=clusters,
                chosen_items=chosen_items,
                rejected_items=rejected_items,
                chosen_cluster_ids=chosen_cluster_ids,
                rejected_cluster_ids=response.cluster_ids,
                edited_candidate=cluster.representative,
                edited_candidate_normalized=normalize_answer_surface(cluster.representative),
                edited_gold_group_id=None,
                candidate_label=LABEL_METRIC_NEGATIVE,
                positive_source=None,
            )
            counter += 1
            if pair is None:
                continue
            if pair.delta_f1 < min_negative_addition_delta_f1:
                audit["filtered_low_delta_negative_pairs"] += 1
                continue
            audit["candidate_negative_pairs_after_filters"] += 1
            rank = (
                int(not evidence_supported),
                int(question_stats["negative_frequency"].get(cluster_id, 0)),
                int(round(pair.delta_f1 * 10_000)),
                -int(cluster.uncertain),
                cluster.representative,
            )
            ranked_pairs.append((rank, pair))
    selected = dedupe_and_trim_pairs(ranked_pairs, max_pairs=max_pairs_per_question)
    audit["emitted_negative_addition_pairs"] = len(selected)
    return selected, audit


def choose_positive_surface(
    *,
    gold_group_id: int,
    response: ClusteredResponse,
    question: QuestionExample,
    clusters: Mapping[str, ClusterInfo],
    question_stats: Mapping[str, Any],
    allow_gold_alias_fallback: bool,
    allow_evidence_unsupported_omission_pairs: bool,
    allow_uncertain_cluster_edits: bool,
) -> tuple[str | None, str | None, str | None]:
    rejected_cluster_ids = set(response.cluster_ids)
    for cluster_id in response.cluster_ids:
        cluster = clusters[cluster_id]
        if gold_group_id in cluster.uncertain_gold_group_ids and not allow_uncertain_cluster_edits:
            return None, None, "semantic_overlap_with_rejected_cluster"

    for surface, _count in question_stats["positive_surface_by_gold_group"].get(gold_group_id, Counter()).most_common():
        if not surface:
            continue
        evidence_supported = candidate_is_evidence_supported(question, surface, question_stats)
        if not evidence_supported and not allow_evidence_unsupported_omission_pairs:
            continue
        surface_cluster_id = None
        for candidate_cluster_id, cluster in clusters.items():
            cluster_surfaces = tuple(cluster.surfaces) + (cluster.representative,)
            if cluster.aligned_gold_group_id == gold_group_id and surface in cluster_surfaces:
                surface_cluster_id = candidate_cluster_id
                break
        if surface_cluster_id is not None and surface_cluster_id in rejected_cluster_ids:
            continue
        return surface, "sampled_valid_cluster", None

    if allow_gold_alias_fallback:
        gold_group = next(group for group in question.gold_groups if group.group_id == gold_group_id)
        return gold_group.canonical_alias, "gold_alias_fallback", None
    return None, None, "positive_cluster_not_observed"


def build_valid_omission_pairs(
    *,
    question: QuestionExample,
    clustered_responses: Sequence[ClusteredResponse],
    clusters: Mapping[str, ClusterInfo],
    question_stats: Mapping[str, Any],
    max_pairs_per_question: int,
    pair_id_prefix: str,
    allow_gold_alias_fallback: bool,
    allow_evidence_unsupported_omission_pairs: bool,
    allow_uncertain_cluster_edits: bool,
) -> tuple[list[PreferencePair], Counter[str]]:
    audit = Counter()
    ranked_pairs: list[tuple[tuple[Any, ...], PreferencePair]] = []
    counter = 0
    gold_group_to_cluster_id = {
        cluster.aligned_gold_group_id: cluster.cluster_id
        for cluster in clusters.values()
        if cluster.aligned_gold_group_id is not None
    }
    for response in clustered_responses:
        if not response.matched_response.pair_eligible:
            continue
        rejected_items = response_items(response)
        for gold_group_id in response.missing_gold_group_ids:
            audit["missing_gold_group_instances_total"] += 1
            positive_surface, positive_source, filter_reason = choose_positive_surface(
                gold_group_id=gold_group_id,
                response=response,
                question=question,
                clusters=clusters,
                question_stats=question_stats,
                allow_gold_alias_fallback=allow_gold_alias_fallback,
                allow_evidence_unsupported_omission_pairs=allow_evidence_unsupported_omission_pairs,
                allow_uncertain_cluster_edits=allow_uncertain_cluster_edits,
            )
            if not positive_surface:
                audit[f"filtered_{filter_reason or 'no_positive_surface'}"] += 1
                continue
            positive_cluster_id = gold_group_to_cluster_id.get(gold_group_id)
            if positive_cluster_id is None:
                audit["filtered_missing_gold_cluster"] += 1
                continue
            chosen_items = rejected_items + [positive_surface]
            chosen_cluster_ids = list(response.cluster_ids) + [positive_cluster_id]
            pair = build_pair(
                pair_id=f"{pair_id_prefix}-cluster-omit-{counter:04d}",
                pair_type=PAIR_TYPE_VALID_OMISSION,
                question=question,
                response=response,
                clusters=clusters,
                chosen_items=chosen_items,
                rejected_items=rejected_items,
                chosen_cluster_ids=chosen_cluster_ids,
                rejected_cluster_ids=response.cluster_ids,
                edited_candidate=positive_surface,
                edited_candidate_normalized=normalize_answer_surface(positive_surface),
                edited_gold_group_id=gold_group_id,
                candidate_label=LABEL_GOLD_MISSING,
                positive_source=positive_source,
            )
            counter += 1
            if pair is None:
                continue
            audit["candidate_valid_omission_pairs_after_filters"] += 1
            rank = (
                1 if positive_source == "sampled_valid_cluster" else 0,
                int(question_stats["positive_surface_by_gold_group"].get(gold_group_id, Counter()).get(positive_surface, 0)),
                int(round(pair.delta_f1 * 10_000)),
                normalize_answer_surface(positive_surface),
            )
            ranked_pairs.append((rank, pair))
    selected = dedupe_and_trim_pairs(ranked_pairs, max_pairs=max_pairs_per_question)
    audit["emitted_valid_omission_pairs"] = len(selected)
    return selected, audit


def build_cluster_response_audit(clustered_responses_by_question: Mapping[str, Sequence[ClusteredResponse]]) -> dict[str, Any]:
    parser_status = Counter()
    eligible = 0
    total = 0
    cluster_counts = []
    for responses in clustered_responses_by_question.values():
        for response in responses:
            total += 1
            parser_status[response.matched_response.parsed.status] += 1
            if response.matched_response.pair_eligible:
                eligible += 1
                cluster_counts.append(len(response.cluster_ids))
    return {
        "total_responses": total,
        "pair_eligible_responses": eligible,
        "pair_ineligible_responses": total - eligible,
        "parser_status_distribution": dict(sorted(parser_status.items())),
        "eligible_response_cluster_count_distribution": summarize_numeric(cluster_counts),
    }


def main() -> None:
    args = parse_args()
    questions_by_id, responses_by_question, responses_by_question_and_checkpoint = group_responses_by_question(
        question_input=args.question_input,
        candidate_input=args.candidate_input,
        dataset_name=args.dataset_name,
        allow_fallback_split=bool(args.allow_fallback_split),
        allow_fallback_pair_construction=bool(args.allow_fallback_pair_construction),
        max_resources=int(args.max_resources),
        max_resource_chars=int(args.max_resource_chars),
        gold_support_policy=str(args.gold_support_policy),
        question_limit=int(args.limit) if args.limit is not None else None,
    )
    clusters_by_question, surface_to_cluster_by_question = load_clusters(Path(args.cluster_input))

    clustered_by_question_and_checkpoint: dict[tuple[str, str], list[ClusteredResponse]] = defaultdict(list)
    clustered_by_question: dict[str, list[ClusteredResponse]] = defaultdict(list)
    for question_id, responses in responses_by_question.items():
        question = questions_by_id[question_id]
        clusters = clusters_by_question.get(question_id, {})
        surface_to_cluster = surface_to_cluster_by_question.get(question_id, {})
        for response in responses:
            clustered = clustered_response_from_matched(response, question, clusters, surface_to_cluster)
            checkpoint = clean_text(response.record.generator_checkpoint) or "unknown"
            clustered_by_question_and_checkpoint[(question_id, checkpoint)].append(clustered)
            clustered_by_question[question_id].append(clustered)

    all_pairs: list[PreferencePair] = []
    negative_audit_total = Counter()
    omission_audit_total = Counter()
    for (question_id, checkpoint), clustered_responses in clustered_by_question_and_checkpoint.items():
        question = questions_by_id[question_id]
        clusters = clusters_by_question.get(question_id, {})
        question_stats = build_question_statistics(clustered_responses, clusters, question)
        pair_id_prefix = f"{slugify(question.dataset)}-{slugify(question.question_id)}-{slugify(checkpoint)}"
        negative_pairs, negative_audit = build_negative_addition_pairs(
            question=question,
            clustered_responses=clustered_responses,
            clusters=clusters,
            question_stats=question_stats,
            max_pairs_per_question=int(args.max_negative_addition_pairs_per_question),
            pair_id_prefix=pair_id_prefix,
            allow_evidence_supported_negative_pairs=bool(args.allow_evidence_supported_negative_pairs),
            allow_uncertain_cluster_edits=bool(args.allow_uncertain_cluster_edits),
            min_negative_addition_delta_f1=float(args.min_negative_addition_delta_f1),
        )
        omission_pairs, omission_audit = build_valid_omission_pairs(
            question=question,
            clustered_responses=clustered_responses,
            clusters=clusters,
            question_stats=question_stats,
            max_pairs_per_question=int(args.max_valid_omission_pairs_per_question),
            pair_id_prefix=pair_id_prefix,
            allow_gold_alias_fallback=bool(args.allow_gold_alias_fallback),
            allow_evidence_unsupported_omission_pairs=bool(args.allow_evidence_unsupported_omission_pairs),
            allow_uncertain_cluster_edits=bool(args.allow_uncertain_cluster_edits),
        )
        all_pairs.extend(negative_pairs)
        all_pairs.extend(omission_pairs)
        negative_audit_total.update(negative_audit)
        omission_audit_total.update(omission_audit)

    pair_rows = [to_jsonable(pair) for pair in sorted(all_pairs, key=lambda item: (item.question_id, item.pair_type, item.pair_id))]
    write_jsonl(Path(args.output_jsonl), pair_rows)

    question_gold_counts = {question_id: len(question.gold_groups) for question_id, question in questions_by_id.items()}
    summary = build_summary(pair_rows, question_gold_counts=question_gold_counts)
    summary["cluster_response_audit"] = build_cluster_response_audit(clustered_by_question)
    summary["pair_filter_audit"] = {
        "negative_addition": dict(sorted(negative_audit_total.items())),
        "valid_omission": dict(sorted(omission_audit_total.items())),
    }
    summary["pair_construction_policy"] = {
        "proposal_style_semantic_clusters": True,
        "cluster_input": args.cluster_input,
        "gold_alignment_policy": "exact_bioasq_alias_only",
        "uncertain_clusters_excluded_from_edits": not bool(args.allow_uncertain_cluster_edits),
        "allow_gold_alias_fallback": bool(args.allow_gold_alias_fallback),
        "allow_evidence_supported_negative_pairs": bool(args.allow_evidence_supported_negative_pairs),
        "allow_evidence_unsupported_omission_pairs": bool(args.allow_evidence_unsupported_omission_pairs),
        "min_negative_addition_delta_f1": float(args.min_negative_addition_delta_f1),
        "max_resources": int(args.max_resources),
        "max_resource_chars": int(args.max_resource_chars),
        "gold_support_policy": str(args.gold_support_policy),
    }
    write_json(Path(args.summary_json), summary)

    markdown = build_manual_audit_markdown(
        pair_rows,
        question_input=args.question_input,
        dataset_name=args.dataset_name,
        sample_per_type=int(args.manual_audit_sample_per_type),
        seed=int(args.seed),
    )
    Path(args.manual_audit_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manual_audit_md).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
