from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.utility.data import clean_text

from .construct_set_edit_pairs import surface_overlap_reason
from .hybrid_semantic_labeler import HybridSemanticEquivalenceLabeler
from .normalize_set_answers import normalize_answer_surface, parse_list_output
from .schemas import CandidateBankRecord, QuestionExample


@dataclass(frozen=True)
class SurfaceEntry:
    surface: str
    normalized: str
    source_types: tuple[str, ...]
    gold_group_ids: tuple[int, ...]
    generated_count: int
    first_seen_index: int


@dataclass
class SemanticCluster:
    cluster_id: str
    surfaces: list[str]
    normalized_surfaces: set[str]
    source_types: set[str]
    gold_group_ids: set[int]
    generated_count: int
    uncertain: bool
    uncertain_reasons: list[str]
    uncertain_gold_group_ids: set[int]

    @property
    def aligned_gold_group_id(self) -> int | None:
        return next(iter(self.gold_group_ids)) if len(self.gold_group_ids) == 1 else None


def surface_key(surface: str) -> str:
    normalized = normalize_answer_surface(surface)
    return normalized or clean_text(surface).lower()


def _entry_sort_key(entry: SurfaceEntry) -> tuple[int, int, int, str]:
    return (
        0 if "gold_alias" in entry.source_types else 1,
        min(entry.gold_group_ids) if entry.gold_group_ids else 10_000,
        -entry.generated_count,
        entry.normalized,
    )


def collect_surface_entries(
    *,
    question: QuestionExample,
    records: Sequence[CandidateBankRecord],
    allow_fallback_split: bool,
    eligible_only: bool = True,
) -> list[SurfaceEntry]:
    surface_sources: dict[str, set[str]] = defaultdict(set)
    surface_gold_ids: dict[str, set[int]] = defaultdict(set)
    surface_counts: Counter[str] = Counter()
    surface_by_key: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    index = 0

    for gold_group in question.gold_groups:
        for alias in gold_group.aliases:
            key = surface_key(alias)
            if not key:
                continue
            surface_by_key.setdefault(key, clean_text(alias))
            surface_sources[key].add("gold_alias")
            surface_gold_ids[key].add(gold_group.group_id)
            first_seen.setdefault(key, index)
            index += 1

    for record in records:
        parsed = parse_list_output(record.raw_output, allow_fallback_split=allow_fallback_split)
        if eligible_only and parsed.status != "ok":
            continue
        for item in parsed.items:
            key = surface_key(item)
            if not key:
                continue
            surface_by_key.setdefault(key, clean_text(item))
            surface_sources[key].add("generated")
            surface_counts[key] += 1
            first_seen.setdefault(key, index)
            index += 1

    entries: list[SurfaceEntry] = []
    for key, surface in surface_by_key.items():
        entries.append(
            SurfaceEntry(
                surface=surface,
                normalized=key,
                source_types=tuple(sorted(surface_sources[key])),
                gold_group_ids=tuple(sorted(surface_gold_ids[key])),
                generated_count=int(surface_counts[key]),
                first_seen_index=first_seen[key],
            )
        )
    return sorted(entries, key=_entry_sort_key)


def _new_cluster(index: int, entry: SurfaceEntry, uncertain: bool = False, reason: str | None = None) -> SemanticCluster:
    return SemanticCluster(
        cluster_id=f"c{index:04d}",
        surfaces=[entry.surface],
        normalized_surfaces={entry.normalized},
        source_types=set(entry.source_types),
        gold_group_ids=set(entry.gold_group_ids),
        generated_count=entry.generated_count,
        uncertain=uncertain,
        uncertain_reasons=[reason] if reason else [],
        uncertain_gold_group_ids=set(),
    )


def _add_entry(cluster: SemanticCluster, entry: SurfaceEntry) -> None:
    if entry.surface not in cluster.surfaces:
        cluster.surfaces.append(entry.surface)
    cluster.normalized_surfaces.add(entry.normalized)
    cluster.source_types.update(entry.source_types)
    cluster.gold_group_ids.update(entry.gold_group_ids)
    cluster.generated_count += entry.generated_count


def _cluster_gold_ids(cluster: SemanticCluster) -> set[int]:
    return set(cluster.gold_group_ids) | set(cluster.uncertain_gold_group_ids)


def _mark_uncertain(cluster: SemanticCluster, reason: str, gold_group_ids: Iterable[int] = ()) -> None:
    cluster.uncertain = True
    if reason not in cluster.uncertain_reasons:
        cluster.uncertain_reasons.append(reason)
    cluster.uncertain_gold_group_ids.update(gold_group_ids)


def _compare_surfaces(
    *,
    question: QuestionExample,
    left: SurfaceEntry,
    right_surface: str,
    right_gold_group_ids: set[int],
    labeler: HybridSemanticEquivalenceLabeler | None,
    mark_lexical_overlap_uncertain: bool,
) -> tuple[str, str]:
    right_normalized = surface_key(right_surface)
    if left.normalized and left.normalized == right_normalized:
        return "equivalent", "exact_normalized"

    left_gold_ids = set(left.gold_group_ids)
    shared_gold_ids = left_gold_ids & right_gold_group_ids
    if shared_gold_ids:
        return "equivalent", "same_bioasq_gold_alias_group"

    involves_gold = bool(left_gold_ids or right_gold_group_ids)

    if labeler is not None:
        decision = labeler.answer_equivalent(left.surface, right_surface, question=question)
        if decision.matched:
            if involves_gold:
                return "uncertain", f"semantic_close_to_gold:{decision.match_type}"
            return "equivalent", decision.match_type
        if decision.uncertain:
            return "uncertain", decision.match_type

    if mark_lexical_overlap_uncertain:
        overlap_reason = surface_overlap_reason(question, left.surface, right_surface)
        if overlap_reason is not None:
            return "uncertain", overlap_reason

    return "different", "different"


def _entry_relation_to_cluster(
    *,
    question: QuestionExample,
    entry: SurfaceEntry,
    cluster: SemanticCluster,
    labeler: HybridSemanticEquivalenceLabeler | None,
    mark_lexical_overlap_uncertain: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for surface in cluster.surfaces:
        relation, reason = _compare_surfaces(
            question=question,
            left=entry,
            right_surface=surface,
            right_gold_group_ids=cluster.gold_group_ids,
            labeler=labeler,
            mark_lexical_overlap_uncertain=mark_lexical_overlap_uncertain,
        )
        if relation == "different":
            return "different", reasons
        reasons.append(reason)
        if relation == "uncertain":
            return "uncertain", reasons
    return "equivalent", reasons


def build_semantic_clusters_for_question(
    *,
    question: QuestionExample,
    records: Sequence[CandidateBankRecord],
    labeler: HybridSemanticEquivalenceLabeler | None,
    allow_fallback_split: bool = False,
    mark_lexical_overlap_uncertain: bool = True,
) -> list[SemanticCluster]:
    entries = collect_surface_entries(
        question=question,
        records=records,
        allow_fallback_split=allow_fallback_split,
    )
    clusters: list[SemanticCluster] = []

    for entry in entries:
        uncertain_against: list[tuple[SemanticCluster, list[str]]] = []
        joined = False
        for cluster in clusters:
            relation, reasons = _entry_relation_to_cluster(
                question=question,
                entry=entry,
                cluster=cluster,
                labeler=labeler,
                mark_lexical_overlap_uncertain=mark_lexical_overlap_uncertain,
            )
            if relation == "equivalent" and not cluster.uncertain:
                _add_entry(cluster, entry)
                joined = True
                break
            if relation == "uncertain":
                uncertain_against.append((cluster, reasons))

        if joined:
            continue

        cluster = _new_cluster(len(clusters), entry)
        for other_cluster, reasons in uncertain_against:
            reason = ";".join(dict.fromkeys(reasons)) or "uncertain_semantic_relation"
            _mark_uncertain(cluster, reason, gold_group_ids=_cluster_gold_ids(other_cluster))
            _mark_uncertain(other_cluster, reason, gold_group_ids=entry.gold_group_ids)
        clusters.append(cluster)

    gold_clusters = [cluster for cluster in clusters if cluster.gold_group_ids]
    for cluster in clusters:
        if len(cluster.gold_group_ids) > 1:
            _mark_uncertain(cluster, "multiple_gold_groups_in_cluster", gold_group_ids=cluster.gold_group_ids)
        if cluster.gold_group_ids:
            continue
        for gold_cluster in gold_clusters:
            relation, reasons = _entry_relation_to_cluster(
                question=question,
                entry=SurfaceEntry(
                    surface=cluster.surfaces[0],
                    normalized=surface_key(cluster.surfaces[0]),
                    source_types=tuple(cluster.source_types),
                    gold_group_ids=(),
                    generated_count=cluster.generated_count,
                    first_seen_index=0,
                ),
                cluster=gold_cluster,
                labeler=labeler,
                mark_lexical_overlap_uncertain=mark_lexical_overlap_uncertain,
            )
            if relation in {"equivalent", "uncertain"}:
                reason = ";".join(dict.fromkeys(reasons)) or "semantic_close_to_gold"
                _mark_uncertain(cluster, reason, gold_group_ids=gold_cluster.gold_group_ids)

    return clusters


def cluster_to_row(question: QuestionExample, cluster: SemanticCluster) -> dict[str, Any]:
    aligned_gold_group_id = cluster.aligned_gold_group_id
    representative = cluster.surfaces[0]
    if aligned_gold_group_id is not None:
        gold_group = next((group for group in question.gold_groups if group.group_id == aligned_gold_group_id), None)
        if gold_group is not None:
            representative = gold_group.canonical_alias
    return {
        "cluster_id": cluster.cluster_id,
        "representative": representative,
        "surfaces": list(cluster.surfaces),
        "normalized_surfaces": sorted(cluster.normalized_surfaces),
        "source_types": sorted(cluster.source_types),
        "gold_group_ids": sorted(cluster.gold_group_ids),
        "aligned_gold_group_id": aligned_gold_group_id,
        "generated_count": int(cluster.generated_count),
        "uncertain": bool(cluster.uncertain),
        "uncertain_reasons": list(cluster.uncertain_reasons),
        "uncertain_gold_group_ids": sorted(cluster.uncertain_gold_group_ids),
    }


def cluster_rows_by_question(
    *,
    questions_by_id: Mapping[str, QuestionExample],
    records_by_question: Mapping[str, Sequence[CandidateBankRecord]],
    labeler: HybridSemanticEquivalenceLabeler | None,
    allow_fallback_split: bool = False,
    mark_lexical_overlap_uncertain: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id, question in sorted(questions_by_id.items()):
        clusters = build_semantic_clusters_for_question(
            question=question,
            records=records_by_question.get(question_id, ()),
            labeler=labeler,
            allow_fallback_split=allow_fallback_split,
            mark_lexical_overlap_uncertain=mark_lexical_overlap_uncertain,
        )
        rows.append(
            {
                "dataset": question.dataset,
                "question_id": question.question_id,
                "question_text": question.question_text,
                "gold_group_count": len(question.gold_groups),
                "clusters": [cluster_to_row(question, cluster) for cluster in clusters],
            }
        )
    return rows


def summarize_cluster_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cluster_count = 0
    uncertain_count = 0
    gold_aligned_count = 0
    generated_cluster_count = 0
    reason_counts: Counter[str] = Counter()
    multi_gold_count = 0
    for row in rows:
        for cluster in row.get("clusters", []):
            cluster_count += 1
            if "generated" in set(cluster.get("source_types", [])):
                generated_cluster_count += 1
            if cluster.get("aligned_gold_group_id") is not None:
                gold_aligned_count += 1
            if cluster.get("uncertain"):
                uncertain_count += 1
                reason_counts.update(cluster.get("uncertain_reasons", []))
            if len(cluster.get("gold_group_ids", [])) > 1:
                multi_gold_count += 1
    return {
        "question_count": len(rows),
        "cluster_count": cluster_count,
        "generated_cluster_count": generated_cluster_count,
        "gold_aligned_cluster_count": gold_aligned_count,
        "uncertain_cluster_count": uncertain_count,
        "uncertain_cluster_rate": uncertain_count / cluster_count if cluster_count else 0.0,
        "multi_gold_cluster_count": multi_gold_count,
        "uncertain_reason_counts": dict(sorted(reason_counts.items())),
    }
