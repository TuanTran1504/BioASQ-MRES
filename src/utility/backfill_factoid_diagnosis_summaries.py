from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import get_project_root, resolve_repo_path, utc_now_iso


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill factoid candidate-diagnosis summaries with submission-aligned "
            "official BioASQ metrics from the corresponding generation manifests."
        )
    )
    parser.add_argument(
        "--summary-root",
        action="append",
        default=[],
        help="Directory to scan recursively for summary.json files. May be provided multiple times.",
    )
    parser.add_argument(
        "--summary-file",
        action="append",
        default=[],
        help="Explicit summary.json file to backfill. May be provided multiple times.",
    )
    parser.add_argument(
        "--generation-root",
        default="Artifacts/Factoid_SFT/candidate_diagnosis/generations",
        help="Root containing the paired generation manifests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_summary_paths(args: argparse.Namespace, project_root: Path) -> list[Path]:
    summary_paths: list[Path] = []

    for root_value in args.summary_root:
        resolved_root = resolve_repo_path(root_value, project_root=project_root) or Path(root_value)
        if not resolved_root.exists():
            raise FileNotFoundError(f"Summary root does not exist: {resolved_root}")
        summary_paths.extend(sorted(resolved_root.rglob("summary.json")))

    for file_value in args.summary_file:
        resolved_file = resolve_repo_path(file_value, project_root=project_root) or Path(file_value)
        if not resolved_file.exists():
            raise FileNotFoundError(f"Summary file does not exist: {resolved_file}")
        summary_paths.append(resolved_file)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in summary_paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_generation_manifest(
    summary_path: Path,
    *,
    project_root: Path,
    generation_root: Path,
) -> Path:
    parts = summary_path.parts
    if "analysis" in parts:
        anchor = parts.index("analysis")
        rel_parts = parts[anchor + 1 : anchor + 4]
    elif "reanalysis" in parts:
        anchor = parts.index("reanalysis")
        rel_parts = parts[anchor + 1 : anchor + 4]
    else:
        raise ValueError(f"Could not infer generation manifest for summary: {summary_path}")

    manifest_path = generation_root.joinpath(*rel_parts, "manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Generation manifest not found for {summary_path}: {manifest_path}")
    return manifest_path


def extract_official_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    models = manifest.get("models") or []
    if not models:
        raise ValueError("Generation manifest does not contain any model summaries.")
    model_summary = models[0]
    if not isinstance(model_summary, Mapping):
        raise ValueError("Generation manifest model summary is malformed.")
    scorers = model_summary.get("scorers") or {}
    official = scorers.get("bioasq_java")
    if not isinstance(official, Mapping):
        return None
    return official


def build_submission_aligned_block(official_payload: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = dict(official_payload.get("aggregate") or {})
    by_type = dict(aggregate.get("by_type") or {})
    factoid = dict(by_type.get("factoid") or {})
    return {
        "source": "official_saved_prediction",
        "question_count": int(factoid.get("question_count") or aggregate.get("question_count") or 0),
        "primary_metric": factoid.get("primary_metric") or "mrr",
        "metrics": dict(factoid.get("metrics") or {}),
        "note": (
            "Use these official BioASQ metrics as the submission-aligned headline results. "
            "The ranking, candidate_pool, and output_format sections remain parser-based diagnostics."
        ),
    }


def remove_legacy_local_score_fields(block: dict[str, Any]) -> None:
    for key in ("mean_saved_strict_accuracy", "mean_saved_lenient_accuracy", "mean_saved_mrr"):
        block.pop(key, None)


def normalize_legacy_union_block(block: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(block)
    remove_legacy_local_score_fields(normalized)
    if "mrr_from_rank_positions" in normalized and "mrr" not in normalized:
        normalized["mrr"] = normalized.pop("mrr_from_rank_positions")
    if "gold_absent_from_top_k_count" in normalized and "gold_absent_from_pool_count" not in normalized:
        normalized["gold_absent_from_pool_count"] = normalized.pop("gold_absent_from_top_k_count")
    if "gold_absent_from_top_k_rate" in normalized and "gold_absent_from_pool_rate" not in normalized:
        normalized["gold_absent_from_pool_rate"] = normalized.pop("gold_absent_from_top_k_rate")
    normalized.setdefault("gold_below_top_k_but_present_count", 0)
    normalized.setdefault("gold_below_top_k_but_present_rate", 0.0)
    return normalized


def promote_direct_generation_to_union_ranking(summary: dict[str, Any]) -> None:
    ranking = summary.get("ranking")
    if isinstance(ranking, Mapping):
        ranking_block = dict(ranking)
    else:
        ranking_block = {}

    union_block = ranking_block.get("union")
    if isinstance(union_block, dict):
        ranking_block["union"] = normalize_legacy_union_block(union_block)

    direct_generation = summary.pop("direct_generation", None)
    if not isinstance(direct_generation, dict):
        if ranking_block:
            summary["ranking"] = ranking_block
        return

    normalized_union = normalize_legacy_union_block(direct_generation)
    if "union" not in ranking_block:
        updated_ranking = {"union": normalized_union}
        updated_ranking.update(ranking_block)
        summary["ranking"] = updated_ranking
        return

    summary["ranking"] = ranking_block


def backfill_summary(
    summary_path: Path,
    *,
    project_root: Path,
    generation_root: Path,
    dry_run: bool,
) -> bool:
    summary = dict(load_json(summary_path))
    manifest_path = resolve_generation_manifest(
        summary_path,
        project_root=project_root,
        generation_root=generation_root,
    )
    manifest = load_json(manifest_path)
    official_payload = extract_official_payload(manifest)

    promote_direct_generation_to_union_ranking(summary)
    if official_payload is not None:
        summary["official_saved_prediction"] = {
            "aggregate": official_payload.get("aggregate"),
            "paths": official_payload.get("paths"),
            "command": official_payload.get("command"),
        }
        summary["submission_aligned"] = build_submission_aligned_block(official_payload)
    summary["metric_notes"] = {
        "headline": (
            "submission_aligned and official_saved_prediction use the official BioASQ "
            "Java evaluation and should be treated as the primary scores."
        ),
        "diagnostic": (
            "ranking, candidate_pool, and output_format quantify "
            "candidate discovery and reranking behavior; they are not identical to the "
            "official submission metric."
        ),
    }
    summary["repaired_at"] = utc_now_iso()

    if not dry_run:
        write_json(summary_path, summary)
    print(f"[updated] {summary_path}")
    return True


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    summary_paths = resolve_summary_paths(args, project_root=project_root)
    if not summary_paths:
        raise ValueError("No summary.json files matched the requested roots/files.")

    generation_root = resolve_repo_path(args.generation_root, project_root=project_root) or Path(args.generation_root)
    if not generation_root.exists():
        raise FileNotFoundError(f"Generation root does not exist: {generation_root}")

    updated_count = 0
    for summary_path in summary_paths:
        if backfill_summary(
            summary_path,
            project_root=project_root,
            generation_root=generation_root,
            dry_run=args.dry_run,
        ):
            updated_count += 1

    print(f"Backfilled {updated_count} summary file(s).")


if __name__ == "__main__":
    main()
