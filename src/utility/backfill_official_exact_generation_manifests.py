from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_registry import get_project_root, resolve_repo_path, utc_now_iso
from src.prompt_registry import resolve_prompt_bundle
from src.utility.bioasq_official import OFFICIAL_EXACT_TYPES, evaluate_with_bioasq_java
from src.utility.config import QUESTION_INSTRUCTIONS
from src.utility.data import clean_text
from src.utility.eval_dataset import load_eval_examples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill existing generation manifests and scores.json files so exact-answer "
            "runs use the official BioASQ Java metrics as the primary aggregate."
        )
    )
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        help=(
            "Directory to scan recursively for manifest.json files. May be provided "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--manifest-file",
        action="append",
        default=[],
        help="Explicit manifest.json path to backfill. May be provided multiple times.",
    )
    parser.add_argument(
        "--bioasq-java-jar",
        default="third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
    )
    parser.add_argument(
        "--bioasq-java-version",
        type=int,
        default=5,
        choices=[2, 3, 5, 8, 9],
    )
    parser.add_argument("--bioasq-java-heap", default="4G")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_manifest_paths(args: argparse.Namespace, project_root: Path) -> list[Path]:
    manifest_paths: list[Path] = []

    for root_value in args.manifest_root:
        resolved_root = resolve_repo_path(root_value, project_root=project_root) or Path(root_value)
        if not resolved_root.exists():
            raise FileNotFoundError(f"Manifest root does not exist: {resolved_root}")
        manifest_paths.extend(sorted(resolved_root.rglob("manifest.json")))

    for file_value in args.manifest_file:
        resolved_file = resolve_repo_path(file_value, project_root=project_root) or Path(file_value)
        if not resolved_file.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {resolved_file}")
        manifest_paths.append(resolved_file)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in manifest_paths:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def build_eval_args(manifest: Mapping[str, Any]) -> argparse.Namespace:
    dataset = dict(manifest.get("dataset") or {})
    prompt = dict(manifest.get("prompt") or {})
    generation = dict(manifest.get("generation") or {})
    scoring = dict(manifest.get("scoring") or {})
    return argparse.Namespace(
        eval_input=list(dataset.get("eval_input") or []),
        question_types=list(dataset.get("question_types") or []),
        max_resources=int(dataset.get("max_resources") or 0),
        max_resource_chars=int(dataset.get("max_resource_chars") or 0),
        resource_selection=str(dataset.get("resource_selection") or "first"),
        resource_granularity=str(dataset.get("resource_granularity") or "document"),
        resource_window_mode=str(dataset.get("resource_window_mode") or "single"),
        resource_window_step=int(dataset.get("resource_window_step") or 0),
        resource_reranker_model=dataset.get("resource_reranker_model") or "sentence-transformers/all-MiniLM-L12-v2",
        resource_reranker_article_model=dataset.get("resource_reranker_article_model"),
        resource_reranker_device=dataset.get("resource_reranker_device") or "auto",
        resource_reranker_batch_size=32,
        max_summary_answers=1,
        max_factoid_answers=5,
        max_list_items=100,
        summary_reference_mode=str(dataset.get("summary_reference_mode") or generation.get("summary_reference_mode") or "first"),
        limit=None,
        chat_template=prompt.get("model_chat_templates", [None])[0] if isinstance(prompt.get("model_chat_templates"), list) else None,
        prompt_format=prompt.get("prompt_format"),
        prompt_registry_path=prompt.get("registry_path"),
        prompt_file=prompt.get("registry_path"),
        prompt=prompt.get("prompt_id"),
        bioasq_java_jar=scoring.get("bioasq_java_jar")
        or "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",
        bioasq_java_version=int(scoring.get("bioasq_java_version") or 5),
        bioasq_java_heap=str(scoring.get("bioasq_java_heap") or "4G"),
    )


def exact_only_question_types(question_types: Iterable[str]) -> bool:
    normalized = {clean_text(question_type).lower() for question_type in question_types if clean_text(question_type)}
    return bool(normalized) and normalized.issubset(set(OFFICIAL_EXACT_TYPES))


def load_examples_by_key(manifest: Mapping[str, Any], project_root: Path) -> dict[tuple[str, str], Any]:
    eval_args = build_eval_args(manifest)
    prompt_path_value = eval_args.prompt_file or eval_args.prompt_registry_path
    prompt_registry_path = resolve_repo_path(prompt_path_value, project_root=project_root)
    prompt_bundle = resolve_prompt_bundle(
        registry_path=prompt_registry_path,
        prompt_ref=eval_args.prompt,
        fallback_instructions=QUESTION_INSTRUCTIONS,
    )
    eval_paths = [Path(path) for path in eval_args.eval_input]
    examples = load_eval_examples(eval_paths, eval_args, prompt_instructions=prompt_bundle["instructions"])
    return {
        (clean_text(example.question_id), clean_text(example.question_type).lower()): example
        for example in examples
    }


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prediction payload must be a JSON list: {path}")
    rows = [dict(row) for row in payload if isinstance(row, Mapping)]
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def update_scores_payload(
    *,
    payload: Mapping[str, Any],
    official_payload: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(payload)
    updated["aggregate"] = official_payload["aggregate"]
    updated["batches"] = official_payload["batches"]
    updated["scorers"] = {"bioasq_java": dict(official_payload)}
    updated["scoring"] = {
        **dict(payload.get("scoring") or {}),
        "requested_backend": "bioasq_java",
        "selected_backend": "bioasq_java",
        "available_backends": ["bioasq_java"],
        "exact_answer_official_only": True,
    }
    return updated


def update_model_summary(
    *,
    model_summary: Mapping[str, Any],
    official_payload: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(model_summary)
    updated["aggregate"] = official_payload["aggregate"]
    updated["batches"] = official_payload["batches"]
    updated["scorers"] = {"bioasq_java": dict(official_payload)}
    updated["scoring"] = {
        **dict(model_summary.get("scoring") or {}),
        "requested_backend": "bioasq_java",
        "selected_backend": "bioasq_java",
        "available_backends": ["bioasq_java"],
        "exact_answer_official_only": True,
    }
    paths = dict(model_summary.get("paths") or {})
    paths["official_scores"] = str(Path(official_payload["paths"]["dir"]) / "official_scores.json")
    updated["paths"] = paths
    return updated


def resolve_existing_official_payload(model_summary: Mapping[str, Any], scores_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    model_scorers = model_summary.get("scorers") or {}
    official_payload = model_scorers.get("bioasq_java")
    if isinstance(official_payload, Mapping):
        return official_payload
    score_scorers = scores_payload.get("scorers") or {}
    official_payload = score_scorers.get("bioasq_java")
    if isinstance(official_payload, Mapping):
        return official_payload
    return None


def backfill_manifest(manifest_path: Path, args: argparse.Namespace, project_root: Path) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Manifest payload must be a JSON object: {manifest_path}")

    dataset = dict(manifest.get("dataset") or {})
    if not exact_only_question_types(dataset.get("question_types") or []):
        print(f"[skip] {manifest_path}: question types are not exact-answer-only.")
        return False

    updated_models: list[dict[str, Any]] = []
    examples_by_key: dict[tuple[str, str], Any] | None = None

    for model_summary in manifest.get("models") or []:
        if not isinstance(model_summary, Mapping):
            continue
        paths = dict(model_summary.get("paths") or {})
        predictions_path = Path(str(paths.get("predictions") or ""))
        scores_path = Path(str(paths.get("scores") or ""))
        model_dir = Path(str(paths.get("model_dir") or predictions_path.parent))
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores file not found: {scores_path}")

        scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
        official_payload = resolve_existing_official_payload(model_summary, scores_payload)
        if official_payload is None:
            if examples_by_key is None:
                try:
                    examples_by_key = load_examples_by_key(manifest, project_root=project_root)
                except FileNotFoundError as exc:
                    print(f"[skip] {manifest_path}: cannot reconstruct official scoring inputs ({exc}).")
                    return False
            prediction_rows = load_prediction_rows(predictions_path)
            model_label = str((model_summary.get("model") or {}).get("label") or model_dir.name)

            official_args = argparse.Namespace(
                bioasq_java_jar=args.bioasq_java_jar,
                bioasq_java_version=args.bioasq_java_version,
                bioasq_java_heap=args.bioasq_java_heap,
            )
            official_payload = evaluate_with_bioasq_java(
                prediction_rows=prediction_rows,
                examples_by_key=examples_by_key,
                model_label=model_label,
                model_dir=model_dir,
                args=official_args,
            )

        updated_scores_payload = update_scores_payload(payload=scores_payload, official_payload=official_payload)
        updated_model_summary = update_model_summary(model_summary=model_summary, official_payload=official_payload)

        if not args.dry_run:
            write_json(scores_path, updated_scores_payload)
        updated_models.append(updated_model_summary)

    updated_manifest = dict(manifest)
    updated_manifest["models"] = updated_models
    updated_manifest["scoring"] = {
        **dict(manifest.get("scoring") or {}),
        "requested_backend": "bioasq_java",
        "bioasq_java_jar": args.bioasq_java_jar,
        "bioasq_java_version": args.bioasq_java_version,
        "bioasq_java_heap": args.bioasq_java_heap,
    }
    updated_manifest["repaired_at"] = utc_now_iso()

    if not args.dry_run:
        write_json(manifest_path, updated_manifest)
    print(f"[updated] {manifest_path}")
    return True


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    manifest_paths = resolve_manifest_paths(args, project_root=project_root)
    if not manifest_paths:
        raise ValueError("No manifest files matched the requested roots/files.")

    updated_count = 0
    for manifest_path in manifest_paths:
        if backfill_manifest(manifest_path, args=args, project_root=project_root):
            updated_count += 1

    print(f"Backfilled {updated_count} manifest(s).")


if __name__ == "__main__":
    main()
