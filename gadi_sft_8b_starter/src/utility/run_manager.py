from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.model_registry import (
    build_run_id,
    get_project_root,
    resolve_repo_path,
    to_project_relative,
    upsert_run,
    utc_now_iso,
    write_manifest,
)

from .config import TASK_NAME


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    created_at: str
    artifacts_root: Path
    registry_path: Path
    run_dir: Path
    manifest_path: Path
    trainer_output_dir: Path
    adapter_dir: Path
    prepared_output_dir: Optional[Path]


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: value for key, value in vars(args).items()}


def resolve_run_layout(args: argparse.Namespace, project_root: Path) -> RunLayout:
    run_id = build_run_id(
        task=TASK_NAME,
        model_name=args.model_name,
        question_types=args.question_types,
        run_name=args.run_name,
    )
    artifacts_root = resolve_repo_path(args.artifacts_root, project_root=project_root)
    registry_path = resolve_repo_path(args.registry_path, project_root=project_root)
    if artifacts_root is None or registry_path is None:
        raise ValueError("Managed artifacts root and registry path must be set.")

    run_dir = artifacts_root / "runs" / run_id
    trainer_output_dir = resolve_repo_path(args.output_dir, project_root=project_root) or (run_dir / "trainer_output")
    adapter_dir = resolve_repo_path(args.save_model_dir, project_root=project_root) or (run_dir / "adapter")
    prepared_output_dir = resolve_repo_path(args.prepared_output_dir, project_root=project_root)

    return RunLayout(
        run_id=run_id,
        created_at=utc_now_iso(),
        artifacts_root=artifacts_root,
        registry_path=registry_path,
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        trainer_output_dir=trainer_output_dir,
        adapter_dir=adapter_dir,
        prepared_output_dir=prepared_output_dir,
    )


def build_run_manifest(
    args: argparse.Namespace,
    run_layout: RunLayout,
    project_root: Path,
    status: str,
    train_examples: Optional[int] = None,
    eval_examples: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    best_checkpoint: Optional[str] = None,
    best_metric: Optional[float] = None,
    best_checkpoints: Optional[Dict[str, Dict[str, Any]]] = None,
    eval_metrics: Optional[Dict[str, Any]] = None,
    training_artifacts: Optional[Dict[str, str]] = None,
    promoted_aliases: Optional[List[str]] = None,
    error: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    paths = {
        "artifacts_root": to_project_relative(run_layout.artifacts_root, project_root=project_root),
        "run_dir": to_project_relative(run_layout.run_dir, project_root=project_root),
        "manifest_path": to_project_relative(run_layout.manifest_path, project_root=project_root),
        "registry_path": to_project_relative(run_layout.registry_path, project_root=project_root),
        "trainer_output_dir": to_project_relative(run_layout.trainer_output_dir, project_root=project_root),
        "adapter_dir": to_project_relative(run_layout.adapter_dir, project_root=project_root),
        "prepared_output_dir": to_project_relative(run_layout.prepared_output_dir, project_root=project_root),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_layout.run_id,
        "run_name": args.run_name or run_layout.run_id,
        "task": TASK_NAME,
        "status": status,
        "created_at": run_layout.created_at,
        "updated_at": utc_now_iso(),
        "base_model": args.model_name,
        "chat_template": args.chat_template,
        "prompt_format": args.prompt_format,
        "question_types": list(args.question_types),
        "train_input": list(args.train_input),
        "eval_input": list(args.eval_input or []),
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "paths": paths,
        "config": namespace_to_dict(args),
    }
    if metrics:
        manifest["metrics"] = metrics
    if best_checkpoint:
        manifest["best_checkpoint"] = to_project_relative(best_checkpoint, project_root=project_root)
    if best_metric is not None:
        manifest["best_metric"] = best_metric
    if best_checkpoints:
        manifest["best_checkpoints"] = {}
        for metric_name, checkpoint_data in best_checkpoints.items():
            normalized_entry = dict(checkpoint_data)
            checkpoint_path = normalized_entry.get("checkpoint")
            if checkpoint_path:
                normalized_entry["checkpoint"] = to_project_relative(
                    checkpoint_path,
                    project_root=project_root,
                )
            manifest["best_checkpoints"][metric_name] = normalized_entry
    if eval_metrics:
        manifest["eval_metrics"] = eval_metrics
    if training_artifacts:
        manifest["training_artifacts"] = {
            key: to_project_relative(Path(value), project_root=project_root)
            for key, value in training_artifacts.items()
        }
    if promoted_aliases:
        manifest["promoted_aliases"] = promoted_aliases
    if error:
        manifest["error"] = error
    return manifest


def persist_run_manifest(run_layout: RunLayout, manifest: Dict[str, Any]) -> None:
    write_manifest(run_layout.manifest_path, manifest)
    upsert_run(run_layout.registry_path, manifest)


def get_repo_root() -> Path:
    return get_project_root()
