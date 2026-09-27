from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List

from src.model_registry import promote_alias, to_project_relative

from .adapter_save import save_adapter_and_tokenizer
from .data import prepare_train_eval_rows, save_prepared_records
from .dataset_builder import prepare_dataset
from .run_manager import (
    RunLayout,
    build_run_manifest,
    get_repo_root,
    persist_run_manifest,
    resolve_run_layout,
)
from .training import (
    build_trainer,
    load_model_and_tokenizer,
    resolve_selection_metric,
    save_training_curves,
    train_trainer,
    uses_generated_selection,
)


def apply_managed_paths(args: argparse.Namespace, run_layout: RunLayout) -> None:
    args.output_dir = str(run_layout.trainer_output_dir)
    args.save_model_dir = str(run_layout.adapter_dir)
    if run_layout.prepared_output_dir is not None:
        args.prepared_output_dir = str(run_layout.prepared_output_dir)


def resolve_selection_metric_from_rows(
    args: argparse.Namespace,
    train_rows: list[dict],
    eval_rows: list[dict],
) -> None:
    requested_metric = str(getattr(args, "selection_metric", "") or "").strip().lower()
    if requested_metric != "auto":
        return

    observed_question_types = {
        str(row.get("type", "")).strip().lower()
        for row in [*train_rows, *eval_rows]
        if str(row.get("type", "")).strip()
    }
    if observed_question_types == {"list"}:
        args.selection_metric = "generated_mean_f1"
    else:
        args.selection_metric = "eval_loss"


def maybe_save_prepared_outputs(
    args: argparse.Namespace,
    train_rows: list[dict],
    eval_rows: list[dict],
) -> None:
    if not args.prepared_output_dir:
        return
    prepared_dir = Path(args.prepared_output_dir)
    save_prepared_records(train_rows, prepared_dir / "train_prepared.json")
    save_prepared_records(eval_rows, prepared_dir / "eval_prepared.json")


def mirror_training_artifacts_to_model_dir(
    training_artifacts: dict[str, str],
    save_model_dir: str | os.PathLike[str],
) -> dict[str, str]:
    target_dir = Path(save_model_dir)
    mirrored: dict[str, str] = {}
    artifact_keys = ("log_history_json", "metrics_csv", "training_curves_png")

    for artifact_key in artifact_keys:
        artifact_path = training_artifacts.get(artifact_key)
        if not artifact_path:
            continue
        source_path = Path(artifact_path)
        if not source_path.exists():
            continue

        target_path = target_dir / source_path.name
        if source_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        mirrored[f"adapter_{artifact_key}"] = str(target_path)

    return mirrored


def write_training_completion_marker(
    save_model_dir: str | os.PathLike[str],
    *,
    run_layout: RunLayout,
    run_name: str | None,
    best_checkpoint: str | None,
    best_metric: float | None,
    train_examples: int,
    eval_examples: int,
    selection_metric: str | None,
    best_checkpoints: dict[str, dict] | None = None,
) -> str:
    marker_path = Path(save_model_dir) / "training_complete.json"
    marker_payload = {
        "best_checkpoint": best_checkpoint,
        "best_metric": best_metric,
        "eval_examples": eval_examples,
        "run_id": run_layout.run_id,
        "run_name": run_name or run_layout.run_id,
        "selection_metric": selection_metric,
        "status": "completed",
        "train_examples": train_examples,
    }
    if best_checkpoints:
        marker_payload["best_checkpoints"] = best_checkpoints
    marker_path.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(marker_path)


def resolve_best_checkpoints(trainer: object, generated_metric_callback: object | None) -> dict[str, dict]:
    best_checkpoints: dict[str, dict] = {}

    trainer_best_checkpoint = getattr(getattr(trainer, "state", None), "best_model_checkpoint", None)
    trainer_best_metric = getattr(getattr(trainer, "state", None), "best_metric", None)
    if trainer_best_checkpoint and trainer_best_metric is not None:
        best_checkpoints["eval_loss"] = {
            "checkpoint": str(trainer_best_checkpoint),
            "metric": float(trainer_best_metric),
        }

    if generated_metric_callback is None:
        return best_checkpoints

    generated_metric = getattr(generated_metric_callback, "best_metric", None)
    generated_checkpoint = getattr(generated_metric_callback, "best_model_path", None)
    generated_summary = getattr(generated_metric_callback, "best_summary", None)
    generated_metric_name = getattr(generated_metric_callback, "selection_metric", None)
    if (
        generated_metric_name
        and generated_checkpoint
        and isinstance(generated_metric, (int, float))
        and isinstance(generated_summary, dict)
    ):
        best_checkpoints[str(generated_metric_name)] = {
            "checkpoint": str(generated_checkpoint),
            "epoch": generated_summary.get("epoch"),
            "metric": float(generated_metric),
            "step": generated_summary.get("step"),
        }

    eval_loss_summary = getattr(generated_metric_callback, "best_eval_loss_summary", None)
    eval_loss_checkpoint = getattr(generated_metric_callback, "best_eval_loss_model_path", None)
    if isinstance(eval_loss_summary, dict) and eval_loss_checkpoint:
        metric_value = eval_loss_summary.get("metric_value")
        if isinstance(metric_value, (int, float)):
            best_checkpoints["eval_loss"] = {
                "checkpoint": str(eval_loss_checkpoint),
                "epoch": eval_loss_summary.get("epoch"),
                "metric": float(metric_value),
                "step": eval_loss_summary.get("step"),
            }

    return best_checkpoints


def register_status(
    args: argparse.Namespace,
    run_layout: RunLayout,
    project_root: Path,
    status: str,
    train_examples: int = 0,
    eval_examples: int = 0,
    promoted_aliases: List[str] | None = None,
    metrics: dict | None = None,
    best_checkpoint: str | None = None,
    best_metric: float | None = None,
    best_checkpoints: dict[str, dict] | None = None,
    eval_metrics: dict | None = None,
    training_artifacts: dict | None = None,
    error: dict | None = None,
) -> None:
    persist_run_manifest(
        run_layout,
        build_run_manifest(
            args=args,
            run_layout=run_layout,
            project_root=project_root,
            status=status,
            train_examples=train_examples,
            eval_examples=eval_examples,
            promoted_aliases=promoted_aliases,
            metrics=metrics,
            best_checkpoint=best_checkpoint,
            best_metric=best_metric,
            best_checkpoints=best_checkpoints,
            eval_metrics=eval_metrics,
            training_artifacts=training_artifacts,
            error=error,
        ),
    )


def run_training(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = get_repo_root()
    run_layout = resolve_run_layout(args, project_root=project_root)
    apply_managed_paths(args, run_layout)

    train_examples = 0
    eval_examples = 0
    promoted_aliases: List[str] = []
    training_artifacts: dict[str, str] = {}
    register_status(
        args=args,
        run_layout=run_layout,
        project_root=project_root,
        status="initializing",
    )

    try:
        train_rows, eval_rows = prepare_train_eval_rows(args)
        resolve_selection_metric_from_rows(args, train_rows=train_rows, eval_rows=eval_rows)
        train_examples = len(train_rows)
        eval_examples = len(eval_rows)
        maybe_save_prepared_outputs(args, train_rows, eval_rows)

        print(f"Prepared {train_examples:,} training examples.")
        print(f"Prepared {eval_examples:,} evaluation examples.")
        print(f"Managed run id: {run_layout.run_id}")
        print(
            "Run artifacts will be stored in "
            f"{to_project_relative(run_layout.run_dir, project_root=project_root)}"
        )

        register_status(
            args=args,
            run_layout=run_layout,
            project_root=project_root,
            status="prepared",
            train_examples=train_examples,
            eval_examples=eval_examples,
        )

        model, tokenizer = load_model_and_tokenizer(args)
        train_dataset = prepare_dataset(
            train_rows,
            tokenizer=tokenizer,
            num_proc=args.dataset_num_proc,
            max_seq_length=args.max_seq_length,
            prompt_format=args.prompt_format,
            response_template=args.response_template,
            response_template_trim_tokens=args.response_template_trim_tokens,
        )
        eval_dataset = None
        if eval_rows:
            eval_dataset = prepare_dataset(
                eval_rows,
                tokenizer=tokenizer,
                num_proc=args.dataset_num_proc,
                max_seq_length=args.max_seq_length,
                prompt_format=args.prompt_format,
                response_template=args.response_template,
                response_template_trim_tokens=args.response_template_trim_tokens,
            )

        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_rows=eval_rows,
            args=args,
        )

        trainer_stats = train_trainer(trainer, args)
        print(trainer_stats)

        eval_metrics = {}
        generated_metric_callback = getattr(trainer, "generated_metric_callback", None)
        if generated_metric_callback is not None and generated_metric_callback.best_summary is not None:
            eval_metrics = {
                "selection_metric": generated_metric_callback.selection_metric,
                "generated_dev": generated_metric_callback.best_summary.get("aggregate", {}),
                "generated_dev_best_metric": generated_metric_callback.best_metric,
                "generated_dev_best_step": generated_metric_callback.best_summary.get("step"),
                "generated_dev_best_epoch": generated_metric_callback.best_summary.get("epoch"),
            }
            if generated_metric_callback.best_eval_loss_summary is not None:
                eval_metrics["eval_loss_best"] = {
                    "eval_loss": generated_metric_callback.best_eval_loss_summary.get("metric_value"),
                    "model_dir": generated_metric_callback.best_eval_loss_model_path,
                    "step": generated_metric_callback.best_eval_loss_summary.get("step"),
                    "epoch": generated_metric_callback.best_eval_loss_summary.get("epoch"),
                }
            print(
                "Best generated-dev selection metrics: "
                f"{eval_metrics['selection_metric']}={generated_metric_callback.best_metric}"
            )
            if generated_metric_callback.best_eval_loss_summary is not None:
                print(
                    "Best validation loss during generated-dev selection: "
                    f"eval_loss={generated_metric_callback.best_eval_loss_summary.get('metric_value')}"
                )
        elif eval_dataset is not None and len(eval_dataset) > 0:
            eval_metrics = dict(trainer.evaluate())
            print(f"Best-model eval metrics: {eval_metrics}")

        if not (
            uses_generated_selection(args)
            and generated_metric_callback is not None
            and generated_metric_callback.best_metric is not None
        ):
            save_adapter_and_tokenizer(
                model,
                tokenizer,
                args.save_model_dir,
                save_dtype=getattr(args, "save_dtype", "float32"),
            )
        training_artifacts = save_training_curves(trainer, run_layout.run_dir)
        training_artifacts.update(
            mirror_training_artifacts_to_model_dir(
                training_artifacts,
                args.save_model_dir,
            )
        )
        if generated_metric_callback is not None:
            training_artifacts.update(generated_metric_callback.artifact_paths)
        print(f"Saved adapter and tokenizer to {args.save_model_dir}")

        if args.register_alias:
            alias_data = promote_alias(
                run_layout.registry_path,
                alias=args.register_alias,
                run_id=run_layout.run_id,
            )
            promoted_aliases.append(alias_data["alias"])
            print(f"Promoted alias {alias_data['alias']} -> {alias_data['run_id']}")

        trainer_metrics = dict(getattr(trainer_stats, "metrics", {}) or {})
        best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
        best_metric = getattr(trainer.state, "best_metric", None)
        if generated_metric_callback is not None and generated_metric_callback.best_metric is not None:
            best_checkpoint = generated_metric_callback.best_model_path
            best_metric = generated_metric_callback.best_metric
        best_checkpoints = resolve_best_checkpoints(trainer, generated_metric_callback)
        training_artifacts["completion_marker_json"] = write_training_completion_marker(
            args.save_model_dir,
            run_layout=run_layout,
            run_name=args.run_name,
            best_checkpoint=best_checkpoint,
            best_metric=best_metric,
            train_examples=train_examples,
            eval_examples=eval_examples,
            selection_metric=resolve_selection_metric(args),
            best_checkpoints=best_checkpoints,
        )
        register_status(
            args=args,
            run_layout=run_layout,
            project_root=project_root,
            status="completed",
            train_examples=train_examples,
            eval_examples=eval_examples,
            promoted_aliases=promoted_aliases,
            metrics=trainer_metrics,
            best_checkpoint=best_checkpoint,
            best_metric=best_metric,
            best_checkpoints=best_checkpoints,
            eval_metrics=eval_metrics,
            training_artifacts=training_artifacts,
        )
        print(
            "Saved run manifest to "
            f"{to_project_relative(run_layout.manifest_path, project_root=project_root)}"
        )
    except Exception as exc:
        register_status(
            args=args,
            run_layout=run_layout,
            project_root=project_root,
            status="failed",
            train_examples=train_examples,
            eval_examples=eval_examples,
            promoted_aliases=promoted_aliases,
            error={
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
