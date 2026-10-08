from __future__ import annotations

import argparse
import gc
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import Unsloth before PyTorch in the portable starter environment.
import unsloth

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments
from trl import SFTTrainer
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:  # pragma: no cover - depends on local TRL version
    DataCollatorForCompletionOnlyLM = None
try:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as TRLDataCollatorForLanguageModeling
except ImportError:  # pragma: no cover - depends on local TRL version
    TRLDataCollatorForLanguageModeling = None
try:
    from trl import SFTConfig
except ImportError:  # pragma: no cover - compatibility with older TRL
    SFTConfig = None

from .adapter_save import save_adapter_and_tokenizer
from .bioasq_official import OFFICIAL_EXACT_TYPES, evaluate_with_bioasq_java
from .config import QUESTION_INSTRUCTIONS
from .data import clean_multiline_text, clean_text, list_record_resources
from .eval_models import generate_answer_samples
from .eval_types import EvalExample
from .text_tokenizer import text_only_tokenizer


def resolve_selection_metric(args: argparse.Namespace) -> str:
    metric = clean_text(getattr(args, "selection_metric", "auto")).lower() or "auto"
    if metric != "auto":
        return metric

    normalized_question_types = [
        clean_text(question_type).lower()
        for question_type in getattr(args, "question_types", []) or []
        if clean_text(question_type)
    ]
    return "generated_mean_f1" if normalized_question_types == ["list"] else "eval_loss"


def uses_generated_selection(args: argparse.Namespace) -> bool:
    return resolve_selection_metric(args).startswith("generated_")


def resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name in {None, ""}:
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_save_strategy(args: argparse.Namespace) -> str:
    strategy = clean_text(getattr(args, "save_strategy", "epoch")).lower() or "epoch"
    if strategy not in {"epoch", "steps"}:
        raise ValueError(f"Unsupported save strategy: {strategy}")
    return strategy


def resolve_eval_strategy(
    args: argparse.Namespace,
    *,
    should_run_trainer_eval: bool,
    save_strategy: str,
) -> str:
    if not should_run_trainer_eval:
        return "no"

    strategy = clean_text(getattr(args, "eval_strategy", "auto")).lower() or "auto"
    if strategy == "auto":
        return save_strategy
    if strategy == "no":
        raise ValueError(
            "Trainer-side evaluation is required to track validation loss during "
            "training. Use --eval-strategy epoch or steps."
        )
    if strategy not in {"epoch", "steps"}:
        raise ValueError(f"Unsupported eval strategy: {strategy}")
    return strategy


def resolve_step_interval(raw_value: Any, *, name: str, fallback: Optional[int] = None) -> int:
    if raw_value in {None, ""}:
        if fallback is not None:
            return fallback
        raise ValueError(f"{name} must be provided.")

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str | None) -> str | None:
    requested = clean_text(resume_from_checkpoint)
    if not requested:
        return None

    normalized = requested.lower()
    if normalized in {"latest", "auto"}:
        from transformers.trainer_utils import get_last_checkpoint

        checkpoint_path = get_last_checkpoint(output_dir)
        if checkpoint_path is None:
            if normalized == "auto":
                return None
            raise ValueError(f"No checkpoint found under output_dir={output_dir!r} to resume from.")
        return checkpoint_path

    checkpoint_path = Path(requested)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise ValueError(f"Resume checkpoint does not exist: {checkpoint_path}")
    return str(checkpoint_path)


def build_training_arguments(args: argparse.Namespace, has_eval: bool) -> Any:
    selection_metric = resolve_selection_metric(args)
    should_run_trainer_eval = has_eval
    load_best_model_at_end = has_eval and selection_metric == "eval_loss"
    generated_selection_active = uses_generated_selection(args)
    save_strategy = resolve_save_strategy(args)
    eval_strategy = resolve_eval_strategy(
        args,
        should_run_trainer_eval=should_run_trainer_eval,
        save_strategy=save_strategy,
    )
    kwargs: Dict[str, Any] = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "save_strategy": save_strategy,
        "report_to": "none",
        "save_total_limit": 1,
        "load_best_model_at_end": load_best_model_at_end,
    }
    if save_strategy == "steps":
        kwargs["save_steps"] = resolve_step_interval(args.save_steps, name="--save-steps")

    args_class = SFTConfig if SFTConfig is not None else TrainingArguments
    signature = inspect.signature(args_class.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    if eval_strategy == "steps":
        kwargs["eval_steps"] = resolve_step_interval(
            getattr(args, "eval_steps", None),
            name="--eval-steps",
            fallback=kwargs.get("save_steps"),
        )

    if args_class is not TrainingArguments:
        if "dataset_text_field" in signature.parameters:
            kwargs["dataset_text_field"] = "text"
        if "dataset_kwargs" in signature.parameters:
            kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}
        if "dataset_num_proc" in signature.parameters:
            # Defensive fallback in case TRL still touches the dataset.
            kwargs["dataset_num_proc"] = 1
        if "max_length" in signature.parameters:
            kwargs["max_length"] = args.max_seq_length
        if "packing" in signature.parameters:
            kwargs["packing"] = False
        if "completion_only_loss" in signature.parameters:
            kwargs["completion_only_loss"] = clean_text(args.prompt_format).lower() == "unitor_plain"

    if should_run_trainer_eval:
        if save_strategy != eval_strategy:
            if generated_selection_active:
                raise ValueError(
                    "When generated dev checkpoint tracking is active, "
                    "--save-strategy and --eval-strategy must match so "
                    "generated metrics and eval_loss are measured on the same checkpoints."
                )
            raise ValueError(
                "When --selection-metric eval_loss is active, --save-strategy and "
                "--eval-strategy must match so the best checkpoint can be reloaded."
            )
        if save_strategy == "steps":
            save_steps = int(kwargs["save_steps"])
            eval_steps = int(kwargs["eval_steps"])
            if generated_selection_active and save_steps != eval_steps:
                raise ValueError(
                    "When generated dev checkpoint tracking is active, "
                    "--save-steps must match --eval-steps so every saved checkpoint "
                    "has both generated metrics and eval_loss."
                )
            if save_steps % eval_steps != 0:
                raise ValueError(
                    "When using step-based eval_loss selection, --save-steps must be "
                    "a positive multiple of --eval-steps."
                )
        if selection_metric == "eval_loss":
            kwargs["metric_for_best_model"] = "eval_loss"
            kwargs["greater_is_better"] = False

    return args_class(**kwargs)


def resolve_response_only_parts(
    chat_template: str,
    instruction_part: Optional[str],
    response_part: Optional[str],
) -> Tuple[str, str]:
    if instruction_part and response_part:
        return instruction_part, response_part

    normalized = clean_text(chat_template).lower()

    if normalized == "phi-4":
        return (
            instruction_part or "<|im_start|>user<|im_sep|>",
            response_part or "<|im_start|>assistant<|im_sep|>",
        )

    if normalized in {"phi-3", "phi-35", "phi-3.5"}:
        return (
            instruction_part or "<|user|>\n",
            response_part or "<|assistant|>\n",
        )

    if normalized in {"llama-3", "llama3"}:
        return (
            instruction_part or "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part or "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    if normalized in {"qwen", "qwen2", "qwen-2", "qwen2.5", "qwen-2.5"}:
        return (
            instruction_part or "<|im_start|>user\n",
            response_part or "<|im_start|>assistant\n",
        )

    raise ValueError(
        "Could not infer response-only markers for this chat template. "
        "Please pass both --instruction-part and --response-part explicitly."
    )


def resolve_completion_only_template_ids(tokenizer: Any, args: argparse.Namespace) -> list[int]:
    template = str(getattr(args, "response_template", "") or "")
    template_ids = tokenizer.encode(template, add_special_tokens=False)
    trim_tokens = max(0, int(getattr(args, "response_template_trim_tokens", 0) or 0))
    if trim_tokens:
        template_ids = template_ids[trim_tokens:]
    if not template_ids:
        raise ValueError(
            "The encoded response template is empty after trimming. "
            "Adjust --response-template or --response-template-trim-tokens."
        )
    return template_ids


def build_completion_only_collator(tokenizer: Any, args: argparse.Namespace) -> Any:
    if DataCollatorForCompletionOnlyLM is not None:
        return DataCollatorForCompletionOnlyLM(
            resolve_completion_only_template_ids(tokenizer, args),
            tokenizer=tokenizer,
        )

    if TRLDataCollatorForLanguageModeling is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is required for TRL's language-modeling collator.")
        return TRLDataCollatorForLanguageModeling(
            pad_token_id=pad_token_id,
            completion_only_loss=True,
        )

    raise ImportError(
        "Neither DataCollatorForCompletionOnlyLM nor TRL's DataCollatorForLanguageModeling "
        "is available in this environment."
    )


class CompatSFTTrainer(SFTTrainer):
    """Compatibility shim for TRL/Transformers/Unsloth version mismatches."""

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        inputs["use_cache"] = False
        return Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def build_eval_examples_from_rows(rows: Sequence[Dict[str, Any]]) -> List[EvalExample]:
    examples: List[EvalExample] = []
    for index, row in enumerate(rows):
        question_type = clean_text(row.get("type", "")).lower()
        body = clean_text(row.get("input_1", ""))
        gold_output = clean_text(row.get("output", ""))
        instruction = clean_multiline_text(
            row.get("instruction")
            or QUESTION_INSTRUCTIONS.get(question_type)
        )
        if not question_type or not body or not gold_output or not instruction:
            continue

        question_id = clean_text(row.get("id", "")) or f"prepared-eval:{index}"
        source_path = clean_text(row.get("source_path", "")) or "prepared-eval"
        examples.append(
            EvalExample(
                question_id=question_id,
                question_type=question_type,
                body=body,
                instruction=instruction,
                resources=tuple(list_record_resources(row)),
                gold_output=gold_output,
                source_path=source_path,
                raw_question=None,
            )
        )
    return examples


def build_selection_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_args = argparse.Namespace(**vars(args))
    eval_args.max_seq_length = int(
        getattr(args, "selection_max_seq_length", None)
        or getattr(args, "max_seq_length", 0)
        or 0
    )
    eval_args.max_new_tokens = int(getattr(args, "selection_max_new_tokens", 512) or 512)
    eval_args.num_generations = int(getattr(args, "selection_num_generations", 1) or 1)
    eval_args.aggregation_strategy = str(
        getattr(args, "selection_aggregation_strategy", "union") or "union"
    )
    eval_args.aggregation_min_frequency = int(
        getattr(args, "selection_aggregation_min_frequency", 2) or 2
    )
    eval_args.do_sample = bool(getattr(args, "selection_do_sample", False))
    eval_args.temperature = float(getattr(args, "selection_temperature", 0.7) or 0.7)
    eval_args.top_p = float(getattr(args, "selection_top_p", 0.9) or 0.9)
    eval_args.summary_reference_mode = str(
        getattr(args, "summary_reference_mode", "first") or "first"
    )
    eval_args.use_cache = False
    eval_args.empty_cuda_cache_per_generation = True
    return eval_args


class GeneratedDevSelectionCallback(TrainerCallback):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        eval_rows: Sequence[Dict[str, Any]],
    ) -> None:
        self.args = args
        self.eval_args = build_selection_eval_args(args)
        self.selection_metric = resolve_selection_metric(args)
        self.eval_examples = build_eval_examples_from_rows(eval_rows)
        self.generated_metric_subset_ids: set[str] = set()
        subset_path_value = clean_text(getattr(args, "generated_metric_subset_ids", None))
        if subset_path_value:
            subset_path = Path(subset_path_value)
            if not subset_path.exists():
                raise FileNotFoundError(f"Generated metric subset file does not exist: {subset_path}")
            subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
            if isinstance(subset_payload, dict):
                subset_payload = subset_payload.get("question_ids", [])
            if not isinstance(subset_payload, list):
                raise ValueError("Generated metric subset JSON must be a list or contain a question_ids list.")
            self.generated_metric_subset_ids = {
                clean_text(question_id)
                for question_id in subset_payload
                if clean_text(question_id)
            }
        self.examples_by_key = {
            (clean_text(example.question_id), clean_text(example.question_type).lower()): example
            for example in self.eval_examples
        }
        self.output_root = Path(args.output_dir).resolve().parent / "generated_dev_selection"
        self.best_model_dir = Path(args.save_model_dir)
        self.history_path = self.output_root / "history.json"
        self.best_summary_path = self.output_root / "best_summary.json"
        self.evidence_output_root = Path(args.output_dir).resolve().parent / "evidence_mrr_selection"
        self.best_evidence_model_dir = self.best_model_dir.parent / f"{self.best_model_dir.name}_best_evidence_mrr"
        self.best_evidence_summary_path = self.evidence_output_root / "best_summary.json"
        self.eval_loss_output_root = Path(args.output_dir).resolve().parent / "eval_loss_selection"
        self.eval_loss_best_model_dir = (
            self.best_model_dir.parent / f"{self.best_model_dir.name}_best_eval_loss"
        )
        self.eval_loss_history_path = self.eval_loss_output_root / "history.json"
        self.eval_loss_best_summary_path = self.eval_loss_output_root / "best_summary.json"
        self.summary_by_step: Dict[int, Dict[str, Any]] = {}
        self.eval_loss_summary_by_step: Dict[int, Dict[str, Any]] = {}
        self.best_metric: Optional[float] = None
        self.best_model_path: Optional[str] = None
        self.best_summary: Optional[Dict[str, Any]] = None
        self.best_evidence_mrr: Optional[float] = None
        self.best_evidence_model_path: Optional[str] = None
        self.best_evidence_summary: Optional[Dict[str, Any]] = None
        self.bad_rounds = 0
        self.last_scored_step: Optional[int] = None
        self.best_eval_loss: Optional[float] = None
        self.best_eval_loss_model_path: Optional[str] = None
        self.best_eval_loss_summary: Optional[Dict[str, Any]] = None
        self.eval_loss_bad_rounds = 0
        self.last_eval_loss_step: Optional[int] = None
        self._load_existing_history()

    def _load_history_file(self, path: Path) -> Dict[int, Dict[str, Any]]:
        summaries: Dict[int, Dict[str, Any]] = {}
        if not path.exists():
            return summaries
        try:
            history_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return summaries
        if not isinstance(history_payload, list):
            return summaries

        for entry in history_payload:
            if not isinstance(entry, dict):
                continue
            step_value = entry.get("step")
            if isinstance(step_value, int):
                summaries[step_value] = entry
            elif isinstance(step_value, float) and step_value.is_integer():
                summaries[int(step_value)] = entry
        return summaries

    def _load_summary_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _load_existing_history(self) -> None:
        self.summary_by_step = self._load_history_file(self.history_path)
        best_payload = self._load_summary_file(self.best_summary_path)
        if best_payload is not None:
            self.best_summary = best_payload
            metric_value = best_payload.get("metric_value")
            if isinstance(metric_value, (int, float)):
                self.best_metric = float(metric_value)
                self.best_model_path = str(self.best_model_dir)

        evidence_payload = self._load_summary_file(self.best_evidence_summary_path)
        if evidence_payload is not None:
            self.best_evidence_summary = evidence_payload
            metric_value = evidence_payload.get("generated_subset_mrr")
            if isinstance(metric_value, (int, float)):
                self.best_evidence_mrr = float(metric_value)
                self.best_evidence_model_path = str(self.best_evidence_model_dir)

        self.eval_loss_summary_by_step = self._load_history_file(self.eval_loss_history_path)
        eval_loss_best_payload = self._load_summary_file(self.eval_loss_best_summary_path)
        if eval_loss_best_payload is not None:
            self.best_eval_loss_summary = eval_loss_best_payload
            metric_value = eval_loss_best_payload.get("metric_value")
            if isinstance(metric_value, (int, float)):
                self.best_eval_loss = float(metric_value)
                self.best_eval_loss_model_path = str(self.eval_loss_best_model_dir)

        if self.summary_by_step:
            self.last_scored_step = max(self.summary_by_step)
        if self.eval_loss_summary_by_step:
            self.last_eval_loss_step = max(self.eval_loss_summary_by_step)

    def _cleanup_cuda(self) -> None:
        gc.collect()
        try:
            import torch._dynamo as torch_dynamo

            torch_dynamo.reset()
        except Exception:
            pass
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def _set_inference_mode(self, model: Any) -> None:
        bound_handler = getattr(model, "for_inference", None)
        if callable(bound_handler):
            try:
                bound_handler()
                return
            except Exception:
                pass

        class_handler = getattr(FastLanguageModel, "for_inference", None)
        if callable(class_handler):
            try:
                class_handler(model)
            except Exception:
                pass

    def _set_training_mode(self, model: Any) -> None:
        bound_handler = getattr(model, "for_training", None)
        if callable(bound_handler):
            try:
                bound_handler()
                return
            except Exception:
                pass

        class_handler = getattr(FastLanguageModel, "for_training", None)
        if callable(class_handler):
            try:
                class_handler(model)
            except Exception:
                pass

    def _extract_metric(self, aggregate: Dict[str, Any]) -> Optional[float]:
        if self.selection_metric == "generated_mean_f1":
            return (
                aggregate.get("by_type", {})
                .get("list", {})
                .get("metrics", {})
                .get("mean_f1")
            )
        if self.selection_metric == "generated_primary_score":
            return aggregate.get("overall_average_primary_score")
        if self.selection_metric == "generated_macro_primary_score":
            return aggregate.get("overall_macro_average_primary_score")
        raise ValueError(f"Unsupported generated selection metric: {self.selection_metric}")

    def _use_official_exact_selection(self) -> bool:
        question_types = {
            clean_text(example.question_type).lower()
            for example in self.eval_examples
            if clean_text(example.question_type)
        }
        return bool(question_types) and question_types.issubset(set(OFFICIAL_EXACT_TYPES))

    def _evaluate_model(self, model: Any, tokenizer: Any) -> Dict[str, Any]:
        prediction_rows = []
        was_training = bool(getattr(model, "training", False))
        use_official_exact = self._use_official_exact_selection()
        if not use_official_exact:
            raise ValueError(
                "Generated checkpoint selection requires official BioASQ Phase-B question types "
                "(yesno, factoid, or list); local Python metric scoring has been removed."
            )
        self._set_inference_mode(model)
        if hasattr(model, "eval"):
            model.eval()
        self._cleanup_cuda()

        try:
            for example in self.eval_examples:
                prediction, generation_samples, _generation_telemetry = generate_answer_samples(
                    model=model,
                    tokenizer=tokenizer,
                    example=example,
                    args=self.eval_args,
                    chat_template=clean_text(self.args.chat_template) or None,
                    prompt_format=clean_text(self.args.prompt_format) or "chat",
                )
                prediction_rows.append(
                    {
                        "question_id": example.question_id,
                        "question_type": example.question_type,
                        "body": example.body,
                        "source_path": example.source_path,
                        "prediction": prediction,
                        "generation_samples": generation_samples,
                        "gold_output": example.gold_output,
                    }
                )
        finally:
            self._set_training_mode(model)
            if was_training and hasattr(model, "train"):
                model.train()
            self._cleanup_cuda()

        official_payload = evaluate_with_bioasq_java(
            prediction_rows=prediction_rows,
            examples_by_key=self.examples_by_key,
            model_label="generated-dev-selection",
            model_dir=self.output_root / "official_eval_tmp",
            args=self.eval_args,
        )
        aggregate = official_payload["aggregate"]
        subset_mrr = None
        subset_question_count = 0
        if self.generated_metric_subset_ids:
            # Reuse the full-dev predictions; this adds scoring only, never another decode pass.
            subset_rows = [
                row
                for row in prediction_rows
                if clean_text(row.get("question_id", "")) in self.generated_metric_subset_ids
            ]
            subset_question_count = len(subset_rows)
            subset_keys = {
                (clean_text(row.get("question_id", "")), clean_text(row.get("question_type", "")).lower())
                for row in subset_rows
            }
            subset_examples_by_key = {
                key: example
                for key, example in self.examples_by_key.items()
                if key in subset_keys
            }
            subset_aggregate = evaluate_with_bioasq_java(
                prediction_rows=subset_rows,
                examples_by_key=subset_examples_by_key,
                model_label="generated-dev-selection-subset",
                model_dir=self.output_root / "official_eval_subset_tmp",
                args=self.eval_args,
            )["aggregate"]
            subset_mrr = (
                subset_aggregate.get("by_type", {})
                .get("factoid", {})
                .get("metrics", {})
                .get("mrr")
            )

        metric_value = self._extract_metric(aggregate)
        return {
            "metric_name": self.selection_metric,
            "metric_value": metric_value,
            "aggregate": aggregate,
            "question_count": len(prediction_rows),
            "generated_subset_question_count": subset_question_count,
            "generated_subset_mrr": subset_mrr,
            "scoring_backend": "bioasq_java",
        }

    def _log_history(self, state: Any, summary: Dict[str, Any]) -> None:
        aggregate = summary["aggregate"]
        list_metrics = (
            aggregate.get("by_type", {})
            .get("list", {})
            .get("metrics", {})
        )
        log_entry: Dict[str, Any] = {
            "epoch": state.epoch,
            "step": state.global_step,
            "generated_selection_metric": summary["metric_name"],
            summary["metric_name"]: summary["metric_value"],
            "generated_overall_average_primary_score": aggregate.get("overall_average_primary_score"),
            "generated_overall_macro_average_primary_score": aggregate.get("overall_macro_average_primary_score"),
            "generated_mean_precision": list_metrics.get("mean_precision"),
            "generated_mean_recall": list_metrics.get("mean_recall"),
            "generated_mean_f1": list_metrics.get("mean_f1"),
            "generated_subset_question_count": summary.get("generated_subset_question_count"),
            "generated_subset_mrr": summary.get("generated_subset_mrr"),
        }
        state.log_history.append(log_entry)

    def _persist_history(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.evidence_output_root.mkdir(parents=True, exist_ok=True)
        self.eval_loss_output_root.mkdir(parents=True, exist_ok=True)
        ordered_history = [self.summary_by_step[step] for step in sorted(self.summary_by_step)]
        self.history_path.write_text(
            json.dumps(ordered_history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.best_summary is not None:
            self.best_summary_path.write_text(
                json.dumps(self.best_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if self.best_evidence_summary is not None:
            self.best_evidence_summary_path.write_text(
                json.dumps(self.best_evidence_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        eval_loss_history = [
            self.eval_loss_summary_by_step[step]
            for step in sorted(self.eval_loss_summary_by_step)
        ]
        self.eval_loss_history_path.write_text(
            json.dumps(eval_loss_history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.best_eval_loss_summary is not None:
            self.eval_loss_best_summary_path.write_text(
                json.dumps(self.best_eval_loss_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _save_model_to_dir(self, model: Any, tokenizer: Any, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        save_adapter_and_tokenizer(
            model,
            tokenizer,
            target_dir,
            save_dtype=getattr(self.args, "save_dtype", "float32"),
        )

    def _maybe_stop_early(self, control: Any, *, current_step: int) -> None:
        patience = int(getattr(self.args, "early_stopping_patience", 0) or 0)
        if patience <= 0:
            return
        if self.last_scored_step != current_step or self.last_eval_loss_step != current_step:
            return
        if self.bad_rounds >= patience and self.eval_loss_bad_rounds >= patience:
            control.should_training_stop = True
            print(
                "Stopping early because neither generated dev metrics nor eval_loss "
                f"improved for {patience} aligned checkpoint evaluation(s)."
            )

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        metrics = kwargs.get("metrics")
        if not isinstance(metrics, dict):
            return control

        metric_value = metrics.get("eval_loss")
        if not isinstance(metric_value, (int, float)):
            return control

        current_step = int(state.global_step)
        if current_step == self.last_eval_loss_step:
            return control

        summary = {
            "epoch": state.epoch,
            "metric_name": "eval_loss",
            "metric_value": float(metric_value),
            "step": current_step,
        }
        self.eval_loss_summary_by_step[current_step] = summary
        self.last_eval_loss_step = current_step

        threshold = float(getattr(self.args, "early_stopping_threshold", 0.0) or 0.0)
        improved = (
            self.best_eval_loss is None
            or float(metric_value) < float(self.best_eval_loss) - threshold
        )

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")

        if improved:
            self.best_eval_loss = float(metric_value)
            self.best_eval_loss_model_path = str(self.eval_loss_best_model_dir)
            self.best_eval_loss_summary = {
                **summary,
                "model_dir": str(self.eval_loss_best_model_dir),
            }
            self.eval_loss_bad_rounds = 0
            if model is not None and tokenizer is not None:
                self._save_model_to_dir(model, tokenizer, self.eval_loss_best_model_dir)
                self._cleanup_cuda()
            print(
                f"Validation loss improved to {metric_value:.4f} "
                f"at step {state.global_step} (epoch {state.epoch})."
            )
        else:
            self.eval_loss_bad_rounds += 1
            print(
                "Validation loss did not improve "
                f"(best={self.best_eval_loss}, current={metric_value}); "
                f"patience={self.eval_loss_bad_rounds}/{getattr(self.args, 'early_stopping_patience', 0)}."
            )

        self._persist_history()
        self._maybe_stop_early(control, current_step=current_step)
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self.eval_examples or state.global_step == self.last_scored_step:
            return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            return control

        summary = self._evaluate_model(model=model, tokenizer=tokenizer)
        summary.update(
            {
                "epoch": state.epoch,
                "step": state.global_step,
            }
        )
        self.summary_by_step[int(state.global_step)] = summary
        self.last_scored_step = int(state.global_step)
        self._log_history(state, summary)
        self._persist_history()

        metric_value = summary.get("metric_value")
        improved = (
            isinstance(metric_value, (int, float))
            and (
                self.best_metric is None
                or float(metric_value) > float(self.best_metric) + float(getattr(self.args, "early_stopping_threshold", 0.0) or 0.0)
            )
        )

        evidence_mrr = summary.get("generated_subset_mrr")
        evidence_improved = (
            bool(self.generated_metric_subset_ids)
            and isinstance(evidence_mrr, (int, float))
            and (
                self.best_evidence_mrr is None
                or float(evidence_mrr) > float(self.best_evidence_mrr) + float(getattr(self.args, "early_stopping_threshold", 0.0) or 0.0)
            )
        )

        if improved:
            self.best_metric = float(metric_value)
            self.best_model_path = str(self.best_model_dir)
            self.best_summary = {
                **summary,
                "model_dir": str(self.best_model_dir),
            }
            self.bad_rounds = 0
            self._save_model_to_dir(model, tokenizer, self.best_model_dir)
            self._cleanup_cuda()
            self._persist_history()
            state.best_metric = float(metric_value)
            state.best_model_checkpoint = str(self.best_model_dir)
            print(
                f"Generated dev selection improved to {metric_value:.4f} "
                f"at step {state.global_step} (epoch {state.epoch})."
            )
        if evidence_improved:
            self.best_evidence_mrr = float(evidence_mrr)
            self.best_evidence_model_path = str(self.best_evidence_model_dir)
            self.best_evidence_summary = {
                **summary,
                "model_dir": str(self.best_evidence_model_dir),
            }
            self._save_model_to_dir(model, tokenizer, self.best_evidence_model_dir)
            self._persist_history()
            print(
                f"Evidence-supported generated MRR improved to {evidence_mrr:.4f} "
                f"at step {state.global_step} (epoch {state.epoch})."
            )
        if improved or evidence_improved:
            self.bad_rounds = 0
        else:
            self.bad_rounds += 1
            print(
                "Generated dev selection did not improve "
                f"(best={self.best_metric}, current={metric_value}); "
                f"patience={self.bad_rounds}/{getattr(self.args, 'early_stopping_patience', 0)}."
            )

        self._maybe_stop_early(control, current_step=int(state.global_step))
        self._cleanup_cuda()
        return control

    @property
    def artifact_paths(self) -> Dict[str, str]:
        paths = {}
        if self.history_path.exists():
            paths["generated_dev_history_json"] = str(self.history_path)
        if self.best_summary_path.exists():
            paths["generated_dev_best_summary_json"] = str(self.best_summary_path)
        if self.best_model_dir.exists():
            paths["generated_dev_best_model_dir"] = str(self.best_model_dir)
        if self.best_evidence_summary_path.exists():
            paths["evidence_mrr_best_summary_json"] = str(self.best_evidence_summary_path)
        if self.best_evidence_model_dir.exists():
            paths["evidence_mrr_best_model_dir"] = str(self.best_evidence_model_dir)
        if self.eval_loss_history_path.exists():
            paths["eval_loss_history_json"] = str(self.eval_loss_history_path)
        if self.eval_loss_best_summary_path.exists():
            paths["eval_loss_best_summary_json"] = str(self.eval_loss_best_summary_path)
        if self.eval_loss_best_model_dir.exists():
            paths["eval_loss_best_model_dir"] = str(self.eval_loss_best_model_dir)
        return paths


def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    training_args: TrainingArguments,
    args: argparse.Namespace,
) -> SFTTrainer:
    signature = inspect.signature(SFTTrainer.__init__)
    parameters = signature.parameters

    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    if "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in parameters and SFTConfig is None:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in parameters and SFTConfig is None:
        kwargs["max_seq_length"] = args.max_seq_length
    if "dataset_num_proc" in parameters and SFTConfig is None:
        kwargs["dataset_num_proc"] = args.dataset_num_proc
    if "packing" in parameters and SFTConfig is None:
        kwargs["packing"] = False

    return CompatSFTTrainer(**kwargs)


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    loader_name = getattr(args, "model_loader", "fast_language_model")
    if loader_name == "fast_language_model":
        loader = FastLanguageModel
    elif loader_name == "fast_model":
        from unsloth import FastModel
        loader = FastModel
    else:
        raise ValueError(f"Unsupported model_loader: {loader_name}")
    model, tokenizer = loader.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=resolve_dtype(args.dtype),
        load_in_4bit=not args.no_4bit,
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    tokenizer = text_only_tokenizer(tokenizer)

    peft_options = dict(
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    if loader_name == "fast_model":
        # Ministral 3 is multimodal; this experiment adapts only its language path.
        language_targets = [name for name, _ in model.named_modules()
                            if name.rsplit(".", 1)[-1] in peft_options["target_modules"]
                            and not any(part in name.lower() for part in ("vision", "visual", "projector"))]
        if not language_targets:
            raise ValueError("No language LoRA modules found in the FastModel backbone")
        # FastModel passes this list to get_peft_regex, which interprets entries
        # as leaf names. Fully qualified paths produce an empty selection.
        # The language/vision flags scope these shared projection names.
        peft_options.update(finetune_vision_layers=False, finetune_language_layers=True,
                            finetune_attention_modules=True, finetune_mlp_modules=True)
    model = loader.get_peft_model(model, **peft_options)
    if loader_name == "fast_model":
        trainable_lora = [name for name, parameter in model.named_parameters()
                          if parameter.requires_grad and "lora_" in name]
        if not trainable_lora:
            raise ValueError("FastModel attached no trainable LoRA parameters")
        if any(any(part in name.lower() for part in ("vision", "visual", "projector"))
               for name in trainable_lora):
            raise ValueError("Language-only SFT unexpectedly attached vision/projector LoRA adapters")

    if clean_text(args.prompt_format).lower() == "chat" and not bool(
        getattr(args, "preserve_native_chat_template", False)
    ):
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=args.chat_template,
        )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    eval_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> SFTTrainer:
    has_eval_dataset = eval_dataset is not None and len(eval_dataset) > 0
    has_generated_eval = bool(eval_rows)
    training_args = build_training_arguments(args, has_eval=has_eval_dataset)
    if clean_text(args.prompt_format).lower() == "unitor_plain":
        trainer = build_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=build_completion_only_collator(tokenizer, args),
            training_args=training_args,
            args=args,
        )
    else:
        instruction_part, response_part = resolve_response_only_parts(
            chat_template=args.chat_template,
            instruction_part=args.instruction_part,
            response_part=args.response_part,
        )

        trainer = build_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            training_args=training_args,
            args=args,
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            num_proc=1,
        )

    trainer.generated_metric_callback = None
    if uses_generated_selection(args) and has_generated_eval:
        callback = GeneratedDevSelectionCallback(args=args, eval_rows=eval_rows)
        trainer.add_callback(callback)
        trainer.generated_metric_callback = callback
    elif getattr(args, "early_stopping_patience", 0) > 0 and has_eval_dataset:
        from transformers import EarlyStoppingCallback

        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    return trainer


def train_trainer(trainer: Any, args: argparse.Namespace) -> Any:
    resume_checkpoint = resolve_resume_checkpoint(
        args.output_dir,
        getattr(args, "resume_from_checkpoint", None),
    )
    if resume_checkpoint is not None:
        print(f"Resuming SFT training from checkpoint: {resume_checkpoint}")
        return trainer.train(resume_from_checkpoint=resume_checkpoint)
    return trainer.train()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_training_curves(trainer: Any, save_dir: Path) -> dict[str, str]:
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    save_json(save_dir / "training_log_history.json", log_history)
    if not log_history:
        return {}

    import pandas as pd

    frame = pd.DataFrame(log_history)
    frame.to_csv(save_dir / "training_metrics.csv", index=False)

    saved_paths: dict[str, str] = {
        "log_history_json": str(save_dir / "training_log_history.json"),
        "metrics_csv": str(save_dir / "training_metrics.csv"),
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return saved_paths

    plotted_sections = []
    if "loss" in frame.columns or "eval_loss" in frame.columns:
        plotted_sections.append(
            (
                "Loss",
                [
                    ("loss", "train loss"),
                    ("eval_loss", "eval loss"),
                ],
            )
        )
    if "learning_rate" in frame.columns:
        plotted_sections.append(
            (
                "Learning Rate",
                [
                    ("learning_rate", "learning rate"),
                ],
            )
        )
    if "grad_norm" in frame.columns:
        plotted_sections.append(
            (
                "Gradient Norm",
                [
                    ("grad_norm", "grad norm"),
                ],
            )
        )
    generated_metric_columns = [
        column
        for column in (
            "generated_mean_f1",
            "generated_mean_precision",
            "generated_mean_recall",
            "generated_overall_average_primary_score",
            "generated_overall_macro_average_primary_score",
            "generated_subset_mrr",
        )
        if column in frame.columns
    ]
    if generated_metric_columns:
        plotted_sections.append(
            (
                "Generated Dev Metrics",
                [
                    (
                        column,
                        {
                            "generated_overall_average_primary_score": "all-dev MRR (160)",
                            "generated_overall_macro_average_primary_score": "all-dev macro MRR (160)",
                            "generated_subset_mrr": "evidence-supported MRR (129)",
                        }.get(column, column.replace("_", " ")),
                    )
                    for column in generated_metric_columns
                ],
            )
        )

    if not plotted_sections:
        return saved_paths

    figure, axes = plt.subplots(len(plotted_sections), 1, figsize=(10, 4 * len(plotted_sections)), squeeze=False)
    x_steps = pd.to_numeric(frame.get("step"), errors="coerce") if "step" in frame.columns else None
    fallback_x = pd.Series(range(len(frame)))

    for axis, (title, metric_specs) in zip(axes[:, 0], plotted_sections):
        plotted_any = False
        for metric_key, metric_label in metric_specs:
            if metric_key not in frame.columns:
                continue
            y_values = pd.to_numeric(frame[metric_key], errors="coerce")
            valid_mask = y_values.notna()
            if not valid_mask.any():
                continue
            x_values = x_steps[valid_mask] if x_steps is not None else fallback_x[valid_mask]
            axis.plot(x_values, y_values[valid_mask], marker="o", linewidth=1.5, markersize=3, label=metric_label)
            plotted_any = True

        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        if plotted_any:
            axis.legend()
        else:
            axis.set_visible(False)

    figure.tight_layout()
    figure_path = save_dir / "training_curves.png"
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    saved_paths["training_curves_png"] = str(figure_path)
    return saved_paths
