from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.model_registry import (
    list_runs,
    load_registry,
    resolve_repo_path,
    short_model_name,
    slugify,
)

from .data import clean_text
from .eval_dataset import render_prompt
from .eval_types import EvalExample, ModelSpec
from .factoid_output_parsing import aggregate_factoid_candidates


def prime_unsloth_runtime() -> None:
    """Import Unsloth early so it can patch Transformers before other model loads."""
    try:
        import unsloth  # noqa: F401
    except Exception:
        return


def _extract_unpadded_token_lengths(tokenized: Mapping[str, Any]) -> List[int]:
    input_ids = tokenized.get("input_ids")
    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, list):
        return []
    if not input_ids:
        return []
    if isinstance(input_ids[0], int):
        return [len(input_ids)]
    return [len(row) for row in input_ids]


def _extract_effective_token_lengths(tokenized: Mapping[str, Any]) -> List[int]:
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        if hasattr(attention_mask, "dim"):
            if attention_mask.dim() == 1:
                return [int(attention_mask.sum().item())]
            return [int(value) for value in attention_mask.sum(dim=-1).tolist()]
        if isinstance(attention_mask, list):
            if not attention_mask:
                return []
            if isinstance(attention_mask[0], int):
                return [int(sum(attention_mask))]
            return [int(sum(row)) for row in attention_mask]

    input_ids = tokenized.get("input_ids")
    if input_ids is None:
        return []
    if hasattr(input_ids, "shape"):
        shape = tuple(input_ids.shape)
        if len(shape) == 1:
            return [int(shape[0])]
        if len(shape) >= 2:
            return [int(shape[-1])] * int(shape[0])
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, list):
        return []
    if not input_ids:
        return []
    if isinstance(input_ids[0], int):
        return [len(input_ids)]
    return [len(row) for row in input_ids]


def tokenize_prompts_for_generation(
    tokenizer: Any,
    prompts: str | Sequence[str],
    *,
    max_seq_length: int,
    encode_kwargs: Mapping[str, Any],
) -> Tuple[Any, List[Dict[str, Any]]]:
    prompt_payload: str | List[str]
    if isinstance(prompts, str):
        prompt_payload = prompts
    else:
        prompt_payload = list(prompts)

    raw_tokenized = tokenizer(
        prompt_payload,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_token_counts = _extract_unpadded_token_lengths(raw_tokenized)

    active_encode_kwargs = dict(encode_kwargs)
    previous_truncation_side = getattr(tokenizer, "truncation_side", None)
    try:
        if max_seq_length > 0:
            active_encode_kwargs.update({"truncation": True, "max_length": max_seq_length})
            if previous_truncation_side is not None:
                tokenizer.truncation_side = "left"
        encoded = tokenizer(prompt_payload, **active_encode_kwargs)
    finally:
        if previous_truncation_side is not None:
            tokenizer.truncation_side = previous_truncation_side

    effective_prompt_token_counts = _extract_effective_token_lengths(encoded)
    if not prompt_token_counts and effective_prompt_token_counts:
        prompt_token_counts = list(effective_prompt_token_counts)

    paired_count = min(len(prompt_token_counts), len(effective_prompt_token_counts))
    prompt_token_counts = prompt_token_counts[:paired_count]
    effective_prompt_token_counts = effective_prompt_token_counts[:paired_count]

    prompt_telemetry: List[Dict[str, Any]] = []
    for original_count, effective_count in zip(prompt_token_counts, effective_prompt_token_counts):
        truncated_token_count = max(0, int(original_count) - int(effective_count))
        prompt_telemetry.append(
            {
                "prompt_token_count": int(original_count),
                "effective_prompt_token_count": int(effective_count),
                "prompt_truncated": bool(truncated_token_count > 0),
                "truncated_token_count": truncated_token_count,
                "max_seq_length": int(max_seq_length) if max_seq_length > 0 else None,
            }
        )

    return encoded, prompt_telemetry


def resolve_torch_dtype(dtype_name: Optional[str]) -> Any:
    if dtype_name in {None, ""}:
        return None

    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _offline_env_enabled() -> bool:
    offline_values = {"1", "true", "yes", "on"}
    return any(
        os.environ.get(name, "").strip().lower() in offline_values
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    )


def _huggingface_hub_cache_dir() -> Path:
    explicit_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_model_snapshot_exists(model_name: str) -> bool:
    normalized_name = str(model_name or "").strip()
    if not normalized_name or "/" not in normalized_name:
        return False

    cache_dir = _huggingface_hub_cache_dir() / f"models--{normalized_name.replace('/', '--')}" / "snapshots"
    return cache_dir.exists() and any(cache_dir.iterdir())


def _adapter_base_model_name(load_target: str) -> Optional[str]:
    adapter_config_path = Path(load_target) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    try:
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    base_model_name = clean_text(payload.get("base_model_name_or_path", ""))
    return base_model_name or None


def _should_use_local_files_only(load_target: str, args: argparse.Namespace) -> bool:
    if bool(getattr(args, "local_files_only", False)):
        return True
    if _offline_env_enabled():
        return True

    if Path(load_target).exists():
        base_model_name = _adapter_base_model_name(load_target)
        return bool(base_model_name and _cached_model_snapshot_exists(base_model_name))

    return _cached_model_snapshot_exists(load_target)


def _should_try_transformers_fallback(exc: Exception, local_files_only: bool) -> bool:
    if local_files_only:
        return True

    error_text = str(exc).lower()
    network_markers = (
        "huggingface seems to be down",
        "timed out",
        "connection",
        "dns",
        "temporary failure",
        "name or service not known",
    )
    return any(marker in error_text for marker in network_markers)


def _repair_unsloth_attention_modules(model: Any) -> None:
    """Restore per-layer helpers expected by Unsloth fast-forward hooks after fallback loads."""
    try:
        from unsloth.models.llama import original_apply_o, original_apply_qkv
    except Exception:
        return

    repaired_qkv = 0
    repaired_o = 0
    for module in model.modules():
        if any(getattr(module, name, None) is None for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue

        forward = getattr(module, "forward", None)
        forward_impl = getattr(forward, "__func__", forward)
        forward_module = str(getattr(forward_impl, "__module__", ""))
        forward_name = str(getattr(forward_impl, "__name__", ""))
        if "unsloth.models.llama" not in forward_module and "fast_forward" not in forward_name:
            continue

        if not hasattr(module, "apply_qkv"):
            module.apply_qkv = original_apply_qkv
            repaired_qkv += 1
        if not hasattr(module, "apply_o"):
            module.apply_o = original_apply_o
            repaired_o += 1

    if repaired_qkv or repaired_o:
        print(
            "Repaired Unsloth attention hooks after Transformers fallback "
            f"(apply_qkv={repaired_qkv}, apply_o={repaired_o})."
        )


def load_model_and_tokenizer_for_eval(model_spec: ModelSpec, args: argparse.Namespace) -> Tuple[Any, Any]:
    load_target = model_spec.load_target
    local_files_only = _should_use_local_files_only(load_target, args)

    try:
        from unsloth import FastLanguageModel
    except Exception:
        FastLanguageModel = None

    if FastLanguageModel is not None:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_target,
                max_seq_length=args.max_seq_length,
                dtype=resolve_torch_dtype(args.dtype),
                load_in_4bit=not args.no_4bit,
                local_files_only=local_files_only,
            )
            if hasattr(model, "for_inference"):
                model.for_inference()
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            return model.eval(), tokenizer
        except Exception as exc:
            if not _should_try_transformers_fallback(exc, local_files_only=local_files_only):
                raise
            print(
                "Unsloth model load failed; falling back to Transformers/PEFT. "
                f"Reason: {type(exc).__name__}: {exc}"
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        load_target,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    adapter_config_path = Path(load_target) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError(
                "This model reference points to a LoRA adapter, but 'peft' is not installed. "
                "Install peft or run the script in the same environment used for training."
            ) from exc

        model = AutoPeftModelForCausalLM.from_pretrained(
            load_target,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_target,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

    _repair_unsloth_attention_modules(model)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def first_model_device(model: Any) -> Any:
    try:
        return model.device
    except Exception:
        pass

    for parameter in model.parameters():
        return parameter.device
    return None


def normalize_generated_item(text: str) -> str:
    normalized = clean_text(text).lower()
    normalized = re.sub(r"\[[A-Z]{2,3}\]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def parse_tagged_items(text: str, begin_tag: str, end_tag: str) -> List[str]:
    pattern = re.compile(re.escape(begin_tag) + r"(.*?)" + re.escape(end_tag), flags=re.DOTALL | re.IGNORECASE)
    return [clean_text(match.group(1)) for match in pattern.finditer(text or "") if clean_text(match.group(1))]


def fallback_split_items(text: str) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [item for item in [clean_text(part) for part in re.split(r"\n|;", cleaned)] if item]


def ordered_unique_items(items: Sequence[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for item in items:
        cleaned = clean_text(item)
        key = normalize_generated_item(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def items_passing_frequency(items_by_sample: Sequence[Sequence[str]], min_frequency: int) -> List[str]:
    representative_by_key: Dict[str, str] = {}
    first_seen_by_key: Dict[str, int] = {}
    frequency: Counter[str] = Counter()

    for sample_index, sample_items in enumerate(items_by_sample):
        sample_keys = set()
        for item in sample_items:
            cleaned = clean_text(item)
            key = normalize_generated_item(cleaned)
            if not cleaned or not key or key in sample_keys:
                continue
            sample_keys.add(key)
            representative_by_key.setdefault(key, cleaned)
            first_seen_by_key.setdefault(key, sample_index)
        frequency.update(sample_keys)

    passing_keys = [
        key
        for key, count in frequency.items()
        if count >= min_frequency
    ]
    passing_keys.sort(key=lambda key: (first_seen_by_key[key], list(representative_by_key).index(key)))
    return [representative_by_key[key] for key in passing_keys]


def aggregate_generated_answers(samples: Sequence[str], question_type: str, args: argparse.Namespace) -> str:
    if not samples:
        return ""
    strategy = str(getattr(args, "aggregation_strategy", "union") or "union")
    min_frequency = max(1, int(getattr(args, "aggregation_min_frequency", 2) or 2))

    if question_type == "list":
        items_by_sample: List[List[str]] = []
        for sample in samples:
            parsed = parse_tagged_items(sample, "[BI]", "[EI]") or fallback_split_items(sample)
            items_by_sample.append(parsed)
        items = (
            items_passing_frequency(items_by_sample, min_frequency=min_frequency)
            if strategy == "frequency"
            else ordered_unique_items([item for sample_items in items_by_sample for item in sample_items])
        )
        if not items and strategy == "frequency":
            items = ordered_unique_items(items_by_sample[0] if items_by_sample else [])
        return " ".join(f"[BI] {item} [EI]" for item in items)

    if question_type == "factoid":
        items = aggregate_factoid_candidates(
            samples,
            strategy=strategy,
            min_frequency=min_frequency,
            max_candidates=max(1, int(getattr(args, "max_factoid_answers", 5) or 5)),
            parser_mode="current",
        )
        return " ".join(f"[BE] {item} [EE]" for item in items)

    if question_type == "yesno":
        votes = []
        for sample in samples:
            normalized = clean_text(sample).lower()
            if normalized.startswith("yes"):
                votes.append("yes")
            elif normalized.startswith("no"):
                votes.append("no")
        if votes:
            yes_votes = votes.count("yes")
            no_votes = votes.count("no")
            return "yes" if yes_votes >= no_votes else "no"
        return clean_text(samples[0])

    return "\n\n".join(ordered_unique_items(samples))


def generate_answer(
    model: Any,
    tokenizer: Any,
    example: EvalExample,
    args: argparse.Namespace,
    chat_template: Optional[str],
    prompt_format: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    import torch

    prompt = render_prompt(
        tokenizer,
        example,
        chat_template=chat_template,
        prompt_format=prompt_format,
    )
    max_seq_length = int(getattr(args, "max_seq_length", 0) or 0)
    encoded, prompt_telemetry_batch = tokenize_prompts_for_generation(
        tokenizer,
        prompt,
        max_seq_length=max_seq_length,
        encode_kwargs={"return_tensors": "pt"},
    )
    prompt_telemetry = prompt_telemetry_batch[0] if prompt_telemetry_batch else {
        "prompt_token_count": int(encoded["input_ids"].shape[-1]),
        "effective_prompt_token_count": int(encoded["input_ids"].shape[-1]),
        "prompt_truncated": False,
        "truncated_token_count": 0,
        "max_seq_length": int(max_seq_length) if max_seq_length > 0 else None,
    }
    device = first_model_device(model)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "do_sample": bool(args.do_sample),
        "use_cache": bool(getattr(args, "use_cache", True)),
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    prompt_length = int(prompt_telemetry.get("effective_prompt_token_count") or encoded["input_ids"].shape[-1])
    generated_ids = output_ids[0][prompt_length:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    cleaned = clean_text(decoded)
    answer = cleaned[7:].strip() if cleaned.lower().startswith("answer:") else cleaned

    del generated_ids
    del output_ids
    del encoded
    gc.collect()

    if bool(getattr(args, "empty_cuda_cache_per_generation", False)) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answer, prompt_telemetry


def generate_answer_samples(
    model: Any,
    tokenizer: Any,
    example: EvalExample,
    args: argparse.Namespace,
    chat_template: Optional[str],
    prompt_format: Optional[str],
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    num_generations = max(1, int(getattr(args, "num_generations", 1) or 1))
    samples: List[str] = []
    prompt_telemetry: List[Dict[str, Any]] = []
    for _ in range(num_generations):
        sample_text, sample_telemetry = generate_answer(
            model=model,
            tokenizer=tokenizer,
            example=example,
            args=args,
            chat_template=chat_template,
            prompt_format=prompt_format,
        )
        samples.append(sample_text)
        prompt_telemetry.append(sample_telemetry)
    if num_generations == 1:
        return samples[0], samples, prompt_telemetry
    return (
        aggregate_generated_answers(samples, question_type=example.question_type, args=args),
        samples,
        prompt_telemetry,
    )


def resolve_model_specs(args: argparse.Namespace, project_root: Path) -> List[ModelSpec]:
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    registry_path = resolve_repo_path(args.registry_path, project_root=project_root)
    if registry_path is None:
        raise ValueError("Could not resolve --registry-path.")

    registry = load_registry(registry_path)
    specs: List[ModelSpec] = []

    def add_run(run: Mapping[str, Any], alias_name: Optional[str] = None) -> None:
        run_id = clean_text(run.get("run_id", ""))
        paths = run.get("paths", {}) if isinstance(run.get("paths"), dict) else {}
        adapter_dir = clean_text(paths.get("adapter_dir", ""))
        load_target = resolve_repo_path(adapter_dir, project_root=project_root) if adapter_dir else None
        load_target_str = str(load_target) if load_target is not None else clean_text(run.get("base_model", ""))
        if not load_target_str:
            raise ValueError(f"Run {run_id or alias_name or '<unknown>'} has no adapter_dir or base_model.")

        chat_template = clean_text(run.get("chat_template", ""))
        if not chat_template and isinstance(run.get("config"), dict):
            chat_template = clean_text(run["config"].get("chat_template", ""))
        prompt_format = clean_text(run.get("prompt_format", ""))
        if not prompt_format and isinstance(run.get("config"), dict):
            prompt_format = clean_text(run["config"].get("prompt_format", ""))

        specs.append(
            ModelSpec(
                ref=alias_name or run_id or load_target_str,
                label=alias_name or run_id or short_model_name(load_target_str),
                source="registry-alias" if alias_name else "registry-run",
                load_target=load_target_str,
                run_id=run_id or None,
                alias=alias_name,
                base_model=clean_text(run.get("base_model", "")) or None,
                adapter_dir=adapter_dir or None,
                chat_template=chat_template or None,
                prompt_format=prompt_format or None,
            )
        )

    if args.all_registry_runs:
        for run in list_runs(registry_path, task="answer_generation", status="completed"):
            add_run(run)

    aliases = registry.get("aliases", {}) if isinstance(registry.get("aliases"), dict) else {}
    runs = registry.get("runs", {}) if isinstance(registry.get("runs"), dict) else {}
    for model_ref in args.model_ref or []:
        normalized_ref = slugify(model_ref, fallback=model_ref)
        if normalized_ref in aliases:
            alias_payload = aliases[normalized_ref]
            run_id = clean_text(alias_payload.get("run_id", "")) if isinstance(alias_payload, dict) else ""
            run = runs.get(run_id)
            if not isinstance(run, dict):
                raise KeyError(f"Alias {model_ref} points to unknown run id: {run_id}")
            add_run(run, alias_name=normalized_ref)
            continue

        if model_ref in runs and isinstance(runs[model_ref], dict):
            add_run(runs[model_ref])
            continue

        resolved_path = resolve_repo_path(model_ref, project_root=project_root)
        model_ref_text = clean_text(model_ref)
        path_candidate = Path(model_ref_text)
        project_candidate = (project_root / path_candidate).resolve() if model_ref_text else None
        first_segment = path_candidate.parts[0] if path_candidate.parts else ""
        first_segment_exists = bool(first_segment) and (project_root / first_segment).exists()
        looks_like_local_path = (
            path_candidate.is_absolute()
            or model_ref_text.startswith(".")
            or "\\" in model_ref_text
            or (
                "/" in model_ref_text
                and (
                    first_segment_exists
                    or (project_candidate is not None and project_candidate.exists())
                )
            )
        )
        if looks_like_local_path and (resolved_path is None or not resolved_path.exists()):
            attempted_path = resolved_path or Path(model_ref_text)
            raise FileNotFoundError(
                "Model reference looks like a local path, but it does not exist: "
                f"{attempted_path}. Check the directory name and whether it uses "
                "underscores or hyphens."
            )
        load_target = str(resolved_path) if resolved_path is not None and resolved_path.exists() else model_ref
        label_source = Path(load_target)
        if label_source.name == "adapter" and label_source.parent.name:
            label = slugify(label_source.parent.name)
        else:
            label = slugify(label_source.name if "/" in load_target or load_target.startswith(".") else load_target)
        specs.append(
            ModelSpec(
                ref=model_ref,
                label=label,
                source="direct",
                load_target=load_target,
                chat_template=clean_text(args.chat_template) or None,
                prompt_format=clean_text(args.prompt_format) or None,
            )
        )

    deduped: List[ModelSpec] = []
    seen = set()
    for spec in specs:
        dedupe_key = (spec.label, spec.load_target, spec.run_id, spec.alias)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(spec)

    if not deduped:
        raise ValueError("No models resolved. Pass --model-ref or use --all-registry-runs.")
    return deduped
