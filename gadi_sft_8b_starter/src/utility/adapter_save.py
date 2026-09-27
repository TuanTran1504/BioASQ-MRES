from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from peft import get_peft_model_state_dict
except ImportError:  # pragma: no cover - depends on local env
    get_peft_model_state_dict = None

try:
    from safetensors.torch import save_file as safe_save_file
except ImportError:  # pragma: no cover - depends on local env
    safe_save_file = None


ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"


def resolve_save_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    normalized = str(dtype_name or "").strip().lower()
    if normalized in {"", "auto", "same"}:
        return None
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {dtype_name}")


def _convert_tensor_for_save(
    value: torch.Tensor,
    *,
    target_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    tensor = value.detach().cpu()
    if target_dtype is not None and torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=target_dtype)
    return tensor


def build_adapter_state_dict_for_save(
    model: Any,
    *,
    adapter_name: str = "default",
    save_dtype: Optional[str],
) -> Optional[Dict[str, Any]]:
    target_dtype = resolve_save_dtype(save_dtype)

    if get_peft_model_state_dict is None:
        raise ImportError(
            "Saving adapters with an explicit dtype requires `peft` to expose "
            "`get_peft_model_state_dict` in this environment."
        )

    state_dict = get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
        save_embedding_layers="auto",
    )
    converted: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            converted[key] = _convert_tensor_for_save(value, target_dtype=target_dtype)
        else:
            converted[key] = value
    return converted


def validate_adapter_state_dict_for_save(
    state_dict: Optional[Dict[str, Any]],
    *,
    save_dtype: Optional[str],
) -> None:
    if state_dict is None:
        return

    if len(state_dict) == 0:
        raise RuntimeError(
            "Refusing to save an empty PEFT adapter state_dict. "
            "This usually means adapter extraction failed, which would otherwise "
            "produce a tiny unusable adapter_model.safetensors file. "
            f"Requested save_dtype={save_dtype!r}."
        )


def _adapter_output_dir(target_dir: Path, adapter_name: str) -> Path:
    return target_dir if adapter_name == "default" else target_dir / adapter_name


def _ensure_peft_save_dependencies() -> None:
    if get_peft_model_state_dict is None:
        raise ImportError(
            "Saving PEFT adapters requires `peft.get_peft_model_state_dict` in this environment."
        )
    if safe_save_file is None:
        raise ImportError(
            "Saving PEFT adapters requires `safetensors` in this environment."
        )


def _adapter_auto_mapping(model: Any, peft_config: Any) -> Optional[Dict[str, str]]:
    if getattr(peft_config, "task_type", None) is not None:
        return None
    if not hasattr(model, "_get_base_model_class"):
        return None

    base_model_class = model._get_base_model_class(
        is_prompt_tuning=getattr(peft_config, "is_prompt_learning", False),
    )
    return {
        "base_model_class": base_model_class.__name__,
        "parent_library": base_model_class.__module__,
    }


def _save_adapter_config(model: Any, adapter_name: str, output_dir: Path) -> None:
    peft_config = model.peft_config[adapter_name]
    if peft_config.base_model_name_or_path is None:
        peft_config.base_model_name_or_path = (
            model.base_model.__dict__.get("name_or_path", None)
            if getattr(peft_config, "is_prompt_learning", False)
            else model.base_model.model.__dict__.get("name_or_path", None)
        )

    inference_mode = peft_config.inference_mode
    peft_config.inference_mode = True
    try:
        peft_config.save_pretrained(str(output_dir), auto_mapping_dict=_adapter_auto_mapping(model, peft_config))
    finally:
        peft_config.inference_mode = inference_mode


def save_peft_adapters(
    model: Any,
    save_directory: str | Path,
    *,
    save_dtype: Optional[str],
) -> None:
    _ensure_peft_save_dependencies()

    target_dir = Path(save_directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "create_or_update_model_card"):
        model.create_or_update_model_card(str(target_dir))

    adapter_names = list(getattr(model, "peft_config", {}).keys())
    if not adapter_names:
        raise RuntimeError("Expected a PEFT model with at least one loaded adapter, but found none.")

    for adapter_name in adapter_names:
        adapter_state_dict = build_adapter_state_dict_for_save(
            model,
            adapter_name=adapter_name,
            save_dtype=save_dtype,
        )
        validate_adapter_state_dict_for_save(adapter_state_dict, save_dtype=save_dtype)

        output_dir = _adapter_output_dir(target_dir, adapter_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        contiguous_state_dict = {
            key: value.contiguous() if isinstance(value, torch.Tensor) and not value.is_contiguous() else value
            for key, value in adapter_state_dict.items()
        }
        safe_save_file(
            contiguous_state_dict,
            str(output_dir / ADAPTER_SAFE_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )
        _save_adapter_config(model, adapter_name, output_dir)


def save_adapter_and_tokenizer(
    model: Any,
    tokenizer: Any,
    save_directory: str | Path,
    *,
    save_dtype: Optional[str],
) -> None:
    target_dir = Path(save_directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    if getattr(model, "peft_config", None):
        save_peft_adapters(
            model,
            target_dir,
            save_dtype=save_dtype,
        )
    else:
        model.save_pretrained(str(target_dir))
    tokenizer.save_pretrained(str(target_dir))
