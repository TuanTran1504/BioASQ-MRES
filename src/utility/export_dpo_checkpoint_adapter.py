from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_POLICY_ADAPTER_NAME = "default"
DEFAULT_REFERENCE_ADAPTER_NAME = "reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-export a DPO trainer checkpoint into a clean adapter directory. "
            "Useful when the trainer checkpoint is valid but the final exported adapter folder is not."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Path to a DPO trainer checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the exported adapter.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default=None)
    parser.add_argument("--save-dtype", default="auto")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name in {None, ""}:
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    from src.utility.adapter_save import save_adapter_and_tokenizer

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    reference_dir = checkpoint_dir / DEFAULT_REFERENCE_ADAPTER_NAME

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=args.max_seq_length,
        dtype=resolve_dtype(args.dtype),
        load_in_4bit=not args.no_4bit,
        local_files_only=args.local_files_only,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

    if reference_dir.exists():
        if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
            raise TypeError(
                "The loaded checkpoint does not expose PEFT adapter-management APIs, "
                "so the shared DPO reference adapter cannot be restored."
            )
        model.load_adapter(
            str(reference_dir),
            adapter_name=DEFAULT_REFERENCE_ADAPTER_NAME,
            is_trainable=False,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        model.set_adapter(DEFAULT_POLICY_ADAPTER_NAME, inference_mode=False)

    save_adapter_and_tokenizer(
        model,
        tokenizer,
        output_dir,
        save_dtype=args.save_dtype,
    )
    save_json(
        output_dir / "export_manifest.json",
        {
            "status": "completed",
            "source_checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "reference_adapter_restored": reference_dir.exists(),
        },
    )
    print(f"Exported adapter to {output_dir}")


if __name__ == "__main__":
    main()
