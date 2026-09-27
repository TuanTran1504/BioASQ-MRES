from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unsloth reads this during import, so it must be set before pipeline/training
# modules import FastLanguageModel.
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
# PyTorch reads this during CUDA allocator initialization, so set it before
# the training stack imports torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.utility.config import parse_args


def main() -> None:
    args = parse_args()

    # Import the runtime pipeline after CLI parsing so `--help` stays lightweight
    # and does not eagerly load the full training stack.
    from src.utility.pipeline import run_training

    run_training(args)


if __name__ == "__main__":
    main()
