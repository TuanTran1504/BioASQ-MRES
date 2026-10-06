#!/usr/bin/env python3
"""Pre-cache an experiment model on a Gadi login node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    credentials.add_argument(
        "--token-file",
        type=Path,
        help="Read the Hugging Face token from a protected local file.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check the configured Hugging Face cache without downloading files.",
    )
    args = parser.parse_args()
    token = args.token
    if args.token_file is not None:
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"Token file is empty: {args.token_file}")
    path = snapshot_download(
        repo_id=args.model,
        token=token,
        local_files_only=args.check_only,
    )
    action = "Found cached" if args.check_only else "Cached"
    print(
        f"{action} {args.model} at {path} "
        f"under HF_HOME={os.environ.get('HF_HOME', '<default>')}"
    )


if __name__ == "__main__":
    main()
