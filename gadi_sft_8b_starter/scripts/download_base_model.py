#!/usr/bin/env python3
"""Pre-cache the gated 8B base model on a Gadi login node."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    snapshot_download(repo_id=args.model, token=args.token)
    print(f"Cached {args.model} under HF_HOME={os.environ.get('HF_HOME', '<default>')}")


if __name__ == "__main__":
    main()
