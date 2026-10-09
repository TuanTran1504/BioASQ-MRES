#!/usr/bin/env python3
"""Train one paired 8B condition from an explicitly pinned cached snapshot."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from prepare_matched_8b_sft import DEFAULT_CONFIG, ROOT, digest, read, resolved_config, validate
from run_expansion_sft_qwen3 import train, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama31", "qwen3", "ministral3"), required=True)
    parser.add_argument("--formulation", choices=("original", "expansion"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--execution-mode", choices=("default", "eager"), default="default")
    args = parser.parse_args()
    if args.execution_mode == "eager" and args.model != "ministral3":
        raise ValueError("Eager recovery is restricted to Ministral")
    matrix = read(args.config)
    validate(matrix)
    pins = read(args.pins)
    if pins["matrix_sha256"] != digest(args.config):
        raise ValueError("Training matrix changed after base snapshots were pinned")
    pin = pins["models"][args.model]
    if pin["model_name"] != matrix["models"][args.model]["model_name"]:
        raise ValueError("Pinned backbone differs from the requested model")
    snapshot = Path(pin["snapshot"])
    if snapshot.name != pin["revision"] or not (snapshot / "config.json").is_file():
        raise ValueError("Pinned base snapshot is unavailable locally")
    config = resolved_config(matrix, args.model, args.formulation)
    config.update(base_revision=pin["revision"], base_snapshot=str(snapshot),
                  matrix_sha256=pins["matrix_sha256"], checkpoint_selection="internal validation loss",
                  execution_mode=args.execution_mode)
    train_rows, eval_rows, validation = validate_inputs(config)
    output_root = "outputs/matched_8b_sft"
    if (ROOT / output_root / args.run_name).exists():
        raise FileExistsError("Choose a new run name; matched experiments never reuse an output directory")
    training_args = SimpleNamespace(
        run_name=args.run_name, output_root=output_root, model_name=str(snapshot),
        allow_download=False, smoke_test=args.smoke_test, max_train_samples=None,
        max_eval_samples=None, resume_from_checkpoint="")
    from run_extractive_expansion_8b import configure_job_local_compiler_cache
    configure_job_local_compiler_cache()
    train(training_args, config, train_rows, eval_rows, validation)


if __name__ == "__main__":
    main()
