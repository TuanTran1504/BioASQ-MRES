#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Study5/pipeline_runs/${TIMESTAMP}-study5-full-gold-dpo-train-eval"

python scripts/run_command_pipeline.py \
  scripts/pipelines/study5_full_gold_dpo_train_eval.json \
  --run-dir "${RUN_DIR}" \
  "$@"
