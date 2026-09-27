#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Study4/pipeline_runs/${TIMESTAMP}-study4-original-first5-cardinality-dpo-train-eval"

python scripts/run_command_pipeline.py \
  scripts/pipelines/study4_original_first5_cardinality_dpo_train_eval.json \
  --run-dir "${RUN_DIR}" \
  "$@"
