#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Factoid_SFT/pipeline_runs/${TIMESTAMP}-factoid-direct-top5-qwen25-05b-train-eval"

python scripts/run_command_pipeline.py \
  scripts/pipelines/factoid_direct_top5_qwen25_05b_train_eval.json \
  --run-dir "${RUN_DIR}" \
  "$@"
