#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Factoid_SFT/pipeline_runs/${TIMESTAMP}-factoid-single-answer-full-resource-train-eval"

python scripts/run_command_pipeline.py \
  scripts/pipelines/factoid_single_answer_full_resource_train_eval.json \
  --run-dir "${RUN_DIR}" \
  "$@"
