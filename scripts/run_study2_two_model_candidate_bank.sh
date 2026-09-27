#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Study2/pipeline_runs/${TIMESTAMP}-study2-two-model-candidate-bank"

python scripts/run_command_pipeline.py \
  scripts/pipelines/study2_two_model_candidate_bank.json \
  --run-dir "${RUN_DIR}" \
  "$@"
