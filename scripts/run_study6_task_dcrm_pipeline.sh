#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Study6/pipeline_runs/${TIMESTAMP}-study6-task-dcrm-pair-generation"

python scripts/run_command_pipeline.py \
  scripts/pipelines/study6_task_dcrm_pair_generation.json \
  --run-dir "${RUN_DIR}" \
  "$@"
