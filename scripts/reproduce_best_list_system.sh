#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/reproductions/best_list_system/pipeline_runs/${TIMESTAMP}"

python scripts/run_command_pipeline.py \
  scripts/pipelines/reproduce_best_list_system.json \
  --run-dir "${RUN_DIR}" \
  "$@"
