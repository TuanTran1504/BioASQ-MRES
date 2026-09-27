#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="Artifacts/Study3/pipeline_runs/${TIMESTAMP}-study3-resource-strategy-sweep"

python scripts/run_command_pipeline.py \
  scripts/pipelines/study3_resource_strategy_sweep.json \
  --run-dir "${RUN_DIR}" \
  "$@"
