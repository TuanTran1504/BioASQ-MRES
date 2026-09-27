#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_SLUG="${1:-mr5_resources}"
TARGET="${2:-dev_mr5}"

"${SCRIPT_DIR}/run_factoid_multigeneration_eval.sh" "${MODEL_SLUG}" "${TARGET}" greedy_single \
  --num-generations 1 \
  --temperature 0.0 \
  --top-p 1.0 \
  --aggregation-strategy union

"${SCRIPT_DIR}/run_factoid_multigeneration_eval.sh" "${MODEL_SLUG}" "${TARGET}" sample5_union_t07 \
  --num-generations 5 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --aggregation-strategy union

"${SCRIPT_DIR}/run_factoid_multigeneration_eval.sh" "${MODEL_SLUG}" "${TARGET}" sample5_frequency2_t07 \
  --num-generations 5 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --aggregation-strategy frequency \
  --aggregation-min-frequency 2

"${SCRIPT_DIR}/run_factoid_multigeneration_eval.sh" "${MODEL_SLUG}" "${TARGET}" sample8_frequency2_t07 \
  --num-generations 8 \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --aggregation-strategy frequency \
  --aggregation-min-frequency 2
