#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_SLUG="${1:-sft_full_single_answer}"
TARGET="${2:-dev_full}"
SEED="${3:-3407}"
shift $(( $# >= 3 ? 3 : $# ))

PARSER_MODE="${FACTOID_CANDIDATE_DIAGNOSIS_PARSER_MODE:-agnostic}"
STRATEGY_LIST="${FACTOID_CANDIDATE_DIAGNOSIS_STRATEGIES:-s1_single_greedy s2_top5_greedy s3_single_sample_n20}"
read -r -a STRATEGIES <<<"${STRATEGY_LIST}"

echo "Running factoid candidate-diagnosis suite"
echo "  model slug:  ${MODEL_SLUG}"
echo "  target:      ${TARGET}"
echo "  seed:        ${SEED}"
echo "  parser mode: ${PARSER_MODE}"
echo "  strategies:  ${STRATEGIES[*]}"

for strategy in "${STRATEGIES[@]}"; do
  echo
  echo "=== ${MODEL_SLUG} :: ${TARGET} :: ${strategy} ==="
  bash "${SCRIPT_DIR}/run_factoid_candidate_diagnosis_eval.sh" \
    "${MODEL_SLUG}" \
    "${TARGET}" \
    "${strategy}" \
    "${SEED}" \
    "$@"
  bash "${SCRIPT_DIR}/run_factoid_candidate_diagnosis_analyze.sh" \
    "${MODEL_SLUG}" \
    "${TARGET}" \
    "${strategy}" \
    "$@"
  bash "${SCRIPT_DIR}/run_factoid_candidate_diagnosis_reanalyze_protocols.sh" \
    "${MODEL_SLUG}" \
    "${TARGET}" \
    "${strategy}" \
    "${PARSER_MODE}" \
    "$@"
done
