#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE="${FACTOID_CANDIDATE_DIAGNOSIS_PROFILE:-aligned_single_answer}"
TARGET="${1:-}"
SEED="${2:-3407}"
shift $(( $# >= 2 ? 2 : $# ))

MODELS=()
STRATEGIES=()

case "${PROFILE}" in
  aligned_single_answer)
    TARGET="${TARGET:-dev_full}"
    MODELS=(base sft_full_single_answer)
    STRATEGIES=(
      s1_single_greedy
      s3_single_sample_n1
      s3_single_sample_n5
      s3_single_sample_n10
      s3_single_sample_n20
    )
    ;;
  prompt_mismatch_ablation)
    TARGET="${TARGET:-dev_mr5}"
    MODELS=(base sft_mr5)
    STRATEGIES=(
      s1_single_greedy
      s2_top5_greedy
      s3_single_sample_n1
      s3_single_sample_n5
      s3_single_sample_n10
      s3_single_sample_n20
    )
    ;;
  *)
    echo "Unsupported FACTOID_CANDIDATE_DIAGNOSIS_PROFILE: ${PROFILE}" >&2
    echo "Supported profiles: aligned_single_answer, prompt_mismatch_ablation" >&2
    exit 1
    ;;
esac

echo "Running factoid candidate-diagnosis validation suite"
echo "  profile: ${PROFILE}"
echo "  target: ${TARGET}"
echo "  seed:   ${SEED}"
echo "  models: ${MODELS[*]}"

for model_slug in "${MODELS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    echo
    echo "=== ${model_slug} :: ${strategy} ==="
    bash "${SCRIPT_DIR}/run_factoid_candidate_diagnosis_eval.sh" \
      "${model_slug}" \
      "${TARGET}" \
      "${strategy}" \
      "${SEED}" \
      "$@"
    bash "${SCRIPT_DIR}/run_factoid_candidate_diagnosis_analyze.sh" \
      "${model_slug}" \
      "${TARGET}" \
      "${strategy}" \
      "$@"
  done
done
