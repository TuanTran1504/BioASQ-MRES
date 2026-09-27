#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE="${FACTOID_CANDIDATE_DIAGNOSIS_PROFILE:-aligned_single_answer}"
ARTIFACT_ROOT="${FACTOID_CANDIDATE_DIAGNOSIS_ROOT:-Artifacts/Factoid_SFT/candidate_diagnosis}"

TARGET="${1:-}"
STRATEGY="${2:-}"
SAMPLE_SOURCE="${3:-prediction}"
shift $(( $# >= 3 ? 3 : $# ))

case "${PROFILE}" in
  aligned_single_answer)
    TARGET="${TARGET:-dev_full}"
    STRATEGY="${STRATEGY:-s1_single_greedy}"
    SFT_SLUG="sft_full_single_answer"
    ;;
  prompt_mismatch_ablation)
    TARGET="${TARGET:-dev_mr5}"
    STRATEGY="${STRATEGY:-s2_top5_greedy}"
    SFT_SLUG="sft_mr5"
    ;;
  *)
    echo "Unsupported FACTOID_CANDIDATE_DIAGNOSIS_PROFILE: ${PROFILE}" >&2
    echo "Supported profiles: aligned_single_answer, prompt_mismatch_ablation" >&2
    exit 1
    ;;
esac

case "${TARGET}" in
  dev_mr5|dev_full)
    ;;
  *)
    echo "Unsupported target: ${TARGET}" >&2
    echo "Supported targets: dev_mr5, dev_full" >&2
    exit 1
    ;;
esac

BASE_ROOT="${ARTIFACT_ROOT}/generations/${TARGET}/base/${STRATEGY}"
SFT_ROOT="${ARTIFACT_ROOT}/generations/${TARGET}/${SFT_SLUG}/${STRATEGY}"
BASE_PREDICTIONS="$(find "${BASE_ROOT}" -name predictions.json -print | head -n 1)"
SFT_PREDICTIONS="$(find "${SFT_ROOT}" -name predictions.json -print | head -n 1)"

if [[ -z "${BASE_PREDICTIONS}" || -z "${SFT_PREDICTIONS}" ]]; then
  echo "Could not find both base and ${SFT_SLUG} prediction files for ${TARGET}/${STRATEGY}" >&2
  exit 1
fi

OUTPUT_ROOT="${ARTIFACT_ROOT}/parser_audits/${TARGET}/${STRATEGY}/${SAMPLE_SOURCE}"
EXTRA_FLAGS=()
if [[ "${SAMPLE_SOURCE}" == "all_samples" ]]; then
  EXTRA_FLAGS+=(--include-generation-samples)
elif [[ "${SAMPLE_SOURCE}" != "prediction" ]]; then
  echo "Unsupported sample source: ${SAMPLE_SOURCE}" >&2
  echo "Supported values: prediction, all_samples" >&2
  exit 1
fi

echo "Running factoid parser audit"
echo "  profile:       ${PROFILE}"
echo "  target:        ${TARGET}"
echo "  strategy:      ${STRATEGY}"
echo "  sample source: ${SAMPLE_SOURCE}"
echo "  output dir:    ${OUTPUT_ROOT}"

python src/utility/factoid_parser_audit.py \
  --prediction "base=${BASE_PREDICTIONS}" \
  --prediction "${SFT_SLUG}=${SFT_PREDICTIONS}" \
  --output-dir "${OUTPUT_ROOT}" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
