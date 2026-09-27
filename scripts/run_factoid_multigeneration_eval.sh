#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_SLUG="${1:-mr5_resources}"
TARGET="${2:-dev_mr5}"
RUN_SLUG="${3:-sample5_frequency_t07}"
shift $(( $# >= 3 ? 3 : $# ))

case "${MODEL_SLUG}" in
  mr5_resources)
    MODEL_REF="Artifacts/Factoid_SFT/models/mr5_resources/adapter"
    ;;
  full_resources)
    MODEL_REF="Artifacts/Factoid_SFT/models/full_resources/adapter"
    ;;
  *)
    echo "Unsupported model slug: ${MODEL_SLUG}" >&2
    echo "Supported model slugs: mr5_resources, full_resources" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="Artifacts/Factoid_SFT/evaluations/multigeneration/${TARGET}/${MODEL_SLUG}/${RUN_SLUG}"
DEFAULT_SAMPLE_FLAGS=(--do-sample)
if [[ "${RUN_SLUG}" == "greedy_single" ]]; then
  DEFAULT_SAMPLE_FLAGS=()
fi

case "${TARGET}" in
  dev_mr5)
    EVAL_INPUT=(data/BioASQ_factoid_sft_prepared/original_mr5/eval_prepared.json)
    SCORE_BACKEND="bioasq_java"
    TARGET_FLAGS=()
    ;;
  dev_full)
    EVAL_INPUT=(data/BioASQ_factoid_sft_prepared/original_full_resources/eval_prepared.json)
    SCORE_BACKEND="bioasq_java"
    TARGET_FLAGS=()
    ;;
  test_mr5)
    EVAL_INPUT=(
      data/Task13BTest/13B1_golden.json
      data/Task13BTest/13B2_golden.json
      data/Task13BTest/13B3_golden.json
      data/Task13BTest/13B4_golden.json
    )
    SCORE_BACKEND="bioasq_java"
    TARGET_FLAGS=(
      --max-resources 5
      --max-resource-chars 1200
      --resource-selection first
      --resource-granularity document
    )
    ;;
  test_full)
    EVAL_INPUT=(
      data/Task13BTest/13B1_golden.json
      data/Task13BTest/13B2_golden.json
      data/Task13BTest/13B3_golden.json
      data/Task13BTest/13B4_golden.json
    )
    SCORE_BACKEND="bioasq_java"
    TARGET_FLAGS=(
      --max-resources 0
      --max-resource-chars 0
      --resource-selection first
      --resource-granularity document
    )
    ;;
  *)
    echo "Unsupported target: ${TARGET}" >&2
    echo "Supported targets: dev_mr5, dev_full, test_mr5, test_full" >&2
    exit 1
    ;;
esac

echo "Running factoid multigeneration evaluation"
echo "  model slug:  ${MODEL_SLUG}"
echo "  target:      ${TARGET}"
echo "  run slug:    ${RUN_SLUG}"
echo "  model ref:   ${MODEL_REF}"
echo "  output dir:  ${OUTPUT_ROOT}"
echo "  note:        factoid aggregation now uses the shared cleaned parser path and caps saved outputs to 5 answers"

python src/utility/evaluate_models.py \
  --model-ref "${MODEL_REF}" \
  --eval-input "${EVAL_INPUT[@]}" \
  --question-types factoid \
  --chat-template llama-3 \
  --prompt-format chat \
  --max-factoid-answers 5 \
  --max-seq-length 4096 \
  --max-new-tokens 64 \
  --num-generations 5 \
  --temperature 0.7 \
  --top-p 0.9 \
  --aggregation-strategy frequency \
  --aggregation-min-frequency 2 \
  --score-backend "${SCORE_BACKEND}" \
  --local-files-only \
  "${DEFAULT_SAMPLE_FLAGS[@]}" \
  "${TARGET_FLAGS[@]}" \
  --output-dir "${OUTPUT_ROOT}" \
  "$@"
