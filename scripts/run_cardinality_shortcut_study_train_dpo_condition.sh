#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: ./scripts/run_cardinality_shortcut_study_train_dpo_condition.sh <condition> [extra dpo_train.py args...]

Supported conditions:
  ordinary
  ordinary_pilot
  equal_cardinality
  direction_balanced
  shorter_skewed
  longer_skewed
EOF
  exit 1
fi

CONDITION="$1"
shift

PAIR_ROOT="Artifacts/Cardinality_Shortcut_Study/pairs/original_full_resources_all_resources"
BASE_MODEL_PATH="Artifacts/Study2/models/original_full_resources/adapter"

case "${CONDITION}" in
  ordinary|ordinary_pilot)
    PAIRS_PATH="${PAIR_ROOT}/ordinary_pairs.jsonl"
    MODEL_SLUG="ordinary_dpo_pilot"
    ;;
  equal_cardinality)
    PAIRS_PATH="${PAIR_ROOT}/equal_cardinality_pairs.jsonl"
    MODEL_SLUG="equal_cardinality_dpo"
    ;;
  direction_balanced)
    PAIRS_PATH="${PAIR_ROOT}/direction_mixture/direction_balanced_pairs.jsonl"
    MODEL_SLUG="direction_balanced_dpo"
    ;;
  shorter_skewed)
    PAIRS_PATH="${PAIR_ROOT}/direction_mixture/shorter_skewed_pairs.jsonl"
    MODEL_SLUG="shorter_skewed_dpo"
    ;;
  longer_skewed)
    PAIRS_PATH="${PAIR_ROOT}/direction_mixture/longer_skewed_pairs.jsonl"
    MODEL_SLUG="longer_skewed_dpo"
    ;;
  *)
    echo "Unsupported condition: ${CONDITION}" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="Artifacts/Cardinality_Shortcut_Study/models/${MODEL_SLUG}"
TRAINER_OUTPUT_DIR="${OUTPUT_ROOT}/trainer_output"
ADAPTER_OUTPUT_DIR="${OUTPUT_ROOT}/adapter"

echo "Training cardinality shortcut DPO condition"
echo "  condition:  ${CONDITION}"
echo "  pair file:  ${PAIRS_PATH}"
echo "  model slug: ${MODEL_SLUG}"
echo "  output dir: ${OUTPUT_ROOT}"

python src/dpo_train.py \
  --preference-input "${PAIRS_PATH}" \
  --model-name "${BASE_MODEL_PATH}" \
  --output-dir "${TRAINER_OUTPUT_DIR}" \
  --save-model-dir "${ADAPTER_OUTPUT_DIR}" \
  --resume-from-checkpoint auto \
  --max-seq-length 2048 \
  --max-prompt-length 1536 \
  --max-completion-length 192 \
  --validation-ratio 0.0 \
  --split-by question \
  --beta 0.1 \
  --learning-rate 5e-6 \
  --num-train-epochs 3.0 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-steps 5 \
  --precompute-ref-log-probs \
  --precompute-ref-batch-size 1 \
  --logging-steps 5 \
  --save-steps 20 \
  --torch-empty-cache-steps 1 \
  --seed 3407 \
  --local-files-only \
  --save-dtype float32 \
  "$@"
