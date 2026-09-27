#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: ./scripts/run_cardinality_shortcut_study_eval_dpo_condition_original_test.sh <condition> [extra evaluate_models.py args...]

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

case "${CONDITION}" in
  ordinary|ordinary_pilot)
    MODEL_SLUG="ordinary_dpo_pilot"
    ;;
  equal_cardinality)
    MODEL_SLUG="equal_cardinality_dpo"
    ;;
  direction_balanced)
    MODEL_SLUG="direction_balanced_dpo"
    ;;
  shorter_skewed)
    MODEL_SLUG="shorter_skewed_dpo"
    ;;
  longer_skewed)
    MODEL_SLUG="longer_skewed_dpo"
    ;;
  *)
    echo "Unsupported condition: ${CONDITION}" >&2
    exit 1
    ;;
esac

MODEL_REF="Artifacts/Cardinality_Shortcut_Study/models/${MODEL_SLUG}/adapter"
OUTPUT_DIR="Artifacts/Cardinality_Shortcut_Study/evaluations/${MODEL_SLUG}_original_test_mr5"

echo "Running MR5 original-test free generation for cardinality shortcut DPO condition"
echo "  condition:  ${CONDITION}"
echo "  model slug: ${MODEL_SLUG}"
echo "  model ref:  ${MODEL_REF}"
echo "  output dir: ${OUTPUT_DIR}"

python src/utility/evaluate_models.py \
  --model-ref "${MODEL_REF}" \
  --eval-input \
    data/Task13BTest/13B1_golden.json \
    data/Task13BTest/13B2_golden.json \
    data/Task13BTest/13B3_golden.json \
    data/Task13BTest/13B4_golden.json \
  --question-types list \
  --max-resources 5 \
  --max-resource-chars 1200 \
  --resource-selection first \
  --resource-granularity document \
  --max-seq-length 4096 \
  --max-new-tokens 512 \
  --num-generations 1 \
  --temperature 0.0 \
  --top-p 1.0 \
  --score-backend both \
  --local-files-only \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
