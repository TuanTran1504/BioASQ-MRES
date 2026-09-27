#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: ./scripts/run_cardinality_shortcut_study_score_validation_probes_dpo_condition.sh <condition> [extra scorer args...]

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

OUTPUT_ROOT="Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources"
OUTPUT_JSONL="${OUTPUT_ROOT}/validation_probe_scores_${MODEL_SLUG}.jsonl"
SUMMARY_JSON="${OUTPUT_ROOT}/validation_probe_scores_${MODEL_SLUG}_summary.json"
MODEL_REF="Artifacts/Cardinality_Shortcut_Study/models/${MODEL_SLUG}/adapter"

echo "Scoring validation probes for cardinality shortcut DPO condition"
echo "  condition:    ${CONDITION}"
echo "  model slug:   ${MODEL_SLUG}"
echo "  model ref:    ${MODEL_REF}"
echo "  output jsonl: ${OUTPUT_JSONL}"

python src/utility/cardinality_shortcut_study_score_probe_bank.py \
  --probe-input \
    Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources/natural_probes_validation.jsonl \
    Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources/constructed_minimal_edit_probes_validation.jsonl \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --model-ref "${MODEL_REF}" \
  --registry-path models/registry.json \
  --batch-size 1 \
  --max-seq-length 4096 \
  --eos-policy excluded \
  --verbose-every 100 \
  "$@"
