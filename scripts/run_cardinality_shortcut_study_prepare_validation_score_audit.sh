#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SCORED_INPUT="${1:-Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/validation_probe_scores_sft_eos_excluded.jsonl}"
if [[ $# -ge 1 ]]; then
  shift 1
fi

SCORED_STEM="$(basename "${SCORED_INPUT}" .jsonl)"
AUDIT_DIR="Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/manual_audit"

python src/utility/cardinality_shortcut_study_prepare_score_audit.py \
  --scored-input "${SCORED_INPUT}" \
  --output-jsonl "${AUDIT_DIR}/${SCORED_STEM}_manual_audit.jsonl" \
  --output-csv "${AUDIT_DIR}/${SCORED_STEM}_manual_audit.csv" \
  --summary-json "${AUDIT_DIR}/${SCORED_STEM}_manual_audit_summary.json" \
  --per-cell 4 \
  --seed 3407 \
  "$@"
