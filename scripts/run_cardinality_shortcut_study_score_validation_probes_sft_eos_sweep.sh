#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

COMMON_ARGS=(
  --probe-input
  Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources/natural_probes_validation.jsonl
  Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources/constructed_minimal_edit_probes_validation.jsonl
  --model-ref Artifacts/Study2/models/original_full_resources/adapter
  --registry-path models/registry.json
  --batch-size 1
  --max-seq-length 4096
  --verbose-every 100
)

python src/utility/cardinality_shortcut_study_score_probe_bank.py \
  "${COMMON_ARGS[@]}" \
  --eos-policy excluded \
  --output-jsonl Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/validation_probe_scores_sft_eos_excluded.jsonl \
  --summary-json Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/validation_probe_scores_sft_eos_excluded_summary.json \
  "$@"

python src/utility/cardinality_shortcut_study_score_probe_bank.py \
  "${COMMON_ARGS[@]}" \
  --eos-policy included \
  --output-jsonl Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/validation_probe_scores_sft_eos_included.jsonl \
  --summary-json Artifacts/Cardinality_Shortcut_Study/probe_scores/original_full_resources_all_resources/validation_probe_scores_sft_eos_included_summary.json \
  "$@"
