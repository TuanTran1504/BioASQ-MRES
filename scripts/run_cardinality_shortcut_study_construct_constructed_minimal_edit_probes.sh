#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/cardinality_shortcut_study_construct_constructed_minimal_edit_probes.py \
  --question-input data/training13b.json \
  --candidate-bank-jsonl Artifacts/Cardinality_Shortcut_Study/candidate_banks/original_full_resources_all_resources/original-full-resources/candidate_bank.jsonl \
  --standardized-input-jsonl data/Cardinality_Shortcut_Study/standardized_candidates/original_full_resources_all_resources/standardized_candidates_validation.jsonl \
  --output-dir Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources \
  --split validation \
  --dataset-name bioasq \
  --max-resources 0 \
  --max-resource-chars 0 \
  --gold-support-policy all \
  --min-delta-f1 0.05 \
  --max-probes-per-question-per-operation 4 \
  --seed 3407 \
  "$@"
