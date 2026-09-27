#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/cardinality_shortcut_study_construct_natural_probe_bank.py \
  --input-jsonl data/Cardinality_Shortcut_Study/standardized_candidates/original_full_resources_all_resources/standardized_candidates_validation.jsonl \
  --candidate-bank-jsonl Artifacts/Cardinality_Shortcut_Study/candidate_banks/original_full_resources_all_resources/original-full-resources/candidate_bank.jsonl \
  --output-dir Artifacts/Cardinality_Shortcut_Study/probes/original_full_resources_all_resources \
  --split validation \
  --min-delta-f1 0.05 \
  --min-response-f1 0.0 \
  --max-probes-per-question-per-operation 4 \
  --response-dedupe-mode exact_unordered \
  --seed 3407 \
  "$@"
