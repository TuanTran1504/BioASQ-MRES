#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/cardinality_shortcut_study_construct_direction_mixture_pairs.py \
  --input-jsonl data/Cardinality_Shortcut_Study/standardized_candidates/original_full_resources_all_resources/standardized_candidates_train.jsonl \
  --output-dir Artifacts/Cardinality_Shortcut_Study/pairs/original_full_resources_all_resources \
  --split train \
  --min-delta-f1 0.05 \
  --min-response-f1 0.0 \
  --max-pairs-per-question 8 \
  --max-entity-gap 8 \
  --max-entity-ratio 2.5 \
  --response-dedupe-mode exact_unordered \
  --seed 3407 \
  "$@"
