#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/cardinality_shortcut_study_standardize_candidates.py \
  --question-input data/training13b.json \
  --candidate-input Artifacts/Cardinality_Shortcut_Study/candidate_banks/original_full_resources_all_resources/original-full-resources/candidate_bank.jsonl \
  --split-manifest data/Cardinality_Shortcut_Study/original_list_question_splits/question_split_manifest.csv \
  --output-dir data/Cardinality_Shortcut_Study/standardized_candidates/original_full_resources_all_resources \
  --dataset-name bioasq \
  --max-resources 0 \
  --max-resource-chars 0 \
  --gold-support-policy all \
  "$@"
