#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/cardinality_shortcut_study_freeze_question_splits.py \
  --train-pool-input data/training13b.json \
  --test-input \
    data/Task13BTest/13B1_golden.json \
    data/Task13BTest/13B2_golden.json \
    data/Task13BTest/13B3_golden.json \
    data/Task13BTest/13B4_golden.json \
  --output-dir data/Cardinality_Shortcut_Study/original_list_question_splits \
  --question-types list \
  --validation-ratio 0.2 \
  "$@"
