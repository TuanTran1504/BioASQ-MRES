#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -gt 0 ]]; then
  CONDITIONS=("$@")
else
  CONDITIONS=(
    equal_cardinality
    direction_balanced
    shorter_skewed
    longer_skewed
  )
fi

for condition in "${CONDITIONS[@]}"; do
  echo
  echo "============================================================"
  echo "Running MR5 original-test eval for condition: ${condition}"
  echo "============================================================"
  "${SCRIPT_DIR}/run_cardinality_shortcut_study_eval_dpo_condition_original_test.sh" "${condition}"
done
