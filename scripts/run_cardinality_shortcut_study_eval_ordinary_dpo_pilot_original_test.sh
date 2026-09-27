#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

"${SCRIPT_DIR}/run_cardinality_shortcut_study_eval_dpo_condition_original_test.sh" ordinary_pilot "$@"
