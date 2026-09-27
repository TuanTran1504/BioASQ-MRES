#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FACTOID_CANDIDATE_DIAGNOSIS_PROFILE="prompt_mismatch_ablation"
export FACTOID_CANDIDATE_DIAGNOSIS_ROOT="${FACTOID_CANDIDATE_DIAGNOSIS_ROOT:-Abalations/Factoid_Prompt_Mismatch/candidate_diagnosis}"
export FACTOID_CANDIDATE_DIAGNOSIS_PROMPT_FILE="${FACTOID_CANDIDATE_DIAGNOSIS_PROMPT_FILE:-prompts/factoid_candidate_diagnosis.json}"

"${SCRIPT_DIR}/run_factoid_candidate_diagnosis_eval.sh" "$@"
