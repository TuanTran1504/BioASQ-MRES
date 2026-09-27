#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/backfill_factoid_diagnosis_summaries.py \
  --summary-root "Artifacts/Factoid_SFT/candidate_diagnosis/analysis" \
  --summary-root "Artifacts/Factoid_SFT/candidate_diagnosis/reanalysis" \
  "$@"
