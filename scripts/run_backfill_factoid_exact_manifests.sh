#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python src/utility/backfill_official_exact_generation_manifests.py \
  --manifest-root "Artifacts/Factoid_SFT/candidate_diagnosis/generations" \
  --manifest-root "Artifacts/Factoid_SFT/evaluations" \
  "$@"
