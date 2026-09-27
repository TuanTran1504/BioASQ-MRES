#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="Artifacts/cse_dpo/class_judgments/candidate_banks_train_rest_after400_c3_c1_v1_gold_c3_gold_supported"
mkdir -p "$LOG_DIR"

python -u -m cse_dpo.annotate_remaining_candidate_bank_questions \
  --archive-incomplete-chunks \
  --overwrite-merged \
  2>&1 | tee "$LOG_DIR/full_annotation.log"
