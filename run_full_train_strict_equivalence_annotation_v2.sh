#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ROOT="Artifacts/cse_dpo/class_judgments/candidate_banks_train_full_fresh_c3_c1_v2_strict_equivalence_gold_c3_gold_supported_1130"
ELIGIBLE_SOURCE="data/BioASQ_factoid_sft_prepared/evidence_grounded_single_answer_qwen25_05b/train_prepared.json"

mkdir -p "$RUN_ROOT"

EXTRA_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

python -u -m cse_dpo.annotate_remaining_candidate_bank_questions \
  --fresh-full-run \
  --eligible-question-source "$ELIGIBLE_SOURCE" \
  --rest-root "$RUN_ROOT/chunks" \
  --merged-root "$RUN_ROOT/merged" \
  --chunk-size 25 \
  --filter-gold-supported \
  --require-c3-extractive \
  --gold-c3-policy always_first_extractive_alias \
  --archive-incomplete-chunks \
  --overwrite-merged \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_ROOT/full_strict_equivalence_annotation.log"
