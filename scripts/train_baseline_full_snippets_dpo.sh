#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  scripts/train_baseline_full_snippets_dpo.sh <preference_jsonl> [run_name] [extra_dpo_train_args...]

Example:
  scripts/train_baseline_full_snippets_dpo.sh \
    Artifacts/cse_dpo/pairs/train_8b_unitutor_whole_response_recall_balanced_dpo.jsonl \
    bioasq-8b-full-snippets-dpo-recall-balanced

Notes:
  - This continues training from the adapter:
      Artifacts/models/runs/20260803-100510-bioasq-8b-baseline-full-snippets/adapter
  - For the cleanest experiment, prefer DPO pairs generated from the same
    full-snippets prompt/evidence regime as this SFT model.
EOF
  exit 1
fi

PREFERENCE_INPUT="$1"
PAIR_STEM="$(basename "${PREFERENCE_INPUT}" .jsonl)"
RUN_NAME="${2:-bioasq-8b-baseline-full-snippets-dpo-${PAIR_STEM}}"

if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

MODEL_NAME="Artifacts/models/runs/20260803-100510-bioasq-8b-baseline-full-snippets/adapter"
OUTPUT_DIR="Artifacts/models/dpo_runs/${RUN_NAME}/trainer_output"
SAVE_MODEL_DIR="Artifacts/models/dpo_runs/${RUN_NAME}/adapter"

python src/dpo_train.py \
  --preference-input "${PREFERENCE_INPUT}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --save-model-dir "${SAVE_MODEL_DIR}" \
  --max-seq-length 2048 \
  --max-prompt-length 1536 \
  --validation-ratio 0.20 \
  --split-by question \
  --beta 0.1 \
  --learning-rate 5e-6 \
  --num-train-epochs 1.0 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --warmup-steps 5 \
  --logging-steps 5 \
  --save-steps 100 \
  --local-files-only \
  --save-dtype float32 \
  "$@"
