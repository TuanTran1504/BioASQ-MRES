#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="data/BioASQ_factoid_sft_prepared/direct_top5_hard_negative_full_resources"
REUSE_AUDIT_PATH="${OUTPUT_DIR}/question_audit.jsonl"
REUSE_AUDIT_FLAGS=()
if [[ -f "${REUSE_AUDIT_PATH}" ]]; then
  echo "Reusing saved sampled outputs from ${REUSE_AUDIT_PATH}"
  REUSE_AUDIT_FLAGS=(--reuse-question-audit "${REUSE_AUDIT_PATH}")
fi

python src/utility/build_factoid_direct_top5_sft_data.py \
  --train-input data/training13b.json \
  --output-dir "${OUTPUT_DIR}" \
  --model-ref unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --sample-prompt-file prompts/factoid_single_answer_aligned.json \
  --sample-prompt factoid-single-answer-aligned-v1 \
  --target-prompt-file prompts/factoid_single_answer_aligned.json \
  --target-prompt factoid-top-five-eval-v1 \
  --num-generations 20 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 64 \
  --max-resources 0 \
  --max-resource-chars 0 \
  --resource-granularity document \
  --single-answer-policy strict \
  --factoid-parser-mode agnostic \
  --negative-count 4 \
  --negative-fill-strategy leave_short \
  --validation-ratio 0.1 \
  --seed 3407 \
  --max-seq-length 4096 \
  --batch-size 1 \
  --local-files-only \
  "${REUSE_AUDIT_FLAGS[@]}" \
  "$@"
