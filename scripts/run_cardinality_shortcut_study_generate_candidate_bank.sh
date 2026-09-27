#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m cse_dpo.generate_candidate_bank \
  --eval-input data/training13b.json \
  --model-ref Artifacts/Study2/models/original_full_resources/adapter \
  --output-dir Artifacts/Cardinality_Shortcut_Study/candidate_banks/original_full_resources_all_resources \
  --dataset-name cardinality-shortcut-original-full-resources-all-resources \
  --question-types list \
  --chat-template llama-3 \
  --prompt-format chat \
  --samples-per-question-total 12 \
  --max-resources 0 \
  --max-resource-chars 0 \
  --resource-selection first \
  --resource-granularity snippet \
  --max-seq-length 131072 \
  --max-new-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.9 \
  --batch-size 1 \
  --local-files-only \
  "$@"
