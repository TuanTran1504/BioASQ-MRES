#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PROFILE="${FACTOID_CANDIDATE_DIAGNOSIS_PROFILE:-aligned_single_answer}"
ARTIFACT_ROOT="${FACTOID_CANDIDATE_DIAGNOSIS_ROOT:-Artifacts/Factoid_SFT/candidate_diagnosis}"
PROMPT_FILE="${FACTOID_CANDIDATE_DIAGNOSIS_PROMPT_FILE:-prompts/factoid_single_answer_aligned.json}"
BASE_MODEL_REF="${FACTOID_BASE_MODEL_REF:-unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit}"

MODEL_SLUG="${1:-}"
TARGET="${2:-}"
STRATEGY="${3:-}"
PARSER_MODE="${4:-agnostic}"
shift $(( $# >= 4 ? 4 : $# ))

PROMPT_REF=""
CANDIDATE_MODE=""
POLICIES=()

case "${PROFILE}" in
  aligned_single_answer)
    MODEL_SLUG="${MODEL_SLUG:-sft_full_single_answer}"
    TARGET="${TARGET:-dev_full}"
    STRATEGY="${STRATEGY:-s3_single_sample_n20}"
    case "${MODEL_SLUG}" in
      base)
        MODEL_REF="${BASE_MODEL_REF}"
        ;;
      sft_full_single_answer)
        MODEL_REF="Artifacts/Factoid_SFT/models/full_resources_single_answer/adapter"
        ;;
      sft_full_single_answer_per_alias)
        MODEL_REF="Artifacts/Factoid_SFT/models/full_resources_single_answer_per_alias/adapter"
        ;;
      sft_direct_top5)
        MODEL_REF="Artifacts/Factoid_SFT/models/direct_top5_full_resources_gold_eval/adapter"
        ;;
      sft_direct_top5_qwen25_05b)
        MODEL_REF="Artifacts/Factoid_SFT/models/direct_top5_full_resources_gold_eval_qwen25_05b/adapter"
        ;;
      dpo_representative_qwen25_05b)
        MODEL_REF="Artifacts/cse_dpo/exact_orbit_runs/factoid_direct_top5_full_resources_qwen25_05b/representative_dpo/final_adapter"
        ;;
      dpo_softmax_llama8b_run1)
        MODEL_REF="Artifacts/cse_dpo/softmax_dpo_paper_style_llama8b/runs/llama31_8b_full_single_answer_run1/best_model_by_loss"
        ;;
      dpo_standard_llama8b)
        MODEL_REF="Artifacts/cse_dpo/pairwise_dpo_llama8b/runs/standard_dpo_llama31_8b_full_single_answer_train_same_question_only_val_dev_full_run1/best_model_by_mrr"
        ;;
      dpo_cal_llama8b)
        MODEL_REF="Artifacts/cse_dpo/pairwise_dpo_llama8b/runs/cal_dpo_llama31_8b_full_single_answer_train_same_question_only_val_dev_full_run1/best_model_by_mrr"
        ;;
      *)
        echo "Unsupported model slug for aligned_single_answer: ${MODEL_SLUG}" >&2
        echo "Supported model slugs: base, sft_full_single_answer, sft_full_single_answer_per_alias, sft_direct_top5, sft_direct_top5_qwen25_05b, dpo_representative_qwen25_05b, dpo_softmax_llama8b_run1, dpo_standard_llama8b, dpo_cal_llama8b" >&2
        exit 1
        ;;
    esac
    case "${STRATEGY}" in
      s1_single_greedy|s3_single_sample_n1|s3_single_sample_n5|s3_single_sample_n10|s3_single_sample_n20)
        PROMPT_REF="factoid-single-answer-aligned-v1"
        CANDIDATE_MODE="single_answer"
        POLICIES=(permissive strict first_entity)
        ;;
      s2_top5_greedy)
        PROMPT_REF="factoid-top-five-eval-v1"
        CANDIDATE_MODE="top_five"
        POLICIES=(permissive)
        ;;
      *)
        echo "Unsupported strategy for aligned_single_answer: ${STRATEGY}" >&2
        echo "Supported strategies: s1_single_greedy, s2_top5_greedy, s3_single_sample_n1, s3_single_sample_n5, s3_single_sample_n10, s3_single_sample_n20" >&2
        exit 1
        ;;
    esac
    ;;
  prompt_mismatch_ablation)
    MODEL_SLUG="${MODEL_SLUG:-sft_mr5}"
    TARGET="${TARGET:-dev_mr5}"
    STRATEGY="${STRATEGY:-s3_single_sample_n20}"
    case "${MODEL_SLUG}" in
      base)
        MODEL_REF="${BASE_MODEL_REF}"
        ;;
      sft_mr5)
        MODEL_REF="Artifacts/Factoid_SFT/models/mr5_resources/adapter"
        ;;
      sft_full)
        MODEL_REF="Artifacts/Factoid_SFT/models/full_resources/adapter"
        ;;
      *)
        echo "Unsupported model slug for prompt_mismatch_ablation: ${MODEL_SLUG}" >&2
        echo "Supported model slugs: base, sft_mr5, sft_full" >&2
        exit 1
        ;;
    esac
    case "${STRATEGY}" in
      s1_single_greedy|s3_single_sample_n1|s3_single_sample_n5|s3_single_sample_n10|s3_single_sample_n20)
        PROMPT_REF="factoid-single-answer-v1"
        CANDIDATE_MODE="single_answer"
        POLICIES=(permissive strict first_entity)
        ;;
      s2_top5_greedy)
        PROMPT_REF="factoid-top-five-v1"
        CANDIDATE_MODE="top_five"
        POLICIES=(permissive)
        ;;
      *)
        echo "Unsupported strategy for prompt_mismatch_ablation: ${STRATEGY}" >&2
        echo "Supported strategies: s1_single_greedy, s2_top5_greedy, s3_single_sample_n1, s3_single_sample_n5, s3_single_sample_n10, s3_single_sample_n20" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported FACTOID_CANDIDATE_DIAGNOSIS_PROFILE: ${PROFILE}" >&2
    echo "Supported profiles: aligned_single_answer, prompt_mismatch_ablation" >&2
    exit 1
    ;;
esac

case "${TARGET}" in
  dev_mr5)
    EVAL_INPUT=(data/BioASQ_factoid_sft_prepared/original_mr5/eval_prepared.json)
    TARGET_FLAGS=()
    ;;
  dev_full)
    if [[ "${PROFILE}" == "aligned_single_answer" ]]; then
      EVAL_INPUT=(data/BioASQ_factoid_sft_prepared/single_answer_full_resources/eval_prepared.json)
    else
      EVAL_INPUT=(data/BioASQ_factoid_sft_prepared/original_full_resources/eval_prepared.json)
    fi
    TARGET_FLAGS=(
      --max-resources 0
      --max-resource-chars 0
    )
    ;;
  test_full)
    EVAL_INPUT=(
      data/Task13BTest/13B1_golden.json
      data/Task13BTest/13B2_golden.json
      data/Task13BTest/13B3_golden.json
      data/Task13BTest/13B4_golden.json
    )
    TARGET_FLAGS=(
      --max-resources 0
      --max-resource-chars 0
    )
    ;;
  *)
    echo "Unsupported target: ${TARGET}" >&2
    echo "Supported targets: dev_mr5, dev_full, test_full" >&2
    exit 1
    ;;
esac

GENERATION_ROOT="${ARTIFACT_ROOT}/generations/${TARGET}/${MODEL_SLUG}/${STRATEGY}"
MANIFEST_JSON="${GENERATION_ROOT}/manifest.json"
PREDICTION_JSON=""
MANIFEST_MODEL_REF=""

if [[ -f "${MANIFEST_JSON}" ]]; then
  MANIFEST_VALUES="$(python - "${MANIFEST_JSON}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
models = payload.get("models") or []
for model_summary in models:
    if not isinstance(model_summary, dict):
        continue
    paths = model_summary.get("paths") or {}
    predictions = paths.get("predictions")
    model_block = model_summary.get("model") or {}
    load_target = model_block.get("load_target") or model_block.get("ref")
    if isinstance(predictions, str) and predictions and isinstance(load_target, str) and load_target:
        print(predictions)
        print(load_target)
        break
PY
)"
  if [[ -n "${MANIFEST_VALUES}" ]]; then
    mapfile -t MANIFEST_LINES <<<"${MANIFEST_VALUES}"
    PREDICTION_JSON="${MANIFEST_LINES[0]:-}"
    MANIFEST_MODEL_REF="${MANIFEST_LINES[1]:-}"
  fi
fi

if [[ -z "${PREDICTION_JSON}" ]]; then
  PREDICTION_JSON="$(find "${GENERATION_ROOT}" -name predictions.json -print | sort | head -n 1)"
fi
if [[ -n "${MANIFEST_MODEL_REF}" ]]; then
  MODEL_REF="${MANIFEST_MODEL_REF}"
fi

if [[ -z "${PREDICTION_JSON}" ]]; then
  echo "Could not find predictions.json under ${GENERATION_ROOT}" >&2
  echo "Run the generation step first." >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UNSLOTH_DISABLE_STATISTICS=1

echo "Reanalyzing factoid candidate diagnosis"
echo "  profile:      ${PROFILE}"
echo "  model slug:   ${MODEL_SLUG}"
echo "  target:       ${TARGET}"
echo "  strategy:     ${STRATEGY}"
echo "  parser mode:  ${PARSER_MODE}"
echo "  predictions:  ${PREDICTION_JSON}"

for policy in "${POLICIES[@]}"; do
  OUTPUT_ROOT="${ARTIFACT_ROOT}/reanalysis/${TARGET}/${MODEL_SLUG}/${STRATEGY}/parser_${PARSER_MODE}/policy_${policy}"
  echo
  echo "=== policy: ${policy} ==="
  python src/utility/factoid_candidate_diagnosis.py \
    --prediction-json "${PREDICTION_JSON}" \
    --output-dir "${OUTPUT_ROOT}" \
    --model-ref "${MODEL_REF}" \
    --candidate-extraction-mode "${CANDIDATE_MODE}" \
    --single-answer-policy "${policy}" \
    --factoid-parser-mode "${PARSER_MODE}" \
    --ranker logp_mean logp_sum frequency generation_order \
    --keep-top-k 5 \
    --max-recall-k 5 \
    --eos-policy excluded \
    --eval-input "${EVAL_INPUT[@]}" \
    --question-types factoid \
    --prompt-file "${PROMPT_FILE}" \
    --prompt "${PROMPT_REF}" \
    --chat-template llama-3 \
    --prompt-format chat \
    --max-resources 5 \
    --max-resource-chars 1200 \
    --resource-selection first \
    --resource-granularity document \
    --max-factoid-answers 5 \
    --max-seq-length 4096 \
    --batch-size 1 \
    --local-files-only \
    "${TARGET_FLAGS[@]}" \
    "$@"
done
