#!/bin/bash
# Train each base once; score all epochs, then export two independent winners.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models=()
reuse_record=""
while (( $# )); do
  case "$1" in
    --reuse-smoke-record)
      if (( $# < 2 )); then echo 'Missing smoke submission record' >&2; exit 2; fi
      reuse_record="$2"; shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) models+=("$1"); shift ;;
  esac
done
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
declare -A seen=()
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 2; fi
  seen[$model]=1
done
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
matrix="${DUAL_TRAINING_MATRIX:-configs/matched_8b_sft.json}"
configuration="${DUAL_SELECTION_CONFIG:-configs/sft_dual_selection_8b.json}"
output_root="${DUAL_OUTPUT_ROOT:-outputs/sft_dual_selection_8b}"
python scripts/prepare_matched_8b_sft.py --mode build --config "$matrix"
submission="${PWD}/${output_root}/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then
  echo 'Submission path cannot contain a comma or newline' >&2; exit 2
fi
# Verify data, historical backbone pins, scorer and all models before any qsub.
python scripts/prepare_sft_dual_selection.py --config "$configuration" --submission-dir "$submission" --models "${models[@]}"
if [[ "${DUAL_EVALUATE_DEV:-0}" == 1 ]]; then
  python scripts/run_selected_sft_evaluation.py --mode validate-protocol --submission-dir "$submission" --models "${models[@]}"
fi
declare -A reusable=()
if [[ -n "$reuse_record" ]]; then
  if [[ ! -f "$reuse_record" ]]; then echo 'Smoke submission record not found' >&2; exit 2; fi
  for model in "${models[@]}"; do
    job=$(awk -F '\t' -v model="$model" 'NR>1 && $1==model && $2=="smoke" {print $3}' "$reuse_record")
    if [[ ! "$job" =~ ^[0-9]+\.gadi-pbs$ ]]; then echo "Expected one smoke job for $model" >&2; exit 2; fi
    training_run="${PWD}/${output_root}/${model}-expansion-sft-${job}-smoke"
    stored_run=$(awk -F '\t' -v model="$model" 'NR>1 && $1==model && $2=="smoke" {print $5}' "$reuse_record")
    if [[ -n "$stored_run" ]]; then training_run="$stored_run"; fi
    python scripts/select_sft_epoch_checkpoints.py --model "$model" \
      --training-run "$training_run" --submission-dir "$submission" --smoke-test --validate-only
    reusable[$model]="$training_run"
  done
fi
record="${submission}/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\ttraining_run\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke_env="MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=1,TRAIN_OUTPUT_ROOT=${output_root}"
  if [[ -n "${reusable[$model]:-}" ]]; then smoke_env+=",REUSE_TRAINING_RUN=${reusable[$model]}"; fi
  smoke="$(qsub -N "${model}-sft-dual-smoke" -l walltime=02:00:00 \
    -v "$smoke_env" jobs/train_sft_dual_selection_8b.pbs)"
  printf '%s\tsmoke\t%s\t\t%s\n' "$model" "$smoke" "${reusable[$model]:-}" | tee -a "$record"
  full="$(qsub -N "${model}-sft-dual-train" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=0,TRAIN_OUTPUT_ROOT=${output_root}" jobs/train_sft_dual_selection_8b.pbs)"
  training_run="${PWD}/${output_root}/${model}-expansion-sft-${full}"
  printf '%s\ttrain\t%s\t%s\t%s\n' "$model" "$full" "$smoke" "$training_run" | tee -a "$record"
  select="$(qsub -N "${model}-sft-dual-select" -W "depend=afterok:${full}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},TRAINING_RUN=${training_run}" jobs/select_sft_dual_selection_8b.pbs)"
  printf '%s\tselect\t%s\t%s\t%s\n' "$model" "$select" "$full" "$training_run" | tee -a "$record"
  if [[ "${DUAL_EVALUATE_DEV:-0}" == 1 ]]; then
    evaluation="$(qsub -N "${model}-sft-dev160" -W "depend=afterok:${select}" \
      -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},TRAINING_RUN=${training_run},SELECTION_RUN=${training_run}/selection-${select}" \
      jobs/evaluate_selected_sft_8b.pbs)"
    printf '%s\tevaluate\t%s\t%s\t%s\n' "$model" "$evaluation" "$select" "$training_run" | tee -a "$record"
  fi
done
echo 'Submitted fresh expansion SFT and dual checkpoint selection. Smoke -> training -> validation selection.'
if [[ "${DUAL_EVALUATE_DEV:-0}" == 1 ]]; then echo 'Dev160 evaluation follows successful checkpoint selection.'; fi
