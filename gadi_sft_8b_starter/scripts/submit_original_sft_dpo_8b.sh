#!/bin/bash
# Original single-answer banks first; reviewed concept/form/mixed DPO second.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-generate}"
if (( $# )); then shift; fi
preferences=""
strategy=mixed
select_mrr=0
case "$mode" in
  generate) ;;
  train)
    preferences="${1:?Provide the reviewed preference package}"; shift
    preferences="$(cd "$preferences" && pwd)"
    strategy="${1:-mixed}"; if (( $# )); then shift; fi
    case "$strategy" in concept|form|mixed|all|separate) ;; *) echo 'Strategy must be concept, form, mixed, all or separate' >&2; exit 2 ;; esac ;;
  *) echo 'Usage: submit_original_sft_dpo_8b.sh generate [models...] | train preferences [concept|form|mixed|all|separate] [--select-mrr] [models...]' >&2; exit 2 ;;
esac
if [[ "$preferences" == *','* || "$preferences" == *$'\n'* ]]; then echo 'Unsupported preference path' >&2; exit 2; fi
models=()
for option in "$@"; do
  if [[ "$option" == --select-mrr ]]; then select_mrr=1; else models+=("$option"); fi
done
if [[ "$mode" != train && "$select_mrr" == 1 ]]; then echo '--select-mrr requires train mode' >&2; exit 2; fi
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
declare -A seen=()
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 2; fi
  seen[$model]=1
done
strategies=("$strategy")
if [[ "$strategy" == all ]]; then strategies=(concept form mixed); fi
if [[ "$strategy" == separate ]]; then strategies=(concept form); fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
config="${PWD}/configs/original_sft_dpo_8b.json"
training_root="outputs/original_sft_dpo_8b"
if [[ "$select_mrr" == 1 ]]; then
  config="${PWD}/configs/original_sft_dpo_selection_8b.json"
  training_root="outputs/original_sft_dpo_selection_8b"
fi
# Refuse invalid banks, empty strata or warm-up-only schedules before any qsub.
for model in "${models[@]}"; do
  for condition in "${strategies[@]}"; do
    options=(--config "$config" --mode validate --model "$model" --full-dataset --pair-strategy "$condition")
    if [[ -n "$preferences" ]]; then options+=(--preferences "$preferences"); fi
    python scripts/run_expansion_dpo_8b.py "${options[@]}"
  done
done
mkdir -p "$training_root"
record="${training_root}/submission-${mode}-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstrategy\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  for condition in "${strategies[@]}"; do
    variables="MODEL_KEY=${model},DPO_MODE=${mode},PAIR_STRATEGY=${condition},DPO_CONFIG=${config}"
    if [[ -n "$preferences" ]]; then variables+=",PREFERENCES=${preferences}"; fi
    job_label="${model}-origdpo-${mode}"
    if [[ "$mode" == train ]]; then job_label="${model}-origdpo-${condition}"; fi
    smoke="$(qsub -N "${job_label}-smoke" -l walltime=02:00:00 \
      -v "${variables},SMOKE_TEST=1" jobs/original_sft_dpo_8b.pbs)"
    printf '%s\t%s\tsmoke\t%s\t\n' "$model" "$condition" "$smoke" | tee -a "$record"
    full="$(qsub -N "${job_label}-full" -W "depend=afterok:${smoke}" \
      -v "${variables},SMOKE_TEST=0" jobs/original_sft_dpo_8b.pbs)"
    printf '%s\t%s\tfull\t%s\t%s\n' "$model" "$condition" "$full" "$smoke" | tee -a "$record"
    if [[ "$mode" == train ]]; then
      training="${PWD}/${training_root}/${model}-${condition}-train-${full}"
      evaluation="$(qsub -N "${model}-origdpo-${condition}-eval" -W "depend=afterok:${full}" \
        -v "MODEL_KEY=${model},DPO_TRAINING_RUN=${training},DPO_CONFIG=${config}" jobs/evaluate_original_sft_dpo_8b.pbs)"
      printf '%s\t%s\tevaluation\t%s\t%s\n' "$model" "$condition" "$evaluation" "$full" | tee -a "$record"
    fi
  done
done
echo 'Generation submits banks only. Training uses reviewed pairs and its own frozen original-SFT reference.'
