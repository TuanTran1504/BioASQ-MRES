#!/bin/bash
# Original single-answer banks first; reviewed concept/form/mixed DPO second.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-generate}"
if (( $# )); then shift; fi
preferences=""
strategy=mixed
case "$mode" in
  generate) ;;
  train)
    preferences="${1:?Provide the reviewed preference package}"; shift
    preferences="$(cd "$preferences" && pwd)"
    strategy="${1:-mixed}"; if (( $# )); then shift; fi
    case "$strategy" in concept|form|mixed|all) ;; *) echo 'Strategy must be concept, form, mixed or all' >&2; exit 2 ;; esac ;;
  *) echo 'Usage: submit_original_sft_dpo_8b.sh generate [models...] | train preferences [concept|form|mixed|all] [models...]' >&2; exit 2 ;;
esac
if [[ "$preferences" == *','* || "$preferences" == *$'\n'* ]]; then echo 'Unsupported preference path' >&2; exit 2; fi
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
declare -A seen=()
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 2; fi
  seen[$model]=1
done
strategies=("$strategy")
if [[ "$strategy" == all ]]; then strategies=(concept form mixed); fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
config="${PWD}/configs/original_sft_dpo_8b.json"
# Refuse invalid banks, empty strata or warm-up-only schedules before any qsub.
for model in "${models[@]}"; do
  for condition in "${strategies[@]}"; do
    options=(--config "$config" --mode validate --model "$model" --full-dataset --pair-strategy "$condition")
    if [[ -n "$preferences" ]]; then options+=(--preferences "$preferences"); fi
    python scripts/run_expansion_dpo_8b.py "${options[@]}"
  done
done
mkdir -p outputs/original_sft_dpo_8b
record="outputs/original_sft_dpo_8b/submission-${mode}-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstrategy\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  for condition in "${strategies[@]}"; do
    variables="MODEL_KEY=${model},DPO_MODE=${mode},PAIR_STRATEGY=${condition},DPO_CONFIG=${config}"
    if [[ -n "$preferences" ]]; then variables+=",PREFERENCES=${preferences}"; fi
    smoke="$(qsub -N "${model}-origdpo-${mode}-smoke" -l walltime=02:00:00 \
      -v "${variables},SMOKE_TEST=1" jobs/original_sft_dpo_8b.pbs)"
    printf '%s\t%s\tsmoke\t%s\t\n' "$model" "$condition" "$smoke" | tee -a "$record"
    full="$(qsub -N "${model}-origdpo-${mode}-full" -W "depend=afterok:${smoke}" \
      -v "${variables},SMOKE_TEST=0" jobs/original_sft_dpo_8b.pbs)"
    printf '%s\t%s\tfull\t%s\t%s\n' "$model" "$condition" "$full" "$smoke" | tee -a "$record"
    if [[ "$mode" == train ]]; then
      training="${PWD}/outputs/original_sft_dpo_8b/${model}-${condition}-train-${full}"
      evaluation="$(qsub -N "${model}-origdpo-${condition}-eval" -W "depend=afterok:${full}" \
        -v "MODEL_KEY=${model},DPO_TRAINING_RUN=${training}" jobs/evaluate_original_sft_dpo_8b.pbs)"
      printf '%s\t%s\tevaluation\t%s\t%s\n' "$model" "$condition" "$evaluation" "$full" | tee -a "$record"
    fi
  done
done
echo 'Generation submits banks only. Training uses reviewed pairs and its own frozen original-SFT reference.'
