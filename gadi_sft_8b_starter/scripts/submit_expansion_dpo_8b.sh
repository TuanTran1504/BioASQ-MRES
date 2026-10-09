#!/bin/bash
# Stage one generates responses. Stage two requires explicit reviewed preferences.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-generate}"
if (( $# > 0 )); then shift; fi
full=0
preferences=""
case "$mode" in
  generate)
    if [[ "${1:-}" == --full-dataset ]]; then full=1; shift; fi
    ;;
  train)
    preferences="${1:?Provide the reviewed preference directory}"
    preferences="$(cd "$preferences" && pwd)"
    shift
    ;;
  *) echo 'Usage: submit_expansion_dpo_8b.sh generate [--full-dataset] [models...] | train preferences [models...]' >&2; exit 1 ;;
esac
if [[ "$preferences" == *','* || "$preferences" == *$'\n'* ]]; then
  echo 'Preference path cannot contain a comma or newline in PBS variables' >&2; exit 1
fi
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
declare -A seen
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 1 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 1; fi
  seen[$model]=1
  args=(--mode validate --model "$model")
  if [[ -n "$preferences" ]]; then args+=(--preferences "$preferences"); fi
  if [[ "$full" == 1 ]]; then args+=(--full-dataset); fi
  python scripts/run_expansion_dpo_8b.py "${args[@]}"
done
mkdir -p outputs/expansion_dpo_8b
record="outputs/expansion_dpo_8b/submission-${mode}-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tmode\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  variables="MODEL_KEY=${model},DPO_MODE=${mode},FULL_DATASET=${full}"
  if [[ -n "$preferences" ]]; then variables+=",PREFERENCES=${preferences}"; fi
  smoke="$(qsub -N "${model}-dpo-${mode}-smoke" -l walltime=02:00:00 \
    -v "${variables},SMOKE_TEST=1" jobs/expansion_dpo_8b.pbs)"
  printf '%s\t%s\tsmoke\t%s\t\n' "$model" "$mode" "$smoke" | tee -a "$record"
  full_job="$(qsub -N "${model}-dpo-${mode}-full" -W "depend=afterok:${smoke}" \
    -v "${variables},SMOKE_TEST=0" jobs/expansion_dpo_8b.pbs)"
  printf '%s\t%s\tfull\t%s\t%s\n' "$model" "$mode" "$full_job" "$smoke" | tee -a "$record"
done
if [[ "$mode" == generate ]]; then
  echo 'Submitted response banks only. Review biomedical labels and construct shared preferences before DPO training.'
else
  echo 'Submitted DPO pilot training with each expansion SFT checkpoint as its own frozen reference.'
fi
