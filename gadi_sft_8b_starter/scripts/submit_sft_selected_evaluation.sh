#!/bin/bash
# Attach dev160 evaluations to an existing batch without restarting training.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record="${1:-}"
if [[ -z "$record" ]]; then
  record=$(ls -t outputs/sft_max5_8b/submission-*/jobs.tsv 2>/dev/null | head -n 1) || true
fi
if [[ -z "$record" || ! -f "$record" ]]; then echo 'No max-five submission record found; pass its jobs.tsv path.' >&2; exit 2; fi
record="$(cd "$(dirname "$record")" && pwd)/$(basename "$record")"
submission="$(dirname "$record")"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
models=()
declare -A selects=() trains=()
while IFS=$'\t' read -r model job training; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model in record: $model" >&2; exit 2 ;; esac
  if [[ -n "${selects[$model]:-}" || ! "$job" =~ ^[0-9]+\.gadi-pbs$ || -z "$training" ]]; then
    echo 'Expected one selection job and training path per model' >&2; exit 2
  fi
  models+=("$model")
  selects[$model]="$job"
  trains[$model]="$training"
done < <(awk -F '\t' 'NR>1 && $2=="select" {print $1 "\t" $3 "\t" $5}' "$record")
if (( ${#models[@]} == 0 )); then echo 'No checkpoint-selection jobs in record' >&2; exit 2; fi
python scripts/run_selected_sft_evaluation.py --mode validate-protocol --submission-dir "$submission" --models "${models[@]}"
for model in "${models[@]}"; do
  existing=$(awk -F '\t' -v model="$model" 'NR>1 && $1==model && $2=="evaluate" {print $3}' "$record")
  if [[ -n "$existing" ]]; then echo "$model: evaluation already recorded: $existing"; continue; fi
  select="${selects[$model]}"
  training="${trains[$model]}"
  evaluation="$(qsub -N "${model}-sft-dev160" -W "depend=afterok:${select}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},TRAINING_RUN=${training},SELECTION_RUN=${training}/selection-${select}" \
    jobs/evaluate_selected_sft_8b.pbs)"
  printf '%s\tevaluate\t%s\t%s\t%s\n' "$model" "$evaluation" "$select" "$training" | tee -a "$record"
done
echo "Evaluation jobs recorded in: $record"
