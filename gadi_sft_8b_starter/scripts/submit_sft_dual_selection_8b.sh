#!/bin/bash
# Train each base once; score all epochs, then export two independent winners.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models=("$@")
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
python scripts/prepare_matched_8b_sft.py --mode build
submission="${PWD}/outputs/sft_dual_selection_8b/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then
  echo 'Submission path cannot contain a comma or newline' >&2; exit 2
fi
# Verify data, historical backbone pins, scorer and all models before any qsub.
python scripts/prepare_sft_dual_selection.py --submission-dir "$submission" --models "${models[@]}"
record="${submission}/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\ttraining_run\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke="$(qsub -N "${model}-sft-dual-smoke" -l walltime=02:00:00 \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=1" jobs/train_sft_dual_selection_8b.pbs)"
  printf '%s\tsmoke\t%s\t\t\n' "$model" "$smoke" | tee -a "$record"
  full="$(qsub -N "${model}-sft-dual-train" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=0" jobs/train_sft_dual_selection_8b.pbs)"
  training_run="${PWD}/outputs/sft_dual_selection_8b/${model}-expansion-sft-${full}"
  printf '%s\ttrain\t%s\t%s\t%s\n' "$model" "$full" "$smoke" "$training_run" | tee -a "$record"
  select="$(qsub -N "${model}-sft-dual-select" -W "depend=afterok:${full}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},TRAINING_RUN=${training_run}" jobs/select_sft_dual_selection_8b.pbs)"
  printf '%s\tselect\t%s\t%s\t%s\n' "$model" "$select" "$full" "$training_run" | tee -a "$record"
done
echo 'Submitted fresh expansion SFT and dual checkpoint selection. Smoke -> training -> validation selection.'
