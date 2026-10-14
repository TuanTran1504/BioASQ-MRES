#!/bin/bash
# Original + max-five SFT, three backbones; test scoring never selects models.
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
submission="${PWD}/outputs/sft_test_evaluation/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python scripts/run_sft_test_evaluation.py --mode prepare --submission-dir "$submission" --models "${models[@]}"
record="${submission}/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke=$(qsub -N "${model}-sfttest-smoke" -l walltime=01:00:00 \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=1" jobs/evaluate_sft_test.pbs)
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full=$(qsub -N "${model}-sfttest-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=0" jobs/evaluate_sft_test.pbs)
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted original + max-five SFT greedy test evaluation. Technical smoke uses internal validation; full runs use 95 official test factoids. No training.'
