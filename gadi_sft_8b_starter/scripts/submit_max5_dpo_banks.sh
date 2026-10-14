#!/bin/bash
# Generate response banks only; judging and the two preference arms come later.
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
for model in "${models[@]}"; do
  python scripts/run_max5_dpo_generation.py --model "$model" --validate-only
done
mkdir -p outputs/max5_dpo_banks_8b
record="outputs/max5_dpo_banks_8b/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke=$(qsub -N "${model}-max5bank-smoke" -l walltime=02:00:00 \
    -v "MODEL_KEY=${model},SMOKE_TEST=1" jobs/generate_max5_dpo_banks.pbs)
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full=$(qsub -N "${model}-max5bank-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SMOKE_TEST=0" jobs/generate_max5_dpo_banks.pbs)
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted max-five banks only: one greedy + four sampled responses per fitting/validation question. No DPO training or paid judging.'
