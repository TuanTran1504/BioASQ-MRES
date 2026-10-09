#!/bin/bash
# Submit inference only: three arms for each completed matched 8B SFT pair.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
config="$(pwd)/configs/matched_8b_evaluation.json"
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3); fi
declare -A seen
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unsupported completed model: $model" >&2; exit 1 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 1; fi
  seen[$model]=1
  python scripts/run_matched_8b_evaluation.py --config "$config" --model "$model" --mode validate
done
mkdir -p outputs/matched_8b_evaluation
record="outputs/matched_8b_evaluation/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke="$(qsub -N "${model}-8b-eval-smoke" -l walltime=02:00:00 \
    -v "MODEL_KEY=${model},SMOKE_TEST=1,EVAL_CONFIG=${config}" jobs/evaluate_matched_8b_sft.pbs)"
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full="$(qsub -N "${model}-8b-eval-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SMOKE_TEST=0,EVAL_CONFIG=${config}" jobs/evaluate_matched_8b_sft.pbs)"
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted greedy original, ten-sample original, and greedy expansion evaluation. No training submitted.'
