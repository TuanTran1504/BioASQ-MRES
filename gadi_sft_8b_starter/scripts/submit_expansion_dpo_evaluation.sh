#!/bin/bash
# Evaluate all sources before submitting any GPU work.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
config="$(pwd)/configs/expansion_dpo_evaluation.json"
if [[ "$config" == *','* || "$config" == *$'\n'* ]]; then
  echo 'Configuration path cannot contain a comma or newline' >&2; exit 1
fi
declare -A seen
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 1 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 1; fi
  seen[$model]=1
  python scripts/run_expansion_dpo_evaluation.py --config "$config" --model "$model" --mode validate
done
mkdir -p outputs/expansion_dpo_evaluation
record="outputs/expansion_dpo_evaluation/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke="$(qsub -N "${model}-dpo-eval-smoke" -l walltime=02:00:00 \
    -v "MODEL_KEY=${model},SMOKE_TEST=1,EVAL_CONFIG=${config}" jobs/evaluate_expansion_dpo_8b.pbs)"
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full="$(qsub -N "${model}-dpo-eval-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SMOKE_TEST=0,EVAL_CONFIG=${config}" jobs/evaluate_expansion_dpo_8b.pbs)"
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted matched expansion SFT/DPO greedy and ten-sample evaluations, gated by smoke jobs.'
