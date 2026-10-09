#!/bin/bash
# Add ten expansion draws; reuse all completed matched baseline evaluations.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
config="$(pwd)/configs/expansion_sampling_8b.json"
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3); fi
declare -A seen
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3) ;; *) echo "Unsupported completed model: $model" >&2; exit 1 ;; esac
  if [[ -n "${seen[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 1; fi
  seen[$model]=1
  python scripts/run_matched_8b_evaluation.py --config "$config" --model "$model" \
    --mode validate --expansion-sampling-only
done
mkdir -p outputs/expansion_sampling_8b
record="outputs/expansion_sampling_8b/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  smoke="$(qsub -N "${model}-exp10-smoke" -l walltime=02:00:00 \
    -v "MODEL_KEY=${model},SMOKE_TEST=1,EXPANSION_SAMPLING_ONLY=1,EVAL_CONFIG=${config}" jobs/evaluate_matched_8b_sft.pbs)"
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full="$(qsub -N "${model}-exp10-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SMOKE_TEST=0,EXPANSION_SAMPLING_ONLY=1,EVAL_CONFIG=${config}" jobs/evaluate_matched_8b_sft.pbs)"
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted ten-sample expansion inference only, gated by smoke jobs; completed baseline arms are reused.'
