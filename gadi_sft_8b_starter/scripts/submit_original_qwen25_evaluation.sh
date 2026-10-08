#!/bin/bash
# Submit a four-question smoke, then both full inference arms per model size.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
sizes=("$@")
if (( ${#sizes[@]} == 0 )); then sizes=(05b 3b); fi
for size in "${sizes[@]}"; do
  python scripts/run_original_qwen25_inference.py --model-size "$size" --mode validate
done
mkdir -p outputs/original_qwen25
record="outputs/original_qwen25/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model_size\tcondition\tjob_id\tdepends_on\n' > "$record"
echo "Submission record: $record"
for size in "${sizes[@]}"; do
  smoke="$(qsub -N "qwen25-${size}-orig-smoke" -l walltime=01:00:00 \
    -v "MODEL_SIZE=${size},SMOKE_TEST=1" jobs/evaluate_original_qwen25.pbs)"
  printf '%s\tsmoke\t%s\t\n' "$size" "$smoke" | tee -a "$record"
  full="$(qsub -N "qwen25-${size}-orig-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_SIZE=${size},SMOKE_TEST=0" jobs/evaluate_original_qwen25.pbs)"
  printf '%s\tfull\t%s\t%s\n' "$size" "$full" "$smoke" | tee -a "$record"
done
echo 'Submitted original SFT expansion plus ten-sample evaluation, gated by smoke jobs.'
