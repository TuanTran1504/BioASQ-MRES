#!/bin/bash
# Validate both saved adapters, then submit smoke -> full evaluation dependencies.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
python3 scripts/run_matched_qwen25_expansion.py --model-size 05b --mode validate
python3 scripts/run_matched_qwen25_expansion.py --model-size 3b --mode validate

mkdir -p outputs/matched_qwen25
submission_log="outputs/matched_qwen25/submission-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model_size\tcondition\tjob_id\tdepends_on\n' > "$submission_log"
echo "Submission record: $submission_log"
for model_size in 05b 3b; do
  smoke_job="$(qsub "jobs/evaluate_qwen25_${model_size}_expansion_smoke.pbs")"
  printf '%s\tsmoke\t%s\t\n' "$model_size" "$smoke_job" | tee -a "$submission_log"
  full_job="$(qsub -W "depend=afterok:${smoke_job}" "jobs/evaluate_qwen25_${model_size}_expansion.pbs")"
  printf '%s\tfull\t%s\t%s\n' "$model_size" "$full_job" "$smoke_job" | tee -a "$submission_log"
done
echo 'Submitted two smoke jobs and two full evaluations. Full jobs require successful smoke exit.'
