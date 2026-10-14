#!/bin/bash
# Three smoke jobs, three generation jobs, one dependent CPU scoring job.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
submission="${PWD}/outputs/sft_ensemble_test/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python scripts/run_sft_ensemble_test.py --mode prepare --submission "$submission"
record="${submission}/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
full_jobs=()
echo "Submission record: $record"
for model in llama31 qwen3 ministral3; do
  smoke=$(qsub -N "${model}-sftens-smoke" -l walltime=01:00:00 \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=1" jobs/generate_sft_ensemble_test.pbs)
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full=$(qsub -N "${model}-sftens-full" -W "depend=afterok:${smoke}" \
    -v "MODEL_KEY=${model},SUBMISSION_DIR=${submission},SMOKE_TEST=0" jobs/generate_sft_ensemble_test.pbs)
  printf '%s\tfull\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
  full_jobs+=("$full")
done
python scripts/run_sft_ensemble_test.py --mode record --submission "$submission"
dependency=$(IFS=:; echo "${full_jobs[*]}")
score=$(qsub -W "depend=afterok:${dependency}" -v "SUBMISSION_DIR=${submission}" jobs/score_sft_ensemble_test.pbs)
printf 'combined\tscore\t%s\t%s\n' "$score" "$dependency" | tee -a "$record"
echo 'Submitted ten-sample original/max-five generation and fixed ensemble scoring. Existing greedy tests are reused. No reranker training or paid judging.'
