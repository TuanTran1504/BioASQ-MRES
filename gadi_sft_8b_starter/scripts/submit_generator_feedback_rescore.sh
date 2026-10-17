#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submission="${1:?Usage: submit_generator_feedback_rescore.sh ORIGINAL_SUBMISSION SAVED_GENERATION}"
generation="${2:?Usage: submit_generator_feedback_rescore.sh ORIGINAL_SUBMISSION SAVED_GENERATION}"
submission="$(cd "$submission" && pwd)"
generation="$(cd "$generation" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
recovery="${PWD}/outputs/generator_feedback_dpo_evaluation/scoring-recovery-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$recovery" == *','* || "$recovery" == *$'\n'* ]]; then echo 'Unsupported recovery path' >&2; exit 2; fi
python ../scripts/recover_generator_feedback_scoring.py --mode prepare \
  --submission "$submission" --generation "$generation" --recovery "$recovery"
job=$(qsub -v "SCORING_RECOVERY=${recovery}" "$recovery/job.pbs")
printf 'stage\tjob_id\nscore\t%s\n' "$job" > "$recovery/jobs.tsv"
echo "CPU rescore job: $job"
echo "Submission record: $recovery/jobs.tsv"
echo 'Reusing completed GPU responses. No generation, training or paid judging.'
