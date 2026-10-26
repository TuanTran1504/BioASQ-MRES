#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
comparison="${1:-outputs/answer_selector_pilot/comparison-20261026-085714/comparison_summary.json}"
submission="$PWD/outputs/answer_selector_fusion/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 ../scripts/fuse_answer_selector_pilots.py --mode prepare \
  --repository "$(cd .. && pwd)" --comparison "$comparison" --submission "$submission"
record="$submission/jobs.tsv"
printf 'stage\tjob_id\n' > "$record"
job=$(qsub -v "SELECTOR_FUSION_SUBMISSION=$submission" "$submission/job.pbs")
printf 'fusion\t%s\n' "$job" | tee -a "$record"
echo "Submission record: $record"
echo "Report after completion: $submission/fusion-$job/report.md"
echo 'One CPU job: four fusion conditions, individual/logistic baselines, complementarity and gains/losses.'
echo 'Saved validation rankings only; no GPU generation, training, paid judging, DPO pairs or test95 evaluation.'
