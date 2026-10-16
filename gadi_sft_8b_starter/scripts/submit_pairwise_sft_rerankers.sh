#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
baseline="${1:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}"
comparison="${2:-outputs/sft_ensemble_test/comparison-181287698.gadi-pbs}"
submission="${PWD}/outputs/pairwise_sft_rerankers/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python ../scripts/compare_pairwise_sft_rerankers.py --mode prepare \
  --repository "$(cd .. && pwd)" --baseline "$baseline" --comparison "$comparison" --submission "$submission"
job=$(qsub -v "PAIRWISE_SUBMISSION=${submission}" "$submission/job.pbs")
printf 'stage\tjob_id\ncomparison\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'One CPU job compares pairwise ranking and typography redundancy on unchanged pools. No GPU generation, API calls or DPO.'
