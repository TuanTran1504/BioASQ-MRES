#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
baseline="${1:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}"
comparison="${2:-outputs/sft_ensemble_test/comparison-181287698.gadi-pbs}"
submission="${PWD}/outputs/reranker_relation_ablation/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python ../scripts/ablate_reranker_relation_labels.py --mode prepare \
  --repository "$(cd .. && pwd)" --baseline "$baseline" --comparison "$comparison" --submission "$submission"
job=$(qsub -v "ABLATION_SUBMISSION=${submission}" "$submission/job.pbs")
printf 'stage\tjob_id\nablation\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'One CPU job retrains tag-blind rerankers and compares cached validation/test pools. No GPU generation or DPO.'
