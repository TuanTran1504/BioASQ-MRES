#!/bin/bash
# One CPU audit only; no GPU generation, paid judging or DPO training.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rerankers="${1:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}"
submission="${PWD}/outputs/reranker_guided_dpo/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python ../scripts/prepare_reranker_guided_dpo.py --mode prepare \
  --repository "$(cd .. && pwd)" --rerankers "$rerankers" --submission "$submission"
job=$(qsub -v "PILOT_SUBMISSION=${submission}" jobs/audit_reranker_guided_dpo.pbs)
printf 'stage\tjob_id\naudit\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'Submitted one CPU audit: five question-disjoint reranker fits and provisional preferences. No DPO training or paid judging.'
