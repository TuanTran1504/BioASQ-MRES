#!/bin/bash
# Freeze complete existing SFT banks, then submit one CPU data audit only.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rerankers="${1:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}"
submission="${PWD}/outputs/answer_selector_data/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 ../scripts/prepare_answer_selector_data.py --mode prepare \
  --repository "$(cd .. && pwd)" --rerankers "$rerankers" --submission "$submission"
job=$(qsub -v "SELECTOR_SUBMISSION=${submission}" "$submission/job.pbs")
printf 'stage\tjob_id\naudit\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'Submitted one CPU audit and selector dataset export: fitting1296/validation144, existing four-sample SFT banks.'
echo 'No GPU training, new generation, DPO pairs or paid judging.'
