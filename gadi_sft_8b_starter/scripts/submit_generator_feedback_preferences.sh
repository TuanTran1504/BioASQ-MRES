#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
models=("$@")
if (( ${#models[@]} == 0 )); then models=(llama31); fi
submission="${PWD}/outputs/generator_feedback_dpo/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python ../scripts/prepare_generator_feedback_preferences.py --mode prepare \
  --repository "$(cd .. && pwd)" --submission "$submission" --models "${models[@]}" \
  --rerankers "${FEEDBACK_RERANKERS:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}" \
  --previous-audit "${FEEDBACK_AUDIT:-outputs/reranker_guided_dpo/audit-181379068.gadi-pbs}"
job=$(qsub -v "FEEDBACK_SUBMISSION=${submission}" "$submission/job.pbs")
printf 'stage\tjob_id\npreferences\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'One CPU job reuses saved responses and frozen feedback models. Provisional preferences only; no GPU training or paid judging.'
