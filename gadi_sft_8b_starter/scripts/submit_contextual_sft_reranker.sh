#!/bin/bash
# Fixed max-five SFT pools: contextual scoring and validation-selected logistic fusion.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_CACHE="/scratch/nl78/${USER}/hf_cache/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# New ~110M encoder checkpoint; download once on the login node, never on workers.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
baseline="${1:-outputs/sft_ensemble_reranker/train-181350943.gadi-pbs}"
comparison="${2:-outputs/sft_ensemble_test/comparison-181287698.gadi-pbs}"
submission="$PWD/outputs/contextual_sft_reranker/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 ../scripts/contextual_sft_reranker.py --mode prepare --repository "$(cd .. && pwd)" \
  --baseline "$baseline" --comparison "$comparison" --submission "$submission" --download-model
record="$submission/jobs.tsv"
printf 'stage\tjob_id\tdepends_on\n' > "$record"
smoke=$(qsub -N max5-ce-smoke -l walltime=01:00:00 \
  -v "CONTEXTUAL_SUBMISSION=$submission,CONTEXTUAL_STAGE=train,SMOKE_TEST=1" "$submission/job.pbs")
printf 'smoke\t%s\t\n' "$smoke" | tee -a "$record"
train=$(qsub -N max5-ce-train -W "depend=afterok:$smoke" \
  -v "CONTEXTUAL_SUBMISSION=$submission,CONTEXTUAL_STAGE=train,SMOKE_TEST=0" "$submission/job.pbs")
printf 'train\t%s\t%s\n' "$train" "$smoke" | tee -a "$record"
score=$(qsub -N max5-ce-test95 -l walltime=02:00:00 -W "depend=afterok:$train" \
  -v "CONTEXTUAL_SUBMISSION=$submission,CONTEXTUAL_STAGE=evaluate,CONTEXTUAL_TRAIN_JOB=$train" "$submission/job.pbs")
printf 'score\t%s\t%s\n' "$score" "$train" | tee -a "$record"
echo "Submission record: $record"
echo 'Three dependent GPU jobs: smoke, contextual reranker fitting/validation selection, fixed test95 rescore.'
echo 'Existing max-five SFT pools only. No generator training, new answer generation, DPO or paid judging.'
