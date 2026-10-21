#!/bin/bash
# Supervised selector pilot: CPU baseline, GPU smoke/train, CPU validation audit.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_CACHE="/scratch/nl78/${USER}/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
audit="${1:-outputs/answer_selector_data/audit-181824213.gadi-pbs}"
submission="$PWD/outputs/answer_selector_pilot/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 ../scripts/run_answer_selector_pilot.py --mode prepare --repository "$(cd .. && pwd)" \
  --audit "$audit" --submission "$submission"
record="$submission/jobs.tsv"
printf 'stage\tjob_id\tdepends_on\n' > "$record"
baseline=$(qsub -v "SELECTOR_SUBMISSION=$submission" "$submission/baseline_answer_selector.pbs")
printf 'baseline\t%s\t\n' "$baseline" | tee -a "$record"
smoke=$(qsub -N answer-sel-smoke -l walltime=01:00:00 \
  -v "SELECTOR_SUBMISSION=$submission,SMOKE_TEST=1" "$submission/train_answer_selector.pbs")
printf 'smoke\t%s\t\n' "$smoke" | tee -a "$record"
train=$(qsub -N answer-sel-train -W "depend=afterok:$baseline:$smoke" \
  -v "SELECTOR_SUBMISSION=$submission,SMOKE_TEST=0,SELECTOR_SMOKE_JOB=$smoke,SELECTOR_BASELINE_JOB=$baseline" \
  "$submission/train_answer_selector.pbs")
printf 'train\t%s\t%s:%s\n' "$train" "$baseline" "$smoke" | tee -a "$record"
report=$(qsub -W "depend=afterok:$train" \
  -v "SELECTOR_SUBMISSION=$submission,SELECTOR_TRAIN_JOB=$train,SELECTOR_BASELINE_JOB=$baseline" \
  "$submission/report_answer_selector.pbs")
printf 'report\t%s\t%s\n' "$report" "$train" | tee -a "$record"
echo "Submission record: $record"
echo 'Fresh Llama selector LoRA; tie-aware supervised loss; all144 validation questions; existing four-sample pools.'
echo 'No generator changes, DPO training, paid judging or test95 evaluation.'
