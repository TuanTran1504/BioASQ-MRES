#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
preferences="${1:?Usage: submit_expanded_generator_dpo.sh REVIEWED_EXPANDED_PREFERENCES}"
preferences="$(cd "$preferences" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
submission="${PWD}/outputs/generator_feedback_dpo_expanded/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 scripts/prepare_expanded_generator_dpo.py --repository "$(cd .. && pwd)" --preferences "$preferences" --submission "$submission"
record="$submission/jobs.tsv"
printf 'stage\tjob_id\tdepends_on\n' > "$record"
smoke=$(qsub -N llama-expanded-smoke -l walltime=01:00:00 \
  -v "DPO_SUBMISSION=${submission},SMOKE_TEST=1" "$submission/train_expanded_generator_dpo.pbs")
printf 'smoke\t%s\t\n' "$smoke" | tee -a "$record"
train=$(qsub -N llama-expanded-train -W "depend=afterok:${smoke}" \
  -v "DPO_SUBMISSION=${submission},SMOKE_TEST=0" "$submission/train_expanded_generator_dpo.pbs")
printf 'train\t%s\t%s\n' "$train" "$smoke" | tee -a "$record"
training="${PWD}/outputs/generator_feedback_dpo_expanded/llama31-compactness-train-${train}"
select=$(qsub -W "depend=afterok:${train}" \
  -v "DPO_SUBMISSION=${submission},TRAINING_DIRECTORY=${training}" "$submission/select_expanded_generator_dpo.pbs")
printf 'select\t%s\t%s\n' "$select" "$train" | tee -a "$record"
echo "Submission record: $record"
echo 'Three GPU jobs: training/selection smoke, expanded DPO, then all-epoch validation144 selection. Frozen companions/reranker; no dev160 or test evaluation.'
