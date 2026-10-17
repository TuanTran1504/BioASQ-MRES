#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
preferences="${1:?Usage: submit_generator_feedback_dpo.sh REVIEWED_PREFERENCES_DIRECTORY}"
preferences="$(cd "$preferences" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub" PYTHONUNBUFFERED=1
submission="${PWD}/outputs/generator_feedback_dpo_training/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python scripts/run_generator_feedback_dpo.py --mode prepare --repository "$(cd .. && pwd)" \
  --preferences "$preferences" --submission "$submission"
record="$submission/jobs.tsv"
printf 'stage\tjob_id\tdepends_on\n' > "$record"
smoke=$(qsub -N llama-compact-smoke -l walltime=01:00:00 -v "DPO_SUBMISSION=${submission},SMOKE_TEST=1" "$submission/job.pbs")
printf 'smoke\t%s\t\n' "$smoke" | tee -a "$record"
full=$(qsub -N llama-compact-train -W "depend=afterok:${smoke}" -v "DPO_SUBMISSION=${submission},SMOKE_TEST=0" "$submission/job.pbs")
printf 'train\t%s\t%s\n' "$full" "$smoke" | tee -a "$record"
echo "Submission record: $record"
echo 'Submitted Llama compactness DPO smoke and dependent training. No new judging or generation.'
