#!/bin/bash
# Recover one interrupted bank. Existing bank files remain unchanged.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if (( $# != 2 )); then
  echo 'Usage: bash scripts/resume_max5_dpo_bank.sh MODEL INTERRUPTED_BANK' >&2
  exit 2
fi
model="$1"
case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
resume=$(cd "$2" && pwd)
case "$resume" in *','*|*$'\n'*) echo 'Unsupported character in bank path' >&2; exit 2 ;; esac
name=$(basename "$resume")
if [[ ! "$name" =~ ^${model}-generate-([0-9]+\.gadi-pbs)$ ]]; then
  echo 'Expected an interrupted full PBS bank for this model' >&2
  exit 2
fi
old_job="${BASH_REMATCH[1]}"
old_state=$(qstat -fx "$old_job" | awk '$1 == "job_state" {print $3}')
if [[ "$old_state" != F ]]; then
  echo "Old job must be finished before recovery; state=${old_state:-unknown}" >&2
  exit 2
fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/run_max5_dpo_generation.py --model "$model" --resume-from "$resume" --validate-only
mkdir -p outputs/max5_dpo_banks_8b
record="outputs/max5_dpo_banks_8b/submission-recovery-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
job=$(qsub -N "${model}-max5bank-resume" -l "walltime=${BANK_WALLTIME:-12:00:00}" \
  -v "MODEL_KEY=${model},SMOKE_TEST=0,RESUME_BANK=${resume}" jobs/generate_max5_dpo_banks.pbs)
printf '%s\tfull\t%s\t\n' "$model" "$job" | tee -a "$record"
echo "Submission record: $record"
echo 'Submitted one recovery job. Saved draws stay unchanged; only missing draws are generated.'
echo 'Recreate the cancelled reranker with: bash scripts/submit_sft_ensemble_reranker.sh'
