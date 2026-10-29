#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
original="${1:?Usage: submit_generator_form_rescore.sh ORIGINAL_SUBMISSION}"
original="$(cd "$original" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
recovery="$PWD/outputs/generator_form_designs/scoring-recovery-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$original" == *','* || "$recovery" == *','* || "$original" == *$'\n'* || "$recovery" == *$'\n'* ]]; then
  echo 'Unsupported recovery path' >&2; exit 2
fi
python3 ../scripts/recover_generator_form_scoring.py --mode prepare --submission "$original" --recovery "$recovery"
job=$(qsub -v "FORM_RECOVERY=${recovery}" "$recovery/job.pbs")
printf 'model\tdesign\tstage\tjob_id\tdepends_on\toutput\nensemble\tboth\trecover\t%s\t\t%s\n' "$job" "$recovery/run-$job" > "$recovery/jobs.tsv"
echo "CPU recovery job: $job"
echo "Submission record: $recovery/jobs.tsv"
echo "Report after completion: $recovery/run-$job/comparison/report.md"
echo 'One CPU job; six saved validation runs plus joint comparison. No training, generation or paid judging.'
