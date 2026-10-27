#!/bin/bash
# Reuse complete SFT banks and existing official labels; CPU audit only.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
audit="${1:-outputs/answer_selector_data/audit-181824213.gadi-pbs}"
submission="${PWD}/outputs/generator_coverage_dpo/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 ../scripts/audit_generator_coverage_dpo.py --mode prepare \
  --repository "$(cd .. && pwd)" --selector-audit "$audit" --submission "$submission"
job=$(qsub -v "COVERAGE_SUBMISSION=${submission}" "$submission/job.pbs")
printf 'stage\tjob_id\naudit\t%s\n' "$job" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo "Report after completion: ${PWD}/outputs/generator_coverage_dpo/audit-${job}/report.md"
echo 'One CPU fitting-only coverage preference audit; all three generators, saved four-sample banks.'
echo 'No GPU training, new generation, paid judging or held-out preference construction.'
