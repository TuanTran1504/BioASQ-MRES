#!/bin/bash
# One CPU job, dependent on outstanding max-five generation jobs.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export PYTHONUNBUFFERED=1
submission="${PWD}/outputs/sft_ensemble_reranker/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
python ../scripts/train_sft_ensemble_reranker.py --mode prepare \
  --repository "$(cd .. && pwd)" --submission "$submission"
dependency=$(cat "$submission/dependencies.txt")
options=()
if [[ -n "$dependency" ]]; then options=(-W "depend=afterok:${dependency}"); fi
job=$(qsub "${options[@]}" -v "RERANKER_SUBMISSION=${submission}" jobs/train_sft_ensemble_reranker.pbs)
printf 'stage\tjob_id\tdepends_on\ntrain\t%s\t%s\n' "$job" "$dependency" > "$submission/jobs.tsv"
echo "Submission record: $submission/jobs.tsv"
cat "$submission/jobs.tsv"
echo 'One CPU reranker training job. No GPU generation, paid API calls, or test evaluation.'
