#!/bin/bash
# Three fixed test95 ensembles; one epoch-one generation per model, six SFT archives reused.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
submission="${PWD}/outputs/compactness_test95/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python3 scripts/run_compactness_test95.py --mode prepare --repository "$(cd .. && pwd)" --submission "$submission" "$@"
record="$submission/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
full_jobs=()
for model in llama31 qwen3 ministral3; do
  smoke=$(qsub -N "${model}-compact95-smoke" -l walltime=01:00:00 \
    -v "EVALUATION_SUBMISSION=${submission},EVALUATION_MODEL=${model},SMOKE_TEST=1" "$submission/generate.pbs")
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full=$(qsub -N "${model}-compact95-full" -W "depend=afterok:${smoke}" \
    -v "EVALUATION_SUBMISSION=${submission},EVALUATION_MODEL=${model},SMOKE_TEST=0" "$submission/generate.pbs")
  printf '%s\tgeneration\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
  full_jobs+=("$full")
done
python3 "$submission/code/gadi_sft_8b_starter/scripts/run_compactness_test95.py" --mode record --submission "$submission"
dependencies=$(IFS=:; echo "${full_jobs[*]}")
score=$(qsub -W "depend=afterok:${dependencies}" -v "EVALUATION_SUBMISSION=${submission}" "$submission/score.pbs")
printf 'ensemble\tscore\t%s\t%s\n' "$score" "$dependencies" | tee -a "$record"
echo "Submission record: $record"
echo 'Submitted three smoke/full epoch-one generation chains and one CPU scorer for test95.'
echo 'Baseline, strict 0/1/0 and alternative 1/1/1 are fixed. Frozen reranker and SFT archives reused; no training or paid judging.'
