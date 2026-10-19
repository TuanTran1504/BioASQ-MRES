#!/bin/bash
# New ten-sample fitting/validation banks; two matched rerankers; cached test95 rescore.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "${1:-}" == --resume-submission ]]; then
  [[ $# -eq 2 ]] || { echo 'Use --resume-submission PATH' >&2; exit 2; }
  submission=$(cd "$2" && pwd)
  # Never run two writers against the same resumable bank.
  for prior in "$submission"/jobs*.tsv; do
    [[ -f "$prior" ]] || continue
    while IFS=$'\t' read -r model stage job rest; do
      [[ "$job" =~ ^[0-9]+\.gadi-pbs$ ]] || continue
      if ! state=$(qstat -fx "$job" 2>&1); then
        if [[ "$state" != *'Unknown Job Id'* && "$state" != *'Unknown Job id'* ]]; then
          echo "Cannot verify whether $job is inactive: $state" >&2; exit 2
        fi
      fi
      if [[ "$state" =~ job_state[[:space:]]*=[[:space:]]*[QHREWTSB] ]]; then
        echo "Prior job $job is still active; do not resume concurrently." >&2; exit 2
      fi
    done < "$prior"
  done
  python3 "$submission/code/scripts/compactness_matched_rerankers.py" --mode check --submission "$submission"
else
  submission="${PWD}/outputs/compactness_matched_rerankers/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
  python3 ../scripts/compactness_matched_rerankers.py --mode prepare --repository "$(cd .. && pwd)" --submission "$submission" "$@"
fi
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
record="$submission/jobs-$(date -u +%Y%m%d-%H%M%S)-$$.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
full_jobs=()
for model in llama31 qwen3 ministral3; do
  smoke=$(qsub -N "${model}-rr10-smoke" -l walltime=01:00:00 \
    -v "RERANK_SUBMISSION=${submission},RERANK_MODEL=${model},SMOKE_TEST=1" "$submission/generate.pbs")
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  full=$(qsub -N "${model}-rr10-bank" -W "depend=afterok:${smoke}" \
    -v "RERANK_SUBMISSION=${submission},RERANK_MODEL=${model},SMOKE_TEST=0" "$submission/generate.pbs")
  printf '%s\tgeneration\t%s\t%s\n' "$model" "$full" "$smoke" | tee -a "$record"
  full_jobs+=("$full")
done
dependencies=$(IFS=:; echo "${full_jobs[*]}")
train=$(qsub -W "depend=afterok:${dependencies}" -v "RERANK_SUBMISSION=${submission}" "$submission/train.pbs")
printf 'ensemble\ttrain\t%s\t%s\n' "$train" "$dependencies" | tee -a "$record"
python3 "$submission/code/scripts/compactness_matched_rerankers.py" --mode record --submission "$submission" --job-id "$train"
score=$(qsub -W "depend=afterok:${train}" -v "RERANK_SUBMISSION=${submission}" "$submission/score.pbs")
printf 'ensemble\tscore\t%s\t%s\n' "$score" "$train" | tee -a "$record"
echo "Submission: $submission"
echo "Submission record: $record"
echo 'Three resumable GPU banks (greedy plus ten samples), one CPU job fitting two rerankers, one CPU test rescore.'
echo 'Fitting=1296; validation=144 with ten samples; test95 responses reused unchanged. No generator training or paid judging.'
