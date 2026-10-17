#!/bin/bash
# Reuse all SFT dev archives; generate only the new Llama continuation.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
training="${1:-outputs/generator_feedback_dpo_training/llama31-compactness-train-181456254.gadi-pbs}"
training="$(cd "$training" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HOME}/hub" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
submission="${PWD}/outputs/generator_feedback_dpo_evaluation/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
python scripts/run_generator_feedback_evaluation.py --mode prepare --repository "$(cd .. && pwd)" \
  --training "$training" --submission "$submission"
record="$submission/jobs.tsv"
printf 'stage\tjob_id\tdepends_on\n' > "$record"
smoke=$(qsub -N llama-compact-eval-smoke -l walltime=01:00:00 \
  -v "EVALUATION_SUBMISSION=${submission},SMOKE_TEST=1" "$submission/generate.pbs")
printf 'smoke\t%s\t\n' "$smoke" | tee -a "$record"
full=$(qsub -N llama-compact-eval-full -W "depend=afterok:${smoke}" \
  -v "EVALUATION_SUBMISSION=${submission},SMOKE_TEST=0" "$submission/generate.pbs")
printf 'generation\t%s\t%s\n' "$full" "$smoke" | tee -a "$record"
generation="${PWD}/outputs/generator_feedback_dpo_evaluation/generation-${full}"
score=$(qsub -W "depend=afterok:${full}" \
  -v "EVALUATION_SUBMISSION=${submission},GENERATION_DIRECTORY=${generation}" "$submission/score.pbs")
printf 'score\t%s\t%s\n' "$score" "$full" | tee -a "$record"
echo "Submission record: $record"
echo 'Submitted DPO greedy/ten-sample dev160 generation and frozen-reranker scoring; six SFT archives reused. No training, test evaluation or paid judging.'
