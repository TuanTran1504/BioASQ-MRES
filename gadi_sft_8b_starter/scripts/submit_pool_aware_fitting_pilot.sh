#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="${1:-outputs/generator_form_designs/reviewed-20261029-v2}"
bundle="$(cd "$bundle" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_CACHE="/scratch/nl78/${USER}/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
submission="$PWD/outputs/pool_aware_fitting_pilot/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$bundle" == *','* || "$submission" == *$'\n'* || "$bundle" == *$'\n'* ]]; then
  echo 'Unsupported submission/package path' >&2; exit 2
fi
python3 ../scripts/pool_aware_fitting_pilot.py --mode prepare --repository "$(cd .. && pwd)" --bundle "$bundle" --submission "$submission"
record="$submission/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\toutput\n' > "$record"
generation_jobs=()
for model in llama31 qwen3 ministral3; do
  smoke=$(qsub -N "${model}-pool-smoke" -l walltime=01:00:00 -v "POOL_SUBMISSION=${submission},POOL_MODEL=${model},POOL_SMOKE=1" "$submission/generate.pbs")
  printf '%s\tsmoke\t%s\t\t%s\n' "$model" "$smoke" "$submission/$model/smoke-$smoke" | tee -a "$record"
  generation=$(qsub -N "${model}-pool-fit" -W "depend=afterok:${smoke}" -v "POOL_SUBMISSION=${submission},POOL_MODEL=${model},POOL_SMOKE=0" "$submission/generate.pbs")
  printf '%s\tgeneration\t%s\t%s\t%s\n' "$model" "$generation" "$smoke" "$submission/$model/generation-$generation" | tee -a "$record"
  generation_jobs+=("$generation")
done
depends=$(IFS=:; echo "${generation_jobs[*]}")
job=$(qsub -W "depend=afterok:${depends}" -v "POOL_SUBMISSION=${submission}" "$submission/score.pbs")
printf 'ensemble\tscore\t%s\t%s\t%s\n' "$job" "$depends" "$submission/comparison-$job" | tee -a "$record"
echo "Submission record: $record"
echo "Report after completion: $submission/comparison-$job/report.md"
echo 'Three GPU smoke/full chains plus one CPU scorer. 192 fitting questions; independent, correction-without-pool, pool-aware correction.'
echo '12 logical responses/question/condition; 1152 fresh full correction responses total. No DPO training, paid judging or held-out generation.'
