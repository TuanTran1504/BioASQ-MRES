#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="${1:?Usage: submit_generator_form_designs.sh REVIEWED_DESIGNS_DIRECTORY}"
bundle="$(cd "$bundle" && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_CACHE="/scratch/nl78/${USER}/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
submission="$PWD/outputs/generator_form_designs/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$bundle" == *','* || "$submission" == *$'\n'* || "$bundle" == *$'\n'* ]]; then
  echo 'Unsupported submission/package path' >&2; exit 2
fi
python3 scripts/run_generator_form_designs.py --mode prepare --repository "$(cd .. && pwd)" \
  --bundle "$bundle" --submission "$submission"
record="$submission/jobs.tsv"
printf 'model\tdesign\tstage\tjob_id\tdepends_on\toutput\n' > "$record"
validation_jobs=()
for model in llama31 qwen3 ministral3; do
  for design in model_specific shared; do
    child="$submission/$model/$design"
    smoke=$(qsub -N "${model}-form-smoke" -l walltime=01:00:00 -v "FORM_SUBMISSION=${child},SMOKE_TEST=1" "$child/train.pbs")
    printf '%s\t%s\tsmoke\t%s\t\t%s\n' "$model" "$design" "$smoke" "$child/smoke-$smoke" | tee -a "$record"
    train=$(qsub -N "${model}-form-dpo" -W "depend=afterok:${smoke}" -v "FORM_SUBMISSION=${child},SMOKE_TEST=0" "$child/train.pbs")
    printf '%s\t%s\ttrain\t%s\t%s\t%s\n' "$model" "$design" "$train" "$smoke" "$child/train-$train" | tee -a "$record"
    val=$(qsub -N "${model}-form-val144" -W "depend=afterok:${train}" -v "FORM_SUBMISSION=${child},FORM_TRAINING=${child}/train-${train}" "$child/validate.pbs")
    printf '%s\t%s\tvalidate\t%s\t%s\t%s\n' "$model" "$design" "$val" "$train" "$child/validation-$val" | tee -a "$record"
    validation_jobs+=("$val")
  done
done
depends=$(IFS=:; echo "${validation_jobs[*]}")
compare=$(qsub -W "depend=afterok:${depends}" -v "FORM_COMPARISON=${submission}" "$submission/compare.pbs")
printf 'ensemble\tboth\tcompare\t%s\t%s\t%s\n' "$compare" "$depends" "$submission/comparison-$compare" | tee -a "$record"
echo "Submission record: $record"
echo 'Submitted six size-matched DPO smoke/train/validation144 chains and one CPU comparison.'
echo '132 training pairs, 99 updates/run; own SFT reference per target. No paid calls or test95 evaluation.'
