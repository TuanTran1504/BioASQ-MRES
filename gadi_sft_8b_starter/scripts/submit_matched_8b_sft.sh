#!/bin/bash
# Submit six paired training conditions, each gated by its training/generation smoke.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models=()
formulations=(original expansion)
execution_mode=default
while (( $# )); do
  case "$1" in
    --formulation)
      if (( $# < 2 )); then echo '--formulation requires original or expansion' >&2; exit 2; fi
      case "$2" in original|expansion) formulations=("$2") ;; *) echo 'Invalid formulation' >&2; exit 2 ;; esac
      shift 2 ;;
    --eager) execution_mode=eager; shift ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) models+=("$1"); shift ;;
  esac
done
if (( ${#models[@]} == 0 )); then models=(llama31 qwen3 ministral3); fi
declare -A seen_models=()
for model in "${models[@]}"; do
  case "$model" in llama31|qwen3|ministral3) ;; *) echo "Unknown model: $model" >&2; exit 2 ;; esac
  if [[ -n "${seen_models[$model]:-}" ]]; then echo "Duplicate model: $model" >&2; exit 2; fi
  seen_models[$model]=1
done
if [[ "$execution_mode" == eager && "${models[*]}" != ministral3 ]]; then
  echo '--eager is restricted to a Ministral-only retry' >&2; exit 2
fi
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/prepare_matched_8b_sft.py --mode build
mkdir -p outputs/matched_8b_sft
submission="$(date -u +%Y%m%d-%H%M%S)-$$"
pins="${PWD}/outputs/matched_8b_sft/base-pins-${submission}.json"
# All data/cache checks finish before the first qsub. Both arms share each pin.
python scripts/prepare_matched_8b_sft.py --mode pin-cache --pin-output "$pins" --models "${models[@]}"
record="outputs/matched_8b_sft/submission-${submission}.tsv"
printf 'model\tformulation\tstage\tjob_id\tdepends_on\texecution_mode\n' > "$record"
echo "Submission record: $record"
for model in "${models[@]}"; do
  for formulation in "${formulations[@]}"; do
    smoke="$(qsub -N "${model}-${formulation}-smoke" -l walltime=02:00:00 \
      -v "MODEL_KEY=${model},SFT_FORMULATION=${formulation},BASE_PINS=${pins},SMOKE_TEST=1,EXECUTION_MODE=${execution_mode}" \
      jobs/train_matched_8b_sft.pbs)"
    printf '%s\t%s\tsmoke\t%s\t\t%s\n' "$model" "$formulation" "$smoke" "$execution_mode" | tee -a "$record"
    full="$(qsub -N "${model}-${formulation}-sft" -W "depend=afterok:${smoke}" \
      -v "MODEL_KEY=${model},SFT_FORMULATION=${formulation},BASE_PINS=${pins},SMOKE_TEST=0,EXECUTION_MODE=${execution_mode}" \
      jobs/train_matched_8b_sft.pbs)"
    printf '%s\t%s\tfull\t%s\t%s\t%s\n' "$model" "$formulation" "$full" "$smoke" "$execution_mode" | tee -a "$record"
  done
done
echo "Submitted ${formulations[*]} SFT (${execution_mode}); full jobs require successful training and generation smoke."
