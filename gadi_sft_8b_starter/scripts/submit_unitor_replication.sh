#!/bin/bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
export HF_HOME="/scratch/nl78/${USER}/hf_cache" HF_HUB_CACHE="/scratch/nl78/${USER}/hf_cache/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ $# == 0 ]]; then models=(llama31); else models=("$@"); fi
submission="$PWD/outputs/unitor_replication/submission-$(date -u +%Y%m%d-%H%M%S)-$$"
if [[ "$submission" == *','* || "$submission" == *$'\n'* ]]; then echo 'Unsupported submission path' >&2; exit 2; fi
# Source files are small and pinned. Backbone downloads are a separate preparation step.
bundle="$PWD/data/official_factoid_test95_v1"
if [[ ! -f "$bundle/manifest.json" ]]; then
  python3 ../scripts/prepare_paper_test95_bundle.py --output "$bundle"
fi
python3 ../scripts/run_unitor_replication.py prepare --submission "$submission" \
  --sources "$PWD/data/paper_replication_sources" --bundle "$bundle" \
  --models "${models[@]}" --download
record="$submission/jobs.tsv"
printf 'model\tstage\tjob_id\tdepends_on\n' > "$record"
for model in "${models[@]}"; do
  common="REPLICATION_SUBMISSION=${submission},REPLICATION_MODEL=${model}"
  smoke=$(qsub -N "${model}-ut3-smoke" -l walltime=01:00:00 -v "$common,REPLICATION_STAGE=train,SMOKE_TEST=1" "$submission/job.pbs")
  printf '%s\tsmoke\t%s\t\n' "$model" "$smoke" | tee -a "$record"
  train=$(qsub -N "${model}-ut3-train" -W "depend=afterok:${smoke}" -v "$common,REPLICATION_STAGE=train,SMOKE_TEST=0" "$submission/job.pbs")
  printf '%s\ttrain\t%s\t%s\n' "$model" "$train" "$smoke" | tee -a "$record"
  evaluation=$(qsub -N "${model}-ut3-test95" -l walltime=04:00:00 -W "depend=afterok:${train}" -v "$common,REPLICATION_STAGE=evaluate,SMOKE_TEST=0" "$submission/job.pbs")
  printf '%s\tevaluation\t%s\t%s\n' "$model" "$evaluation" "$train" | tee -a "$record"
done
echo "Submission record: $record"
echo 'UniTor released multitask SFT and UT3 test95 generation. No DPO, reranker or paid API calls.'
