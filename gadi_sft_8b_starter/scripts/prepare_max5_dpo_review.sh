#!/bin/bash
# Score completed banks with official matching and package them for local judging.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record="${1:-$(ls -t outputs/max5_dpo_banks_8b/submission-*.tsv | head -n 1)}"
output="${2:-outputs/max5_dpo_banks_8b/review-v1}"
module purge
module load "${PYTHON_MODULE:-python3/3.12.13}"
source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"
banks=()
while IFS=$'\t' read -r model stage job dependency; do
  if [[ "$stage" == full ]]; then banks+=("outputs/max5_dpo_banks_8b/${model}-generate-${job}"); fi
done < "$record"
if (( ${#banks[@]} != 3 )); then echo 'Require a three-model full-bank submission record' >&2; exit 2; fi
python scripts/max5_dpo_data.py --mode review --banks "${banks[@]}" --output "$output"
archive="/scratch/nl78/${USER}/max5-dpo-review-v1.tar.gz"
tar -czf "$archive" -C "$output" .
echo "Portable review archive: $archive"
echo 'Generation and official matching are complete; semantic labels are still pending. No paid calls were made.'
