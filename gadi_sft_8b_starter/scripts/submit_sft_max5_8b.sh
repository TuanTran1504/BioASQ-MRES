#!/bin/bash
# Fresh max-five SFT, reusing the validated teacher examples and pinned backbones.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DUAL_TRAINING_MATRIX=configs/matched_8b_sft_max5.json
export DUAL_SELECTION_CONFIG=configs/sft_max5_selection_8b.json
export DUAL_OUTPUT_ROOT=outputs/sft_max5_8b
exec bash scripts/submit_sft_dual_selection_8b.sh "$@"
