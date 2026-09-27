#!/bin/bash
set -euo pipefail

# Run once on a Gadi login node. Override PYTHON_MODULE if your available
# Python module has a different version. This bundle was checked against the
# current Gadi module python3/3.12.13.
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.13}"
VENV_DIR="${VENV_DIR:-/scratch/nl78/${USER}/venvs/bioasq-8b}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/nl78/${USER}/pip_cache}"

module purge
module load "${PYTHON_MODULE}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel

# Install a CUDA-enabled PyTorch wheel before Unsloth. If this fails, inspect
# the Gadi GPU driver with nvidia-smi and choose the matching PyTorch wheel.
PIP_CACHE_DIR="${PIP_CACHE_DIR}" python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
PIP_CACHE_DIR="${PIP_CACHE_DIR}" python3 -m pip install -r requirements-gadi.txt

python3 - <<'PY'
import torch
import transformers
import trl
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('Unsloth import is deferred to the GPU smoke-test job.')
PY
