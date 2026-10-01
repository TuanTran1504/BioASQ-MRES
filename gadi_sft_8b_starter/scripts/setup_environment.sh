#!/bin/bash
set -euo pipefail

# Run once on a Gadi login node. Override PYTHON_MODULE if your available
# Python module has a different version. This bundle was checked against the
# current Gadi module python3/3.12.13.
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.12.13}"
VENV_DIR="${VENV_DIR:-/scratch/nl78/${USER}/venvs/bioasq-8b}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/nl78/${USER}/pip_cache}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"

module purge
module load "${PYTHON_MODULE}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel

# Gadi's gpuvolta nodes use V100 (compute capability 7.0). CUDA 13 PyTorch
# wheels no longer include that architecture, so use the CUDA 12.6 build and
# constrain downstream packages from replacing it with the PyPI CUDA 13 wheel.
PIP_CACHE_DIR="${PIP_CACHE_DIR}" python3 -m pip install \
  "torch==${PYTORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}" --force-reinstall
PIP_CACHE_DIR="${PIP_CACHE_DIR}" python3 -m pip install \
  -r requirements-gadi.txt -c constraints-gadi-v100.txt
python3 -m pip check

python3 - <<'PY'
from importlib.metadata import requires, version
for package in ("torch", "torchvision", "transformers", "trl", "unsloth"):
    print(f"{package}: {version(package)}")
torch_requirements = requires("torch") or []
cuda_requirements = [item for item in torch_requirements if "cuda" in item or "nvidia" in item]
print("torch CUDA requirements:", ", ".join(cuda_requirements))
if any("cu13" in item for item in cuda_requirements):
    raise RuntimeError("The installed PyTorch wheel still targets CUDA 13, which excludes Gadi V100 GPUs")
print('CUDA loading and the Unsloth import are deferred to the GPU smoke-test job.')
PY
