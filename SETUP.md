# Setup

This repository can be moved to a new machine without changing any project paths.

## Linux or macOS

Install Python 3.10 or newer, clone or copy the project, then run:

```bash
cd "Task-Structured Counterfactual Preference Mining"
./setup.sh
source .venv/bin/activate
```

The script installs the packages listed in `src/requirements.txt` and `pytest` into a local `.venv`. It is safe to run again after copying the project to another location. To use a particular Python executable:

```bash
PYTHON_BIN=/path/to/python3.11 ./setup.sh
```

## Verify the checkout

Run the tests and a lightweight import check:

```bash
python -m pytest
python -m compileall -q src scripts/run_command_pipeline.py
```

Training and generation require a compatible PyTorch/CUDA installation and locally available model/data artifacts. The dependency file installs the project’s current ML stack; GPU-specific wheel selection may need to follow the destination machine’s CUDA version.

For offline model use:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UNSLOTH_DISABLE_STATISTICS=1
```