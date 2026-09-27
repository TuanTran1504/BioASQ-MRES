#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  printf 'Install Python 3.10+ or set PYTHON_BIN to its executable.\n' >&2
  exit 1
fi

python_version="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_major="${python_version%%.*}"
python_minor="${python_version##*.}"
if (( python_major < 3 || (python_major == 3 && python_minor < 10) )); then
  printf 'Python 3.10+ is required; found %s.\n' "${python_version}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/src/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install pytest
"${VENV_DIR}/bin/python" -m pip check

cat <<EOF

Environment ready:
  ${VENV_DIR}/bin/python

Activate it with:
  source "${VENV_DIR}/bin/activate"

Run the test suite with:
  python -m pytest
EOF