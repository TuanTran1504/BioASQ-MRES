"""Test PBS wiring using mocks; never submit a real GPU job."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("mode,models,fail,expected", [
    ("generate", [], False, 6), ("generate", ["ministral3"], False, 2),
    ("generate", ["qwen3", "qwen3"], False, 0), ("generate", [], True, 0),
    ("train", [], False, 6), ("train", [], True, 0),
])
def test_validate_all_sources_before_submission(tmp_path, mode, models, fail, expected):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    root = Path(__file__).resolve().parents[1]
    source = root / "gadi_sft_8b_starter/scripts/submit_expansion_dpo_8b.sh"
    text = source.read_text().replace('source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"', ':')
    mocks = r'''
module() { :; }
python() { printf 'validate %s\n' "$*" >> "$MOCK_LOG"; [[ "$MOCK_FAIL" == 0 ]]; }
qsub() {
  printf 'qsub %s\n' "$*" >> "$MOCK_LOG"
  number="$(cat "$MOCK_COUNT")"
  printf '%s\n' "$((number + 1))" > "$MOCK_COUNT"
  printf '%s.gadi-pbs\n' "$number"
}
'''
    text = text.replace("set -euo pipefail", "set -euo pipefail\n" + mocks)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / source.name
    script.write_text(text, newline="\n")
    log, count = tmp_path / "log", tmp_path / "count"
    count.write_text("1\n")
    preferences = tmp_path / "preferences"
    preferences.mkdir()
    arguments = [mode, *([preferences.as_posix()] if mode == "train" else []), *models]
    env = {**os.environ, "USER": "mock", "MOCK_FAIL": str(int(fail)),
           "MOCK_LOG": log.as_posix(), "MOCK_COUNT": count.as_posix()}
    result = subprocess.run([bash, script.as_posix(), *arguments], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    lines = log.read_text().splitlines() if log.exists() else []
    jobs = [line for line in lines if line.startswith("qsub ")]
    assert len(jobs) == expected, result.stderr
    assert result.returncode == (0 if expected else 1), result.stderr
    if jobs:
        assert next(i for i, line in enumerate(lines) if line.startswith("qsub ")) == len(models or data_models())
        for i in range(0, expected, 2):
            assert f"DPO_MODE={mode}" in jobs[i]
            assert "SMOKE_TEST=1" in jobs[i] and "walltime=02:00:00" in jobs[i]
            assert "SMOKE_TEST=0" in jobs[i + 1] and f"depend=afterok:{i + 1}.gadi-pbs" in jobs[i + 1]
            assert ("PREFERENCES=" in jobs[i]) == (mode == "train")


def data_models():
    return ("llama31", "qwen3", "ministral3")
