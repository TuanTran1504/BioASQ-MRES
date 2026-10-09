"""Exercise submission/dependency wiring without invoking a real scheduler."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("models,fail_validation,expected_jobs", [
    ([], False, 4), (["llama31"], False, 2), (["ministral3"], False, 0),
    (["qwen3", "qwen3"], False, 0), ([], True, 0),
])
def test_only_new_expansion_jobs_are_submitted_after_validation(tmp_path, models, fail_validation, expected_jobs):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = root / "gadi_sft_8b_starter/scripts/submit_expansion_sampling_8b.sh"
    text = source.read_text().replace('source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"', ':')
    mocks = r'''
module() { :; }
python() {
  printf 'validate %s\n' "$*" >> "$MOCK_LOG"
  [[ "$MOCK_FAIL" == 0 ]]
}
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
    log, counter = tmp_path / "mock.log", tmp_path / "mock.count"
    counter.write_text("1\n")
    env = {**os.environ, "USER": "mock-user", "MOCK_LOG": log.as_posix(),
           "MOCK_COUNT": counter.as_posix(), "MOCK_FAIL": str(int(fail_validation))}
    result = subprocess.run([bash, script.as_posix(), *models], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    lines = log.read_text().splitlines() if log.exists() else []
    jobs = [line for line in lines if line.startswith("qsub ")]
    assert len(jobs) == expected_jobs, result.stderr
    assert result.returncode == (0 if expected_jobs else 1), result.stderr
    if not jobs:
        return
    assert all("EXPANSION_SAMPLING_ONLY=1" in line for line in jobs)
    assert all("jobs/evaluate_matched_8b_sft.pbs" in line for line in jobs)
    assert all("--expansion-sampling-only" in line for line in lines if line.startswith("validate "))
    first_job = next(i for i, line in enumerate(lines) if line.startswith("qsub "))
    assert first_job == len(models or ["llama31", "qwen3"])
    for i in range(0, expected_jobs, 2):
        assert "SMOKE_TEST=1" in jobs[i] and "walltime=02:00:00" in jobs[i]
        assert "SMOKE_TEST=0" in jobs[i + 1] and f"depend=afterok:{i + 1}.gadi-pbs" in jobs[i + 1]
