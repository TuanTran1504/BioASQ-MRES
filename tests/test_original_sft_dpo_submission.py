import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("mode,strategy,fail,expected", [("generate", "mixed", False, 6),
    ("generate", "mixed", True, 0), ("train", "mixed", False, 9), ("train", "all", False, 27),
    ("train", "all", True, 0)])
def test_original_dpo_dependencies_and_preflight(tmp_path, mode, strategy, fail, expected):
    root = Path(__file__).resolve().parents[1]
    path = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(path) if path.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = root / "gadi_sft_8b_starter/scripts/submit_original_sft_dpo_8b.sh"
    text = source.read_text().replace("${USER}", "${TEST_USER}")
    stub = r'''
module() { :; }
source() { :; }
python() { printf 'validate %s\n' "$*" >> "$TEST_LOG"; [[ "$TEST_FAIL" == 0 ]]; }
qsub() {
  printf 'qsub %s\n' "$*" >> "$TEST_LOG"
  number=$(cat "$TEST_COUNT")
  printf '%s' "$((number+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\n' "$number"
}
'''
    script = scripts / source.name
    script.write_text(text.replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    count, log = tmp_path / "count", tmp_path / "log"
    count.write_text("1")
    preferences = tmp_path / "preferences"
    preferences.mkdir()
    arguments = [mode, *([preferences.as_posix(), strategy] if mode == "train" else [])]
    result = subprocess.run([bash, script.as_posix(), *arguments], text=True, capture_output=True,
        env={**os.environ, "TEST_USER": "test", "TEST_FAIL": str(int(fail)),
             "TEST_LOG": log.as_posix(), "TEST_COUNT": count.as_posix()})
    lines = log.read_text().splitlines() if log.exists() else []
    jobs = [r for r in lines if r.startswith("qsub ")]
    assert len(jobs) == expected, result.stderr
    if not fail:
        assert result.returncode == 0, result.stderr
        first = next(i for i, row in enumerate(lines) if row.startswith("qsub "))
        assert first == (9 if strategy == "all" else 3)
        stride = 3 if mode == "train" else 2
        for i in range(0, len(jobs), stride):
            assert "SMOKE_TEST=1" in jobs[i]
            assert f"depend=afterok:{i+1}.gadi-pbs" in jobs[i+1]
            if mode == "train":
                assert f"depend=afterok:{i+2}.gadi-pbs" in jobs[i+2]
                assert "DPO_TRAINING_RUN=" in jobs[i+2]
            assert "original_sft_dpo_8b.json" in jobs[i]
    else:
        assert result.returncode != 0
