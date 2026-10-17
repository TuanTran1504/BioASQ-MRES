"""CPU scoring recovery preserves generation and changes exactly the filesystem fix."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts import recover_generator_feedback_scoring as recovery

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved(tmp_path):
    submission = tmp_path / "submitted"; (submission / "code/scripts").mkdir(parents=True)
    generation = tmp_path / "generation"; generation.mkdir()
    text = (ROOT / recovery.SCORER).read_text(encoding="utf-8").replace(recovery.CORRECTION, "")
    scorer = submission / "code" / recovery.SCORER
    scorer.write_text(text, encoding="utf-8")
    marker = submission / "code/scripts/marker.py"; marker.write_text("frozen other code")
    weights = tmp_path / "weights"; weights.write_bytes(b"unchanged adapter")
    recovery.write(submission / "data.json", {"pinned": "data"})
    recovery.write(submission / "prepared.json", {
        "code_hashes": {recovery.SCORER: recovery.digest(scorer), "scripts/marker.py": recovery.digest(marker)},
        "file_hashes": {"data.json": recovery.digest(submission / "data.json")},
        "external_hashes": {str(weights): recovery.digest(weights)}})
    runs, hashes = {}, {}
    for mode in ("greedy", "sampling10"):
        path = generation / mode; path.mkdir()
        (path / "generations.jsonl").write_text('saved raw responses\n')
        runs[mode] = mode
        hashes[mode] = {"generations.jsonl": recovery.digest(path / "generations.jsonl")}
    recovery.write(generation / "manifest.json", {
        "status": "complete", "smoke_test": False, "expected_questions": 160,
        "submission": str(submission), "prepared_sha256": recovery.digest(submission / "prepared.json"),
        "runs": runs, "arm_hashes": hashes})
    return submission, generation, tmp_path / "recovery"


def test_exact_single_correction_leaves_original_and_raw_files_unchanged(saved):
    submission, generation, destination = saved
    original = recovery.digest(submission / "code" / recovery.SCORER)
    raw_hashes = {str(p): recovery.digest(p) for p in generation.rglob("*") if p.is_file()}
    recovery.prepare(submission, generation, destination)
    state = recovery.validate_recovery(destination)
    expected = (submission / "code" / recovery.SCORER).read_text(encoding="utf-8").replace(
        recovery.ANCHOR, recovery.CORRECTION + recovery.ANCHOR)
    assert (destination / "code" / recovery.SCORER).read_text(encoding="utf-8") == expected
    assert recovery.digest(submission / "code" / recovery.SCORER) == original
    assert raw_hashes == {str(p): recovery.digest(p) for p in generation.rglob("*") if p.is_file()}
    assert state["GPU_jobs"] == 0 and state["generation_reused"]


@pytest.mark.parametrize("change", ["generation", "original_code", "recovery_code", "correction", "weights"])
def test_recovery_rejects_changed_inputs_or_additional_code(saved, change):
    submission, generation, destination = saved
    recovery.prepare(submission, generation, destination)
    if change == "generation":
        (generation / "greedy/generations.jsonl").write_text("changed responses")
    elif change == "original_code":
        (submission / "code/scripts/marker.py").write_text("changed original")
    elif change == "recovery_code":
        (destination / "code/scripts/marker.py").write_text("changed features")
    elif change == "correction":
        path = destination / "code" / recovery.SCORER
        path.write_text(path.read_text(encoding="utf-8") + "\n# additional edit\n", encoding="utf-8")
    else:
        (submission.parent / "weights").write_bytes(b"changed weights")
    with pytest.raises(ValueError):
        recovery.validate_recovery(destination)


def test_recovery_runs_copied_scorer_and_records_correction(saved, monkeypatch):
    submission, generation, destination = saved
    recovery.prepare(submission, generation, destination)
    output = destination / "comparison"
    argv_before = list(recovery.sys.argv)
    paths_before = list(recovery.sys.path)
    def execute(path, run_name):
        assert path == str(destination / "code" / recovery.SCORER) and run_name == "__main__"
        assert recovery.sys.argv[recovery.sys.argv.index("--generation") + 1] == str(generation)
        output.mkdir()
        recovery.write(output / "manifest.json", {"status": "complete"})
    monkeypatch.setattr(recovery.runpy, "run_path", execute)
    try:
        recovery.run(destination, output)
    finally:
        recovery.sys.argv = argv_before
        recovery.sys.path[:] = paths_before
    manifest = recovery.read(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["scoring_recovery"]["generation_reused"]


@pytest.mark.parametrize("fail", [True, False])
def test_launcher_submits_one_cpu_job_after_validation(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    submission = tmp_path / "original"; submission.mkdir()
    generation = tmp_path / "generation"; generation.mkdir()
    launcher = ROOT / "gadi_sft_8b_starter/scripts/submit_generator_feedback_rescore.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --recovery ]]; then
      shift; mkdir -p "$1"; printf frozen > "$1/job.pbs"; return 0
    fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '456.gadi-pbs\\n'; }
'''
    target = scripts / launcher.name
    target.write_text(launcher.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "calls.log"
    result = subprocess.run([executable, target.as_posix(), submission.as_posix(), generation.as_posix()],
        capture_output=True, text=True, env={**os.environ, "TEST_USER": "test", "TEST_FAIL": str(int(fail)),
                                           "TEST_LOG": log.as_posix()}, timeout=30)
    if fail:
        assert result.returncode and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert len(calls) == 1 and "SCORING_RECOVERY=" in calls[0] and "job.pbs" in calls[0]
        assert "SMOKE_TEST" not in calls[0] and "afterok" not in calls[0]
