"""Missing-library reproduction and CPU-only reuse of frozen generation artifacts."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import recover_generator_form_scoring as recovery
from scripts import generator_form_scorer_runtime as runtime


def jar_fixture(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("META-INF/MANIFEST.MF","Manifest-Version: 1.0\r\nClass-Path: lib/gson.jar \r\n lib/cli.jar\r\n\r\n")
    (path.parent/"lib").mkdir()
    for n in ("gson.jar","cli.jar"): (path.parent/"lib"/n).write_bytes(n.encode())


def test_full_scorer_dependencies_are_pinned_and_missing_or_changed_libraries_fail(tmp_path):
    source,bundle=tmp_path/"source",tmp_path/"bundle"
    jar=source/"scorer/BioASQEvaluation.jar";jar_fixture(jar)
    hashes={p.relative_to(source).as_posix():recovery.digest(p) for p in source.rglob("*") if p.is_file()}
    recovery.write(source/"prepared.json",{"files_sha256":hashes})
    recovery.write(bundle/"selector_audit_manifest.json",{"submission":str(source),"prepared_sha256":recovery.digest(source/"prepared.json")})
    assert runtime.source(bundle)==jar
    assert runtime.dependency_paths(jar)==["lib/gson.jar","lib/cli.jar"]
    output=tmp_path/"frozen"
    frozen=runtime.freeze(jar,output)
    assert len(frozen)==3 and (output/"scorer/lib/gson.jar").exists()
    (jar.parent/"lib/gson.jar").write_bytes(b"different")
    with pytest.raises(ValueError): runtime.source(bundle)
    (output/"scorer/lib/gson.jar").unlink()
    with pytest.raises(ValueError,match="dependency missing"): runtime.dependency_paths(output/"scorer/BioASQEvaluation.jar")


def test_real_java_requires_sibling_libraries_and_full_runtime_smoke_passes(tmp_path):
    if not shutil.which("java"): pytest.skip("Java required for evaluator runtime integration")
    dist=ROOT/"third_party/Evaluation-Measures/flat/BioASQEvaluation/dist"
    bad=tmp_path/"broken";bad.mkdir()
    shutil.copy2(dist/"BioASQEvaluation.jar",bad/"scorer.jar")
    with pytest.raises(RuntimeError,match="JsonReader"):
        runtime.smoke(bad/"scorer.jar")
    good=tmp_path/"complete runtime"
    runtime.freeze(dist/"BioASQEvaluation.jar",good)
    runtime.smoke(good/"scorer/BioASQEvaluation.jar")


def original_fixture(tmp_path):
    sub=tmp_path/"original";sub.mkdir()
    bundle=tmp_path/"bundle"
    source=tmp_path/"source"
    jar=source/"scorer/BioASQEvaluation.jar";jar_fixture(jar)
    recovery.write(source/"prepared.json",{"files_sha256":{p.relative_to(source).as_posix():recovery.digest(p)
        for p in source.rglob("*") if p.is_file()}})
    recovery.write(bundle/"selector_audit_manifest.json",dict(submission=str(source),prepared_sha256=recovery.digest(source/"prepared.json")))
    recovery.write(bundle/"manifest.json",{"fixture":"reviewed data package"})
    for name in recovery.PATCHES:
        target=sub/"code"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    code={n:recovery.digest(sub/"code"/n) for n in recovery.PATCHES}
    (sub/"compare.pbs").write_text("frozen comparison")
    jobs,pins=[],{}
    ids=[f"v{i}" for i in range(144)]
    for model in recovery.MODELS:
        for design in recovery.DESIGNS:
            child=sub/model/design
            recovery.write(child/"prepared.json",{"code_hashes":{}})
            pins[str(child)]=recovery.digest(child/"prepared.json")
            train,saved=child/"train-1",child/"validation-1"
            (train/"adapter").mkdir(parents=True)
            (train/"adapter/adapter_model.safetensors").write_text("weights")
            recovery.write(train/"manifest.json",dict(status="complete",smoke_test=False,phase="save_adapter",global_steps=99,
                trained_pair_counts={"train":132,"validation":96},prepared_sha256=pins[str(child)],
                adapter_files_sha256={"adapter_model.safetensors":recovery.digest(train/"adapter/adapter_model.safetensors")}))
            recovery.write(saved/"manifest.json",dict(status="failed",smoke_test=False,prepared_sha256=pins[str(child)],
                submission=str(child),training=str(train),pool_policy="verbatim surfaces",
                training_manifest_sha256=recovery.digest(train/"manifest.json"),question_ids=ids,
                error="RuntimeError: java.lang.NoClassDefFoundError: com/google/gson/stream/JsonReader"))
            for epoch in range(1,4):
                for mode,count in (("greedy",1),("sampling4",4)):
                    run=saved/f"epoch-{epoch}"/mode
                    recovery.write(run/"status.json",{"status":"complete"})
                    (run/"generations.jsonl").write_text("\n".join(json.dumps(dict(question_id=q,samples=[{}]*count)) for q in ids)+"\n")
            jobs.extend(dict(model=model,design=design,stage=stage,output=str(path)) for stage,path in (("train",train),("validate",saved)))
    with (sub/"jobs.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=tuple(jobs[0]),delimiter="\t");w.writeheader();w.writerows(jobs)
    recovery.write(sub/"prepared.json",dict(code_hashes=code,child_prepared_sha256=pins,bundle=str(bundle),
        bundle_manifest_sha256=recovery.digest(bundle/"manifest.json"),
        comparison_job_sha256=recovery.digest(sub/"compare.pbs"),scorer_sha256=recovery.digest(jar)))
    return sub


def test_recovery_preparation_preserves_originals_pins_responses_and_allows_only_declared_code_changes(tmp_path,monkeypatch):
    sub=original_fixture(tmp_path)
    before={p.relative_to(sub).as_posix():recovery.digest(p) for p in sub.rglob("*") if p.is_file()}
    monkeypatch.setattr(runtime,"smoke",lambda _:None)
    output=tmp_path/"recovery"
    recovery.prepare(sub,output)
    state=recovery.validate(output)
    assert len(state["panels"])==6 and state["GPU_jobs"]==0 and state["new_generation"] is False
    assert set(state["scorer_files_sha256"])=={"scorer/BioASQEvaluation.jar","scorer/lib/gson.jar","scorer/lib/cli.jar"}
    assert {p.relative_to(sub).as_posix():recovery.digest(p) for p in sub.rglob("*") if p.is_file()}==before
    target=sub/"llama31/model_specific/validation-1/epoch-1/greedy/generations.jsonl"
    target.write_text("changed")
    with pytest.raises(ValueError,match="Frozen recovery input changed"): recovery.validate(output)


@pytest.mark.parametrize("change",["missing_draw","wrong_error","failed_training","duplicate_question"])
def test_incomplete_or_unrelated_failures_cannot_start_cpu_recovery(tmp_path,change):
    sub=original_fixture(tmp_path)
    root=sub/"qwen3/shared"
    if change=="missing_draw":
        (root/"validation-1/epoch-3/sampling4/generations.jsonl").write_text("\n")
    elif change=="wrong_error":
        p=root/"validation-1/manifest.json";d=recovery.read(p);d["error"]="Out of memory";recovery.write(p,d)
    elif change=="duplicate_question":
        p=root/"validation-1/manifest.json";d=recovery.read(p);d["question_ids"][-1]=d["question_ids"][0];recovery.write(p,d)
    else:
        p=root/"train-1/manifest.json";d=recovery.read(p);d["status"]="failed";recovery.write(p,d)
    with pytest.raises(ValueError): recovery.original_inputs(sub)


def test_recovery_cannot_change_checkpoint_selection_or_pool_rules(tmp_path):
    original,corrected=tmp_path/"original",tmp_path/"corrected"
    for root in (original,corrected):
        for name in recovery.PATCHES:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    recovery.verify_corrections(original,corrected)
    path=corrected/recovery.PATCHES[1]
    path.write_text(path.read_text().replace('r["ensemble_full_pool_coverage"], r["focal_full_pool_coverage"]','r["focal_full_pool_coverage"], r["ensemble_full_pool_coverage"]'))
    with pytest.raises(ValueError,match="pool/coverage rules"): recovery.verify_corrections(original,corrected)


@pytest.mark.parametrize("fail",[False,True])
def test_launcher_submits_exactly_one_cpu_recovery_after_preflight(tmp_path,fail):
    bash=Path("C:/Program Files/Git/bin/bash.exe")
    executable=str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    script=tmp_path/"scripts/submit_generator_form_rescore.sh";script.parent.mkdir()
    original=tmp_path/"original";original.mkdir()
    stub='''
module() { :; }
source() { :; }
python3() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --recovery ]]; then shift; mkdir -p "$1"; return 0; fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '900.gadi-pbs\\n'; }
'''
    script.write_text((ROOT/"gadi_sft_8b_starter/scripts/submit_generator_form_rescore.sh").read_text()
        .replace("${USER}","${TEST_USER}").replace("set -euo pipefail","set -euo pipefail\n"+stub),newline="\n")
    log=tmp_path/"submitted.log"
    result=subprocess.run([executable,script.as_posix(),original.as_posix()],capture_output=True,text=True,
        env={**os.environ,"TEST_USER":"test","TEST_LOG":log.as_posix(),"TEST_FAIL":str(int(fail))},timeout=30)
    if fail: assert result.returncode!=0 and not log.exists()
    else:
        assert result.returncode==0,result.stderr
        assert len(log.read_text().splitlines())==1 and "FORM_RECOVERY=" in log.read_text()
        job=(ROOT/"gadi_sft_8b_starter/jobs/recover_generator_form_scoring.pbs").read_text()
        assert "#PBS -q normal" in job and "ngpus" not in job
