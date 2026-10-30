"""Reproduce missing duplicate metadata and rescore saved generations on CPU."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/"gadi_sft_8b_starter/scripts"))
from scripts import recover_pool_aware_scoring as recovery
from scripts import pool_aware_fitting_pilot as pilot


def response(answer): return json.dumps({"answers":[{"answer":answer,"relation_type":"original"}]})


@pytest.fixture
def original(tmp_path):
    if not shutil.which("java"): pytest.skip("Java required for full recovery integration")
    import joblib
    import numpy as np
    from run_generator_form_designs import snapshot
    from scripts.train_candidate_reranker_pilot import fit_components
    sub=tmp_path/"submission";code=snapshot(ROOT,sub/"code")
    script=sub/"code"/recovery.SCORER
    patched=script.read_text(encoding="utf-8")
    assert patched.count(recovery.NEW)==1
    script.write_text(patched.replace(recovery.NEW,recovery.OLD),encoding="utf-8",newline="\n")
    code[recovery.SCORER]=pilot.digest(script)
    ids=["gain","loss"]
    examples=[dict(question_id=q,question="Which symbol is the accepted answer?",gold_aliases=["Accepted-B"],
        snippets=[{"id":"s1","text":"The accepted symbol is Accepted-B."}]) for q in ids]
    config=pilot.read(ROOT/"gadi_sft_8b_starter/configs/pool_aware_fitting_pilot.json")
    config.update(questions=2,discovery_questions=1,audit_questions=1)
    baseline=[dict(question_id=q,model_key=m,split="train",draw=d,seed=pilot.sample_seed(q,d+1,config["seed"]),
        raw_response=response("Accepted-B" if q=="loss" and m=="llama31" and d==4 else "Other-A"),
        output_tokens=30,prompt_tokens=20) for q in ids for m in pilot.MODELS for d in range(1,5)]
    baseline[0]["raw_response"]=json.dumps({"answers":[{"answer":"Other-A","relation_type":"original"},{"answer":"other-a"}]})
    requests=pilot.make_requests(examples,baseline,config,"Repair from evidence.")
    plans={r["request_key"]:{"prompt_tokens":20,"prompt_ids_sha256":"tokens"} for r in requests}
    pilot.jsonl(sub/"scoring_examples.jsonl",examples);pilot.jsonl(sub/"baseline_responses.jsonl",baseline)
    pilot.write(sub/"question_split.json",{"gain":"discovery","loss":"audit"})
    (sub/"repair_prompt.txt").write_text("Repair from evidence.")
    for stage in ("generate","score"):
        shutil.copy2(ROOT/f"gadi_sft_8b_starter/jobs/{stage}_pool_aware_fitting_pilot.pbs",sub/f"{stage}.pbs")
    generation_pins={"generate.pbs":pilot.digest(sub/"generate.pbs")}
    for m in pilot.MODELS:
        pilot.jsonl(sub/f"inputs/{m}.jsonl",requests);pilot.write(sub/f"plans/{m}.json",plans)
        for n in (f"inputs/{m}.jsonl",f"plans/{m}.json"): generation_pins[n]=pilot.digest(sub/n)
    runtime=pilot.scorer.freeze(ROOT/"third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",sub)
    bundle=tmp_path/"bundle";pilot.write(bundle/"manifest.json",{})
    rr=tmp_path/"reranker";rr.mkdir()
    features=pilot.pool_rows(baseline,set(ids))
    labels=np.array([int(r["answer"]=="Accepted-B") for r in features])
    joblib.dump({"selector":"max5","components":fit_components(features,{e["question_id"]:e for e in examples},labels)},rr/"model.joblib")
    pilot.write(rr/"manifest.json",{"selectors":{"max5":{"model_path":"model.joblib"}}})
    p=dict(schema=pilot.SCHEMA,configuration=config,question_ids=ids,
        sources={m:dict(adapter=str(tmp_path/"adapter"),adapter_files_sha256={}) for m in pilot.MODELS},
        bundle=str(bundle),bundle_manifest_sha256=pilot.digest(bundle/"manifest.json"),
        feedback=dict(rerankers=str(rr),reranker_hashes={n:pilot.digest(rr/n) for n in ("model.joblib","manifest.json")}),
        code_hashes=code,generation_files_sha256=generation_pins,scorer_files_sha256=runtime,
        files_sha256={n:pilot.digest(sub/n) for n in ("scoring_examples.jsonl","baseline_responses.jsonl","question_split.json","repair_prompt.txt","generate.pbs","score.pbs")})
    pilot.write(sub/"prepared.json",p)
    jobs=[]
    for m in pilot.MODELS:
        path=sub/m/"generation-1";rows=[]
        for r in requests:
            rows.append({k:r[k] for k in ("question_id","arm","seed","request_key")} | dict(model_key=m,
                raw_response=response("Accepted-B" if m=="ministral3" and r["question_id"]=="gain" and r["arm"]=="repair_with_pool" else "Other-A"),
                output_tokens=30,generation_seconds=.1,**plans[r["request_key"]]))
        pilot.jsonl(path/"generations.jsonl",rows)
        pilot.write(path/"manifest.json",dict(status="complete",smoke_test=False,model_key=m,configuration=config,
            source=p["sources"][m],prepared_sha256=pilot.digest(sub/"prepared.json"),
            output_sha256={"generations.jsonl":pilot.digest(path/"generations.jsonl")}))
        jobs.append(dict(model=m,stage="generation",output=str(path)))
    with (sub/"jobs.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=tuple(jobs[0]),delimiter="\t");w.writeheader();w.writerows(jobs)
    return sub


def test_full_cpu_recovery_reproduces_failure_then_reuses_unchanged_requests_and_real_scorer(original,tmp_path):
    pins={p.relative_to(original).as_posix():pilot.digest(p) for p in original.rglob("*") if p.is_file()}
    bad=subprocess.run([sys.executable,str(original/"code"/recovery.SCORER),"--mode","score","--submission",str(original),
        "--output",str(tmp_path/"broken")],capture_output=True,text=True,timeout=120)
    assert bad.returncode!=0 and "KeyError: 'relation_type'" in bad.stderr,bad.stderr
    fixed=tmp_path/"recovery"
    recovery.prepare(original,fixed,ROOT)
    state=recovery.validate(fixed)
    assert state["generation_reused"] is True and state["GPU_jobs"]==0
    assert set(state["code_hashes"])==set(state["original_code_sha256"])
    assert [k for k in state["code_hashes"] if state["code_hashes"][k]!=state["original_code_sha256"][k]]==[recovery.SCORER]
    output=fixed/"comparison-2"
    result=subprocess.run([sys.executable,str(fixed/"runner.py"),"--mode","run","--recovery",str(fixed),"--output",str(output)],
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    report=pilot.read(output/"comparison_summary.json")
    t=report["paired_contrasts"]["repair_with_pool_minus_independent"]["coverage_transitions"]["coverage_full_pool"]
    assert t["gained_question_ids"]==["gain"] and t["lost_question_ids"]==["loss"]
    assert report["training_ready"] is False
    assert report["costs"]["independent"]["responses_per_question"]==12
    assert report["costs"]["independent"]["parser_diagnostics"]["duplicate_answer_surface"]==1
    manifest=pilot.read(output/"manifest.json")
    assert manifest["status"]=="complete" and manifest["scoring_recovery"]["generation_reused"]
    assert manifest["training"] is False and manifest["generation"] is False
    assert {p.relative_to(original).as_posix():pilot.digest(p) for p in original.rglob("*") if p.is_file() and "__pycache__" not in str(p)}=={
        k:v for k,v in pins.items() if "__pycache__" not in k}


def test_recovery_rejects_changed_inputs_extra_code_patches_and_incomplete_generation(original,tmp_path,monkeypatch):
    monkeypatch.setattr(recovery.subprocess,"run",lambda *a,**kw:None)
    fixed=tmp_path/"recovery";recovery.prepare(original,fixed,ROOT)
    recovery.validate(fixed)
    target=fixed/"code/scripts/prepare_answer_selector_data.py"
    target.write_text(target.read_text()+"\n# undeclared change\n")
    with pytest.raises(ValueError,match="Frozen"): recovery.validate(fixed)
    shutil.copy2(original/"code/scripts/prepare_answer_selector_data.py",target)
    raw=original/"qwen3/generation-1/generations.jsonl"
    text=raw.read_text();raw.write_text(text+"\n")
    with pytest.raises(ValueError,match="Frozen"): recovery.validate(fixed)
    raw.write_text(text,encoding="utf-8",newline="\n")
    manifest=original/"qwen3/generation-1/manifest.json"
    state=pilot.read(manifest);state["status"]="running";pilot.write(manifest,state)
    with pytest.raises(ValueError,match="complete"): recovery.frozen_inputs(original)


@pytest.mark.parametrize("mode",["missing","already_fixed"])
def test_correction_is_exact_and_idempotent_application_is_rejected(mode):
    text="nothing" if mode=="missing" else recovery.NEW
    with pytest.raises(ValueError,match="original"): recovery.corrected(text)


def test_recovery_job_is_cpu_only_and_preflight_runs_before_qsub():
    job=(ROOT/"gadi_sft_8b_starter/jobs/recover_pool_aware_scoring.pbs").read_text()
    assert "#PBS -q normal" in job and "ngpus" not in job and "--mode run" in job
    script=(ROOT/"gadi_sft_8b_starter/scripts/submit_pool_aware_rescore.sh").read_text()
    assert script.index("--mode prepare")<script.index("job=$(qsub")
    assert script.count("qsub")==1


@pytest.mark.parametrize("fail",[False,True])
def test_recovery_launcher_submits_only_one_cpu_job_after_successful_preflight(tmp_path,fail):
    bash=Path("C:/Program Files/Git/bin/bash.exe");exe=str(bash) if bash.exists() else shutil.which("bash")
    if not exe: pytest.skip("Bash unavailable")
    script=tmp_path/"scripts/submit_pool_aware_rescore.sh";script.parent.mkdir()
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
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '999.gadi-pbs\\n'; }
'''
    script.write_text((ROOT/"gadi_sft_8b_starter/scripts/submit_pool_aware_rescore.sh").read_text()
        .replace("${USER}","${TEST_USER}").replace("set -euo pipefail","set -euo pipefail\n"+stub),newline="\n")
    log=tmp_path/"submitted.log"
    result=subprocess.run([exe,script.as_posix(),original.as_posix()],capture_output=True,text=True,
        env={**os.environ,"TEST_USER":"test","TEST_LOG":log.as_posix(),"TEST_FAIL":str(int(fail))},timeout=30)
    if fail:
        assert result.returncode!=0 and not log.exists()
    else:
        assert result.returncode==0,result.stderr
        assert len(log.read_text().splitlines())==1
        assert "POOL_RECOVERY=" in log.read_text() and "generation" not in log.read_text()
        jobs=next(tmp_path.glob("outputs/pool_aware_fitting_pilot/scoring-recovery-*/jobs.tsv"))
        with jobs.open() as f: rows=list(csv.DictReader(f,delimiter="\t"))
        assert len(rows)==1 and rows[0]["stage"]=="score" and rows[0]["job_id"]=="999.gadi-pbs"
