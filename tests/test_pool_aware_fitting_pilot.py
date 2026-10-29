"""Gold-blind pool conditioning and fixed-budget fitting-only generation."""
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/"gadi_sft_8b_starter/scripts"))
from scripts import pool_aware_fitting_pilot as pilot
import run_pool_aware_fitting_pilot as worker


def config(): return pilot.read(ROOT/"gadi_sft_8b_starter/configs/pool_aware_fitting_pilot.json")
def response(answers):
    return json.dumps({"answers":[{"answer":a,"relation_type":"original" if i==0 else "synonym"} for i,a in enumerate(answers)]})
def examples(ids):
    return [{"question_id":q,"question":"Which symbol?","snippets":[{"id":"s1","text":"Evidence describes Accepted-B."}],
             "gold_aliases":["Accepted-B"],"review_notes":"NEVER INCLUDE THIS"} for q in ids]
def baseline(ids):
    return [dict(question_id=q,model_key=m,split="train",draw=d,seed=pilot.sample_seed(q,d+1,3407),
        raw_response=response(["Other-A"]),output_tokens=20,prompt_tokens=10,
        prompt=[{"role":"system","content":"original"},{"role":"user","content":"question/evidence"}],response_id=f"{q}-{m}-{d}")
        for q in ids for m in pilot.MODELS for d in range(1,5)]
def plan(requests): return {r["request_key"]:{"prompt_tokens":20,"prompt_ids_sha256":"tokens"} for r in requests}
def generated(request,model,raw,plans):
    return {k:request[k] for k in ("request_key","question_id","arm","seed")} | dict(model_key=model,raw_response=raw,
        output_tokens=30,generation_seconds=.1,**plans[request["request_key"]])


def test_panel_is_id_selected_without_labels_and_excludes_protected_and_old_preference_holdout():
    fitting=[f"q{i}" for i in range(300)]
    protected={"validation144":["q0"],"dev160":["q1"],"test95":["q2"]}
    preference={"q3":"inner_validation","q4":"train"}
    ids,split=pilot.select_questions(fitting,protected,preference,config())
    assert len(ids)==len(set(ids))==192
    assert not set(ids)&{"q0","q1","q2","q3"}
    assert sum(s=="discovery" for s in split.values())==160
    assert sum(s=="audit" for s in split.values())==32
    assert pilot.select_questions(list(reversed(fitting)),protected,preference,config())==(ids,split)


def test_pool_uses_exact_surfaces_from_first_nine_draws_without_labels_or_fourth_draw():
    rr=baseline(["q"])
    rr[0]["raw_response"]=response(["HER2","HER-2"," HER2 "])
    for r in rr:
        if r["draw"]==4: r["raw_response"]=response(["FOURTH-ONLY"])
    pool=pilot.initial_pool(rr,"q",3407)
    assert {r["answer"] for r in pool}=={"HER2","HER-2"," HER2 ","Other-A"}
    assert pilot.initial_pool(list(reversed(rr)),"q",3407)==pool
    requests=pilot.make_requests(examples(["q"]),rr,config(),"repair prompt")
    assert len(requests)==2 and requests[0]["seed"]==requests[1]["seed"]==pilot.sample_seed("q",5,3407)
    a,b=(json.loads(r["messages"][1]["content"]) for r in requests)
    assert a["previous_proposals"]==[] and b["previous_proposals"]==pool
    assert a["question"]==b["question"] and a["snippets"]==b["snippets"]
    assert set(a)==set(b)=={"question","snippets","previous_proposals"}
    assert "NEVER INCLUDE THIS" not in json.dumps(requests) and "gold_aliases" not in json.dumps(requests)


def test_arm_assembly_replaces_fourth_responses_instead_of_adding_a_thirteenth():
    rr=baseline(["q"]);requests=pilot.make_requests(examples(["q"]),rr,config(),"repair")
    pp=plan(requests)
    rows={(m,"q",r["arm"]):generated(r,m,response(["Accepted-B"]),pp) for m in pilot.MODELS for r in requests}
    arms=pilot.assemble_arms(rr,rows,["q"])
    for name,records in arms.items():
        assert len(records)==12
        assert {(r["model_key"],r["draw"]) for r in records}=={(m,d) for m in pilot.MODELS for d in range(1,5)}
        assert sum(r["draw"]<=3 for r in records)==9
        pilot.pool_rows(records,{"q"})
    assert all(r["raw_response"]==response(["Other-A"]) for r in arms["independent"])
    assert sum(r["raw_response"]==response(["Accepted-B"]) for r in arms["focal_qwen3"])==1
    assert sum(r["raw_response"]==response(["Accepted-B"]) for r in arms["repair_with_pool"])==3
    with pytest.raises(ValueError,match="Missing"): pilot.pool_rows(rr[:-1],{"q"})
    with pytest.raises(ValueError,match="Invalid"): pilot.pool_rows(rr+[rr[0]],{"q"})


def test_failed_draws_count_and_pool_frequency_uses_four_requested_draws():
    rr=baseline(["q"]);rr[0]["raw_response"]=""
    pooled=pilot.pool_rows(rr,{"q"})
    assert len(pooled)==1 and pooled[0]["sample_frequencies"]["llama31"]==.75
    with pytest.raises(ValueError,match="Incomplete"): pilot.check_baseline(rr[:-1],{"q"},config())


def test_collect_never_resamples_parse_failures_and_resume_preserves_saved_draws():
    requests=pilot.make_requests(examples(["q"]),baseline(["q"]),config(),"repair")
    pp=plan(requests);calls=[]
    def produce(r):
        calls.append(r["request_key"])
        return {"raw_response":"", "output_tokens":1,"generation_seconds":.1,**pp[r["request_key"]]}
    saved=[]
    rows=worker.collect(requests,pp,"qwen3",config(),produce,lambda rr:saved.append(copy.deepcopy(rr)))
    assert len(rows)==len(calls)==2 and all(r["raw_response"]=="" for r in rows)
    again=worker.collect(requests,pp,"qwen3",config(),lambda r:pytest.fail("completed request resampled"),lambda _:None,rows)
    assert again==rows
    bad=copy.deepcopy(rows);bad[0]["seed"]+=1
    with pytest.raises(ValueError,match="differs"): worker.collect(requests,pp,"qwen3",config(),produce,lambda _:None,bad)


def test_longest_smoke_keeps_both_conditions_and_rejects_gold_fields(tmp_path):
    requests=pilot.make_requests(examples(["short","long","middle"]),baseline(["short","long","middle"]),config(),"repair")
    pp=plan(requests)
    for r in requests: pp[r["request_key"]]["prompt_tokens"]={"short":1,"long":30,"middle":20}[r["question_id"]]
    pilot.jsonl(tmp_path/"inputs/llama31.jsonl",requests);pilot.write(tmp_path/"plans/llama31.json",pp)
    state={"question_ids":["short","long","middle"],"configuration":config()}
    chosen,_=worker.selected_requests(tmp_path,state,"llama31",True)
    assert len(chosen)==4 and {r["question_id"] for r in chosen}=={"long","middle"}
    r=requests[0];payload=json.loads(r["messages"][1]["content"]);payload["gold_aliases"]=["Accepted-B"]
    r["messages"][1]["content"]=json.dumps(payload);r["request_key"]=pilot.identity({k:v for k,v in r.items() if k!="request_key"})
    pilot.jsonl(tmp_path/"inputs/llama31.jsonl",requests)
    pilot.write(tmp_path/"plans/llama31.json",plan(requests))
    with pytest.raises(ValueError,match="context fields"): worker.selected_requests(tmp_path,state,"llama31",False)


@pytest.mark.parametrize("overlong",[False,True])
def test_preparation_pins_gold_blind_inputs_complete_scorer_and_common_nontruncated_context(tmp_path,monkeypatch,overlong):
    repo,bundle,sub=(tmp_path/n for n in ("repo","bundle","submission"))
    fitting=[f"q{i:04d}" for i in range(1296)]
    ex=examples(fitting);raw=baseline(fitting)
    sources={m:dict(adapter=str(tmp_path/m),base_snapshot=m,model_loader="fast_language_model",chat_template_kwargs={},adapter_files_sha256={}) for m in pilot.MODELS}
    monkeypatch.setattr(pilot,"source_inputs",lambda *a:({e["question_id"]:e for e in ex},raw,
        {q:["Accepted-B"] for q in fitting},sources,{"banks":{},"rerankers":"unused","reranker_hashes":{}}))
    pilot.write(bundle/"manifest.json",{})
    pilot.write(bundle/"fitting_ids.json",fitting)
    pilot.write(bundle/"protected_ids.json",{"validation144":["v"],"dev160":["d"],"test95":["t"]})
    pilot.write(bundle/"review/question_split.json",{q:"inner_validation" for q in fitting[:40]})
    conf=repo/"config.json";pilot.write(conf,config())
    prompt=repo/"gadi_sft_8b_starter/prompts/pool_aware_repair_v1.txt";prompt.parent.mkdir(parents=True);prompt.write_text("repair prompt")
    for stage in ("generate","score"):
        p=repo/f"gadi_sft_8b_starter/jobs/{stage}_pool_aware_fitting_pilot.pbs";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(stage)
    jar=ROOT/"third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    monkeypatch.setattr(pilot.scorer,"source",lambda _:jar)
    monkeypatch.setattr(pilot.scorer,"smoke",lambda _:None)
    import run_base_original_prompt as inference
    import run_generator_form_designs as original
    class NativeTokenizer:
        def apply_chat_template(self,*a,**kw): assert kw["tokenize"] is False;return "baseline"
        def __call__(self,*a,**kw): return {"input_ids":[1]*10}
    monkeypatch.setattr(inference,"load_text_tokenizer",lambda _:NativeTokenizer())
    monkeypatch.setattr(inference,"native_context",lambda _:32768)
    monkeypatch.setattr(original,"snapshot",lambda *a:{})
    def tokens(tok,requests,source):
        return {r["request_key"]:{"prompt_tokens":(16384 if overlong else 9000) if source["base_snapshot"]=="qwen3" and r["arm"]=="repair_with_pool" else 10,
            "prompt_ids_sha256":"tokens"} for r in requests}
    monkeypatch.setattr(pilot,"token_plan",tokens)
    if overlong:
        with pytest.raises(ValueError,match="refusing truncation"): pilot.prepare(repo,bundle,sub,conf)
        assert not sub.exists()
    else:
        pilot.prepare(repo,bundle,sub,conf)
        p=pilot.read(sub/"prepared.json")
        assert p["configuration"]["max_seq_length"]==9728
        assert p["condition_budget"]["fresh_full_responses"]==1152
        assert "scorer/lib/gson-2.2.4.jar" in p["scorer_files_sha256"]
        ids=p["question_ids"];assert len(ids)==192 and not set(ids)&set(fitting[:40])
        for m in pilot.MODELS:
            requests=pilot.records(sub/f"inputs/{m}.jsonl")
            assert len(requests)==384 and "gold_aliases" not in json.dumps(requests)
            assert "NEVER INCLUDE THIS" not in json.dumps(requests)
        pilot.validate_submission(sub,generation_only=True)


@pytest.mark.parametrize("actual_java",[False,True])
def test_cpu_scoring_reports_coverage_swaps_costs_and_audit_exclusions_without_training(tmp_path,monkeypatch,actual_java):
    if actual_java and not shutil.which("java"): pytest.skip("Java required")
    sub,out=tmp_path/"sub",tmp_path/"out"
    ids=["gained","lost"];rr=baseline(ids)
    for r in rr:
        if r["question_id"]=="lost" and r["model_key"]=="llama31" and r["draw"]==4:
            r["raw_response"]=response(["Accepted-B"])
    request_rows=pilot.make_requests(examples(ids),rr,config(),"repair");pp=plan(request_rows)
    p={"schema":pilot.SCHEMA,"configuration":{**config(),"questions":2,"discovery_questions":1,"audit_questions":1},
       "question_ids":ids,"feedback":{"rerankers":str(tmp_path/"rr"),"reranker_hashes":{}},
       "sources":{m:{"adapter":str(tmp_path/"adapter"),"adapter_files_sha256":{}} for m in pilot.MODELS},
       "code_hashes":{},"generation_files_sha256":{},"files_sha256":{},"scorer_files_sha256":{},"bundle":str(tmp_path/"bundle")}
    pilot.write(tmp_path/"bundle/manifest.json",{});p["bundle_manifest_sha256"]=pilot.digest(tmp_path/"bundle/manifest.json")
    if actual_java:
        p["scorer_files_sha256"]=pilot.scorer.freeze(ROOT/"third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar",sub)
    pilot.write(sub/"prepared.json",p)
    pilot.jsonl(sub/"baseline_responses.jsonl",rr)
    pilot.jsonl(sub/"scoring_examples.jsonl",[{k:v for k,v in e.items() if k!="review_notes"} for e in examples(ids)])
    pilot.write(sub/"question_split.json",{"gained":"discovery","lost":"audit"})
    pilot.write(tmp_path/"rr/manifest.json",{"selectors":{"max5":{"model_path":"model.joblib"}}})
    jobs=[]
    for m in pilot.MODELS:
        path=sub/m/"generation-1"
        pilot.jsonl(sub/f"inputs/{m}.jsonl",request_rows);pilot.write(sub/f"plans/{m}.json",pp)
        fresh=[generated(r,m,response(["Accepted-B"] if m=="ministral3" and r["question_id"]=="gained" and r["arm"]=="repair_with_pool" else ["Other-A"]),pp) for r in request_rows]
        pilot.jsonl(path/"generations.jsonl",fresh)
        pilot.write(path/"manifest.json",dict(status="complete",smoke_test=False,prepared_sha256=pilot.digest(sub/"prepared.json"),
            model_key=m,source=p["sources"][m],configuration=p["configuration"],
            output_sha256={"generations.jsonl":pilot.digest(path/"generations.jsonl")}))
        jobs.append(dict(model=m,stage="generation",output=str(path)))
    with (sub/"jobs.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=tuple(jobs[0]),delimiter="\t");w.writeheader();w.writerows(jobs)
    import joblib
    monkeypatch.setattr(joblib,"load",lambda _: {"selector":"max5","components":{}})
    from scripts import evaluate_sft_ensemble_rerankers as frozen
    def values(components,rows,contexts):
        assert all("label" not in r for r in rows) and all("gold_aliases" not in e for e in contexts.values())
        return np.array([float(r["answer"]=="Accepted-B") for r in rows])
    monkeypatch.setattr(frozen,"score_components",values)
    if not actual_java:
        from src.notebook_workflows import coverage_comparison
        monkeypatch.setattr(coverage_comparison,"official_candidate_matches",lambda e,c,o,**kw:{(r["question_id"],r["answer"]):r["answer"]=="Accepted-B" for r in c})
    pilot.score(sub,out)
    summary=pilot.read(out/"comparison_summary.json")
    t=summary["paired_contrasts"]["repair_with_pool_minus_independent"]["coverage_transitions"]["coverage_full_pool"]
    assert t["gained_question_ids"]==["gained"] and t["lost_question_ids"]==["lost"] and t["net"]==0
    assert all(summary["costs"][arm]["responses_per_question"]==12 for arm in pilot.ARMS)
    cases=pilot.records(out/"recovery_review_cases.jsonl")
    assert next(r for r in cases if r["question_id"]=="gained")["same_prompt_positive_negative_available"]
    assert next(r for r in cases if r["question_id"]=="lost")["eligible_split_for_future_training"] is False
    assert summary["training_ready"] is False and pilot.read(out/"manifest.json")["status"]=="complete"
    assert not list(out.glob("*dpo*"))


@pytest.mark.parametrize("fail",[False,True])
def test_launcher_preflights_before_three_smoke_generation_chains_and_one_cpu_job(tmp_path,fail):
    bash=Path("C:/Program Files/Git/bin/bash.exe");exe=str(bash) if bash.exists() else shutil.which("bash")
    if not exe: pytest.skip("Bash unavailable")
    script=tmp_path/"scripts/submit_pool_aware_fitting_pilot.sh";script.parent.mkdir()
    bundle=tmp_path/"bundle";bundle.mkdir()
    stub='''
module() { :; }
source() { :; }
python3() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift; mkdir -p "$1"; return 0; fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '%s.gadi-pbs\\n' "$(wc -l < "$TEST_LOG")"; }
'''
    script.write_text((ROOT/"gadi_sft_8b_starter/scripts/submit_pool_aware_fitting_pilot.sh").read_text()
        .replace("${USER}","${TEST_USER}").replace("set -euo pipefail","set -euo pipefail\n"+stub),newline="\n")
    log=tmp_path/"submitted.log"
    result=subprocess.run([exe,script.as_posix(),bundle.as_posix()],capture_output=True,text=True,
        env={**os.environ,"TEST_USER":"test","TEST_LOG":log.as_posix(),"TEST_FAIL":str(int(fail))},timeout=30)
    if fail: assert result.returncode!=0 and not log.exists()
    else:
        assert result.returncode==0,result.stderr
        lines=log.read_text().splitlines();assert len(lines)==7
        assert all(f"afterok:{i}.gadi-pbs" in lines[i] for i in (1,3,5))
        assert "afterok:2.gadi-pbs:4.gadi-pbs:6.gadi-pbs" in lines[-1]


@pytest.mark.parametrize("name",["scripts/pool_aware_fitting_pilot.py","gadi_sft_8b_starter/scripts/run_pool_aware_fitting_pilot.py"])
def test_fresh_cli_imports(name):
    result=subprocess.run([sys.executable,str(ROOT/name),"--help"],cwd=ROOT,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
