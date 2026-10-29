#!/usr/bin/env python3
"""Run one frozen SFT's gold-blind pool-aware/control requests; no training."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import sys
import time

REPOSITORY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPOSITORY))
sys.path.insert(1,str(REPOSITORY/"gadi_sft_8b_starter/scripts"))
from scripts.pool_aware_fitting_pilot import (read,records,write,jsonl,digest,identity,MODELS,
    validate_submission,token_plan,verify_result,sample_seed)


def selected_requests(submission,p,model,smoke):
    rr=records(submission/f"inputs/{model}.jsonl")
    plans=read(submission/f"plans/{model}.json")
    expected={(q,a) for q in p["question_ids"] for a in p["configuration"]["new_arms"]}
    if len(rr)!=len(expected) or {(r["question_id"],r["arm"]) for r in rr}!=expected:
        raise ValueError("Gold-blind generation request membership differs")
    if set(plans)!={r["request_key"] for r in rr}: raise ValueError("Token plan membership differs")
    for r in rr:
        if set(r)!={"question_id","arm","seed","messages","request_key"} or identity({k:v for k,v in r.items() if k!="request_key"})!=r["request_key"]:
            raise ValueError("Generation request includes unexpected data or changed prompt")
        if r["seed"]!=sample_seed(r["question_id"],5,p["configuration"]["seed"]):
            raise ValueError("Correction must use the original fourth-slot seed")
        if len(r["messages"])!=2 or [m["role"] for m in r["messages"]]!=["system","user"]:
            raise ValueError("Expected system/user messages only")
        import json
        payload=json.loads(r["messages"][1]["content"])
        if set(payload)!={"question","snippets","previous_proposals"} or any(set(s)!={"id","text"} for s in payload["snippets"]):
            raise ValueError("Unexpected generation context fields")
        if any(set(c)!={"id","answer"} for c in payload["previous_proposals"]):
            raise ValueError("Pool must contain unlabelled exact surfaces only")
        if r["arm"]=="repair_no_pool" and payload["previous_proposals"]:
            raise ValueError("No-pool control contains previous answers")
    if smoke:
        lengths={q:max(plans[r["request_key"]]["prompt_tokens"] for r in rr if r["question_id"]==q) for q in p["question_ids"]}
        ids=set(sorted(lengths,key=lambda q:(-lengths[q],q))[:p["configuration"]["smoke_questions"]])
        rr=[r for r in rr if r["question_id"] in ids]
    return rr,plans


def collect(requests,plans,model,config,produce,save,existing=()):
    """All attempts, including parse failures, count; never sample until success."""
    lookup={r["request_key"]:r for r in requests}
    rows=list(existing);seen=set()
    for row in rows:
        key=row["request_key"]
        if key in seen or key not in lookup: raise ValueError("Duplicate or unexpected resumed response")
        verify_result(row,lookup[key],plans[key],model,config);seen.add(key)
    for request in requests:
        key=request["request_key"]
        if key in seen: continue
        row={**produce(request),"request_key":key,"question_id":request["question_id"],
             "arm":request["arm"],"model_key":model,"seed":request["seed"]}
        verify_result(row,request,plans[key],model,config)
        rows.append(row);seen.add(key);save(rows)
    if seen!=set(lookup): raise ValueError("Incomplete generation")
    return rows


def run(submission,model,output,smoke=False,resume=False):
    p=validate_submission(submission,generation_only=True)
    source=p["sources"][model];config=p["configuration"]
    requests,plans=selected_requests(submission,p,model,smoke)
    if output.exists() and not resume: raise FileExistsError(output)
    if resume:
        previous=read(output/"manifest.json")
        if previous["prepared_sha256"]!=digest(submission/"prepared.json") or previous["model_key"]!=model or previous["smoke_test"]!=smoke:
            raise ValueError("Cannot resume a different generation protocol")
        existing=records(output/"generations.jsonl") if (output/"generations.jsonl").exists() else []
        if len(existing)!=previous["completed_requests"]: raise ValueError("Resume response count differs from recorded progress")
    else:
        output.mkdir(parents=True);existing=[]
    jsonl(output/"requests.jsonl",requests)
    state={"status":"running","model_key":model,"submission":str(submission),"prepared_sha256":digest(submission/"prepared.json"),
        "smoke_test":smoke,"expected_requests":len(requests),"completed_requests":len(existing),
        "source":source,"configuration":config,"training":False,"paid_judging":False,"gold_blind_generation":True,
        "resumed":resume,"versions":{n:version(n) for n in ("torch","transformers","peft","unsloth")}}
    write(output/"manifest.json",state)
    model_object=tokenizer=torch=None
    rows=list(existing)
    try:
        from run_generator_feedback_dpo import activate_portable_helpers
        activate_portable_helpers()
        from run_extractive_expansion_8b import configure_job_local_compiler_cache,prepare_inference_execution,load_model,tokenize_text
        configure_job_local_compiler_cache()
        activate=prepare_inference_execution(source)
        import torch
        activate(torch)
        if not torch.cuda.is_available(): raise RuntimeError("Submit generation through the Gadi GPU queue")
        model_object,tokenizer=load_model(source["adapter"],config["max_seq_length"],False,source["model_loader"])
        configure_job_local_compiler_cache()
        if hasattr(model_object,"gradient_checkpointing_disable"): model_object.gradient_checkpointing_disable()
        actual=token_plan(tokenizer,requests,source)
        if any(actual[k]!=plans[k] for k in actual): raise ValueError("Runtime native tokenizer differs from preflight; no truncation allowed")
        device=next(model_object.parameters()).device
        def produce(request):
            text=tokenizer.apply_chat_template(request["messages"],tokenize=False,add_generation_prompt=True,**source["chat_template_kwargs"])
            encoded=tokenize_text(tokenizer,text,return_tensors="pt",add_special_tokens=True)
            encoded={k:v.to(device) for k,v in encoded.items()}
            torch.manual_seed(request["seed"]);torch.cuda.manual_seed_all(request["seed"])
            started=time.perf_counter()
            pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            with torch.inference_mode():
                ids=model_object.generate(**encoded,max_new_tokens=config["max_new_tokens"],do_sample=True,
                    temperature=config["temperature"],top_p=config["top_p"],top_k=0,num_beams=1,repetition_penalty=1.,
                    use_cache=True,pad_token_id=pad,eos_token_id=tokenizer.eos_token_id)
            tail=ids[0,encoded["input_ids"].shape[-1]:]
            return {"raw_response":tokenizer.decode(tail,skip_special_tokens=True).strip(),"output_tokens":int(tail.numel()),
                    "generation_seconds":time.perf_counter()-started,**actual[request["request_key"]]}
        def save(saved):
            jsonl(output/"generations.jsonl",saved)
            state["completed_requests"]=len(saved);write(output/"manifest.json",state)
            print(f"{model}: {len(saved)}/{len(requests)} frozen requests complete",flush=True)
        rows=collect(requests,plans,model,config,produce,save,existing)
        validate_submission(submission,generation_only=True)
        state.update(status="complete",output_sha256={n:digest(output/n) for n in ("generations.jsonl","requests.jsonl")})
    except BaseException as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        write(output/"manifest.json",state)
        del model_object,tokenizer
        if torch is not None and torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission",type=Path,required=True)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--smoke-test",action="store_true")
    p.add_argument("--resume-output",type=Path)
    a=p.parse_args();sub=a.submission.resolve()
    out=a.resume_output.resolve() if a.resume_output else sub/a.model/f"{'smoke' if a.smoke_test else 'generation'}-{os.environ.get('PBS_JOBID','manual')}"
    run(sub,a.model,out,a.smoke_test,bool(a.resume_output))
