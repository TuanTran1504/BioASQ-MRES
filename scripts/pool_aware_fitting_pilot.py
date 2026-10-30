#!/usr/bin/env python3
"""Prepare/score a fitting-only, fixed-response-budget pool-aware generation pilot."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(1,str(ROOT/"gadi_sft_8b_starter/scripts"))
from scripts.prepare_answer_selector_data import read,records,write,jsonl,digest,verify_files,parse_surfaces,MODELS
from scripts import prepare_generator_form_designs as designs
from scripts import generator_form_scorer_runtime as scorer
from scripts.prepare_reranker_guided_dpo import load_inputs,configure_data_root
from run_single_answer_sampling import sample_seed

SCHEMA="pool-aware-fitting-pilot-v1"
ARMS=("independent","repair_no_pool","repair_with_pool")


def identity(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def select_questions(fitting,protected,preference_split,config):
    banned=set().union(*(set(v) for v in protected.values()))
    reserved={q for q,s in preference_split.items() if s=="inner_validation"}
    eligible=set(fitting)-banned-reserved
    ordered=sorted(eligible,key=lambda q:identity([config["seed"],"pool-aware-fitting-pilot-v1",q]))
    if config["questions"]!=config["discovery_questions"]+config["audit_questions"] or len(ordered)<config["questions"]:
        raise ValueError("Invalid fitting pilot panel size")
    chosen=ordered[:config["questions"]]
    split={q:"discovery" if i<config["discovery_questions"] else "audit" for i,q in enumerate(chosen)}
    return chosen,split


def initial_pool(rows,qid,seed):
    surfaces=[]
    for row in sorted((r for r in rows if r["question_id"]==qid and 1<=r["draw"]<=3),
                      key=lambda r:(MODELS.index(r["model_key"]),r["draw"])):
        for c in parse_surfaces(row["raw_response"])[0]:
            if c["answer"] not in surfaces: surfaces.append(c["answer"])
    surfaces.sort(key=lambda a:identity([seed,qid,"pool-order",a]))
    return [{"id":f"P{i:03d}","answer":a} for i,a in enumerate(surfaces,1)]


def messages(example,pool,prompt):
    # Build a whitelist rather than forwarding gold/review fields from examples.
    payload={"question":example["question"],"snippets":[{"id":s.get("id",s.get("snippet_id")),"text":s["text"]}
              for s in example["snippets"]],"previous_proposals":pool}
    return [{"role":"system","content":prompt},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]


def make_requests(examples,baseline,config,prompt):
    requests=[]
    for e in examples:
        q=e["question_id"];pool=initial_pool(baseline,q,config["seed"])
        for arm in config["new_arms"]:
            row={"question_id":q,"arm":arm,"seed":sample_seed(q,5,config["seed"]),
                 "messages":messages(e,pool if arm=="repair_with_pool" else [],prompt)}
            requests.append({**row,"request_key":identity(row)})
    return requests


def token_plan(tokenizer,requests,source):
    from run_extractive_expansion_8b import tokenize_text
    result={}
    for r in requests:
        text=tokenizer.apply_chat_template(r["messages"],tokenize=False,add_generation_prompt=True,**source["chat_template_kwargs"])
        ids=tokenize_text(tokenizer,text,add_special_tokens=True)["input_ids"]
        if not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):
            raise ValueError("Expected one flat native prompt token sequence")
        result[r["request_key"]]={"prompt_tokens":len(ids),"prompt_ids_sha256":identity(ids)}
    return result


def source_inputs(repository,bundle):
    designs.validate(bundle)
    provenance=read(bundle/"source_provenance.json")
    rr=Path(provenance["rerankers"])
    if digest(rr/"manifest.json")!=provenance["reranker_manifest_sha256"]:
        raise ValueError("Original reranker changed")
    fit,_,raw,_,rr_hashes,banks,data=load_inputs(repository,rr,read(repository/"gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json"))
    for m,pin in banks.items():
        original=provenance["native_banks"][m]
        if pin["path"]!=original["directory"] or any(pin[k]!=original[k] for k in ("manifest_sha256","responses_sha256")):
            raise ValueError("Original SFT bank differs")
    generation=read(data.DEFAULT_CONFIG)
    if generation!=provenance["generation_config"]: raise ValueError("Original decoding configuration changed")
    _,aliases=data.panels(generation)
    sources={m:data.selected_source(generation,m) for m in MODELS}
    from matched_8b_provenance import validate_provenance
    for m in MODELS:
        validate_provenance({"expansion":read(Path(banks[m]["path"])/"manifest.json")["source"]},{"expansion":sources[m]})
    return fit,raw,aliases,sources,{"rerankers":str(rr),"reranker_hashes":rr_hashes,"banks":banks}


def prepare(repository,bundle,submission,configuration):
    if submission.exists(): raise FileExistsError(submission)
    config=read(configuration)
    if (config["prefix_draws_per_model"]!=3 or config["total_draws_per_model"]!=4 or config["new_draws_per_model_per_arm"]!=1
            or config["new_arms"]!=list(ARMS[1:]) or config["temperature"]!=.8 or config["top_p"]!=.95
            or config["max_new_tokens"]!=512 or config["max_candidates"]!=5 or config["future_dpo_training"] or config["paid_judging"]):
        raise ValueError("Unsupported fixed-response-budget pilot protocol")
    fit,raw,aliases,sources,feedback=source_inputs(repository,bundle)
    fitting=read(bundle/"fitting_ids.json")
    if set(fit)!=set(fitting) or len(fit)!=1296: raise ValueError("Fitting membership changed")
    ids,split=select_questions(fitting,read(bundle/"protected_ids.json"),read(bundle/"review/question_split.json"),config)
    examples=[{**fit[q],"gold_aliases":aliases[q]} for q in ids]
    baseline=[r for r in raw if r["question_id"] in split and 1<=r["draw"]<=4]
    check_baseline(baseline,set(ids),config)
    prompt=(repository/"gadi_sft_8b_starter/prompts/pool_aware_repair_v1.txt").read_text(encoding="utf-8").strip()
    requests=make_requests(examples,baseline,config,prompt)
    jar=scorer.source(bundle);scorer.smoke(jar)
    from run_base_original_prompt import load_text_tokenizer,native_context
    from run_generator_form_designs import snapshot
    plans={}
    required=config["minimum_allocation"]
    for m in MODELS:
        print("Checking complete fitting inputs and frozen SFT tokenizer:",m,flush=True)
        tokenizer=load_text_tokenizer(sources[m]["adapter"])
        plans[m]=token_plan(tokenizer,requests,sources[m])
        # Check the saved baseline used the same native tokenizer/serialization.
        from run_extractive_expansion_8b import tokenize_text
        for row in baseline:
            if row["model_key"]!=m or row["draw"]!=1: continue
            text=tokenizer.apply_chat_template(row["prompt"],tokenize=False,add_generation_prompt=True,**sources[m]["chat_template_kwargs"])
            if len(tokenize_text(tokenizer,text,add_special_tokens=True)["input_ids"])!=row["prompt_tokens"]:
                raise ValueError("Baseline prompt tokens differ from frozen SFT tokenizer")
        required=max(required,max(p["prompt_tokens"] for p in plans[m].values())+config["max_new_tokens"])
    allocation=math.ceil(required/256)*256
    if allocation>config["maximum_allocation"] or any(allocation>native_context(s["base_snapshot"]) for s in sources.values()):
        raise ValueError("Complete snippets and pool exceed allocation/native context; refusing truncation")
    submission.mkdir(parents=True)
    jsonl(submission/"scoring_examples.jsonl",examples)
    jsonl(submission/"baseline_responses.jsonl",baseline)
    write(submission/"question_split.json",split)
    (submission/"repair_prompt.txt").write_text(prompt,encoding="utf-8")
    code=snapshot(repository,submission/"code")
    files={}
    for m in MODELS:
        jsonl(submission/f"inputs/{m}.jsonl",requests)
        write(submission/f"plans/{m}.json",plans[m])
        files.update({n:digest(submission/n) for n in (f"inputs/{m}.jsonl",f"plans/{m}.json")})
    runtime_hashes=scorer.freeze(jar,submission)
    for stage in ("generate","score"):
        shutil.copy2(repository/f"gadi_sft_8b_starter/jobs/{stage}_pool_aware_fitting_pilot.pbs",submission/f"{stage}.pbs")
    files["generate.pbs"]=digest(submission/"generate.pbs")
    write(submission/"prepared.json",{"schema":SCHEMA,"repository":str(repository),"bundle":str(bundle),
        "bundle_manifest_sha256":digest(bundle/"manifest.json"),"configuration":{**config,"max_seq_length":allocation},
        "question_ids":ids,"ordered_question_ids_sha256":identity(ids),"sources":sources,"feedback":feedback,"code_hashes":code,"generation_files_sha256":files,
        "scorer_files_sha256":runtime_hashes,"files_sha256":{n:digest(submission/n) for n in
            ("scoring_examples.jsonl","baseline_responses.jsonl","question_split.json","repair_prompt.txt","generate.pbs","score.pbs")},
        "source_data_use":"fitting-only diagnostic; no held-out generation, pair construction or DPO training",
        "condition_budget":{"responses_per_question":12,"max_output_tokens_per_response":512,
                            "input_tokens_matched":False,"fresh_full_responses":len(ids)*len(MODELS)*2}})
    print(f"Prepared {len(ids)} fitting questions: {config['discovery_questions']} discovery / {config['audit_questions']} audit; three conditions x 12 responses/question.",flush=True)
    print("Reuse first nine SFT draws; independent fourth draws also reused. Two fresh correction prompts/model/question.",flush=True)


def check_baseline(rows,ids,config):
    expected={(q,m,d) for q in ids for m in MODELS for d in range(1,5)}
    seen=set()
    for r in rows:
        k=(r["question_id"],r["model_key"],r["draw"])
        if (k in seen or k not in expected or r["split"]!="train" or r["seed"]!=sample_seed(r["question_id"],r["draw"]+1,config["seed"])):
            raise ValueError("Baseline request membership/seed differs")
        seen.add(k)
    if seen!=expected: raise ValueError("Incomplete saved baseline; failed draws must remain in denominator")


def validate_submission(submission,generation_only=False):
    p=read(submission/"prepared.json")
    if p["schema"]!=SCHEMA: raise ValueError("Wrong pool-aware pilot submission")
    verify_files(submission/"code",p["code_hashes"])
    verify_files(submission,p["generation_files_sha256"])
    for source in p["sources"].values(): verify_files(Path(source["adapter"]),source["adapter_files_sha256"])
    if not generation_only:
        verify_files(submission,p["files_sha256"])
        verify_files(submission,p["scorer_files_sha256"])
        if digest(Path(p["bundle"])/"manifest.json")!=p["bundle_manifest_sha256"]: raise ValueError("Source package changed")
        verify_files(Path(p["feedback"]["rerankers"]),p["feedback"]["reranker_hashes"])
    return p


def pool_rows(rows,ids):
    seen=set();result={};ranks=Counter()
    for r in sorted(rows,key=lambda r:(r["question_id"],MODELS.index(r["model_key"]),r["draw"])):
        k=(r["question_id"],r["model_key"],r["draw"])
        if k in seen or r["question_id"] not in ids or r["draw"] not in range(1,5): raise ValueError("Invalid evaluation draw")
        seen.add(k);emitted=set()
        for c in parse_surfaces(r["raw_response"])[0]:
            q,m,a=r["question_id"],r["model_key"],c["answer"]
            t=result.setdefault((q,a),dict(question_id=q,answer=a,sources=[],source_ranks={},relation_types=[],sample_frequencies={},
                                         surface_operations=[],is_format_variant=False))
            if m not in t["sources"]:
                t["sources"].append(m);ranks[q,m]+=1;t["source_ranks"][m]=ranks[q,m]
            relation=str(c.get("relation_type") or "unrecognized")
            if relation not in t["relation_types"]: t["relation_types"].append(relation)
            if a not in emitted:
                t["sample_frequencies"][m]=t["sample_frequencies"].get(m,0)+.25;emitted.add(a)
    if seen!={(q,m,d) for q in ids for m in MODELS for d in range(1,5)}: raise ValueError("Missing evaluation draws")
    return list(result.values())


def verify_result(row,request,plan,model,config):
    if (row["request_key"]!=request["request_key"] or row["question_id"]!=request["question_id"] or row["arm"]!=request["arm"]
            or row["model_key"]!=model or row["seed"]!=request["seed"]
            or row["prompt_tokens"]!=plan["prompt_tokens"] or row["prompt_ids_sha256"]!=plan["prompt_ids_sha256"]
            or not isinstance(row["raw_response"],str) or type(row["output_tokens"])is not int
            or not 0<=row["output_tokens"]<=config["max_new_tokens"] or not math.isfinite(row["generation_seconds"]) or row["generation_seconds"]<0):
        raise ValueError("Saved pool-aware response differs from requested prompt/budget")


def assemble_arms(baseline,generated,ids):
    arms={"independent":baseline}
    for arm in ARMS[1:]:
        arms[arm]=[r for r in baseline if r["draw"]<=3]+[
            {**generated[m,q,arm],"draw":4,"split":"train"} for q in ids for m in MODELS]
    # Marginal replacement changes exactly one of twelve responses per question.
    for m in MODELS:
        arms["focal_"+m]=[r for r in baseline if not(r["model_key"]==m and r["draw"]==4)]+[
            {**generated[m,q,"repair_with_pool"],"draw":4,"split":"train"} for q in ids]
    return arms


def score(submission,output):
    if output.exists(): raise FileExistsError(output)
    p=validate_submission(submission);config=p["configuration"]
    with (submission/"jobs.tsv").open() as handle:
        import csv
        jobs=list(csv.DictReader(handle,delimiter="\t"))
    runs={r["model"]:Path(r["output"]) for r in jobs if r["stage"]=="generation"}
    if set(runs)!=set(MODELS): raise ValueError("Need all three full generation jobs")
    generated={};run_hashes={};input_requests={}
    for m,path in runs.items():
        state=read(path/"manifest.json")
        if (state["status"]!="complete" or state["smoke_test"] or state["prepared_sha256"]!=digest(submission/"prepared.json")
                or state["model_key"]!=m or state["source"]!=p["sources"][m] or state["configuration"]!=config):
            raise ValueError("Need complete full prompt-only generation")
        verify_files(path,state["output_sha256"]);run_hashes[m]=digest(path/"manifest.json")
        rr=records(path/"generations.jsonl");requests={r["request_key"]:r for r in records(submission/f"inputs/{m}.jsonl")}
        plans=read(submission/f"plans/{m}.json")
        if len(rr)!=len(requests) or {r["request_key"] for r in rr}!=set(requests): raise ValueError("Generation request coverage differs")
        for r in rr:
            verify_result(r,requests[r["request_key"]],plans[r["request_key"]],m,config)
            generated[m,r["question_id"],r["arm"]]=r
            input_requests[m,r["request_key"]]=requests[r["request_key"]]
    examples={r["question_id"]:r for r in records(submission/"scoring_examples.jsonl")}
    ids=set(p["question_ids"]);baseline=records(submission/"baseline_responses.jsonl")
    check_baseline(baseline,ids,config)
    arms=assemble_arms(baseline,generated,p["question_ids"])
    pools={a:pool_rows(rr,ids) for a,rr in arms.items()}
    distinct={(r["question_id"],r["answer"]):r for pp in pools.values() for r in pp}
    output.mkdir(parents=True)
    state={"status":"running","submission":str(submission),"question_count":len(ids),"generation":False,"training":False,"paid_judging":False}
    write(output/"manifest.json",state)
    try:
        from src.notebook_workflows.coverage_comparison import official_candidate_matches
        from scripts.evaluate_sft_ensemble_rerankers import rank_frozen
        import joblib
        (output/"official").mkdir()
        matches=official_candidate_matches(list(examples.values()),list(distinct.values()),output/"official",
                                           jar_path=submission/"scorer/BioASQEvaluation.jar") if distinct else {}
        for pp in pools.values():
            for r in pp: r["label"]=int(matches[r["question_id"],r["answer"]])
        rr=Path(p["feedback"]["rerankers"]);meta=read(rr/"manifest.json")["selectors"]["max5"]
        conditions,perq,rankings=rank_frozen(joblib.load(rr/meta["model_path"]),pools,examples)
        from scripts.evaluate_generator_form_designs import contrast
        contrasts={a+"_minus_independent":contrast("independent_max5_reranker",a+"_max5_reranker",perq) for a in ARMS[1:]}
        contrasts["with_pool_minus_no_pool"]=contrast("repair_no_pool_max5_reranker","repair_with_pool_max5_reranker",perq)
        for m in MODELS: contrasts["focal_"+m+"_minus_independent"]=contrast("independent_max5_reranker","focal_"+m+"_max5_reranker",perq)
        split=read(submission/"question_split.json")
        if (set(examples)!=ids or set(split)!=ids or len(ids)!=config["questions"]
                or Counter(split.values())!={"discovery":config["discovery_questions"],"audit":config["audit_questions"]}):
            raise ValueError("Pilot fitting question membership/split differs")
        split_metrics={};split_contrasts={}
        for panel in ("discovery","audit"):
            split_metrics[panel]={}
            for a in ARMS:
                rows=[r for r in perq[a+"_max5_reranker"] if split[r["question_id"]]==panel]
                split_metrics[panel][a]={"questions":len(rows),"metrics":{k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k!="question_id"}}
            subset={name:[r for r in rows if split[r["question_id"]]==panel] for name,rows in perq.items()}
            split_contrasts[panel]={a+"_minus_independent":contrast("independent_max5_reranker",a+"_max5_reranker",subset) for a in ARMS[1:]}
        costs={}
        for a,rows in arms.items():
            costs[a]={"responses":len(rows),"responses_per_question":len(rows)/len(ids),
                "input_tokens":sum(r["prompt_tokens"] for r in rows),"output_tokens":sum(r["output_tokens"] for r in rows),
                "ceiling_responses":sum(r["output_tokens"]>=512 for r in rows),
                "empty_or_unparseable_responses":sum(not parse_surfaces(r["raw_response"])[0] for r in rows)}
            costs[a]["parser_diagnostics"]=dict(Counter(w for r in rows for w in parse_surfaces(r["raw_response"])[1]))
            costs[a]["mean_candidates_per_response"]=sum(len(parse_surfaces(r["raw_response"])[0]) for r in rows)/len(rows)
            costs[a]["mean_unique_pool_size"]=len(pools[a])/len(ids)
        review=[]
        for q in p["question_ids"]:
            a,b=(next(r for r in perq[n+"_max5_reranker"] if r["question_id"]==q) for n in ("independent","repair_with_pool"))
            if a["coverage_full_pool"]==b["coverage_full_pool"]: continue
            # Diagnostics only, never automatic DPO pairs. Cross-prompt
            # independent/no-pool completions are not relabeled as DPO negatives.
            replies=[generated[m,q,"repair_with_pool"] for m in MODELS]
            from expansion_dpo_data import strict_answers
            eligible=[]
            for r in replies:
                try: strict_answers(r["raw_response"])
                except (ValueError,TypeError): continue
                if r["output_tokens"]<config["max_new_tokens"]: eligible.append(r)
            pos=[r for r in eligible if any(matches.get((q,c["answer"]),False) for c in parse_surfaces(r["raw_response"])[0])]
            neg=[r for r in eligible if not any(matches.get((q,c["answer"]),False) for c in parse_surfaces(r["raw_response"])[0])]
            review.append({"question_id":q,"split":split[q],"transition":"gained" if b["coverage_full_pool"] else "lost",
                "question":examples[q]["question"],"snippets":examples[q]["snippets"],"gold_aliases":examples[q]["gold_aliases"],
                "pool_aware_replies":replies,"same_prompt_positive_negative_available":bool(pos and neg),
                "pool_conditioned_messages":input_requests[MODELS[0],replies[0]["request_key"]]["messages"],
                "baseline_candidate_surfaces":[{"answer":r["answer"],"official_accepted":bool(r["label"])}
                    for r in pools["independent"] if r["question_id"]==q],
                "training_ready":False,"semantic_review":"unresolved; official nonacceptance does not establish medical incorrectness",
                "eligible_split_for_future_training":split[q]=="discovery"})
        jsonl(output/"recovery_review_cases.jsonl",review)
        jsonl(output/"rankings.jsonl",rankings)
        write(output/"comparison_per_question.json",perq)
        write(output/"comparison_summary.json",{"question_count":len(ids),"conditions":conditions,"paired_contrasts":contrasts,
            "split_metrics":split_metrics,"split_contrasts":split_contrasts,"costs":costs,"review_cases":len(review),"training_ready":False,
            "same_prompt_discovery_opportunity_questions":sorted({r["question_id"] for r in review
                if r["transition"]=="gained" and r["eligible_split_for_future_training"] and r["same_prompt_positive_negative_available"]}),
            "limitations":["Fitting-only feasibility diagnostics, not held-out performance.","Response/output limits matched; pool-aware input tokens and experiment-wide costs are not matched.",
                "Saved independent draws are reused; common fourth-slot seeds do not guarantee identical sampling across GPU hardware.",
                "No automatic DPO pairs, semantic correctness labels or new training."]})
        lines=["# Prompt-only pool-aware generation: fitting pilot","",
               "Same original max-five SFTs and first nine saved draws; 12 responses/question/condition. Frozen logistic, verbatim pools.","",
               "| Condition | Full-pool coverage | Top-five coverage | MRR@5 | Mean pool | Input tokens | Output tokens |","|---|---:|---:|---:|---:|---:|---:|"]
        for a in ARMS:
            m=conditions[a+"_max5_reranker"]["metrics"]
            lines.append(f"| {a} | {round(m['coverage_full_pool']*len(ids))}/{len(ids)} | {round(m['coverage_at5']*len(ids))}/{len(ids)} | {m['mrr_at5']:.6f} | {costs[a]['mean_unique_pool_size']:.2f} | {costs[a]['input_tokens']} | {costs[a]['output_tokens']} |")
        for name,c in contrasts.items():
            t=c["coverage_transitions"]["coverage_full_pool"]
            lines += ["",f"{name}: full-pool gained {t['gained']}, lost {t['lost']}, net {t['net']:+d}.",
                      f"Gained IDs: {t['gained_question_ids']}",f"Lost IDs: {t['lost_question_ids']}"]
        lines += ["",f"{config['discovery_questions']} discovery / {config['audit_questions']} audit fitting questions; neither is unseen by the original SFTs. Audit cases are excluded from future preference training.",
                  "More candidates or novel wording alone is not a coverage success. New accepted forms require evidence/scope review. No DPO training launched."]
        (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        validate_submission(submission)
        for m,path in runs.items():
            if digest(path/"manifest.json")!=run_hashes[m]: raise ValueError("Generation metadata changed during scoring")
            verify_files(path,read(path/"manifest.json")["output_sha256"])
        state.update(status="complete",output_sha256={x.relative_to(output).as_posix():digest(x) for x in output.rglob("*") if x.is_file() and x!=output/"manifest.json"})
    except BaseException as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally: write(output/"manifest.json",state)
    print("Fitting pilot report:",output/"report.md",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("prepare","score"),required=True)
    p.add_argument("--repository",type=Path,default=ROOT)
    p.add_argument("--bundle",type=Path)
    p.add_argument("--submission",type=Path,required=True)
    p.add_argument("--configuration",type=Path,default=ROOT/"gadi_sft_8b_starter/configs/pool_aware_fitting_pilot.json")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.mode=="prepare":
        if not a.bundle: p.error("--bundle required")
        prepare(a.repository.resolve(),a.bundle.resolve(),a.submission.resolve(),a.configuration.resolve())
    else:
        if not a.output: p.error("--output required")
        score(a.submission.resolve(),a.output.resolve())
