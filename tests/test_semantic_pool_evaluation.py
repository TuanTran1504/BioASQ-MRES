"""Strict semantic metrics, frozen validation export and bounded cached judging."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import semantic_pool_evaluation as data
from scripts import judge_semantic_answer_equivalence as judge


def questions():
    return {f"q{i}":dict(question_id=f"q{i}",split="validation",question="How many exomes?" if i==0 else "Which symbol?",
        gold_aliases=[">54000","More than 54,000"] if i==0 else ["Accepted"],
        snippets=[{"id":"S001","text":"The study analysed greater than 54,000 exomes. Accepted is the symbol."}]) for i in range(144)}


def ranked(conditions):
    rows=[]
    for name in conditions:
        for q in range(144):
            if q==2: continue  # Empty pools must remain in the denominator.
            answers=([("At least 54,000",0),("Greater than 54000",0),(">54000",1)] if q==0 else
                [(f"Wrong{i}",0) for i in range(1,6)]+[("Accepted",1)] if q==1 else [("Accepted",1)])
            rows.extend(dict(condition=name,question_id=f"q{q}",rank=i,answer=a,label=label) for i,(a,label) in enumerate(answers,1))
    return rows


@pytest.fixture
def package(tmp_path):
    source=tmp_path/"recovery";output=tmp_path/"package"
    qq=questions();conditions=[f"joint_{s}_{m}_max5_reranker" for s in
        ("baseline","model_specific_selected","shared_selected") for m in ("greedy","sampling4")]
    rows=ranked(conditions);groups,_=data.group_rankings(rows,qq,conditions)
    metrics,perq=data.score_rows(groups,lambda r:bool(r["label"]))
    data.write(source/"comparison/comparison_summary.json",dict(question_count=144,conditions=metrics))
    data.write(source/"comparison/comparison_per_question.json",perq)
    data.jsonl(source/"comparison/joint_rankings.jsonl",rows)
    data.jsonl(source/"llama31/model_specific/validation_inputs.jsonl",
        [{k:e[k] for k in ("question_id","question","snippets")} for e in qq.values()])
    data.write(source/"llama31/model_specific/manifest.json",dict(status="complete",smoke_test=False,
        validation_question_count=144,question_ids=list(qq)))
    data.write(source/"comparison/official_joint/scorer_gold.json",{"questions":[dict(id=f"g{i}",body=e["question"],exact_answer=[e["gold_aliases"]]) for i,e in enumerate(qq.values())]})
    data.write(source/"comparison/official_joint/scorer_id_map.json",{f"g{i}":[q,"Accepted"] for i,q in enumerate(qq)})
    pins={p.relative_to(source).as_posix():data.digest(p) for p in source.rglob("*") if p.is_file()}
    data.write(source/"manifest.json",dict(status="complete",output_sha256=pins))
    data.export(source,output,tmp_path/"semantic.tar.gz")
    return source,output


def decision(index=0,equivalence="equivalent",answers="yes",support="supported"):
    return dict(index=index,equivalence=equivalence,answers_question=answers,matched_alias_indices=[0] if equivalence=="equivalent" else [],
        error_kind="none" if equivalence=="equivalent" else "numeric_or_unit",evidence_support=support,confidence="high",
        evidence=[{"snippet_id":"S001","quote":"greater than 54,000 exomes"}] if support=="supported" else [],
        rationale="Same strict numerical bound and requested quantity.")


def test_export_preserves_full_ranks_all_gold_aliases_and_validation_only_membership(package,tmp_path):
    source,p=package
    state,qq,groups,units=data.load_package(p)
    assert state["question_count"]==144 and state["split"]=="validation" and len(groups)==6
    assert qq["q0"]["gold_aliases"]==[">54000","More than 54,000"]
    assert all(len(g)==144 and g["q2"]==[] and len(g["q1"])==6 for g in groups.values())
    assert len(units)==len({(r["question_id"],r["answer"]) for r in data.records(p/"rankings.jsonl")})
    with tarfile.open(tmp_path/"semantic.tar.gz") as tar:
        assert set(tar.getnames())=={*state["files_sha256"],"manifest.json"}
        assert not any("train" in n or "api" in n for n in tar.getnames())
    raw=p/"rankings.jsonl";raw.write_text(raw.read_text()+"\n")
    with pytest.raises(ValueError,match="Frozen"): data.load_package(p)


def test_semantic_mrr_coverage_and_uncertainty_keep_all144_questions_and_original_order(package,tmp_path):
    _,p=package
    _,_,groups,units=data.load_package(p)
    alias=next(uid for uid,u in units.items() if u["question_id"]=="q0" and u["answer"]=="Greater than 54000")
    annotations={uid:{"class":"C1"} for uid,u in units.items() if not u["official_accepted"]}
    annotations[alias]={"class":"C2"}
    summary=data.evaluate(p,tmp_path/"scored",annotations,"test judgments")
    name=next(iter(groups));a=summary["official"][name]["metrics"];b=summary["semantic_confirmed"][name]["metrics"]
    assert a["mrr_at5"]==pytest.approx((141+1/3)/144)
    assert b["mrr_at5"]==pytest.approx((141+1/2)/144)
    assert b["coverage_at5"]==142/144 and b["coverage_at10"]==143/144 and b["coverage_full_pool"]==143/144
    assert summary["all_candidates_resolved"] is True
    annotations.pop(alias)
    bounds=data.evaluate(p,tmp_path/"pending",annotations,"partial test judgments")
    assert bounds["semantic_scores_provisional"] is True
    assert bounds["semantic_confirmed"][name]["metrics"]["mrr_at5"]==a["mrr_at5"]
    assert bounds["semantic_optimistic_bound"][name]["metrics"]["mrr_at5"]==b["mrr_at5"]
    unknown=data.records(tmp_path/"pending/uncertain_candidates.jsonl")
    assert len(unknown)==1 and unknown[0]["answer"]=="Greater than 54000"


def test_judge_plan_is_blind_deduplicated_and_sends_all_aliases_and_complete_evidence(package,tmp_path,monkeypatch):
    _,p=package
    _,qq,_,units=data.load_package(p)
    jobs=judge.prepare_jobs(qq,units)
    ids=[uid for j in jobs for uid in j["unit_ids"]]
    assert len(ids)==len(set(ids))==sum(not u["official_accepted"] for u in units.values())
    assert not any(units[uid]["official_accepted"] for uid in ids)
    for j in jobs:
        payload=json.loads(j["payload"]["messages"][1]["content"])
        assert set(payload)=={"question","gold_aliases","snippets","candidates"}
        assert all(set(c)=={"index","answer"} for c in payload["candidates"])
        q=units[j["unit_ids"][0]]["question_id"]
        assert payload["snippets"]==qq[q]["snippets"] and payload["gold_aliases"]==qq[q]["gold_aliases"]
    monkeypatch.setattr(judge,"BudgetClient",lambda *a,**kw:pytest.fail("Dry plan contacted paid client"))
    state=judge.run(p,tmp_path/"plan")
    assert state["status"]=="prepared" and state["max_usd"]==0 and state["questions"]==144
    with pytest.raises(ValueError,match="explicit"): judge.run(p,tmp_path/"bad",True)


@pytest.mark.parametrize("change,expected",[
    ({},"C2"),({"confidence":"medium"},"uncertain"),({"evidence_support":"contradicted","evidence":[]},"uncertain"),
    ({"equivalence":"not_equivalent","answers_question":"no","matched_alias_indices":[],"error_kind":"numeric_or_unit"},"C1"),
    ({"equivalence":"not_equivalent","answers_question":"yes","matched_alias_indices":[],"error_kind":"other"},"uncertain"),
    ({"matched_alias_indices":[]},"uncertain"),({"evidence":[]},"uncertain")])
def test_only_high_confidence_consistent_equivalent_answers_become_c2(change,expected):
    qq={"q0":questions()["q0"]}
    units={"a":dict(question_id="q0",answer="Greater than 54000",official_accepted=False)}
    j=judge.prepare_jobs(qq,units)[0]
    d={**decision(),**change}
    checked=judge.validate_decisions({"decisions":[d]},j)["a"]
    assert judge.classify(checked)==expected


def test_resume_reuses_validated_cached_gpt_decisions_without_paying_twice(package,tmp_path,monkeypatch):
    _,p=package;output=tmp_path/"judge";calls=[]
    class OfflineClient:
        def __init__(self,*a,**kw): self.spent=.01;self.blocked_reason=None
        def call(self,j):
            calls.append(j["request_sha256"])
            public=json.loads(j["payload"]["messages"][1]["content"])
            ds=[]
            for c in public["candidates"]:
                ok=c["answer"]=="Greater than 54000"
                ds.append(decision(c["index"],"equivalent" if ok else "not_equivalent","yes" if ok else "no"))
            value={"decisions":ds}
            data.write(output/"cache"/(j["request_sha256"]+".json"),{"payload":j["payload"],"decisions":value})
            return judge.validate_decisions(value,j)
    monkeypatch.setattr(judge,"BudgetClient",OfflineClient)
    first=judge.run(p,output,True,1.,100)
    assert first["status"]=="complete" and first["all_candidates_resolved"]
    count=len(calls)
    second=judge.run(p,output,True,1.,100)
    assert len(calls)==count and second["status"]=="complete"
    assert (output/"review_sample.jsonl").exists()
    with pytest.raises(ValueError,match="budget"): judge.run(p,output,True,2.,100)


@pytest.mark.parametrize("limited",[False,True])
def test_real_spending_client_honours_cap_and_persists_usage_with_mocked_transport(package,tmp_path,monkeypatch,limited):
    from scripts import judge_expansion_dpo_annotations as shared
    import requests
    _,p=package;calls=[];output=tmp_path/"budgeted"
    monkeypatch.setattr(shared,"read_api_key",lambda _:"test-key-never-sent-to-network")
    class Reply:
        status_code=200
        def __init__(self,payload): self.payload=payload
        def json(self):
            public=json.loads(self.payload["messages"][1]["content"])
            ds=[]
            for c in public["candidates"]:
                ok=c["answer"]=="Greater than 54000"
                ds.append(decision(c["index"],"equivalent" if ok else "not_equivalent","yes" if ok else "no"))
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"decisions":ds})}}],
                    "usage":{"prompt_tokens":10,"completion_tokens":10}}
    def transport(url,**kw):
        assert url=="https://api.openai.com/v1/chat/completions"
        calls.append(kw["json"]);return Reply(kw["json"])
    monkeypatch.setattr(requests,"post",transport)
    cap=.00000001 if limited else 1.
    state=judge.run(p,output,True,cap,100)
    assert state["charged_or_reserved_usd"]<=cap
    if limited:
        assert calls==[] and state["status"]=="incomplete" and not state["all_candidates_resolved"]
        assert state["semantic_class_counts"]["uncertain"]==state["candidates_to_judge"]
    else:
        assert calls and state["status"]=="complete" and state["all_candidates_resolved"]
        ledger=data.read(output/"spend-ledger.json")
        assert len(ledger["attempts"])==len(calls) and all(a["status"]=="complete" for a in ledger["attempts"])
        assert state["charged_or_reserved_usd"]==pytest.approx(len(calls)*.00002)
        assert (output/"responses").exists()


@pytest.mark.parametrize("fault",["rank_gap","official_label_conflict","missing_question","changed_reference"])
def test_export_rejects_corrupt_or_mismatched_saved_results(package,tmp_path,fault):
    source,_=package
    if fault in ("rank_gap","official_label_conflict"):
        path=source/"comparison/joint_rankings.jsonl";rr=data.records(path)
        if fault=="rank_gap": rr[0]["rank"]=99
        else: rr[-1]["label"]=0
        data.jsonl(path,rr)
    elif fault=="missing_question":
        path=source/"llama31/model_specific/validation_inputs.jsonl";data.jsonl(path,data.records(path)[:-1])
    else:
        path=source/"comparison/official_joint/scorer_gold.json";v=data.read(path);v["questions"][0]["body"]="Different question";data.write(path,v)
    state=data.read(source/"manifest.json");state["output_sha256"][path.relative_to(source).as_posix()]=data.digest(path);data.write(source/"manifest.json",state)
    with pytest.raises(ValueError): data.export(source,tmp_path/"rejected")


@pytest.mark.parametrize("script",["semantic_pool_evaluation.py","judge_semantic_answer_equivalence.py"])
def test_fresh_cli_imports_without_api_key(script):
    r=subprocess.run([sys.executable,str(ROOT/"scripts"/script),"--help"],capture_output=True,text=True,timeout=40)
    assert r.returncode==0,r.stderr
