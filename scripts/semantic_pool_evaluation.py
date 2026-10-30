#!/usr/bin/env python3
"""Export frozen answer pools and evaluate official and strict semantic matches."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tarfile

SCHEMA = "semantic-answer-pool-v1"
METRICS = ("mrr_at5", "coverage_at5", "coverage_at10", "coverage_full_pool")


def sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as f: return [json.loads(line) for line in f if line.strip()]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")


def verify(root,pins):
    for name,expected in pins.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=expected:
            raise ValueError("Frozen semantic-evaluation input changed: "+name)


def context(example):
    aliases=example["gold_aliases"]
    if not aliases or not all(isinstance(a,str) and a.strip() for a in aliases):
        raise ValueError("Need all nonempty gold alias strings")
    snippets=[{"id":f"S{i:03d}","text":s["text"]} for i,s in enumerate(example["snippets"],1)]
    if not isinstance(example["question"],str) or not all(isinstance(s["text"],str) for s in snippets):
        raise ValueError("Invalid question/evidence")
    return dict(question=example["question"],gold_aliases=aliases,snippets=snippets)


def group_rankings(rows,questions,conditions):
    groups={name:{q:[] for q in questions} for name in conditions}
    units={}
    for r in rows:
        name,q=r["condition"],r["question_id"]
        if name not in groups or q not in questions or type(r["rank"])is not int or r["rank"]<1:
            raise ValueError("Unknown condition/question or invalid rank")
        if (not isinstance(r["answer"],str) or not r["answer"].strip()
                or type(r["label"])not in (int,bool) or r["label"]not in (0,1)):
            raise ValueError("Invalid candidate surface or official acceptance")
        unit=sha({"question_id":q,"context":context(questions[q]),"answer":r["answer"]})
        item=dict(unit_id=unit,question_id=q,answer=r["answer"],official_accepted=bool(r["label"]))
        if unit in units and units[unit]!=item:
            raise ValueError("Official acceptance differs across conditions")
        units[unit]=item
        groups[name][q].append({**r,"unit_id":unit})
    for qq in groups.values():
        for rr in qq.values():
            rr.sort(key=lambda r:r["rank"])
            if [r["rank"] for r in rr]!=list(range(1,len(rr)+1)) or len({r["answer"] for r in rr})!=len(rr):
                raise ValueError("Missing ranks or duplicate exact surfaces")
    return groups,units


def score_rows(groups,match):
    conditions,perq={},{}
    for name,qq in groups.items():
        rows=[]
        for q,rr in qq.items():
            first=next((r["rank"] for r in rr if match(r)),None)
            rows.append(dict(question_id=q,mrr_at5=1/first if first and first<=5 else 0.,
                coverage_at5=int(first is not None and first<=5),coverage_at10=int(first is not None and first<=10),
                coverage_full_pool=int(first is not None)))
        perq[name]=rows
        conditions[name]={"metrics":{k:sum(r[k] for r in rows)/len(rows) for k in METRICS}}
    return conditions,perq


def check_official(groups,summary):
    conditions,perq=score_rows(groups,lambda r:bool(r["label"]))
    if len(next(iter(groups.values())))!=summary["question_count"]:
        raise ValueError("Question denominator differs from original comparison")
    for name,c in conditions.items():
        for k in METRICS:
            value=summary["conditions"][name]["metrics"][k]
            if not math.isfinite(value) or abs(value-c["metrics"][k])>1e-12:
                raise ValueError("Rankings do not reproduce official metrics: "+name+"/"+k)
    return conditions,perq


def form_validation144(source):
    """Read only frozen validation scores, complete evidence and scorer references."""
    state=read(source/"manifest.json")
    conditions=[f"joint_{setting}_{mode}_max5_reranker" for setting in
        ("baseline","model_specific_selected","shared_selected") for mode in ("greedy","sampling4")]
    needed={"comparison/comparison_summary.json","comparison/comparison_per_question.json","comparison/joint_rankings.jsonl",
        "comparison/official_joint/scorer_gold.json","comparison/official_joint/scorer_id_map.json",
        "llama31/model_specific/validation_inputs.jsonl","llama31/model_specific/manifest.json"}
    if state.get("status")!="complete" or not needed<=set(state.get("output_sha256",{})):
        raise ValueError("Need the complete six-run generator-form validation recovery")
    pins={name:state["output_sha256"][name] for name in needed}
    verify(source,pins)
    all_summary=read(source/"comparison/comparison_summary.json")
    if all_summary["question_count"]!=144 or not set(conditions)<=set(all_summary["conditions"]):
        raise ValueError("Need all 144 validation questions and six frozen selected conditions")
    summary={"question_count":144,"conditions":{n:all_summary["conditions"][n] for n in conditions}}
    examples=records(source/"llama31/model_specific/validation_inputs.jsonl")
    if len(examples)!=144 or len({e["question_id"] for e in examples})!=144:
        raise ValueError("Incomplete validation144 evidence membership")
    child=read(source/"llama31/model_specific/manifest.json")
    if (child.get("status")!="complete" or child.get("smoke_test")is not False
            or child.get("validation_question_count")!=144 or len(child["question_ids"])!=144
            or set(child["question_ids"])!={e["question_id"] for e in examples}):
        raise ValueError("Saved validation144 selection membership differs")
    mapping=read(source/"comparison/official_joint/scorer_id_map.json")
    aliases={}
    bodies={}
    for row in read(source/"comparison/official_joint/scorer_gold.json")["questions"]:
        q=mapping[row["id"]][0]
        aa=[a for group in row["exact_answer"] for a in group]
        if q in aliases and (aliases[q]!=aa or bodies[q]!=row["body"]): raise ValueError("Inconsistent scorer gold references")
        aliases[q],bodies[q]=aa,row["body"]
    questions={}
    for e in examples:
        q=e["question_id"]
        if q not in aliases or bodies[q]!=e["question"]: raise ValueError("Missing or mismatched gold references for validation question")
        questions[q]={"question_id":q,"split":"validation",**context({**e,"gold_aliases":aliases[q]})}
    rankings=[{k:r[k] for k in ("condition","question_id","rank","answer","label")}
        for r in records(source/"comparison/joint_rankings.jsonl") if r["condition"] in conditions]
    return state,questions,summary,rankings,read(source/"comparison/comparison_per_question.json"),pins


def export(source,output,archive=None):
    if output.exists(): raise FileExistsError(output)
    if archive is not None and archive.exists(): raise FileExistsError(archive)
    state,questions,summary,rankings,saved,source_pins=form_validation144(source)
    ids=list(questions);conditions=list(summary["conditions"])
    groups,units=group_rankings(rankings,questions,conditions)
    _,perq=check_official(groups,summary)
    for name,rr in perq.items():
        old={r["question_id"]:r for r in saved[name]}
        if len(old)!=len(saved[name]) or set(old)!=set(ids): raise ValueError("Wrong saved question scores")
        if any(abs(r[k]-old[r["question_id"]][k])>1e-12 for r in rr for k in METRICS):
            raise ValueError("Saved per-question scores differ")
    output.mkdir(parents=True)
    jsonl(output/"questions.jsonl",[questions[q] for q in ids])
    jsonl(output/"rankings.jsonl",rankings)
    write(output/"source_summary.json",summary)
    jsonl(output/"candidate_units.jsonl",[units[k] for k in sorted(units)])
    manifest=dict(schema=SCHEMA,status="complete",source_comparison=str(source),source_manifest_sha256=digest(source/"manifest.json"),
        source_files_sha256=source_pins,question_count=len(ids),split="validation",conditions=conditions,
        unique_candidates=len(units),official_C3=sum(u["official_accepted"] for u in units.values()),
        candidates_requiring_GPT=sum(not u["official_accepted"] for u in units.values()),question_splits={"validation":144},
        generation=False,training=False,paid_calls=0,
        files_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        interpretation="Validation144 only, baseline and selected model-specific/shared generator DPO, greedy/sampling4. Full saved pools and exact rankings. No training preferences or checkpoint reselection.")
    write(output/"manifest.json",manifest)
    if archive is not None:
        archive.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,"w:gz") as tar:
            for name in [*manifest["files_sha256"],"manifest.json"]: tar.add(output/name,arcname=name)
    print(json.dumps({k:manifest[k] for k in ("question_count","unique_candidates","official_C3","candidates_requiring_GPT")},indent=2))
    if archive is not None: print("Archive:",archive)


def load_package(directory):
    p=read(directory/"manifest.json")
    if (p["schema"]!=SCHEMA or p["status"]!="complete" or p["question_count"]!=144 or p["split"]!="validation"):
        raise ValueError("Need a validation144 semantic pool package")
    verify(directory,p["files_sha256"])
    required={"questions.jsonl","rankings.jsonl","candidate_units.jsonl","source_summary.json"}
    if not required<=set(p["files_sha256"]) or not p["conditions"] or len(p["conditions"])!=len(set(p["conditions"])):
        raise ValueError("Incomplete or duplicate semantic package conditions")
    rows=records(directory/"questions.jsonl")
    questions={r["question_id"]:r for r in rows}
    if (len(rows)!=len(questions) or len(rows)!=p["question_count"] or any(r["split"]!="validation" for r in rows)):
        raise ValueError("Question membership differs")
    groups,units=group_rankings(records(directory/"rankings.jsonl"),questions,p["conditions"])
    stored=records(directory/"candidate_units.jsonl")
    if len(stored)!=len(units) or {r["unit_id"]:r for r in stored}!=units: raise ValueError("Candidate units differ from frozen pools")
    if len(units)!=p["unique_candidates"] or sum(u["official_accepted"] for u in units.values())!=p["official_C3"]:
        raise ValueError("Candidate counts differ")
    check_official(groups,read(directory/"source_summary.json"))
    return p,questions,groups,units


def evaluate(directory,output,annotations,annotation_source):
    p,questions,groups,units=load_package(directory)
    if not set(annotations)<=set(units): raise ValueError("Annotation outside frozen candidate pool")
    classes={uid:"C3" if u["official_accepted"] else annotations.get(uid,{}).get("class","uncertain") for uid,u in units.items()}
    if not set(classes.values())<={"C3","C2","C1","uncertain"}: raise ValueError("Invalid semantic class")
    if any(classes[u]=="C3" and not units[u]["official_accepted"] for u in units): raise ValueError("GPT cannot assign official C3")
    official,original=score_rows(groups,lambda r:bool(r["label"]))
    confirmed,perq=score_rows(groups,lambda r:classes[r["unit_id"]]in("C3","C2"))
    optimistic,upper=score_rows(groups,lambda r:classes[r["unit_id"]]!="C1")
    unresolved=[{**units[uid],**annotations.get(uid,{})} for uid in sorted(units) if classes[uid]=="uncertain"]
    labels=[{**u,**annotations.get(uid,{}),"class":classes[uid]} for uid,u in sorted(units.items())]
    changed=[]
    for name,rr in perq.items():
        before={r["question_id"]:r for r in original[name]}
        for r in rr:
            if any(r[k]!=before[r["question_id"]][k] for k in METRICS):
                changed.append({"condition":name,"question_id":r["question_id"],"official":before[r["question_id"]],"semantic_confirmed":r})
    output.mkdir(parents=True,exist_ok=True)
    jsonl(output/"annotated_candidates.jsonl",labels)
    jsonl(output/"uncertain_candidates.jsonl",unresolved)
    jsonl(output/"semantic_gain_cases.jsonl",changed)
    write(output/"comparison_per_question.json",{"official":original,"semantic_confirmed":perq,"semantic_optimistic_bound":upper})
    summary=dict(question_count=len(questions),annotation_source=annotation_source,source_manifest_sha256=digest(directory/"manifest.json"),
        official=official,semantic_confirmed=confirmed,semantic_optimistic_bound=optimistic,class_counts=dict(Counter(classes.values())),
        all_candidates_resolved=not unresolved,semantic_scores_provisional=bool(unresolved),training=False,
        interpretation="C3 or high-confidence strict-equivalent C2 is a semantic match. Unknown judgments remain unresolved; all questions stay in the denominator. GPT labels are not independent expert review.")
    write(output/"semantic_summary.json",summary)
    lines=["# Official and strict semantic answer evaluation","",
        "Saved rankings unchanged. C3 = official acceptance; C2 = strict question-conditioned equivalence. GPT-only annotations require audit.","",
        "| Condition | Official MRR@5 | Semantic confirmed MRR@5 | Official coverage@5 | Semantic coverage@5 | Official coverage@10 | Semantic coverage@10 | Official full pool | Semantic full pool |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    n=len(questions)
    for name in p["conditions"]:
        a,b=official[name]["metrics"],confirmed[name]["metrics"]
        display=name.removeprefix("joint_").removesuffix("_max5_reranker")
        lines.append(f"| {display} | {a['mrr_at5']:.6f} | {b['mrr_at5']:.6f} | {round(a['coverage_at5']*n)}/{n} | {round(b['coverage_at5']*n)}/{n} | {round(a['coverage_at10']*n)}/{n} | {round(b['coverage_at10']*n)}/{n} | {round(a['coverage_full_pool']*n)}/{n} | {round(b['coverage_full_pool']*n)}/{n} |")
    lines += ["",f"Class counts: {summary['class_counts']}",
        "Uncertain/pending candidates are not C1. Confirmed semantic values are lower bounds while any uncertainty remains; optimistic bounds are in semantic_summary.json.",
        "All questions, including empty/uncovered pools, remain in the denominator. C3 is retained by definition and is not re-judged by GPT. Official scores remain unchanged. No DPO or generator/reranker training."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("export",))
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--archive",type=Path)
    args=parser.parse_args()
    export(args.comparison.resolve(),args.output.resolve(),args.archive.resolve() if args.archive else None)
