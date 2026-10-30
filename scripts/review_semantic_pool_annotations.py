#!/usr/bin/env python3
"""Conservative review holds on frozen GPT semantic labels; no model calls."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from decimal import Decimal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import semantic_pool_evaluation as data


def canonical(text): return re.sub(r"[^a-z0-9]","",text.casefold())


ONES=dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),range(20)))
TENS=dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),range(20,100,10)))
SCALES={"thousand":1000,"million":1000000,"billion":1000000000}
APPROX=r"\b(about|around|approximately|approximate|roughly|estimated|estimate)\b|~"
QUANTITY_QUESTION=r"how many|how much|how long|what percentage|what proportion|what fraction|what is the (number|length|size|duration|concentration|percentage)"


def number_values(text):
    """Review hint, not an unrestricted mathematical equivalence evaluator."""
    text=re.sub(r"\bbetween\b(.*?)\band\b",r"between \1 separator ",text.casefold())
    tokens=re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+",text)
    result=[];i=0
    while i<len(tokens):
        t=tokens[i]
        if t[0].isdigit():
            v=Decimal(t.replace(",",""));i+=1
            if i<len(tokens) and tokens[i] in SCALES: v*=SCALES[tokens[i]];i+=1
            result.append(v);continue
        if t in ONES or t in TENS or t=="hundred" or (t=="a" and i+1<len(tokens) and tokens[i+1]in("hundred",*SCALES)):
            total=current=0
            while i<len(tokens):
                word=tokens[i]
                if word in ONES: current+=ONES[word]
                elif word in TENS: current+=TENS[word]
                elif word=="a": current+=1
                elif word=="hundred": current=max(1,current)*100
                elif word in SCALES: total+=max(1,current)*SCALES[word];current=0
                elif word!="and": break
                i+=1
            result.append(Decimal(total+current));continue
        i+=1
    return tuple(result)


def bound(text):
    t=text.casefold()
    for name,pattern in (("le",r"<=|≤|\bat most\b|\bno more than\b|\bnot greater than\b|\bup to\b"),
                         ("ge",r">=|≥|\bat least\b|\bno fewer than\b|\bnot less than\b"),
                         ("lt",r"<|\bless than\b|\bfewer than\b"),("gt",r">|\bmore than\b|\bgreater than\b")):
        if re.search(pattern,t): return name
    return "none"


def quantity_units(text):
    aliases={"nucleotide":"nt","nucleotides":"nt","nt":"nt","base pair":"bp","base pairs":"bp","bp":"bp",
        "kilobase":"kb","kilobases":"kb","kb":"kb","kbp":"kb","megabase":"mb","megabases":"mb","mb":"mb","mbp":"mb",
        "cell division":"divisions","cell divisions":"divisions","interphase":"interphase","cycle":"cycles","cycles":"cycles",
        "year":"years","years":"years","month":"months","months":"months","day":"days","days":"days",
        "hour":"hours","hours":"hours","minute":"minutes","minutes":"minutes","second":"seconds","seconds":"seconds",
        "percent":"percent","percentage":"percent","%":"percent","mg":"mg","kg":"kg","ml":"ml"}
    pattern="|".join(re.escape(k) for k in sorted(aliases,key=len,reverse=True))
    return {aliases[m.group()] for m in re.finditer(r"(?<!\w)(?:"+pattern+r")(?!\w)",text.casefold())}


def numerical_holds(answer,question,aliases):
    if not re.search(QUANTITY_QUESTION,question,re.I): return []
    reasons=[]
    refs=[a for a in aliases if number_values(a)]
    if refs and not any(number_values(answer)==number_values(a) and bound(answer)==bound(a) for a in refs):
        reasons.append("numeric_value_range_or_bound_differs_from_references")
    if re.search(APPROX,answer,re.I) and not any(re.search(APPROX,a,re.I) for a in [question,*aliases]):
        reasons.append("approximation_not_established_by_question_or_references")
    actual=quantity_units(answer)
    expected=[quantity_units(a) or quantity_units(question) for a in aliases]
    if actual and any(expected) and not any(actual==u for u in expected):
        reasons.append("quantitative_unit_or_event_requires_review")
    return reasons


def review(source,judge_directory,output,holds_file=None):
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(judge_directory.resolve()):
        raise ValueError("Write review results separately from the frozen source and raw GPT outputs")
    p,questions,_,units=data.load_package(source)
    state=data.read(judge_directory/"manifest.json")
    if (state["source_manifest_sha256"]!=data.digest(source/"manifest.json")
            or state["status"]not in ("complete","incomplete")):
        raise ValueError("Need saved GPT annotation results from these frozen validation pools")
    data.verify(judge_directory,state["output_sha256"])
    raw=data.records(judge_directory/"evaluation/annotated_candidates.jsonl")
    if len(raw)!=len(units) or {r["unit_id"] for r in raw}!=set(units): raise ValueError("Wrong annotation membership")
    annotations={}
    for r in raw:
        u=units[r["unit_id"]]
        if any(r[k]!=u[k] for k in ("question_id","answer","official_accepted")):
            raise ValueError("Annotated candidate identity or official label changed")
        annotations[r["unit_id"]]={k:r[k] for k in ("class","judge_decision","annotation_source") if k in r}
    flags=defaultdict(list)
    manual=data.records(holds_file) if holds_file else []
    for row in manual:
        uid=row["unit_id"]
        if uid not in units or units[uid]["official_accepted"] or not row["reason"].strip():
            raise ValueError("Invalid assistant review hold; official C3 stays unchanged")
        flags[uid].append("assistant_review: "+row["reason"])
    same_surface=defaultdict(list)
    for uid,u in units.items():
        if not u["official_accepted"]:
            same_surface[u["question_id"]," ".join(u["answer"].casefold().split())].append(uid)
    for ids in same_surface.values():
        if {annotations[uid]["class"] for uid in ids}>={"C1","C2"}:
            for uid in ids: flags[uid].append("conflicting_C1_C2_for_case_spacing_variants")
    for uid,a in annotations.items():
        if a["class"]!="C2": continue
        d=a.get("judge_decision",{})
        if re.search(r"\b(subset|narrower|broader|partial|part of|omits|excludes)\b",d.get("rationale",""),re.I):
            flags[uid].append("positive_rationale_mentions_substantive_scope_difference")
        q=questions[units[uid]["question_id"]]
        if "\ufffd" in units[uid]["answer"]: flags[uid].append("replacement_character_in_candidate_must_not_be_silently_repaired")
        numeric_reasons=numerical_holds(units[uid]["answer"],q["question"],q["gold_aliases"])
        if numeric_reasons: flags[uid].extend(numeric_reasons)
        evidence=canonical(" ".join([*q["gold_aliases"],*(s["text"] for s in q["snippets"])]))
        codes=re.findall(r"\b(?:[A-Za-z]{2,}[- ]?\d[A-Za-z0-9-]*|NCT\d+)\b",units[uid]["answer"])
        if any(canonical(c)not in evidence for c in codes):
            flags[uid].append("identifier_not_established_by_references_or_supplied_evidence")
    held=[]
    for uid,reasons in sorted(flags.items()):
        if not reasons: raise ValueError("A conservative hold must have a concrete review reason")
        a=annotations[uid]
        held.append({**units[uid],"original_GPT_class":a["class"],"review_reasons":reasons,
            "judge_decision":a.get("judge_decision"),"question":questions[units[uid]["question_id"]]["question"],
            "gold_aliases":questions[units[uid]["question_id"]]["gold_aliases"]})
        annotations[uid]={**a,"original_GPT_class":a["class"],"class":"uncertain","review_holds":reasons}
    summary=data.evaluate(source,output,annotations,
        state["model"]+" with assistant/diagnostic abstentions; not independent biomedical expert review")
    perq=data.read(output/"comparison_per_question.json")
    paired={};uncertainty={}
    for name,rr in perq["semantic_confirmed"].items():
        lo={r["question_id"]:r for r in rr};hi={r["question_id"]:r for r in perq["semantic_optimistic_bound"][name]}
        uncertainty[name]={k:sum(hi[q][k]>lo[q][k] for q in lo) for k in data.METRICS}
        if "joint_baseline_" in name: continue
        mode="sampling4" if "sampling4" in name else "greedy"
        base="joint_baseline_"+mode+"_max5_reranker"
        before={r["question_id"]:r for r in perq["semantic_confirmed"][base]}
        paired[name+"_minus_baseline"]={
            "mrr_at5_difference":sum(lo[q]["mrr_at5"]-before[q]["mrr_at5"] for q in lo)/len(lo),
            "mrr_improved_ids":[q for q in lo if lo[q]["mrr_at5"]>before[q]["mrr_at5"]],
            "mrr_worsened_ids":[q for q in lo if lo[q]["mrr_at5"]<before[q]["mrr_at5"]],
            "coverage_transitions":{k:{"gained_ids":[q for q in lo if not before[q][k] and lo[q][k]],
                "lost_ids":[q for q in lo if before[q][k] and not lo[q][k]]} for k in data.METRICS if k!="mrr_at5"}}
    data.jsonl(output/"review_holds.jsonl",held)
    audit=dict(judge_manifest_sha256=data.digest(judge_directory/"manifest.json"),source_manifest_sha256=data.digest(source/"manifest.json"),
        code_sha256=data.digest(Path(__file__)),manual_holds_sha256=data.digest(holds_file) if holds_file else None,
        model=state["model"],cumulative_charged_or_reserved_usd=state["charged_or_reserved_usd"],
        GPT_candidates_received=state["judged_candidates"],GPT_candidates_expected=state["candidates_to_judge"],
        assistant_or_diagnostic_held_candidates=len(held),hold_reason_counts=dict(Counter(r for row in held for r in row["review_reasons"])),
        unresolved_question_counts=uncertainty,paired_confirmed_semantic_changes=paired,
        interpretation="All holds abstain; none establishes factual incorrectness or manufactures a positive. Same labels/rankings across conditions. Validation diagnostics, not independent test evidence.")
    data.write(output/"review_audit.json",audit)
    with (output/"report.md").open("a",encoding="utf-8") as f:
        f.write(f"\nReviewed with conservative abstentions: {len(held)} candidates held. Mechanical hints are not proof of incorrectness.\n")
        f.write(f"GPT candidate judgments received: {state['judged_candidates']}/{state['candidates_to_judge']}. Total batch charged/reserved: US${state['charged_or_reserved_usd']:.4f}.\n")
    data.write(output/"manifest.json",dict(status="complete",source_manifest_sha256=audit["source_manifest_sha256"],
        judge_manifest_sha256=audit["judge_manifest_sha256"],code_sha256=audit["code_sha256"],
        all_GPT_requests_received=state["judged_candidates"]==state["candidates_to_judge"],
        semantic_scores_provisional=summary["semantic_scores_provisional"],
        output_sha256={p.name:data.digest(p) for p in output.iterdir() if p.is_file() and p.name!="manifest.json"}))
    return summary,audit


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path);p.add_argument("--judge",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--holds",type=Path)
    a=p.parse_args()
    review(a.source.resolve(),a.judge.resolve(),a.output.resolve(),a.holds.resolve() if a.holds else None)
