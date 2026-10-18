"""Saved validation diagnostics must distinguish initial wording and omitted alternatives."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from scripts import export_expanded_dpo_coverage_loss as audit

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.write(path, value)


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v)+"\n" for v in values), encoding="utf-8")


def raw(answers):
    return json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"} for i,a in enumerate(answers)]})


def metrics(qid, covered, mrr=0):
    return dict(question_id=qid, coverage_full_pool=float(covered), coverage_at5=float(covered), mrr_at5=mrr)


@pytest.fixture
def bundle(tmp_path):
    comparison, submission, training = [tmp_path / name for name in ("comparison", "submission", "training")]
    comparison.mkdir(); submission.mkdir(); training.mkdir()
    write(training / "manifest.json", {"status":"complete"})
    parser_rel = "gadi_sft_8b_starter/scripts/run_extractive_expansion_8b.py"
    parser = submission / "code" / parser_rel; parser.parent.mkdir(parents=True)
    shutil.copy2(ROOT / parser_rel, parser)
    prepared = {"source":{"system_prompt":"Return FIVE."},"code_hashes":{parser_rel:audit.digest(parser)},"native_banks":{}}
    examples = [{"question_id":f"q{i}","question":f"Which entity {i}?", "snippets":[{"snippet_id":"s1","text":"ShortName means Accepted full name."}]} for i in range(144)]
    context = {"question":examples[0]["question"],"snippets":[{"id":"s1","text":examples[0]["snippets"][0]["text"]}]}
    for model in ("llama31","qwen3","ministral3"):
        bank = tmp_path / model; bank.mkdir()
        write(bank / "manifest.json", {"status":"complete"})
        bank_rows = [dict(question_id="q0", split="validation", model_key=model, draw=d,seed=audit.seed("q0",d+1),
            raw_response=raw(["ShortName","Accepted full name"] if model == "llama31" else ["Accepted full name"]), output_tokens=40,
            prompt=[{"role":"system","content":"Return FIVE."},{"role":"user","content":json.dumps(context)}]) for d in range(5)]
        rows(bank / "responses.jsonl",bank_rows)
        prepared["native_banks"][model] = dict(path=str(bank),manifest_sha256=audit.digest(bank / "manifest.json"),responses_sha256=audit.digest(bank / "responses.jsonl"))
    write(submission / "prepared.json", prepared)
    rows(comparison / "validation_inputs.jsonl",examples)
    aa = [{"answer":"ShortName","relation_type":"original","raw_position":1}]
    for mode,count in (("greedy",1),("sampling4",4)):
        run = comparison / "epoch-1" / mode
        write(run / "config.json",dict(num_generations=count, seed=3407,max_candidates=5,sample_seed_offset=0 if count==1 else 1))
        write(run / "status.json", {"status":"complete"})
        rows(run / "generations.jsonl",[dict(question_id=e["question_id"],snippets_truncated=False,included_snippets=1,total_snippets=1,
            samples=[dict(draw=d,seed=audit.seed(e["question_id"],d if count==1 else d+1),raw_response=raw(["ShortName"]),answers=aa,output_tokens=20) for d in range(1,count+1)]) for e in examples])
    write(comparison / "official/scorer_id_map.json",{"coverage-0":["q0","ShortName"],"coverage-1":["q0","Accepted full name"]})
    rows(comparison / "official/official_candidate_scores.jsonl",[dict(question_id="coverage-0",lenient_accuracy=0),dict(question_id="coverage-1",lenient_accuracy=1)])
    write(comparison / "official/scorer_gold.json", {"questions":[dict(id=f"coverage-{i}",body=examples[0]["question"],exact_answer=[["Accepted full name"]]) for i in range(2)]})
    scores = {}
    for epoch in (0,1):
        scores[f"llama_epoch{epoch}_sampling4"] = [metrics(e["question_id"],i==0 and epoch==0,.5 if i==0 and epoch==0 else 0) for i,e in enumerate(examples)]
        scores[f"epoch{epoch}_sampling4_max5_reranker"] = [metrics(e["question_id"],i==0,1 if i==0 else 0) for i,e in enumerate(examples)]
    write(comparison / "comparison_per_question.json",scores)
    write(comparison / "comparison_summary.json",dict(question_count=144,selected_epoch=0,checkpoint_scores=[dict(epoch=1,llama_full_pool_lost=1)]))
    rows(comparison / "rankings.jsonl", [dict(condition=f"epoch{e}_sampling4_max5_reranker",question_id="q0",answer=a,rank=i,label=int(i==1),score=float(3-i),sources=["qwen3"] if i==1 else ["llama31"])
        for e in (0,1) for i,a in enumerate(("Accepted full name","ShortName"),1)])
    write(comparison / "manifest.json",dict(status="complete",smoke_test=False,validation_question_count=144,question_ids=[e["question_id"] for e in examples],
        selected_epoch=0,submission=str(submission),prepared_sha256=audit.digest(submission / "prepared.json"),training=str(training),training_manifest_sha256=audit.digest(training / "manifest.json"),
        output_sha256={p.relative_to(comparison).as_posix():audit.digest(p) for p in comparison.rglob("*") if p.is_file()}))
    return comparison, tmp_path / "export", tmp_path / "case.tar.gz"


def test_export_accepted_alternative_loss_keeps_source_and_selection_unchanged(bundle):
    comparison,out,archive = bundle
    before = {str(p):audit.digest(p) for p in comparison.parent.rglob("*") if p.is_file()}
    audit.export(comparison,out,archive)
    case = audit.records(out / "cases.jsonl")[0]
    assert case["question_id"] == "q0"
    assert all(d["same_initial_text"] and d["accepted_alternative_removed_with_same_initial"] for d in case["paired_draws"])
    assert case["llama_sampling4"]["sft"]["first_accepted_rank"] == 2
    assert case["llama_sampling4"]["dpo"]["first_accepted_rank"] is None
    assert audit.read(out / "summary.json")["selected_epoch_unchanged"] == 0
    assert len(audit.records(out / "source_responses.jsonl")) == 20
    assert all(audit.digest(Path(p)) == h for p,h in before.items())
    assert not list((comparison.parent / "submission").rglob("__pycache__"))
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == {"cases.jsonl","source_responses.jsonl","report.md","summary.json","manifest.json"}
    with pytest.raises(ValueError,match="new export"): audit.export(comparison,out,archive)


def test_initial_change_is_separate_from_same_initial_expansion_deletion():
    before = dict(draw=1,seed=1,answers=[dict(answer="Accepted",official_accepted=True)])
    after = dict(draw=1,seed=1,answers=[dict(answer="Other wording",official_accepted=False)])
    result = audit.compare_draws([before],[after])[0]
    assert not result["same_initial_text"] and not result["accepted_alternative_removed_with_same_initial"]
    assert result["removed_accepted_expressions"] == before["answers"]


@pytest.mark.parametrize("tamper",["bank","duplicate_score","missing_draw","false_label"])
def test_corrupt_sources_or_nonreproducible_metrics_fail_closed(bundle,tamper):
    comparison,out,archive = bundle
    state = audit.read(comparison / "manifest.json")
    if tamper == "bank":
        (comparison.parent / "llama31/responses.jsonl").write_text("changed")
    else:
        relative = {"duplicate_score":"comparison_per_question.json","missing_draw":"epoch-1/sampling4/generations.jsonl","false_label":"official/official_candidate_scores.jsonl"}[tamper]
        p=comparison / relative
        if tamper == "duplicate_score":
            d=audit.read(p);d["llama_epoch1_sampling4"].append(d["llama_epoch1_sampling4"][0]);write(p,d)
        else:
            d=audit.records(p)
            if tamper == "missing_draw": d[0]["samples"].pop()
            else: d[1]["lenient_accuracy"]=0
            rows(p,d)
        state["output_sha256"][relative]=audit.digest(p);write(comparison / "manifest.json",state)
    with pytest.raises(ValueError): audit.export(comparison,out,archive)
    assert not out.exists() and not archive.exists()


def test_fresh_cli_needs_only_standard_library():
    script=ROOT / "scripts/export_expanded_dpo_coverage_loss.py"
    p=subprocess.run([sys.executable,"-S",str(script),"--help"],capture_output=True,text=True,timeout=30)
    assert p.returncode == 0,p.stderr
