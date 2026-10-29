"""Matched preference designs, native provenance, coverage selection and job graph."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))
from scripts import prepare_generator_form_designs as designs
from scripts import evaluate_generator_form_designs as evaluation
import run_generator_form_designs as runner


def pairs():
    return [dict(question_id=f"q{i}", model_key=m, pair_id=f"{m}-{i}",
        prompt=[{"role": "user", "content": "Question and evidence"}],
        chosen=" accepted ", rejected=f"other {i % 2}",
        chosen_id=f"{m}-{i}-a", rejected_id=f"{m}-{i}-b",
        source_pair_ids=[f"{m}-{i}"], source_objectives=["sampled_response_coverage"])
        for m in designs.data.MODELS for i in range(10)]


def test_size_matched_designs_deduplicate_cross_author_pairs_and_keep_origins():
    rows = pairs()
    splits = {f"q{i}": "train" if i < 8 else "inner_validation" for i in range(10)}
    specific, shared, holdout = designs.construct(rows, splits, count=6)
    assert all(len(rr) == 6 for rr in specific.values()) and len(shared) == 6
    assert len(holdout) == 2  # identical content from three authors is one preference
    assert all(len(r["origins"]) == 3 for r in shared + holdout)
    assert {m: sum(p["sampling_origin"] == m for p in shared) for m in specific} == dict.fromkeys(specific, 2)
    assert len({p["preference_key"] for p in shared}) == 6
    assert not {r["question_id"] for r in shared} & {r["question_id"] for r in holdout}
    assert all(all(o["model_key"] == m for r in rr for o in r["origins"]) for m, rr in specific.items())
    assert designs.construct(list(reversed(rows)), splits, count=6) == (specific, shared, holdout)
    assert all(r["chosen"] == " accepted " for r in shared)


def test_shared_training_and_holdout_files_are_identical_for_every_target(tmp_path):
    rr = designs.pooled(pairs())
    designs.data.jsonl(tmp_path / "shared/train.jsonl", rr[:6])
    designs.data.jsonl(tmp_path / "inner_validation.jsonl", rr[-2:])
    a = [runner.preference_rows(tmp_path, m, "shared") for m in designs.data.MODELS]
    assert a[0] == a[1] == a[2]
    with pytest.raises(ValueError): runner.preference_rows(tmp_path, "unknown", "shared")


def test_export_reconstructs_every_sample_and_detects_changed_shared_file(tmp_path, monkeypatch):
    inspection,selector,output = (tmp_path / n for n in ("inspection","selector","output"))
    designs.data.write(selector / "manifest.json", {"output_sha256":{}})
    state = dict(inspection_scope="all_pairs",uninspected_pair_records=0,
        selector_manifest_sha256=designs.data.digest(selector / "manifest.json"))
    designs.data.write(inspection / "manifest.json",state)
    designs.data.jsonl(inspection / "selected_pairs.jsonl",pairs())
    designs.data.write(inspection / "question_split.json",{f"q{i}":"train" if i<8 else "inner_validation" for i in range(10)})
    designs.data.jsonl(selector / "train_inputs.jsonl",[{"question_id":f"q{i}"} for i in range(10)])
    designs.data.jsonl(selector / "validation_inputs.jsonl",[{"question_id":"v"}])
    designs.data.write(selector / "heldout_ids.json",{"dev160":["d"],"test95":["t"]})
    designs.data.write(selector / "source_provenance.json",{})
    monkeypatch.setattr(designs.reviewed,"validate",lambda p: designs.data.read(p / "manifest.json"))
    summary = designs.build(inspection,selector,output,count=6)
    assert summary["training_launched"] is False
    assert designs.validate(output)["training_ready"] is False
    assert designs.data.digest(output / "selector_audit_manifest.json") == state["selector_manifest_sha256"]
    (output / "shared/train.jsonl").write_text("changed")
    with pytest.raises(ValueError): designs.validate(output)


def native_rows(rows):
    return {p[k + "_id"]: dict(split="train", question_id=p["question_id"], model_key=p["model_key"],
                               raw_response=p[k], prompt=p["prompt"])
            for p in rows for k in ("chosen", "rejected")}


def test_shared_responses_keep_author_identity_and_train_holdout_is_disjoint():
    rr = pairs()
    splits = {"train": designs.pooled([r for r in rr if r["question_id"] == "q0"]),
              "validation": designs.pooled([r for r in rr if r["question_id"] == "q9"])}
    native = native_rows(rr)
    runner.verify_preferences_native(splits, native, {f"q{i}" for i in range(10)})
    native[rr[0]["chosen_id"]]["raw_response"] = "accepted"  # whitespace is provenance too
    with pytest.raises(ValueError, match="provenance"):
        runner.verify_preferences_native(splits, native, {f"q{i}" for i in range(10)})
    with pytest.raises(ValueError, match="outside fitting"):
        runner.verify_preferences_native(splits, native, {"q9"})
    splits["validation"] = splits["train"]
    with pytest.raises(ValueError, match="leakage"):
        runner.verify_preferences_native(splits, native_rows(rr), {f"q{i}" for i in range(10)})


class TargetTokenizer:
    def __init__(self, marker): self.marker = marker
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        # Common assistant prefix; different target tokenization is intentional.
        prefix = [self.marker, 7, 8]
        return prefix if add_generation_prompt else prefix + list(messages[-1]["content"].encode()) + [9]


def test_foreign_completions_render_with_target_tokenizer_without_truncation():
    rows = {"train": designs.pooled(pairs())[:2], "validation": designs.pooled(pairs())[-1:]}
    a = runner.preflight(TargetTokenizer(1), rows, {"chat_template_kwargs": {}}, 100)
    assert a["train"] > 4 and a["validation"] > 4
    from run_expansion_dpo_8b import render_preference, training_schedule
    r = rows["train"][0]
    assert render_preference(TargetTokenizer(2), r, {}, 100)["chosen_input_ids"][0] == 2
    with pytest.raises(ValueError, match="refusing truncation"):
        runner.preflight(TargetTokenizer(2), rows, {"chat_template_kwargs": {}}, 4)
    assert training_schedule(dict(gradient_accumulation_steps=4,num_train_epochs=3,warmup_ratio=.1),132) == (99,10)


def test_scorer_comes_from_original_audit_and_changed_jar_is_rejected(tmp_path):
    original, bundle = tmp_path / "original", tmp_path / "bundle"
    jar = original / "scorer/BioASQEvaluation.jar"
    jar.parent.mkdir(parents=True)
    with zipfile.ZipFile(jar,"w") as z: z.writestr("META-INF/MANIFEST.MF","Manifest-Version: 1.0\n\n")
    designs.data.write(original / "prepared.json", {"files_sha256": {"scorer/BioASQEvaluation.jar": designs.data.digest(jar)}})
    designs.data.write(bundle / "selector_audit_manifest.json", {"submission": str(original),
        "prepared_sha256": designs.data.digest(original / "prepared.json")})
    assert runner.scorer_source(bundle) == jar
    jar.write_bytes(b"changed")
    with pytest.raises(ValueError): runner.scorer_source(bundle)


def test_prepare_checks_all_six_target_contexts_before_freezing_common_allocation(tmp_path,monkeypatch):
    import run_base_original_prompt as inference
    repo,bundle,sub = (tmp_path/n for n in ("repo","bundle","submission"))
    protocol=dict(seed=3407,training_pairs=132,num_train_epochs=3,gradient_accumulation_steps=4,
        review_scope="official_form",max_seq_length=8704,warmup_ratio=.1)
    designs.data.write(bundle / "manifest.json",{"fixture":True})
    designs.data.write(bundle / "training_protocol.json",protocol)
    designs.data.write(repo / "results/generator_form_full_inspection_20261029.json",{"full_export_manifest_sha256":"review"})
    designs.data.write(repo / "results/generator_form_designs_20261029.json",{"bundle_manifest_sha256":designs.data.digest(bundle / "manifest.json")})
    for stage in ("train","validate","compare"):
        path=repo/f"gadi_sft_8b_starter/jobs/{stage}_generator_form_designs.pbs"
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(stage)
    jar=repo/"runtime/BioASQEvaluation.jar";jar.parent.mkdir(parents=True)
    with zipfile.ZipFile(jar,"w") as z: z.writestr("META-INF/MANIFEST.MF","Manifest-Version: 1.0\nClass-Path: lib/gson.jar\n\n")
    (jar.parent/"lib").mkdir();(jar.parent/"lib/gson.jar").write_bytes(b"frozen dependency")
    monkeypatch.setattr(runner.designs,"validate",lambda _: {"inspection_manifest_sha256":"review"})
    monkeypatch.setattr(runner,"scorer_source",lambda _:jar)
    monkeypatch.setattr(runner.scorer_runtime,"smoke",lambda _:None)
    calls=[]
    def native(repository,package,model,design):
        calls.append((model,design))
        source=dict(adapter=model,base_snapshot=model,chat_template_kwargs={})
        row={**designs.pooled(pairs())[0],"chosen":"x"*(9000 if model=="qwen3" and design=="shared" else 40)}
        splits={"train":[{**row,"question_id":f"q{i}"} for i in range(132)],
                "validation":[{**row,"question_id":f"h{i}"} for i in range(96)]}
        return {"pair_counts":{"train":132,"validation":96}},splits,source,{"banks":{}}
    monkeypatch.setattr(runner,"native_inputs",native)
    monkeypatch.setattr(runner,"snapshot",lambda *a:{})
    monkeypatch.setattr(inference,"load_text_tokenizer",lambda _:TargetTokenizer(1))
    monkeypatch.setattr(inference,"native_context",lambda _:16384)
    def snapshot(destination):
        worker=destination/"inference_code/scripts/run_single_answer_sampling.py"
        worker.parent.mkdir(parents=True);worker.write_text("frozen worker")
        return {"directory":"inference_code","files_sha256":{"scripts/run_single_answer_sampling.py":designs.data.digest(worker)}}
    monkeypatch.setattr(inference,"snapshot_inference_code",snapshot)
    runner.prepare(repo,bundle,sub)
    state=designs.data.read(sub/"prepared.json")
    assert len(calls)==6 and len(state["children"])==6
    assert state["common_context_allocation"]==9216 and state["planned_steps_per_run"]==99
    for child in map(Path,state["children"]):
        p=designs.data.read(child/"prepared.json")
        assert p["configuration"]["max_seq_length"]==9216
        assert p["source"]["adapter"]==p["configuration"]["model_key"]
        assert p["job_hashes"]["scorer/BioASQEvaluation.jar"]==designs.data.digest(jar)
        assert "scorer/lib/gson.jar" in p["job_hashes"]
    child=Path(state["children"][0]);(child/"scorer/lib/gson.jar").write_bytes(b"changed")
    with pytest.raises(ValueError): runner.validate_child(child)


def response(answers):
    return json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                   for i, a in enumerate(answers)]})


def draws(models=("llama31", "qwen3", "ministral3")):
    return [dict(split="validation",question_id="q",model_key=m,draw=d,
        raw_response=response(["HER2", "HER-2", " HER2 "]),output_tokens=24)
        for m in models for d in range(5)]


def test_exact_pool_keeps_wording_variants_and_fixed_draw_denominator():
    rows = draws()
    rows[1]["raw_response"] = ""  # one failed sampling response still counts
    pp, _ = evaluation.exact_pools(rows, {"validation": {"q": {}}})
    rr = pp["validation"]["max5_sampling4"]
    assert [r["answer"] for r in rr] == ["HER2", "HER-2", " HER2 "]
    assert all(r["sample_frequencies"] == {"llama31": .75, "qwen3": 1, "ministral3": 1} for r in rr)
    assert all(r["source_ranks"]["llama31"] == i for i,r in enumerate(rr,1))
    solo, _ = evaluation.exact_pools(draws(("qwen3",)), {"validation": {"q": {}}})
    assert solo["validation"]["max5_greedy"][0]["sources"] == ["qwen3"]
    with pytest.raises(ValueError, match="Missing"):
        evaluation.exact_pools(rows[:-1], {"validation": {"q": {}}})
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.exact_pools(rows + [rows[0]], {"validation": {"q": {}}})


def test_coverage_selection_allows_longer_outputs_and_partial_losses_but_guards_failures():
    base = dict(epoch=0, ensemble_full_pool_coverage=.7,focal_full_pool_coverage=.6,mrr_at5=.58,
        mean_output_tokens=60,ensemble_full_pool_lost=0,llama_full_pool_lost=0,
        generation_quality=dict(empty_responses=0,ceiling_responses=0,surplus_responses=0))
    trial = {**copy.deepcopy(base), "epoch":1,"ensemble_full_pool_coverage":.72,"mrr_at5":.57,
             "mean_output_tokens":80,"ensemble_full_pool_lost":1,"llama_full_pool_lost":1}
    assert evaluation.choose_coverage([base,trial])["epoch"] == 1
    trial["generation_quality"]["ceiling_responses"] = 1
    assert evaluation.choose_coverage([base,trial])["epoch"] == 0
    assert evaluation.choose_coverage([base,{**copy.deepcopy(base),"epoch":2}])["epoch"] == 0


def test_paired_contrast_sign_and_coverage_gains_are_consistent():
    a = [dict(question_id="q",mrr_at5=0.,coverage_at5=0.,coverage_full_pool=0.)]
    b = [dict(question_id="q",mrr_at5=.5,coverage_at5=1.,coverage_full_pool=1.)]
    c = evaluation.contrast("specific","shared",{"specific":a,"shared":b})
    assert c["metrics"]["mrr_at5"]["difference"] == .5
    assert c["coverage_transitions"]["coverage_full_pool"]["gained"] == 1


@pytest.mark.parametrize("changed_baseline",[False,True])
def test_comparison_collects_selected_and_fixed_epochs_and_checks_common_baseline(tmp_path,monkeypatch,changed_baseline):
    sub,out = tmp_path / "sub",tmp_path / "out"
    (sub / "compare.pbs").parent.mkdir(parents=True)
    (sub / "compare.pbs").write_text("compare")
    baseline = [dict(question_id=f"v{i}",mrr_at5=.5,coverage_at5=1.,coverage_full_pool=1.) for i in range(144)]
    jobs,childpins = [],{}
    for model in designs.data.MODELS:
        for design in designs.DESIGNS:
            child = sub / model / design
            designs.data.write(child / "prepared.json",{})
            childpins[str(child)] = designs.data.digest(child / "prepared.json")
            val = child / "validation-1"
            q = {f"epoch{e}_{mode}_max5_reranker":copy.deepcopy(baseline) for e in range(4) for mode in ("greedy","sampling4")}
            if changed_baseline and model=="ministral3" and design=="shared": q["epoch0_greedy_max5_reranker"][0]["mrr_at5"] = 0.
            conditions = {n:{"metrics":{k:sum(r[k] for r in rr)/144 for k in rr[0] if k!="question_id"}} for n,rr in q.items()}
            summary = dict(pool_policy="verbatim surfaces",selected_epoch=1,conditions=conditions,
                checkpoint_scores=[dict(epoch=e,focal_full_pool_coverage=1.,mean_output_tokens=60.) for e in range(4)])
            designs.data.write(val / "comparison_summary.json",summary)
            designs.data.write(val / "comparison_per_question.json",q)
            designs.data.write(val / "manifest.json",dict(status="complete",smoke_test=False,
                question_ids=[r["question_id"] for r in baseline],submission=str(child),
                prepared_sha256=childpins[str(child)],output_sha256={n:designs.data.digest(val/n) for n in ("comparison_summary.json","comparison_per_question.json")}))
            jobs.append(dict(model=model,design=design,stage="validate",output=str(val)))
    designs.data.write(sub / "prepared.json",dict(code_hashes={},child_prepared_sha256=childpins,
        comparison_job_sha256=designs.data.digest(sub / "compare.pbs")))
    with (sub / "jobs.tsv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=tuple(jobs[0]),delimiter="\t");writer.writeheader();writer.writerows(jobs)
    def joint(*args):
        names = ["baseline"]+[f"{d}_{w}" for d in designs.DESIGNS for w in ("selected","final_epoch")]
        q = {f"joint_{n}_{mode}_max5_reranker":copy.deepcopy(baseline) for n in names for mode in ("greedy","sampling4")}
        conditions = {n:{"metrics":{"mrr_at5":.5,"coverage_at5":1.,"coverage_full_pool":1.}} for n in q}
        return conditions,q
    monkeypatch.setattr(evaluation,"joint_scores",joint)
    monkeypatch.setattr(evaluation,"contrast",lambda a,b,q:{"before":a,"after":b})
    if changed_baseline:
        with pytest.raises(ValueError,match="Baseline"): evaluation.compare(sub,out)
        assert designs.data.read(out / "manifest.json")["status"] == "failed"
    else:
        evaluation.compare(sub,out)
        result = designs.data.read(out / "comparison_summary.json")
        assert len(result["conditions"]) == 36
        assert result["conditions"]["qwen3_shared_final_epoch_sampling4"]["epoch"] == 3
        assert result["conditions"]["qwen3_shared_selected_sampling4"]["epoch"] == 1
        assert "joint_selected_sampling4_shared_minus_specific" in result["paired_contrasts"]
        assert designs.data.read(out / "manifest.json")["status"] == "complete"


@pytest.mark.parametrize("fail", [False, True])
def test_launcher_preflights_before_six_chains_and_cpu_comparison(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    bundle = tmp_path / "bundle"; bundle.mkdir()
    source = ROOT / "gadi_sft_8b_starter/scripts/submit_generator_form_designs.sh"
    stub = '''
module() { :; }
source() { :; }
python3() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift; mkdir -p "$1"; return 0; fi
    shift
  done
}
qsub() {
  printf '%s\\n' "$*" >> "$TEST_LOG"
  printf '%s.gadi-pbs\\n' "$(wc -l < "$TEST_LOG")"
}
'''
    script = scripts / source.name
    script.write_text(source.read_text().replace("${USER}", "${TEST_USER}")
        .replace("set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "jobs.log"
    result = subprocess.run([executable, script.as_posix(), bundle.as_posix()],capture_output=True,text=True,
        env={**os.environ,"TEST_USER":"test","TEST_LOG":log.as_posix(),"TEST_FAIL":str(int(fail))},timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
        return
    assert result.returncode == 0, result.stderr
    lines = log.read_text().splitlines()
    assert len(lines) == 19
    for i in range(6):
        assert f"depend=afterok:{3*i+1}.gadi-pbs" in lines[3*i+1]
        assert f"depend=afterok:{3*i+2}.gadi-pbs" in lines[3*i+2]
    assert "depend=afterok:" + ":".join(f"{i}.gadi-pbs" for i in range(3,19,3)) in lines[-1]
    record = next(tmp_path.glob("outputs/generator_form_designs/submission-*/jobs.tsv"))
    with record.open() as f: jobs = list(csv.DictReader(f,delimiter="\t"))
    assert len(jobs) == 19 and len({r["output"] for r in jobs}) == 19
    assert {(r["model"],r["design"]) for r in jobs if r["stage"] == "validate"} == {
        (m,d) for m in designs.data.MODELS for d in designs.DESIGNS}


@pytest.mark.parametrize("script", ["scripts/evaluate_generator_form_designs.py",
                                   "gadi_sft_8b_starter/scripts/run_generator_form_designs.py"])
def test_fresh_cli_imports(script):
    result = subprocess.run([sys.executable,str(ROOT / script),"--help"],cwd=ROOT,capture_output=True,text=True,timeout=40)
    assert result.returncode == 0, result.stderr
