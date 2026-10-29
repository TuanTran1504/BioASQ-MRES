#!/usr/bin/env python3
"""Recover six completed validation generations with the full audited Java runtime."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ("gadi_sft_8b_starter/scripts/select_generator_feedback_dpo.py",
           "scripts/evaluate_generator_form_designs.py")
RUNTIME_HELPER = "scripts/generator_form_scorer_runtime.py"
MODELS = ("llama31","qwen3","ministral3")
DESIGNS = ("model_specific","shared")


def read(path): return json.loads(path.read_text(encoding="utf-8"))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def verify(root,hashes):
    root=root.resolve()
    for name,value in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path)!=value:
            raise ValueError("Frozen recovery input changed: "+name)


def original_inputs(submission):
    p=read(submission/"prepared.json")
    if digest(Path(p["bundle"])/"manifest.json")!=p["bundle_manifest_sha256"]:
        raise ValueError("Original reviewed data package changed")
    verify(submission/"code",p["code_hashes"])
    if digest(submission/"compare.pbs")!=p["comparison_job_sha256"]:
        raise ValueError("Original comparison job changed")
    with (submission/"jobs.tsv").open() as f: jobs=list(csv.DictReader(f,delimiter="\t"))
    panels={}
    hashes={"prepared.json":digest(submission/"prepared.json"),"jobs.tsv":digest(submission/"jobs.tsv")}
    common_ids=None
    for model in MODELS:
        for design in DESIGNS:
            child=submission/model/design
            if digest(child/"prepared.json")!=p["child_prepared_sha256"][str(child)]:
                raise ValueError("Original child preparation changed")
            verify(child/"code",read(child/"prepared.json")["code_hashes"])
            paths={r["stage"]:Path(r["output"]).resolve() for r in jobs if r["model"]==model and r["design"]==design}
            if not {"train","validate"}<=set(paths): raise ValueError("Missing original training/validation record")
            for path in paths.values():
                if not path.is_relative_to(child): raise ValueError("Original job output escaped its child")
            trained,saved=read(paths["train"]/"manifest.json"),read(paths["validate"]/"manifest.json")
            if (trained["status"]!="complete" or trained["smoke_test"] or trained["phase"]!="save_adapter"
                    or trained["global_steps"]!=99 or trained["trained_pair_counts"]!={"train":132,"validation":96}
                    or trained["prepared_sha256"]!=digest(child/"prepared.json")):
                raise ValueError("Need complete original 132-pair/99-update training")
            if (saved["status"]!="failed" or saved["smoke_test"] or saved["prepared_sha256"]!=digest(child/"prepared.json")
                    or saved["submission"]!=str(child) or saved["training"]!=str(paths["train"])
                    or saved["pool_policy"]!="verbatim surfaces"
                    or saved["training_manifest_sha256"]!=digest(paths["train"]/"manifest.json")
                    or "com/google/gson/stream/JsonReader" not in saved.get("error","")
                    or not ("NoClassDefFoundError" in saved["error"] or "ClassNotFoundException" in saved["error"])):
                raise ValueError("Recovery requires the known missing-Gson scoring failure for all six runs")
            ids=set(saved["question_ids"])
            if len(saved["question_ids"])!=144 or len(ids)!=144 or (common_ids is not None and common_ids!=ids):
                raise ValueError("Validation144 membership differs")
            common_ids=ids
            verify(paths["train"]/"adapter",trained["adapter_files_sha256"])
            for epoch in range(1,4):
                for mode,count in (("greedy",1),("sampling4",4)):
                    directory=paths["validate"]/f"epoch-{epoch}"/mode
                    rows=[json.loads(x) for x in (directory/"generations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
                    if (read(directory/"status.json")["status"]!="complete" or len(rows)!=144
                            or {r["question_id"] for r in rows}!=ids or any(len(r["samples"])!=count for r in rows)):
                        raise ValueError("Incomplete saved generation; no GPU fallback is permitted")
            # Pin all existing training/validation artifacts, including epoch
            # weights, tokenizer files, raw responses and failure metadata.
            train_files=[paths["train"]/"manifest.json",*(paths["train"]/"adapter").glob("*")]
            for checkpoint in (paths["train"]/"trainer_output").glob("checkpoint-*"):
                train_files.extend(checkpoint/n for n in ("trainer_state.json","adapter_config.json","adapter_model.safetensors"))
            for path in [*train_files,*paths["validate"].rglob("*")]:
                if path.is_file(): hashes[path.relative_to(submission).as_posix()]=digest(path)
            hashes[(child/"prepared.json").relative_to(submission).as_posix()]=digest(child/"prepared.json")
            panels[model+"/"+design]={"training":str(paths["train"]),"saved":str(paths["validate"])}
    return p,panels,hashes


def verify_corrections(original, corrected):
    # Everything after pool construction remains the original scoring/selection
    # implementation. Recovery only changes how completed draws are loaded.
    marker="        contexts = {e[\"question_id\"]: e for e in examples}"
    old=(original/PATCHES[0]).read_text(encoding="utf-8")
    new=(corrected/PATCHES[0]).read_text(encoding="utf-8")
    if old.count(marker)!=1 or new.count(marker)!=1 or old.split(marker)[1]!=new.split(marker)[1]:
        raise ValueError("Recovery changed frozen scoring or checkpoint selection")
    def definitions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=definitions(original/PATCHES[1]),definitions(corrected/PATCHES[1])
    for name in ("exact_pools","choose_coverage","selector_native","contrast"):
        if a[name]!=b[name]: raise ValueError("Recovery changed frozen pool/coverage rules: "+name)


def prepare(submission,recovery,repository=ROOT):
    if recovery.exists(): raise FileExistsError(recovery)
    original,panels,hashes=original_inputs(submission)
    sys.path.insert(0,str(repository))
    from scripts import generator_form_scorer_runtime as runtime
    jar=runtime.source(Path(original["bundle"]))
    if digest(jar)!=original["scorer_sha256"]: raise ValueError("Recovery evaluator differs from original frozen JAR")
    runtime.smoke(jar)
    recovery.mkdir(parents=True)
    code=dict(original["code_hashes"])
    for name in code:
        target=recovery/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(submission/"code"/name,target)
    for name in (*PATCHES,RUNTIME_HELPER):
        target=recovery/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repository/name,target);code[name]=digest(target)
    verify_corrections(submission/"code",recovery/"code")
    scorer_hashes=runtime.freeze(jar,recovery)
    shutil.copy2(Path(__file__),recovery/"runner.py")
    shutil.copy2(repository/"gadi_sft_8b_starter/jobs/recover_generator_form_scoring.pbs",recovery/"job.pbs")
    write(recovery/"prepared.json",{"submission":str(submission),"panels":panels,"original_files_sha256":hashes,
        "original_code_sha256":original["code_hashes"],"code_hashes":code,"scorer_files_sha256":scorer_hashes,
        "runner_sha256":digest(recovery/"runner.py"),"job_sha256":digest(recovery/"job.pbs"),
        "correction":"Restore audited scorer distribution/libraries; reuse all saved epoch generations on CPU",
        "training":False,"new_generation":False,"GPU_jobs":0,"paid_judging":False})
    print("Prepared one CPU recovery: all 36 saved generation panels complete; full scorer runtime smoke passed.",flush=True)


def validate(recovery):
    p=read(recovery/"prepared.json")
    verify(recovery,{"runner.py":p["runner_sha256"],"job.pbs":p["job_sha256"],**p["scorer_files_sha256"]})
    verify(recovery/"code",p["code_hashes"])
    original=Path(p["submission"])
    verify(original,p["original_files_sha256"])
    verify(original/"code",p["original_code_sha256"])
    verify_corrections(original/"code",recovery/"code")
    expected={**p["original_code_sha256"],**{n:digest(recovery/"code"/n) for n in (*PATCHES,RUNTIME_HELPER)}}
    if expected!=p["code_hashes"]: raise ValueError("Unexpected recovery code change")
    return p


def run(recovery,output):
    if output.exists(): raise FileExistsError(output)
    p=validate(recovery);prepared_sha=digest(recovery/"prepared.json")
    original=Path(p["submission"])
    sys.path.insert(0,str(recovery/"code"))
    sys.path.insert(1,str(recovery/"code/gadi_sft_8b_starter/scripts"))
    # This entry point has not imported evaluation modules before selecting the
    # pinned recovery snapshot. CPU recovery cannot import a GPU training worker.
    from scripts import evaluate_generator_form_designs as evaluator
    from select_generator_feedback_dpo import select
    from run_generator_form_designs import validate_child
    from scripts import generator_form_scorer_runtime as runtime
    if not Path(evaluator.__file__).resolve().is_relative_to((recovery/"code").resolve()):
        raise ValueError("Recovery must run in a fresh process using its frozen evaluation code")
    output.mkdir(parents=True)
    state={"status":"running","recovery":str(recovery),"prepared_sha256":prepared_sha,
        "training":False,"new_generation":False,"GPU_jobs":0,"paid_judging":False,"completed_panels":[]}
    write(output/"manifest.json",state)
    try:
        jar=recovery/"scorer/BioASQEvaluation.jar";runtime.smoke(jar)
        paths={}
        for name,panel in p["panels"].items():
            model,design=name.split("/");child=original/model/design
            validate_child(child)
            destination=output/model/design
            select(child,Path(panel["training"]),destination,native_validator=evaluator.selector_native,
                checkpoint_chooser=evaluator.choose_coverage,pool_builder=evaluator.exact_pools,preserve_surfaces=True,
                jar_path_override=jar,saved_generation_root=Path(panel["saved"]))
            paths[model,design]=destination
            state["completed_panels"].append(name);write(output/"manifest.json",state)
        evaluator.compare(original,output/"comparison",validation_paths=paths,jar_path_override=jar)
        validate(recovery)
        if digest(recovery/"prepared.json")!=prepared_sha: raise ValueError("Recovery metadata changed")
        state.update(status="complete",report=str(output/"comparison/report.md"),
            output_sha256={x.relative_to(output).as_posix():digest(x) for x in output.rglob("*") if x.is_file() and x!=output/"manifest.json"})
    except BaseException as exc:
        state.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally: write(output/"manifest.json",state)
    print("Recovery report:",state["report"],flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--mode",choices=("prepare","run"),required=True)
    a.add_argument("--submission",type=Path)
    a.add_argument("--recovery",type=Path,required=True)
    a.add_argument("--output",type=Path)
    x=a.parse_args()
    if x.mode=="prepare":
        if not x.submission: a.error("--submission is required")
        prepare(x.submission.resolve(),x.recovery.resolve())
    else:
        run(x.recovery.resolve(),(x.output or x.recovery/f"run-{os.environ.get('PBS_JOBID','manual')}").resolve())
