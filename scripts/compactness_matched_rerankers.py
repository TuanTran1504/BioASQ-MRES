#!/usr/bin/env python3
"""Ten-sample fitting banks, two matching logistic selectors, fixed test95 rescore."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gadi_sft_8b_starter/scripts"))

import joblib
import numpy as np
import sklearn

from prepare_matched_8b_sft import read, records, write, digest
from run_extractive_expansion_8b import write_jsonl, parse_equivalent_response
from run_single_answer_sampling import sample_seed, expansion_candidates
from run_base_original_prompt import full_input_plan, snapshot_inference_code, inference_worker
from run_compactness_test95 import CONFIGURATIONS, MODELS, RUN_FILES, validate_epoch_one, validate_prepared
from select_generator_feedback_dpo import verify_generated
from scripts import train_sft_ensemble_reranker as rr
from scripts.prepare_reranker_guided_dpo import load_inputs
from scripts.ablate_reranker_relation_labels import verify_files
from scripts.analyze_sft_ensemble_test import merge_pool
from scripts.evaluate_sft_ensemble_rerankers import rank_frozen, load_frozen
from scripts.analyze_generator_feedback_evaluation import contrast, coverage_transitions, METRICS
from scripts.analyze_compactness_test95 import add_batches
from scripts.compare_expansion_pool_rerankers import question_weights
from scripts.train_candidate_reranker_pilot import fit_components

OUTPUT = "outputs/compactness_matched_rerankers"
MODES = ("greedy", "sampling10")
SELECTORS = ("strict", "alternative")
DRAW_IDS = set(range(11))  # 0 = greedy; 1..10 are stochastic seed indices.
C_GRID = [0.1, 1.0, 10.0]


def save_state(path, value):
    """A killed job must leave its last committed chunk index readable."""
    temporary = path.with_name(path.name + ".tmp")
    write(temporary, value)
    os.replace(temporary, path)


def pin_files(directory, hashes, pins):
    verify_files(directory, hashes)
    pins.update({str(directory / n): h for n, h in hashes.items()})


def policies(model):
    # Only these SFT companions occur in the strict 0/1/0 ensemble.
    return ("sft", "dpo") if model != "qwen3" else ("dpo",)


def input_panel(contexts):
    return [{"question_id": q, "question": e["question"], "snippets": [
        {"snippet_id": s.get("snippet_id", s.get("id")), "text": s["text"]}
        for s in e["snippets"]]} for q, e in sorted(contexts.items())]


def check_membership(examples, heldout_ids, expected=None):
    groups = {s: set(e) for s, e in examples.items()}
    if expected is not None and {s: len(e) for s, e in groups.items()} != expected:
        raise ValueError("Wrong fitting/validation membership")
    if groups["train"] & groups["validation"] or any(set(heldout_ids) & e for e in groups.values()):
        raise ValueError("Fitting, validation and held-out questions must be disjoint")


def raw_row(qid, split, draw, sample):
    return {"question_id": qid, "split": split, "draw": draw,
            "seed": sample["seed"], "raw_response": sample["raw_response"],
            "output_tokens": sample["output_tokens"],
            "generation_seconds": sample.get("generation_seconds", 0)}


def prepare(repository, comparison, submission):
    """Pin complete prior experiment and reuse validated native response banks."""
    old_state = read(comparison / "manifest.json")
    if old_state.get("status") != "complete" or old_state.get("configurations") != CONFIGURATIONS:
        raise ValueError("Need the completed, fixed compactness test95 comparison")
    test_submission = Path(old_state["submission"])
    old = validate_prepared(test_submission)
    if old_state["prepared_sha256"] != digest(test_submission / "prepared.json"):
        raise ValueError("Test result/preparation mismatch")
    test_pins = {str(comparison / "manifest.json"): digest(comparison / "manifest.json"),
                 str(test_submission / "prepared.json"): digest(test_submission / "prepared.json")}
    pin_files(comparison, old_state["output_sha256"], test_pins)
    for n in ("test_inputs.jsonl", "test_gold.jsonl"):
        test_pins[str(test_submission / n)] = digest(test_submission / n)
    heldout = records(test_submission / "test_inputs.jsonl")
    if len(heldout) != 95 or len({r["question_id"] for r in heldout}) != 95:
        raise ValueError("Need exactly test95")
    rerankers = Path(old["rerankers"])
    audit_cfg = read(repository / "gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json")
    _, _, _, _, rr_hashes, banks, data = load_inputs(repository, rerankers, audit_cfg)
    if rr_hashes != old["reranker_hashes"]:
        raise ValueError("Starting reranker differs from completed test experiment")
    raw = {m: records(Path(banks[m]["path"]) / "responses.jsonl") for m in MODELS}
    examples = rr.contexts_from_rows([{**r, "arm": "max5"} for rows in raw.values() for r in rows])
    check_membership(examples, [r["question_id"] for r in heldout], {"train": 1296, "validation": 144})
    _, aliases = data.panels(read(data.DEFAULT_CONFIG))
    panel = {s: input_panel(e) for s, e in examples.items()}
    pins, sources, dpo_sources, plans, saved = {}, {}, {}, {}, {}
    pin_files(rerankers, rr_hashes, pins)
    jar = repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    from src.utility.bioasq_official import _official_adapter_classes
    _official_adapter_classes(jar)
    pins[str(jar)] = digest(jar)
    all_inputs = panel["train"] + panel["validation"]
    for m in MODELS:
        source, dpo, _, _, extra = validate_epoch_one(Path(old["selections"][m]), m, repository)
        if source != old["sources"][m] or dpo != old["dpo_sources"][m]:
            raise ValueError("Epoch-one generator differs from tested source: " + m)
        sources[m], dpo_sources[m] = source, dpo
        pins.update(extra)
        for n in ("manifest.json", "responses.jsonl"):
            p = Path(banks[m]["path"]) / n; pins[str(p)] = digest(p)
        pin_files(Path(source["adapter"]), source["adapter_files_sha256"], pins)
        plan = full_input_plan(dpo, all_inputs, 512)
        sft_plan = full_input_plan(source, all_inputs, 512)
        if plan != sft_plan or plan["max_seq_length"] > 8704:
            raise ValueError("Native SFT/DPO tokenization or full-evidence allocation differs")
        plan["max_seq_length"] = 8704
        plans[m] = plan
        base_cfg = Path(source["base_snapshot"]) / "config.json"
        pins[str(base_cfg)] = digest(base_cfg)
        for r in raw[m]:
            if r["prompt_tokens"] != plan["expected_prompt_tokens"][r["question_id"]]:
                raise ValueError("Native SFT prompt token count differs")
        saved[m] = {"dpo": []}
        if "sft" in policies(m):
            # Native banks use draw0 greedy and draws1..4 with seed indices2..5.
            saved[m]["sft"] = [raw_row(r["question_id"], r["split"],
                0 if r["draw"] == 0 else r["draw"] + 1, r) for r in raw[m]]
        selection = Path(old["selections"][m])
        validation = records(selection / "validation_inputs.jsonl")
        if {r["question_id"]: r for r in validation} != {r["question_id"]: r for r in panel["validation"]}:
            raise ValueError("Reused DPO validation evidence differs")
        for mode in ("greedy", "sampling4"):
            run = selection / "epoch-1" / mode
            cfg = read(run / "config.json")
            old_plan = {k: cfg[k] for k in ("max_seq_length", "expected_prompt_tokens", "expected_prompt_ids_sha256")}
            if old_plan["max_seq_length"] != 8704 or any(
                any(v != plan[k][q] for q, v in old_plan[k].items())
                for k in ("expected_prompt_tokens", "expected_prompt_ids_sha256")):
                raise ValueError("Saved DPO validation tokenization differs")
            generated = verify_generated(run, validation, dpo, mode, old_plan)
            pin_files(run, {n: digest(run / n) for n in RUN_FILES}, pins)
            for r in generated:
                saved[m]["dpo"].extend(raw_row(r["question_id"], "validation",
                    0 if mode == "greedy" else s["draw"] + 1, s) for s in r["samples"])
        for policy, rows in saved[m].items():
            validate_raw(rows, examples, complete=False, policy=policy)
    submission.mkdir(parents=True, exist_ok=False)
    for s in examples:
        write_jsonl(submission / f"{s}_inputs.jsonl", panel[s])
    write(submission / "aliases.json", {q: aliases[q] for e in examples.values() for q in e})
    for m in MODELS:
        (submission / f"{m}_prompt.txt").write_text(sources[m]["system_prompt"], encoding="utf-8")
        for policy, rows in saved[m].items():
            write_jsonl(submission / f"{m}_{policy}_reused.jsonl", rows)
    for stage in ("generate", "train", "score"):
        shutil.copy2(repository / f"gadi_sft_8b_starter/jobs/{stage}_compactness_rerankers.pbs",
                     submission / f"{stage}.pbs")
    files = {p.name: digest(p) for p in submission.iterdir() if p.is_file()}
    inference = snapshot_inference_code(submission)
    write(submission / "prepared.json", {
        "repository": str(repository), "comparison": str(comparison), "test_submission": str(test_submission),
        "rerankers": str(rerankers), "sources": sources, "dpo_sources": dpo_sources, "input_plans": plans,
        "configurations": CONFIGURATIONS, "sampled_responses": 10, "C_grid": C_GRID, "seed": 3407,
        "question_counts": {s: len(e) for s, e in examples.items()}, "chunk_size": 64,
        "file_hashes": files, "external_hashes": pins, "test_hashes": test_pins,
        "code_hashes": rr.snapshot_code(repository, submission / "code"),
        "code_snapshot": inference, "worker_sha256": inference["files_sha256"]["scripts/run_single_answer_sampling.py"],
        "policy": "Fitting-only logistic fits; validation144 selects C on greedy/sampling10 equally; test95 scoring only",
    })
    print("Prepared 1296 fitting / 144 validation, one greedy plus ten samples per model/policy.", flush=True)
    print("Saved draws reused; new draws only. Two matching rerankers; existing test95 pools stay fixed.", flush=True)


def validate(submission, test=False):
    p = read(submission / "prepared.json")
    if (p["configurations"] != CONFIGURATIONS or p["sampled_responses"] != 10
            or p["C_grid"] != C_GRID or p["question_counts"] != {"train": 1296, "validation": 144}):
        raise ValueError("Declared ten-sample protocol changed")
    verify_files(submission, p["file_hashes"])
    verify_files(submission / "code", p["code_hashes"])
    for path, value in {**p["external_hashes"], **(p["test_hashes"] if test else {})}.items():
        if digest(Path(path)) != value:
            raise ValueError("Pinned input changed: " + path)
    inference_worker(submission, p)
    return p


def panels(submission):
    return {s: {r["question_id"]: r for r in records(submission / f"{s}_inputs.jsonl")}
            for s in ("train", "validation")}


def validate_raw(rows, examples, complete=True, policy="dpo"):
    seen = set()
    for r in rows:
        identity = (r["split"], r["question_id"], r["draw"])
        if (r["split"] not in examples or r["question_id"] not in examples[r["split"]]
                or type(r["draw"]) is not int or r["draw"] not in DRAW_IDS or identity in seen
                or r["seed"] != sample_seed(r["question_id"], max(1, r["draw"]), 3407)
                or not isinstance(r["raw_response"], str) or r["output_tokens"] < 0
                or r.get("generation_seconds", 0) < 0):
            raise ValueError("Invalid, duplicated, out-of-partition or mis-seeded response: " + policy)
        seen.add(identity)
    if complete and seen != {(s, q, d) for s, e in examples.items() for q in e for d in DRAW_IDS}:
        raise ValueError("Bank must contain all 11 draws for every fitting/validation question")


def missing_tasks(reused, examples, chunk_size=64):
    seen = {(r["split"], r["question_id"], r["draw"]) for r in reused}
    tasks = []
    for split, ee in examples.items():
        groups = defaultdict(list)
        for q in sorted(ee):
            missing = sorted(DRAW_IDS - {d for s, qq, d in seen if s == split and qq == q})
            # Greedy is a separate decoding condition. Consecutive sampled
            # indices can share a worker invocation and retain their true seeds.
            ranges = []
            for d in missing:
                if ranges and d != 1 and ranges[-1][-1] != 0 and d == ranges[-1][-1] + 1:
                    ranges[-1].append(d)
                else:
                    ranges.append([d])
            for indices in ranges:
                groups[tuple(indices)].append(q)
        for indices, qq in sorted(groups.items()):
            for start in range(0, len(qq), chunk_size):
                tasks.append({"split": split, "indices": list(indices), "question_ids": qq[start:start+chunk_size]})
    return tasks


def worker_config(submission, model, source, plan, task, inputs):
    greedy = task["indices"] == [0]
    return {"input": str(inputs), "prompt": str(submission / f"{model}_prompt.txt"),
        **({} if greedy else {"sampled_draw_indices": task["indices"]}),
        "response_mode": "equivalent_greedy" if greedy else "equivalent_sampling",
        "num_generations": len(task["indices"]), "sample_seed_offset": 0 if greedy else task["indices"][0] - 1,
        "temperature": 0 if greedy else .8, "top_p": .95, "top_k": 0,
        "seed": 3407, "max_new_tokens": 512, "max_candidates": 5,
        "model_loader": source["model_loader"], "chat_template_kwargs": source["chat_template_kwargs"],
        "execution_mode": source.get("execution_mode", "default"), "mark_snippets": False,
        "require_all_snippets": True, "max_seq_length": plan["max_seq_length"],
        "expected_prompt_tokens": {q: plan["expected_prompt_tokens"][q] for q in task["question_ids"]},
        "expected_prompt_ids_sha256": {q: plan["expected_prompt_ids_sha256"][q] for q in task["question_ids"]},
        "input_token_policy": "complete_evidence_auto_size", "prompt_version": "compactness-reranker-ten-samples-v1"}


def read_chunk(run, cfg, panel, source, task):
    saved = read(run / "config.json")
    if (any(saved.get(k) != v for k, v in cfg.items()) or saved.get("model_name") != source["adapter"]
            or saved.get("gold_blind_generation") is not True or saved.get("local_files_only") is not True
            or saved.get("question_count") != len(panel)
            or saved.get("input_sha256") != digest(Path(cfg["input"]))
            or saved.get("prompt_sha256") != digest(Path(cfg["prompt"]))
            or read(run / "status.json").get("status") != "complete"
            or records(run / "examples.jsonl") != panel):
        raise ValueError("Incomplete or changed chunk inputs/configuration")
    rows = records(run / "generations.jsonl")
    if len(rows) != len(panel) or [r["question_id"] for r in rows] != [e["question_id"] for e in panel]:
        raise ValueError("Duplicate or missing chunk questions")
    projected, candidates, invalid = [], [], []
    for r, e in zip(rows, panel):
        q = e["question_id"]; count = len(task["indices"])
        if (len(r["samples"]) != count or r["request_count"] != count or r["snippets_truncated"]
                or r["prompt_tokens"] != cfg["expected_prompt_tokens"][q]
                or r["included_snippets"] != len(e["snippets"]) or r["total_snippets"] != len(e["snippets"])
                or r["input_tokens"] != count * r["prompt_tokens"]):
            raise ValueError("Chunk dropped evidence or draws")
        for i, (s, draw) in enumerate(zip(r["samples"], task["indices"]), 1):
            if s["draw"] != i or s["seed"] != sample_seed(q, max(1, draw), 3407):
                raise ValueError("Chunk draw indices or seeds changed")
            try:
                aa, bad, _, _ = parse_equivalent_response(s["raw_response"], 5)
            except ValueError:
                aa, bad = [], []
            if (s["answers"] != aa or bool(s.get("parse_error")) == bool(aa)
                    or s["output_tokens"] < 0 or s["generation_seconds"] < 0):
                raise ValueError("Chunk parsing/cost differs from raw response")
            projected.append(raw_row(q, task["split"], draw, s))
            invalid.extend({"question_id": q, "draw": i, **b} for b in bad)
        if (r["output_tokens"] != sum(s["output_tokens"] for s in r["samples"])
                or abs(r["generation_seconds"] - sum(s["generation_seconds"] for s in r["samples"])) > 1e-6):
            raise ValueError("Chunk cost sums differ")
        candidates.extend({"question_id": q, **c} for c in expansion_candidates(r["samples"]))
    if records(run / "candidates.jsonl") != candidates or records(run / "invalid_candidates.jsonl") != invalid:
        raise ValueError("Chunk candidate serialization differs")
    return projected


def generate(submission, model, smoke=False):
    p = validate(submission); examples = panels(submission)
    output = submission / (f"smoke-{model}-{os.environ.get('PBS_JOBID', 'manual')}" if smoke else f"banks/{model}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists() and read(manifest).get("status") == "complete":
        load_bank(submission, model, examples)
        print(model + ": complete bank verified; no new generation", flush=True)
        return
    state = {"status": "running", "model_key": model, "smoke_test": smoke,
             "prepared_sha256": digest(submission / "prepared.json"), "sampled_responses": 10,
             "completed_chunks": {}, "policies": list(policies(model))}
    if manifest.exists():
        previous = read(manifest)
        if any(previous.get(k) != state[k] for k in ("model_key", "smoke_test", "prepared_sha256", "sampled_responses", "policies")):
            raise ValueError("Existing bank belongs to a different preparation")
        state["completed_chunks"] = previous["completed_chunks"]
    save_state(manifest, state)
    try:
        for policy in policies(model):
            source = p["sources" if policy == "sft" else "dpo_sources"][model]
            reused = records(submission / f"{model}_{policy}_reused.jsonl")
            tasks = missing_tasks(reused, examples, p["chunk_size"])
            if smoke:
                qq = sorted(examples["train"], key=lambda q: (-p["input_plans"][model]["expected_prompt_tokens"][q], q))[:4]
                tasks = [{"split": "train", "question_ids": qq, "indices": indices} for indices in ([0], [1, 2])]
            combined = list(reused)
            for i, task in enumerate(tasks):
                name = f"{policy}-{i:04d}"; directory = output / name
                panel = [examples[task["split"]][q] for q in task["question_ids"]]
                cfg = worker_config(submission, model, source, p["input_plans"][model], task, directory / "inputs.jsonl")
                run = directory / "run"
                if name in state["completed_chunks"]:
                    pin_files(directory, state["completed_chunks"][name], {})
                    combined.extend(read_chunk(run, cfg, panel, source, task))
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                if run.exists():
                    if (run / "status.json").exists() and read(run / "status.json").get("status") == "complete":
                        combined.extend(read_chunk(run, cfg, panel, source, task))
                        state["completed_chunks"][name] = {n: digest(directory / n) for n in
                            ("inputs.jsonl", "config.json", *("run/" + f for f in RUN_FILES))}
                        save_state(manifest, state)
                        continue
                    # Preserve an interrupted chunk for inspection; rerun only
                    # this chunk, never a previously committed complete chunk.
                    backup = directory / f"interrupted-{os.environ.get('PBS_JOBID', 'manual')}"
                    if backup.exists():
                        raise ValueError("Interrupted chunk backup already exists; use a fresh PBS job")
                    run.rename(backup)
                write_jsonl(directory / "inputs.jsonl", panel)
                write(directory / "config.json", cfg)
                subprocess.run([sys.executable, str(inference_worker(submission, p)), "--config", str(directory / "config.json"),
                    "--model-name", source["adapter"], "--output-dir", str(run), "--limit", str(len(panel)), "--preflight-all"], check=True)
                projected = read_chunk(run, cfg, panel, source, task)
                if smoke and not any(parse_answers(r["raw_response"]) for r in projected):
                    raise ValueError("No usable smoke responses")
                combined.extend(projected)
                state["completed_chunks"][name] = {n: digest(directory / n) for n in
                    ("inputs.jsonl", "config.json", *("run/" + f for f in RUN_FILES))}
                save_state(manifest, state)
                print(f"{model}/{policy}: completed chunk {i+1}/{len(tasks)}", flush=True)
            if not smoke:
                validate_raw(combined, examples, policy=policy)
                write_jsonl(output / f"{policy}_responses.jsonl", combined)
        validate(submission)
        state.update(status="complete", output_sha256={} if smoke else {
            f"{policy}_responses.jsonl": digest(output / f"{policy}_responses.jsonl") for policy in policies(model)})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_state(manifest, state)


def parse_answers(raw):
    try:
        return parse_equivalent_response(raw, 5)[0]
    except (ValueError, TypeError):
        return []


def load_bank(submission, model, examples):
    root = submission / "banks" / model; state = read(root / "manifest.json")
    if (state.get("status") != "complete" or state.get("smoke_test") is not False
            or state.get("model_key") != model or state.get("sampled_responses") != 10
            or state["prepared_sha256"] != digest(submission / "prepared.json")
            or state["policies"] != list(policies(model))
            or set(state["output_sha256"]) != {f"{v}_responses.jsonl" for v in policies(model)}):
        raise ValueError("Need a complete full ten-sample bank: " + model)
    verify_files(root, state["output_sha256"])
    result = {policy: records(root / f"{policy}_responses.jsonl") for policy in policies(model)}
    for policy, rows in result.items():
        validate_raw(rows, examples, policy=policy)
    return result, {str(root / "manifest.json"): digest(root / "manifest.json"), **{
        str(root / n): v for n, v in state["output_sha256"].items()}}


def make_pools(banks, examples):
    pools = {s: {} for s in examples}
    for setting in SELECTORS:
        for split, ee in examples.items():
            for mode in MODES:
                by_model = {}
                for model in MODELS:
                    policy = "dpo" if CONFIGURATIONS[setting][model] else "sft"
                    grouped = {q: [] for q in ee}
                    for r in banks[model][policy]:
                        if r["split"] == split and (r["draw"] == 0) == (mode == "greedy"):
                            grouped[r["question_id"]].append(r)
                    generated = [{"question_id": q, "samples": [{"answers": parse_answers(r["raw_response"])}
                        for r in sorted(rows, key=lambda r: r["draw"])]} for q, rows in sorted(grouped.items())]
                    if any(len(r["samples"]) != (1 if mode == "greedy" else 10) for r in generated):
                        raise ValueError("Candidate pooling dropped a requested draw")
                    by_model[model] = {"max5": generated}
                pools[split][f"{setting}_{mode}"] = [
                    {**r, "surface_operations": [], "is_format_variant": False}
                    for r in merge_pool(by_model, "max5")]
    return pools


def fit_one(pools, examples, setting):
    names = [f"{setting}_{mode}" for mode in MODES]
    rows = [r for n in names for r in pools["train"][n]]
    strata = [n for n in names for _ in pools["train"][n]]
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Need accepted and unaccepted fitting candidates")
    trials, best = [], None
    for C in C_GRID:
        print(f"Training {setting}, C={C}, {len(rows)} fitting candidates", flush=True)
        components = fit_components(rr.feature_rows(rows), examples["train"], labels,
            sample_weight=question_weights(rows, strata), C=C)
        payload = {"schema": "compactness-matched-logistic-v1", "selector": setting + "_matched",
            "components": components, "selected_C": C, "models": MODELS, "sampled_responses": 10,
            "configuration": CONFIGURATIONS[setting], "selection_conditions": names}
        metrics, perq, rankings = rank_frozen(payload, {n: pools["validation"][n] for n in names}, examples["validation"])
        score = float(np.mean([v["metrics"]["mrr_at5"] for v in metrics.values()]))
        trials.append({"C": C, "mean_validation_mrr_at5": score, "conditions": metrics})
        if best is None or score > best[0]:
            best = (score, payload, perq, rankings)
    return best, trials


def train(submission, output):
    p = validate(submission); examples = panels(submission)
    banks, bank_hashes = {}, {}
    for m in MODELS:
        banks[m], hashes = load_bank(submission, m, examples); bank_hashes.update(hashes)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = {"status": "running", "submission": str(submission), "prepared_sha256": digest(submission / "prepared.json"),
             "bank_hashes": bank_hashes, "question_counts": p["question_counts"], "sklearn_version": sklearn.__version__}
    write(output / "manifest.json", state)
    try:
        pools = make_pools(banks, examples)
        rr.label_pools(pools, examples, read(submission / "aliases.json"), output,
            Path(p["repository"]) / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
        summaries = {}
        for setting in SELECTORS:
            (score, payload, perq, rankings), trials = fit_one(pools, examples, setting)
            directory = output / setting; directory.mkdir()
            joblib.dump(payload, directory / "reranker.joblib")
            result = {"selected_C": payload["selected_C"], "mean_validation_mrr_at5": score,
                      "trials": trials, "configuration": CONFIGURATIONS[setting]}
            write(directory / "validation_selection.json", result)
            write(directory / "validation_per_question.json", perq)
            write_jsonl(directory / "validation_rankings.jsonl", rankings)
            summaries[setting] = result
            print(f"Frozen {setting}: C={payload['selected_C']}; validation MRR={score:.6f}", flush=True)
        for split in examples:
            write_jsonl(output / f"{split}_examples.jsonl", examples[split].values())
            for name, rows in pools[split].items():
                write_jsonl(output / f"{split}_{name}_candidates.jsonl", rows)
        write(output / "validation_summary.json", {"question_count": 144, "selectors": summaries,
            "sampling": "one greedy plus ten sampled responses per question/model",
            "selection": "Maximize mean validation MRR@5 across greedy and sampling10; first C on ties; fitting only, no refit on validation"})
        validate(submission)
        if any(digest(Path(n)) != h for n, h in bank_hashes.items()):
            raise ValueError("Generation changed while training")
        state.update(status="complete", artifacts_sha256={x.relative_to(output).as_posix(): digest(x)
            for x in output.rglob("*") if x.is_file() and x != output / "manifest.json"})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


def score(submission, training, output, resamples=10000):
    p = validate(submission, test=True)
    trained = read(training / "manifest.json")
    if (trained.get("status") != "complete" or trained.get("sklearn_version") != sklearn.__version__
            or trained["prepared_sha256"] != digest(submission / "prepared.json")
            or trained["question_counts"] != {"train": 1296, "validation": 144}):
        raise ValueError("Need two completed matching frozen rerankers")
    verify_files(training, trained["artifacts_sha256"])
    comparison = Path(p["comparison"]); old = read(comparison / "comparison_summary.json")
    test_submission = Path(p["test_submission"]); old_prep = read(test_submission / "prepared.json")
    examples = {r["question_id"]: r for r in records(test_submission / "test_inputs.jsonl")}
    gold = {r["question_id"]: r for r in records(test_submission / "test_gold.jsonl")}
    check_membership(panels(submission), examples, {"train": 1296, "validation": 144})
    pools = {f"{s}_{mode}": records(comparison / f"{s}_{mode}_candidates.jsonl") for s in CONFIGURATIONS for mode in MODES}
    if len(examples) != 95 or set(gold) != set(examples) or any(
            r["question_id"] not in examples or type(r.get("label")) is not int or r["label"] not in (0, 1)
            for rows in pools.values() for r in rows):
        raise ValueError("Invalid cached official test labels/membership")
    _, frozen, _ = load_frozen(Path(p["rerankers"]), examples, Path(old_prep["comparison"]))
    payloads = [frozen["max5"]]
    for setting in SELECTORS:
        payload = joblib.load(training / setting / "reranker.joblib")
        if (payload["schema"] != "compactness-matched-logistic-v1" or payload["selector"] != setting + "_matched"
                or payload["configuration"] != CONFIGURATIONS[setting] or payload["sampled_responses"] != 10
                or payload["selection_conditions"] != [f"{setting}_{m}" for m in MODES]):
            raise ValueError("Unexpected matched selector configuration")
        payloads.append(payload)
    if output.exists() or resamples < 2:
        raise ValueError("Need fresh scoring output and at least two bootstrap resamples")
    output.mkdir(parents=True)
    state = {"status": "running", "submission": str(submission), "training": str(training),
             "prepared_sha256": digest(submission / "prepared.json"),
             "training_manifest_sha256": digest(training / "manifest.json"),
             "test_generation": False, "test_selection": False, "official_labels": "unchanged cached labels"}
    write(output / "manifest.json", state)
    try:
        conditions, scores, rankings = {}, {}, []
        for payload in payloads:
            cc, ss, rows = rank_frozen(payload, pools, examples)
            conditions.update(cc); scores.update(ss); rankings.extend(rows)
        old_scores = read(comparison / "comparison_per_question.json")
        for name, condition in conditions.items():
            pool = name.split("_max5_reranker")[0] if name.endswith("_max5_reranker") else None
            if pool and (condition["metrics"] != old["conditions"][name]["metrics"] or scores[name] != old_scores[name]):
                raise ValueError("Old reranker did not reproduce the completed test result")
        for pool in pools:
            full = [conditions[f"{pool}_{v['selector']}_reranker"]["metrics"]["coverage_full_pool"] for v in payloads]
            if len(set(full)) != 1:
                raise ValueError("Reranking changed full-pool membership")
        add_batches(conditions, scores, gold)
        comparisons, transitions = {}, {}
        for mode in MODES:
            for setting in SELECTORS:
                matched = f"{setting}_{mode}_{setting}_matched_reranker"
                pairs = [(f"{setting}_{mode}_max5_reranker", matched, f"{setting}_{mode}_new_minus_old_same_pool"),
                         (f"baseline_{mode}_max5_reranker", matched, f"{setting}_{mode}_matched_pipeline_minus_baseline")]
                for before, after, name in pairs:
                    comparisons[name] = {"before": before, "after": after,
                        "metrics": contrast(scores[before], scores[after], METRICS, resamples)}
                    transitions[name] = {k: coverage_transitions(scores[before], scores[after], k) for k in METRICS if k != "mrr_at5"}
        result = {"question_count": 95, "conditions": conditions, "paired_contrasts": comparisons,
                  "coverage_transitions": transitions, "ensemble_generation": old["ensemble_generation"],
                  "quality_warnings": old["quality_warnings"], "training_sampled_responses": 10,
                  "limitations": ["Exploratory results on historically inspected test95; no choices selected on these scores.",
                      "New selectors change both generator-pool distribution and fitting sample budget from four to ten.",
                      "Same-pool comparisons isolate reranker replacement; pipeline comparisons include generator changes.",
                      "Frozen SFT baseline reranker retains its original four-sample training budget.",
                      "Official nonacceptance is not semantic incorrectness; historical features and pooling unchanged."]}
        validate(submission, test=True)
        if digest(training / "manifest.json") != state["training_manifest_sha256"]:
            raise ValueError("Frozen models changed while scoring")
        verify_files(training, trained["artifacts_sha256"])
        write(output / "comparison_summary.json", result)
        write(output / "comparison_per_question.json", scores)
        write_jsonl(output / "rankings.jsonl", rankings)
        state.update(status="complete", output_sha256={n: digest(output / n) for n in
            ("comparison_summary.json", "comparison_per_question.json", "rankings.jsonl")})
        for mode in MODES:
            print("\n" + mode, flush=True)
            for setting, selector in (("baseline", "max5"), ("strict", "strict_matched"), ("alternative", "alternative_matched")):
                m = conditions[f"{setting}_{mode}_{selector}_reranker"]["metrics"]
                print(f"  {setting}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, full pool={m['coverage_full_pool']:.4f}", flush=True)
            for setting in SELECTORS:
                c = comparisons[f"{setting}_{mode}_new_minus_old_same_pool"]["metrics"]["mrr_at5"]
                print(f"  {setting} new minus old on identical pool: {c['difference']:+.6f}; CI={c['paired_bootstrap_95ci']}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(output / "manifest.json", state)


def record(submission, stage, job):
    if stage != "train" or not re.fullmatch(r"\d+\.gadi-pbs", job):
        raise ValueError("Invalid training PBS job")
    write(submission / "training_job.json", {"job_id": job})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("prepare", "generate", "train", "score", "record", "check"))
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--repository", type=Path, default=ROOT)
    ap.add_argument("--comparison", type=Path)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--job-id")
    args = ap.parse_args(); sub = args.submission.resolve()
    if args.mode == "prepare":
        comparison = args.comparison or args.repository / "gadi_sft_8b_starter/outputs/compactness_test95/comparison-181558664.gadi-pbs"
        prepare(args.repository.resolve(), comparison.resolve(), sub)
    elif args.mode == "generate":
        if not args.model:
            ap.error("--model required")
        generate(sub, args.model, args.smoke_test)
    elif args.mode == "record":
        record(sub, "train", args.job_id or "")
    elif args.mode == "check":
        validate(sub)
    elif args.mode == "train":
        train(sub, sub / f"train-{os.environ.get('PBS_JOBID', 'manual')}")
    else:
        training = sub / f"train-{read(sub / 'training_job.json')['job_id']}"
        score(sub, training, sub / f"comparison-{os.environ.get('PBS_JOBID', 'manual')}")


if __name__ == "__main__":
    main()
