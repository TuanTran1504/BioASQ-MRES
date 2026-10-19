"""Budget-matched banks must preserve seed slots, membership and test pools."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compactness_matched_rerankers as run
from scripts.train_candidate_reranker_pilot import candidate_features
from run_single_answer_sampling import generation_options


def examples(n=4):
    return {s: {f"{s}{i}": {"question_id": f"{s}{i}", "question": "Which entity is correct?",
            "snippets": [{"snippet_id": "s1", "text": "Core is correct; Distractor is unrelated."}]}
        for i in range(n)} for s in ("train", "validation")}


def raw(qid, split, draw, answers=("Core", "Distractor")):
    text = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                  for i, a in enumerate(answers)]}) if answers else "invalid JSON"
    return {"question_id": qid, "split": split, "draw": draw, "raw_response": text,
        "seed": run.sample_seed(qid, max(1, draw), 3407), "output_tokens": 30, "generation_seconds": .1}


def banks(ee):
    return {m: {p: [raw(q, s, d, ("Core", f"{m} {p}"))
        for s, e in ee.items() for q in e for d in range(11)] for p in run.policies(m)} for m in run.MODELS}


def test_reused_seed_slots_need_six_new_samples():
    ee = examples(2)
    reused = [raw(q, s, d) for s, e in ee.items() for q in e for d in (0, 2, 3, 4, 5)]
    before = copy.deepcopy(reused)
    run.validate_raw(reused, ee, complete=False)
    tasks = run.missing_tasks(reused, ee, chunk_size=1)
    assert reused == before
    assert {tuple(t["indices"]) for t in tasks} == {(1,), (6, 7, 8, 9, 10)}
    generated = [raw(q, t["split"], d) for t in tasks for q in t["question_ids"] for d in t["indices"]]
    run.validate_raw(reused + generated, ee)
    assert len(reused + generated) == 4 * 11


@pytest.mark.parametrize("change", ["duplicate", "missing", "seed", "dev", "negative", "unknown_split"])
def test_raw_bank_fails_closed(change):
    ee = examples(1); rows = banks(ee)["qwen3"]["dpo"]
    if change == "duplicate": rows.append(rows[0])
    elif change == "missing": rows.pop()
    elif change == "seed": rows[0]["seed"] += 1
    elif change == "dev": rows[0]["question_id"] = "dev160"
    elif change == "negative": rows[0]["output_tokens"] = -1
    else: rows[0]["split"] = "test"
    with pytest.raises(ValueError): run.validate_raw(rows, ee)


def test_strict_sources_and_failed_draw_denominators_match_test_pooling():
    ee = examples(1); bb = banks(ee)
    bb["qwen3"]["dpo"] = [raw(r["question_id"], r["split"], r["draw"],
        () if r["draw"] == 10 else ("Core", "Qwen DPO")) for r in bb["qwen3"]["dpo"]]
    pools = run.make_pools(bb, ee)
    strict = {r["answer"]: r for r in pools["train"]["strict_sampling10"]}
    assert set(strict) == {"Core", "llama31 sft", "Qwen DPO", "ministral3 sft"}
    assert strict["Core"]["sources"] == list(run.MODELS)
    assert strict["Core"]["sample_frequencies"]["qwen3"] == pytest.approx(.9)
    assert strict["Core"]["sample_frequencies"]["llama31"] == pytest.approx(1.)
    assert strict["Core"]["source_ranks"] == dict.fromkeys(run.MODELS, 1)
    for r in pools["train"]["alternative_sampling10"]:
        candidate_features(r, ee["train"][r["question_id"]])
    with pytest.raises(ValueError, match="dropped"):
        run.make_pools({**bb, "qwen3": {"dpo": bb["qwen3"]["dpo"][:-1]}}, ee)


def chunk(tmp_path, indices, broken=False):
    source = {"adapter": "adapter", "model_loader": "fast_language_model", "chat_template_kwargs": {}}
    ee = examples(1); q = "train0"; panel = [ee["train"][q]]
    plan = {"max_seq_length": 8704, "expected_prompt_tokens": {q: 100}, "expected_prompt_ids_sha256": {q: "hash"}}
    (tmp_path / "llama31_prompt.txt").write_text("Native prompt", encoding="utf-8")
    inp = tmp_path / "inputs.jsonl"; run.write_jsonl(inp, panel)
    task = {"split": "train", "question_ids": [q], "indices": indices}
    cfg = run.worker_config(tmp_path, "llama31", source, plan, task, inp)
    assert generation_options(cfg)["do_sample"] == (indices != [0])
    saved = {**cfg, "model_name": "adapter", "question_count": 1, "gold_blind_generation": True,
        "local_files_only": True, "input_sha256": run.digest(inp),
        "prompt_sha256": run.digest(tmp_path / "llama31_prompt.txt")}
    samples = []
    for i, d in enumerate(indices, 1):
        response = raw(q, "train", d, () if broken and i == 1 else ("Core",))
        aa = run.parse_answers(response["raw_response"])
        samples.append({**response, "draw": i, "answers": aa, "parse_error": None if aa else "invalid"})
    row = {"question_id": q, "samples": samples, "request_count": len(samples), "snippets_truncated": False,
        "prompt_tokens": 100, "included_snippets": 1, "total_snippets": 1,
        "input_tokens": 100*len(samples), "output_tokens": 30*len(samples), "generation_seconds": .1*len(samples)}
    directory = tmp_path / "run"; directory.mkdir()
    run.write(directory / "config.json", saved)
    run.write(directory / "status.json", {"status": "complete"})
    run.write_jsonl(directory / "examples.jsonl", panel)
    run.write_jsonl(directory / "generations.jsonl", [row])
    run.write_jsonl(directory / "candidates.jsonl", [{"question_id": q, **c} for c in run.expansion_candidates(samples)])
    run.write_jsonl(directory / "invalid_candidates.jsonl", [])
    return directory, cfg, panel, source, task


@pytest.mark.parametrize("indices", [[0], [1], [6, 7, 8, 9, 10], list(range(1, 11))])
def test_chunk_validation_reconstructs_true_draw_slots(tmp_path, indices):
    rows = run.read_chunk(*chunk(tmp_path, indices, broken=True))
    assert [r["draw"] for r in rows] == indices
    assert rows[0]["raw_response"] == "invalid JSON"
    assert [r["seed"] for r in rows] == [run.sample_seed("train0", max(1, d)) for d in indices]


@pytest.mark.parametrize("change", ["tokenization", "draw", "seed", "parsing", "cost", "candidates", "evidence", "source"])
def test_chunk_tampering_fails_closed(tmp_path, change):
    directory, cfg, panel, ss, task = chunk(tmp_path, [6, 7])
    rows = run.records(directory / "generations.jsonl")
    if change == "tokenization": cfg["expected_prompt_ids_sha256"] = {"train0": "other"}
    elif change == "draw": rows[0]["samples"].pop()
    elif change == "seed": rows[0]["samples"][0]["seed"] += 1
    elif change == "parsing": rows[0]["samples"][0]["answers"] = []
    elif change == "cost": rows[0]["output_tokens"] = 0
    elif change == "candidates": run.write_jsonl(directory / "candidates.jsonl", [])
    elif change == "evidence": rows[0]["snippets_truncated"] = True
    else: ss["adapter"] = "wrong"
    run.write_jsonl(directory / "generations.jsonl", rows)
    with pytest.raises(ValueError): run.read_chunk(directory, cfg, panel, ss, task)


def test_fit_uses_fitting_labels_only_and_tunes_on_ten_sample_validation(monkeypatch):
    ee = examples(); pools = run.make_pools(banks(ee), ee)
    for pp in pools.values():
        for rows in pp.values():
            for r in rows: r["label"] = int(r["answer"] == "Core")
    original_fit = run.fit_components; calls = []
    def fitting(rows, contexts, labels, **kw):
        assert set(contexts) == set(ee["train"])
        assert {r["question_id"] for r in rows} <= set(ee["train"])
        assert all("label" not in r and "gold_aliases" not in r for r in rows)
        assert np.sum(kw["sample_weight"]) > 0
        calls.append(kw["C"])
        return original_fit(rows, contexts, labels, **kw)
    monkeypatch.setattr(run, "fit_components", fitting)
    (score, payload, perq, _), trials = run.fit_one(pools, ee, "strict")
    assert calls == [.1, 1., 10.]
    assert payload["selection_conditions"] == ["strict_greedy", "strict_sampling10"]
    assert payload["sampled_responses"] == 10
    assert all({r["question_id"] for r in rows} == set(ee["validation"]) for rows in perq.values())
    assert score == max(t["mean_validation_mrr_at5"] for t in trials)


def test_membership_rejects_dev_test_and_validation_overlap():
    ee = examples(); run.check_membership(ee, ["test0"])
    with pytest.raises(ValueError): run.check_membership(ee, ["train0"])
    ee["validation"]["train0"] = ee["train"]["train0"]
    with pytest.raises(ValueError): run.check_membership(ee, [])


def test_native_snippet_ids_become_worker_ids_without_gold():
    assert run.input_panel({"q": {"question": "Q", "gold_aliases": ["SECRET"],
        "snippets": [{"id": "S", "text": "Evidence"}]}}) == [
            {"question_id": "q", "question": "Q", "snippets": [{"snippet_id": "S", "text": "Evidence"}]}]


def test_prepare_reuses_native_four_samples_and_epoch_one_validation(tmp_path, monkeypatch):
    ee = examples(1)
    for s, n in (("train", 1296), ("validation", 144)):
        ee[s] = {f"{s}{i}": {**ee[s][f"{s}0"], "question_id": f"{s}{i}"} for i in range(n)}
    repo = tmp_path / "repo"; starter = repo / "gadi_sft_8b_starter"
    (starter / "configs").mkdir(parents=True)
    run.write(starter / "configs/reranker_guided_dpo_pilot.json", {})
    cfg_path = starter / "configs/banks.json"; run.write(cfg_path, {})
    (starter / "jobs").mkdir()
    for stage in ("generate", "train", "score"):
        (starter / f"jobs/{stage}_compactness_rerankers.pbs").write_text("#!/bin/bash\nexit 0\n")
    jar = repo / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    jar.parent.mkdir(parents=True); jar.write_bytes(b"mock jar")
    monkeypatch.setattr("src.utility.bioasq_official._official_adapter_classes", lambda *a: None)
    old_sub = tmp_path / "old-sub"; old_sub.mkdir()
    old_result = tmp_path / "old-result"; old_result.mkdir()
    run.write(old_sub / "prepared.json", {})
    test = [{"question_id": f"test{i}", "question": "Test?", "snippets": []} for i in range(95)]
    for name in ("test_inputs", "test_gold"): run.write_jsonl(old_sub / f"{name}.jsonl", test)
    run.write(old_result / "manifest.json", {"status": "complete", "configurations": run.CONFIGURATIONS,
        "submission": str(old_sub), "prepared_sha256": run.digest(old_sub / "prepared.json"), "output_sha256": {}})
    rerankers = tmp_path / "rerankers"; rerankers.mkdir()
    sources, dpo, selections, bank_paths = {}, {}, {}, {}
    for m in run.MODELS:
        base = tmp_path / f"base-{m}"; base.mkdir(); run.write(base / "config.json", {})
        for policy, target in (("sft", sources), ("dpo", dpo)):
            adapter = tmp_path / f"{m}-{policy}"; adapter.mkdir(); (adapter / "weights").write_bytes(b"weights")
            target[m] = {"adapter": str(adapter), "adapter_files_sha256": {"weights": run.digest(adapter / "weights")},
                "base_snapshot": str(base), "system_prompt": "Return up to FIVE answers", "model_loader": "unsloth", "chat_template_kwargs": {}}
        selection = tmp_path / f"{m}-sel"; selection.mkdir(); selections[m] = str(selection)
        panel = list(ee["validation"].values()); run.write_jsonl(selection / "validation_inputs.jsonl", panel)
        for mode in ("greedy", "sampling4"):
            count = 1 if mode == "greedy" else 4
            directory = selection / "epoch-1" / mode; directory.mkdir(parents=True)
            config = {"model_name": dpo[m]["adapter"], "num_generations": count, "max_candidates": 5,
                "temperature": 0 if count == 1 else .8, "seed": 3407, "sample_seed_offset": 0 if count == 1 else 1,
                "max_new_tokens": 512, "require_all_snippets": True, "mark_snippets": False,
                "response_mode": "equivalent_greedy" if count == 1 else "equivalent_sampling", "top_p": .95, "top_k": 0,
                "model_loader": "unsloth", "chat_template_kwargs": {}, "execution_mode": "default", "max_seq_length": 8704,
                "expected_prompt_tokens": {e["question_id"]: 100 for e in panel},
                "expected_prompt_ids_sha256": {e["question_id"]: "tokenhash" for e in panel}}
            run.write(directory / "config.json", config); run.write(directory / "status.json", {"status": "complete"})
            run.write_jsonl(directory / "examples.jsonl", panel)
            generated = []
            for e in panel:
                q = e["question_id"]
                samples = []
                for i in range(1, count+1):
                    r = raw(q, "validation", 0 if count == 1 else i+1)
                    samples.append({**r, "draw": i, "answers": run.parse_answers(r["raw_response"])})
                generated.append({"question_id": q, "samples": samples, "snippets_truncated": False,
                    "included_snippets": 1, "total_snippets": 1})
            run.write_jsonl(directory / "generations.jsonl", generated)
            for name in ("candidates", "invalid_candidates"): run.write_jsonl(directory / f"{name}.jsonl", [])
        bank = tmp_path / f"{m}-native"; bank.mkdir(); run.write(bank / "manifest.json", {})
        native = []
        for s, rows in ee.items():
            for q, e in rows.items():
                prompt = [{"role": "system", "content": sources[m]["system_prompt"]},
                    {"role": "user", "content": json.dumps({"question": e["question"], "snippets": [{"id": "s1", "text": e["snippets"][0]["text"]}]})}]
                native.extend({**raw(q, s, 0 if d == 0 else d+1), "draw": d, "prompt_tokens": 100,
                    "prompt": prompt, "model_key": m} for d in range(5))
        run.write_jsonl(bank / "responses.jsonl", native)
        bank_paths[m] = {"path": str(bank)}
    old = {"rerankers": str(rerankers), "reranker_hashes": {}, "selections": selections, "sources": sources, "dpo_sources": dpo}
    monkeypatch.setattr(run, "validate_prepared", lambda *a: old)
    data = SimpleNamespace(DEFAULT_CONFIG=cfg_path, panels=lambda c: ({}, {q: ["Core"] for e in ee.values() for q in e}))
    monkeypatch.setattr(run, "load_inputs", lambda *a: ({}, {}, [], {}, {}, bank_paths, data))
    monkeypatch.setattr(run, "validate_epoch_one", lambda s, m, r: (sources[m], dpo[m], {}, {}, {}))
    def plan(source, panel, count):
        assert count == 512 and len(panel) == 1440 and all("gold_aliases" not in e for e in panel)
        return {"max_seq_length": 612, "expected_prompt_tokens": {e["question_id"]: 100 for e in panel},
                "expected_prompt_ids_sha256": {e["question_id"]: "tokenhash" for e in panel}}
    monkeypatch.setattr(run, "full_input_plan", plan)
    monkeypatch.setattr(run, "snapshot_inference_code", lambda s: {"files_sha256": {"scripts/run_single_answer_sampling.py": "workerhash"}})
    monkeypatch.setattr(run.rr, "snapshot_code", lambda *a: {})
    sub = tmp_path / "new"; run.prepare(repo, old_result, sub)
    p = run.read(sub / "prepared.json")
    assert p["sampled_responses"] == 10
    for m in run.MODELS:
        reused = run.records(sub / f"{m}_dpo_reused.jsonl")
        assert len(reused) == 144*5 and {r["draw"] for r in reused} == {0, 2, 3, 4, 5}
        assert {r["split"] for r in reused} == {"validation"}
        if m != "qwen3": assert len(run.records(sub / f"{m}_sft_reused.jsonl")) == 1440*5
    assert not (sub / "qwen3_sft_reused.jsonl").exists()
    assert all("gold_aliases" not in e for e in run.records(sub / "train_inputs.jsonl"))


def test_resumption_reuses_finished_bank_without_rerunning_worker(tmp_path, monkeypatch):
    ee = examples(1); sub = tmp_path / "s"; sub.mkdir(); run.write(sub / "prepared.json", {})
    reused = [raw(q, s, d) for s, e in ee.items() for q in e for d in range(11)]
    run.write_jsonl(sub / "qwen3_dpo_reused.jsonl", reused)
    p = {"sources": {}, "dpo_sources": {"qwen3": {}}, "chunk_size": 64}
    monkeypatch.setattr(run, "validate", lambda *a, **k: p)
    monkeypatch.setattr(run, "panels", lambda s: ee)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("Completed bank must not regenerate"))
    run.generate(sub, "qwen3")
    first = run.digest(sub / "banks/qwen3/dpo_responses.jsonl")
    run.generate(sub, "qwen3")
    assert first == run.digest(sub / "banks/qwen3/dpo_responses.jsonl")


def test_recovery_commits_orphan_complete_chunk_and_skips_it_on_resume(tmp_path, monkeypatch):
    ee = examples(1); sub = tmp_path / "s"; sub.mkdir(); run.write(sub / "prepared.json", {})
    reused = [raw(q, s, d) for s, e in ee.items() for q in e for d in range(11)
              if (s, d) != ("train", 1)]
    run.write_jsonl(sub / "qwen3_dpo_reused.jsonl", reused)
    directory = sub / "banks/qwen3/dpo-0000"; directory.mkdir(parents=True)
    run_dir, cfg, _, source, _ = chunk(directory, [1])
    run.write(directory / "config.json", cfg)
    p = {"sources": {}, "dpo_sources": {"qwen3": source}, "chunk_size": 64, "input_plans": {"qwen3": {}}}
    monkeypatch.setattr(run, "validate", lambda *a, **k: p)
    monkeypatch.setattr(run, "panels", lambda s: ee)
    monkeypatch.setattr(run, "worker_config", lambda *a: cfg)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("Saved complete chunk must not rerun"))
    run.generate(sub, "qwen3")
    manifest = sub / "banks/qwen3/manifest.json"; state = run.read(manifest)
    assert "dpo-0000" in state["completed_chunks"]
    state["status"] = "failed"; run.write(manifest, state)
    run.generate(sub, "qwen3")
    assert run.read(manifest)["status"] == "complete"
    assert len(run.records(sub / "banks/qwen3/dpo_responses.jsonl")) == 22


@pytest.mark.parametrize("indices", [[2, 1], [0], [10, 11], [], [1, 3]])
def test_partial_sampler_requires_exact_contiguous_seed_slots(indices):
    cfg = {"num_generations": len(indices), "sample_seed_offset": (indices[0]-1 if indices else 0),
           "sampled_draw_indices": indices, "temperature": .8, "max_new_tokens": 512, "top_p": .95}
    with pytest.raises(ValueError, match="subset"):
        generation_options(cfg)


def test_unmarked_partial_sampling_still_rejected():
    with pytest.raises(ValueError, match="four or ten"):
        generation_options({"num_generations": 5, "temperature": .8})


def test_train_and_test_rescore_with_real_classifier_and_official_labels(tmp_path, monkeypatch):
    jar = run.ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    if not jar.is_file() or not shutil.which("java"):
        pytest.skip("Official Java scorer required")
    # Full membership sizes exercise the production disjointness guards. Only
    # four fitting questions supply candidates; failed/empty pools remain in
    # validation/test denominators throughout the production ranking code.
    ee = examples(1)
    for split, n in (("train", 1296), ("validation", 144)):
        ee[split] = {f"{split}{i}": {**ee[split][f"{split}0"], "question_id": f"{split}{i}"} for i in range(n)}
    bb = banks({s: dict(list(e.items())[:4]) for s, e in ee.items()})
    sub = tmp_path / "s"; sub.mkdir(); run.write(sub / "prepared.json", {})
    for split, rows in ee.items(): run.write_jsonl(sub / f"{split}_inputs.jsonl", rows.values())
    run.write(sub / "aliases.json", {q: ["Core"] for e in ee.values() for q in e})
    # Empty responses represent all other questions, preserving 11 requested
    # draws even when they yield no usable candidate.
    for m in bb:
        for policy, rows in bb[m].items():
            present = {r["question_id"] for r in rows}
            rows.extend(raw(q, s, d, ()) for s, e in ee.items() for q in e if q not in present for d in range(11))
        root = sub / "banks" / m; root.mkdir(parents=True)
        for policy, rows in bb[m].items(): run.write_jsonl(root / f"{policy}_responses.jsonl", rows)
        run.write(root / "manifest.json", {"status": "complete", "smoke_test": False, "model_key": m,
            "sampled_responses": 10, "prepared_sha256": run.digest(sub / "prepared.json"),
            "policies": list(run.policies(m)), "output_sha256": {
                f"{p}_responses.jsonl": run.digest(root / f"{p}_responses.jsonl") for p in run.policies(m)}})
    p = {"repository": str(run.ROOT), "question_counts": {"train": 1296, "validation": 144}}
    monkeypatch.setattr(run, "validate", lambda *a, **kw: p)
    training = sub / "train-1"; run.train(sub, training)
    state = run.read(training / "manifest.json")
    assert state["status"] == "complete"
    assert all((training / s / "reranker.joblib").exists() for s in run.SELECTORS)
    summary = run.read(training / "validation_summary.json")
    assert set(summary["selectors"]) == set(run.SELECTORS)
    # Cached test pools and the old baseline classifier are independent inputs.
    comparison = sub / "old"; comparison.mkdir(); test_sub = sub / "test"; test_sub.mkdir()
    p.update(comparison=str(comparison), test_submission=str(test_sub), rerankers="old-rerankers")
    test = {f"test{i}": {"question_id": f"test{i}", "question": "Which entity is correct?", "snippets": []} for i in range(95)}
    gold = {q: {**e, "gold_aliases": ["Core"], "batch": "batch1" if i < 48 else "batch2"} for i, (q, e) in enumerate(test.items())}
    run.write_jsonl(test_sub / "test_inputs.jsonl", test.values())
    run.write_jsonl(test_sub / "test_gold.jsonl", gold.values())
    run.write(test_sub / "prepared.json", {"comparison": "old-sft-comparison"})
    prototype = run.records(training / "validation_alternative_sampling10_candidates.jsonl")[:4]
    pools = {f"{s}_{mode}": [{**r, "question_id": q} for q in test for r in prototype]
             for s in run.CONFIGURATIONS for mode in run.MODES}
    for name, rows in pools.items(): run.write_jsonl(comparison / f"{name}_candidates.jsonl", rows)
    payload = run.joblib.load(training / "strict/reranker.joblib"); payload["selector"] = "max5"
    conditions, scores, _ = run.rank_frozen(payload, pools, test)
    run.write(comparison / "comparison_summary.json", {"conditions": conditions,
        "ensemble_generation": {}, "quality_warnings": []})
    run.write(comparison / "comparison_per_question.json", scores)
    monkeypatch.setattr(run, "load_frozen", lambda *a: ({}, {"max5": payload}, {}))
    original = {n: run.digest(comparison / f"{n}_candidates.jsonl") for n in pools}
    output = sub / "score"; run.score(sub, training, output, resamples=20)
    assert run.read(output / "manifest.json")["status"] == "complete"
    d = run.read(output / "comparison_summary.json")
    assert len(d["conditions"]) == 18
    assert len(d["paired_contrasts"]) == 8
    assert all("per_batch" in r for r in d["conditions"].values())
    assert original == {n: run.digest(comparison / f"{n}_candidates.jsonl") for n in pools}
    assert run.read(output / "manifest.json")["test_generation"] is False


def test_launcher_schedules_three_smoke_banks_then_train_then_score(tmp_path):
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not bash: pytest.skip("Git Bash required")
    starter = tmp_path / "starter"; (starter / "scripts").mkdir(parents=True)
    launcher = run.ROOT / "gadi_sft_8b_starter/scripts/submit_compactness_matched_rerankers.sh"
    target = starter / "scripts/submit.sh"
    text = launcher.read_text(encoding="utf-8").replace('source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"', ":")
    text = text.replace("${USER}", "${TEST_USER}")
    target.write_text(text, encoding="utf-8", newline="\n")
    mocks = tmp_path / "bin"; mocks.mkdir()
    def executable(name, code):
        (mocks / name).write_text("#!/bin/bash\n" + code + "\n", encoding="utf-8", newline="\n")
    executable("module", "exit 0")
    executable("python3", 'for ((i=1;i<=$#;i++)); do\n  if [[ "${!i}" == --submission ]]; then j=$((i+1)); sub="${!j}"; mkdir -p "$sub"; fi\ndone\nexit 0')
    executable("qsub", 'n=$(cat "$MOCK_COUNTER" 2>/dev/null || echo 1000)\nn=$((n+1)); echo "$n" > "$MOCK_COUNTER"\nprintf "%s\\n" "$*" >> "$MOCK_LOG"\necho "$n.gadi-pbs"')
    env = {**os.environ, "PATH": str(mocks) + os.pathsep + os.environ["PATH"], "TEST_USER": "tester",
           "MOCK_COUNTER": str(tmp_path / "count"), "MOCK_LOG": str(tmp_path / "calls")}
    result = subprocess.run([bash, str(target)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == 8
    assert "afterok:1001.gadi-pbs" in calls[1]
    assert "afterok:1002.gadi-pbs:1004.gadi-pbs:1006.gadi-pbs" in calls[6]
    assert "afterok:1007.gadi-pbs" in calls[7]
