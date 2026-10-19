"""Fixed test95 ensembles must reuse SFT, preserve seeds and never pick on test."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import run_compactness_test95 as run
from scripts import analyze_compactness_test95 as score
from test_generator_feedback_evaluation import archive, panel, source


def banks(tmp_path, examples):
    baseline, dpo = {}, {}
    for mode in run.MODES:
        for model in run.MODELS:
            baseline[f"{model}_{mode}"], _ = archive(tmp_path / f"sft-{model}-{mode}", examples, source(), mode,
                lambda q, m=model: [m + " SFT", "Core"])
            dpo[f"{model}_{mode}"], _ = archive(tmp_path / f"dpo-{model}-{mode}", examples, source(), mode,
                lambda q, m=model: [m + " DPO"])
    return baseline, dpo


def test_three_fixed_configurations_reuse_each_model_bank(tmp_path):
    examples = panel(); baseline, dpo = banks(tmp_path, examples)
    before = copy.deepcopy((baseline, dpo))
    pools = score.make_pools(baseline, dpo)
    assert (baseline, dpo) == before
    assert run.CONFIGURATIONS == {"baseline": dict.fromkeys(run.MODELS, 0),
        "strict": {"llama31": 0, "qwen3": 1, "ministral3": 0}, "alternative": dict.fromkeys(run.MODELS, 1)}
    for mode in run.MODES:
        strict = {r["answer"]: r for r in pools[f"strict_{mode}"] if r["question_id"] == "q0"}
        assert set(strict) == {"llama31 SFT", "Core", "qwen3 DPO", "ministral3 SFT"}
        assert strict["Core"]["sources"] == ["llama31", "ministral3"]
        assert strict["qwen3 DPO"]["sample_frequencies"]["qwen3"] == pytest.approx(1.)
        assert {r["answer"] for r in pools[f"alternative_{mode}"]} == {m + " DPO" for m in run.MODELS}


def test_test95_disjoint_membership_and_preference_checks():
    examples = panel(95)
    membership = {"train": [{"question_id": f"t{i}"} for i in range(1296)],
                  "validation": [{"question_id": f"v{i}"} for i in range(144)]}
    splits = {"train": [{"question_id": "t0"}], "validation": [{"question_id": "v0"}]}
    run.validate_partition(examples, membership, splits)
    with pytest.raises(ValueError, match="95 distinct"):
        run.validate_partition(examples[:-1], membership, splits)
    membership["validation"][0]["question_id"] = "q0"
    with pytest.raises(ValueError, match="overlap"):
        run.validate_partition(examples, membership, splits)
    membership["validation"][0]["question_id"] = "v0"
    splits["train"][0]["question_id"] = "v0"
    with pytest.raises(ValueError, match="outside"):
        run.validate_partition(examples, membership, splits)


def predictor(monkeypatch):
    from scripts import evaluate_sft_ensemble_rerankers as ev
    def blind(components, features, contexts):
        assert all("label" not in r for r in features)
        assert all(set(e) == {"question", "snippets"} for e in contexts.values())
        return np.array([2. if r["answer"] == "Core" else 0. for r in features])
    monkeypatch.setattr(ev, "score_components", blind)


def test_scoring_records_pool_losses_and_per_batch_without_gold_features(tmp_path, monkeypatch):
    predictor(monkeypatch)
    examples = panel(3); ee = {e["question_id"]: e for e in examples}
    gold = {q: {**e, "batch": "B1" if q == "q0" else "B2"} for q, e in ee.items()}
    baseline, dpo = banks(tmp_path, examples)
    pools = score.make_pools(baseline, dpo)
    calls = []
    def matcher(ee, cc, output, jar_path):
        assert output.is_dir()
        calls.append(cc)
        return {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in cc}
    conditions, scores, rankings = score.score_pools(pools, ee, gold,
        {"selector": "max5", "components": {}}, tmp_path / "result", Path("jar"), matcher)
    assert len(calls) == 1 and len(conditions) == 24
    assert conditions["baseline_sampling10_max5_reranker"]["metrics"]["mrr_at5"] == 1.
    assert conditions["strict_sampling10_max5_reranker"]["metrics"]["coverage_full_pool"] == 1.
    assert conditions["alternative_sampling10_max5_reranker"]["metrics"]["coverage_full_pool"] == 0.
    t = score.coverage_transitions(scores["baseline_sampling10_max5_reranker"],
        scores["alternative_sampling10_max5_reranker"], "coverage_full_pool")
    assert t["lost"] == 3 and t["gained"] == 0
    assert conditions["baseline_greedy_max5_reranker"]["per_batch"]["B2"]["question_count"] == 2
    assert rankings


def test_actual_java_scoring_creates_its_report_directory(tmp_path, monkeypatch):
    predictor(monkeypatch)
    examples = {e["question_id"]: e for e in panel(2)}
    gold = {q: {**e, "batch": "B1"} for q, e in examples.items()}
    rows = [{"question_id": q, "answer": a, "sources": ["llama31"], "source_ranks": {"llama31": i},
        "sample_frequencies": {"llama31": 1.}, "relation_types": ["original"]}
        for q in examples for i, a in enumerate(("Wrong", "Core"), 1)]
    output = tmp_path / "not-yet-created/result"
    conditions, _, _ = score.score_pools({"baseline_greedy": rows}, examples, gold,
        {"selector": "max5", "components": {}}, output,
        ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    assert (output / "official/scorer_gold.json").is_file()
    assert conditions["baseline_greedy_max5_reranker"]["metrics"]["mrr_at5"] == 1.


@pytest.fixture
def epoch_run(tmp_path, monkeypatch, request):
    model = getattr(request, "param", "qwen3")
    prefix = "llama" if model == "llama31" else model
    training = tmp_path / "training"; training.mkdir()
    submission = tmp_path / "training-submission"; submission.mkdir()
    selection = training / "selection"; selection.mkdir()
    preferences = tmp_path / "preferences"; preferences.mkdir()
    run.write(preferences / "manifest.json", {})
    ss = source("native-sft")
    config = {"model_key": model, "retain_epoch_checkpoints": True, "num_train_epochs": 3}
    splits = {"train": [{"question_id": "t0"}], "validation": [{"question_id": "v0"}]}
    feedback = {"banks": {}, "reranker_hashes": {"manifest.json": "reranker"}}
    prepared = {"preferences": str(preferences), "configuration": config, "source": ss, "native_banks": {},
        "preferences_manifest_sha256": run.digest(preferences / "manifest.json")}
    run.write(submission / "prepared.json", prepared)
    for directory, weights in ((training / "adapter", b"final"), (selection / "epoch-1/adapter", b"epoch1")):
        directory.mkdir(parents=True)
        run.write(directory / "adapter_config.json", {"base_model_name_or_path": "base"})
        run.write(directory / "tokenizer_config.json", {})
        (directory / "adapter_model.safetensors").write_bytes(weights)
    for epoch in (1, 2, 3):
        cp = training / f"trainer_output/checkpoint-{epoch}"
        cp.mkdir(parents=True)
        run.write(cp / "trainer_state.json", {"global_step": epoch, "epoch": epoch,
            "log_history": [{"step": epoch, "epoch": epoch, "eval_loss": .69}]})
        run.write(cp / "adapter_config.json", {"base_model_name_or_path": "base"})
        (cp / "adapter_model.safetensors").write_bytes(f"epoch{epoch}".encode())
    (training / "reference_logps.jsonl").write_text("[]")
    state = {"status": "complete", "phase": "save_adapter", "smoke_test": False, "model_key": model,
        "global_steps": 3, "submission": str(submission), "prepared_sha256": run.digest(submission / "prepared.json"),
        "configuration": config, "source": ss, "trained_pair_counts": {"train": 1, "validation": 1},
        "preference_pair_counts": {"train": 1, "validation": 1},
        "preferences_manifest_sha256": prepared["preferences_manifest_sha256"],
        "reference_logps_sha256": run.digest(training / "reference_logps.jsonl"),
        "adapter_files_sha256": {p.name: run.digest(p) for p in (training / "adapter").iterdir()}}
    run.write(training / "manifest.json", state)
    rows = [{"epoch": e, "mrr_at5": .6, "mean_output_tokens": 60-e*5,
        f"{prefix}_full_pool_lost": int(e > (1 if model == "qwen3" else 0)),
        "ensemble_full_pool_lost": 0, "generation_quality": dict(empty_responses=0, ceiling_responses=0, surplus_responses=0)} for e in (0, 1, 2, 3)]
    winner = run.CONFIGURATIONS["strict"][model]
    run.write(selection / "comparison_summary.json", {"question_count": 144, "selected_epoch": winner, "checkpoint_scores": rows})
    selected = {"status": "complete", "smoke_test": False, "validation_question_count": 144, "focal_model": model,
        "training": str(training), "submission": str(submission), "prepared_sha256": state["prepared_sha256"],
        "training_manifest_sha256": run.digest(training / "manifest.json"), "selected_epoch": winner,
        "reranker_hashes": feedback["reranker_hashes"],
        "output_sha256": {p.relative_to(selection).as_posix(): run.digest(p) for p in selection.rglob("*") if p.is_file()}}
    run.write(selection / "manifest.json", selected)
    monkeypatch.setattr(run, "validate_native", lambda *args: ({}, splits, ss, feedback))
    return selection, training


@pytest.mark.parametrize("corrupt", [None, "epoch_weights", "reference", "summary", "training", "preference", "smoke", "source"])
def test_epoch_one_is_exact_retained_checkpoint_and_strict_selection_is_pinned(epoch_run, corrupt):
    selection, training = epoch_run
    if corrupt == "epoch_weights": (selection / "epoch-1/adapter/adapter_model.safetensors").write_bytes(b"final")
    if corrupt == "reference": (training / "reference_logps.jsonl").write_text("changed")
    if corrupt == "summary": (selection / "comparison_summary.json").write_text("{}")
    if corrupt == "training":
        state = run.read(training / "manifest.json"); state["model_key"] = "ministral3"; run.write(training / "manifest.json", state)
    if corrupt == "preference":
        preferences = training.parent / "preferences/manifest.json"; run.write(preferences, {"changed": True})
    if corrupt in ("smoke", "source"):
        state = run.read(selection / "manifest.json")
        state["smoke_test" if corrupt == "smoke" else "focal_model"] = True if corrupt == "smoke" else "llama31"
        run.write(selection / "manifest.json", state)
    if corrupt:
        with pytest.raises((ValueError, KeyError)):
            run.validate_epoch_one(selection, "qwen3", ROOT)
    else:
        ss, dpo, _, _, pins = run.validate_epoch_one(selection, "qwen3", ROOT)
        assert dpo["adapter"] == str(selection / "epoch-1/adapter") and dpo["dpo_epoch"] == 1
        assert ss["adapter"] == "native-sft" and pins


@pytest.mark.parametrize("epoch_run", ["llama31", "qwen3", "ministral3"], indirect=True)
def test_legacy_llama_manifest_without_focal_model_is_read_without_rewriting(epoch_run):
    selection, training = epoch_run
    selected = run.read(selection / "manifest.json")
    model = selected.pop("focal_model")
    run.write(selection / "manifest.json", selected)
    original_hash = run.digest(selection / "manifest.json")
    if model == "llama31":
        _, dpo, _, _, pins = run.validate_epoch_one(selection, model, ROOT)
        assert dpo["adapter"] == str(selection / "epoch-1/adapter")
        assert pins[str(selection / "manifest.json")] == original_hash
    else:
        with pytest.raises(ValueError, match="model identity"):
            run.validate_epoch_one(selection, model, ROOT)
    assert run.digest(selection / "manifest.json") == original_hash


@pytest.mark.parametrize("epoch_run", ["llama31"], indirect=True)
@pytest.mark.parametrize("change", ["wrong_training_model", "explicit_null", "wrong_count", "smoke", "incomplete", "weights"])
def test_legacy_compatibility_keeps_model_completion_and_checkpoint_guards(epoch_run, change):
    selection, training = epoch_run
    selected = run.read(selection / "manifest.json")
    selected.pop("focal_model")
    if change == "wrong_training_model":
        state = run.read(training / "manifest.json")
        state["model_key"] = "qwen3"
        run.write(training / "manifest.json", state)
        selected["training_manifest_sha256"] = run.digest(training / "manifest.json")
    if change == "explicit_null": selected["focal_model"] = None
    if change == "wrong_count": selected["validation_question_count"] = 4
    if change == "smoke": selected["smoke_test"] = True
    if change == "incomplete": selected["status"] = "running"
    if change == "weights": (selection / "epoch-1/adapter/adapter_model.safetensors").write_bytes(b"modified")
    run.write(selection / "manifest.json", selected)
    with pytest.raises(ValueError):
        run.validate_epoch_one(selection, "llama31", ROOT)


def test_complete_generate_and_score_pipeline_without_gpu(tmp_path, monkeypatch):
    predictor(monkeypatch)
    repository = tmp_path / "repo"; starter = repository / "gadi_sft_8b_starter"
    submission = starter / "outputs/compactness_test95/submission-test"
    submission.mkdir(parents=True)
    examples = panel(95); ee = {e["question_id"]: e for e in examples}
    gold = {q: {**e, "batch": "B1" if int(q[1:]) < 40 else "B2"} for q, e in ee.items()}
    run.write_jsonl(submission / "test_inputs.jsonl", run.input_view(examples))
    run.write_jsonl(submission / "test_gold.jsonl", list(gold.values()))
    baseline, baseline_paths, sources, dpo_sources, plans = {}, {}, {}, {}, {}
    for model in run.MODELS:
        sources[model] = source("sft-" + model); dpo_sources[model] = source("epoch1-" + model)
        (submission / f"{model}_prompt.txt").write_text(sources[model]["system_prompt"])
        for mode in run.MODES:
            path = tmp_path / f"sft-{model}-{mode}"
            baseline[f"{model}_{mode}"], cfg = archive(path, examples, sources[model], mode, lambda q: ["Core", "alias"])
            baseline_paths[f"{model}_{mode}"] = str(path)
        plans[model] = {k: cfg[k] for k in ("max_seq_length", "expected_prompt_tokens", "expected_prompt_ids_sha256")}
    comparison = tmp_path / "comparison"; comparison.mkdir()
    historical = score.make_pools(baseline, baseline)
    expected = {}
    for mode in run.MODES:
        rows = historical[f"baseline_{mode}"]
        for r in rows: r["label"] = int(r["answer"] == "Core")
        run.write_jsonl(comparison / f"max5_{mode}_candidates.jsonl", rows)
        cc, _, _ = score.summarize_pools({"x": rows}, ee, gold, methods=("round_robin",))
        expected[f"max5_{mode}"] = cc["x_round_robin"]["metrics"]
    (submission / "inference_code/scripts").mkdir(parents=True)
    worker = submission / "inference_code/scripts/run_single_answer_sampling.py"; worker.write_text("# frozen")
    prepared = {"repository": str(repository), "comparison": str(comparison), "rerankers": str(tmp_path / "rerankers"),
        "configurations": run.CONFIGURATIONS, "question_count": 95, "sources": sources, "dpo_sources": dpo_sources,
        "baseline_runs": baseline_paths, "input_plans": plans, "reranker_hashes": {}, "selected_C": .1,
        "file_hashes": {n: run.digest(submission / n) for n in ("test_inputs.jsonl", "test_gold.jsonl")},
        "code_hashes": {}, "external_hashes": {}, "code_snapshot": {"directory": "inference_code",
            "files_sha256": {"scripts/run_single_answer_sampling.py": run.digest(worker)}},
        "worker_sha256": run.digest(worker), "baseline_expected_metrics": expected}
    run.write(submission / "prepared.json", prepared)
    calls = []
    def fake_worker(command, check):
        calls.append(command)
        cfg = run.read(Path(command[command.index("--config") + 1]))
        assert "--preflight-all" in command and "inference_code" in command[1]
        inputs = run.records(Path(cfg["input"]))
        assert all(set(e) == {"question_id", "question", "snippets"} for e in inputs)
        count = int(command[command.index("--limit") + 1])
        if "--longest-first" in command: inputs = sorted(inputs, key=lambda e: (-100, e["question_id"]))[:count]
        path = Path(command[command.index("--output-dir") + 1])
        mode = "greedy" if cfg["num_generations"] == 1 else "sampling10"
        model = next(m for m in run.MODELS if dpo_sources[m]["adapter"] == command[command.index("--model-name") + 1])
        archive(path, inputs, dpo_sources[model], mode, lambda q: ["Core"] if q != "q1" else ["Wrong"])
    monkeypatch.setattr(run.subprocess, "run", fake_worker)
    run.generate(submission, "qwen3", starter / "smoke", True)
    jobs = dict(zip(run.MODELS, ("101.gadi-pbs", "102.gadi-pbs", "103.gadi-pbs")))
    for model in run.MODELS:
        output = starter / run.OUTPUT / f"{model}-generation-{jobs[model]}"
        run.generate(submission, model, output)
    assert len(calls) == 8  # one smoke plus three full jobs, each with two modes
    run.write(submission / "generation_jobs.json", jobs)
    monkeypatch.setattr(score, "load_frozen", lambda *args: ({}, {"max5": {"selector": "max5", "components": {}}}, {}))
    matcher = lambda ee, cc, out, jar_path: {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in cc}
    output = starter / "results"
    score.analyze(submission, output, resamples=4, matcher=matcher)
    result = run.read(output / "comparison_summary.json")
    assert run.read(output / "manifest.json")["status"] == "complete"
    assert len(result["conditions"]) == 24 and result["question_count"] == 95
    assert result["coverage_transitions"]["strict_minus_baseline_sampling10_max5_reranker"]["coverage_full_pool"]["lost"] == 0
    assert result["coverage_transitions"]["alternative_minus_baseline_sampling10_max5_reranker"]["coverage_full_pool"]["lost"] == 1
    assert result["ensemble_generation"]["alternative_sampling10"]["total_output_tokens"] < result["ensemble_generation"]["baseline_sampling10"]["total_output_tokens"]
    assert result["conditions"]["alternative_sampling10_max5_reranker"]["per_batch"]["B1"]["question_count"] == 40
    assert (output / "rankings.jsonl").is_file()
    # Code snapshot and inputs remain pinned despite later checkout edits.
    (submission / "test_inputs.jsonl").write_text("tamper")
    with pytest.raises(ValueError): run.validate_prepared(submission)


def test_preparation_reuses_six_archives_and_freezes_gold_blind_inputs(tmp_path, monkeypatch):
    repository = tmp_path / "repo"; starter = repository / "gadi_sft_8b_starter"
    (repository / "scripts").mkdir(parents=True)
    (repository / "scripts/marker.py").write_text("frozen = True")
    (starter / "jobs").mkdir(parents=True)
    for kind in ("generate", "score"):
        (starter / f"jobs/{kind}_compactness_test95.pbs").write_text("frozen job")
    jar = repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    jar.parent.mkdir(parents=True); jar.write_bytes(b"jar")
    from src.utility import bioasq_official
    monkeypatch.setattr(bioasq_official, "_official_adapter_classes", lambda path: None)
    examples = panel(95); ee = {e["question_id"]: {k: e[k] for k in ("question_id", "question", "snippets")} for e in examples}
    gold = {e["question_id"]: {**e, "batch": "B1"} for e in examples}
    comparison = starter / "comparison"; comparison.mkdir()
    (comparison / "manifest.json").write_text("{}")
    rerankers = starter / "rerankers"; rerankers.mkdir()
    membership = {"train": [{"question_id": f"t{i}"} for i in range(1296)],
                  "validation": [{"question_id": f"v{i}"} for i in range(144)]}
    for split, rr in membership.items(): run.write_jsonl(rerankers / f"{split}_examples.jsonl", rr)
    run.write(rerankers / "manifest.json", {})
    hashes = {p.name: run.digest(p) for p in rerankers.iterdir()}
    sources, dpo, loaded = {}, {}, {}
    selections = {m: starter / ("selection-" + m) for m in run.MODELS}
    states = {"sources": {}}
    base = starter / "base"; base.mkdir(); run.write(base / "config.json", {})
    for model in run.MODELS:
        adapter = starter / (model + "_sft"); adapter.mkdir(); (adapter / "weights").write_bytes(b"sft")
        sources[model] = {**source(adapter), "base_snapshot": str(base), "adapter_files_sha256": {"weights": run.digest(adapter / "weights")}}
        adapter1 = starter / (model + "_epoch1"); adapter1.mkdir(); (adapter1 / "weights").write_bytes(b"dpo")
        dpo[model] = {**sources[model], "adapter": str(adapter1), "adapter_files_sha256": {"weights": run.digest(adapter1 / "weights")}}
        for mode in run.MODES:
            directory = starter / f"{model}-{mode}"
            rows, cfg = archive(directory / "max5", list(ee.values()), sources[model], mode)
            saved = {"runs": {"max5": "max5"}}
            run.write(directory / "manifest.json", saved)
            loaded[str(directory)] = (saved, {"sources": {model: {"max5": {"source": sources[model]}}}},
                list(ee.values()), gold, {"max5": rows}, {})
            states["sources"][f"{model}_{mode}"] = {"directory": str(directory)}
    run.write(comparison / "manifest.json", states)
    original = {"conditions": {f"max5_{m}_round_robin": {"metrics": {"mrr_at5": .5}} for m in run.MODES}}
    monkeypatch.setattr(run, "load_comparison", lambda path: (original, {}, ee, gold, {"manifest.json": run.digest(comparison / "manifest.json")}))
    feedback = {"rerankers": str(rerankers), "reranker_hashes": hashes}
    splits = {"train": [{"question_id": "t0"}], "validation": [{"question_id": "v0"}]}
    monkeypatch.setattr(run, "validate_epoch_one", lambda selection, model, repo: (sources[model], dpo[model], feedback, splits, {}))
    monkeypatch.setattr(run, "load_frozen", lambda *args: ({}, {"max5": {"selected_C": .1}}, hashes))
    monkeypatch.setattr(run, "load_archive", lambda directory, mode: loaded[str(directory)])
    monkeypatch.setattr(run, "full_input_plan", lambda *args: {k: cfg[k] for k in (
        "max_seq_length", "expected_prompt_tokens", "expected_prompt_ids_sha256")})
    def snapshot(path):
        worker = path / "inference_code/scripts/run_single_answer_sampling.py"
        worker.parent.mkdir(parents=True); worker.write_text("# frozen")
        return {"directory": "inference_code", "files_sha256": {"scripts/run_single_answer_sampling.py": run.digest(worker)}}
    monkeypatch.setattr(run, "snapshot_inference_code", snapshot)
    submission = starter / "submission"
    run.prepare(repository, comparison, rerankers, selections, submission)
    prepared = run.validate_prepared(submission)
    assert len(prepared["baseline_runs"]) == 6 and prepared["configurations"] == run.CONFIGURATIONS
    assert all(set(e) == {"question_id", "question", "snippets"} for e in run.records(submission / "test_inputs.jsonl"))
    assert run.records(submission / "test_gold.jsonl") == list(gold.values())
    (repository / "scripts/marker.py").write_text("new checkout edit")
    assert (submission / "code/scripts/marker.py").read_text() == "frozen = True"
    run.validate_prepared(submission)
    (starter / "qwen3_epoch1/weights").write_bytes(b"modified")
    with pytest.raises(ValueError): run.validate_prepared(submission)


@pytest.mark.parametrize("fail_preflight", [True, False])
def test_launcher_preflights_all_models_before_seven_dependent_jobs(tmp_path, fail_preflight):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable: pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    launcher = ROOT / "gadi_sft_8b_starter/scripts/submit_compactness_test95.sh"
    stub = '''
module() { :; }
source() { :; }
python3() {
  if [[ "$*" == *'--mode record'* ]]; then return 0; fi
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"; printf frozen > "$1/generate.pbs"; printf frozen > "$1/score.pbs"; return 0
    fi
    shift
  done
}
qsub() {
  printf '%s\\n' "$*" >> "$TEST_LOG"
  local count; count=$(wc -l < "$TEST_LOG")
  printf '%s.gadi-pbs\\n' "$count"
}
'''
    target = scripts / launcher.name
    target.write_text(launcher.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "calls.log"
    result = subprocess.run([executable, target.as_posix()], capture_output=True, text=True, timeout=30,
        env={**os.environ, "TEST_USER": "test", "TEST_FAIL": str(int(fail_preflight)), "TEST_LOG": log.as_posix()})
    if fail_preflight:
        assert result.returncode and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert len(calls) == 7
        assert "afterok:1.gadi-pbs" in calls[1] and "afterok:3.gadi-pbs" in calls[3] and "afterok:5.gadi-pbs" in calls[5]
        assert "afterok:2.gadi-pbs:4.gadi-pbs:6.gadi-pbs" in calls[6]
