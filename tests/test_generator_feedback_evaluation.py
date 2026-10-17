"""The compactness continuation must be evaluated against its actual starting policy."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gadi_sft_8b_starter/scripts"))
import run_generator_feedback_evaluation as run
from scripts import analyze_generator_feedback_evaluation as score


def panel(n=2):
    return [{"question_id": f"q{i}", "question": f"Which entity {i}?",
             "snippets": [{"snippet_id": "s1", "text": "Core is the entity."}],
             "gold_aliases": ["Core"]} for i in range(n)]


def source(path="source_adapter"):
    return {"adapter": str(path), "system_prompt": "Return at most five equivalent answers.",
            "model_loader": "unsloth", "chat_template_kwargs": {}, "base_snapshot": "base"}


def archive(path, examples, ss, mode, answer_fn=None):
    path.mkdir(parents=True)
    prompt = path / "prompt.txt"; prompt.write_text(ss["system_prompt"])
    draws = 1 if mode == "greedy" else 10
    cfg = {"model_name": ss["adapter"], "max_candidates": 5, "max_new_tokens": 512,
        "seed": 3407, "response_mode": "equivalent_greedy" if draws == 1 else "equivalent_sampling",
        "num_generations": draws, "temperature": 0 if draws == 1 else .8, "top_p": .95, "top_k": 0,
        "model_loader": ss["model_loader"], "chat_template_kwargs": ss["chat_template_kwargs"],
        "require_all_snippets": True, "mark_snippets": False, "question_count": len(examples),
        "gold_blind_generation": True, "local_files_only": True, "prompt": str(prompt),
        "prompt_sha256": run.digest(prompt), "expected_prompt_tokens": {e["question_id"]: 100 for e in examples},
        "expected_prompt_ids_sha256": {e["question_id"]: "tokenhash" for e in examples}, "max_seq_length": 612}
    rows, candidates, invalid = [], [], []
    for ex in examples:
        qid = ex["question_id"]
        answers = answer_fn(qid) if answer_fn else ["Core", "alias"]
        raw = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"}
                                      for i, a in enumerate(answers)]}) if answers is not None else "bad JSON"
        try:
            aa, bad, _, _ = score.parse_equivalent_response(raw, 5)
        except ValueError:
            aa, bad = [], []
        samples = [{"draw": d, "seed": score.sample_seed(qid, d, 3407), "raw_response": raw,
                    "answers": aa, "parse_error": None if aa else "No usable candidates",
                    "output_tokens": 5 + 5*len(aa), "generation_seconds": .1} for d in range(1, draws + 1)]
        row = {"question_id": qid, "question": ex["question"], "samples": samples,
            "parse_error": None if aa else "All samples failed", "prompt_tokens": 100,
            "included_snippets": 1, "total_snippets": 1, "snippets_truncated": False,
            "request_count": draws, "input_tokens": 100*draws,
            "output_tokens": sum(s["output_tokens"] for s in samples), "generation_seconds": .1*draws}
        rows.append(row)
        candidates.extend({"question_id": qid, **r} for r in score.expansion_candidates(samples))
        invalid.extend({"question_id": qid, "draw": d, **r} for d in range(1, draws + 1) for r in bad)
    run.write(path / "config.json", cfg)
    run.write(path / "status.json", {"status": "complete"})
    for name, rr in (("examples", examples), ("generations", rows), ("candidates", candidates), ("invalid_candidates", invalid)):
        score.write_jsonl(path / f"{name}.jsonl", rr)
    return rows, cfg


@pytest.mark.parametrize("mode", score.MODES)
def test_archives_reparse_and_preserve_failed_draws(tmp_path, mode):
    examples = panel(); ss = source()
    rows, _ = archive(tmp_path / "arm", examples, ss, mode, lambda q: None if q == "q1" else ["Core"])
    assert score.load_generation(tmp_path / "arm", ss, examples, mode) == rows
    means, perq = score.compactness(rows)
    assert means["mean_candidates_per_response"] == .5
    assert len(perq) == 2 and perq[1]["unique_candidates"] == 0


@pytest.mark.parametrize("change", ["seed", "evidence", "draw", "raw", "source", "truncation", "candidates", "duplicate"])
def test_archive_mismatches_fail_closed(tmp_path, change):
    examples = panel(); ss = source(); path = tmp_path / "arm"
    rows, cfg = archive(path, examples, ss, "sampling10")
    if change == "seed":
        cfg["seed"] = 0; run.write(path / "config.json", cfg)
    elif change == "evidence":
        changed = copy.deepcopy(examples); changed[0]["snippets"][0]["text"] = "Changed"
        score.write_jsonl(path / "examples.jsonl", changed)
    elif change == "draw":
        rows[0]["samples"].pop(); score.write_jsonl(path / "generations.jsonl", rows)
    elif change == "raw":
        rows[0]["samples"][0]["raw_response"] = "bad"; score.write_jsonl(path / "generations.jsonl", rows)
    elif change == "source":
        ss["adapter"] = "other"
    elif change == "truncation":
        rows[0]["snippets_truncated"] = True; score.write_jsonl(path / "generations.jsonl", rows)
    elif change == "duplicate":
        score.write_jsonl(path / "generations.jsonl", [rows[0], rows[0]])
    else:
        score.write_jsonl(path / "candidates.jsonl", [])
    with pytest.raises(ValueError):
        score.load_generation(path, ss, examples, "sampling10")


def test_only_llama_changes_and_frequency_is_draw_normalized(tmp_path):
    examples = panel(); baseline, dpo = {}, {}
    for mode in score.MODES:
        for model in score.MODELS:
            baseline[f"{model}_{mode}"], _ = archive(tmp_path / f"{model}-{mode}", examples, source(), mode)
        dpo[mode], _ = archive(tmp_path / f"dpo-{mode}", examples, source(), mode, lambda q: ["Core"])
    frozen = copy.deepcopy(baseline)
    pools = score.make_pools(baseline, dpo)
    assert baseline == frozen
    for mode in score.MODES:
        a = {r["answer"]: r for r in pools[f"sft_{mode}"] if r["question_id"] == "q0"}
        b = {r["answer"]: r for r in pools[f"dpo_{mode}"] if r["question_id"] == "q0"}
        assert a["Core"] == b["Core"]
        assert b["alias"]["sources"] == ["qwen3", "ministral3"]
        for model in ("qwen3", "ministral3"):
            assert a["alias"]["source_ranks"][model] == b["alias"]["source_ranks"][model]
            assert a["alias"]["sample_frequencies"][model] == pytest.approx(b["alias"]["sample_frequencies"][model])
        assert all(v == pytest.approx(1) for v in b["Core"]["sample_frequencies"].values())


def test_membership_checks_full_disjoint_partitions():
    examples = panel(160)
    membership = {"train": [{"question_id": f"t{i}"} for i in range(1296)],
                  "validation": [{"question_id": f"v{i}"} for i in range(144)]}
    pref = {"train": [{"question_id": "t1"}], "validation": [{"question_id": "v1"}]}
    score.validate_membership(examples, membership, pref)
    membership["train"][0]["question_id"] = "q1"
    with pytest.raises(ValueError, match="overlap"):
        score.validate_membership(examples, membership, pref)
    membership["train"][0]["question_id"] = "t0"
    pref["validation"][0]["question_id"] = "q3"
    with pytest.raises(ValueError, match="outside"):
        score.validate_membership(examples, membership, pref)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    training = tmp_path / "training"; training.mkdir()
    submission = tmp_path / "submission"; submission.mkdir()
    preferences = tmp_path / "preferences"; preferences.mkdir()
    run.write(preferences / "manifest.json", {})
    adapter = training / "adapter"; adapter.mkdir()
    run.write(adapter / "adapter_config.json", {"base_model_name_or_path": "base"})
    run.write(adapter / "tokenizer_config.json", {})
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    score.write_jsonl(training / "reference_logps.jsonl", [])
    ss, feedback = source(), {"banks": {"llama31": "native"}}
    splits = {"train": [{"question_id": "t1"}] * 36, "validation": [{"question_id": "v1"}] * 4}
    prepared = {"configuration": {"beta": .1}, "source": ss, "preferences": str(preferences),
                "native_banks": feedback["banks"], "preferences_manifest_sha256": run.digest(preferences / "manifest.json")}
    run.write(submission / "prepared.json", prepared)
    state = {"status": "complete", "smoke_test": False, "model_key": "llama31", "phase": "save_adapter",
        "global_steps": 27, "submission": str(submission), "prepared_sha256": run.digest(submission / "prepared.json"),
        "configuration": prepared["configuration"], "source": ss, "trained_pair_counts": {"train": 36, "validation": 4},
        "preference_pair_counts": {"train": 36, "validation": 4},
        "preferences_manifest_sha256": prepared["preferences_manifest_sha256"],
        "reference_logps_sha256": run.digest(training / "reference_logps.jsonl"),
        "adapter_files_sha256": {p.name: run.digest(p) for p in adapter.iterdir()}}
    run.write(training / "manifest.json", state)
    monkeypatch.setattr(run, "validate_native", lambda *a: ({}, splits, ss, feedback))
    return training, state


@pytest.mark.parametrize("corrupt", [None, "source", "counts", "weights", "reference", "smoke"])
def test_dpo_requires_exact_training_lineage(trained, corrupt):
    training, state = trained
    if corrupt == "source": state["source"] = source("unrelated_sft")
    if corrupt == "counts": state["trained_pair_counts"]["train"] = 1
    if corrupt == "smoke": state["smoke_test"] = True
    if corrupt == "weights": (training / "adapter/adapter_model.safetensors").write_bytes(b"changed")
    if corrupt == "reference": (training / "reference_logps.jsonl").write_text("changed")
    run.write(training / "manifest.json", state)
    if corrupt:
        with pytest.raises(ValueError): run.validate_training(training, ROOT)
    else:
        assert run.validate_training(training, ROOT)[0]["global_steps"] == 27


def test_full_scoring_keeps_gains_losses_and_blinds_features(tmp_path, monkeypatch):
    examples = panel(3); baseline, dpo = {}, {}
    for mode in score.MODES:
        for model in score.MODELS:
            baseline[f"{model}_{mode}"], _ = archive(tmp_path / f"{model}-{mode}", examples, source(), mode,
                lambda q, model=model: ["Core", "alias"] if model == "llama31" and q != "q2" else ["Wrong"])
        dpo[mode], _ = archive(tmp_path / f"dpo-{mode}", examples, source(), mode,
                              lambda q: ["Wrong"] if q == "q1" else ["Core"])
    from scripts import evaluate_sft_ensemble_rerankers as ev
    def blind_predict(components, features, contexts):
        assert all("label" not in r for r in features)
        assert all(set(e) == {"question", "snippets"} for e in contexts.values())
        return np.asarray([2. if r["answer"] == "Core" else 0. for r in features])
    monkeypatch.setattr(ev, "score_components", blind_predict)
    calls = []
    def matcher(ee, cc, output, jar_path):
        assert output.is_dir()
        (output / "scorer_gold.json").write_text("matcher needs an existing directory")
        calls.append(cc)
        return {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in cc}
    conditions, scores, rankings = score.score_pools(score.make_pools(baseline, dpo),
        {e["question_id"]: e for e in examples}, {"selector": "max5", "components": {}}, tmp_path, Path("jar"), matcher)
    assert len(calls) == 1
    a, b = (scores[f"{p}_sampling10_max5_reranker"] for p in ("sft", "dpo"))
    trans = score.coverage_transitions(a, b, "coverage_full_pool")
    assert trans["gained_question_ids"] == ["q2"] and trans["lost_question_ids"] == ["q1"]
    assert trans["net"] == 0
    assert conditions["sft_sampling10_max5_reranker"]["metrics"]["mrr_at5"] == pytest.approx(2/3)
    assert conditions["dpo_sampling10_max5_reranker"]["metrics"]["mrr_at5"] == pytest.approx(2/3)
    assert rankings and len(conditions) == 12


def test_real_official_matcher_creates_nested_report_directory(tmp_path, monkeypatch):
    # Use the actual Java adapter and writer here: a matcher stub previously
    # hid the missing output-directory failure on Gadi.
    from scripts import evaluate_sft_ensemble_rerankers as ev
    monkeypatch.setattr(ev, "score_components", lambda components, rows, contexts: np.asarray(
        [1. if r["answer"] == "Core" else 0. for r in rows]))
    examples = {e["question_id"]: e for e in panel(2)}
    rows = [{"question_id": q, "answer": answer, "sources": ["llama31"],
             "source_ranks": {"llama31": rank}, "sample_frequencies": {"llama31": 1.},
             "relation_types": ["original"]}
            for q in examples for rank, answer in enumerate(("Wrong", "Core"), 1)]
    output = tmp_path / "new/nested/result"
    conditions, _, _ = score.score_pools({"sft_greedy": rows}, examples,
        {"selector": "max5", "components": {}}, output,
        ROOT / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar")
    assert (output / "official/scorer_gold.json").is_file()
    assert (output / "official/official_candidate_scores.jsonl").is_file()
    assert conditions["sft_greedy_max5_reranker"]["metrics"]["mrr_at5"] == 1
    assert {r["label"] for r in rows} == {0, 1}


def test_preparation_freezes_code_and_gold_blind_inputs(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    starter = repository / "gadi_sft_8b_starter"
    for folder in ("configs", "jobs", "scripts", "data"):
        (starter / folder).mkdir(parents=True)
    (repository / "scripts").mkdir()
    (repository / "src").mkdir()
    (repository / "scripts/marker.py").write_text("pinned = True")
    for kind in ("generate", "score"):
        (starter / f"jobs/{kind}_generator_feedback_evaluation.pbs").write_text("frozen job")
    jar = repository / "third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar"
    jar.parent.mkdir(parents=True); jar.write_bytes(b"jar")
    from src.utility import bioasq_official
    monkeypatch.setattr(bioasq_official, "_official_adapter_classes", lambda path: None)
    examples = panel(160)
    score.write_jsonl(starter / "data/dev.jsonl", examples)
    run.write(starter / "configs/matrix.json", {"outer_dev": "data/dev.jsonl"})
    run.write(starter / "configs/reranker_guided_dpo_pilot.json", {})
    config = {"training_matrix": "configs/matrix.json", "models": {m: m for m in score.MODELS}}
    run.write(starter / "configs/generation.json", config)
    base = starter / "base"; base.mkdir(); run.write(base / "config.json", {})
    training = starter / "training"; (training / "adapter").mkdir(parents=True)
    run.write(training / "manifest.json", {})
    score.write_jsonl(training / "reference_logps.jsonl", [])
    (training / "adapter/weights").write_bytes(b"dpo")
    sources, archives = {}, {}
    for model in score.MODELS:
        adapter = starter / f"{model}_adapter"; adapter.mkdir()
        (adapter / "weights").write_bytes(b"sft")
        sources[model] = {**source(adapter), "base_snapshot": str(base),
                          "adapter_files_sha256": {"weights": run.digest(adapter / "weights")}}
        directory = starter / model
        paths = {}; hashes = {}
        for mode in score.MODES:
            path = directory / mode
            _, cfg = archive(path, examples, sources[model], mode)
            paths[f"best_eval_loss_{mode}"] = path
            hashes[f"best_eval_loss_{mode}"] = {n: run.digest(path / n) for n in (
                "config.json", "examples.jsonl", "generations.jsonl", "candidates.jsonl", "invalid_candidates.jsonl", "status.json")}
        saved = {"model_key": model, "common_settings": {"max_candidates": 5}, "arm_hashes": hashes}
        run.write(directory / "manifest.json", saved)
        archives[str(directory)] = (saved, paths)
    rerankers = starter / "rerankers"; rerankers.mkdir()
    membership = {"train": [{"question_id": f"t{i}"} for i in range(1296)],
                  "validation": [{"question_id": f"v{i}"} for i in range(144)]}
    for s, rr in membership.items(): score.write_jsonl(rerankers / f"{s}_examples.jsonl", rr)
    run.write(rerankers / "manifest.json", {})
    hashes = {"manifest.json": run.digest(rerankers / "manifest.json")}
    feedback = {"rerankers": str(rerankers), "reranker_hashes": hashes, "banks": {}}
    trainstate = {"adapter_files_sha256": {"weights": run.digest(training / "adapter/weights")}}
    monkeypatch.setattr(run, "validate_training", lambda *args: (
        trainstate, {}, sources["llama31"], feedback,
        {"train": [{"question_id": "t0"}], "validation": [{"question_id": "v0"}]}))
    monkeypatch.setattr(run, "configure_data_root", lambda repo: SimpleNamespace(
        DEFAULT_CONFIG=starter / "configs/generation.json", selected_source=lambda cfg, m: sources[m]))
    monkeypatch.setattr(run, "load_inputs", lambda *args: (None, None, None, None, hashes, {}, None))
    monkeypatch.setattr(run, "load_selector", lambda *args: ({"selected_C": .1}, {}))
    monkeypatch.setattr(run, "validate_experiment", lambda directory: archives[str(directory)])
    plan = {k: cfg[k] for k in ("expected_prompt_tokens", "expected_prompt_ids_sha256", "max_seq_length")}
    monkeypatch.setattr(run, "full_input_plan", lambda *args: plan)
    submission = starter / "evaluation"
    run.prepare(repository, training, submission)
    prepared = run.validate_prepared(submission)
    assert len(prepared["baseline_runs"]) == 6
    assert all(set(e) == {"question_id", "question", "snippets"} for e in run.records(submission / "dev_inputs.jsonl"))
    assert run.records(submission / "dev_gold.jsonl") == examples
    (repository / "scripts/marker.py").write_text("later checkout edit")
    assert (submission / "code/scripts/marker.py").read_text() == "pinned = True"
    run.validate_prepared(submission)
    # Exercise smoke -> full inference orchestration and the CPU result writer
    # without loading GPU weights or invoking the official Java process here.
    invocations = []
    def fake_inference(command, check):
        invocations.append(command)
        cfg = run.read(Path(command[command.index("--config") + 1]))
        assert "--preflight-all" in command and "inference_code" in command[1]
        rows = run.records(Path(cfg["input"]))
        assert all("gold_aliases" not in row for row in rows)
        mode = "greedy" if cfg["num_generations"] == 1 else "sampling10"
        if "--longest-first" in command:
            rows = sorted(rows, key=lambda e: (-100, e["question_id"]))[:4]
        archive(Path(command[command.index("--output-dir") + 1]), rows,
                {**sources["llama31"], "adapter": str(training / "adapter")}, mode, lambda q: ["Core"])
    monkeypatch.setattr(run.subprocess, "run", fake_inference)
    smoke = starter / "smoke"
    run.generate(submission, smoke, True)
    assert run.read(smoke / "manifest.json")["expected_questions"] == 4
    generation = starter / "generation"
    run.generate(submission, generation)
    assert len(invocations) == 4
    monkeypatch.setattr(score, "load_selector", lambda *args: ({"selector": "max5", "components": {}, "selected_C": .1}, {}))
    from scripts import evaluate_sft_ensemble_rerankers as ev
    monkeypatch.setattr(ev, "score_components", lambda components, rows, contexts: np.asarray(
        [1. if r["answer"] == "Core" else 0. for r in rows]))
    actual_score = score.score_pools
    def fake_match(examples, candidates, output, jar_path):
        return {(r["question_id"], r["answer"]): r["answer"] == "Core" for r in candidates}
    monkeypatch.setattr(score, "score_pools", lambda *args: actual_score(*args, matcher=fake_match))
    output = starter / "scored"
    score.analyze(submission, generation, output, resamples=2)
    summary = run.read(output / "comparison_summary.json")
    assert run.read(output / "manifest.json")["status"] == "complete"
    assert summary["question_count"] == 160 and len(summary["conditions"]) == 12
    assert summary["llama_generation"]["dpo_sampling10"]["mean_candidates_per_response"] == 1
    assert summary["coverage_transitions"]["dpo_minus_sft_sampling10_max5_reranker"]["coverage_full_pool"]["lost"] == 0
    assert (output / "rankings.jsonl").is_file()
    (training / "adapter/weights").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="Pinned evaluation input changed"):
        run.validate_prepared(submission)


@pytest.mark.parametrize("fail", [True, False])
def test_submission_preflight_before_three_dependent_jobs(tmp_path, fail):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash unavailable")
    scripts = tmp_path / "scripts"; scripts.mkdir()
    training = tmp_path / "trained"; training.mkdir()
    launcher = ROOT / "gadi_sft_8b_starter/scripts/submit_generator_feedback_evaluation.sh"
    stub = '''
module() { :; }
source() { :; }
python() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"; printf frozen > "$1/generate.pbs"; printf frozen > "$1/score.pbs"; return 0
    fi
    shift
  done
}
qsub() { printf '%s\\n' "$*" >> "$TEST_LOG"; printf '123.gadi-pbs\\n'; }
'''
    target = scripts / launcher.name
    target.write_text(launcher.read_text().replace("${USER}", "${TEST_USER}").replace(
        "set -euo pipefail", "set -euo pipefail\n" + stub), newline="\n")
    log = tmp_path / "calls.log"
    result = subprocess.run([executable, target.as_posix(), training.as_posix()], capture_output=True, text=True,
        env={**os.environ, "TEST_USER": "test", "TEST_FAIL": str(int(fail)), "TEST_LOG": log.as_posix()}, timeout=30)
    if fail:
        assert result.returncode and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert len(calls) == 3 and "afterok:123.gadi-pbs" in calls[1] and "afterok:123.gadi-pbs" in calls[2]
        assert "generation-123.gadi-pbs" in calls[2]
        assert "generate.pbs" in calls[0] and "score.pbs" in calls[2]
