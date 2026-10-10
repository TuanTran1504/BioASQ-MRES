"""Verify DPO provenance and matched inference without a GPU or scheduler."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gadi_sft_8b_starter/scripts"))
import run_expansion_dpo_evaluation as runner
from run_extractive_expansion_8b import parse_equivalent_response
from run_single_answer_sampling import expansion_candidates, sample_seed
from scripts import analyze_expansion_dpo_evaluation as analysis


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DPO_CONFIG", tmp_path / "dpo.json")
    write(tmp_path / "dpo.json", {})
    evaluation = {"input": "dev.jsonl"}
    write(tmp_path / "sft.json", evaluation)
    jsonl(tmp_path / "dev.jsonl", [{"question_id": "dev"}])
    source = {"adapter": "sft adapter", "base_snapshot": "base", "base_revision": "rev",
              "matrix_sha256": "matrix", "system_prompt": "expansion", "model_loader": "fast_language_model",
              "chat_template_kwargs": {}, "adapter_files_sha256": {"weights": "sft"},
              "status_sha256": "sft status", "execution_mode": "default"}
    monkeypatch.setattr(runner, "validate_training_pair", lambda *args: {"expansion": copy.deepcopy(source)})
    prefs = {"pair_counts": {"train": 159, "validation": 48},
             "banks": {"llama31": {"manifest": {"source": source}}}}
    write(tmp_path / "preferences/manifest.json", prefs)
    pairs = {"train": [{"question_id": "train"}], "validation": [{"question_id": "validation"}]}
    monkeypatch.setattr(runner, "validate_pairs", lambda *args: (prefs, pairs))
    adapter = tmp_path / "dpo/adapter"
    write(adapter / "adapter_config.json", {"base_model_name_or_path": "base"})
    (adapter / "adapter_model.safetensors").write_bytes(b"DPO weights")
    write(adapter / "tokenizer_config.json", {})
    jsonl(tmp_path / "dpo/reference_logps.jsonl", [{"ref_chosen": -1, "ref_rejected": -2}])
    state = {"status": "complete", "phase": "save_adapter", "model_key": "llama31", "smoke_test": False,
             "source": source, "global_steps": 10, "dpo_config_sha256": runner.digest(tmp_path / "dpo.json"),
             "preferences_manifest_sha256": runner.digest(tmp_path / "preferences/manifest.json"),
             "preference_pair_counts": prefs["pair_counts"], "trained_pair_counts": prefs["pair_counts"],
             "reference": "precomputed initial expansion SFT completion log probabilities",
             "reference_logps_sha256": runner.digest(tmp_path / "dpo/reference_logps.jsonl"),
             "adapter_files_sha256": {p.name: runner.digest(p) for p in adapter.iterdir()}}
    write(tmp_path / "dpo/manifest.json", state)
    config = {"sft_evaluation_config": "sft.json", "dpo_config": "dpo.json",
              "preferences": "preferences", "models": {"llama31": "dpo"}}
    return tmp_path, config, pairs


@pytest.mark.parametrize("failure", [None, "weights", "reference", "preferences", "smoke", "base", "overlap"])
def test_completed_dpo_requires_exact_source_and_preferences(training, failure):
    root, config, pairs = training
    if failure == "weights":
        (root / "dpo/adapter/adapter_model.safetensors").write_bytes(b"changed")
    elif failure == "reference":
        (root / "dpo/reference_logps.jsonl").write_text("changed")
    elif failure == "preferences":
        (root / "preferences/manifest.json").write_text("changed")
    elif failure in ("smoke", "base"):
        state = json.loads((root / "dpo/manifest.json").read_text())
        if failure == "smoke":
            state["smoke_test"] = True
        else:
            state["source"]["base_revision"] = "other"
        write(root / "dpo/manifest.json", state)
    elif failure == "overlap":
        pairs["train"][0]["question_id"] = "dev"
    if failure:
        with pytest.raises(ValueError):
            runner.validate_sources(config, "llama31")
    else:
        _, sources, state = runner.validate_sources(config, "llama31")
        assert sources["dpo"]["base_snapshot"] == sources["sft"]["base_snapshot"]
        assert sources["dpo"]["adapter"] != sources["sft"]["adapter"]
        assert state["trained_pair_counts"] == {"train": 159, "validation": 48}


@pytest.fixture
def experiment(tmp_path):
    source = {"adapter": "sft adapter", "base_snapshot": "base", "base_revision": "rev",
              "matrix_sha256": "matrix", "system_prompt": "expansion", "model_loader": "fast_language_model",
              "chat_template_kwargs": {"enable_thinking": False}, "adapter_files_sha256": {"weights": "sft"}}
    dpo = {**source, "adapter": "dpo adapter", "adapter_files_sha256": {"weights": "dpo"}}
    manifest = {"status": "complete", "smoke_test": False, "expected_questions": 160,
                "model_key": "qwen3", "input_sha256": analysis.INPUT_SHA, "runs": {},
                "evaluation": "development pilot", "selection": "first five in generation order",
                "provenance": {"sft": source, "dpo": dpo},
                "dpo_training_manifest": {"status": "complete", "smoke_test": False, "model_key": "qwen3",
                    "global_steps": 10, "source": source, "adapter_files_sha256": dpo["adapter_files_sha256"]}}
    prompt = tmp_path / "expansion_prompt.txt"
    prompt.write_text("expansion")
    examples = [{"question_id": str(i), "question": "How many?", "gold_aliases": ["15"],
                 "snippets": [{"snippet_id": "s", "text": "There are 15."}]} for i in range(160)]
    for name in runner.CONDITIONS:
        directory = tmp_path / name
        directory.mkdir()
        sampled = name.endswith("sampling10")
        policy = name.split("_", 1)[0]
        config = runner.arm_config({"input": "unused", "max_seq_length": 6144, "max_new_tokens": 512, "seed": 3407},
                                  {"expansion": manifest["provenance"][policy]},
                                  "expansion_sampling10" if sampled else "expansion_greedy", prompt)
        config.update(model_name=manifest["provenance"][policy]["adapter"], input_sha256=analysis.INPUT_SHA,
                      prompt_sha256=analysis.digest(prompt), gold_blind_generation=True)
        write(directory / "config.json", config)
        write(directory / "status.json", {"status": "complete"})
        jsonl(directory / "examples.jsonl", examples)
        generations, candidates = [], []
        for i in range(160):
            # Preserve actual format failures: SFT 80 correct, DPO 120 correct.
            hit = i < (80 if policy == "sft" else 120)
            raw = json.dumps({"answers": [{"answer": "15", "relation_type": "original"}]}) if hit else "bad output"
            common = {"question_id": str(i), "question": "How many?", "parse_error": None if hit else "bad format",
                      "snippets_truncated": False, "included_snippets": 1, "total_snippets": 1,
                      "request_count": 10 if sampled else 1, "input_tokens": 50 if sampled else 5,
                      "output_tokens": 30 if sampled else 3, "generation_seconds": 1 if sampled else 0.1}
            answers = parse_equivalent_response(raw)[0] if hit else []
            if sampled:
                samples = [{"draw": d, "seed": sample_seed(str(i), d), "raw_response": raw, "answers": answers,
                            "parse_error": common["parse_error"], "schema_compliant": hit} for d in range(1, 11)]
                common["samples"] = samples
                candidates.extend({"question_id": str(i), **c} for c in expansion_candidates(samples))
            else:
                common["raw_response"] = raw
                candidates.extend({"question_id": str(i), "position": c["raw_position"], **c} for c in answers)
            generations.append(common)
        jsonl(directory / "generations.jsonl", generations)
        jsonl(directory / "candidates.jsonl", candidates)
        manifest["runs"][name] = name
    write(tmp_path / "manifest.json", manifest)
    return tmp_path


@pytest.mark.parametrize("failure", ["seed", "draws", "evidence", "ordering", "adapter", "prompt", "reference", "budget", "truncation"])
def test_analysis_rejects_changed_protocol(experiment, failure):
    directory = experiment / "dpo_sampling10"
    if failure == "evidence":
        rows = analysis.records(directory / "examples.jsonl")
        rows[0]["snippets"][0]["text"] = "changed"
        jsonl(directory / "examples.jsonl", rows)
    elif failure in ("adapter", "budget"):
        config = analysis.read(directory / "config.json")
        config["model_name" if failure == "adapter" else "max_new_tokens"] = "wrong" if failure == "adapter" else 256
        write(directory / "config.json", config)
    elif failure == "prompt":
        (experiment / "expansion_prompt.txt").write_text("changed")
    elif failure == "reference":
        manifest = analysis.read(experiment / "manifest.json")
        manifest["dpo_training_manifest"]["source"]["base_revision"] = "different"
        write(experiment / "manifest.json", manifest)
    elif failure == "ordering":
        rows = analysis.records(directory / "candidates.jsonl")
        rows[0]["position"] = 2
        jsonl(directory / "candidates.jsonl", rows)
    else:
        rows = analysis.records(directory / "generations.jsonl")
        if failure == "seed": rows[0]["samples"][0]["seed"] = 0
        if failure == "draws": rows[0]["samples"].pop()
        if failure == "truncation": rows[0]["snippets_truncated"] = True
        jsonl(directory / "generations.jsonl", rows)
    with pytest.raises(ValueError):
        analysis.validate_experiment(experiment)


def test_scoring_keeps_failures_and_compares_equal_budgets(experiment, monkeypatch):
    from src.notebook_workflows import local_expansion
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
                        {(r["question_id"], r["answer"]): r["answer"] == "15" for r in candidates})
    result = analysis.score_experiment(experiment, Path("unused.jar"))
    assert result["conditions"]["sft_greedy"]["metrics"]["mrr_at5"] == 0.5
    assert result["conditions"]["dpo_sampling10"]["metrics"]["mrr_at5"] == 0.75
    assert result["conditions"]["sft_greedy"]["strict_training_schema"]["rate"] == 0.5
    assert result["conditions"]["dpo_sampling10"]["efficiency"]["request_count"] == 1600
    for inference in ("greedy", "sampling10"):
        contrast = result["paired_contrasts"]["dpo_minus_sft_" + inference]["mrr_at5"]
        assert contrast["dpo_minus_sft"] == 0.25
        assert contrast["paired_bootstrap_95ci"][0] > 0
    assert len(result["paired_contrasts"]) == 2


def test_full_pool_hit_beyond_ten_does_not_count_as_submission_success(experiment, monkeypatch):
    directory = experiment / "dpo_sampling10"
    generations = analysis.records(directory / "generations.jsonl")
    candidates = []
    for row in generations:
        row["parse_error"] = None
        for sample in row["samples"]:
            values = [{"answer": f"wrong-{i}", "relation_type": "original" if i == 1 else "synonym"}
                      for i in range(1, 11)] if sample["draw"] == 1 else [{"answer": "15", "relation_type": "original"}]
            sample.update(raw_response=json.dumps({"answers": values}), parse_error=None, schema_compliant=True)
            sample["answers"] = parse_equivalent_response(sample["raw_response"])[0]
        candidates.extend({"question_id": row["question_id"], **c} for c in expansion_candidates(row["samples"]))
    jsonl(directory / "generations.jsonl", generations)
    jsonl(directory / "candidates.jsonl", candidates)
    from src.notebook_workflows import local_expansion
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
                        {(r["question_id"], r["answer"]): r["answer"] == "15" for r in candidates})
    metrics = analysis.score_experiment(experiment, Path("unused.jar"))["conditions"]["dpo_sampling10"]["metrics"]
    assert metrics["mrr_at5"] == 0
    assert metrics["coverage_at10"] == 0
    assert metrics["coverage_full_pool"] == 1


@pytest.mark.parametrize("models,fail,expected", [([], False, 6), (["ministral3"], False, 2),
                        ([], True, 0), (["qwen3", "qwen3"], False, 0)])
def test_submission_validates_all_before_scheduling(tmp_path, models, fail, expected):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable")
    root = Path(__file__).resolve().parents[1]
    source = root / "gadi_sft_8b_starter/scripts/submit_expansion_dpo_evaluation.sh"
    text = source.read_text().replace('source "/scratch/nl78/${USER}/venvs/bioasq-8b/bin/activate"', ':')
    mocks = r'''
module() { :; }
python() { printf 'validate %s\n' "$*" >> "$MOCK_LOG"; [[ "$MOCK_FAIL" == 0 ]]; }
qsub() { printf 'qsub %s\n' "$*" >> "$MOCK_LOG"; n="$(cat "$MOCK_COUNT")"; printf '%s\n' "$((n + 1))" > "$MOCK_COUNT"; printf '%s.gadi-pbs\n' "$n"; }
'''
    text = text.replace("set -euo pipefail", "set -euo pipefail\n" + mocks)
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / source.name
    script.write_text(text, newline="\n")
    log, count = tmp_path / "log", tmp_path / "count"
    count.write_text("1\n")
    result = subprocess.run([bash, script.as_posix(), *models], capture_output=True, text=True,
        env={**os.environ, "USER": "mock", "MOCK_FAIL": str(int(fail)), "MOCK_LOG": log.as_posix(), "MOCK_COUNT": count.as_posix()})
    lines = log.read_text().splitlines() if log.exists() else []
    jobs = [s for s in lines if s.startswith("qsub ")]
    assert len(jobs) == expected, result.stderr
    assert result.returncode == (0 if expected else 1), result.stderr
    if jobs:
        assert next(i for i, line in enumerate(lines) if line.startswith("qsub ")) == len(models or [1, 2, 3])
        for i in range(0, expected, 2):
            assert "SMOKE_TEST=1" in jobs[i] and "walltime=02:00:00" in jobs[i]
            assert "SMOKE_TEST=0" in jobs[i+1] and f"depend=afterok:{i+1}.gadi-pbs" in jobs[i+1]
