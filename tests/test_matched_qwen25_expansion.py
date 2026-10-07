import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from scripts.analyze_matched_qwen_expansion import paired_contrast, ranking_metrics, validate_pair


ROOT = Path(__file__).resolve().parents[1]


def test_ranking_scores_keep_failures_and_enforce_top_five():
    rows = [
        {"question_id": "first", "answers": ["gold"], "matching_answers": ["gold"]},
        {"question_id": "third", "answers": ["a", "b", "gold"], "matching_answers": ["gold"]},
        {"question_id": "sixth", "answers": ["a", "b", "c", "d", "e", "gold"], "matching_answers": ["gold"]},
        {"question_id": "parse_failed", "answers": [], "matching_answers": []},
    ]
    metrics, per_question = ranking_metrics(rows)
    assert metrics["mrr_at5"] == pytest.approx(1 / 3)
    assert metrics["strict_accuracy"] == 0.25
    assert metrics["lenient_accuracy"] == metrics["coverage_at5"] == 0.5
    assert metrics["coverage_at10"] == 0.75
    assert len(per_question) == 4
    assert per_question[-1]["mrr_at5"] == 0


def test_paired_contrast_uses_ids_rather_than_row_order():
    _, base = ranking_metrics([
        {"question_id": "a", "answers": [], "matching_answers": []},
        {"question_id": "b", "answers": ["gold"], "matching_answers": ["gold"]},
    ])
    result = paired_contrast(base, list(reversed(base)), resamples=200)
    assert result["mrr_at5"]["sft_minus_base"] == 0
    assert result["mrr_at5"]["paired_bootstrap_95ci"] == [0, 0]
    with pytest.raises(ValueError, match="different or duplicate"):
        paired_contrast(base, [base[0], base[0]], resamples=200)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pair_validation_rejects_changed_evidence_or_missing_questions(tmp_path):
    paths = [tmp_path / "base", tmp_path / "sft"]
    for path in paths:
        write_json(path / "config.json", {"input_sha256": "same", "prompt_sha256": "same"})
        write_json(path / "status.json", {"status": "complete"})
        (path / "examples.jsonl").write_text(json.dumps({"question_id": "a", "snippets": ["evidence"]}) + "\n")
        (path / "generations.jsonl").write_text(json.dumps({"question_id": "a", "parse_error": "bad JSON"}) + "\n")
    validate_pair(paths, 1)
    (paths[1] / "examples.jsonl").write_text(json.dumps({"question_id": "a", "snippets": ["changed"]}) + "\n")
    with pytest.raises(ValueError, match="identical questions"):
        validate_pair(paths, 1)
    (paths[1] / "examples.jsonl").write_bytes((paths[0] / "examples.jsonl").read_bytes())
    (paths[1] / "generations.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete generation"):
        validate_pair(paths, 1)


def test_completed_adapter_validation_rejects_development_leakage(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("matched_runner", ROOT / "gadi_sft_8b_starter/scripts/run_matched_qwen25_expansion.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: None)
    config = {"model_name": "base", "input": "data/dev.jsonl"}
    write_json(tmp_path / "configs/equivalent_expansion_qwen25_05b.json", config)

    def records(path, prefix, count):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"question_id": f"{prefix}-{i}"}) + "\n" for i in range(count)))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    records(tmp_path / "data/dev.jsonl", "dev", 160)
    hashes = {name: records(tmp_path / f"data/{name}.jsonl", name, count)
              for name, count in (("train", 1296), ("validation", 144))}
    training = tmp_path / "outputs/expansion_sft" / runner.TRAINING_RUNS["05b"]
    status = {"model": "base", "smoke_test": False,
              "configuration": {"train_input": "data/train.jsonl", "eval_input": "data/validation.jsonl"},
              "dataset_validation": {f"{name}_sha256": digest for name, digest in hashes.items()}}
    write_json(training / "status.json", status)
    write_json(training / "adapter/training_complete.json", {"status": "completed"})
    write_json(training / "adapter/adapter_config.json", {"base_model_name_or_path": "base"})
    (training / "adapter/adapter_model.safetensors").write_bytes(b"fixture")
    runner.validate("05b")
    train_file = tmp_path / "data/train.jsonl"
    train_file.write_text(train_file.read_text().replace('train-0"', 'dev-0"'))
    status["dataset_validation"]["train_sha256"] = hashlib.sha256(train_file.read_bytes()).hexdigest()
    write_json(training / "status.json", status)
    with pytest.raises(ValueError, match="overlap"):
        runner.validate("05b")
