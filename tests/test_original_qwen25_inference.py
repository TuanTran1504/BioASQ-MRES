import importlib.util
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.analyze_original_qwen25_inference import sampling_diagnostics, validate_inference_pair
from scripts.reparse_original_qwen25_sampling import reparse_samples

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "gadi_sft_8b_starter"


def load_script(name):
    original_path = sys.path[:]
    scripts = str(BUNDLE / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, BUNDLE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


def test_sampling_parser_rejects_multiple_answers_and_preserves_failed_draw_budget():
    sampler = load_script("run_single_answer_sampling")
    assert sampler.parse_single_answer("[BE] alpha  beta [EE]") == "alpha beta"
    assert sampler.parse_single_answer("Answer: [BE] alpha beta [EE]") == "alpha beta"
    assert sampler.parse_single_answer("\nanswer: [BE] alpha [EE]\n") == "alpha"
    for raw in ("alpha", "[BE][EE]", "[BE]alpha[EE][BE]beta[EE]", "Reasoning [BE]alpha[EE]"):
        with pytest.raises(ValueError):
            sampler.parse_single_answer(raw)
    result = sampler.unique_candidates([
        {"draw": 1, "answer": None}, {"draw": 2, "answer": "Alpha"},
        {"draw": 3, "answer": "alpha"}, {"draw": 4, "answer": "Beta"}])
    assert [(r["answer"], r["position"], r["raw_position"]) for r in result] == [("Alpha", 1, 2), ("Beta", 2, 4)]
    assert sampler.sample_seed("q", 2) == sampler.sample_seed("q", 2)
    assert sampler.sample_seed("q", 2) != sampler.sample_seed("q", 3)


@pytest.mark.parametrize("draw_count,temperature,mark_snippets,expansion", [(10, 0.8, True, False), (10, 0.8, False, False), (1, 0, False, False), (10, 0.8, False, True)])
def test_sampler_actually_makes_seeded_calls_and_saves_failures(tmp_path, monkeypatch, draw_count, temperature, mark_snippets, expansion):
    sampler = load_script("run_single_answer_sampling")
    monkeypatch.setattr(sampler, "ROOT", tmp_path)
    monkeypatch.setattr(sampler, "configure_job_local_compiler_cache", lambda: None)
    calls, seeds, rendered = [], [], []

    class Tokens:
        shape = (1, 5)
        def to(self, device): return self
        def numel(self): return 3

    class Output:
        def __getitem__(self, item): return Tokens()

    class Model:
        def parameters(self): return iter([SimpleNamespace(device="fake")])
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Output()

    class Tokenizer:
        pad_token_id = eos_token_id = 1
        def apply_chat_template(self, messages, **kwargs):
            rendered.append(messages)
            return "question and evidence"
        def __call__(self, **kwargs): return {"input_ids": Tokens()}
        def decode(self, ids, **kwargs):
            if expansion:
                if len(calls) == 2:
                    return "invalid"
                return json.dumps({"answers": [{"answer": "Alpha", "relation_type": "original"},
                    {"answer": "Beta" if len(calls) == 10 else "alpha", "relation_type": "synonym"}]})
            return "invalid" if len(calls) == 2 else "Answer: [BE] Beta [EE]" if len(calls) == 10 else "Answer: [BE] Alpha [EE]"

    fake_torch = SimpleNamespace(inference_mode=nullcontext, manual_seed=seeds.append,
                                 cuda=SimpleNamespace(is_available=lambda: True,
                                                      manual_seed_all=lambda seed: None, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sampler, "load_model", lambda *a: (Model(), Tokenizer()))
    (tmp_path / "dev.jsonl").write_text(json.dumps({"question_id": "q", "question": "Question?",
        "snippets": [{"snippet_id": "s", "text": "Evidence"}], "gold_aliases": ["SECRET_GOLD"]}) + "\n")
    (tmp_path / "prompt.txt").write_text("Single answer")
    config = {"input": "dev.jsonl", "prompt": "prompt.txt", "num_generations": draw_count,
              "temperature": temperature, "top_p": 0.95, "seed": 3407, "max_seq_length": 6144,
              "max_new_tokens": 512, "mark_snippets": mark_snippets,
              "response_mode": "single_answer_greedy" if draw_count == 1 else "single_answer_sampling"}
    if expansion:
        config["response_mode"] = "equivalent_sampling"
    (tmp_path / "config.json").write_text(json.dumps(config))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["sampling", "--config", str(tmp_path / "config.json"),
                                      "--model-name", "adapter", "--output-dir", str(output), "--limit", "1"])
    sampler.main()
    assert len(calls) == len(set(seeds)) == draw_count
    if draw_count == 10:
        assert all(c["do_sample"] and c["temperature"] == 0.8 and c["top_p"] == 0.95 and c["top_k"] == 0 for c in calls)
    else:
        assert calls[0]["do_sample"] is False
        assert "temperature" not in calls[0] and "top_p" not in calls[0]
    assert "SECRET_GOLD" not in str(rendered)
    assert ("[BS] Evidence [ES]" in str(rendered)) == mark_snippets
    assert "Evidence" in str(rendered)
    generation = json.loads((output / "generations.jsonl").read_text())
    if draw_count == 10:
        assert generation["samples"][1]["parse_error"]
    assert generation["request_count"] == draw_count and generation["input_tokens"] == 5 * draw_count
    assert len(generation["samples"]) == draw_count
    assert [json.loads(line)["answer"] for line in (output / "candidates.jsonl").read_text().splitlines()] == (["Alpha", "Beta"] if draw_count == 10 else ["Alpha"])
    if expansion:
        candidates = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
        assert [(c["position"], c["draw"], c["within_draw_position"]) for c in candidates] == [(1, 1, 1), (2, 10, 2)]
        assert generation["samples"][1]["answers"] == []
        assert (output / "invalid_candidates.jsonl").is_file()


def test_sampling_diagnostics_distinguish_draws_from_unique_ranks(tmp_path):
    samples = [{"draw": i, "answer": "gold" if i == 10 else "wrong", "parse_error": None} for i in range(1, 11)]
    (tmp_path / "generations.jsonl").write_text(json.dumps({"question_id": "q", "samples": samples}) + "\n")
    result = sampling_diagnostics(tmp_path, [{"question_id": "q", "matching_answers": ["gold"]}])
    assert result["coverage_within_first_5_draws"] == 0
    assert result["coverage_within_first_10_draws"] == 1
    assert result["attempted_draws"] == 10


def test_inference_validation_rejects_nine_draws_and_changed_evidence(tmp_path):
    paths = [tmp_path / "expansion", tmp_path / "sampling"]
    base = {"model_name": "adapter", "input_sha256": "same", "max_seq_length": 6144,
            "max_new_tokens": 512, "require_all_snippets": True, "chat_template_kwargs": {}}
    for path, mode in zip(paths, ("equivalent", "single_answer_sampling")):
        path.mkdir()
        config = {**base, "response_mode": mode, "temperature": 0 if mode == "equivalent" else 0.8,
                  "num_generations": 10, "top_p": 0.95, "seed": 3407}
        (path / "config.json").write_text(json.dumps(config))
        (path / "status.json").write_text(json.dumps({"status": "complete"}))
        (path / "examples.jsonl").write_text("".join(json.dumps({"question_id": str(i), "snippets": ["same"]}) + "\n" for i in range(160)))
        (path / "generations.jsonl").write_text("".join(json.dumps({"question_id": str(i), "samples": [{"draw": d} for d in range(1, 11)]}) + "\n" for i in range(160)))
    validate_inference_pair(*paths)
    lines = (paths[1] / "generations.jsonl").read_text().splitlines()
    row = json.loads(lines[0]); row["samples"].pop(); lines[0] = json.dumps(row)
    (paths[1] / "generations.jsonl").write_text("\n".join(lines))
    with pytest.raises(ValueError, match="exactly ten"):
        validate_inference_pair(*paths)


def test_coverage_uses_compact_submission_ranks_after_deduplication(tmp_path, monkeypatch):
    from src.notebook_workflows import local_expansion
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda *a, **k: {("q", "gold"): True})
    files = {"config.json": {"model_name": "model", "response_mode": "equivalent"},
             "status.json": {"status": "complete"}}
    for filename, value in files.items():
        (tmp_path / filename).write_text(json.dumps(value))
    for name, rows in {
        "examples": [{"question_id": "q", "question": "?", "gold_aliases": ["gold"], "snippets": []}],
        "generations": [{"question_id": "q", "parse_error": None}],
        "candidates": [{"question_id": "q", "answer": "wrong", "position": 1},
                       {"question_id": "q", "answer": "gold", "position": 6}],
    }.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, summary, rows = local_expansion.analyze_local_expansion(tmp_path, jar_path=tmp_path / "unused.jar")
    assert summary["coverage_at5"] == 1
    assert rows[0]["raw_positions"] == [1, 6]


def test_historical_identity_rejects_changed_files_and_fitting_leakage(tmp_path, monkeypatch):
    runner = load_script("run_original_qwen25_inference")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: None)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/expansion_dev160.jsonl").write_text("".join(
        json.dumps({"question_id": f"dev-{i}"}) + "\n" for i in range(160)))
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "base"}))
    (adapter / "training_complete.json").write_text(json.dumps({"status": "completed"}))
    identity = {"model_size": "3b", "historical_run": "run", "train_question_ids": ["fit"],
                "files": {name: hashlib.sha256((adapter / name).read_bytes()).hexdigest()
                          for name in ("adapter_config.json", "training_complete.json")}}

    def pin():
        (adapter / "identity.json").write_text(json.dumps(identity))
        (tmp_path / "configs/original_qwen25_adapters.json").write_text(json.dumps({"3b": {
            "historical_run": "run", "base_model": "base",
            "identity_sha256": hashlib.sha256((adapter / "identity.json").read_bytes()).hexdigest()}}))

    pin()
    runner.validate_adapter("3b", adapter)
    identity["train_question_ids"] = ["dev-0"]
    pin()
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_adapter("3b", adapter)
    (adapter / "adapter_config.json").write_text("changed")
    with pytest.raises(ValueError, match="file differs"):
        runner.validate_adapter("3b", adapter)


def test_smoke_guard_rejects_completed_zero_answer_arm(tmp_path):
    runner = load_script("run_original_qwen25_inference")
    (tmp_path / "generations.jsonl").write_text(json.dumps({"question_id": "q", "samples": [
        {"raw_response": "Answer: [BE] alpha [EE]", "parse_error": "Unexpected prefix"}]}) + "\n")
    (tmp_path / "candidates.jsonl").write_text("")
    with pytest.raises(ValueError, match="zero parseable"):
        runner.validate_smoke_outputs({"sampling": tmp_path})
    (tmp_path / "candidates.jsonl").write_text(json.dumps({"question_id": "q", "answer": "alpha"}) + "\n")
    runner.validate_smoke_outputs({"sampling": tmp_path})


def test_reparsing_preserves_raw_samples_costs_and_previous_errors():
    original = {"question_id": "q", "parse_error": "All ten samples failed", "request_count": 10,
                "samples": [{"draw": 1, "raw_response": "Answer: [BE] alpha [EE]", "answer": None,
                             "parse_error": "Unexpected prefix", "seed": 123, "output_tokens": 10},
                            {"draw": 2, "raw_response": "Reasoning [BE] beta [EE]", "answer": None,
                             "parse_error": "Unexpected text", "seed": 456, "output_tokens": 20}]}
    generations, candidates = reparse_samples([original])
    assert generations[0]["samples"][0]["answer"] == "alpha"
    assert generations[0]["samples"][0]["previous_parse_error"] == "Unexpected prefix"
    assert generations[0]["samples"][0]["raw_response"] == original["samples"][0]["raw_response"]
    assert generations[0]["samples"][0]["seed"] == 123
    assert generations[0]["samples"][0]["output_tokens"] == 10
    assert generations[0]["request_count"] == 10
    assert generations[0]["samples"][1]["parse_error"]
    assert original["samples"][0]["answer"] is None
    assert generations[0]["parse_error"] is None
    assert [c["answer"] for c in candidates] == ["alpha"]


def test_recovery_creates_auditable_sibling_without_changing_original(tmp_path, monkeypatch):
    from scripts import reparse_original_qwen25_sampling as recovery
    original = tmp_path / "experiment"
    sampling = original / "sampling10"
    expansion = original / "expansion"
    sampling.mkdir(parents=True)
    expansion.mkdir()
    manifest = {"status": "complete", "smoke_test": False, "expected_questions": 160,
                "runs": {"original_sft_sampling10": "sampling10", "original_sft_expansion": "expansion"}}
    (original / "manifest.json").write_text(json.dumps(manifest))
    (sampling / "status.json").write_text(json.dumps({"status": "complete"}))
    (sampling / "config.json").write_text(json.dumps({"temperature": 0.8, "num_generations": 10}))
    examples = [{"question_id": str(i)} for i in range(160)]
    rows = [{"question_id": str(i), "request_count": 10, "samples": [
        {"draw": draw, "seed": draw, "raw_response": "Answer: [BE]15[EE]", "answer": None,
         "parse_error": "Unexpected text outside the single answer"} for draw in range(1, 11)]}
        for i in range(160)]
    for name, values in (("examples", examples), ("generations", rows)):
        (sampling / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    before = {path: path.read_bytes() for path in original.rglob("*") if path.is_file()}
    monkeypatch.setattr(sys, "argv", ["recover", str(original)])
    recovery.main()
    output = original.with_name("experiment-sampling-parser-v2")
    revised = json.loads((output / "manifest.json").read_text())
    assert (output / revised["runs"]["original_sft_expansion"]).resolve() == expansion
    assert revised["generation_unchanged"] is True
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert len((output / "sampling10/candidates.jsonl").read_text().splitlines()) == 160
    config = json.loads((output / "sampling10/config.json").read_text())
    assert config["source_generations_sha256"] == hashlib.sha256(before[sampling / "generations.jsonl"]).hexdigest()
    with pytest.raises(FileExistsError):
        recovery.main()
