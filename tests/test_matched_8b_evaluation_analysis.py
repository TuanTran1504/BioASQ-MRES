import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_matched_8b_evaluation import CONDITIONS, score_experiment, validate_experiment


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path):
    provenance = {}
    for formulation in ("original", "expansion"):
        prompt = formulation + " trained prompt"
        (tmp_path / (formulation + "_prompt.txt")).write_text(prompt)
        provenance[formulation] = {"adapter": formulation + " adapter", "system_prompt": prompt,
                                   "model_loader": "fast_language_model",
                                   "chat_template_kwargs": {"enable_thinking": False},
                                   "base_snapshot": "shared", "base_revision": "pinned", "matrix_sha256": "matrix"}
    manifest = {"status": "complete", "smoke_test": False, "expected_questions": 160,
                "model_key": "qwen3", "evaluation": "development-only", "selection": "first five",
                "input_sha256": "same input", "provenance": provenance, "runs": {}}
    examples = [{"question_id": str(i), "question": "How many?", "gold_aliases": ["15"],
                 "snippets": [{"snippet_id": "s", "text": "There are 15."}]} for i in range(160)]
    for name, hit_count in zip(CONDITIONS, (80, 160, 120)):
        path = tmp_path / name
        path.mkdir()
        formulation = "expansion" if name == "expansion_greedy" else "original"
        source = provenance[formulation]
        count = 10 if name == "original_sampling10" else 1
        config = {"model_name": source["adapter"], "model_loader": source["model_loader"],
                  "chat_template_kwargs": source["chat_template_kwargs"],
                  "prompt": "old host path", "prompt_sha256": hashlib.sha256(source["system_prompt"].encode()).hexdigest(),
                  "input_sha256": "same input", "max_seq_length": 6144, "max_new_tokens": 512,
                  "require_all_snippets": True, "mark_snippets": False, "seed": 3407,
                  "temperature": 0.8 if count == 10 else 0, "num_generations": count, "top_p": 0.95,
                  "question_count": 160, "gold_blind_generation": True,
                  "response_mode": "equivalent" if formulation == "expansion" else
                                   "single_answer_sampling" if count == 10 else "single_answer_greedy"}
        write(path / "config.json", config)
        write(path / "status.json", {"status": "complete"})
        jsonl(path / "examples.jsonl", examples)
        rows = []
        candidates = []
        for i in range(160):
            # Greedy parse failures stay in the denominator; other arms recover.
            failure = name == "original_greedy" and i >= hit_count
            answer = "15" if i < hit_count else "wrong"
            row = {"question_id": str(i), "question": "How many?", "parse_error": "bad format" if failure else None,
                   "snippets_truncated": False, "included_snippets": 1, "total_snippets": 1,
                   "request_count": count, "input_tokens": count * 5, "output_tokens": count * 3,
                   "generation_seconds": count * 0.1}
            if formulation == "original":
                row["samples"] = [{"draw": d, "seed": int.from_bytes(hashlib.sha256(f"3407:{i}:{d}".encode()).digest()[:4], "big"),
                                   "answer": None if failure else answer, "parse_error": row["parse_error"]}
                                  for d in range(1, count + 1)]
            rows.append(row)
            if not failure:
                candidates.append({"question_id": str(i), "position": 1, "answer": answer, "relation_type": "original"})
        jsonl(path / "generations.jsonl", rows)
        jsonl(path / "candidates.jsonl", candidates)
        manifest["runs"][name] = name
    write(tmp_path / "manifest.json", manifest)
    return tmp_path


@pytest.mark.parametrize("error", ["evidence", "nine_draws", "seed", "prompt", "backbone", "truncation"])
def test_analysis_rejects_unmatched_inputs_or_incomplete_draws(experiment, error):
    path = experiment / "original_sampling10"
    if error == "evidence":
        rows = [json.loads(line) for line in (path / "examples.jsonl").read_text().splitlines()]
        rows[0]["snippets"][0]["text"] = "changed evidence"
        jsonl(path / "examples.jsonl", rows)
    elif error == "prompt":
        (experiment / "original_prompt.txt").write_text("changed system prompt")
    elif error == "backbone":
        manifest = json.loads((experiment / "manifest.json").read_text())
        manifest["provenance"]["expansion"]["base_revision"] = "other revision"
        write(experiment / "manifest.json", manifest)
    else:
        rows = [json.loads(line) for line in (path / "generations.jsonl").read_text().splitlines()]
        if error == "nine_draws": rows[0]["samples"].pop()
        if error == "seed": rows[0]["samples"][0]["seed"] = 0
        if error == "truncation": rows[0]["snippets_truncated"] = True
        jsonl(path / "generations.jsonl", rows)
    with pytest.raises(ValueError):
        validate_experiment(experiment)


def test_all_three_paired_contrasts_include_failures_and_request_costs(experiment, monkeypatch):
    from src.notebook_workflows import local_expansion
    # Isolate Java matching; exercise the actual candidate analysis and bootstrap.
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
                        {(row["question_id"], row["answer"]): row["answer"] == "15" for row in candidates})
    result = score_experiment(experiment, Path("unused.jar"))
    assert [result["conditions"][name]["metrics"]["mrr_at5"] for name in CONDITIONS] == [0.5, 1.0, 0.75]
    assert [result["conditions"][name]["efficiency"]["request_count"] for name in CONDITIONS] == [160, 1600, 160]
    contrasts = result["paired_contrasts"]
    assert len(contrasts) == 3
    assert contrasts["expansion_greedy_minus_original_greedy"]["mrr_at5"]["second_minus_first"] == 0.25
    assert contrasts["expansion_greedy_minus_original_sampling10"]["mrr_at5"]["second_minus_first"] == -0.25
    assert contrasts["original_sampling10_minus_original_greedy"]["mrr_at5"]["paired_bootstrap_95ci"][0] > 0
    assert result["conditions"]["original_greedy"]["candidate_diagnostics"]["parse_success_rate"] == 0.5
    assert (experiment / "comparison_summary.json").is_file()


@pytest.fixture
def expansion_sampling(experiment):
    directory = experiment / "extra"
    directory.mkdir()
    baseline_manifest = json.loads((experiment / "manifest.json").read_text())
    manifest = {**baseline_manifest, "runs": {"expansion_sampling10": "sampled"},
                "baseline_experiment": str(experiment),
                "baseline_manifest_sha256": hashlib.sha256((experiment / "manifest.json").read_bytes()).hexdigest()}
    # Reproduce a new sampler reusing manifests written before execution_mode.
    manifest["provenance"] = {name: {**source, "execution_mode": "default"}
                              for name, source in baseline_manifest["provenance"].items()}
    path = directory / "sampled"
    path.mkdir()
    (directory / "expansion_prompt.txt").write_bytes((experiment / "expansion_prompt.txt").read_bytes())
    config = json.loads((experiment / "expansion_greedy/config.json").read_text())
    config.update(response_mode="equivalent_sampling", num_generations=10, temperature=0.8, top_p=0.95, top_k=0)
    write(path / "config.json", config)
    write(path / "status.json", {"status": "complete"})
    (path / "examples.jsonl").write_bytes((experiment / "expansion_greedy/examples.jsonl").read_bytes())
    generations, candidates = [], []
    for i in range(160):
        samples = []
        for d in range(1, 11):
            # An accepted answer appears after ten unique wrong candidates.
            answers = [{"answer": "15" if d == 2 else f"wrong-{j}", "relation_type": "original" if j == 1 else "synonym",
                        "raw_position": j} for j in range(1, 11 if d == 1 else 2)]
            samples.append({"draw": d, "seed": int.from_bytes(hashlib.sha256(f"3407:{i}:{d}".encode()).digest()[:4], "big"),
                            "answers": answers, "parse_error": None})
        generations.append({"question_id": str(i), "samples": samples, "parse_error": None,
                            "snippets_truncated": False, "included_snippets": 1, "total_snippets": 1,
                            "request_count": 10, "input_tokens": 50, "output_tokens": 100, "generation_seconds": 1})
        for j, answer in enumerate(samples[0]["answers"] + samples[1]["answers"], 1):
            candidates.append({"question_id": str(i), **answer, "position": j, "draw": 1 if j <= 10 else 2,
                               "within_draw_position": answer["raw_position"]})
    jsonl(path / "generations.jsonl", generations)
    jsonl(path / "candidates.jsonl", candidates)
    write(directory / "manifest.json", manifest)
    return directory, experiment


@pytest.mark.parametrize("failure", [None, "order", "prompt", "seed", "nine_draws", "baseline"])
def test_expansion_sampling_checks_pool_protocol(expansion_sampling, failure):
    from scripts.analyze_expansion_sampling_8b import validate_sampling_comparison
    directory, baseline = expansion_sampling
    path = directory / "sampled"
    if failure == "order":
        rows = [json.loads(line) for line in (path / "candidates.jsonl").read_text().splitlines()]
        rows[0]["position"] = 2
        jsonl(path / "candidates.jsonl", rows)
    elif failure == "prompt":
        (directory / "expansion_prompt.txt").write_text("changed")
    elif failure == "baseline":
        (baseline / "manifest.json").write_text((baseline / "manifest.json").read_text() + '\n')
    elif failure:
        rows = [json.loads(line) for line in (path / "generations.jsonl").read_text().splitlines()]
        if failure == "seed": rows[0]["samples"][0]["seed"] = 0
        if failure == "nine_draws": rows[0]["samples"].pop()
        jsonl(path / "generations.jsonl", rows)
    if failure:
        with pytest.raises(ValueError):
            validate_sampling_comparison(directory, baseline)
    else:
        _, runs = validate_sampling_comparison(directory, baseline)
        assert len(runs) == 4


def test_expansion_sampling_separates_full_pool_from_top_ten(expansion_sampling, monkeypatch):
    from scripts.analyze_expansion_sampling_8b import validate_sampling_comparison
    from scripts.analyze_matched_8b_evaluation import score_validated_runs
    from src.notebook_workflows import local_expansion
    directory, baseline = expansion_sampling
    manifest, runs = validate_sampling_comparison(directory, baseline)
    monkeypatch.setattr(local_expansion, "official_candidate_matches", lambda examples, candidates, *a, **k:
        {(r["question_id"], r["answer"]): r["answer"] == "15" for r in candidates})
    result = score_validated_runs(directory, manifest, runs, Path("unused.jar"))
    arm = result["conditions"]["expansion_sampling10"]
    assert arm["metrics"]["coverage_at10"] == arm["metrics"]["mrr_at5"] == 0
    assert arm["metrics"]["coverage_full_pool"] == 1
    assert arm["efficiency"]["request_count"] == 1600
    assert arm["sampling_diagnostics"]["coverage_within_first_1_draws"] == 0
    assert arm["sampling_diagnostics"]["coverage_within_first_5_draws"] == 1
    assert len(result["paired_contrasts"]) == 6
