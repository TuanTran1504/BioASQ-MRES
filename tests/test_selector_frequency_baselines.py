"""Untrained baselines must preserve exact pools, provenance and failed draws."""
import copy
import json

import pytest

from scripts import prepare_answer_selector_data as data
from scripts import score_selector_frequency_baselines as frequency


def candidate(identity, ranks, counts):
    return {"id": identity, "answer": identity, "source_ranks": ranks,
            "sources": list(ranks), "draw_support": counts}


def test_generator_vote_rules_and_exact_round_robin():
    a = candidate("A", {"llama31": 1}, {"llama31": 4})
    b = candidate("B", {"llama31": 2, "qwen3": 1, "ministral3": 1}, {m: 1 for m in data.MODELS})
    c = candidate("C", {"qwen3": 2, "ministral3": 2}, {"qwen3": 4, "ministral3": 4})
    rows = [c, a, b]
    ids = lambda method: [r["id"] for r in frequency.order(rows, method)]
    assert ids("round_robin") == ["A", "B", "C"]
    assert ids("model_vote") == ["B", "C", "A"]
    assert ids("rank_vote") == ["B", "A", "C"]
    assert ids("sample_frequency") == ["C", "A", "B"]
    a["answer"], b["answer"], c["answer"] = "HER-2", "her-2", " HER-2 "
    for method in frequency.METHODS:
        assert len(frequency.order(rows, method)) == 3
        before = frequency.order(copy.deepcopy(rows), method)
        for r in rows: r["official_accepted"] = not r.get("official_accepted", False)
        assert [r["id"] for r in before] == [r["id"] for r in frequency.order(rows[::-1], method)]
    with pytest.raises(ValueError): frequency.order(rows, "unknown")


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "COUNTS", {"train": 1, "validation": 2})
    examples = {s: [{"question_id": q, "question": "Which answer?", "snippets": [{"id": "s", "text": "HER-2"}],
                     "gold_aliases": ["HER-2"]} for q in qs]
                for s, qs in (("train", ("t",)), ("validation", ("v0", "v1")))}
    raw = []
    for split, panel in examples.items():
        for e in panel:
            for model in data.MODELS:
                for draw in range(5):
                    aa = ["HER-2", "her-2", "HER2", "decoy"] if model == "llama31" else ["decoy", "HER2"]
                    response = json.dumps({"answers": [{"answer": a, "relation_type": "original" if i == 0 else "synonym"} for i, a in enumerate(aa)]})
                    if e["question_id"] == "v1" and draw > 0: response = "failed request"
                    raw.append({"split": split, "question_id": e["question_id"], "model_key": model, "draw": draw,
                        "response_id": f"{e['question_id']}-{model}-{draw}", "raw_response": response, "output_tokens": 40})
    pools, _, response_audit = data.pool_responses(raw, examples)
    for modes in pools.values():
        for rows in modes.values():
            for r in rows: r["official_accepted"] = r["answer"] == "HER-2"
    directory = tmp_path / "audit"
    data.construct_dataset(pools, examples, directory)
    data.jsonl(directory / "response_audit.jsonl", response_audit)
    data.write(directory / "heldout_ids.json", {"dev160": ["dev"], "test95": ["test"]})
    data.write(directory / "source_provenance.json", {})
    data.write(directory / "manifest.json", {"status": "complete", "schema": "max5-answer-selector-audit-v1",
        "question_counts": data.COUNTS, "output_sha256": {p.name: data.digest(p) for p in directory.iterdir() if p.is_file()}})
    return directory


def test_actual_support_failures_and_scoring_export(tmp_path, monkeypatch):
    directory = fixture(tmp_path, monkeypatch)
    output = tmp_path / "scored"
    result = frequency.run(directory, output)
    for method in frequency.METHODS:
        assert result["conditions"][method]["max5_sampling4"]["coverage_full_pool"] == 0.5
        predictions = data.records(output / f"{method}_predictions.jsonl")
        assert len(predictions) == 4
        assert next(p for p in predictions if p["question_id"] == "v1" and p["pool"] == "max5_sampling4")["selected_source_ids"] == []
    state = data.read(output / "manifest.json")
    assert state["status"] == "complete" and not state["training"]
    data.verify_files(output, state["artifacts_sha256"])
    with pytest.raises(FileExistsError): frequency.run(directory, output)


@pytest.mark.parametrize("mutation", ["frequency", "draw_count", "source_rank", "missing_request", "duplicate_occurrence", "missing_surface"])
def test_rejects_changed_support_or_pool_membership(tmp_path, monkeypatch, mutation):
    directory = fixture(tmp_path, monkeypatch)
    view = next(r for r in data.records(directory / "validation_supervision.jsonl") if r["pool"] == "max5_sampling4" and r["question_id"] == "v0")
    requests = data.records(directory / "response_audit.jsonl")
    if mutation == "frequency": view["candidates"][0]["sample_frequencies"]["llama31"] = 0.1
    elif mutation == "draw_count": view["candidates"][0]["draw_support"]["llama31"] = 1
    elif mutation == "source_rank": view["candidates"][0]["source_ranks"]["llama31"] = 99
    elif mutation == "missing_request": requests = [r for r in requests if not (r["question_id"] == "v0" and r["model_key"] == "llama31" and r["draw"] == 1)]
    elif mutation == "duplicate_occurrence": view["candidates"][0]["occurrences"].append(view["candidates"][0]["occurrences"][0])
    elif mutation == "missing_surface": view["candidates"].pop()
    with pytest.raises(ValueError): frequency.validate_support(view, requests)
