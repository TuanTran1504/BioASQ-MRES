"""Pool absence means no accepted candidate anywhere, regardless of rank."""
import copy
from pathlib import Path
import tarfile

import pytest

from scripts import export_sft_ensemble_pool_absences as audit


def candidate(q, answer, label):
    return {"question_id": q, "answer": answer, "label": label,
            "sources": ["llama31"], "source_ranks": {"llama31": 1},
            "sample_frequencies": {"llama31": 0.1}, "relation_types": ["original"]}


@pytest.fixture
def inputs():
    examples = {f"q{i:02}": {"question_id": f"q{i:02}", "question": f"Question {i}?",
                 "snippets": [{"snippet_id": "1", "text": "Supplied evidence, not a semantic approval."}]} for i in range(95)}
    gold = {q: {"question_id": q, "batch": "B1" if i < 26 else "B2", "gold_aliases": ["gold", "accepted alias"]}
            for i, q in enumerate(examples)}
    pools = {f"{arm}_{mode}": [candidate(q, "gold", 1) for q in examples]
             for arm in audit.source.ARMS for mode in audit.source.MODES}
    target = [r for r in pools[audit.TARGET] if int(r["question_id"][1:]) >= 20]
    target.extend(candidate(q, "semantically plausible unaccepted wording", 0) for q in list(examples)[1:20])
    # q00 has a completely empty parsed pool; q20 has gold below the top five.
    target.extend(candidate("q20", f"frequent distractor {i}", 0) for i in range(8))
    for r in target:
        if r["question_id"] == "q20" and not r["label"]:
            r["sample_frequencies"] = {"llama31": 1.0}
    pools[audit.TARGET] = target
    return pools, examples, gold


def test_all_pool_absences_including_empty_and_no_semantic_assumptions(inputs):
    pools, examples, gold = inputs
    old = copy.deepcopy(pools)
    cases, summary = audit.collect_cases(pools, examples, gold)
    assert len(cases) == 20 and summary["covered_in_full_pool"] == 75
    assert [c["question_id"] for c in cases] == list(examples)[:20]
    assert "q20" not in summary["question_ids"]
    assert cases[0]["conditions"][audit.TARGET]["pool_size"] == 0
    assert cases[1]["conditions"][audit.TARGET]["candidates"][0]["answer"].startswith("semantically plausible")
    assert all(c["review"]["status"] == "unreviewed" and
               c["review"]["supported_but_officially_unaccepted_answer_present"] is None for c in cases)
    assert summary["among_target_absences_covered_by"]["original_sampling10"] == 20
    assert not summary["training_ready"] and pools == old


@pytest.mark.parametrize("corruption", ["count", "label", "unknown", "duplicate", "gold"])
def test_changed_counts_membership_or_labels_fail(inputs, corruption):
    pools, examples, gold = inputs
    if corruption == "count":
        pools[audit.TARGET].append(candidate("q00", "gold", 1))
    elif corruption == "label":
        pools[audit.TARGET][0]["label"] = "0"
    elif corruption == "unknown":
        pools[audit.TARGET].append(candidate("unknown", "x", 0))
    elif corruption == "duplicate":
        pools[audit.TARGET].append(copy.deepcopy(pools[audit.TARGET][0]))
    else:
        gold.pop("q00")
    with pytest.raises(ValueError):
        audit.collect_cases(pools, examples, gold)


@pytest.fixture
def export_inputs(inputs, monkeypatch, tmp_path):
    pools, examples, gold = inputs
    comparison = tmp_path / "comparison"; comparison.mkdir()
    sources = {}
    for model in audit.source.MODELS:
        for mode in audit.source.MODES:
            directory = tmp_path / f"{model}-{mode}"; directory.mkdir()
            arms = {}
            for arm in audit.source.ARMS:
                run = directory / arm; run.mkdir()
                audit.source.write_jsonl(run / "generations.jsonl", [
                    {"question_id": q, "samples": [{"raw_response": "saved natural generation"}]}
                    for q in examples])
                arms[arm] = {"generations.jsonl": audit.source.digest(run / "generations.jsonl")}
            audit.source.write(directory / "manifest.json", {
                "runs": {a: a for a in audit.source.ARMS}, "arm_hashes": arms})
            sources[f"{model}_{mode}"] = {"directory": str(directory),
                "manifest_sha256": audit.source.digest(directory / "manifest.json")}
    audit.source.write(comparison / "manifest.json", {"sources": sources})
    hashes = {"manifest.json": audit.source.digest(comparison / "manifest.json")}
    monkeypatch.setattr(audit.source, "load_comparison", lambda _: ({}, pools, examples, gold, hashes))
    return comparison, sources


def test_export_contains_all_twenty_complete_questions_and_raw_traces(export_inputs, tmp_path):
    comparison, _ = export_inputs
    output, archive = tmp_path / "out", tmp_path / "absences.tar.gz"
    audit.export(comparison, output, archive)
    cases = audit.source.records(output / "cases.jsonl")
    raw = audit.source.records(output / "source_generations.jsonl")
    assert len(cases) == 20 and len(raw) == 240
    assert {r["question_id"] for r in raw} == {r["question_id"] for r in cases}
    assert all(r["samples"][0]["raw_response"] == "saved natural generation" for r in raw)
    state = audit.source.read(output / "manifest.json")
    assert state["status"] == "complete" and not state["new_official_scoring"] and not state["paid_judging"]
    for name, digest in state["output_sha256"].items():
        assert audit.source.digest(output / name) == digest
    with tarfile.open(archive) as tar:
        assert {m.name for m in tar.getmembers()} == {"manifest.json", *state["output_sha256"]}
    with pytest.raises(ValueError):
        audit.export(comparison, output, archive)


def test_changed_raw_source_is_rejected_before_publication(export_inputs, tmp_path):
    comparison, sources = export_inputs
    path = Path(next(iter(sources.values()))["directory"]) / "max5/generations.jsonl"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="generation changed"):
        audit.export(comparison, tmp_path / "out")
    assert not (tmp_path / "out").exists()
