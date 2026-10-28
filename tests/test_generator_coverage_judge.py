"""Blinding, evidence validation, source identity and provisional categories."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from scripts import judge_generator_coverage_dpo as judge
from scripts import prepare_answer_selector_data as data
from scripts import audit_generator_coverage_dpo as audit
from test_generator_coverage_dpo_audit import fixture


def decision(correctness="correct", support="supported", equivalence="equivalent", relation="valid", confidence="high"):
    return {"index": 0, "correctness": correctness, "support": support, "equivalence": equivalence,
        "relation_valid": relation, "confidence": confidence, "evidence": [{"snippet_id": "s", "quote": "Marker HER-2."}],
        "rationale": "Evidence establishes the answer.", "validation_issues": [], "request_sha256": "request"}


def make_source(tmp_path, monkeypatch):
    selector, _ = fixture(tmp_path, monkeypatch)
    import scripts.train_sft_ensemble_reranker as rr
    def snapshot(repo, destination):
        for name in ("audit_generator_coverage_dpo.py", "prepare_answer_selector_data.py"):
            path = destination / "scripts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit.ROOT / "scripts" / name, path)
        return {}
    monkeypatch.setattr(rr, "snapshot_code", snapshot)
    sub = tmp_path / "sub"
    audit.prepare(audit.ROOT, selector, sub)
    source = tmp_path / "source_export"
    audit.run(source, submission=sub)
    return source, selector


def test_source_validation_and_blind_plan(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    pairs, rows = judge.load_source(source, selector)
    jobs, units = judge.prepare_jobs(rows)
    assert pairs and jobs and units
    before = copy.deepcopy(rows)
    for r in rows:
        r["model_key"] = "HIDDEN_MODEL"
        r["response_id"] = "HIDDEN_ID"
        for c in r["candidates"]: c["official_accepted"] = not c["official_accepted"]
    jobs2, units2 = judge.prepare_jobs(rows)
    assert jobs2 == jobs and units2 == units
    for job in jobs:
        payload = job["payload"]
        public = json.loads(payload["messages"][1]["content"])
        assert set(public) == {"question", "snippets", "candidates"}
        assert "HIDDEN_MODEL" not in json.dumps(payload)
        assert "official_accepted" not in json.dumps(public) and "gold_aliases" not in json.dumps(public)
        assert payload["model"] == "gpt-4.1-mini-2025-04-14" and not payload["store"]
        assert payload["messages"][0]["content"] == judge.SYSTEM
        assert job["request_sha256"] == judge.shared.sha(payload)
        assert job["maximum_cost_reserve_usd"] > 0
    # Flipping an official label cannot survive the frozen source validation.
    p = source / "review_responses.jsonl"
    data.jsonl(p, before[:-1])
    with pytest.raises(ValueError, match="changed"):
        judge.load_source(source, selector)


def test_categories_separate_semantic_from_benchmark_and_bad_preferred(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    pairs, rows = judge.load_source(source, selector)
    p = pairs[0]
    decisions = {judge.unit_id(r, c): decision() for r in rows for c in r["candidates"]}
    annotated = judge.annotate(rows, decisions)
    assert judge.classify_pair(p, annotated)[0] == "benchmark_form_preference"
    # The rejected answer may be a real mistake, supported by a diagnostic quote.
    rejected = next(r for r in rows if r["response_id"] == p["rejected_id"])
    key = judge.unit_id(rejected, rejected["candidates"][0])
    decisions[key] = decision("incorrect", "contradicted")
    # All equal question/answer contexts must agree, even across originals.
    for r in rows:
        if r["question_id"] == rejected["question_id"]:
            for c in r["candidates"]:
                if c["answer"] == rejected["candidates"][0]["answer"]:
                    decisions[judge.unit_id(r, c)] = decision("incorrect", "contradicted")
    annotated = judge.annotate(rows, decisions)
    assert judge.classify_pair(p, annotated)[0] == "semantic_coverage_repair_candidate"
    chosen = next(r for r in rows if r["response_id"] == p["chosen_id"])
    decisions[judge.unit_id(chosen, chosen["candidates"][0])] = decision("correct", "unsupported")
    assert judge.classify_pair(p, judge.annotate(rows, decisions))[0] in ("preferred_not_clean", "unresolved")
    del decisions[judge.unit_id(chosen, chosen["candidates"][0])]
    assert judge.classify_pair(p, judge.annotate(rows, decisions))[0] == "pending"


def test_assistant_hold_and_uncertainty_never_autoapprove(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    pairs, rows = judge.load_source(source, selector)
    p = pairs[0]
    decisions = {judge.unit_id(r, c): decision(confidence="medium") for r in rows for c in r["candidates"]}
    annotated = judge.annotate(rows, decisions)
    assert judge.classify_pair(p, annotated)[0] == "unresolved"
    held = {**p, "question_id": "534e364e288f4dae47000001"}
    assert judge.classify_pair(held, annotated)[0] == "assistant_evidence_hold"
    out = tmp_path / "review"
    out.mkdir()
    result = judge.export_review(out, pairs, rows, decisions, complete=True)
    assert not result["training_ready"] and not result["full_semantic_pilot_approved"]
    assert not data.records(out / "semantic_coverage_repair_candidate.jsonl")
    assert all(not p["training_ready"] for p in data.records(out / "pair_review.jsonl"))


def test_plan_resume_and_caps_do_not_make_paid_calls(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    out = tmp_path / "judge"
    first = judge.run(source, selector, out, execute=False, maximum=10)
    assert first["status"] == "prepared" and not first["training"]
    assert not (out / "spend-ledger.json").exists()
    assert judge.run(source, selector, out, execute=False, maximum=10)["plan_sha256"] == first["plan_sha256"]
    state = data.read(out / "manifest.json")
    state.update(status="partial", blocking_reason="credit_balance_exhausted", judged_units=0, charged_or_reserved_usd=0.01)
    data.write(out / "manifest.json", state)
    after = judge.run(source, selector, out, execute=False, maximum=10)
    assert after["blocking_reason"] == "credit_balance_exhausted" and after["charged_or_reserved_usd"] == 0.01
    with pytest.raises(ValueError, match="Cannot resume"):
        judge.run(source, selector, out, execute=False, maximum=9)
    for cap in (-1, 0, 11):
        with pytest.raises(ValueError): judge.run(source, selector, out, maximum=cap)
    assert judge.run(source, selector, out, execute=False, maximum=10, workers=12)["plan_sha256"] == first["plan_sha256"]
    with pytest.raises(ValueError): judge.run(source, selector, out, maximum=10, workers=13)


def test_evidence_quote_validation_catches_bad_judgments(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    _, rows = judge.load_source(source, selector)
    jobs, _ = judge.prepare_jobs(rows)
    job = jobs[0]
    decisions = [{k: v for k, v in decision().items() if k not in ("validation_issues", "request_sha256")} for _ in job["unit_ids"]]
    for i, d in enumerate(decisions):
        d["index"] = i
        d["evidence"][0]["quote"] = "fabricated quote not in evidence"
    checked = judge.shared.validate_decisions({"decisions": decisions}, job)
    assert all(v["validation_issues"] for v in checked.values())
    with pytest.raises(ValueError): judge.shared.validate_decisions({"decisions": decisions[:-1]}, job)


def test_credit_exhaustion_stops_requests_without_exposing_credentials(tmp_path, monkeypatch):
    import requests
    source, selector = make_source(tmp_path, monkeypatch)
    _, rows = judge.load_source(source, selector)
    jobs, _ = judge.prepare_jobs(rows)
    directory = tmp_path / "budget"
    directory.mkdir()
    calls = []
    class Reply:
        status_code = 429
        def json(self): return {"error": {"code": "credit_balance_exhausted"}}
    def post(*args, **kwargs):
        calls.append(1)
        return Reply()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(judge.shared, "read_api_key", lambda p: "test-secret-never-print")
    client = judge.shared.BudgetClient(directory, 10, tmp_path / "unused-key", input_rate=judge.INPUT_RATE,
                                     output_rate=judge.OUTPUT_RATE, validator=judge.shared.validate_decisions)
    assert client.call(jobs[0]) == {}
    assert client.blocked_reason == "credit_balance_exhausted"
    assert client.call(jobs[-1]) == {} and len(calls) == 1
    ledger = (directory / "spend-ledger.json").read_text()
    assert "test-secret-never-print" not in ledger and client.spent <= 10
    # Failed requests retain a conservative budget reservation, not an assertion
    # that this amount was actually billed. A resumed run must preserve it.
    resumed = judge.shared.BudgetClient(directory, 10, tmp_path / "unused-key", input_rate=judge.INPUT_RATE, output_rate=judge.OUTPUT_RATE)
    assert resumed.spent == client.spent


def test_conservative_reserve_enforces_cap_before_call(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    _, rows = judge.load_source(source, selector)
    jobs, _ = judge.prepare_jobs(rows)
    directory = tmp_path / "small-budget"
    directory.mkdir()
    monkeypatch.setattr(judge.shared, "read_api_key", lambda p: pytest.fail("Budget must be checked before reading credentials"))
    client = judge.shared.BudgetClient(directory, 0.0000001, tmp_path / "unused-key", input_rate=judge.INPUT_RATE, output_rate=judge.OUTPUT_RATE)
    assert client.call(jobs[0]) == {} and client.spent == 0 and client.reserved == 0


def test_full_cached_review_resume_makes_no_requests_and_pins_outputs(tmp_path, monkeypatch):
    source, selector = make_source(tmp_path, monkeypatch)
    out = tmp_path / "cached-review"
    judge.run(source, selector, out, execute=False)
    jobs = data.read(out / "plan.json")
    for job in jobs:
        decisions = [{k: v for k, v in decision().items() if k not in ("validation_issues", "request_sha256")} for _ in job["unit_ids"]]
        for i, d in enumerate(decisions): d["index"] = i
        judge.shared.save(out / "cache" / (job["request_sha256"] + ".json"), {"payload": job["payload"], "decisions": {"decisions": decisions}})
    monkeypatch.setattr(judge.shared, "read_api_key", lambda p: pytest.fail("Completed caches must not access credentials or network"))
    result = judge.run(source, selector, out, execute=True)
    assert result["status"] == "complete" and result["charged_or_reserved_usd"] == 0
    assert not result["training_ready"] and "pair_review.jsonl" in result["output_sha256"]
    data.verify_files(out, result["output_sha256"])
