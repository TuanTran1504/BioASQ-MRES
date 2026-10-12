import copy
import json
from pathlib import Path
import sys

import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_single_answer_dpo import response
import scripts.judge_single_answer_dpo as judge
import scripts.judge_expansion_dpo_annotations as old_judge


def decision(**changes):
    return {"index": 0, "correctness": "correct", "support": "supported", "confidence": "high",
        "evidence": [{"snippet_id": "s1", "quote": "TNF-alpha is the implicated cytokine."}],
        "rationale": "The stated cytokine answers the question.", **changes}


def jobs(rows):
    return judge.prepare(rows, {"q": ["TNF-alpha"]})


def test_mini_profile_and_blinded_deduplicated_payload():
    a, b = response(), response(model="qwen3", draw=1)
    plan, units = jobs([a, b])
    assert len(units) == 1 and len(plan) == 1
    payload = plan[0]["payload"]
    assert payload["model"] == "gpt-4.1-mini-2025-04-14" and payload["store"] is False
    public = payload["messages"][1]["content"]
    assert all(x not in public for x in ("model_key", "official_accepted", "chosen", "C3"))
    assert judge.INPUT_RATE == 0.4 / 1_000_000


def test_c2_is_correct_unaccepted_and_uncertainty_abstains():
    row = response("full name", "C2", 1)
    row["class"] = None
    plan, _ = jobs([row])
    decisions = judge.validate_decisions({"decisions": [decision()]}, plan[0])
    labeled = judge.apply_labels([row], decisions)[0]
    assert labeled["class"] == "C2" and not labeled["excluded_reason"]
    assert labeled["raw_response"] == row["raw_response"]
    decisions = judge.validate_decisions({"decisions": [decision(correctness="uncertain")]}, plan[0])
    assert judge.apply_labels([row], decisions)[0]["excluded_reason"]


def test_invalid_quotes_or_official_conflict_are_excluded():
    row = response()
    plan, _ = jobs([row])
    decisions = judge.validate_decisions({"decisions": [decision(evidence=[{"snippet_id": "s1", "quote": "fabricated evidence"}])]}, plan[0])
    assert judge.apply_labels([row], decisions)[0]["excluded_reason"] == "uncertain_or_invalid_judgment"
    decisions = judge.validate_decisions({"decisions": [decision(correctness="incorrect", support="contradicted", evidence=[])]}, plan[0])
    assert judge.apply_labels([row], decisions)[0]["excluded_reason"] == "official_semantic_conflict"
    with pytest.raises(ValueError):
        judge.validate_decisions(None, plan[0])


def test_mini_cost_cache_and_cap_use_correct_rates(tmp_path, monkeypatch):
    plan, _ = jobs([response()])
    envelope = {"usage": {"prompt_tokens": 100, "completion_tokens": 100}, "choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps({"decisions": [decision()]})}}]}
    calls = []
    class Result:
        status_code = 200
        def json(self):
            return copy.deepcopy(envelope)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or Result())
    monkeypatch.setattr(old_judge, "read_api_key", lambda *a: "fake-test-key")
    client = judge.BudgetClient(tmp_path, 10, tmp_path / "key", input_rate=judge.INPUT_RATE,
                               output_rate=judge.OUTPUT_RATE, validator=judge.validate_decisions)
    assert client.call(plan[0])
    assert client.spent == pytest.approx(0.0002)
    assert client.call(plan[0]) and len(calls) == 1
    low = tmp_path / "low"
    blocked = judge.BudgetClient(low, 0.0000001, tmp_path / "key", input_rate=judge.INPUT_RATE,
                               output_rate=judge.OUTPUT_RATE, validator=judge.validate_decisions)
    assert blocked.call(plan[0]) == {} and len(calls) == 1
    assert "fake-test-key" not in (tmp_path / "spend-ledger.json").read_text()


def test_process_lock_prevents_parallel_paid_runs(tmp_path):
    with judge.paid_session(tmp_path):
        with pytest.raises(FileExistsError):
            with judge.paid_session(tmp_path):
                pass
        assert (tmp_path / "paid-run.lock").exists()
    assert not (tmp_path / "paid-run.lock").exists()
