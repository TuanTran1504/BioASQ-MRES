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


def test_absent_support_alone_does_not_establish_c1():
    row = response('alternative expression', 'C2', 1)
    plan, _ = jobs([row])
    ds = judge.validate_decisions({'decisions': [decision(correctness='incorrect', support='unsupported',
        evidence=[], rationale='No snippet supports this prevalence estimate.')]}, plan[0])
    assert judge.apply_labels([row], ds)[0]['excluded_reason'] == 'incorrectness_based_only_on_absent_support'
    ds = judge.validate_decisions({'decisions': [decision(correctness='incorrect', support='unsupported',
        evidence=[{'snippet_id':'s1','quote':'TNF-alpha is the implicated cytokine.'}],
        rationale='The quoted implicated cytokine differs from the proposed answer.')]}, plan[0])
    assert judge.apply_labels([row], ds)[0]['class'] == 'C1'


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


def test_completed_cache_survives_output_allowance_increase(tmp_path):
    plan, _ = jobs([response()])
    old = copy.deepcopy(plan[0])
    old["payload"]["max_tokens"] -= 160
    old["request_sha256"] = judge.sha(old["payload"])
    judge.save(tmp_path / "plan.json", [old])
    judge.save(tmp_path / "cache" / (old["request_sha256"] + ".json"), {})
    resumed, count = judge.reuse_completed_jobs(plan, tmp_path)
    assert count == 1 and resumed == [old]
    changed = copy.deepcopy(plan)
    public = json.loads(changed[0]["payload"]["messages"][1]["content"])
    public['question'] += ' Different context'
    changed[0]["payload"]["messages"][1]["content"] = json.dumps(public)
    assert judge.reuse_completed_jobs(changed, tmp_path) == (changed, 0)
    (tmp_path / "cache" / (old["request_sha256"] + ".json")).unlink()
    assert judge.reuse_completed_jobs(plan, tmp_path) == (plan, 0)


def test_singleton_retry_preserves_completed_units_and_context(tmp_path):
    plan, _ = jobs([response(), response('IL-6', 'C1', 1)])
    split, count = judge.reuse_completed_jobs(plan, tmp_path, split_pending=True)
    assert count == 0 and len(split) == 2
    assert all(len(j['unit_ids']) == 1 for j in split)
    for job in split:
        public = json.loads(job['payload']['messages'][1]['content'])
        assert len(public['candidates']) == 1 and public['candidates'][0]['index'] == 0
    judge.save(tmp_path / 'plan.json', split)
    judge.save(tmp_path / 'cache' / (split[0]['request_sha256'] + '.json'), {})
    resumed, count = judge.reuse_completed_jobs(plan, tmp_path, split_pending=True)
    assert count == 1 and resumed[0] == split[0] and resumed[1] == split[1]


def test_dns_failure_is_unbilled_and_stops_queued_calls(tmp_path, monkeypatch):
    import socket
    import requests
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError(socket.gaierror(11001, 'DNS unavailable'))
    monkeypatch.setattr(requests, 'post', failed)
    monkeypatch.setattr(old_judge, 'read_api_key', lambda *a: 'fake-test-key')
    plan, _ = jobs([response()])
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'key', input_rate=judge.INPUT_RATE,
                               output_rate=judge.OUTPUT_RATE, validator=judge.validate_decisions)
    assert client.call(plan[0]) == {}
    assert client.spent == 0 and client.blocked_reason == 'network_dns_failure'
    assert client.call(plan[0]) == {} and len(calls) == 1
    ledger = json.loads((tmp_path / 'spend-ledger.json').read_text())
    assert ledger['attempts'][0]['no_remote_request'] is True


def test_parallel_completed_requests_keep_consistent_ledger(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import requests
    plan, _ = jobs([response()])
    envelope = {'usage': {'prompt_tokens': 100, 'completion_tokens': 100}, 'choices': [{
        'finish_reason': 'stop', 'message': {'content': json.dumps({'decisions': [decision()]})}}]}
    class Result:
        status_code = 200
        def json(self):
            return copy.deepcopy(envelope)
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Result())
    monkeypatch.setattr(old_judge, 'read_api_key', lambda *a: 'fake-test-key')
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'key', input_rate=judge.INPUT_RATE,
                               output_rate=judge.OUTPUT_RATE, validator=judge.validate_decisions)
    requests_to_run = []
    for i in range(40):
        job = copy.deepcopy(plan[0])
        job['request_sha256'] = f'request-{i}'
        requests_to_run.append(job)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(client.call, requests_to_run))
    ledger = json.loads((tmp_path / 'spend-ledger.json').read_text())['attempts']
    assert len(ledger) == 40 and all(a['status'] == 'complete' for a in ledger)
    assert client.spent == pytest.approx(.008) and client.reserved == pytest.approx(0)


def test_atomic_save_retries_brief_windows_reader_lock(tmp_path, monkeypatch):
    if old_judge.os.name != 'nt':
        pytest.skip('Windows file sharing behavior')
    original_replace = old_judge.os.replace
    calls = []
    def replace(*args):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError('temporary reader lock')
        return original_replace(*args)
    monkeypatch.setattr(old_judge.os, 'replace', replace)
    old_judge.save(tmp_path / 'ledger.json', {'complete': True})
    assert json.loads((tmp_path / 'ledger.json').read_text()) == {'complete': True}
    assert len(calls) == 2 and not list(tmp_path.glob('*.tmp'))
