import copy
import importlib.util
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('expansion_judge', ROOT / 'scripts/judge_expansion_dpo_annotations.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def row(model='llama31', answer='TNF-alpha', original=None, accepted=True):
    return {'model_key': model, 'draw': 0, 'split': 'train', 'question_id': 'q1',
            'response_id': model, 'raw_response': 'untouched raw text',
            'prompt': [{'role': 'system', 'content': 'generator system'}, {'role': 'user', 'content': json.dumps({
                'question': 'Which protein?', 'snippets': [{'id': '1.1', 'text': 'TNF-alpha is the relevant protein.'}]})}],
            'excluded_reason': '', 'reviewer': '', 'review_notes': '',
            'candidates': [{'answer': original or answer, 'relation_type': 'original', 'official_accepted': accepted,
                            'class': 'C3' if accepted else None, 'supported': None, 'equivalent_to_original': None, 'evidence': ''}]}


def decision(index=0, **kwargs):
    return {'index': index, 'correctness': 'correct', 'support': 'supported', 'equivalence': 'equivalent',
            'relation_valid': 'valid', 'confidence': 'high',
            'evidence': [{'snippet_id': '1.1', 'quote': 'TNF-alpha is the relevant protein.'}],
            'rationale': 'Question and evidence agree.', **kwargs}


def jobs_for(rows):
    return judge.prepare(rows, {'q1': {'exact_answer': ['TNF-alpha']}})[0]


def test_deduplication_hides_model_and_draw_and_keeps_original_context():
    a, b = row(), row('qwen3')
    jobs = jobs_for([a, b])
    assert len(jobs) == 1 and len(jobs[0]['unit_ids']) == 1
    sent = jobs[0]['payload']['messages'][1]['content']
    assert 'llama31' not in sent and 'qwen3' not in sent and 'official_accepted' not in sent
    assert json.loads(sent)['candidates'][0]['original'] == 'TNF-alpha'


def test_different_originals_cannot_share_equivalence_judgment():
    a, b = row(), row('qwen3')
    b['candidates'].append({'answer': 'TNF-alpha', 'relation_type': 'synonym', 'official_accepted': True})
    b['candidates'][0]['answer'] = 'IL-6'
    jobs = jobs_for([a, b])
    assert len(jobs[0]['unit_ids']) == 3


@pytest.mark.parametrize('evidence', [[], [{'snippet_id': 'unknown', 'quote': 'TNF-alpha'}],
                                    [{'snippet_id': '1.1', 'quote': 'fabricated quotation'}]])
def test_support_requires_real_snippet_and_exact_quote(evidence):
    job = jobs_for([row()])[0]
    result = judge.validate_decisions({'decisions': [decision(evidence=evidence)]}, job)
    assert next(iter(result.values()))['validation_issues']


def test_missing_or_duplicate_decision_indices_fail_closed():
    job = jobs_for([row()])[0]
    with pytest.raises(ValueError):
        judge.validate_decisions({'decisions': []}, job)
    with pytest.raises(ValueError):
        judge.validate_decisions({'decisions': [decision(index=2)]}, job)


def test_unmatched_correct_answer_is_c2_without_raw_repair():
    source = row(accepted=False)
    job = jobs_for([source])[0]
    decisions = judge.validate_decisions({'decisions': [decision()]}, job)
    labeled, _ = judge.apply_labels([source], decisions)
    assert labeled[0]['candidates'][0]['class'] == 'C2'
    assert labeled[0]['raw_response'] == source['raw_response']
    assert not labeled[0]['excluded_reason']
    assert source['candidates'][0]['class'] is None


@pytest.mark.parametrize('changes,reason', [
    ({'confidence': 'medium'}, 'uncertain_or_invalid_judgment'),
    ({'correctness': 'uncertain'}, 'unresolved_attributes'),
    ({'correctness': 'incorrect'}, 'official_semantic_conflict'),
    ({'relation_valid': 'invalid'}, 'invalid_declared_relation'),
    ({'equivalence': 'different'}, 'uncertain_or_invalid_judgment'),
])
def test_uncertainty_conflicts_and_wrong_self_equivalence_are_excluded(changes, reason):
    source = row()
    job = jobs_for([source])[0]
    decisions = judge.validate_decisions({'decisions': [decision(**changes)]}, job)
    labeled, _ = judge.apply_labels([source], decisions)
    assert reason in labeled[0]['excluded_reason']


def test_budget_blocks_calls_before_loading_key(tmp_path):
    client = judge.BudgetClient(tmp_path, 0.001, tmp_path / 'no-key')
    job = jobs_for([row()])[0]
    assert client.call(job) == {}
    assert client.spent == 0 and client.key is None


def test_reserved_failed_requests_count_against_resumed_budget(tmp_path):
    judge.save(tmp_path / 'spend-ledger.json', {'attempts': [{'charged_or_reserved_usd': 10}]})
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'no-key')
    assert client.call(jobs_for([row()])[0]) == {}


def test_same_candidate_cannot_be_correct_and_incorrect_across_originals():
    a, b = row(), row('qwen3', original='IL-6')
    b['candidates'].append({**a['candidates'][0], 'relation_type': 'synonym'})
    jobs = jobs_for([a, b])
    job = jobs[0]
    public = json.loads(job['payload']['messages'][1]['content'])
    ds = []
    for c in public['candidates']:
        bad = c['answer'] == 'TNF-alpha' and c['original'] == 'IL-6'
        ds.append(decision(index=c['index'], correctness='incorrect' if bad else 'correct'))
    decisions = judge.validate_decisions({'decisions': ds}, job)
    labeled, _ = judge.apply_labels([a, b], decisions)
    assert all('inconsistent_repeated_candidate_judgment' in r['excluded_reason'] for r in labeled)


def test_false_synonym_is_kept_as_a_negative_despite_wrong_relation_label():
    a = row()
    a['candidates'].append({'answer': 'IL-6', 'relation_type': 'synonym', 'official_accepted': False,
                            'class': None, 'supported': None, 'equivalent_to_original': None, 'evidence': ''})
    job = jobs_for([a])[0]
    public = json.loads(job['payload']['messages'][1]['content'])
    ds = []
    for c in public['candidates']:
        if c['answer'] == 'IL-6':
            ds.append(decision(index=c['index'], correctness='incorrect', support='unsupported',
                               equivalence='different', relation_valid='invalid', evidence=[]))
        else:
            ds.append(decision(index=c['index']))
    labeled, _ = judge.apply_labels([a], judge.validate_decisions({'decisions': ds}, job))
    assert not labeled[0]['excluded_reason']
    assert labeled[0]['candidates'][1]['class'] == 'C1'
    assert labeled[0]['candidates'][1]['supported'] is False
    assert labeled[0]['candidates'][1]['equivalent_to_original'] is False


def test_correct_non_equivalent_variant_stays_c2_not_c1():
    a = row()
    a['candidates'].append({'answer': 'more specific supported expression', 'relation_type': 'synonym',
                            'official_accepted': False, 'class': None, 'supported': None,
                            'equivalent_to_original': None, 'evidence': ''})
    job = jobs_for([a])[0]
    public = json.loads(job['payload']['messages'][1]['content'])
    ds = [decision(index=c['index'], equivalence='different' if c['answer'] != 'TNF-alpha' else 'equivalent',
                   relation_valid='invalid' if c['answer'] != 'TNF-alpha' else 'valid') for c in public['candidates']]
    labeled, _ = judge.apply_labels([a], judge.validate_decisions({'decisions': ds}, job))
    assert not labeled[0]['excluded_reason']
    assert labeled[0]['candidates'][1]['class'] == 'C2'
    assert labeled[0]['candidates'][1]['equivalent_to_original'] is False


def test_paid_response_is_cached_and_charged_once(tmp_path, monkeypatch):
    job = jobs_for([row()])[0]
    calls = []
    envelope = {'usage': {'prompt_tokens': 100, 'completion_tokens': 100}, 'choices': [{
        'finish_reason': 'stop', 'message': {'content': json.dumps({'decisions': [decision()]})}}]}
    class Response:
        status_code = 200
        def json(self):
            return copy.deepcopy(envelope)
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: calls.append(1) or Response())
    monkeypatch.setattr(judge, 'read_api_key', lambda _: 'test-key')
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'key')
    first = client.call(job)
    assert first and client.spent == pytest.approx(0.001)
    assert client.call(job) == first
    assert len(calls) == 1
    assert 'test-key' not in (tmp_path / 'spend-ledger.json').read_text()


def test_parallel_requests_reserve_budget_before_network(tmp_path, monkeypatch):
    job = jobs_for([row()])[0]
    other = copy.deepcopy(job)
    other['request_sha256'] = 'other-request'
    entered, release = threading.Event(), threading.Event()
    import requests
    def post(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise requests.Timeout('request timed out')
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setattr(judge, 'read_api_key', lambda _: 'test-key')
    cap = job['maximum_cost_reserve_usd'] + 0.000001
    client = judge.BudgetClient(tmp_path, cap, tmp_path / 'key')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.call, job)
        assert entered.wait(5)
        assert pool.submit(client.call, other).result(timeout=5) == {}
        release.set()
        assert first.result(timeout=5) == {}
    assert client.spent == pytest.approx(job['maximum_cost_reserve_usd'])
    assert client.spent <= cap
    resumed = judge.BudgetClient(tmp_path, cap, tmp_path / 'key')
    assert resumed.call(other) == {}


def test_refused_paid_response_keeps_usage_and_raw_audit(tmp_path, monkeypatch):
    job = jobs_for([row()])[0]
    envelope = {'usage': {'prompt_tokens': 100, 'completion_tokens': 10}, 'choices': [{
        'finish_reason': 'stop', 'message': {'refusal': 'Unable to judge.', 'content': None}}]}
    class Response:
        status_code = 200
        def json(self):
            return envelope
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: Response())
    monkeypatch.setattr(judge, 'read_api_key', lambda _: 'test-key')
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'key')
    assert client.call(job) == {}
    assert client.spent == pytest.approx(0.00028)
    assert list((tmp_path / 'responses').glob('*.json'))
    assert not list((tmp_path / 'cache').glob('*.json'))


def test_exhausted_credits_stop_queued_calls_without_logging_error_message(tmp_path, monkeypatch):
    job = jobs_for([row()])[0]
    calls = []
    class Response:
        status_code = 429
        def json(self):
            return {'error': {'code': 'credit_balance_exhausted', 'message': 'sensitive message must not be logged'}}
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **kw: calls.append(1) or Response())
    monkeypatch.setattr(judge, 'read_api_key', lambda _: 'test-key')
    client = judge.BudgetClient(tmp_path, 10, tmp_path / 'key')
    assert client.call(job) == {}
    assert client.blocked_reason == 'credit_balance_exhausted'
    assert client.call(job) == {}
    assert len(calls) == 1
    assert 'sensitive message' not in (tmp_path / 'spend-ledger.json').read_text()
