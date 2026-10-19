import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_replication as p
from scripts import run_gptprompt_replication as paid
from scripts import run_unitor_replication as u


def example(qid='q'):
    return {'question_id': qid, 'question': 'Which enzyme?', 'snippets': [{'text': 'Evidence', 'snippet_id': '1'}]}


def template():
    return {'prefix': [{'role': 'system', 'content': 'System'}], 'factoid_format': 'Format',
            'model': 'gpt-4-turbo', 'max_tokens': 100, 'actual_examples': 2, 'limitation': 'Disclosed'}


def prepared(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for name in ('inputs', 'gold'):
        (bundle / f'test_{name}.jsonl').write_text('{}\n')
    directory = tmp_path / 'prepared'
    directory.mkdir()
    p.save(directory / 'gptprompt_template.json', template())
    p.save(directory / 'prepared.json', {'bundle': str(bundle),
        **{f'test_{n}_sha256': p.digest(bundle / f'test_{n}.jsonl') for n in ('inputs', 'gold')},
        'gpt_template_sha256': p.digest(directory / 'gptprompt_template.json')})
    return directory


def test_payload_has_no_test_answer_or_silent_decoding_change():
    row = {**example(), 'gold_aliases': ['SECRET_GOLD']}
    payload = p.gpt_payload(row, template())
    assert 'SECRET_GOLD' not in json.dumps(payload)
    assert set(payload) == {'model', 'messages', 'max_tokens'}
    assert payload['messages'][-1]['content'] == 'Passage: Evidence\n\n Question:Which enzyme? \n\n Format'
    assert payload['model'] == 'gpt-4-turbo'


@pytest.mark.parametrize('raw,expected', [
    ('{"entities": ["A", "B"]}', ['A', 'B']),
    ('{"entities": []}', []), ('No answer', []),
    ('{"entities": ["A, B"]}', ['A', 'B']),  # Preserve disclosed upstream comma bug.
])
def test_released_gpt_parser(raw, expected):
    assert p.gpt_parse(raw) == expected


def test_unitor_tags_and_answer_order():
    assert p.unitor_parse('Answer: [BE] full name [EE] [BI] ABC[EE]') == ['full name', 'ABC']
    assert p.unitor_parse('untagged') == []


@pytest.mark.parametrize('change', ['id', 'text'])
def test_released_training_cannot_overlap_test(change):
    row = {'id': 'old', 'input_1': 'Other question', 'type': 'factoid'}
    row['id' if change == 'id' else 'input_1'] = 'q' if change == 'id' else 'WHICH enzyme!!'
    with pytest.raises(ValueError, match='overlaps test95'):
        p.audit_training([row], [], [example()])


def test_released_internal_overlap_is_disclosed():
    train = [{'id': 'a', 'input_1': 'Other question?', 'type': 'factoid'}]
    validation = [{'id': 'b', 'input_1': 'OTHER QUESTION', 'type': 'factoid'}]
    audit = p.audit_training(train, validation, [example()])
    assert audit['train_validation_repeated_question_texts'] == ['other question']
    assert audit['test95_overlap_by_id_or_question'] == 0


def test_download_hash_checked_before_writing(tmp_path, monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'wrong'
    monkeypatch.setattr(p.urllib.request, 'urlopen', lambda *a, **k: Response())
    destination = tmp_path / 'source.json'
    with pytest.raises(ValueError, match='differs'):
        p.fetch('owner/repo', 'abc', 'source.json', destination, '0' * 64, True)
    assert not destination.exists()


def test_completion_mask_uses_last_marker_and_never_trains_prompt():
    assert u.labels_after_marker([1, 2, 3, 4, 2, 3, 8, 9], [2, 3]) == [-100] * 6 + [8, 9]
    assert u.labels_after_marker([1, 2, 3], [7]) == [-100] * 3
    assert u.labels_after_marker([1, 2], []) == [-100, -100]


def test_llama_native_prompt_and_phi_native_messages():
    row = {'instruction': 'Rules', 'input_1': 'Question', 'input_2': 'A', 'input_3': 'B', 'input_4': 'C', 'output': '[BE]X[EE]'}
    assert u.llama_text(row) == 'Rules\n\n# Question: Question\n# PubMed resources: A\nB\nC\n# Answer:'
    assert u.llama_text(row, 'EOS', True).endswith('# Answer: [BE]X[EE] EOS')
    assert u.phi_messages(row, True)[-1] == {'role': 'assistant', 'content': 'Answer: [BE]X[EE]'}
    assert len(u.phi_messages(row, False)) == 2


def test_cost_reserve_is_conservative_and_cap_cannot_be_raised():
    amount = paid.reserve(p.gpt_payload(example(), template()))
    assert amount > 2048 * paid.INPUT_RATE + 100 * paid.OUTPUT_RATE
    assert paid.affordable(9, .9, 10)
    assert not paid.affordable(9.9, .2, 10)
    assert not paid.affordable(0, .1, 11)
    assert not paid.affordable(0, .1, float('nan'))


def test_no_paid_call_when_reservation_exceeds_budget(tmp_path, monkeypatch):
    directory = prepared(tmp_path)
    monkeypatch.setattr(paid, 'panel', lambda bundle: ([example()], {}))
    paid.run(directory, tmp_path / 'run', tmp_path / 'missing-key', .001)
    state = p.read(tmp_path / 'run/manifest.json')
    assert state['completed_questions'] == 0
    assert state['charged_or_reserved_usd'] == 0


def test_paid_run_caches_real_usage_and_does_not_repeat_on_resume(tmp_path, monkeypatch):
    directory = prepared(tmp_path)
    monkeypatch.setattr(paid, 'panel', lambda bundle: ([example()], {}))
    import src.utility.eval_openai as key_reader
    monkeypatch.setattr(key_reader, 'read_api_key', lambda path: 'unit-test-secret')
    import requests
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        return SimpleNamespace(status_code=200, json=lambda: {'model': 'gpt-4-turbo-2024-04-09',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 5},
            'choices': [{'message': {'content': '{"entities":["A"]}'}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr(requests, 'post', post)
    output = tmp_path / 'run'
    paid.run(directory, output, tmp_path / 'key', 10)
    paid.run(directory, output, tmp_path / 'key', 10)
    assert len(calls) == 1
    assert p.read(output / 'manifest.json')['charged_or_reserved_usd'] == pytest.approx(.00115)
    assert 'unit-test-secret' not in (output / 'spend-ledger.json').read_text()


def test_uncertain_attempt_is_never_retried_automatically(tmp_path, monkeypatch):
    directory = prepared(tmp_path)
    monkeypatch.setattr(paid, 'panel', lambda bundle: ([example()], {}))
    import src.utility.eval_openai as key_reader
    monkeypatch.setattr(key_reader, 'read_api_key', lambda path: 'test-secret')
    import requests
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise requests.Timeout()
    monkeypatch.setattr(requests, 'post', fail)
    output = tmp_path / 'run'
    paid.run(directory, output, tmp_path / 'key', 10)
    paid.run(directory, output, tmp_path / 'key', 10)
    assert len(calls) == 1
    ledger = p.read(output / 'spend-ledger.json')
    assert ledger['attempts'][0]['charged_or_reserved_usd'] > 0
    assert not (output / 'paid-run.lock').exists()


def test_changed_template_rejected_before_spending(tmp_path):
    directory = prepared(tmp_path)
    p.save(directory / 'gptprompt_template.json', {**template(), 'model': 'gpt-4o'})
    with pytest.raises(ValueError, match='template changed'):
        paid.run(directory, tmp_path / 'run', tmp_path / 'key', 10)


def test_incomplete_generation_cannot_get_a_test_score(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'panel', lambda bundle: ([example()], {'q': {'question_id': 'q'}}))
    generation = tmp_path / 'generations.jsonl'
    generation.write_text(json.dumps({'question_id': 'q', 'raw_response': '[]'}) + '\n')
    with pytest.raises(ValueError, match='complete, unique 95'):
        p.score(tmp_path, generation, tmp_path / 'score', p.gpt_parse)


def test_official_scoring_keeps_failures_and_reports_weighted_and_batch_means(tmp_path, monkeypatch):
    examples, gold = [], {}
    generations = []
    for batch, n in enumerate([26, 27, 20, 22], 1):
        for i in range(n):
            qid = f'b{batch}-{i}'
            row = example(qid)
            examples.append(row)
            gold[qid] = {'question_id': qid, 'batch': f'13B{batch}_golden.json', 'gold_aliases': ['accepted']}
            generations.append({'question_id': qid, 'raw_response': '[BE]wrong[EE][BE]accepted[EE]' if batch == 1 else ''})
    monkeypatch.setattr(p, 'panel', lambda bundle: (examples, gold))
    import src.notebook_workflows.coverage_comparison as coverage
    monkeypatch.setattr(coverage, 'official_candidate_matches', lambda examples, candidates, *args, **kwargs:
                        {(r['question_id'], r['answer']): r['answer'] == 'accepted' for r in candidates})
    path = tmp_path / 'generations.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in generations))
    result = p.score(tmp_path, path, tmp_path / 'score', p.unitor_parse)
    assert result['metrics']['mrr_at5'] == pytest.approx(13 / 95)
    assert result['macro_batch_mrr_at5'] == .125
    assert result['empty_parse_count'] == 69
    assert result['per_batch']['13B1_golden.json']['coverage_at5'] == 1


def test_frozen_unitor_preparation_validates_data_code_and_backbone(tmp_path, monkeypatch):
    data = tmp_path / 'sources/unitor/data/dataset/answer_generation'
    data.mkdir(parents=True)
    for name in ('train_set_answGEN.json', 'dev_set_answGEN.json'):
        p.save(data / name, [{'id': 'released-' + name, 'input_1': 'Released question ' + name,
            'instruction': 'Published instructions', 'type': 'factoid'}])
    snapshot = tmp_path / 'model'
    snapshot.mkdir()
    (snapshot / 'model.safetensors').write_bytes(b'fake test weights')
    p.save(snapshot / 'config.json', {})
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *args, **kwargs: str(snapshot))
    monkeypatch.setattr(u, 'sources', lambda *args, **kwargs: {'models': {'llama31': {'model_id': 'pinned', 'revision': 'a' * 40}}})
    monkeypatch.setattr(u, 'panel', lambda bundle: ([example()], {}))
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for name in ('manifest.json', 'test_inputs.jsonl', 'test_gold.jsonl'):
        (bundle / name).write_text('{}')
    submission = tmp_path / 'submission'
    u.prepare(tmp_path / 'sources', bundle, submission, ['llama31'], False, False)
    state = u.validate_snapshot(submission)
    assert state['models']['llama31']['settings']['r'] == 32
    assert 'output' not in p.read(submission / 'test_rows.json')[0]
    (snapshot / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='backbone changed'):
        u.validate_snapshot(submission)
