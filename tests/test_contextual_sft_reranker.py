from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import contextual_sft_reranker as ce


class Tokenizer:
    cls_token_id, sep_token_id = 101, 102

    def __init__(self):
        self.words, self.reverse = {}, {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.casefold().split():
            if word not in self.words:
                self.words[word] = len(self.words) + 200
                self.reverse[self.words[word]] = word
            ids.append(self.words[word])
        return ids

    def decode(self, ids):
        return ' '.join(self.reverse[i] for i in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


@pytest.fixture
def config():
    result = ce.ev.read(ce.CONFIG)
    result.update(max_length=64, minimum_evidence_tokens=8)
    return result


def candidate(answer='correct', label=1, qid='q'):
    return {'question_id': qid, 'answer': answer, 'label': label, 'sources': ['llama31'],
            'source_ranks': {'llama31': 1}, 'relation_types': ['original'], 'surface_operations': [],
            'is_format_variant': False, 'sample_frequencies': {'llama31': .5}}


def example(qid='q'):
    return {'question_id': qid, 'question': 'Which protein treats disease?',
            'snippets': [{'id': str(i), 'text': text} for i, text in enumerate([
                'irrelevant observation', 'correct protein treats disease', 'another study disease',
                'correct protein also treats disease'], 1)]}


def test_encoding_does_not_read_labels_aliases_or_provenance(config):
    tokenizer = Tokenizer()
    row = candidate()
    e = example()
    first, audit = ce.encode_candidate(e, row, tokenizer, config)
    row.update(label=0, relation_types=['synonym'], sample_frequencies={'qwen3': 1}, gold_aliases=['SECRET'])
    e['gold_aliases'] = ['SECRET']
    second, other = ce.encode_candidate(e, row, tokenizer, config)
    assert first == second and audit == other
    assert 'secret' not in tokenizer.words
    assert audit['selected_snippets'][0]['snippet_id'] == '2'


def test_entire_question_and_answer_are_preserved(config):
    tokenizer = Tokenizer()
    e, row = example(), candidate('long correct name with qualifier')
    encoded, audit = ce.encode_candidate(e, row, tokenizer, config)
    first_segment = encoded['input_ids'][1:1 + audit['question_candidate_tokens']]
    assert tokenizer.decode(first_segment) == f"question: {e['question']} candidate answer: {row['answer']}".casefold()
    assert len(encoded['input_ids']) <= 64
    assert len(encoded['input_ids']) == len(encoded['attention_mask']) == len(encoded['token_type_ids'])


def test_long_head_fails_instead_of_truncating(config):
    with pytest.raises(ValueError, match='no silent head truncation'):
        ce.encode_candidate(example(), candidate('word ' * 100), Tokenizer(), config)


def test_evidence_window_can_select_late_answer_and_reports_omissions(config):
    e = example()
    e['snippets'] = [{'id': 'late', 'text': 'noise ' * 120 + 'correct protein treats disease'}]
    _, audit = ce.encode_candidate(e, candidate(), Tokenizer(), config)
    assert audit['selected_snippets'][0]['start_token'] > 0
    assert audit['retained_evidence_tokens'] < audit['all_evidence_tokens']
    assert audit['encoded_tokens'] <= config['max_length']


def test_lexical_window_ties_keep_earliest_window():
    tokenizer = Tokenizer()
    ids = tokenizer.encode('noise ' * 30)
    window, start = ce.best_window(ids, 10, tokenizer, 'question', 'answer')
    assert start == 0 and len(window) == 10


@pytest.fixture
def ranking():
    pools = {'max5_greedy': [candidate('wrong', 0), candidate('correct', 1)],
             'max5_sampling4': [candidate('wrong', 0), candidate('correct', 1)]}
    examples = {'q': example(), 'empty': example('empty')}
    neural = {n: [0., 2.] for n in pools}
    logistic = {n: [2., 0.] for n in pools}
    return pools, examples, neural, logistic


def test_rank_endpoints_and_full_pool_invariance(ranking):
    pools, examples, neural, logistic = ranking
    before = copy.deepcopy(pools)
    pure_lr = ce.rank_scores(*ranking, alpha=-1)
    pure_ce = ce.rank_scores(*ranking)
    zero = ce.rank_scores(*ranking, alpha=0)
    one = ce.rank_scores(*ranking, alpha=1)
    assert pure_lr[0] == zero[0]
    assert pure_ce[0] == one[0]
    assert pure_lr[0]['max5_greedy']['metrics']['mrr_at5'] == .25
    assert pure_ce[0]['max5_greedy']['metrics']['mrr_at5'] == .5
    for result in (pure_lr, pure_ce, zero, one):
        assert result[0]['max5_greedy']['metrics']['coverage_full_pool'] == .5
    assert before == pools


def test_ranking_never_sorts_using_labels(ranking):
    original = ce.rank_scores(*ranking)[2]
    pools = copy.deepcopy(ranking[0])
    for rows in pools.values():
        for row in rows:
            row['label'] = 1 - row['label']
    changed = ce.rank_scores(pools, *ranking[1:])[2]
    assert [(r['pool'], r['rank'], r['answer']) for r in original] == [(r['pool'], r['rank'], r['answer']) for r in changed]


@pytest.mark.parametrize('values', [[float('nan'), 0], [float('inf'), 1], [1]])
def test_bad_candidate_scores_rejected(values):
    with pytest.raises(ValueError):
        ce.ordered_indices([candidate('a'), candidate('b')], values)


def test_hybrid_ties_prefer_frozen_logistic(ranking):
    pools, examples, _, logistic = ranking
    result = ce.select_blend(pools, examples, logistic, logistic, [0., .5, 1.])
    assert result['alpha'] == 0


def test_hybrid_selection_uses_validation_mrr(ranking):
    selected = ce.select_blend(*ranking, [0., .25, .5, .75, 1.])
    assert selected['alpha'] >= .5
    assert selected['mean_validation_mrr_at5'] == .5


def test_class_weights_balance_acceptance_without_excluding_all_negative_questions():
    rows = [candidate('p', 1), candidate('n1', 0), candidate('n2', 0), candidate('n3', 0, 'all-negative')]
    weights = np.array(ce.loss_weights(rows, ['pool'] * 4))
    labels = np.array([r['label'] for r in rows])
    assert weights[labels == 1].sum() == pytest.approx(weights[labels == 0].sum())
    assert weights[3] > 0 and weights.mean() == pytest.approx(1)


def test_encoded_inputs_contain_no_official_label_or_alias(config):
    pools = {'max5_greedy': [candidate('p', 1), candidate('n', 0)]}
    encoded, _ = ce.encode_split(pools, {'q': example()}, Tokenizer(), config, fitting=True)
    assert not any('label' in r or 'gold_aliases' in r for r in encoded)
    dataset = ce.dataset_records(encoded, pools)
    assert [r['labels'] for r in dataset] == [1., 0.]
    assert set(dataset[0]) == {*ce.INPUT_KEYS, 'labels', 'row_weight'}


def test_prediction_alignment_is_verified(config):
    pools = {'max5_greedy': [candidate('p', 1), candidate('n', 0)]}
    encoded, _ = ce.encode_split(pools, {'q': example()}, Tokenizer(), config)
    encoded.reverse()
    with pytest.raises(ValueError, match='ordering changed'):
        ce.dataset_records(encoded, pools)
    with pytest.raises(ValueError, match='Incomplete'):
        ce.split_predictions(encoded, [1.], pools)


def test_partition_guard_rejects_dev_or_test_leakage():
    examples = {s: {f'{s}-{i}': {'question': f'Unique {s} {i}'} for i in range(n)}
                for s, n in [('train', 1296), ('validation', 144)]}
    test = {f'test-{i}': {'question': f'Unique test {i}'} for i in range(95)}
    ce.validate_membership(examples, test, {'dev-0'})
    with pytest.raises(ValueError, match='overlaps'):
        ce.validate_membership(examples, test, {'train-0'})
    test['test-0']['question'] = 'UNIQUE train 0!!'
    with pytest.raises(ValueError, match='text overlaps'):
        ce.validate_membership(examples, test, {'dev-0'})


def test_comparison_preserves_membership_and_reports_question_transitions(ranking):
    result, scores, ranks = ce.compare(*ranking, alpha=1., resamples=50)
    delta = result['paired_contrasts']['max5_greedy_contextual_minus_logistic']['metrics']['mrr_at5']['difference']
    assert delta == .25
    assert result['question_count'] == 2
    for name in result['conditions']:
        assert result['conditions'][name]['metrics']['coverage_full_pool'] == .5


@pytest.fixture
def real_tokenizer(tmp_path):
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    directory = tmp_path / 'tokenizer'
    directory.mkdir()
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'question', ':', 'which', 'protein', '?',
                  'candidate', 'answer', 'correct', 'wrong', 'evidence', 'the', 'study', '.', 'a', 'b', 'c']
    (directory / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n')
    (directory / 'config.json').write_text('{"model_type":"bert"}')
    (directory / 'tokenizer_config.json').write_text('{"tokenizer_class":"BertTokenizer","do_lower_case":true}')
    return AutoTokenizer.from_pretrained(directory, local_files_only=True)


def test_packed_ids_equal_native_bert_pair_for_untruncated_evidence(real_tokenizer, config):
    e = example()
    e['snippets'] = [{'id': '1', 'text': 'The correct protein.'}]
    row = candidate()
    encoded, _ = ce.encode_candidate(e, row, real_tokenizer, config)
    expected = real_tokenizer(f"Question: {e['question']}\nCandidate answer: {row['answer']}", 'The correct protein.')
    assert all(encoded[k] == expected[k] for k in ce.INPUT_KEYS)


def test_prepare_freezes_inputs_and_validates_before_gpu_jobs(tmp_path, monkeypatch, real_tokenizer, config):
    config.update(epochs=1, batch_size=64, gradient_accumulation_steps=1)
    repository = tmp_path / 'repository'
    starter = repository / 'gadi_sft_8b_starter'
    (starter / 'configs').mkdir(parents=True)
    ce.ev.write(repository / ce.CONFIG.relative_to(ce.ROOT), config)
    ce.ev.write(starter / 'configs/matched_8b_evaluation.json', {'training_matrix': 'configs/matrix.json'})
    ce.ev.write(starter / 'configs/matrix.json', {'outer_dev': 'data/dev.jsonl'})
    ce.ev.write_jsonl(starter / 'data/dev.jsonl', [{'question_id': f'dev-{i}'} for i in range(160)])
    job = repository / 'gadi_sft_8b_starter/jobs/contextual_sft_reranker.pbs'
    job.parent.mkdir()
    job.write_text('#!/bin/bash\n')
    examples = {s: {f'{s}-{i}': {'question_id': f'{s}-{i}', 'question': f'Which protein {s} {i}?',
                'snippets': [{'id': '1', 'text': 'The correct protein.'}]} for i in range(n)}
                for s, n in [('train', 1296), ('validation', 144)]}
    pools = {s: {name: [candidate(a, label, qid) for qid in examples[s] for a, label in [('correct', 1), ('wrong', 0)]]
                 for name in config['training_pools']} for s in examples}
    test = {f'test-{i}': {'question_id': f'test-{i}', 'question': f'Which protein test {i}?',
                         'snippets': [{'id': '1', 'text': 'The correct protein.'}]} for i in range(95)}
    test_pools = {name: [candidate(a, label, q) for q in test for a, label in [('correct', 1), ('wrong', 0)]]
                  for name in config['test_pools']}
    gold = {q: {'batch': 'batch', 'gold_aliases': ['SECRET_GOLD_NEVER_MODEL_INPUT']} for q in test}
    monkeypatch.setattr(ce.ev, 'load_comparison', lambda path: ({}, test_pools, test, gold, {}))
    monkeypatch.setattr(ce.ev, 'load_frozen', lambda *args: ({'question_counts': {'train': 1296, 'validation': 144}}, {'max5': {}}, {}))
    monkeypatch.setattr(ce, 'load_training', lambda *args: (examples, pools))
    def logistic(payload, pools, examples):
        return {n: [float(r['label']) for r in rows] for n, rows in pools.items()}
    monkeypatch.setattr(ce, 'logistic_scores', logistic)
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    scores = logistic({}, pools['validation'], examples['validation'])
    conditions = ce.rank_scores(pools['validation'], examples['validation'], scores, scores, -1)[0]
    ce.ev.write(baseline / 'validation_summary.json', {'metrics': {n + '_max5_reranker': r['metrics'] for n, r in conditions.items()}})
    snapshot = tmp_path / 'snapshot'
    real_tokenizer.save_pretrained(snapshot)
    import torch
    from transformers import BertConfig, BertForMaskedLM
    backbone = BertForMaskedLM(BertConfig(vocab_size=real_tokenizer.vocab_size, hidden_size=12,
        intermediate_size=24, num_hidden_layers=1, num_attention_heads=3,
        max_position_embeddings=64, pad_token_id=real_tokenizer.pad_token_id))
    backbone.save_pretrained(snapshot)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda *args, **kwargs: str(snapshot))
    def snapshot_code(repository, output):
        target = output / 'scripts/contextual_sft_reranker.py'
        target.parent.mkdir(parents=True)
        target.write_text('# frozen worker\n')
        return {'scripts/contextual_sft_reranker.py': ce.ev.digest(target)}
    monkeypatch.setattr(ce.rr, 'snapshot_code', snapshot_code)
    submission = tmp_path / 'submission'
    ce.prepare(repository, baseline, tmp_path / 'comparison', submission)
    assert ce.validate_submission(submission)['dev160_excluded'] is True
    bytecode = submission / 'code/scripts/__pycache__/worker.pyc'
    bytecode.parent.mkdir()
    bytecode.write_bytes(b'benign runtime cache')
    assert ce.validate_submission(submission)['dev160_excluded'] is True
    text = (submission / 'test_inputs.jsonl').read_text()
    assert 'SECRET_GOLD' not in text and '"label"' not in text
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trained = ce.train(submission, smoke=False, cpu=True)
        state = ce.ev.read(trained / 'manifest.json')
        assert state['status'] == 'complete' and state['global_steps'] > 1
        assert ce.ev.read(trained / 'selection.json')['hybrid']['alpha'] in config['blend_weights']
        ce.evaluate(submission, trained, cpu=True, resamples=50)
        result = ce.ev.read(submission / 'comparison-manual/comparison_summary.json')
        assert result['question_count'] == 95
        assert result['conditions']['max5_greedy_logistic']['metrics']['mrr_at5'] == 1
        assert {c['metrics']['coverage_full_pool'] for c in result['conditions'].values()} == {1}
    finally:
        torch.set_num_threads(old_threads)
    (submission / 'validation_pools.json').write_text('{}')
    with pytest.raises(ValueError, match='Frozen input changed'):
        ce.validate_submission(submission)


def test_weighted_loss_matches_direct_bce_and_does_not_forward_labels(real_tokenizer):
    torch = pytest.importorskip('torch')
    WeightedTrainer, _ = ce.training_classes()
    class Model:
        def __call__(self, **kwargs):
            assert 'labels' not in kwargs and 'row_weight' not in kwargs
            return type('Output', (), {'logits': torch.tensor([[0.], [1.]], requires_grad=True)})()
    inputs = {'input_ids': torch.tensor([[1], [2]]), 'labels': torch.tensor([1., 0.]), 'row_weight': torch.tensor([2., .5])}
    loss = WeightedTrainer.compute_loss(None, Model(), inputs)
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0., 1.]), inputs['labels'], reduction='none') * inputs['row_weight']).mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert 'labels' in inputs  # Caller batch was not mutated.


def test_real_tiny_bert_training_checkpoint_selection_and_reload(tmp_path, real_tokenizer, config):
    torch = pytest.importorskip('torch')
    pytest.importorskip('accelerate')
    from transformers import BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = BertForSequenceClassification(BertConfig(vocab_size=real_tokenizer.vocab_size, hidden_size=12,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=3, num_labels=1,
            max_position_embeddings=64, pad_token_id=real_tokenizer.pad_token_id))
        examples = {s: {qid: {'question_id': qid, 'question': 'Which protein?',
                    'snippets': [{'id': '1', 'text': 'The correct protein.'}]}}
                    for s, qid in [('train', 'fit'), ('validation', 'valid')]}
        pools = {s: {name: [candidate('correct', 1, qid), candidate('wrong', 0, qid)]
                    for name in ['max5_greedy', 'max5_sampling4']}
                 for s, qid in [('train', 'fit'), ('validation', 'valid')]}
        encoded = {s: ce.encode_split(pools[s], examples[s], real_tokenizer, config, s == 'train')[0] for s in pools}
        logistic = {n: [1., 0.] for n in pools['validation']}
        config.update(batch_size=2, gradient_accumulation_steps=1, epochs=1)
        output = tmp_path / 'fit'
        trainer, neural, blend = ce.fit_model(model, real_tokenizer, encoded, pools, examples,
            logistic, config, output, smoke=True, cpu=True)
        assert trainer.state.global_step == 2
        assert trainer.state.best_model_checkpoint
        assert set(neural) == set(logistic)
        assert blend['alpha'] in config['blend_weights']
        loaded = AutoModelForSequenceClassification.from_pretrained(output / 'model', local_files_only=True)
        prediction = ce.predict(loaded, real_tokenizer, encoded['validation'], 2, cpu=True)
        assert np.isfinite(prediction).all() and len(prediction) == 4
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('fail', [False, True])
def test_launcher_validates_then_submits_three_dependent_jobs(tmp_path, fail):
    import os
    import shutil
    import subprocess
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    executable = str(bash) if bash.exists() else shutil.which('bash')
    if not executable:
        pytest.skip('Bash unavailable')
    source = ce.ROOT / 'gadi_sft_8b_starter/scripts/submit_contextual_sft_reranker.sh'
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    stub = '''
module() { :; }
source() { :; }
python3() {
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"; printf 'frozen job' > "$1/job.pbs"; return 0
    fi
    shift
  done
}
qsub() {
  printf '%s\\n' "$*" >> "$TEST_LOG"
  count=$(wc -l < "$TEST_LOG")
  printf '%s.gadi-pbs\\n' "$count"
}
'''
    path = scripts / source.name
    path.write_text(source.read_text().replace('${USER}', '${TEST_USER}').replace(
        'set -euo pipefail', 'set -euo pipefail\n' + stub), newline='\n')
    log = tmp_path / 'qsub.log'
    result = subprocess.run([executable, path.as_posix()], capture_output=True, text=True,
        env={**os.environ, 'TEST_USER': 'unit-test', 'TEST_LOG': log.as_posix(), 'TEST_FAIL': str(int(fail))}, timeout=30)
    if fail:
        assert result.returncode != 0 and not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = log.read_text().splitlines()
        assert len(calls) == 3
        assert 'SMOKE_TEST=1' in calls[0]
        assert 'depend=afterok:1.gadi-pbs' in calls[1]
        assert 'depend=afterok:2.gadi-pbs' in calls[2]
        assert 'CONTEXTUAL_TRAIN_JOB=2.gadi-pbs' in calls[2]
        record = next((tmp_path / 'outputs/contextual_sft_reranker').glob('submission-*/jobs.tsv'))
        assert len(record.read_text().splitlines()) == 4
