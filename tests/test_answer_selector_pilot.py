"""Tie-aware ID supervision, constrained decoding and frozen pilot data checks."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts import answer_selector_core as core
from scripts import prepare_answer_selector_data as data
from scripts import run_answer_selector_pilot as pilot


class TinyTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        if len(text) == 3 and text.isdigit():
            ids = [200 + int(text)]
        else:
            ids = [ord(c) + 10 for c in text]
        return {'input_ids': ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        return '\n'.join(r['content'] for r in messages) + '\nAssistant:'

    def decode(self, ids):
        return ''.join(f'{i-200:03d}' if i > 200 else chr(i-10) for i in ids)

    def save_pretrained(self, path):
        data.write(Path(path) / 'tokenizer.json', {'fixture': True})


def view(qid='q', pool='max5_sampling4', split='train', positive=(1, 3), count=7):
    candidates = [{'id': f'C{i:03d}', 'question_id': qid, 'answer': f'answer {i}',
                   'official_accepted': i in positive, 'sources': ['llama31'],
                   'source_ranks': {'llama31': i}, 'sample_frequencies': {'llama31': 1.},
                   'relation_types': ['original'], 'surface_operations': [], 'is_format_variant': False}
                  for i in range(1, count+1)]
    payload = {'question': 'Which answer?', 'snippets': [{'id': 's', 'text': 'Supplied evidence.'}],
               'candidates': [{'id': r['id'], 'answer': r['answer']} for r in candidates]}
    identity = {'question_id': qid, 'pool': pool, 'split': split}
    inp = {**identity, 'messages': [{'role': 'system', 'content': 'old'}, {'role': 'user', 'content': json.dumps(payload)}]}
    labels = {**identity, 'candidates': candidates, 'accepted_ids': [r['id'] for r in candidates if r['official_accepted']],
              'gold_aliases': ['answer 1']}
    return inp, labels


def test_loss_accepts_ties_is_permutation_invariant_and_ignores_other_vocabulary():
    import torch
    logits = torch.zeros(1, 2, 12, requires_grad=True)
    actions = [{'position': 0, 'legal_token_ids': [3, 4, 5, 6], 'accepted_token_ids': [3, 5]}]
    loss = core.marginal_id_loss(logits, actions)
    assert float(loss.detach()) == pytest.approx(float(torch.log(torch.tensor(2.))))
    flipped = [{'position': 0, 'legal_token_ids': [6, 5, 4, 3], 'accepted_token_ids': [5, 3]}]
    assert core.marginal_id_loss(logits, flipped) == loss
    loss.backward()
    assert logits.grad[0, 0, 3] < 0 and logits.grad[0, 0, 5] < 0
    assert logits.grad[0, 0, 4] > 0 and logits.grad[0, 0, 6] > 0
    assert logits.grad[0, 0, 8] == 0 and logits.grad[0, 1].abs().sum() == 0
    more = logits.detach().clone(); more[0, 0, 3] += 5
    assert core.marginal_id_loss(more, actions) < loss
    all_positive = [{'position': 0, 'legal_token_ids': [3, 5], 'accepted_token_ids': [3, 5]}]
    assert core.marginal_id_loss(logits, all_positive).item() == pytest.approx(0.)


def test_prompt_mapping_independent_of_labels_and_training_stops_before_negative_tail():
    tok, (inp, labels) = TinyTokenizer(), view()
    spec = core.encoding_spec(tok, 7)
    case, train = core.render_case(tok, inp, labels, spec, epoch=1)
    other = copy.deepcopy(labels)
    for row in other['candidates']:
        row['official_accepted'] = not row['official_accepted']
    case2, _ = core.render_case(tok, inp, other, spec, epoch=1)
    assert case == case2
    assert len(train['actions']) == 2
    remaining = {r['id'] for r in labels['candidates'] if r['official_accepted']}
    prior = set()
    for action in train['actions']:
        chosen_token = train['input_ids'][action['position'] + 1]
        assert chosen_token in action['accepted_token_ids']
        assert not prior & set(action['legal_token_ids'])
        prior.add(chosen_token)
        assert len(action['accepted_token_ids']) == len(remaining)
        remaining.pop()
    assert len(train['actions'][0]['accepted_token_ids']) == 2
    assert len(train['actions'][1]['accepted_token_ids']) == 1
    assert train['input_ids'][:len(case['input_ids'])] == case['input_ids']
    assert 'gold_aliases' not in tok.decode(case['input_ids'])
    assert 'official_accepted' not in tok.decode(case['input_ids'])
    assert all('answer '+str(i) in tok.decode(case['input_ids']) for i in range(1, 8))
    assert core.render_case(tok, inp, labels, spec, epoch=2)[0] != case


@pytest.mark.parametrize('positive', [(), (1, 2, 3, 4, 5, 6, 7)])
def test_uncovered_and_all_accepted_pools_remain_generation_cases_but_have_no_ranking_loss(positive):
    tok, (inp, labels) = TinyTokenizer(), view(positive=positive)
    case, training = core.render_case(tok, inp, labels, core.encoding_spec(tok, 7))
    assert len(case['code_tokens']) == 7 and training is None


def test_no_silent_context_truncation_and_complete_output_budget():
    tok, (inp, labels) = TinyTokenizer(), view()
    spec = core.encoding_spec(tok, 7)
    case, training = core.render_case(tok, inp, labels, spec)
    needed = len(case['input_ids']) + core.required_generation_tokens(case, spec)
    assert core.check_budget(case, training, spec, needed) == needed
    with pytest.raises(ValueError, match='no truncation'):
        core.check_budget(case, training, spec, needed - 1)


def test_single_token_code_requirement_and_json_roundtrip():
    tok = TinyTokenizer()
    spec = core.encoding_spec(tok, 56)
    assert json.loads(tok.decode(spec['prefix']+[spec['codes']['001']]+spec['closing'])) == ['001']
    class SplitNumbers(TinyTokenizer):
        def __call__(self, text, **kwargs):
            return {'input_ids': [10, 20]}
    with pytest.raises(ValueError, match='not one token'):
        core.encoding_spec(SplitNumbers(), 7)


def test_constraints_reject_repeated_invalid_or_incomplete_outputs():
    tok, (inp, labels) = TinyTokenizer(), view()
    spec = core.encoding_spec(tok, 7)
    case, _ = core.render_case(tok, inp, labels, spec)
    constraint = core.IDConstraint(case, spec)
    selected = list(case['code_tokens'].values())[:5]
    suffix = []
    for i, token in enumerate(selected):
        assert token in constraint.scan(suffix)[0]
        suffix.append(token)
        suffix += spec['closing'] if i == 4 else spec['separator']
    codes, identities = constraint.parse(suffix + [tok.eos_token_id])
    assert len(codes) == len(set(identities)) == 5
    assert constraint.scan(suffix)[0] == [tok.eos_token_id]
    with pytest.raises(ValueError, match='repeated'):
        constraint.scan([selected[0]] + spec['separator'] + [selected[0]])
    with pytest.raises(ValueError, match='invalid'):
        constraint.scan([999])
    with pytest.raises(ValueError, match='Incomplete'):
        constraint.parse([selected[0]])


def test_tiny_real_causal_lm_backward_and_constrained_generation():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(3407)
    tok, (inp, labels) = TinyTokenizer(), view(count=5, positive=(2, 4))
    spec = core.encoding_spec(tok, 5)
    case, training = core.render_case(tok, inp, labels, spec)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=2048,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id))
    tokens = torch.tensor([training['input_ids']])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial = core.marginal_id_loss(model(tokens).logits, training['actions'])
    initial.backward(); optimizer.step(); optimizer.zero_grad()
    changed = core.marginal_id_loss(model(tokens).logits, training['actions'])
    assert torch.isfinite(changed) and changed < initial
    model.eval()
    codes, identities = core.generate_ids(model, case, spec, 'cpu', torch)
    assert len(codes) == len(set(identities)) == 5


def test_validation_denominator_includes_uncovered_questions():
    _, a = view('covered', count=3, positive=(2,))
    _, b = view('uncovered', count=3, positive=())
    predictions = [{'question_id': 'covered', 'pool': a['pool'], 'selected_source_ids': ['C002','C001','C003']},
                   {'question_id': 'uncovered', 'pool': b['pool'], 'selected_source_ids': ['C001','C002','C003']}]
    summary, scores = core.score_predictions(predictions, [a, b])
    assert summary['mean_validation_mrr_at5'] == .5
    assert summary['conditions'][a['pool']]['coverage_full_pool'] == .5
    assert len(scores[a['pool']]) == 2
    with pytest.raises(ValueError, match='Missing'):
        core.score_predictions(predictions[:1], [a, b])


def test_epoch_zero_is_retained_on_ties_and_nonfinite_scores_fail():
    assert core.choose_epoch([{'epoch': i, 'mean_validation_mrr_at5': v} for i,v in enumerate([.6,.6,.59,.6])]) == 0
    assert core.choose_epoch([{'epoch': i, 'mean_validation_mrr_at5': v} for i,v in enumerate([.6,.61,.61,.59])]) == 1
    with pytest.raises(ValueError):
        core.choose_epoch([{'epoch': 1, 'mean_validation_mrr_at5': .7}])
    with pytest.raises(ValueError):
        core.choose_epoch([{'epoch': 0, 'mean_validation_mrr_at5': float('nan')}])


def export_fixture(directory, monkeypatch):
    monkeypatch.setattr(data, 'COUNTS', {'train': 2, 'validation': 2})
    directory.mkdir()
    for split in data.SPLITS:
        views = [view(f'{split}-{i}', p, split, positive=() if i == 1 else (1,3)) for i in range(2) for p in pilot.POOLS]
        data.jsonl(directory/f'{split}_inputs.jsonl', [r[0] for r in views])
        data.jsonl(directory/f'{split}_supervision.jsonl', [r[1] for r in views])
    data.write(directory/'heldout_ids.json', {'dev160': ['dev'], 'test95': ['test']})
    data.write(directory/'source_provenance.json', {'fixture': True})
    data.write(directory/'manifest.json', {'status': 'complete', 'schema': 'max5-answer-selector-audit-v1',
        'model_training': False, 'question_counts': data.COUNTS, 'output_sha256': pilot.hashes(directory)})


def test_export_membership_and_changed_labels_fail_closed(tmp_path, monkeypatch):
    directory = tmp_path/'export'
    export_fixture(directory, monkeypatch)
    pilot.verify_export(directory)
    path = directory/'validation_supervision.jsonl'
    rows = data.records(path); rows[0]['candidates'][0]['official_accepted'] = False
    data.jsonl(path, rows)
    with pytest.raises(ValueError, match='changed'):
        pilot.verify_export(directory)


def test_preparation_snapshots_all_validation_and_never_loads_model_weights(tmp_path, monkeypatch):
    audit = tmp_path/'audit'
    export_fixture(audit, monkeypatch)
    repo = tmp_path/'repo'
    config = data.read(pilot.CONFIG)
    data.write(repo/'gadi_sft_8b_starter/configs/answer_selector_pilot.json', config)
    data.write(repo/'gadi_sft_8b_starter/configs/max5_dpo_banks_8b.json', {'fixture': True})
    revision = 'a'*40
    snapshot = repo/f'hf/models--llama/snapshots/{revision}'
    data.write(snapshot/'config.json', {'model_type': 'llama'})
    (snapshot/'model.safetensors').write_bytes(b'fixture weights, never loaded')
    from scripts import prepare_reranker_guided_dpo as guided
    from scripts import train_sft_ensemble_reranker as rr
    monkeypatch.setattr(guided,'configure_data_root',lambda p: SimpleNamespace(selected_source=lambda *a: {
        'base_snapshot': str(snapshot), 'base_revision': revision, 'model_loader': 'fast_language_model', 'model_name': 'fixture llama'}))
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k: TinyTokenizer())
    monkeypatch.setattr(rr,'snapshot_code',lambda r,d: data.write(d/'code.json',{'fixture':True}))
    for name in ('baseline_answer_selector.pbs','train_answer_selector.pbs','report_answer_selector.pbs'):
        target=repo/'gadi_sft_8b_starter/jobs'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(pilot.ROOT/'gadi_sft_8b_starter/jobs'/name,target)
    output=tmp_path/'submission'
    pilot.prepare(repo,audit,output)
    pilot.validate(output,backbone=True)
    val=data.records(output/'validation_generation_inputs.jsonl')
    assert len(val)==4 and len({r['question_id'] for r in val})==2
    assert all('accepted_ids' not in r and 'official_accepted' not in str(r) for r in val)
    for epoch in range(1,4):
        rows=data.records(output/f'train_epoch{epoch}.jsonl')
        assert len(rows)==2
        assert all(r['question_id']=='train-0' for r in rows)
    assert data.read(output/'prepared.json')['DPO_training'] is False
    baseline_dir=output/'baseline-test'
    pilot.baseline(output,baseline_dir)
    assert data.read(baseline_dir/'manifest.json')['status']=='complete'
    assert set(data.read(baseline_dir/'summary.json')['conditions'])==set(pilot.POOLS)
    # Exercise the final comparison/diagnostic exporter without pretending this
    # fixture is an LLM training run. Real LM gradients/decoding are tested above.
    supervision=data.records(output/'validation_supervision.jsonl')
    predictions=[]
    for row in supervision:
        order=sorted(row['candidates'],key=lambda r:(not r['official_accepted'],r['id']))[:5]
        predictions.append({'question_id':row['question_id'],'pool':row['pool'],
                            'selected_source_ids':[r['id'] for r in order]})
    metrics,scores=core.score_predictions(predictions,supervision)
    training=output/'fixture-trained'
    data.write(training/'epoch-0/validation_summary.json',metrics)
    data.write(training/'epoch-0/validation_per_question.json',scores)
    data.jsonl(training/'epoch-0/validation_predictions.jsonl',predictions)
    data.write(training/'selection.json',{'selected_epoch':0,'checkpoint_scores':[{'epoch':0,**metrics}]})
    data.write(training/'manifest.json',{'status':'complete','smoke':False,
        'prepared_sha256':data.digest(output/'prepared.json'),'artifacts_sha256':pilot.hashes(training)})
    report=output/'report-test'
    pilot.report(output,training,baseline_dir,report)
    assert data.read(report/'comparison_summary.json')['question_count']==2
    cases=data.records(report/'validation_mistakes.jsonl')
    assert cases and all(r['DPO_training_eligible'] is False and r['split']=='validation' for r in cases)
    assert all(r['diagnosis']=='pool_absence' for r in cases)
    (snapshot/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='backbone changed'):
        pilot.validate(output,backbone=True)


def test_backbone_verification_accepts_hf_blob_links_but_rejects_path_escape(tmp_path):
    model=tmp_path/'models--example'
    snap=model/'snapshots'/'rev'
    snap.mkdir(parents=True)
    data.write(snap/'config.json', {'model_type':'llama'})
    pilot.verify_backbone(snap,pilot.hashes(snap))
    with pytest.raises(ValueError):
        pilot.verify_backbone(snap,{'../../../../outside':'fake'})


def test_full_training_requires_completed_smoke_and_baseline_before_loading_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot,'validate',lambda *a,**k:{})
    data.write(tmp_path/'prepared.json',{'fixture':True})
    with pytest.raises(ValueError,match='successful smoke'):
        pilot.train(tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure', ['none','prepare','train'])
def test_launcher_records_frozen_dependencies_and_never_submits_dpo(tmp_path, failure):
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.exists() else shutil.which('bash')
    if not executable:pytest.skip('Bash unavailable')
    stub=r'''
module() { :; }
source() { :; }
python3() {
  [[ "$FAILURE" != prepare ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then
      shift; mkdir -p "$1"
      for name in baseline train report; do printf '# frozen\n' > "$1/${name}_answer_selector.pbs"; done
      return 0
    fi
    shift
  done
  return 1
}
qsub() {
  if [[ "$FAILURE" == train && "$*" == *answer-sel-train* ]]; then return 1; fi
  printf '%s\n' "$*" >> "$TEST_LOG"
  printf '%s.gadi-pbs\n' "$(wc -l < "$TEST_LOG" | tr -d ' ')"
}
'''
    scripts=tmp_path/'scripts'; scripts.mkdir()
    shell=scripts/'submit.sh'
    text=(pilot.ROOT/'gadi_sft_8b_starter/scripts/submit_answer_selector_pilot.sh').read_text()
    shell.write_text(text.replace('${USER}','${TEST_USER}').replace('set -euo pipefail','set -euo pipefail\n'+stub),newline='\n')
    log=tmp_path/'calls.txt'
    result=subprocess.run([executable,shell.as_posix()],capture_output=True,text=True,timeout=30,
        env={**os.environ,'TEST_USER':'fixture','TEST_LOG':log.as_posix(),'FAILURE':failure})
    if failure=='prepare':
        assert result.returncode!=0 and not log.exists()
    elif failure=='train':
        assert result.returncode!=0 and len(log.read_text().splitlines())==2
    else:
        assert result.returncode==0,result.stderr
        calls=log.read_text().splitlines()
        assert len(calls)==4 and 'afterok:1.gadi-pbs:2.gadi-pbs' in calls[2]
        assert 'afterok:3.gadi-pbs' in calls[3] and 'DPO' not in log.read_text()
        record=next(tmp_path.glob('outputs/answer_selector_pilot/submission-*/jobs.tsv'))
        assert len(record.read_text().splitlines())==5
