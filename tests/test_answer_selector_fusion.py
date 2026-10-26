"""Saved-ID fusion, strict frozen-input checks and validation-only diagnostics."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import answer_selector_core as core
from scripts import fuse_answer_selector_pilots as fusion
from scripts import prepare_answer_selector_data as data
from scripts import run_answer_selector_pilot as pilot


def complete(directory, pin, smoke=None):
    directory.mkdir(parents=True, exist_ok=True)
    hashes = pilot.hashes(directory); hashes.pop('manifest.json',None)
    state = {'status':'complete','prepared_sha256':pin,'artifacts_sha256':hashes}
    if smoke is not None:
        state['smoke'] = smoke
    data.write(directory/'manifest.json',state)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(data,'COUNTS',{'train':4,'validation':4})
    inputs, labels = [], []
    for q in range(4):
        for pool in fusion.POOLS:
            qid = f'validation-{q}'
            candidates = [{'id':f'C{i:03d}','question_id':qid,'answer':f'answer {i}',
                'official_accepted':q<2 and i==1} for i in range(1,8)] if q<3 else []
            identity = {'question_id':qid,'pool':pool,'split':'validation'}
            payload = {'question':f'Which answer {q}?','snippets':[{'id':'s','text':'Supplied evidence.'}],
                       'candidates':[{'id':c['id'],'answer':c['answer']} for c in candidates]}
            inputs.append({**identity,'messages':[{'role':'system','content':'rank'},
                {'role':'user','content':json.dumps(payload)}]})
            labels.append({**identity,'candidates':candidates,'gold_aliases':['answer 1']})
    def order(model, q):
        if q==3:
            return []
        indexes = {
            'llama31': [2,3,4,5,6],
            'qwen3': [1,2,3,4,5] if q==0 else [2,3,4,5,6],
            'ministral3':[2,1,3,4,5] if q==0 else [1,2,3,4,5],
            'logistic':[3,4,5,6,7] if q==0 else [1,2,3,4,5],
        }[model]
        return [f'C{i:03d}' for i in indexes]
    comparison = {'question_count':4,'test_evaluation':False,'DPO_training':False,
                  'sources':{},'mean_validation_mrr_at5':{}}
    for index,model in enumerate(fusion.MODELS):
        sub = tmp_path/model
        data.write(sub/'configuration.json', {**data.read(pilot.CONFIG),'model_key':model})
        data.jsonl(sub/'validation_inputs.jsonl',inputs)
        data.jsonl(sub/'validation_supervision.jsonl',labels)
        for name in ('train_inputs.jsonl','train_supervision.jsonl'):
            data.jsonl(sub/name,[{'same_fixture':True}])
        data.write(sub/'heldout_ids.json',{'dev160':['dev'],'test95':['test']})
        encoded, predictions, logistic, full_logistic = [], [], [], []
        for row in labels:
            q = int(row['question_id'].rsplit('-',1)[1])
            ids = [c['id'] for c in row['candidates']]
            # Different display codes and mappings must still refer to identical source IDs.
            codes = [f'{i:03d}' if model=='llama31' else chr(64+i) for i in range(1,len(ids)+1)]
            mapping = dict(zip(codes, ids if model=='llama31' else list(reversed(ids))))
            inverse = {v:k for k,v in mapping.items()}
            identity = {k:row[k] for k in ('question_id','pool')}
            encoded.append({**identity,'code_to_source_id':mapping})
            chosen = order(model,q)
            predictions.append({**identity,'selected_codes':[inverse[c] for c in chosen],'selected_source_ids':chosen})
            rank = order('logistic',q)
            logistic.append({**identity,'selected_source_ids':rank})
            entire = rank + [c for c in ids if c not in rank]
            by_id = {c['id']:c for c in row['candidates']}
            full_logistic.extend({'question_id':row['question_id'],'condition':row['pool']+'_logistic',
                'answer':by_id[c]['answer'],'label':int(by_id[c]['official_accepted']),'rank':r,'score':1/r}
                for r,c in enumerate(entire,1))
        data.jsonl(sub/'validation_generation_inputs.jsonl',encoded)
        static = pilot.hashes(sub)
        data.write(sub/'prepared.json',{'configuration':data.read(sub/'configuration.json'),'files_sha256':static})
        pin = data.digest(sub/'prepared.json')
        (sub/'jobs.tsv').write_text('stage\tjob_id\tdepends_on\nbaseline\tb\t\ntrain\tt\t\nreport\tr\t\n')
        metric,perq = core.score_predictions(predictions,labels)
        base_metric,base_perq = core.score_predictions(logistic,labels)
        data.jsonl(sub/'train-t/epoch-0/validation_predictions.jsonl',predictions)
        data.write(sub/'train-t/epoch-0/validation_per_question.json',perq)
        data.write(sub/'train-t/selection.json',{'selected_epoch':0,'checkpoint_scores':[{'epoch':0,**metric}]})
        data.write(sub/'report-r/comparison_summary.json',{'selected_epoch':0,
            'conditions':{'selector':metric['conditions'],'logistic':base_metric['conditions']}})
        data.jsonl(sub/'baseline-b/rankings.jsonl',full_logistic)
        data.write(sub/'baseline-b/per_question.json',base_perq)
        data.write(sub/'baseline-b/summary.json',{'conditions':base_metric['conditions']})
        for stage in ('train-t','baseline-b','report-r'):
            complete(sub/stage,pin,smoke=False if stage=='train-t' else None)
        comparison['sources'][model] = {'submission':str(sub),'report':str(sub/'report-r'),
            'prepared_sha256':pin,'selected_epoch':0}
        comparison['mean_validation_mrr_at5'][model] = metric['mean_validation_mrr_at5']
    comparison['shared_data_sha256'] = {name:data.digest(tmp_path/'llama31'/name) for name in fusion.DATA_FILES}
    path = tmp_path/'comparison.json'
    data.write(path,comparison)
    return path


def test_vote_and_rank_rules_are_different_and_label_blind():
    surfaces = {str(i):f'answer {i}' for i in range(1,10)}
    rankings = {'a':['1','2','3','4','5'],'b':['6','2','3','4','5'],'c':['7','8','9','2','3']}
    votes,union,support = fusion.fuse(rankings,surfaces,'q','p','model_vote')
    ranks,_,_ = fusion.fuse(rankings,surfaces,'q','p','rank_vote')
    assert votes[:2] == ['2','3'] and ranks[0] == '2' and ranks[1] in ('1','6','7')
    assert votes != ranks and support['2']['votes']==3
    assert support['2']['rank_vote']==pytest.approx(1.25)
    reversed_rankings = dict(reversed(list(rankings.items())))
    assert fusion.fuse(reversed_rankings,dict(reversed(list(surfaces.items()))),'q','p','model_vote')[0] == votes
    assert set(union)==set().union(*map(set,rankings.values()))
    assert len(votes)==len(set(votes))==5


def test_empty_lists_case_variants_and_no_unobserved_tail():
    surfaces = {'1':'Women','2':'women','3':'Females'}
    selected,union,_ = fusion.fuse({'a':['1'],'b':['2']},surfaces,'q','p','rank_vote')
    assert set(selected)==set(union)=={'1','2'} and '3' not in selected
    assert fusion.fuse({'a':[],'b':[]},{},'q','p','model_vote')[:2] == ([],[])
    for bad in (['1','1'],['unknown']):
        with pytest.raises(ValueError,match='Invalid'):
            fusion.fuse({'a':bad},surfaces,'q','p','rank_vote')


def test_real_saved_formats_fusion_gains_losses_and_uncovered_denominator(tmp_path,monkeypatch):
    path = fixture(tmp_path,monkeypatch)
    result = fusion.analyze(path,tmp_path/'out',resamples=100)
    assert result['best_individual_selector']=='ministral3'
    assert len(result['conditions'])==8 and len(result['paired_contrasts'])==16
    for pool in fusion.POOLS:
        c = result['coverage_transitions'][f'selectors3_model_vote_minus_logistic_{pool}']
        assert c['gained']==1 and c['lost']==1
        c = result['complementarity'][pool]
        assert c['source_full_pool_covered']==2 and c['three_selector_union_covered']==2
        assert c['selectors_cover_logistic_misses']==['validation-0']
        assert c['selector_hit_count_histogram']['0']==2
        for condition in result['conditions'].values():
            assert condition[pool]['coverage_full_pool']==.5
    audit = data.records(tmp_path/'out/question_audit.jsonl')
    assert len(audit)==8 and all(r['DPO_training_eligible'] is False and r['split']=='validation' for r in audit)
    assert any(not r['candidates'] for r in audit)
    assert all(len(c)==min(5,len(r['candidates'])) for r in audit for c in r['top_five'].values())
    assert result['DPO_pairs_created']==0 and result['test_evaluation'] is False
    assert data.read(tmp_path/'out/manifest.json')['status']=='complete'


@pytest.mark.parametrize('failure',['mapping','scores','logistic_order','frozen_input','selected_epoch','incomplete'])
def test_corrupted_or_inconsistent_inputs_fail_before_fusion(tmp_path,monkeypatch,failure):
    path = fixture(tmp_path,monkeypatch)
    sub = tmp_path/'qwen3'; pin=data.digest(sub/'prepared.json')
    if failure=='mapping':
        p=sub/'train-t/epoch-0/validation_predictions.jsonl'
        rows=data.records(p); rows[0]['selected_codes'][0]='unknown'; data.jsonl(p,rows)
        complete(sub/'train-t',pin,smoke=False)
    elif failure=='scores':
        p=sub/'train-t/epoch-0/validation_per_question.json'
        value=data.read(p); value[fusion.POOLS[0]][0]['mrr_at5']=.123; data.write(p,value)
        complete(sub/'train-t',pin,smoke=False)
    elif failure=='logistic_order':
        p=sub/'baseline-b/rankings.jsonl'; rows=data.records(p)
        first,second=[r for r in rows if r['question_id']=='validation-0' and r['condition']==fusion.POOLS[0]+'_logistic'][:2]
        first['rank'],second['rank']=second['rank'],first['rank']; data.jsonl(p,rows)
        complete(sub/'baseline-b',pin)
    elif failure=='frozen_input':
        data.write(sub/'validation_inputs.jsonl',{'changed':True})
    elif failure=='selected_epoch':
        value=data.read(path); value['sources']['qwen3']['selected_epoch']=1; data.write(path,value)
    else:
        data.write(sub/'train-t/manifest.json',{'status':'failed'})
    with pytest.raises((ValueError,KeyError)):
        fusion.analyze(path,tmp_path/'out',resamples=100)
    assert data.read(tmp_path/'out/manifest.json')['status']=='failed'
    assert not (tmp_path/'out/comparison_summary.json').exists()


def test_prepare_freezes_comparison_code_and_cpu_job(tmp_path,monkeypatch):
    path = fixture(tmp_path,monkeypatch)
    from scripts import train_sft_ensemble_reranker as rr
    monkeypatch.setattr(rr,'snapshot_code',lambda repo,dest:data.write(dest/'fixture.json',{'snapshot':True}))
    output=tmp_path/'submission'
    fusion.prepare(pilot.ROOT,path,output,100)
    state=data.read(output/'prepared.json')
    data.verify_files(output,state['files_sha256'])
    assert state['resamples']==100 and state['DPO_training'] is False
    assert 'ngpus' not in (output/'job.pbs').read_text()
    (output/'source_comparison.json').write_text('{}')
    with pytest.raises(ValueError):
        data.verify_files(output,state['files_sha256'])


@pytest.mark.parametrize('failure',['none','prepare','qsub'])
def test_launcher_submits_only_one_frozen_cpu_job(tmp_path,failure):
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.exists() else shutil.which('bash')
    if not executable: pytest.skip('Bash unavailable')
    stub=r'''
module() { :; }
source() { :; }
python3() {
  [[ "$FAILURE" != prepare ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission ]]; then shift; mkdir -p "$1"; printf '# CPU only\n' > "$1/job.pbs"; return 0; fi
    shift
  done
  return 1
}
qsub() { [[ "$FAILURE" != qsub ]] || return 1; printf '%s\n' "$*" > "$TEST_LOG"; printf '123.gadi-pbs\n'; }
'''
    scripts=tmp_path/'scripts'; scripts.mkdir()
    path=scripts/'submit.sh'
    text=(pilot.ROOT/'gadi_sft_8b_starter/scripts/submit_answer_selector_fusion.sh').read_text()
    path.write_text(text.replace('${USER}','${TEST_USER}').replace('set -euo pipefail','set -euo pipefail\n'+stub),newline='\n')
    log=tmp_path/'qsub.txt'
    result=subprocess.run([executable,path.as_posix()],capture_output=True,text=True,timeout=30,
        env={**os.environ,'FAILURE':failure,'TEST_USER':'fixture','TEST_LOG':log.as_posix()})
    if failure=='none':
        assert result.returncode==0,result.stderr
        assert len(log.read_text().splitlines())==1 and 'SELECTOR_FUSION_SUBMISSION=' in log.read_text()
        record=next(tmp_path.glob('outputs/answer_selector_fusion/submission-*/jobs.tsv'))
        assert record.read_text().splitlines()==['stage\tjob_id','fusion\t123.gadi-pbs']
    else:
        assert result.returncode!=0 and not log.exists()
