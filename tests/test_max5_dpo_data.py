import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gadi_sft_8b_starter/scripts'))
import max5_dpo_data as data

EVIDENCE='The relevant protein is TNF-alpha in this assay.'


def candidate(answer, *, correct=True, equivalent=True, accepted=False, original=False):
    return {'answer':answer,'relation_type':'original' if original else 'synonym','official_accepted':accepted,
        'judge_decision':{'confidence':'high','validation_issues':[],
            'correctness':'correct' if correct else 'incorrect','support':'supported' if correct else 'contradicted',
            'equivalence':'equivalent' if equivalent else 'different','relation_valid':'valid' if equivalent else 'invalid',
            'evidence':[{'snippet_id':'s','quote':EVIDENCE}],'rationale':'The evidence identifies the answer.'}}


def row(candidates, draw=0, qid='q', model='llama31', split='train'):
    raw=data.serialize(candidates)
    return {'model_key':model,'split':split,'question_id':qid,'draw':draw,
        'prompt':[{'role':'system','content':'Generate up to FIVE equivalents.'},
                  {'role':'user','content':json.dumps({'question':'Which protein?','snippets':[{'id':'s','text':EVIDENCE}]})}],
        'raw_response':raw,'response_id':data.response_id(model,qid,draw,raw),
        'candidates':candidates,'reviewer':'test','excluded_reason':'','output_tokens':30}


def good(first='TNF-alpha'):
    return candidate(first,original=True,accepted=True)


def test_expansion_deletion_preserves_correct_first_and_valid_candidates():
    r=row([good(),candidate('tumor necrosis factor alpha'),candidate('IL-6',correct=False,equivalent=False)])
    pp,_=data.build_pairs([r])
    pair=pp['llama31']['expansion']['train'][0]
    assert [c['answer'] for c in data.answers(pair['chosen'])]==['TNF-alpha','tumor necrosis factor alpha']
    assert pair['rejected']==r['raw_response'] and pair['synthetic_chosen']
    assert pp['llama31']['initial']['train']==[]


def test_equal_length_replacement_uses_judged_donor_no_new_gold_strings():
    rejected=row([good(),candidate('IL-6',correct=False,equivalent=False)])
    donor=row([good(),candidate('tumor necrosis factor alpha')],1)
    pp,_=data.build_pairs([rejected,donor])
    pairs=pp['llama31']['expansion']['train']
    replacement=next(p for p in pairs if p['construction']=='equal_count_validated_replacement')
    assert len(data.answers(replacement['chosen']))==len(data.answers(replacement['rejected']))==2
    assert data.answers(replacement['chosen'])[0]==data.answers(replacement['rejected'])[0]
    assert donor['response_id'] in replacement['source_response_ids']


def test_initial_arm_requires_same_shape_and_coherent_expansion_of_wrong_core():
    correct=row([good(),candidate('tumor necrosis factor alpha')])
    wrong=row([candidate('IL-6',correct=False,original=True),candidate('interleukin 6',correct=False)],1)
    pp,_=data.build_pairs([correct,wrong])
    pair=pp['llama31']['initial']['train'][0]
    assert pair['chosen']==correct['raw_response'] and pair['rejected']==wrong['raw_response']
    assert pair['chosen_id']==correct['response_id'] and not pair['synthetic_chosen']
    wrong['candidates'][1]['judge_decision']['equivalence']='different'
    pp,_=data.build_pairs([correct,wrong])
    assert not pp['llama31']['initial']['train']


def test_correct_unaccepted_first_is_not_a_negative_and_can_be_preserved():
    original=good();original['official_accepted']=False
    r=row([original,candidate('IL-6',correct=False,equivalent=False)])
    pp,_=data.build_pairs([r])
    assert pp['llama31']['expansion']['train']


@pytest.mark.parametrize('problem',['wrong_first','uncertain','unsupported_only','relation_only','bad_quote','official_conflict'])
def test_expansion_pairs_abstain_on_wrong_or_unresolved_core_and_labels(problem):
    r=row([good(),candidate('IL-6',correct=False,equivalent=False)])
    c=r['candidates'][1]
    if problem=='wrong_first':r['candidates'][0]['judge_decision'].update(correctness='incorrect',support='contradicted')
    if problem=='uncertain':c['judge_decision']['confidence']='medium'
    if problem=='unsupported_only':c['judge_decision'].update(support='unsupported',evidence=[])
    if problem=='relation_only':c['judge_decision'].update(correctness='correct',support='supported',equivalence='equivalent')
    if problem=='bad_quote':c['judge_decision']['evidence'][0]['quote']='invented evidence'
    if problem=='official_conflict':c['official_accepted']=True
    pp,_=data.build_pairs([r])
    assert not pp['llama31']['expansion']['train']


def test_pairs_accumulate_across_questions_and_never_cross_models_or_splits():
    rr=[row([good(),candidate('IL-6',correct=False,equivalent=False)],qid=q,model=m,split=s)
        for q in ('q1','q2') for m in data.MODELS for s in ('train','validation')]
    pp,_=data.build_pairs(rr)
    for m in data.MODELS:
        for s in ('train','validation'):
            assert len(pp[m]['expansion'][s])==2
            assert all(p['model_key']==m and p['split']==s for p in pp[m]['expansion'][s])


def test_strict_maxfive_and_output_ceiling_excluded():
    r=row([good()]+[candidate('variant '+str(i)) for i in range(5)])
    with pytest.raises(ValueError,match='five'):data.answers(r['raw_response'])
    assert data.review_template(r)['excluded_reason']=='invalid_max5_schema'
    r=row([good()]);r['output_tokens']=512
    assert data.review_template(r)['excluded_reason']=='output_token_ceiling'


def test_bank_rejects_development_injection_and_missing_draws(monkeypatch):
    rr=row([good()])
    config={'seed':3407,'sampled_responses':4}
    panel={'train':[{'question_id':'q','messages':rr['prompt']}],'validation':[]}
    monkeypatch.setattr(data,'panels',lambda c:(panel,{}))
    monkeypatch.setattr(data,'digest',lambda p:'config')
    manifest={'status':'complete','smoke_test':False,'full_dataset':True,'configuration':config,
        'dpo_config_sha256':'config','model_key':'llama31','question_ids':{'train':['q'],'validation':[]},'response_count':5}
    raw=[]
    for draw in range(5):
        r=row([good()],draw)
        r.update(seed=data.sample_seed('q',draw+1,3407),decoding='sampling' if draw else 'greedy')
        raw.append(r)
    data.validate_bank(manifest,raw,config)
    with pytest.raises(ValueError,match='Incomplete'):data.validate_bank(manifest,raw[:-1],config)
    raw[0]['question_id']='dev160-id'
    with pytest.raises(ValueError,match='out-of-partition'):data.validate_bank(manifest,raw,config)


def test_blinded_mini_plan_pricing_and_candidate_schema():
    from scripts import judge_max5_dpo_banks as judge
    r=row([good(),candidate('IL-6',correct=False,equivalent=False)])
    jobs,units=judge.prepare([r],{'q':['TNF-alpha']})
    assert len(units)==2
    for j in jobs:
        assert j['payload']['model']==judge.MODEL
        public=json.loads(j['payload']['messages'][1]['content'])
        assert 'model_key' not in json.dumps(public) and 'official_accepted' not in json.dumps(public)
        assert j['request_sha256']==judge.shared.sha(j['payload'])
        assert j['maximum_cost_reserve_usd']<.1


@pytest.mark.parametrize('saved_draws',[0,3,5,7,10])
def test_shared_generator_keeps_teacher_answers_out_and_saves_five_draws(tmp_path,monkeypatch,saved_draws):
    import torch
    import run_expansion_dpo_8b as worker
    teacher={'question_id':'fit','messages':[
        {'role':'system','content':'Generate up to FIVE equivalents.'},
        {'role':'user','content':json.dumps({'question':'Which protein?','snippets':[{'id':'s','text':EVIDENCE}]})},
        {'role':'assistant','content':'TARGET_SECRET_NOT_IN_PROMPT'}]}
    panel={'train':[teacher],'validation':[{**teacher,'question_id':'validation'}]}
    monkeypatch.setattr(worker,'split_rows',lambda *a:panel)
    monkeypatch.setattr(worker,'prepare_inference_execution',lambda c:lambda t:None)
    monkeypatch.setattr(worker,'configure_job_local_compiler_cache',lambda:None)
    rendered=[]
    class Tokenizer:
        pad_token_id=0
        eos_token_id=9
        def apply_chat_template(self,prompt,**kwargs):
            assert len(prompt)==2 and all(m['role']!='assistant' for m in prompt)
            rendered.append(json.dumps(prompt));return rendered[-1]
        def __call__(self,*,text,return_tensors=None,**kwargs):
            return {'input_ids':torch.tensor([[1,2]]) if return_tensors else [1,2]}
        def decode(self,ids,**kwargs):return data.serialize([good()])
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=[]
        def generate(self,input_ids,**kwargs):
            self.calls.append(kwargs);return torch.cat([input_ids,torch.tensor([[3]])],dim=1)
    model=Model()
    monkeypatch.setattr(worker,'load_model',lambda *a:(model,Tokenizer()))
    config={'sampled_responses':4,'seed':3407,'max_seq_length':32,'max_new_tokens':4,'temperature':.8,'top_p':.95}
    state={}
    worker.generate(SimpleNamespace(model='llama31',full_dataset=True,smoke_test=False),config,
        {'adapter':'pinned','model_loader':'fast_language_model','chat_template_kwargs':{}},tmp_path,state)
    rr=data.records(tmp_path/'responses.jsonl')
    assert len(rr)==10 and len(rendered)==2 and all('TARGET_SECRET' not in t for t in rendered)
    assert all(sum(r['question_id']==q for r in rr)==5 for q in ('fit','validation'))
    assert sum(not c['do_sample'] for c in model.calls)==2
    assert state['responses_sha256']==data.digest(tmp_path/'responses.jsonl')
    if saved_draws:
        original=rr[:saved_draws]
        resumed=tmp_path/'resumed';resumed.mkdir()
        model.calls.clear()
        state={}
        worker.generate(SimpleNamespace(model='llama31',full_dataset=True,smoke_test=False),config,
            {'adapter':'pinned','model_loader':'fast_language_model','chat_template_kwargs':{}},
            resumed,state,initial_responses=original)
        recovered=data.records(resumed/'responses.jsonl')
        assert recovered[:saved_draws]==original
        assert len(recovered)==10 and len(model.calls)==10-saved_draws
        assert len({(r['split'],r['question_id'],r['draw']) for r in recovered})==10
        assert state['responses_sha256']==data.digest(resumed/'responses.jsonl')


@pytest.mark.parametrize('fail,models,expected',[(False,[],6),(False,['ministral3'],2),(True,[],0)])
def test_max5_generation_submission_dependencies(tmp_path,fail,models,expected):
    import os
    import shutil
    import subprocess
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.exists() else shutil.which('bash')
    if not executable:pytest.skip('Bash unavailable')
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=ROOT/'gadi_sft_8b_starter/scripts/submit_max5_dpo_banks.sh'
    stub='''
module() { :; }
source() { :; }
python() { printf 'validate %s\\n' "$*" >> "$TEST_LOG"; [[ "$TEST_FAIL" == 0 ]]; }
qsub() {
  printf 'qsub %s\\n' "$*" >> "$TEST_LOG"
  number=$(cat "$TEST_COUNT")
  printf '%s' "$((number+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\\n' "$number"
}
'''
    script=scripts/source.name
    script.write_text(source.read_text().replace('${USER}','${TEST_USER}').replace('set -euo pipefail','set -euo pipefail\n'+stub),newline='\n')
    count,log=tmp_path/'count',tmp_path/'log';count.write_text('1')
    result=subprocess.run([executable,script.as_posix(),*models],text=True,capture_output=True,
        env={**os.environ,'TEST_USER':'test','TEST_FAIL':str(int(fail)),'TEST_LOG':log.as_posix(),'TEST_COUNT':count.as_posix()})
    lines=log.read_text().splitlines() if log.exists() else []
    jobs=[line for line in lines if line.startswith('qsub ')]
    assert len(jobs)==expected,result.stderr
    if not fail:
        assert result.returncode==0,result.stderr
        assert next(i for i,line in enumerate(lines) if line.startswith('qsub '))==len(models or data.MODELS)
        for i in range(0,len(jobs),2):
            assert 'SMOKE_TEST=1' in jobs[i] and f'depend=afterok:{i+1}.gadi-pbs' in jobs[i+1]
            assert 'SMOKE_TEST=0' in jobs[i+1]
    else:assert result.returncode!=0
