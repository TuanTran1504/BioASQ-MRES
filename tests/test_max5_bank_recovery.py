"""Recovery preserves all complete draws and rejects changes to their protocol."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gadi_sft_8b_starter/scripts'))
from max5_bank_recovery import recover_rows
from expansion_dpo_data import response_id
from run_single_answer_sampling import sample_seed


@pytest.fixture
def bank(tmp_path):
    config={'seed':3407,'sampled_responses':4,'max_seq_length':8704,'max_new_tokens':512}
    source={'adapter':'frozen','files_sha256':{'adapter':'hash'}}
    panel={s:[{'question_id':s,'messages':[
        {'role':'system','content':'Generate five expressions'},
        {'role':'user','content':'question and evidence'},
        {'role':'assistant','content':'gold must stay excluded'}]}] for s in ('train','validation')}
    manifest={'status':'running','model_key':'ministral3','smoke_test':False,'full_dataset':True,
              'gold_blind_generation':True,'formulation':'expansion','configuration':config,
              'source':source,'dpo_config_sha256':'config_hash',
              'question_ids':{s:[s] for s in panel},'worker_sha256':'old_worker'}
    rows=[]
    for split,rr in panel.items():
        for draw in range(5):
            raw='malformed model text' if draw==1 else '{"answers":[]}'
            rows.append({'split':split,'question_id':split,'model_key':'ministral3','draw':draw,
                         'prompt':rr[0]['messages'][:2],'raw_response':raw,
                         'seed':sample_seed(split,draw+1,3407),'decoding':'sampling' if draw else 'greedy',
                         'response_id':response_id('ministral3',split,draw,raw),
                         'prompt_tokens':200,'output_tokens':512 if draw==2 else 20,
                         'generation_seconds':1.2})
    return tmp_path,config,source,panel,manifest,rows


def save(bank,rows=None,tail=b''):
    directory,_,_,_,manifest,all_rows=bank
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    body=b''.join((json.dumps(r)+'\n').encode() for r in (all_rows if rows is None else rows))
    (directory/'responses.jsonl').write_bytes(body+tail)


def recover(bank):
    directory,config,source,panel,_,_=bank
    return recover_rows(directory,config,source,panel,'ministral3','config_hash')


@pytest.mark.parametrize('count',[1,3,5,7,10])
def test_recovers_partial_question_or_full_prefix_without_reparsing_answers(bank,count):
    directory,_,_,_,_,rows=bank
    save(bank,rows[:count])
    original=(directory/'responses.jsonl').read_bytes()
    restored,audit=recover(bank)
    assert restored==rows[:count]
    assert audit['reused_responses']==count and audit['remaining_responses']==10-count
    assert audit['ignored_unterminated_tail_bytes']==0
    assert (directory/'responses.jsonl').read_bytes()==original


@pytest.mark.parametrize('tail',[b'{"question_id":',b'{"raw_response":"\xff'])
def test_discards_only_interrupted_unterminated_json_tail(bank,tail):
    save(bank,bank[-1][:3],tail)
    restored,audit=recover(bank)
    assert len(restored)==3 and audit['ignored_unterminated_tail_bytes']==len(tail)


def test_valid_final_record_without_newline_is_reused(bank):
    save(bank,bank[-1][:3])
    path=bank[0]/'responses.jsonl'
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    assert len(recover(bank)[0])==3


@pytest.mark.parametrize('kind',['middle','terminated','blank'])
def test_complete_jsonl_corruption_cannot_be_silently_dropped(bank,kind):
    rows=bank[-1]
    save(bank,rows[:2],b'broken\n' if kind=='terminated' else b'\n' if kind=='blank' else b'broken\n'+json.dumps(rows[2]).encode())
    with pytest.raises(ValueError,match='Malformed'):recover(bank)


@pytest.mark.parametrize('field,value',[
    ('status','complete'),('smoke_test',True),('source',{}),('dpo_config_sha256','changed'),
    ('question_ids',{'train':['dev160'],'validation':['validation']}),('configuration',{}),
    ('gold_blind_generation',False),('full_dataset',False),('model_key','qwen3')])
def test_rejects_changed_manifest_or_partitions(bank,field,value):
    bank[-2][field]=value
    save(bank,bank[-1][:3])
    with pytest.raises(ValueError):recover(bank)


@pytest.mark.parametrize('field,value',[
    ('question_id','dev160'),('model_key','llama31'),('draw',True),('seed',0),
    ('decoding','greedy'),('response_id','tampered'),('prompt',[]),('raw_response',None),
    ('prompt_tokens',9000),('output_tokens',513),('generation_seconds',float('nan'))])
def test_rejects_changed_saved_draws(bank,field,value):
    rows=copy.deepcopy(bank[-1][:3]);rows[1][field]=value
    save(bank,rows)
    with pytest.raises(ValueError):recover(bank)


@pytest.mark.parametrize('kind',['duplicate','missing','reordered','empty'])
def test_requires_exact_prefix(bank,kind):
    rows=bank[-1][:4]
    if kind=='duplicate':rows[2]=rows[1]
    elif kind=='missing':rows.pop(1)
    elif kind=='reordered':rows[1],rows[2]=rows[2],rows[1]
    else:rows=[]
    save(bank,rows)
    with pytest.raises(ValueError):recover(bank)


@pytest.mark.parametrize('state,validation_fails,expected_jobs',[
    ('F',False,1),('R',False,0),('F',True,0)])
def test_recovery_submission_only_queues_after_terminal_state_and_validation(tmp_path,state,validation_fails,expected_jobs):
    import os
    import shutil
    import subprocess
    bash=Path('C:/Program Files/Git/bin/bash.exe')
    executable=str(bash) if bash.exists() else shutil.which('bash')
    if not executable:pytest.skip('Bash unavailable')
    starter=tmp_path/'starter';(starter/'scripts').mkdir(parents=True)
    bank=starter/'outputs/max5_dpo_banks_8b/ministral3-generate-181233007.gadi-pbs'
    bank.mkdir(parents=True)
    log=tmp_path/'calls'
    stub='''
module() { :; }
source() { :; }
qstat() { printf '    job_state = %s\\n' "$TEST_STATE"; }
python() { printf 'validate %s\\n' "$*" >> "$TEST_LOG"; [[ "$TEST_FAIL" == 0 ]]; }
qsub() { printf 'qsub %s\\n' "$*" >> "$TEST_LOG"; printf '999.gadi-pbs\\n'; }
'''
    original=Path(__file__).resolve().parents[1]/'gadi_sft_8b_starter/scripts/resume_max5_dpo_bank.sh'
    script=starter/'scripts'/original.name
    script.write_text(original.read_text().replace('${USER}','${TEST_USER}').replace('set -euo pipefail','set -euo pipefail\n'+stub),newline='\n')
    result=subprocess.run([executable,script.as_posix(),'ministral3',bank.as_posix()],capture_output=True,text=True,
                          env={**os.environ,'TEST_USER':'test','TEST_STATE':state,'TEST_FAIL':str(int(validation_fails)),'TEST_LOG':log.as_posix()})
    lines=log.read_text().splitlines() if log.exists() else []
    queued=[line for line in lines if line.startswith('qsub ')]
    assert len(queued)==expected_jobs,result.stderr
    if expected_jobs:
        assert result.returncode==0,result.stderr
        assert '--resume-from' in lines[0] and '--validate-only' in lines[0]
        assert 'RESUME_BANK=' in queued[0] and 'SMOKE_TEST=0' in queued[0]
        assert 'walltime=12:00:00' in queued[0]
        records=list((starter/'outputs/max5_dpo_banks_8b').glob('submission-recovery-*.tsv'))
        assert len(records)==1
        assert records[0].read_text().splitlines()[1]=='ministral3\tfull\t999.gadi-pbs\t'
    else:
        assert result.returncode!=0


def test_recovery_record_replaces_only_interrupted_model_for_reranker(tmp_path):
    import os
    from scripts.train_sft_ensemble_reranker import latest_max5_banks
    root=tmp_path/'outputs/max5_dpo_banks_8b';root.mkdir(parents=True)
    previous=root/'submission-old.tsv'
    previous.write_text('model\tstage\tjob_id\tdepends_on\n'
                        'llama31\tfull\t181233002.gadi-pbs\t\n'
                        'qwen3\tfull\t181233004.gadi-pbs\t\n'
                        'ministral3\tfull\t181233007.gadi-pbs\t\n')
    recovery=root/'submission-recovery-new.tsv'
    recovery.write_text('model\tstage\tjob_id\tdepends_on\nministral3\tfull\t999.gadi-pbs\t\n')
    os.utime(previous,(1000,1000));os.utime(recovery,(1001,1001))
    paths,jobs=latest_max5_banks(tmp_path)
    assert jobs=={'llama31':'181233002.gadi-pbs','qwen3':'181233004.gadi-pbs','ministral3':'999.gadi-pbs'}
    assert paths['ministral3']==root/'ministral3-generate-999.gadi-pbs'
