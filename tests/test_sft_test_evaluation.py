import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gadi_sft_8b_starter/scripts'))
import run_sft_test_evaluation as worker
from scripts import analyze_sft_test_evaluation as scorer


def test_real_official_panel_is_95_disjoint_and_gold_separated():
    from max5_dpo_data import panels,read,DEFAULT_CONFIG
    panel,_=panels(read(DEFAULT_CONFIG))
    forbidden={r['question_id'] for rr in panel.values() for r in rr}
    forbidden.update(r['question_id'] for r in worker.records(worker.ROOT/'data/expansion_dev160.jsonl'))
    inputs,gold,batches=worker.test_panel(ROOT/'data/Task13BTest',forbidden)
    assert len(inputs)==len(gold)==95
    assert [b['factoid_count'] for b in batches]==[26,27,20,22]
    assert all(set(r)=={'question_id','question','snippets'} for r in inputs)
    raw={q['id']:q for p in (ROOT/'data/Task13BTest').glob('*.json') for q in worker.read(p)['questions']}
    for r in inputs:assert [s['text'] for s in r['snippets']]==[s['text'] for s in raw[r['question_id']]['snippets']]
    assert all(g['gold_aliases']==worker.flatten(raw[g['question_id']]['exact_answer']) for g in gold)
    with pytest.raises(ValueError,match='overlap'):
        worker.test_panel(ROOT/'data/Task13BTest',{inputs[0]['question_id']})


def test_portable_bundle_hash_and_overlap_checks(tmp_path):
    inputs,gold,batches=worker.test_panel(ROOT/'data/Task13BTest',set())
    worker.write_jsonl(tmp_path/'test_inputs.jsonl',inputs)
    worker.write_jsonl(tmp_path/'test_gold.jsonl',gold)
    worker.write(tmp_path/'manifest.json',{'question_count':95,'batches':batches,
        'file_hashes':{f:worker.digest(tmp_path/f) for f in ('test_inputs.jsonl','test_gold.jsonl')}})
    assert worker.bundle_panel(tmp_path,set())==(inputs,gold,batches)
    with pytest.raises(ValueError,match='overlap'):
        worker.bundle_panel(tmp_path,{inputs[0]['question_id']})
    with (tmp_path/'test_gold.jsonl').open('a',encoding='utf-8') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed'):worker.bundle_panel(tmp_path,set())


def test_original_and_maxfive_scoring_retains_failures_and_ranks(tmp_path,monkeypatch):
    from run_single_answer_sampling import unique_candidates,expansion_candidates,parse_single_answer
    from run_extractive_expansion_8b import parse_equivalent_response
    submission=tmp_path/'submission';submission.mkdir()
    inputs=[{'question_id':str(i),'question':'Which?','snippets':[{'snippet_id':'s','text':'Complete evidence.'}]} for i in range(95)]
    gold=[{'question_id':r['question_id'],'question':r['question'],'gold_aliases':['accepted'],'batch':'13B1'} for r in inputs]
    worker.write_jsonl(submission/'test_inputs.jsonl',inputs)
    worker.write_jsonl(submission/'test_gold.jsonl',gold)
    worker.write(submission/'manifest.json',{'file_hashes':{f:worker.digest(submission/f) for f in ('test_inputs.jsonl','test_gold.jsonl')},
        'exposure_status':'not certified unseen','batches':[]})
    manifest={'status':'complete','smoke_test':False,'expected_questions':95,'runs':{},'arm_hashes':{},'settings':{},
        'submission_dir':str(submission),'submission_manifest_sha256':worker.digest(submission/'manifest.json'),
        'model_key':'llama31','checkpoint_selection':'frozen validation loss'}
    for arm in ('original','max5'):
        directory=tmp_path/arm;directory.mkdir()
        rr,cc=[],[]
        for i,e in enumerate(inputs):
            raw=('Answer: [BE]accepted[EE]' if i<40 else 'bad format') if arm=='original' else json.dumps({'answers':[
                {'answer':'wrong','relation_type':'original'},{'answer':'accepted' if i<50 else 'other','relation_type':'synonym'}]})
            sample={'draw':1,'raw_response':raw,'output_tokens':20}
            if arm=='original':
                try:sample['answer']=parse_single_answer(raw)
                except ValueError:sample['answer']=None
                sample['parse_error']=None if sample['answer'] else 'bad format'
                ordered=unique_candidates([sample])
            else:
                sample['answers'],*_=parse_equivalent_response(raw,5);sample['parse_error']=None
                ordered=expansion_candidates([sample])
            rr.append({'question_id':e['question_id'],'samples':[sample],
                'snippets_truncated':False,'included_snippets':1,'total_snippets':1,'parse_error':sample['parse_error']})
            cc.extend({'question_id':e['question_id'],**c} for c in ordered)
        for name,values in (('examples.jsonl',inputs),('generations.jsonl',rr),('candidates.jsonl',cc)):
            worker.write_jsonl(directory/name,values)
        worker.write(directory/'config.json',{'num_generations':1})
        worker.write(directory/'status.json',{'status':'complete'})
        manifest['runs'][arm]=arm;manifest['settings'][arm]={'num_generations':1}
        manifest['arm_hashes'][arm]={f:worker.digest(directory/f) for f in ('config.json','status.json','examples.jsonl','generations.jsonl','candidates.jsonl')}
    worker.write(tmp_path/'manifest.json',manifest)
    monkeypatch.setattr(scorer,'official_candidate_matches',lambda ee,cc,*a,**kw:
        {(c['question_id'],c['answer']):c['answer']=='accepted' for c in cc})
    monkeypatch.setattr(scorer,'paired_contrast',lambda aa,bb:
        {'mrr_at5':{'sft_minus_base':sum(b['mrr_at5']-a['mrr_at5'] for a,b in zip(aa,bb))/95}})
    scorer.analyze(tmp_path)
    result=worker.read(tmp_path/'comparison_summary.json')
    assert result['question_count']==95
    assert result['conditions']['original']['metrics']['mrr_at5']==pytest.approx(40/95)
    assert result['conditions']['max5']['metrics']['mrr_at5']==pytest.approx(25/95)
    assert result['conditions']['max5']['metrics']['coverage_at5']==pytest.approx(50/95)
    assert result['conditions']['original']['parse_success_count']==40


@pytest.mark.parametrize('fail,models,expected',[(False,[],6),(False,['qwen3'],2),(True,[],0)])
def test_test_submission_preflight_and_dependency_chain(tmp_path,fail,models,expected):
    bash=Path('C:/Program Files/Git/bin/bash.exe');executable=str(bash) if bash.exists() else shutil.which('bash')
    if not executable:pytest.skip('Bash unavailable')
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=ROOT/'gadi_sft_8b_starter/scripts/submit_sft_test_evaluation.sh'
    stub='''
module() { :; }
source() { :; }
python() {
  printf 'prepare %s\\n' "$*" >> "$TEST_LOG"
  [[ "$TEST_FAIL" == 0 ]] || return 1
  while (( $# )); do
    if [[ "$1" == --submission-dir ]]; then shift;mkdir -p "$1";break;fi
    shift
  done
}
qsub() {
  printf 'qsub %s\\n' "$*" >> "$TEST_LOG"
  number=$(cat "$TEST_COUNT");printf '%s' "$((number+1))" > "$TEST_COUNT"
  printf '%s.gadi-pbs\\n' "$number"
}
'''
    script=scripts/source.name
    script.write_text(source.read_text().replace('${USER}','${TEST_USER}').replace('set -euo pipefail','set -euo pipefail\n'+stub),newline='\n')
    count,log=tmp_path/'count',tmp_path/'log';count.write_text('1')
    result=subprocess.run([executable,script.as_posix(),*models],capture_output=True,text=True,
        env={**os.environ,'TEST_USER':'test','TEST_FAIL':str(int(fail)),'TEST_LOG':log.as_posix(),'TEST_COUNT':count.as_posix()})
    jobs=[s for s in log.read_text().splitlines() if s.startswith('qsub ')] if log.exists() else []
    assert len(jobs)==expected,result.stderr
    if not fail:
        assert result.returncode==0,result.stderr
        for i in range(0,len(jobs),2):assert f'depend=afterok:{i+1}.gadi-pbs' in jobs[i+1]
    else:assert result.returncode!=0
