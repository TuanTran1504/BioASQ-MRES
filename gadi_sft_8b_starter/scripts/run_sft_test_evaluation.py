#!/usr/bin/env python3
"""Original and max-five SFT greedy evaluation on the 95 official factoid tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_matched_8b_sft import ROOT,read,records,write,digest
from run_matched_8b_evaluation import validate_training_pair,DEFAULT_CONFIG as ORIGINAL_CONFIG
from run_base_original_prompt import full_input_plan,snapshot_inference_code,inference_worker
from max5_dpo_data import DEFAULT_CONFIG as MAX5_CONFIG,selected_source
from run_extractive_expansion_8b import write_jsonl


def flatten(value):
    if isinstance(value,str):return [value] if value.strip() else []
    if isinstance(value,list):return [s for v in value for s in flatten(v)]
    raise ValueError('Invalid exact-answer aliases')


def test_panel(test_dir,forbidden_ids):
    inputs,gold,sources,seen=[],[],[],set()
    for path in sorted(test_dir.glob('13B*_golden.json')):
        count=0
        for question in read(path)['questions']:
            if question['type']!='factoid':continue
            qid=question['id']
            if qid in seen or qid in forbidden_ids:raise ValueError('Duplicate test ID or fitting/validation/dev overlap')
            seen.add(qid)
            aliases=flatten(question['exact_answer'])
            snippets=[{'snippet_id':str(i),'pubmed_id':s.get('document','').rsplit('/',1)[-1],
                       'text':s['text']} for i,s in enumerate(question.get('snippets',[]),1)]
            if not aliases or not snippets:raise ValueError('Test question missing aliases/snippets')
            inputs.append({'question_id':qid,'question':question['body'],'snippets':snippets})
            gold.append({'question_id':qid,'question':question['body'],'gold_aliases':aliases,'batch':path.name})
            count+=1
        sources.append({'file':path.name,'sha256':digest(path),'factoid_count':count})
    if len(inputs)!=95 or len(sources)!=4:raise ValueError('Expected 95 factoids across four official 13B test batches')
    return inputs,gold,sources


def bundle_panel(directory,forbidden_ids):
    manifest=read(directory/'manifest.json')
    if manifest['question_count']!=95 or len(manifest['batches'])!=4:
        raise ValueError('Expected 95 official test questions in the portable bundle')
    for name,value in manifest['file_hashes'].items():
        if digest(directory/name)!=value:raise ValueError('Portable test bundle changed')
    inputs,gold=records(directory/'test_inputs.jsonl'),records(directory/'test_gold.jsonl')
    ids={r['question_id'] for r in inputs}
    if len(inputs)!=95 or len(ids)!=95 or len(gold)!=95 or {r['question_id'] for r in gold}!=ids or ids & forbidden_ids:
        raise ValueError('Portable test IDs differ or overlap fitting/validation/dev')
    if any(set(r)!={'question_id','question','snippets'} or not r['snippets'] for r in inputs):
        raise ValueError('Portable inference inputs must contain only question and snippets')
    by_id={r['question_id']:r for r in gold}
    if any(by_id[r['question_id']]['question']!=r['question'] or not by_id[r['question_id']]['gold_aliases'] for r in inputs):
        raise ValueError('Portable gold labels do not match input questions')
    return inputs,gold,manifest['batches']


def prepare(destination,models):
    if not models or len(set(models))!=len(models) or not set(models)<= {'llama31','qwen3','ministral3'}:
        raise ValueError('Select unique supported models')
    original_config=read(ORIGINAL_CONFIG)
    matrix=read(ROOT/original_config['training_matrix'])
    forbidden=set()
    for formulation in ('original','expansion'):
        for split in ('train','validation'):
            forbidden.update(r['question_id'] for r in records(ROOT/matrix['output_dataset']/formulation/f'{split}.jsonl'))
    forbidden.update(r['question_id'] for r in records(ROOT/matrix['outer_dev']))
    max5_config=read(MAX5_CONFIG)
    max5_matrix=read(ROOT/max5_config['training_matrix'])
    for split in ('train','validation'):
        forbidden.update(r['question_id'] for r in records(ROOT/max5_matrix['output_dataset']/'expansion'/f'{split}.jsonl'))
    bundle=ROOT/'data/official_factoid_test95_v1'
    examples,gold,batches=(bundle_panel(bundle,forbidden) if bundle.is_dir() else
                          test_panel(ROOT.parent/'data/Task13BTest',forbidden))
    # Smoke uses validation questions, avoiding test-guided formatting repairs.
    validation=[]
    for row in records(ROOT/matrix['output_dataset']/'original/validation.jsonl'):
        user=json.loads(row['messages'][1]['content'])
        validation.append({'question_id':row['question_id'],'question':user['question'],
            'snippets':[{'snippet_id':s['id'],'text':s['text']} for s in user['snippets']]})
    sources={}
    for model in models:
        original=validate_training_pair(original_config,model)['original']
        max5=selected_source(max5_config,model)
        if original['base_snapshot']!=max5['base_snapshot'] or original['base_revision']!=max5['base_revision']:
            raise ValueError('Original/max-five SFT must use identical pinned backbones')
        sources[model]={}
        for arm,source in (('original',original),('max5',max5)):
            sources[model][arm]={'source':source,'test_plan':full_input_plan(source,examples,512),
                                 'smoke_plan':full_input_plan(source,validation,512)}
    sys.path.insert(0,str(ROOT.parent))
    from src.utility.bioasq_official import _conda_executable,_official_adapter_classes
    subprocess.run([*_conda_executable('java'),'-version'],check=True)
    _official_adapter_classes(ROOT.parent/'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar')
    destination.mkdir(parents=True,exist_ok=False)
    for name,rr in (('test_inputs.jsonl',examples),('test_gold.jsonl',gold),('smoke_inputs.jsonl',validation)):
        write_jsonl(destination/name,rr)
    snapshot=snapshot_inference_code(destination)
    write(destination/'manifest.json',{'ready':True,'models':models,'sources':sources,'question_count':95,
        'code_snapshot':snapshot,'worker_sha256':snapshot['files_sha256']['scripts/run_single_answer_sampling.py'],
        'batches':batches,'file_hashes':{f:digest(destination/f) for f in ('test_inputs.jsonl','test_gold.jsonl','smoke_inputs.jsonl')},
        'fitting_validation_dev_overlap':0,'smoke_partition':'internal validation only',
        'exposure_status':'Historically configured project test batches; not certified unseen',
        'generation_gold_fields':False,'checkpoint_selection':'previously frozen validation-loss checkpoints'})
    print(f'Prepared 95 official test factoids for {len(models)*2} SFT checkpoints; complete evidence; no dev160 evaluation')


def run(submission,model,smoke):
    prepared=read(submission/'manifest.json')
    if not prepared['ready'] or model not in prepared['models']:raise ValueError('Submission not prepared for model')
    for name,value in prepared['file_hashes'].items():
        if digest(submission/name)!=value:raise ValueError('Prepared input/gold files changed')
    worker=inference_worker(submission,prepared)
    name=f"{model}-sft-test-{os.environ.get('PBS_JOBID','manual')}"+('-smoke' if smoke else '')
    output=ROOT/'outputs/sft_test_evaluation'/name
    output.mkdir(parents=True,exist_ok=False)
    state={'status':'running','model_key':model,'smoke_test':smoke,'expected_questions':4 if smoke else 95,
        'submission_dir':str(submission),'submission_manifest_sha256':digest(submission/'manifest.json'),
        'runs':{},'arm_hashes':{},'settings':{},'scope':'internal-validation technical smoke' if smoke else 'official factoid test',
        'checkpoint_selection':'previously frozen validation loss','exposure_status':prepared['exposure_status']}
    write(output/'manifest.json',state)
    try:
        for arm,entry in prepared['sources'][model].items():
            source=entry['source']
            for filename,value in source['adapter_files_sha256'].items():
                if digest(Path(source['adapter'])/filename)!=value:raise ValueError('Frozen source weights/tokenizer changed')
            prompt=output/f'{arm}_prompt.txt';prompt.write_text(source['system_prompt'],encoding='utf-8')
            plan=entry['smoke_plan' if smoke else 'test_plan']
            settings={'input':str(submission/('smoke_inputs.jsonl' if smoke else 'test_inputs.jsonl')),
                'prompt':str(prompt),'seed':3407,'max_new_tokens':512,'max_candidates':5 if arm=='max5' else 1,
                'num_generations':1,'temperature':0,'response_mode':'equivalent_greedy' if arm=='max5' else 'single_answer_greedy',
                'model_loader':source['model_loader'],'chat_template_kwargs':source['chat_template_kwargs'],
                'execution_mode':source.get('execution_mode','default'),'mark_snippets':False,'require_all_snippets':True,
                'max_seq_length':plan['max_seq_length'],'expected_prompt_tokens':plan['expected_prompt_tokens'],
                'expected_prompt_ids_sha256':plan['expected_prompt_ids_sha256'],'input_token_policy':'complete_evidence_auto_size',
                'prompt_version':'sft-official-test-'+arm+'-v1'}
            cfg=output/f'{arm}_config.json';write(cfg,settings)
            arm_dir=output/arm
            subprocess.run([sys.executable,str(worker),'--config',str(cfg),'--model-name',source['adapter'],
                '--output-dir',str(arm_dir),'--limit',str(4 if smoke else 95),'--preflight-all',
                *(['--longest-first'] if smoke else [])],cwd=ROOT,check=True)
            if read(arm_dir/'status.json')['status']!='complete' or len(records(arm_dir/'generations.jsonl'))!=state['expected_questions']:
                raise ValueError('Incomplete generation')
            state['runs'][arm]=arm
            state['settings'][arm]=settings
            state['arm_hashes'][arm]={f:digest(arm_dir/f) for f in ('config.json','examples.jsonl','generations.jsonl','candidates.jsonl','status.json')}
            write(output/'manifest.json',state)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:write(output/'manifest.json',state)
    if not smoke:
        subprocess.run([sys.executable,str(ROOT.parent/'scripts/analyze_sft_test_evaluation.py'),str(output)],cwd=ROOT,check=True)
    print(f'Completed: {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run'),required=True)
    p.add_argument('--submission-dir',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['llama31','qwen3','ministral3'])
    p.add_argument('--model',choices=('llama31','qwen3','ministral3'))
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    if args.mode=='prepare':prepare(args.submission_dir.resolve(),args.models)
    else:
        if not args.model:p.error('--model required for run')
        run(args.submission_dir.resolve(),args.model,args.smoke_test)


if __name__=='__main__':main()
