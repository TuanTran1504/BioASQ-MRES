#!/usr/bin/env python3
"""Blinded GPT-4.1-mini labels for max-five banks; cached shared US$10 maximum."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'gadi_sft_8b_starter/scripts'))
from prepare_matched_8b_sft import read,records,digest
from max5_dpo_data import DEFAULT_CONFIG,MODELS,validate_bank,review_template,panels,reliable
from scripts import judge_expansion_dpo_annotations as shared
from scripts.judge_single_answer_dpo import paid_session

MODEL='gpt-4.1-mini-2025-04-14'
INPUT_RATE,OUTPUT_RATE=.4/1_000_000,1.6/1_000_000
PRICING_SOURCE='https://developers.openai.com/api/docs/models/gpt-4.1-mini'


def prepare(rows, aliases):
    jobs,units=shared.prepare(rows,{q:{'exact_answer':v} for q,v in aliases.items()})
    for job in jobs:
        payload=job['payload']
        payload['model']=MODEL
        payload['max_tokens']=min(32768,256+320*len(job['unit_ids']))
        job['request_sha256']=shared.sha(payload)
        job['maximum_cost_reserve_usd']=(len(json.dumps(payload,ensure_ascii=False).encode())+2048)*INPUT_RATE+payload['max_tokens']*OUTPUT_RATE
    # Full questions are randomised within split; validation first protects
    # pair-validation yield if the cap is reached. Failed/pending units abstain.
    split_by_id={r['question_id']:r['split'] for r in rows}
    random.Random(3407).shuffle(jobs)
    jobs.sort(key=lambda j:split_by_id[units[j['unit_ids'][0]]['question_id']]!='validation')
    return jobs,units


def validate_review(directory,config):
    state=read(directory/'manifest.json')
    if state['status']!='review' or state['configuration']!=config:
        raise ValueError('Expected officially scored max-five review package')
    for name,value in state['files_sha256'].items():
        if digest(directory/name)!=value:raise ValueError('Review hash changed')
    raw,rows=records(directory/'responses.jsonl'),records(directory/'review.jsonl')
    if set(state['banks'])!=set(MODELS) or {r['model_key'] for r in raw}!=set(MODELS):
        raise ValueError('Review requires all three model banks')
    for model,bank in state['banks'].items():
        if bank['manifest']['model_key']!=model:raise ValueError('Bank model identity differs')
        validate_bank(bank['manifest'],[r for r in raw if r['model_key']==model],config)
    by_id={r['response_id']:r for r in raw}
    if len(by_id)!=len(raw) or len(rows)!=len(raw) or {r['response_id'] for r in rows}!=set(by_id):
        raise ValueError('Review must retain each raw response once')
    for row in rows:
        if any(row.get(k)!=v for k,v in by_id[row['response_id']].items()):raise ValueError('Review changed generation')
        expected=review_template(by_id[row['response_id']],config['max_new_tokens'])
        if row['excluded_reason']!=expected['excluded_reason'] or len(row['candidates'])!=len(expected['candidates']):
            raise ValueError('Review changed max-five parsing')
        for c,a in zip(row['candidates'],expected['candidates']):
            if any(c[k]!=a[k] for k in ('answer','relation_type')):raise ValueError('Review changed candidates')
            if not row['excluded_reason'] and type(c['official_accepted']) is not bool:
                raise ValueError('Official matching required before semantic judging')
    return state,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--max-usd',type=float,default=10)
    p.add_argument('--max-new-calls',type=int,default=10000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--api-key-file',type=Path,default=ROOT/'open_ai_api.txt')
    args=p.parse_args()
    if not 0<args.max_usd<=10 or not 1<=args.workers<=8 or args.max_new_calls<1:
        p.error('Use a US$10-or-less cap, 1–8 workers and positive call limit')
    config=read(DEFAULT_CONFIG)
    review,rows=validate_review(args.review_dir,config)
    _,aliases=panels(config)
    jobs,units=prepare(rows,aliases)
    directory=args.output_dir.resolve()
    state={'status':'prepared','model':MODEL,'review_manifest_sha256':digest(args.review_dir/'manifest.json'),
        'system_sha256':shared.sha(shared.SYSTEM),'unit_count':len(units),'planned_requests':len(jobs),
        'max_usd':args.max_usd,'pricing':{'input_per_million_usd':.4,'output_per_million_usd':1.6,'source':PRICING_SOURCE},
        'full_plan_maximum_reserve_usd':sum(j['maximum_cost_reserve_usd'] for j in jobs),
        'annotation_source':'LLM only; no independent human audit','dev160_training_questions':0}
    if (directory/'manifest.json').exists():
        old=read(directory/'manifest.json')
        if any(old[k]!=state[k] for k in ('model','review_manifest_sha256','system_sha256','unit_count')):
            raise ValueError('Cannot resume a changed judging plan')
    shared.save(directory/'plan.json',jobs)
    shared.save(directory/'manifest.json',state)
    if not args.run:
        print(json.dumps(state,indent=2));return
    decisions={}
    with paid_session(directory):
        client=shared.BudgetClient(directory,args.max_usd,args.api_key_file,
            input_rate=INPUT_RATE,output_rate=OUTPUT_RATE,validator=shared.validate_decisions)
        pending=[]
        for job in jobs:
            if (directory/'cache'/(job['request_sha256']+'.json')).exists():decisions.update(client.call(job))
            else:pending.append(job)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(client.call,j) for j in pending[:args.max_new_calls]]
            for number,future in enumerate(as_completed(futures),1):
                decisions.update(future.result())
                if number%20==0:print(f'Processed {number}/{len(futures)}; spend/reserve US${client.spent+client.reserved:.4f}',flush=True)
    annotated,exclusions=shared.apply_labels(rows,decisions)
    for row in annotated:row['reviewer']='LLM:'+MODEL
    for name,rr in [('reviewed-llm.jsonl',annotated),
        ('first_answer_labels.jsonl',[{'response_id':r['response_id'],'model_key':r['model_key'],'split':r['split'],
            'question_id':r['question_id'],'response_excluded_reason':r['excluded_reason'],
            'candidate':r['candidates'][0] if r['candidates'] else None} for r in annotated]),
        ('expansion_labels.jsonl',[{'response_id':r['response_id'],'model_key':r['model_key'],'split':r['split'],
            'question_id':r['question_id'],'position':i+1,'candidate':c,'response_excluded_reason':r['excluded_reason']}
            for r in annotated for i,c in enumerate(r['candidates']) if i>0])]:
        (directory/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr),encoding='utf-8')
    state.update(status='complete' if len(decisions)==len(units) else 'partial',judged_units=len(decisions),
        resolved_responses=sum(reliable(r) for r in annotated),exclusions=exclusions,
        billed_or_reserved_usd=client.spent,blocking_reason=client.blocked_reason,
        files_sha256={f:digest(directory/f) for f in ('reviewed-llm.jsonl','first_answer_labels.jsonl','expansion_labels.jsonl')})
    shared.save(directory/'manifest.json',state)
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()
