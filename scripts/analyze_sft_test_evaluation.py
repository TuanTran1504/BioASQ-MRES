#!/usr/bin/env python3
"""Official scoring of paired original/max-five SFT on the 95-question test panel."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'gadi_sft_8b_starter/scripts'))
from prepare_matched_8b_sft import read,records,write,digest
from run_single_answer_sampling import parse_single_answer,unique_candidates,expansion_candidates
from run_extractive_expansion_8b import parse_equivalent_response
from scripts.analyze_matched_qwen_expansion import ranking_metrics,paired_contrast
from src.notebook_workflows.coverage_comparison import official_candidate_matches


def analyze(directory):
    state=read(directory/'manifest.json')
    if state['status']!='complete' or state['smoke_test'] or state['expected_questions']!=95 or set(state['runs'])!={'original','max5'}:
        raise ValueError('Expected complete paired official test evaluation')
    submission=Path(state['submission_dir'])
    if digest(submission/'manifest.json')!=state['submission_manifest_sha256']:raise ValueError('Prepared manifest changed')
    prepared=read(submission/'manifest.json')
    for name,value in prepared['file_hashes'].items():
        if digest(submission/name)!=value:raise ValueError('Prepared test data changed')
    gold={r['question_id']:r for r in records(submission/'test_gold.jsonl')}
    inputs=records(submission/'test_inputs.jsonl')
    if len(gold)!=95 or len(inputs)!=95 or {r['question_id'] for r in inputs}!=set(gold):raise ValueError('Wrong official test panel')
    expected={r['question_id']:r for r in inputs}
    scored,conditions={},{}
    for arm,relative in state['runs'].items():
        run=(directory/relative).resolve()
        if not run.is_relative_to(directory.resolve()):raise ValueError('Run escaped evaluation')
        for name,value in state['arm_hashes'][arm].items():
            if digest(run/name)!=value:raise ValueError('Archived inference changed')
        config=read(run/'config.json')
        if any(config.get(k)!=v for k,v in state['settings'][arm].items()):
            raise ValueError('Inference configuration differs from prepared settings')
        examples=records(run/'examples.jsonl')
        generations=records(run/'generations.jsonl')
        if len(examples)!=95 or {r['question_id'] for r in examples}!=set(expected):raise ValueError('Input membership differs')
        if any(r!=expected[r['question_id']] for r in examples):raise ValueError('Input text/evidence changed or gold leaked')
        if len(generations)!=95 or {r['question_id'] for r in generations}!=set(expected):raise ValueError('Missing/duplicate generation')
        candidates=[]
        for row in generations:
            if row['snippets_truncated'] or row['included_snippets']!=row['total_snippets'] or len(row['samples'])!=1:
                raise ValueError('Evidence/sample budget differs')
            sample=row['samples'][0]
            if arm=='original':
                try:a=parse_single_answer(sample['raw_response'])
                except ValueError:a=None
                if sample['answer']!=a:raise ValueError('Tagged answer changed')
                cc=unique_candidates([sample])
            else:
                try:aa,*_=parse_equivalent_response(sample['raw_response'],5)
                except ValueError:aa=[]
                if sample['answers']!=aa:raise ValueError('Expansion parsing changed')
                cc=expansion_candidates([sample])
            candidates.extend({'question_id':row['question_id'],**c} for c in cc)
        if candidates!=records(run/'candidates.jsonl'):raise ValueError('Candidate order changed')
        report=directory/(arm+'_official_scores');report.mkdir(exist_ok=True)
        labelled=[{**e,'gold_aliases':gold[e['question_id']]['gold_aliases']} for e in examples]
        matches=official_candidate_matches(labelled,candidates,report,
            jar_path=ROOT/'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar') if candidates else {}
        rows=[]
        for qid in expected:
            ordered=[c['answer'] for c in candidates if c['question_id']==qid]
            rows.append({'question_id':qid,'answers':ordered,'matching_answers':[a for a in ordered if matches[(qid,a)]]})
        metrics,scored[arm]=ranking_metrics(rows)
        per_batch={}
        for batch in {r['batch'] for r in gold.values()}:
            selected=[r for r in scored[arm] if gold[r['question_id']]['batch']==batch]
            per_batch[batch]={'question_count':len(selected),
                **{k:sum(r[k] for r in selected)/len(selected) for k in selected[0] if k!='question_id'}}
        conditions[arm]={'metrics':metrics,'per_batch':per_batch,'parse_success_count':sum(not r.get('parse_error') for r in generations),
            'output_token_ceiling_count':sum(r['samples'][0]['output_tokens']>=512 for r in generations)}
    contrasts=paired_contrast(scored['original'],scored['max5'])
    result={'model_key':state['model_key'],'question_count':95,'conditions':conditions,
        'max5_minus_original':{k:{'difference':v['sft_minus_base'],**{a:b for a,b in v.items() if a!='sft_minus_base'}} for k,v in contrasts.items()},
        'scope':'Official factoid test, not dev160; previously frozen SFT checkpoints',
        'exposure_status':prepared['exposure_status'],'batches':prepared['batches'],
        'checkpoint_selection':state['checkpoint_selection']}
    write(directory/'comparison_summary.json',result)
    write(directory/'comparison_per_question.json',scored)
    print(f"{state['model_key']}: original MRR={conditions['original']['metrics']['mrr_at5']:.6f}, "
          f"max5 MRR={conditions['max5']['metrics']['mrr_at5']:.6f}, "
          f"delta={result['max5_minus_original']['mrr_at5']['difference']:+.6f}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directories',type=Path,nargs='+')
    for directory in p.parse_args().directories:analyze(directory.resolve())
