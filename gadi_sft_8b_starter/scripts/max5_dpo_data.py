#!/usr/bin/env python3
"""Separate max-five initial-answer and expansion-validity preferences."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

from prepare_matched_8b_sft import ROOT, read, records, digest, write, validate, load_source
from expansion_dpo_data import strict_answers, response_id
from run_extractive_expansion_8b import write_jsonl
from run_single_answer_sampling import sample_seed

DEFAULT_CONFIG = ROOT / 'configs/max5_dpo_banks_8b.json'
MODELS = ('llama31', 'qwen3', 'ministral3')
STRATEGIES = ('initial', 'expansion')


def answers(raw):
    result = strict_answers(raw)
    if len(result) > 5:
        raise ValueError('More than five candidates')
    return result


def panels(config):
    matrix = read(ROOT/config['training_matrix'])
    validate(matrix)
    source, _, _ = load_source(matrix)
    audit = source/'audit.jsonl'
    if digest(audit) != read(source/'manifest.json')['output_sha256']['audit.jsonl']:
        raise ValueError('Alias audit changed')
    aliases = {r['question_id']: r['gold_aliases'] for r in records(audit)}
    result = {}
    for split in ('train','validation'):
        result[split] = sorted(records(ROOT/matrix['output_dataset']/'expansion'/f'{split}.jsonl'), key=lambda r:r['question_id'])
        random.Random(config['seed']).shuffle(result[split])
        for row in result[split]:
            if 'up to FIVE' not in row['messages'][0]['content']:
                raise ValueError('Expected trained max-five prompt')
    return result, aliases


def selected_source(config, model):
    """Read only checkpoint metadata from the completed dev run, never its answers."""
    directory = ROOT/config['models'][model]
    evaluation = read(directory/'manifest.json')
    if evaluation['status'] != 'complete' or evaluation['model_key'] != model:
        raise ValueError('Expected completed source evaluation metadata')
    selection = Path(evaluation['selection_run'])
    selected = read(selection/'manifest.json')
    if digest(selection/'manifest.json') != evaluation['selection_manifest_sha256']:
        raise ValueError('SFT selection metadata changed')
    if selected['status'] != 'complete' or selected['smoke_test'] or selected['validation_questions'] != 144:
        raise ValueError('Source must be selected on full internal validation')
    from select_sft_epoch_checkpoints import choose_winners
    winner = choose_winners(selected['checkpoint_scores'])['best_eval_loss']
    saved = selected['selected_adapters']['best_eval_loss']
    recorded = evaluation['sources']['best_eval_loss']
    adapter = Path(saved['path']).resolve()
    if (not adapter.is_relative_to(selection.resolve()) or saved['epoch'] != winner['epoch']
            or saved['epoch'] != recorded['epoch'] or str(adapter) != recorded['adapter']
            or saved['files_sha256'] != recorded['files_sha256']):
        raise ValueError('Source is not the frozen loss-selected SFT adapter')
    for name, value in saved['files_sha256'].items():
        if digest(adapter/name) != value:
            raise ValueError('Selected SFT adapter/tokenizer changed')
    metadata = read(adapter/'training_complete.json')
    training = Path(evaluation['training_run'])
    state = read(training/'status.json')
    if (state['status'] != 'completed' or state['smoke_test']
            or state['selected_train_examples'] != 1296 or state['selected_validation_examples'] != 144
            or digest(training/'status.json') != selected['training_status_sha256']
            or metadata['selection_name'] != 'best_eval_loss' or metadata['model_key'] != model
            or metadata['training_configuration'] != state['configuration']):
        raise ValueError('Source training provenance differs')
    cfg = state['configuration']
    matrix = read(ROOT/config['training_matrix'])
    for field, split in (('train_input','train'),('eval_input','validation')):
        if Path(ROOT/cfg[field]).resolve() != (ROOT/matrix['output_dataset']/'expansion'/f'{split}.jsonl').resolve():
            raise ValueError('Selected model trained on different fitting/validation data')
    if cfg.get('max_candidates') != 5 or not Path(cfg['base_snapshot']).is_dir():
        raise ValueError('Expected cached max-five backbone')
    panel, _ = panels(config)
    prompts = {r['messages'][0]['content'] for rr in panel.values() for r in rr}
    if len(prompts) != 1:
        raise ValueError('Source prompts differ')
    return {**cfg, 'adapter':str(adapter), 'system_prompt':prompts.pop(),
            'selection':'best_eval_loss','epoch':saved['epoch'], 'adapter_files_sha256':saved['files_sha256'],
            'selection_manifest_sha256':digest(selection/'manifest.json'),
            'source_training_status_sha256':digest(training/'status.json')}


def validate_bank(manifest, rows, config):
    if (manifest['status'] != 'complete' or manifest['smoke_test'] or not manifest['full_dataset']
            or manifest['dpo_config_sha256'] != digest(DEFAULT_CONFIG)
            or manifest['configuration'] != config or manifest['model_key'] not in MODELS):
        raise ValueError('Expected completed full max-five bank')
    panel, _ = panels(config)
    wanted_prompts = {(s,r['question_id']):r['messages'][:2] for s,rr in panel.items() for r in rr}
    if manifest['question_ids'] != {s:[r['question_id'] for r in rr] for s,rr in panel.items()}:
        raise ValueError('Question partition differs')
    seen = set()
    for row in rows:
        key = (row['split'],row['question_id'],row['draw'])
        if (key in seen or key[:2] not in wanted_prompts or row['prompt'] != wanted_prompts[key[:2]]
                or row['model_key'] != manifest['model_key']
                or row['seed'] != sample_seed(row['question_id'],row['draw']+1,config['seed'])
                or row['decoding'] != ('sampling' if row['draw'] else 'greedy')
                or row['response_id'] != response_id(row['model_key'],row['question_id'],row['draw'],row['raw_response'])):
            raise ValueError('Changed, duplicate or out-of-partition response')
        seen.add(key)
    wanted = {(*key,draw) for key in wanted_prompts for draw in range(config['sampled_responses']+1)}
    if seen != wanted or manifest['response_count'] != len(rows):
        raise ValueError('Incomplete response bank')


def pack_banks(paths, config):
    banks, raw = {}, []
    for path in paths:
        manifest = read(path/'manifest.json')
        rr = records(path/'responses.jsonl')
        if digest(path/'responses.jsonl') != manifest['responses_sha256']:
            raise ValueError('Bank content hash differs')
        validate_bank(manifest,rr,config)
        model = manifest['model_key']
        if model in banks:
            raise ValueError('Duplicate model bank')
        banks[model] = {'manifest':manifest}
        raw.extend(rr)
    if set(banks) != set(MODELS):
        raise ValueError('Require one bank from each of the three max-five SFT models')
    return banks, raw


def review_template(row, ceiling=512):
    output = {**row, 'excluded_reason':'', 'reviewer':'', 'review_notes':'','candidates':[]}
    try:
        aa = answers(row['raw_response'])
    except (ValueError,TypeError):
        output['excluded_reason'] = 'invalid_max5_schema'
        return output
    if row['output_tokens'] >= ceiling:
        output['excluded_reason'] = 'output_token_ceiling'
    output['candidates'] = [{**a,'official_accepted':None} for a in aa]
    return output


def reliable(row):
    if row.get('excluded_reason') or not row.get('reviewer'):
        return False
    original = answers(row['raw_response'])
    if len(original) != len(row['candidates']):
        raise ValueError('Annotation count differs')
    context = json.loads(row['prompt'][1]['content'])
    snippets = {s['id']:s['text'] for s in context['snippets']}
    for a,c in zip(original,row['candidates']):
        if any(c.get(k) != v for k,v in a.items()):
            raise ValueError('Annotation changed candidate')
        d = c.get('judge_decision',{})
        if (d.get('confidence') != 'high' or d.get('validation_issues') or
                d.get('correctness') not in ('correct','incorrect') or
                d.get('support') not in ('supported','unsupported','contradicted') or
                d.get('equivalence') not in ('equivalent','different') or
                d.get('relation_valid') not in ('valid','invalid') or
                type(c.get('official_accepted')) is not bool):
            return False
        if c['official_accepted'] and d['correctness'] != 'correct':
            return False
        if d['support'] == 'supported' and not d.get('evidence'):
            return False
        for e in d.get('evidence',[]):
            if e['snippet_id'] not in snippets or len(e['quote'].strip()) < 8 or e['quote'] not in snippets[e['snippet_id']]:
                return False
        if d['correctness']=='incorrect' and d['support']=='unsupported' and not d['evidence']:
            return False  # Absence of support alone must not establish incorrectness.
    return True


def valid(c):
    d=c['judge_decision']
    return d['correctness']=='correct' and d['support']=='supported' and d['equivalence']=='equivalent' and d['relation_valid']=='valid'


def harmful(c):
    d=c['judge_decision']
    return not c['official_accepted'] and (d['correctness']=='incorrect' or d['equivalence']=='different' or d['support']=='contradicted')


def serialize(candidates):
    return json.dumps({'answers':[{k:c[k] for k in ('answer','relation_type')} for c in candidates]},ensure_ascii=False)


def build_pairs(rows, cap=2):
    groups, exclusions = defaultdict(list),Counter()
    output = {m:{arm:{s:[] for s in ('train','validation')} for arm in STRATEGIES} for m in MODELS}
    for row in rows:
        if not reliable(row):
            exclusions['excluded_or_unresolved_response']+=1
            continue
        groups[(row['model_key'],row['split'],row['question_id'])].append(row)
    for (model,split,qid), group in sorted(groups.items()):
        group=sorted(group,key=lambda r:r['response_id'])
        per_arm={a:[] for a in STRATEGIES}
        seen={a:set() for a in STRATEGIES}
        def add(arm, chosen, rejected, source, donors, construction):
            if chosen==rejected or (chosen,rejected) in seen[arm]:return
            answers(chosen); answers(rejected)
            seen[arm].add((chosen,rejected))
            per_arm[arm].append({'model_key':model,'split':split,'question_id':qid,'prompt':source['prompt'],
                'chosen':chosen,'rejected':rejected,'chosen_id':(donors[0] if construction=='natural_same_shape'
                    else response_id(model,qid,'derived-'+arm,chosen)),
                'rejected_id':source['response_id'],'source_response_ids':donors,
                'contrast':arm,'construction':construction,'synthetic_chosen':construction!='natural_same_shape'})
        # Initial-answer arm: natural responses, same length and relation-type sequence.
        # Both lists must expand their own core coherently; only the chosen core is correct.
        clean=[r for r in group if all(valid(c) for c in r['candidates'])]
        wrong=[r for r in group if r['candidates'][0]['judge_decision']['correctness']=='incorrect'
               and all(c['judge_decision']['correctness']=='incorrect' and c['judge_decision']['equivalence']=='equivalent' and
                       c['judge_decision']['relation_valid']=='valid' for c in r['candidates'])]
        for a in clean:
            for b in wrong:
                if a['prompt']!=b['prompt']:raise ValueError('Pair prompts differ')
                if [c['relation_type'] for c in a['candidates']] != [c['relation_type'] for c in b['candidates']]:continue
                add('initial',a['raw_response'],b['raw_response'],b,[a['response_id'],b['response_id']],'natural_same_shape')
        # Expansion arm: verified correct first answer unchanged; derived completions are explicit.
        for rejected in group:
            cc=rejected['candidates']
            if not valid(cc[0]):continue
            bad=[i for i,c in enumerate(cc[1:],1) if harmful(c)]
            if not bad or any(not valid(c) and i not in bad for i,c in enumerate(cc)):continue
            originals=answers(rejected['raw_response'])
            # An equally long correction is possible only with independently judged,
            # naturally generated replacements from the same model/question/original.
            donor_candidates=[]
            for donor in group:
                if donor['candidates'][0]['answer'] != cc[0]['answer'] or not valid(donor['candidates'][0]):continue
                for c in donor['candidates'][1:]:
                    if valid(c) and c['answer'].casefold() not in {a['answer'].casefold() for a in originals}:
                        if c['answer'].casefold() not in {a['answer'].casefold() for a,_ in donor_candidates}:
                            donor_candidates.append((c,donor['response_id']))
            if len(donor_candidates)>=len(bad):
                corrected=list(cc)
                for i,(c,_) in zip(bad,donor_candidates):corrected[i]=c
                add('expansion',serialize(corrected),rejected['raw_response'],rejected,
                    [rejected['response_id']]+[r for _,r in donor_candidates[:len(bad)]],'equal_count_validated_replacement')
            cleaned=[c for i,c in enumerate(cc) if i not in bad]
            add('expansion',serialize(cleaned),rejected['raw_response'],rejected,[rejected['response_id']],'delete_verified_harmful')
        for arm in STRATEGIES:
            # Round-robin across construction types avoids filling the cap with deletions only.
            choices=defaultdict(list)
            for pair in per_arm[arm]:choices[pair['construction']].append(pair)
            selected=[]
            while choices and len(selected)<cap:
                for kind in list(choices):
                    if len(selected)<cap:selected.append(choices[kind].pop(0))
                    if not choices[kind]:del choices[kind]
            output[model][arm][split].extend(selected)
    return output,dict(exclusions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('review','pairs'),required=True)
    parser.add_argument('--banks',type=Path,nargs=3)
    parser.add_argument('--annotations',type=Path)
    parser.add_argument('--review-dir',type=Path,help='Officially scored review package; required for pair construction')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=read(DEFAULT_CONFIG)
    official=None
    if args.mode=='review':
        if not args.banks:parser.error('--banks requires three completed model banks')
        banks,raw=pack_banks(args.banks,config)
    else:
        if not args.annotations or not args.review_dir:parser.error('--annotations and --review-dir required for pairs')
        sys.path.insert(0,str(ROOT.parent))
        from scripts.judge_max5_dpo_banks import validate_review
        review,official=validate_review(args.review_dir,config)
        banks,raw=review['banks'],records(args.review_dir/'responses.jsonl')
    args.output.mkdir(parents=True,exist_ok=False)
    write_jsonl(args.output/'responses.jsonl',raw)
    if args.mode=='review':
        rr=[review_template(r,config['max_new_tokens']) for r in raw]
        _,aliases=panels(config)
        contexts={r['question_id']:json.loads(r['prompt'][1]['content']) for r in raw}
        examples=[{'question_id':q,'question':c['question'],'gold_aliases':aliases[q]} for q,c in contexts.items()]
        candidates=[{'question_id':r['question_id'],'answer':c['answer']} for r in rr for c in r['candidates'] if not r['excluded_reason']]
        sys.path.insert(0,str(ROOT.parent))
        from src.notebook_workflows.coverage_comparison import official_candidate_matches
        report=args.output/'official-audit';report.mkdir()
        matches=official_candidate_matches(examples,candidates,report,
            jar_path=ROOT.parent/'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar') if candidates else {}
        for row in rr:
            if not row['excluded_reason']:
                for c in row['candidates']:c['official_accepted']=matches[(row['question_id'],c['answer'])]
        write_jsonl(args.output/'review.jsonl',rr)
        state={'status':'review','banks':banks,'configuration':config,'files_sha256':{
            f:digest(args.output/f) for f in ('responses.jsonl','review.jsonl')}}
    else:
        rr=records(args.annotations)
        by_id={r['response_id']:r for r in raw}
        if len(rr)!=len(raw) or {r['response_id'] for r in rr}!=set(by_id):raise ValueError('Missing/duplicate annotations')
        for row in rr:
            if any(row.get(k)!=v for k,v in by_id[row['response_id']].items()):raise ValueError('Annotation changed raw response')
        official_by_id={r['response_id']:r for r in official}
        if set(official_by_id)!=set(by_id):raise ValueError('Official review bank membership differs')
        for row in rr:
            reviewed=official_by_id[row['response_id']]
            if len(row['candidates'])!=len(reviewed['candidates']) or any(
                    c.get(k)!=a.get(k) for c,a in zip(row['candidates'],reviewed['candidates'])
                    for k in ('answer','relation_type','official_accepted')):
                raise ValueError('Annotations changed official labels or parsed candidate order')
        pairs,exclusions=build_pairs(rr,config['max_pairs_per_question_per_model'])
        write_jsonl(args.output/'annotations.jsonl',rr)
        counts,matched_counts={},{}
        for model in MODELS:
            for arm in STRATEGIES:
                counts[model+'_'+arm]={}
                for split,pp in pairs[model][arm].items():
                    path=args.output/f'{model}-{arm}-{split}.jsonl'
                    write_jsonl(path,pp)
                    counts[model+'_'+arm][split]=len(pp)
            matched_counts[model]={}
            for split in ('train','validation'):
                by_arm={arm:defaultdict(list) for arm in STRATEGIES}
                for arm in STRATEGIES:
                    for pair in pairs[model][arm][split]:by_arm[arm][pair['question_id']].append(pair)
                common=set(by_arm['initial']) & set(by_arm['expansion'])
                count=sum(min(len(by_arm[a][q]) for a in STRATEGIES) for q in common)
                matched_counts[model][split]={'questions':len(common),'pairs_per_arm':count}
                for arm in STRATEGIES:
                    matched=[p for q in sorted(common) for p in by_arm[arm][q][:min(len(by_arm[a][q]) for a in STRATEGIES)]]
                    write_jsonl(args.output/f'{model}-{arm}-matched-{split}.jsonl',matched)
        state={'status':'ready' if all(v[s] for v in counts.values() for s in ('train','validation')) else 'insufficient_pairs',
            'banks':banks,'configuration':config,'pair_counts':counts,'exclusions':exclusions,
            'matched_question_and_pair_counts':matched_counts,
            'construction_counts':dict(Counter(p['construction'] for m in pairs.values() for a in m.values() for rr in a.values() for p in rr)),
            'files_sha256':{p.name:digest(p) for p in args.output.iterdir() if p.is_file()},
            'label_source':'GPT-4.1 mini only; independent human audit pending',
            'dev160_training_questions':0,'cross_model_pairs':False}
    write(args.output/'manifest.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='banks'},indent=2))


if __name__=='__main__':main()
