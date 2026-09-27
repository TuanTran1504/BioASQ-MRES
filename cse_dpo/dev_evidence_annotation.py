"""Reusable dev-only alias occurrence and training-style sufficient evidence annotations.

No student predictions or model identities enter annotation. No SFT rows are emitted.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from cse_dpo import evidence_sft_data as prep, occurrence_sft_data as occ

VERSION = 'factoid-dev-snippet-evidence-v2'
# Share the training rubric; only the evaluation purpose and reference unit change.
JUDGE_PROMPT = (prep.JUDGE_PROMPT
    .replace('for supervised training', 'for reusable development evaluation')
    .replace('resource_3', '3')
    .replace('"resource_id": "3"', '"snippet_id": 3')
    .replace('resources', 'snippets').replace('resource', 'snippet')
    .replace('training example', 'confident evaluation label')
    .replace('Prior flags remain held unless a human explicitly resolves them.', '')
    .replace('training answers', 'new answers')
) + """
This request contains ONE alias: proposed_alias, identified by alias_id. Return
that alias as the only entry in aliases, using the training-style schema above.
Use snippets[] as the authoritative mapping of numeric snippet_id to exact text.
Resource numbers in numbered_context are NOT snippet IDs. Read the full context,
but cite snippet IDs from snippets[]. Never merge or renumber snippets.
Copy a short contiguous quote from the referenced snippets[].text, preserving
HTML and whitespace. Do not insert ellipses. Do not label every snippet: return
one or more sufficient evidence sets and leave uncited snippets unassessed.
"""


def build_packets(project_root, *, dev_path=None, train_path=None):
    root = Path(project_root)
    directory = root/'data/BioASQ_factoid_sft_prepared/single_answer_full_resources_qwen25_05b'
    dev_path = Path(dev_path) if dev_path is not None else directory/'eval_prepared.json'
    train_path = Path(train_path) if train_path is not None else directory/'train_prepared.json'
    dev, train = json.loads(dev_path.read_text(encoding='utf-8')), json.loads(train_path.read_text(encoding='utf-8'))
    dev_ids, train_ids = {r['id'] for r in dev}, {r['id'] for r in train}
    if len(dev_ids)!=len(dev) or len(train_ids)!=len(train):
        raise ValueError('Duplicate question IDs in source data')
    if dev_ids & train_ids:
        raise ValueError('Train/dev overlap: cannot build evaluation annotations')
    packets = []
    for raw in dev:
        packet = {'question_id':raw['id'], 'split':'dev', 'question':raw['input_1'],
                  'resources':prep.raw_resources(raw), 'gold_output':raw['output']}
        snippets, issues = occ.split_snippets(packet)
        if issues:
            raise ValueError(f"Malformed snippets for {raw['id']}: {issues}")
        aliases = []
        for i, alias in enumerate(prep.aliases_of(raw['output']),1):
            aliases.append({'alias_id':f'a{i}', 'text':alias,
                'literal_snippet_ids':[s['snippet_id'] for s in snippets if occ.contains_alias(alias,s['text'],'literal')],
                'normalized_snippet_ids':[s['snippet_id'] for s in snippets if occ.contains_alias(alias,s['text'],'normalized')]})
        if not aliases:
            raise ValueError(f"No accepted aliases for {raw['id']}")
        packet.update(snippets=snippets, aliases=aliases, numbered_context=occ.numbered_context(packet,snippets))
        packet['input_sha256'] = prep.digest(packet)
        packets.append(packet)
    manifest = {'version':VERSION,'split':'dev','intended_use':'evaluation_only',
                'question_count':len(packets),'alias_count':sum(len(p['aliases']) for p in packets),
                'snippet_count':sum(len(p['snippets']) for p in packets),
                'alias_snippet_pairs':sum(len(p['aliases'])*len(p['snippets']) for p in packets),
                'train_dev_disjoint':True,'source_hashes':{str(p.resolve()):prep.file_hash(p) for p in [dev_path,train_path]},
                'packets_sha256':prep.digest(packets),
                'numbering':'individual BS/ES snippets, original resource order; same as shared evidence-answer evaluation',
                'occurrence_policy':'literal case-sensitive token-bounded and normalized token-bounded; only snippet bodies',
                'semantic_support':'training-style sufficient evidence sets per alias; uncited snippets are unreviewed'}
    return packets, manifest


def _verify_packets(packets):
    if len({p['question_id'] for p in packets}) != len(packets):
        raise ValueError('Duplicate dev packet IDs')
    for p in packets:
        if p.get('split')!='dev' or prep.digest({k:v for k,v in p.items() if k!='input_sha256'})!=p['input_sha256']:
            raise ValueError('Not a frozen dev packet')


def initialize_run(run_dir, packets, manifest):
    _verify_packets(packets)
    if manifest.get('split')!='dev' or manifest.get('packets_sha256')!=prep.digest(packets):
        raise ValueError('Dev manifest does not match packets')
    prep.initialize_run(run_dir,packets,manifest)
    run_dir = Path(run_dir)
    prep.write_jsonl(run_dir/'snippet_index.jsonl',[
        {'question_id':p['question_id'],'split':'dev',**s} for p in packets for s in p['snippets']])
    prep.write_jsonl(run_dir/'alias_occurrences.jsonl',[
        {'question_id':p['question_id'],'split':'dev',**a,'semantic_support':'unreviewed'}
        for p in packets for a in p['aliases']])
    prep.write_jsonl(run_dir/'alias_snippet_occurrences.jsonl',[
        {'question_id':p['question_id'],'split':'dev','alias_id':a['alias_id'],'alias':a['text'],
         'snippet_id':s['snippet_id'],'literal_match':s['snippet_id'] in a['literal_snippet_ids'],
         'normalized_match':s['snippet_id'] in a['normalized_snippet_ids']}
        for p in packets for a in p['aliases'] for s in p['snippets']])


def tasks_for(packets):
    _verify_packets(packets)
    tasks = []
    for packet in packets:
        for alias in packet['aliases']:
            payload = {'question_id':packet['question_id'], 'alias_id':alias['alias_id'],
                       'proposed_alias':alias['text'],'numbered_context':packet['numbered_context'],
                       'snippet_ids':[s['snippet_id'] for s in packet['snippets']],
                       'snippets':[{'snippet_id':s['snippet_id'],'resource_id':s['resource_id'],'text':s['text']}
                                   for s in packet['snippets']],
                       'occurrence_hints':{k:alias[k] for k in ['literal_snippet_ids','normalized_snippet_ids']}}
            tasks.append({'task_id':packet['question_id']+'__'+alias['alias_id'],
                          'payload':payload,'input_sha256':prep.digest(payload),
                          'snippets':{s['snippet_id']:s['text'] for s in packet['snippets']}})
    return tasks


def normalize_decision(task, decision):
    """Remove redundant singleton sets only; never repair quotes, IDs or support labels."""
    result=copy.deepcopy(decision)
    changes=[]
    if not isinstance(result,dict) or not isinstance(result.get('snippet_labels'),list) or not isinstance(result.get('joint_evidence_sets'),list):
        return result,changes
    sufficient={v.get('snippet_id') for v in result['snippet_labels']
                if isinstance(v,dict) and type(v.get('snippet_id')) is int and v.get('support')=='sufficient'}
    kept=[]
    for group in result['joint_evidence_sets']:
        ids=group.get('snippet_ids') if isinstance(group,dict) else None
        if isinstance(ids,list) and len(ids)==1 and type(ids[0]) is int and ids[0] in sufficient and ids[0] in task['snippets']:
            changes.append({'type':'redundant_singleton_removed','original':group})
        else:
            kept.append(group)
    result['joint_evidence_sets']=kept
    return result,changes


def validate_decision(task, decision):
    if not isinstance(decision,dict):
        raise ValueError('Decision must be an object')
    if 'aliases' in decision:
        return validate_training_style_decision(task, decision)
    # Preserve validation of historical exhaustive annotations and human imports.
    for key in ['question_id','alias_id']:
        if decision.get(key)!=task['payload'][key]:
            raise ValueError('Wrong question/alias ID')
    for key,allowed in [('question_status',{'clear','ambiguous','conflicting'}),
                        ('answer_status',{'complete','incomplete','wrong','uncertain'}),
                        ('joint_sets_status',{'assessed','uncertain'})]:
        if decision.get(key) not in allowed:
            raise ValueError('Invalid '+key)
    for key in ['question_reason','answer_reason']:
        if not isinstance(decision.get(key),str) or not decision[key].strip():
            raise ValueError('Missing '+key)
    labels = decision.get('snippet_labels')
    if not isinstance(labels,list) or len(labels)!=len(task['snippets']):
        raise ValueError('Every snippet must receive a label')
    seen = {}
    for label in labels:
        if not isinstance(label,dict):
            raise ValueError('Invalid snippet label')
        sid=label.get('snippet_id')
        if type(sid) is not int or sid not in task['snippets'] or sid in seen:
            raise ValueError('Unknown/duplicate snippet ID')
        support=label.get('support')
        if support not in {'sufficient','partial','irrelevant','contradictory','uncertain'}:
            raise ValueError('Invalid support status')
        seen[sid]=support
        quote=label.get('quote')
        if not isinstance(quote,str) or (quote and quote not in task['snippets'][sid]):
            raise ValueError(f'Quote for snippet_id={sid} must occur exactly in snippets[{sid}].text; copy a contiguous substring, preserving HTML and whitespace. Do not use ellipses or another snippet/resource ID.')
        if support in {'sufficient','partial','contradictory'} and not quote.strip():
            raise ValueError('Supporting/contradictory labels need an exact quote')
        if not isinstance(label.get('reason'),str) or not label['reason'].strip():
            raise ValueError('Missing snippet rationale')
    sets=decision.get('joint_evidence_sets')
    if not isinstance(sets,list):
        raise ValueError('joint_evidence_sets must be a list')
    seen_sets=set()
    for group in sets:
        if not isinstance(group,dict):
            raise ValueError('Invalid joint set')
        ids=group.get('snippet_ids')
        if not isinstance(ids,list) or len(ids)<2 or any(type(i) is not int or i not in seen for i in ids):
            raise ValueError('Joint set needs at least two known snippet IDs')
        if len(set(ids))!=len(ids) or tuple(sorted(ids)) in seen_sets:
            raise ValueError('Duplicate joint set or member')
        seen_sets.add(tuple(sorted(ids)))
        if any(seen[i] not in {'sufficient','partial'} for i in ids):
            raise ValueError('Joint sets cannot contain irrelevant/contradictory/uncertain snippets')
        if not isinstance(group.get('reason'),str) or not group['reason'].strip():
            raise ValueError('Missing joint-set rationale')
    return decision


def validate_training_style_decision(task, decision):
    """Use the training validator with a lossless snippet-to-resource adapter."""
    aliases = decision.get('aliases')
    if not isinstance(aliases, list) or len(aliases) != 1 or not isinstance(aliases[0], dict):
        raise ValueError('Return exactly the proposed alias in aliases')
    if aliases[0].get('alias_id') != task['payload']['alias_id']:
        raise ValueError('Wrong alias ID')
    mapped = copy.deepcopy(decision)
    sets = mapped['aliases'][0].get('evidence_sets')
    if not isinstance(sets, list):
        raise ValueError('evidence_sets must be a list')
    seen_sets = set()
    for group in sets:
        if not isinstance(group, list) or not group:
            raise ValueError('Empty evidence set')
        ids = []
        for ref in group:
            if not isinstance(ref, dict):
                raise ValueError('Evidence reference must be an object')
            sid = ref.get('snippet_id')
            if type(sid) is not int or sid not in task['snippets'] or sid in ids:
                raise ValueError('Unknown/duplicate snippet ID')
            ids.append(sid)
            quote = ref.get('quote')
            if not isinstance(quote, str) or not quote.strip() or quote not in task['snippets'][sid]:
                raise ValueError(f'Quote for snippet_id={sid} must occur exactly in that snippets[].text; copy a contiguous substring preserving HTML and whitespace')
            ref['resource_id'] = sid
        key = tuple(sorted(ids))
        if key in seen_sets:
            raise ValueError('Duplicate evidence set')
        seen_sets.add(key)
    packet = {'question_id':task['payload']['question_id'],
              'aliases':[{'alias_id':task['payload']['alias_id']}],
              'resources':[{'resource_id':sid, 'text':text} for sid,text in task['snippets'].items()]}
    prep.validate_decision(packet, mapped)
    return decision


def export_decision_view(decision):
    """Adapt selected evidence sets to export columns without inventing negatives."""
    if not decision or 'aliases' not in decision:
        return decision
    alias = decision['aliases'][0]
    singletons = {group[0]['snippet_id'] for group in alias['evidence_sets'] if len(group) == 1}
    labels = {}
    joint = []
    for group in alias['evidence_sets']:
        for ref in group:
            sid = ref['snippet_id']
            labels[sid] = {'snippet_id':sid, 'support':'sufficient' if sid in singletons else 'partial',
                           'quote':ref['quote'], 'reason':ref['reason']}
        if len(group) > 1:
            joint.append({'snippet_ids':[ref['snippet_id'] for ref in group],
                          'reason':' '.join(ref['reason'] for ref in group)})
    return {'question_status':decision['question_status'], 'question_reason':decision['question_reason'],
            'answer_status':alias['answer_status'], 'answer_reason':alias['rationale'],
            'evidence_status':alias['evidence_status'], 'snippet_labels':list(labels.values()),
            'joint_sets_status':'uncertain', 'joint_evidence_sets':joint}


def exact_match_fallback(task):
    alias=task['payload']['proposed_alias']
    return {'match_mode':'literal_token_bounded', 'semantic_support':'unreviewed',
            'matching_snippet_ids':[sid for sid,text in task['snippets'].items()
                                    if occ.contains_alias(alias,text,'literal')]}


def validate_record(task, record):
    if record.get('task_id')!=task['task_id'] or record.get('input_sha256')!=task['input_sha256']:
        raise ValueError('Invalid annotation fingerprint')
    origin=record.get('origin')
    if origin=='exact_match_fallback':
        if record.get('decision') is not None or record.get('fallback')!=exact_match_fallback(task):
            raise ValueError('Invalid exact-match fallback; must not claim semantic support')
        if len(record.get('validation_attempts',[]))!=3:
            raise ValueError('Fallback requires an initial failure and two failed retries')
    elif origin in {'model','human'}:
        if origin=='human' and (not record.get('reviewer') or not record.get('review_reason')):
            raise ValueError('Missing human review provenance')
        validate_decision(task,record['decision'])
    else:
        raise ValueError('Invalid annotation provenance')


def annotate(packets, run_dir, judge, judge_metadata, *, limit=None, smoke_questions=None, continue_on_invalid=True, validation_retries=0, fallback_to_exact_match=False):
    if limit is not None and (type(limit) is not int or limit<=0):
        raise ValueError('limit must be a positive integer or None')
    if smoke_questions is not None and (type(smoke_questions) is not int or smoke_questions<=0):
        raise ValueError('smoke_questions must be a positive integer or None')
    if type(validation_retries) is not int or not 0<=validation_retries<=2:
        raise ValueError('validation_retries must be 0, 1 or 2')
    if fallback_to_exact_match and validation_retries!=2:
        raise ValueError('Exact-match fallback requires validation_retries=2')
    tasks=tasks_for(packets if smoke_questions is None else packets[:smoke_questions])
    signature=prep.digest({'version':VERSION,'prompt':JUDGE_PROMPT,'judge':judge_metadata})
    cache=Path(run_dir)/'judgments'/signature
    cache.mkdir(parents=True,exist_ok=True)
    prep.write_json(cache/'judge_manifest.json',{'version':VERSION,'prompt':JUDGE_PROMPT,'judge':judge_metadata})
    attempted=saved=invalid=fallback_count=0
    for task in tasks:
        path=cache/(task['task_id']+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            validate_record(task,record)
            continue
        if limit is not None and attempted>=limit:
            break
        failures=[];decision=None
        if fallback_to_exact_match and path.with_suffix('.error.json').exists():
            previous=json.loads(path.with_suffix('.error.json').read_text())
            if previous.get('input_sha256')!=task['input_sha256']:
                raise ValueError('Saved validation failures belong to different inputs')
            failures=previous.get('attempts',[])
            if not isinstance(failures,list) or len(failures)>3:
                raise ValueError('Invalid saved validation-attempt history')
            if any(not isinstance(f,dict) or not {'raw','metadata','validation_error'}<=f.keys() for f in failures):
                raise ValueError('Incomplete saved validation-attempt history')
        max_attempts=1+validation_retries
        for attempt in range(max_attempts-len(failures)):
            if limit is not None and attempted>=limit:
                break
            payload=task['payload']
            if failures:
                payload={**payload,'validation_feedback':failures[-1]['validation_error'],
                         'previous_invalid_response':failures[-1]['raw'],
                         'correction_request':'Return the complete corrected training-style JSON for the proposed alias, with aliases and evidence_sets. Cite authoritative snippet IDs and exact contiguous quotes. Uncited snippets need no labels; accept equivalent wording.'}
            attempted+=1
            # Network/provider failures stop, preserving all prior successful calls.
            raw,metadata=judge(payload)
            try:
                if metadata.get('finish_reason') not in {None,'stop'}:
                    raise ValueError('Incomplete judge response')
                candidate,normalizations=normalize_decision(task,json.loads(raw) if isinstance(raw,str) else raw)
                decision=validate_decision(task,candidate)
                break
            except (ValueError,TypeError,KeyError) as exc:
                failures.append({'validation_error':str(exc),'raw':raw,'metadata':metadata})
                prep.write_json(path.with_suffix('.error.json'),{'task_id':task['task_id'],'input_sha256':task['input_sha256'],
                                **failures[-1],'attempts':failures})
                can_retry=len(failures)<max_attempts and (limit is None or attempted<limit)
                will_fallback=fallback_to_exact_match and len(failures)==3
                outcome='Retrying with feedback.' if can_retry else 'Using exact-match fallback.' if will_fallback else 'Left pending.'
                print(f"Invalid dev annotation: {task['task_id']}: {exc}. "+outcome,flush=True)
                if not can_retry and not continue_on_invalid and not will_fallback:
                    raise
        if decision is None:
            if fallback_to_exact_match and len(failures)==3:
                record={'task_id':task['task_id'],'input_sha256':task['input_sha256'],
                        'origin':'exact_match_fallback','decision':None,'fallback':exact_match_fallback(task),
                        'validation_attempts':failures,'judge_metadata':judge_metadata,
                        'judge_prompt_sha256':prep.digest(JUDGE_PROMPT)}
                validate_record(task,record)
                prep.write_json(path,record)
                fallback_count+=1
                print(f"Exact-match fallback saved: {task['task_id']}; matches={record['fallback']['matching_snippet_ids']}",flush=True)
                continue
            invalid+=1
            continue
        prep.write_json(path,{'task_id':task['task_id'],'input_sha256':task['input_sha256'],
                        'origin':'model','decision':decision,'metadata':metadata,
                        'raw':raw,'normalizations':normalizations,'validation_attempts':failures,
                        'judge_metadata':judge_metadata,'judge_prompt_sha256':prep.digest(JUDGE_PROMPT)})
        path.with_suffix('.error.json').unlink(missing_ok=True)
        saved+=1
        print(f"Dev alias {saved} saved: {task['task_id']}",flush=True)
    print(f'Dev pass: {attempted} calls, {saved} semantic annotations, {fallback_count} exact-match fallbacks, {invalid} invalid/pending',flush=True)
    return cache


def load_decisions(packets, cache_dir=None, manual_reviews=None):
    tasks={t['task_id']:t for t in tasks_for(packets)}
    results={}
    if cache_dir is not None:
        for task_id,task in tasks.items():
            path=Path(cache_dir)/(task_id+'.json')
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('origin') not in {'model','exact_match_fallback'}:
                    raise ValueError('Invalid cache provenance')
                validate_record(task,record)
                results[task_id]=record
    if manual_reviews is not None and Path(manual_reviews).exists():
        seen=set()
        for record in prep.read_jsonl(manual_reviews):
            task_id=record.get('task_id')
            if task_id not in tasks or task_id in seen:
                raise ValueError('Unknown/duplicate manual task')
            seen.add(task_id)
            if record.get('input_sha256')!=tasks[task_id]['input_sha256']:
                raise ValueError('Manual task fingerprint mismatch')
            if record.get('origin')!='human' or not record.get('reviewer') or not record.get('review_reason'):
                raise ValueError('Manual decisions need human reviewer and reason')
            validate_decision(tasks[task_id],record['decision'])
            results[task_id]=record
    return results


def export_annotations(packets, manifest, decisions, output_dir, *, allow_partial=False, occurrence_only=False):
    _verify_packets(packets)
    tasks={t['task_id']:t for t in tasks_for(packets)}
    if manifest.get('split')!='dev' or manifest.get('packets_sha256')!=prep.digest(packets):
        raise ValueError('Dev source manifest mismatch')
    for path,sha in manifest['source_hashes'].items():
        if prep.file_hash(path)!=sha:
            raise ValueError('Annotation source changed')
    if occurrence_only and decisions:
        raise ValueError('Occurrence-only export cannot contain semantic judgments')
    if decisions.keys()-tasks.keys():
        raise ValueError('Unknown annotation task')
    for task_id,item in decisions.items():
        validate_record(tasks[task_id],item)
    pending=len(tasks)-len(decisions)
    fallbacks=sum(item['origin']=='exact_match_fallback' for item in decisions.values())
    semantic_count=len(decisions)-fallbacks
    if pending and not allow_partial and not occurrence_only:
        raise ValueError(f'{pending} dev aliases remain unannotated; finish annotation or explicitly allow partial export')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    alias_rows=[];pair_rows=[];question_rows=[];review=[]
    for packet in packets:
        statuses=[]
        question_needs_review=any(
            (decisions.get(packet['question_id']+'__'+a['alias_id'],{}).get('decision') or {}).get('question_status')
            in {'ambiguous','conflicting'} for a in packet['aliases'])
        for alias in packet['aliases']:
            task_id=packet['question_id']+'__'+alias['alias_id']
            item=decisions.get(task_id);decision=export_decision_view(item['decision']) if item else None
            fallback=item.get('fallback') if item and item['origin']=='exact_match_fallback' else None
            labels={v['snippet_id']:v for v in decision['snippet_labels']} if decision else {}
            singleton=[[sid] for sid,label in labels.items() if label['support']=='sufficient']
            joint=[sorted(v['snippet_ids']) for v in decision['joint_evidence_sets']] if decision else []
            eligible=bool(decision and not question_needs_review and decision['question_status']=='clear' and decision['answer_status']=='complete')
            support_values={label['support'] for label in labels.values()}
            evidence_status=('unreviewed' if not decision else 'conflicting' if 'contradictory' in support_values
                else 'supported' if singleton or joint else 'uncertain' if 'uncertain' in support_values or decision['joint_sets_status']=='uncertain'
                else 'partial' if 'partial' in support_values else 'unsupported')
            if decision and 'evidence_status' in decision:
                evidence_status=decision['evidence_status']
            eligible=eligible and evidence_status=='supported'
            row={'question_id':packet['question_id'],'split':'dev','task_id':task_id,
                 'alias_id':alias['alias_id'],'alias':alias['text'],
                 'literal_snippet_ids':alias['literal_snippet_ids'],'normalized_snippet_ids':alias['normalized_snippet_ids'],
                 'question_status':decision['question_status'] if decision else 'unreviewed',
                 'answer_status':decision['answer_status'] if decision else 'unreviewed',
                 'evidence_status':evidence_status,'question_needs_review':question_needs_review,
                 'joint_sets_status':decision['joint_sets_status'] if decision else 'unreviewed',
                 'eligible_answer_for_evidence_scoring':eligible,
                 'sufficient_evidence_sets':singleton+joint,
                 'supporting_snippet_ids':sorted(sid for sid,label in labels.items() if label['support'] in {'sufficient','partial'}),
                 'uncertain_snippet_ids':sorted(sid for sid,label in labels.items() if label['support']=='uncertain'),
                 'unreviewed_snippet_ids':sorted(sid for sid in tasks[task_id]['snippets'] if sid not in labels),
                 'origin':item['origin'] if item else 'none','input_sha256':tasks[task_id]['input_sha256']}
            row.update(annotation_complete=item is not None,semantic_annotation_complete=decision is not None,
                       fallback_match_snippet_ids=fallback['matching_snippet_ids'] if fallback else [],
                       fallback_evidence_sets=[[sid] for sid in fallback['matching_snippet_ids']] if fallback else [],
                       fallback_sets_are_semantically_verified=False if fallback else None)
            alias_rows.append(row);statuses.append(row['question_status'])
            if not eligible or row['uncertain_snippet_ids'] or row['joint_sets_status']=='uncertain':
                review.append(row)
            for snippet in packet['snippets']:
                sid=snippet['snippet_id'];label=labels.get(sid)
                pair_rows.append({'question_id':packet['question_id'],'split':'dev','alias_id':alias['alias_id'],
                    'alias':alias['text'],'snippet_id':sid,
                    'literal_match':sid in alias['literal_snippet_ids'],'normalized_match':sid in alias['normalized_snippet_ids'],
                    'support':label['support'] if label else 'unreviewed',
                    'fallback_occurrence_label':('match' if sid in fallback['matching_snippet_ids'] else 'no_match') if fallback else None,
                    'quote':label['quote'] if label else '', 'reason':label['reason'] if label else 'Not annotated',
                    'origin':item['origin'] if item else 'none'})
        question_rows.append({'question_id':packet['question_id'],'split':'dev','question':packet['question'],
                              'aliases_annotated':sum(s!='unreviewed' for s in statuses),'alias_count':len(statuses),
                              'annotation_complete':all(packet['question_id']+'__'+a['alias_id'] in decisions for a in packet['aliases']),
                              'semantic_annotation_complete':'unreviewed' not in statuses,
                              'question_status_by_alias':statuses,
                              'needs_question_review':any(s in {'ambiguous','conflicting'} for s in statuses)})
    files={'dev_packets.jsonl':packets,'dev_alias_labels.jsonl':alias_rows,'dev_snippet_alias_labels.jsonl':pair_rows,
           'dev_question_labels.jsonl':question_rows,'dev_review_queue.jsonl':review,
           'dev_annotation_provenance.jsonl':list(decisions.values()),
           'dev_snippet_index.jsonl':[{'question_id':p['question_id'],'split':'dev',**s} for p in packets for s in p['snippets']]}
    for name,rows in files.items():prep.write_jsonl(out/name,rows)
    summary={'version':VERSION,'split':'dev','intended_use':'evaluation_only',
             'question_count':len(packets),'alias_count':len(tasks),'alias_snippet_pairs':len(pair_rows),
             'annotated_aliases':semantic_count,'completed_aliases':len(decisions),'fallback_aliases':fallbacks,'pending_aliases':pending,
             'semantic_pending_aliases':len(tasks)-semantic_count,
             'fully_annotated_questions':sum(q['semantic_annotation_complete'] for q in question_rows),
             'fully_processed_questions':sum(q['annotation_complete'] for q in question_rows),
             'occurrence_labels_complete':True,'annotation_complete':not pending,
             'semantic_alias_annotations_complete':not pending and not fallbacks,
             'semantic_labels_complete':not pending and not fallbacks and all(r['support']!='unreviewed' for r in pair_rows),
             'mode':'occurrence_only' if occurrence_only else 'semantic_and_occurrence',
             'support_counts':dict(Counter(r['support'] for r in pair_rows)),
             'alias_evidence_status_counts':dict(Counter(r['evidence_status'] for r in alias_rows)),
             'judge_prompt_sha256':prep.digest(JUDGE_PROMPT),
             'literal_match_pairs':sum(r['literal_match'] for r in pair_rows),
             'normalized_match_pairs':sum(r['normalized_match'] for r in pair_rows),
             'all_human_reviewed':bool(decisions) and not pending and all(d['origin']=='human' for d in decisions.values()),
             'alternative_set_policy':'selected sufficient sets only; unlisted snippets and alternative sets are unreviewed, not negative; joint members are not automatically sufficient alone',
             'source_manifest':manifest,'created_utc':datetime.now(timezone.utc).isoformat(),
             'output_hashes':{name:prep.file_hash(out/name) for name in files}}
    prep.write_json(out/'summary.json',summary)
    return summary
