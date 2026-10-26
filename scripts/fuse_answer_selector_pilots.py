#!/usr/bin/env python3
"""CPU-only fusion of frozen selector validation rankings; no fitting or DPO."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import answer_selector_core as core
from scripts import prepare_answer_selector_data as data
from scripts import run_answer_selector_pilot as pilot

MODELS = tuple(pilot.MODELS)
POOLS = pilot.POOLS
DATA_FILES = ('train_inputs.jsonl', 'train_supervision.jsonl', 'validation_inputs.jsonl',
              'validation_supervision.jsonl', 'heldout_ids.json')
METRICS = ('mrr_at5', 'coverage_at1', 'coverage_at5', 'coverage_full_pool')
POLICY = {
    'input_depth_per_system': 5, 'output_depth': 5, 'weights': 'equal; no tuning',
    'model_vote': 'vote count, then sum(1/rank), then label-blind SHA256 tie key',
    'rank_vote': 'sum(1/rank), then vote count, then label-blind SHA256 tie key',
    'unlisted_candidates': 'no contribution; never fill output from outside the input union',
    'pool_identity': 'shared source ID with exact answer surface; no semantic/normalized merging',
}


def keyed(rows):
    result = {(r['question_id'], r['pool']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate question/pool view')
    return result


def fuse(rankings, surfaces, question_id, pool, method):
    """Gold-blind fusion. Each system supplies at most five distinct source IDs."""
    if method not in ('model_vote', 'rank_vote') or not rankings:
        raise ValueError('Unknown fusion rule or empty ensemble')
    votes, reciprocal = Counter(), {}
    for ranking in rankings.values():
        if (len(ranking) > 5 or len(set(ranking)) != len(ranking)
                or not set(ranking) <= set(surfaces)):
            raise ValueError('Invalid, duplicate or unknown fusion input ID')
        for rank, identity in enumerate(ranking, 1):
            votes[identity] += 1
            reciprocal[identity] = reciprocal.get(identity, Fraction()) + Fraction(1, rank)
    def order(identity):
        tie = hashlib.sha256(json.dumps([3407, question_id, pool, surfaces[identity]],
            ensure_ascii=False).encode('utf-8')).hexdigest()
        primary = (-votes[identity], -reciprocal[identity]) if method == 'model_vote' else (
            -reciprocal[identity], -votes[identity])
        return (*primary, tie)
    full = sorted(votes, key=order)
    return full[:5], full, {c: {'votes': votes[c], 'rank_vote': float(reciprocal[c])} for c in full}


def check_scores(predictions, labels, saved, expected_metrics=None):
    summary, scores = core.score_predictions(predictions, labels)
    for pool in POOLS:
        a = {r['question_id']: r for r in scores[pool]}
        old = saved[pool]
        if len(old) != len(a) or {r['question_id'] for r in old} != set(a):
            raise ValueError('Saved score panel differs')
        for row in old:
            for metric in METRICS:
                if not math.isclose(row[metric], a[row['question_id']][metric], abs_tol=1e-12):
                    raise ValueError('Saved per-question score disagrees with ranked IDs: ' + metric)
        if expected_metrics is not None:
            for metric in METRICS:
                if not math.isclose(summary['conditions'][pool][metric], expected_metrics[pool][metric], abs_tol=1e-12):
                    raise ValueError('Saved aggregate score disagrees with ranked IDs')
    return summary, scores


def logistic_predictions(path, labels):
    views = keyed(labels)
    groups = {k: [] for k in views}
    for row in data.records(path):
        pool = row['condition'].removesuffix('_logistic')
        key = (row['question_id'], pool)
        if key not in groups or row['condition'] != pool + '_logistic':
            raise ValueError('Unexpected logistic ranking panel')
        groups[key].append(row)
    predicted = []
    for key, rows in groups.items():
        candidates = views[key]['candidates']
        by_surface = {c['answer']: c for c in candidates}
        if len(by_surface) != len(candidates):
            raise ValueError('Duplicate exact answer surface')
        rows.sort(key=lambda r: r['rank'])
        if ([r['rank'] for r in rows] != list(range(1, len(candidates) + 1))
                or len(rows) != len(candidates) or {r['answer'] for r in rows} != set(by_surface)):
            raise ValueError('Logistic must contain each original candidate exactly once')
        for r in rows:
            if type(r['label']) not in (int, bool) or r['label'] not in (0, 1) or bool(r['label']) != by_surface[r['answer']]['official_accepted']:
                raise ValueError('Logistic candidate label differs')
        predicted.append({'question_id': key[0], 'pool': key[1],
            'selected_source_ids': [by_surface[r['answer']]['id'] for r in rows[:5]]})
    return predicted


def load_sources(comparison_path):
    comparison = data.read(comparison_path)
    if (comparison.get('question_count') != data.COUNTS['validation']
            or comparison.get('test_evaluation') is not False or comparison.get('DPO_training') is not False
            or set(comparison.get('sources', {})) != set(MODELS)):
        raise ValueError('Need the completed three-selector validation comparison')
    predictions, provenance = {}, {}
    labels, inputs, reference_hashes, configuration = None, None, None, None
    common_logistic = None
    for model in MODELS:
        source = comparison['sources'][model]
        submission = Path(source['submission']).resolve()
        state = pilot.validate(submission)
        if state['configuration']['model_key'] != model or data.digest(submission / 'prepared.json') != source['prepared_sha256']:
            raise ValueError('Frozen selector source differs')
        shared = {k: v for k, v in state['configuration'].items() if k != 'model_key'}
        files = {name: data.digest(submission / name) for name in DATA_FILES}
        if (files != comparison['shared_data_sha256'] or
                (reference_hashes is not None and (files != reference_hashes or shared != configuration))):
            raise ValueError('Candidate pools, labels or common settings differ')
        reference_hashes, configuration = files, shared
        if labels is None:
            labels = data.records(submission / 'validation_supervision.jsonl')
            inputs = data.records(submission / 'validation_inputs.jsonl')
            views = keyed(labels)
            if len(views) != 2 * data.COUNTS['validation'] or set(views) != set(keyed(inputs)):
                raise ValueError('Incomplete validation input panel')
            heldout = data.read(submission / 'heldout_ids.json')
            excluded = set(heldout['dev160'] + heldout['test95'])
            panels = [{r['question_id'] for r in labels if r['pool'] == p} for p in POOLS]
            if (panels[0] != panels[1] or len(panels[0]) != data.COUNTS['validation']
                    or panels[0] & excluded or any(r['split'] != 'validation' for r in labels + inputs)):
                raise ValueError('Wrong validation or held-out membership')
            for key, item in keyed(inputs).items():
                payload = json.loads(item['messages'][1]['content'])
                if payload['candidates'] != [{'id': c['id'], 'answer': c['answer']} for c in views[key]['candidates']]:
                    raise ValueError('Input candidate surfaces differ from supervision')
        with (submission / 'jobs.tsv').open() as handle:
            jobs = {r['stage']: r['job_id'] for r in csv.DictReader(handle, delimiter='\t')}
        if any('/' in v or '\\' in v or '..' in v for v in jobs.values()):
            raise ValueError('Invalid job record path')
        training, report, baseline = (submission / (stage + '-' + jobs[stage]) for stage in ('train', 'report', 'baseline'))
        if report != Path(source['report']).resolve():
            raise ValueError('Recorded report path differs')
        pin = source['prepared_sha256']
        pilot.verify_result(training, pin, smoke=False)
        pilot.verify_result(report, pin)
        pilot.verify_result(baseline, pin)
        selection = data.read(training / 'selection.json')
        epoch = selection['selected_epoch']
        report_summary = data.read(report / 'comparison_summary.json')
        if (epoch != source['selected_epoch'] or epoch != report_summary['selected_epoch']
                or epoch != core.choose_epoch(selection['checkpoint_scores'])):
            raise ValueError('Selected checkpoint differs')
        directory = training / f'epoch-{epoch}'
        predicted = data.records(directory / 'validation_predictions.jsonl')
        # Validate the model-specific display-ID mapping, then fuse only source IDs.
        encoded = keyed(data.records(submission / 'validation_generation_inputs.jsonl'))
        if set(encoded) != set(views):
            raise ValueError('Encoded validation panel differs')
        for row in predicted:
            key = (row['question_id'], row['pool'])
            if key not in encoded or key not in views:
                raise ValueError('Unexpected selector prediction panel')
            mapping = encoded[key]['code_to_source_id']
            if len(mapping) != len(views[key]['candidates']) or set(mapping.values()) != {c['id'] for c in views[key]['candidates']}:
                raise ValueError('Display mapping omits original candidates')
            if any(c not in mapping for c in row['selected_codes']) or [mapping[c] for c in row['selected_codes']] != row['selected_source_ids']:
                raise ValueError('Display IDs and source IDs disagree')
        summary, _ = check_scores(predicted, labels, data.read(directory / 'validation_per_question.json'),
                                 report_summary['conditions']['selector'])
        if not math.isclose(summary['mean_validation_mrr_at5'], comparison['mean_validation_mrr_at5'][model], abs_tol=1e-12):
            raise ValueError('Previously compared selector score differs')
        predictions[model] = keyed(predicted)
        logistic = logistic_predictions(baseline / 'rankings.jsonl', labels)
        check_scores(logistic, labels, data.read(baseline / 'per_question.json'), data.read(baseline / 'summary.json')['conditions'])
        logistic = keyed(logistic)
        if common_logistic is not None and logistic != common_logistic:
            raise ValueError('Logistic baselines have different top-five rankings')
        common_logistic = logistic
        provenance[model] = {**source, 'training_manifest_sha256': data.digest(training / 'manifest.json'),
            'report_manifest_sha256': data.digest(report / 'manifest.json'),
            'baseline_manifest_sha256': data.digest(baseline / 'manifest.json')}
        print('Verified saved validation rankings:', model, flush=True)
    predictions['logistic'] = common_logistic
    return comparison, labels, inputs, predictions, provenance


def transitions(before, after):
    a, b = ({r['question_id']: r for r in rows} for rows in (before, after))
    if set(a) != set(b):
        raise ValueError('Transition panel differs')
    gained = sorted(q for q in a if not a[q]['coverage_at5'] and b[q]['coverage_at5'])
    lost = sorted(q for q in a if a[q]['coverage_at5'] and not b[q]['coverage_at5'])
    improved = sorted(q for q in a if b[q]['mrr_at5'] > a[q]['mrr_at5'])
    worsened = sorted(q for q in a if b[q]['mrr_at5'] < a[q]['mrr_at5'])
    return {'gained': len(gained), 'lost': len(lost), 'gained_question_ids': gained,
        'lost_question_ids': lost, 'mrr_improved': len(improved), 'mrr_worsened': len(worsened),
        'mrr_improved_question_ids': improved, 'mrr_worsened_question_ids': worsened}


def analyze(comparison_path, output, resamples=10000):
    if not 100 <= resamples <= 100000:
        raise ValueError('Need 100..100000 paired resamples')
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'comparison_sha256': data.digest(comparison_path),
              'DPO_training': False, 'test_evaluation': False, 'new_generation': False}
    data.write(output / 'manifest.json', status)
    try:
        comparison, labels, inputs, expert, provenance = load_sources(comparison_path)
        views = keyed(labels)
        conditions, scores, rankings = {}, {}, dict(expert)
        for model in expert:
            summary, scores[model] = core.score_predictions(list(expert[model].values()), labels)
            conditions[model] = summary['conditions']
        best = max(MODELS, key=lambda m: sum(conditions[m][p]['mrr_at5'] for p in POOLS))
        fusion_names, details = [], {}
        for group, members in (('selectors3', MODELS), ('selectors3_logistic', (*MODELS, 'logistic'))):
            for method in ('model_vote', 'rank_vote'):
                name = group + '_' + method
                fusion_names.append(name)
                predictions, rankings[name], details[name] = [], {}, {}
                for key, row in views.items():
                    surfaces = {c['id']: c['answer'] for c in row['candidates']}
                    votes = {m: expert[m][key]['selected_source_ids'] for m in members}
                    chosen, union, support = fuse(votes, surfaces, *key, method)
                    prediction = {'question_id': key[0], 'pool': key[1], 'selected_source_ids': chosen}
                    predictions.append(prediction); rankings[name][key] = prediction
                    accepted = {c['id'] for c in row['candidates'] if c['official_accepted']}
                    details[name][key] = {'input_union': union, 'input_union_covered': bool(accepted & set(union)), 'support': support}
                summary, scores[name] = core.score_predictions(predictions, labels)
                conditions[name] = summary['conditions']
                for p in POOLS:
                    conditions[name][p]['coverage_input_union'] = sum(d['input_union_covered'] for k,d in details[name].items() if k[1] == p) / data.COUNTS['validation']
        from scripts.analyze_matched_qwen_expansion import paired_contrast
        contrasts, changes = {}, {}
        for name in fusion_names:
            for baseline in dict.fromkeys(('logistic', best)):
                for pool in POOLS:
                    key = f'{name}_minus_{baseline}_{pool}'
                    c = paired_contrast(scores[baseline][pool], scores[name][pool], resamples=resamples,
                                        metrics=('mrr_at5', 'coverage_at5'))
                    contrasts[key] = {metric: {'difference': value['sft_minus_base'], **{k:v for k,v in value.items() if k != 'sft_minus_base'}} for metric,value in c.items()}
                    changes[key] = transitions(scores[baseline][pool], scores[name][pool])
            print('Scored fusion and paired comparisons:', name, flush=True)
        complementarity, audit = {}, []
        score_maps = {name: {(r['question_id'],p):r for p,rr in panels.items() for r in rr} for name,panels in scores.items()}
        for pool in POOLS:
            panel = [k for k in views if k[1] == pool]
            hit = lambda m,k: bool(score_maps[m][k]['coverage_at5'])
            any_hit = lambda k: any(hit(m,k) for m in MODELS)
            complementarity[pool] = {
                'source_full_pool_covered': int(sum(score_maps['logistic'][k]['coverage_full_pool'] for k in panel)),
                'three_selector_union_covered': sum(any_hit(k) for k in panel),
                'four_system_union_covered': sum(any_hit(k) or hit('logistic',k) for k in panel),
                'selector_hit_count_histogram': {str(i):sum(sum(hit(m,k) for m in MODELS)==i for k in panel) for i in range(4)},
                'unique_selector_hits': {m:[k[0] for k in panel if hit(m,k) and not any(hit(other,k) for other in MODELS if other != m)] for m in MODELS},
                'selectors_cover_logistic_misses': [k[0] for k in panel if any_hit(k) and not hit('logistic',k)],
                'logistic_covers_all_selector_misses': [k[0] for k in panel if hit('logistic',k) and not any_hit(k)],
                'source_covered_but_all_four_miss': [k[0] for k in panel if score_maps['logistic'][k]['coverage_full_pool'] and not any_hit(k) and not hit('logistic',k)],
            }
        by_input = keyed(inputs)
        for key,row in views.items():
            candidates = {c['id']: {'id':c['id'], 'answer':c['answer'], 'official_accepted':c['official_accepted']} for c in row['candidates']}
            comparisons = {f'{name}_minus_{base}': {
                'mrr_difference': score_maps[name][key]['mrr_at5'] - score_maps[base][key]['mrr_at5'],
                'top5_gained': not score_maps[base][key]['coverage_at5'] and bool(score_maps[name][key]['coverage_at5']),
                'top5_lost': bool(score_maps[base][key]['coverage_at5']) and not score_maps[name][key]['coverage_at5']}
                for name in fusion_names for base in dict.fromkeys(('logistic',best))}
            audit.append({'question_id':key[0], 'pool':key[1], 'split':'validation',
                'context':json.loads(by_input[key]['messages'][1]['content']), 'gold_aliases':row['gold_aliases'],
                'candidates':list(candidates.values()),
                'top_five': {name:[candidates[c] for c in r[key]['selected_source_ids']] for name,r in rankings.items()},
                'per_condition_scores':{name:r[key] for name,r in score_maps.items()},
                'fusion_support':{name:details[name][key] for name in fusion_names},
                'comparisons':comparisons, 'DPO_training_eligible':False,
                'semantic_review':'Unresolved; official nonacceptance does not establish medical incorrectness',
                'reason':'Validation diagnostic only; never use these examples as fitting preferences'})
        result = {'question_count': data.COUNTS['validation'], 'pools':list(POOLS), 'fusion_policy':POLICY,
            'conditions':conditions, 'best_individual_selector':best, 'paired_contrasts':contrasts,
            'mean_validation_mrr_at5':{name:sum(v[p]['mrr_at5'] for p in POOLS)/len(POOLS) for name,v in conditions.items()},
            'coverage_transitions':changes, 'complementarity':complementarity, 'sources':provenance,
            'comparison_source':str(comparison_path), 'comparison_sha256':data.digest(comparison_path),
            'DPO_pairs_created':0, 'test_evaluation':False, 'new_generation':False,
            'interpretation':'Internal validation diagnostics after checkpoint/backbone selection; CIs are descriptive; union coverage is an oracle bound, not a deployable score'}
        data.write(output/'comparison_summary.json', result)
        data.write(output/'comparison_per_question.json', scores)
        data.jsonl(output/'question_audit.jsonl', audit)
        data.jsonl(output/'changed_cases.jsonl', [r for r in audit if any(c['mrr_difference'] for c in r['comparisons'].values())])
        lines = ['# Saved-selector fusion: validation only', '', 'Equal-weight top-five inputs from every system; original candidate pools unchanged.',
                 'Rank voting here means sum(1/rank), without a smoothing constant.', '']
        for pool in POOLS:
            lines += [f'## {pool}', '', '| Condition | MRR@5 | Top-five hits | Gains / losses vs logistic |', '|---|---:|---:|---:|']
            for name in conditions:
                metric = conditions[name][pool]
                transition = transitions(scores['logistic'][pool],scores[name][pool])
                lines.append(f"| {name} | {metric['mrr_at5']:.6f} | {round(metric['coverage_at5']*data.COUNTS['validation'])}/{data.COUNTS['validation']} | {transition['gained']} / {transition['lost']} |")
            c = complementarity[pool]
            lines += ['', f"Accepted answer available in source pool: {int(c['source_full_pool_covered'])}/{data.COUNTS['validation']}.",
                f"Accepted answer somewhere in three selector lists: {c['three_selector_union_covered']}/{data.COUNTS['validation']}; including logistic: {c['four_system_union_covered']}/{data.COUNTS['validation']}.",
                f"Three-selector union covers {len(c['selectors_cover_logistic_misses'])} logistic misses; logistic covers {len(c['logistic_covers_all_selector_misses'])} questions all three selectors miss.", '']
            for name in fusion_names:
                c = contrasts[f'{name}_minus_logistic_{pool}']['mrr_at5']
                lines.append(f"{name} minus logistic: {c['difference']:+.6f}; descriptive paired 95% CI={c['paired_bootstrap_95ci']}.")
            lines.append('')
        lines += [f'Best individual selector by validation mean: {best}. Comparisons against this selector are also in the JSON summary.',
            'Tie-breaking uses no gold labels. Exact surfaces, including case/spacing variants, stay separate.',
            'Candidates absent from all input top-five lists cannot be recovered by fusion. Source full-pool coverage stays fixed.',
            'No GPU calls, new answers, fitting, paid judging, DPO pairs or test95 evaluation. All audit cases are validation-only.',
            'Checkpoint/backbone/fusion comparisons reuse validation; these are not independent test improvements.', '']
        (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        status.update(status='complete', artifacts_sha256=pilot.hashes(output))
        status['artifacts_sha256'].pop('manifest.json',None)
        print('\n'.join(lines),flush=True)
        return result
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        data.write(output/'manifest.json',status)


def prepare(repository, comparison, submission, resamples=10000):
    if not 100 <= resamples <= 100000:
        raise ValueError('Need 100..100000 paired resamples')
    value = data.read(comparison)
    if set(value.get('sources',{})) != set(MODELS) or value.get('test_evaluation') is not False:
        raise ValueError('Need the completed three-selector validation comparison')
    submission.mkdir(parents=True, exist_ok=False)
    shutil.copy2(comparison, submission/'source_comparison.json')
    from scripts.train_sft_ensemble_reranker import snapshot_code
    snapshot_code(repository,submission/'code')
    shutil.copy2(repository/'gadi_sft_8b_starter/jobs/fuse_answer_selectors.pbs',submission/'job.pbs')
    data.write(submission/'prepared.json',{'comparison_original':str(comparison), 'resamples':resamples,
        'fusion_policy':POLICY, 'files_sha256':pilot.hashes(submission), 'new_generation':False,
        'test_evaluation':False, 'DPO_training':False})
    print('Prepared one CPU audit of the saved validation rankings; no models loaded.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run'),required=True)
    parser.add_argument('--repository',type=Path,default=ROOT)
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--submission',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resamples',type=int,default=10000)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.comparison is None or args.submission is None: parser.error('--comparison and --submission required')
        prepare(args.repository.resolve(),args.comparison.resolve(),args.submission.resolve(),args.resamples)
    else:
        if args.output is None: parser.error('--output required')
        if args.submission is not None:
            sub = args.submission.resolve()
            state = data.read(sub/'prepared.json')
            data.verify_files(sub,state['files_sha256'])
            if state['fusion_policy'] != POLICY: raise ValueError('Frozen fusion rules changed')
            analyze(sub/'source_comparison.json',args.output.resolve(),state['resamples'])
        elif args.comparison is not None:
            analyze(args.comparison.resolve(),args.output.resolve(),args.resamples)
        else: parser.error('--submission or --comparison required')
