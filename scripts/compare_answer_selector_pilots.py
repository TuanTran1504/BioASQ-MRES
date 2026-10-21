#!/usr/bin/env python3
"""Compare the three frozen selector pilots on the same internal validation pools."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_answer_selector_pilot as pilot
from scripts import prepare_answer_selector_data as data


def compare(root, output):
    latest = {}
    for path in sorted(root.glob('submission-*/prepared.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        model = data.read(path)['configuration']['model_key']
        if model in pilot.MODELS:
            latest.setdefault(model, path.parent)
    if set(latest) != set(pilot.MODELS):
        raise ValueError('Need submissions for all three selectors; found ' + ', '.join(sorted(latest)))
    conditions, perq, provenance = {}, {}, {}
    signature = None
    common = None
    for model in pilot.MODELS:
        submission = latest[model]
        state = pilot.validate(submission)
        shared = {k: v for k, v in state['configuration'].items() if k != 'model_key'}
        files = {name: data.digest(submission / name) for name in (
            'train_inputs.jsonl', 'train_supervision.jsonl', 'validation_inputs.jsonl',
            'validation_supervision.jsonl', 'heldout_ids.json')}
        if signature is not None and (signature != files or common != shared):
            raise ValueError('Selector data or common training settings differ; refuse comparison')
        signature, common = files, shared
        with (submission / 'jobs.tsv').open() as handle:
            jobs = {r['stage']: r['job_id'] for r in csv.DictReader(handle, delimiter='\t')}
        report = submission / ('report-' + jobs['report'])
        training = submission / ('train-' + jobs['train'])
        baseline = submission / ('baseline-' + jobs['baseline'])
        pin = data.digest(submission / 'prepared.json')
        pilot.verify_result(report, pin)
        pilot.verify_result(training, pin, smoke=False)
        pilot.verify_result(baseline, pin)
        summary = data.read(report / 'comparison_summary.json')
        epoch = summary['selected_epoch']
        conditions[model] = summary['conditions']['selector']
        perq[model] = data.read(training / f'epoch-{epoch}/validation_per_question.json')
        if any(len(perq[model][p]) != data.COUNTS['validation'] for p in pilot.POOLS):
            raise ValueError('Incomplete validation panel')
        provenance[model] = {'submission': str(submission), 'report': str(report),
                             'selected_epoch': epoch, 'prepared_sha256': pin}
    from scripts.analyze_matched_qwen_expansion import paired_contrast
    contrasts = {f'{model}_minus_llama31_{pool}': paired_contrast(perq['llama31'][pool], perq[model][pool],
        metrics=('mrr_at5', 'coverage_at5', 'coverage_full_pool'))
        for model in ('qwen3', 'ministral3') for pool in pilot.POOLS}
    means = {m: sum(conditions[m][p]['mrr_at5'] for p in pilot.POOLS) / len(pilot.POOLS) for m in pilot.MODELS}
    best = [m for m, v in means.items() if abs(v - max(means.values())) < 1e-12]
    output.mkdir(parents=True, exist_ok=False)
    result = {'question_count': data.COUNTS['validation'], 'conditions': conditions,
        'mean_validation_mrr_at5': means, 'highest_validation_models': best,
        'paired_contrasts': contrasts, 'sources': provenance, 'shared_data_sha256': signature,
        'interpretation': 'Internal validation comparison used for selection; no independent test result',
        'DPO_training': False, 'test_evaluation': False}
    data.write(output / 'comparison_summary.json', result)
    lines = ['# Three supervised selectors: internal validation', '',
        '| Selector | Selected epoch | Greedy MRR@5 | Sampling4 MRR@5 | Mean |',
        '|---|---:|---:|---:|---:|']
    for model in pilot.MODELS:
        a, b = (conditions[model][p]['mrr_at5'] for p in pilot.POOLS)
        lines.append(f"| {model} | {provenance[model]['selected_epoch']} | {a:.6f} | {b:.6f} | {means[model]:.6f} |")
    lines += ['', 'Highest validation mean: ' + ', '.join(best) + '.',
        'Same fitting data, all144 validation questions/mode, common training settings and fixed candidate pools.',
        'Native chat templates and tokenizer-compatible ID codes differ. GPU smoke checks are model-specific.',
        'Validation selects both checkpoints and backbone; these scores do not establish held-out superiority.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print('Saved:', output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'gadi_sft_8b_starter/outputs/answer_selector_pilot')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / ('comparison-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
    compare(args.root.resolve(), destination.resolve())
