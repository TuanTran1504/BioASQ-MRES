#!/usr/bin/env python3
"""Contextual BiomedBERT versus frozen logistic ranking on unchanged max-five SFT pools."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from scripts import evaluate_sft_ensemble_rerankers as ev
from scripts import train_sft_ensemble_reranker as rr
from scripts.ablate_reranker_relation_labels import load_training, verify_files
from scripts.train_candidate_reranker_pilot import score_components

CONFIG = ROOT / 'gadi_sft_8b_starter/configs/contextual_sft_reranker.json'
INPUT_KEYS = ('input_ids', 'attention_mask', 'token_type_ids')
STOP = set('a an the of to in on for and or is are was were what which how with by as from does do'.split())


def tokens(text):
    return set(re.findall(r'[\w]+', text.casefold())) - STOP


def relevance(text, question, answer):
    """Fixed lexical evidence selector; no aliases, official labels or trained scorer."""
    tt, qq, aa = tokens(text), tokens(question), tokens(answer)
    return (len(tt & qq) / max(1, len(qq)) + 2 * len(tt & aa) / max(1, len(aa)))


def best_window(ids, budget, tokenizer, question, answer):
    if len(ids) <= budget:
        return ids, 0
    starts = sorted(set(range(0, len(ids) - budget + 1, max(1, budget // 2))) | {len(ids) - budget})
    start = min(starts, key=lambda i: (-relevance(tokenizer.decode(ids[i:i + budget]), question, answer), i))
    return ids[start:start + budget], start


def encode_candidate(example, candidate, tokenizer, config, snippet_ids=None):
    """Preserve the entire question and answer; select evidence within native BERT length."""
    question, answer = example['question'], candidate['answer']
    head = tokenizer.encode(f'Question: {question}\nCandidate answer: {answer}', add_special_tokens=False)
    snippets = example['snippets']
    k = min(config['evidence_snippets'], len(snippets))
    special = tokenizer.num_special_tokens_to_add(pair=True)
    available = config['max_length'] - special - len(head) - max(0, k - 1)
    if k < 1 or available < config['minimum_evidence_tokens']:
        raise ValueError('Question/candidate leaves insufficient evidence space; no silent head truncation')
    chosen = sorted(range(len(snippets)), key=lambda i: (-relevance(snippets[i]['text'], question, answer), i))[:k]
    all_ids = snippet_ids if snippet_ids is not None else [tokenizer.encode(s['text'], add_special_tokens=False) for s in snippets]
    if len(all_ids) != len(snippets) or not any(all_ids):
        raise ValueError('Missing usable evidence or misaligned tokenized snippets')
    evidence, kept = [], []
    for position, i in enumerate(chosen):
        budget = available // k + int(position < available % k)
        window, offset = best_window(all_ids[i], budget, tokenizer, question, answer)
        if position:
            evidence.append(tokenizer.sep_token_id)
        evidence.extend(window)
        kept.append({'snippet_id': str(snippets[i].get('id', snippets[i].get('snippet_id', i + 1))),
                     'source_index': i, 'start_token': offset, 'retained_tokens': len(window),
                     'original_tokens': len(all_ids[i])})
    # BiomedBERT uses BERT's native pair framing. Construct IDs directly so
    # selected WordPiece windows are not decoded/re-tokenized, and remain
    # compatible with tokenizers before and after Transformers 5.
    if special != 3 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('Expected the pinned BERT pair framing')
    ids = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id] + evidence + [tokenizer.sep_token_id]
    encoded = {'input_ids': ids, 'attention_mask': [1] * len(ids),
               'token_type_ids': [0] * (len(head) + 2) + [1] * (len(evidence) + 1)}
    if len(encoded['input_ids']) > config['max_length']:
        raise ValueError('Encoder input exceeds pinned context length')
    audit = {'question_id': example['question_id'], 'answer': answer, 'question_candidate_tokens': len(head),
             'total_snippets': len(snippets), 'selected_snippets': kept,
             'all_evidence_tokens': sum(map(len, all_ids)),
             'retained_evidence_tokens': sum(s['retained_tokens'] for s in kept),
             'encoded_tokens': len(encoded['input_ids'])}
    return {k: encoded[k] for k in INPUT_KEYS}, audit


def contexts(examples):
    return {q: {'question': e['question'], 'snippets': e['snippets']} for q, e in examples.items()}


def logistic_scores(payload, pools, examples):
    result = {}
    for name, rows in pools.items():
        clean = rr.feature_rows([{**r, 'surface_operations': [], 'is_format_variant': False} for r in rows])
        result[name] = score_components(payload['components'], clean, contexts(examples)).tolist() if rows else []
    return result


def ordered_indices(rows, values):
    if len(rows) != len(values) or not np.isfinite(values).all():
        raise ValueError('Missing or nonfinite candidate scores')
    return sorted(range(len(rows)), key=lambda i: (-float(values[i]), rr.tie_key(rows[i])))


def rank_scores(pools, examples, neural, logistic, alpha=None):
    """None=contextual; -1=logistic; 0..1=fixed reciprocal-rank blend."""
    if alpha is not None and alpha != -1 and not 0 <= alpha <= 1:
        raise ValueError('Invalid blend weight')
    conditions, questions, rankings = {}, {}, []
    for pool, rows in pools.items():
        nvalues = neural[pool]
        lvalues = logistic[pool]
        if len(nvalues) != len(rows) or len(lvalues) != len(rows):
            raise ValueError('Score alignment differs from candidate rows')
        grouped = {q: [] for q in examples}
        for i, row in enumerate(rows):
            if row['question_id'] not in grouped:
                raise ValueError('Unknown candidate question')
            grouped[row['question_id']].append(i)
        ordered = {}
        for qid, indices in grouped.items():
            group = [rows[i] for i in indices]
            nn, ll = [nvalues[i] for i in indices], [lvalues[i] for i in indices]
            ni, li = ordered_indices(group, nn), ordered_indices(group, ll)
            nr, lr = ({i: rank for rank, i in enumerate(order, 1)} for order in (ni, li))
            if alpha is None:
                order, values = ni, nn
            elif alpha == -1:
                order, values = li, ll
            else:
                values = [(1 - alpha) / lr[i] + alpha / nr[i] for i in range(len(group))]
                order = ordered_indices(group, values)
            ordered[qid] = [{**group[i], 'score': float(values[i])} for i in order]
        metrics, scored = rr.evaluate_rankings(ordered)
        conditions[pool] = {'metrics': metrics, 'mean_unique_candidates': len(rows) / len(examples)}
        questions[pool] = scored
        rankings.extend({'pool': pool, 'rank': rank, **r} for group in ordered.values() for rank, r in enumerate(group, 1))
    return conditions, questions, rankings


def select_blend(pools, examples, neural, logistic, weights):
    if not weights or list(weights) != sorted(set(weights)) or weights[0] != 0 or weights[-1] != 1:
        raise ValueError('Blend grid must include ordered unique endpoints zero and one')
    trials, best = [], None
    for alpha in weights:
        cc, _, _ = rank_scores(pools, examples, neural, logistic, alpha)
        value = float(np.mean([r['metrics']['mrr_at5'] for r in cc.values()]))
        trials.append({'alpha': alpha, 'mean_validation_mrr_at5': value,
                       'metrics': {name: r['metrics'] for name, r in cc.items()}})
        if best is None or value > best[0] + 1e-12:
            best = value, alpha
    return {'alpha': best[1], 'mean_validation_mrr_at5': best[0], 'trials': trials,
            'selection': '144 validation questions, greedy/sampling4 equally weighted; ties prefer lower alpha'}


def loss_weights(rows, strata):
    labels = np.array([r['label'] for r in rows])
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Need both official classes in fitting data')
    weights = rr.question_weights(rows, strata)
    mass = [weights[labels == label].sum() for label in (0, 1)]
    weights *= np.array([weights.sum() / (2 * mass[int(label)]) for label in labels])
    return (weights / weights.mean()).tolist()


def encode_split(pools, examples, tokenizer, config, fitting=False):
    rows, strata = [], []
    for name, candidates in pools.items():
        rows.extend(candidates)
        strata.extend([name] * len(candidates))
    weights = loss_weights(rows, strata) if fitting else [1.] * len(rows)
    encoded, audit, snippet_cache, packed_cache = [], [], {}, {}
    for row, pool, weight in zip(rows, strata, weights):
        qid = row['question_id']
        identity = (qid, row['answer'])
        if qid not in snippet_cache:
            snippet_cache[qid] = [tokenizer.encode(s['text'], add_special_tokens=False) for s in examples[qid]['snippets']]
        if identity not in packed_cache:
            packed_cache[identity] = encode_candidate(examples[qid], row, tokenizer, config, snippet_cache[qid])
        fields, record = packed_cache[identity]
        encoded.append({**fields, 'question_id': row['question_id'], 'pool': pool,
                        'candidate_answer': row['answer'], 'row_weight': weight})
        audit.append({**record, 'pool': pool})
    return encoded, audit


def hash_files(directory):
    return {p.relative_to(directory).as_posix(): ev.digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def freeze_backbone(snapshot, destination):
    """Materialize HF cache links as private files, retaining content verification."""
    expected = hash_files(snapshot)
    destination.mkdir(parents=True, exist_ok=False)
    for name in expected:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # HF snapshots normally symlink into ../../blobs. copy2 follows links;
        # never hardlink or symlink the submission back to the mutable cache.
        shutil.copy2(snapshot / name, target, follow_symlinks=True)
    verify_files(destination, expected)
    if hash_files(snapshot) != expected:
        raise ValueError('Backbone cache changed while freezing the submission')
    return destination


def validate_membership(examples, test, dev_ids):
    if {s: len(v) for s, v in examples.items()} != {'train': 1296, 'validation': 144} or len(test) != 95:
        raise ValueError('Need 1296 fitting, 144 validation and 95 test questions')
    train, validation = set(examples['train']), set(examples['validation'])
    if train & validation or (train | validation) & set(test) or (train | validation | set(test)) & set(dev_ids):
        raise ValueError('Fitting/validation overlaps another partition')
    norm = lambda q: re.sub(r'\W+', ' ', q.casefold()).strip()
    tt = {norm(e['question']) for e in test.values()}
    if any(norm(e['question']) in tt for split in examples.values() for e in split.values()):
        raise ValueError('Fitting/validation question text overlaps test95')


def prepare(repository, baseline, comparison, submission, download_model=False):
    _, test_pools, test, gold, test_hashes = ev.load_comparison(comparison)
    baseline_state, payloads, baseline_hashes = ev.load_frozen(baseline, test, comparison)
    examples, pools = load_training(baseline, baseline_hashes)
    print('Validated frozen fitting/validation banks and saved test pools; checking dev160 separation.', flush=True)
    starter = repository / 'gadi_sft_8b_starter'
    # Use the established matched-data configuration to protect dev160 membership.
    from prepare_matched_8b_sft import read as shared_read
    from run_matched_8b_evaluation import DEFAULT_CONFIG
    original = shared_read(starter / 'configs' / DEFAULT_CONFIG.name)
    matrix = shared_read(starter / original['training_matrix'])
    dev_ids = {r['question_id'] for r in ev.records(starter / matrix['outer_dev'])}
    if len(dev_ids) != 160:
        raise ValueError('Need pinned disjoint dev160 membership')
    validate_membership(examples, test, dev_ids)
    pools = {s: {n: rows for n, rows in pp.items() if n.startswith('max5_')} for s, pp in pools.items()}
    test_pools = {n: rows for n, rows in test_pools.items() if n.startswith('max5_')}
    config = ev.read(repository / CONFIG.relative_to(ROOT))
    if set(pools['train']) != set(config['training_pools']) or set(test_pools) != set(config['test_pools']):
        raise ValueError('Unexpected candidate pool names')
    from huggingface_hub import snapshot_download
    print('Resolving pinned BiomedBERT checkpoint (download once if missing).', flush=True)
    snapshot = Path(snapshot_download(config['model_id'], revision=config['revision'],
        allow_patterns=['config.json', 'tokenizer_config.json', 'vocab.txt', 'pytorch_model.bin', '*.safetensors', 'special_tokens_map.json'],
        local_files_only=not download_model))
    if not list(snapshot.glob('*.bin')) and not list(snapshot.glob('*.safetensors')):
        raise ValueError('Pinned BiomedBERT weights are incomplete')
    cache_snapshot = snapshot
    submission.mkdir(parents=True, exist_ok=False)
    snapshot = freeze_backbone(cache_snapshot, submission / 'backbone')
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    backbone = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if config['max_length'] > backbone.max_position_embeddings:
        raise ValueError('Requested encoder length exceeds native allocation')
    # Confirm unchanged scoring before reserving GPU resources.
    lr = {s: logistic_scores(payloads['max5'], pools[s], examples[s]) for s in examples}
    old_validation = ev.read(baseline / 'validation_summary.json')['metrics']
    cc, _, _ = rank_scores(pools['validation'], examples['validation'], lr['validation'], lr['validation'], -1)
    for pool, condition in cc.items():
        if condition['metrics'] != old_validation[pool + '_max5_reranker']:
            raise ValueError('Frozen logistic validation results no longer reproduce')
    audits = {}
    for split in ('train', 'validation'):
        ev.write_jsonl(submission / f'{split}_examples.jsonl', examples[split].values())
        ev.write(submission / f'{split}_pools.json', pools[split])
        encoded, audit = encode_split(pools[split], examples[split], tokenizer, config, split == 'train')
        ev.write_jsonl(submission / f'{split}_inputs.jsonl', encoded)
        ev.write_jsonl(submission / f'{split}_evidence_audit.jsonl', audit)
        ev.write(submission / f'{split}_logistic_scores.json', lr[split])
        audits[split] = evidence_summary(audit)
        print(f"Encoded {split}: {len(encoded)} candidates; evidence audit saved.", flush=True)
    # Test tokenization is gold-blind; labels remain in separate scoring data.
    encoded, audit = encode_split(test_pools, test, tokenizer, config)
    ev.write_jsonl(submission / 'test_inputs.jsonl', encoded)
    ev.write_jsonl(submission / 'test_evidence_audit.jsonl', audit)
    ev.write(submission / 'test_pools.json', test_pools)
    ev.write_jsonl(submission / 'test_examples.jsonl', test.values())
    ev.write(submission / 'test_gold.json', gold)
    ev.write(submission / 'test_logistic_scores.json', logistic_scores(payloads['max5'], test_pools, test))
    audits['test'] = evidence_summary(audit)
    print(f'Encoded test95: {len(encoded)} candidates; no test labels in encoder inputs.', flush=True)
    ev.write(submission / 'evidence_summary.json', audits)
    ev.write(submission / 'configuration.json', config)
    joblib.dump(payloads['max5'], submission / 'logistic.joblib')
    code_hashes = rr.snapshot_code(repository, submission / 'code')
    shutil.copy2(repository / 'gadi_sft_8b_starter/jobs/contextual_sft_reranker.pbs', submission / 'job.pbs')
    files = hash_files(submission)
    state = {'schema': config['schema'], 'repository': str(repository), 'config': config,
        'backbone_snapshot': str(snapshot.resolve()), 'backbone_files': hash_files(snapshot),
        'backbone_cache_source': str(cache_snapshot.resolve()),
        'baseline': str(baseline), 'baseline_hashes': baseline_hashes, 'comparison': str(comparison),
        'comparison_hashes': test_hashes, 'code_hashes': code_hashes, 'files_sha256': files,
        'question_counts': baseline_state['question_counts'], 'dev160_excluded': True,
        'dev_membership_sha256': ev.digest(starter / matrix['outer_dev']),
        'scope': 'Max-five SFT only; existing greedy/four-sample fitting and validation banks; no new generation',
        'test_scope': 'Same saved greedy/ten-sample test95 pools; historically examined, exploratory',
        'labels': config['labels'], 'evidence_summary': audits}
    ev.write(submission / 'prepared.json', state)
    # Exercise the exact worker integrity check before reserving GPU jobs.
    validate_submission(submission)
    print('Prepared BiomedBERT: 1296 fitting / 144 validation; same frozen max-five logistic baseline.', flush=True)
    print('Evidence policy: entire question/candidate plus three lexical evidence windows, at most 512 tokens.', flush=True)


def evidence_summary(audit):
    return {'candidate_rows': len(audit), 'mean_encoded_tokens': float(np.mean([r['encoded_tokens'] for r in audit])),
            'rows_with_omitted_evidence': sum(r['retained_evidence_tokens'] < r['all_evidence_tokens'] for r in audit),
            'total_evidence_tokens': sum(r['all_evidence_tokens'] for r in audit),
            'retained_evidence_tokens': sum(r['retained_evidence_tokens'] for r in audit)}


def validate_submission(submission):
    state = ev.read(submission / 'prepared.json')
    verify_files(submission, state['files_sha256'])
    if state['config'] != ev.read(submission / 'configuration.json') or hash_files(submission / 'code') != state['code_hashes']:
        raise ValueError('Frozen configuration or code membership differs')
    verify_files(Path(state['backbone_snapshot']), state['backbone_files'])
    return state


def dataset_records(encoded, pools):
    rows = [r for rr in pools.values() for r in rr]
    if len(encoded) != len(rows):
        raise ValueError('Encoded and candidate row counts differ')
    result = []
    for item, row, pool in zip(encoded, rows, [n for n, rr in pools.items() for _ in rr]):
        if (item['question_id'], item['candidate_answer'], item['pool']) != (row['question_id'], row['answer'], pool):
            raise ValueError('Encoded candidate ordering changed')
        result.append({**{k: item[k] for k in INPUT_KEYS}, 'labels': float(row['label']), 'row_weight': item['row_weight']})
    return result


def split_predictions(encoded, predictions, pools):
    if len(encoded) != len(predictions):
        raise ValueError('Incomplete neural predictions')
    result = {pool: [] for pool in pools}
    for item, value in zip(encoded, predictions):
        result[item['pool']].append(float(value))
    if any(len(result[n]) != len(rows) for n, rows in pools.items()):
        raise ValueError('Prediction pool alignment differs')
    return result


def training_classes():
    import torch
    from transformers import Trainer, DataCollatorWithPadding

    class Collator:
        def __init__(self, tokenizer):
            self.pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        def __call__(self, features):
            batch = self.pad([{k: r[k] for k in INPUT_KEYS} for r in features])
            for key in ('labels', 'row_weight'):
                if key in features[0]:
                    batch[key] = torch.tensor([r[key] for r in features], dtype=torch.float32)
            return batch

    class WeightedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            fields = dict(inputs)
            labels, weights = fields.pop('labels'), fields.pop('row_weight')
            output = model(**fields)
            logits = output.logits.reshape(-1).float()
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            loss = (losses * weights).mean()
            return (loss, output) if return_outputs else loss

    return WeightedTrainer, Collator


def training_arguments(output, config, smoke, cpu=False):
    from transformers import TrainingArguments
    fields = {'output_dir': str(output / 'checkpoints'), 'learning_rate': config['learning_rate'],
        'per_device_train_batch_size': config['batch_size'], 'per_device_eval_batch_size': config['batch_size'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'], 'num_train_epochs': config['epochs'],
        'max_steps': 2 if smoke else -1, 'weight_decay': config['weight_decay'], 'warmup_ratio': config['warmup_ratio'],
        'optim': 'adamw_torch', 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-8,
        'lr_scheduler_type': 'linear', 'max_grad_norm': 1., 'fp16': not cpu, 'bf16': False,
        'seed': config['seed'], 'data_seed': config['seed'], 'eval_strategy': 'epoch', 'save_strategy': 'epoch',
        'save_total_limit': 2, 'load_best_model_at_end': True, 'metric_for_best_model': 'mean_mrr_at5',
        'greater_is_better': True, 'logging_steps': 20, 'report_to': 'none', 'remove_unused_columns': False,
        'dataloader_num_workers': 0, 'gradient_checkpointing': not cpu}
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if cpu:
        fields['use_cpu' if 'use_cpu' in parameters else 'no_cuda'] = True
    if 'warmup_ratio' not in parameters:
        fields.pop('warmup_ratio')
        raise RuntimeError('Runtime does not support the pinned warm-up configuration')
    return TrainingArguments(**fields)


def fit_model(model, tokenizer, encoded, pools, examples, logistic, config, output, smoke=False, cpu=False):
    from datasets import Dataset
    WeightedTrainer, Collator = training_classes()
    training = Dataset.from_list(dataset_records(encoded['train'], pools['train']))
    validation = Dataset.from_list(dataset_records(encoded['validation'], pools['validation']))

    def metrics(prediction):
        logits = np.asarray(prediction.predictions).reshape(-1)
        neural = split_predictions(encoded['validation'], logits, pools['validation'])
        cc, _, _ = rank_scores(pools['validation'], examples['validation'], neural, logistic)
        result = {'mean_mrr_at5': float(np.mean([r['metrics']['mrr_at5'] for r in cc.values()]))}
        result.update({n + '_mrr_at5': r['metrics']['mrr_at5'] for n, r in cc.items()})
        print('Validation contextual ranking:', result, flush=True)
        return result

    kwargs = {'model': model, 'args': training_arguments(output, config, smoke, cpu),
              'train_dataset': training, 'eval_dataset': validation, 'data_collator': Collator(tokenizer), 'compute_metrics': metrics}
    # Wrapped __init__ has *args/**kwargs, so inspect the underlying Trainer API.
    from transformers import Trainer
    kwargs['processing_class' if 'processing_class' in inspect.signature(Trainer.__init__).parameters else 'tokenizer'] = tokenizer
    trainer = WeightedTrainer(**kwargs)
    trainer.train()
    predicted = trainer.predict(validation)
    neural = split_predictions(encoded['validation'], np.asarray(predicted.predictions).reshape(-1), pools['validation'])
    trainer.save_model(str(output / 'model'))
    tokenizer.save_pretrained(str(output / 'model'))
    blend = select_blend(pools['validation'], examples['validation'], neural, logistic, config['blend_weights'])
    return trainer, neural, blend


def train(submission, smoke=False, cpu=False):
    state = validate_submission(submission)
    output = submission / (('smoke-' if smoke else 'train-') + os.environ.get('PBS_JOBID', 'manual'))
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'smoke': smoke, 'prepared_sha256': ev.digest(submission / 'prepared.json')}
    ev.write(output / 'manifest.json', status)
    try:
        config = state['config']
        pools = {s: ev.read(submission / f'{s}_pools.json') for s in ('train', 'validation')}
        examples = {s: {r['question_id']: r for r in ev.records(submission / f'{s}_examples.jsonl')} for s in pools}
        encoded = {s: ev.records(submission / f'{s}_inputs.jsonl') for s in pools}
        logistic = ev.read(submission / 'validation_logistic_scores.json')
        if smoke:
            for split, count in [('train', 8), ('validation', 4)]:
                keep = set(sorted(examples[split])[:count])
                old = pools[split]
                pools[split] = {n: [r for r in rows if r['question_id'] in keep] for n, rows in old.items()}
                if split == 'validation':
                    logistic = {n: [v for r, v in zip(old[n], logistic[n]) if r['question_id'] in keep] for n in old}
                examples[split] = {q: e for q, e in examples[split].items() if q in keep}
                encoded[split] = [r for r in encoded[split] if r['question_id'] in keep]
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
        import torch
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError('GPU training requires CUDA; CPU is only for technical tests')
        set_seed(config['seed'])
        snapshot = state['backbone_snapshot']
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(snapshot, num_labels=1, local_files_only=True)
        trainer, neural, blend = fit_model(model, tokenizer, encoded, pools, examples, logistic, config, output, smoke, cpu)
        conditions, scores = {}, {}
        for method, alpha in [('logistic', -1), ('contextual', None), ('hybrid', blend['alpha'])]:
            cc, ss, _ = rank_scores(pools['validation'], examples['validation'], neural, logistic, alpha)
            conditions[method], scores[method] = cc, ss
        selection = {'status': 'frozen', 'smoke': smoke, 'selected_checkpoint': trainer.state.best_model_checkpoint,
            'best_mean_validation_mrr_at5': trainer.state.best_metric, 'hybrid': blend,
            'conditions': conditions, 'scope': 'Internal validation only; test labels never read by training',
            'log_history': trainer.state.log_history, 'prepared_sha256': status['prepared_sha256']}
        ev.write(output / 'selection.json', selection)
        ev.write(output / 'validation_per_question.json', scores)
        ev.write(output / 'validation_neural_scores.json', neural)
        status.update(status='complete', global_steps=trainer.state.global_step,
            model_files=hash_files(output / 'model'), selection_sha256=ev.digest(output / 'selection.json'),
            model_id=config['model_id'], model_revision=config['revision'],
            question_counts={s: len(e) for s, e in examples.items()},
            candidate_counts={s: sum(map(len, pp.values())) for s, pp in pools.items()},
            runtime={name: str(__import__(name).__version__) for name in ('torch', 'transformers', 'sklearn')},
            device='cpu' if cpu else torch.cuda.get_device_name(0))
        print('Selected validation checkpoint:', trainer.state.best_model_checkpoint, flush=True)
        print('Selected hybrid alpha:', blend['alpha'], flush=True)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        ev.write(output / 'manifest.json', status)
    return output


def predict(model, tokenizer, encoded, batch_size, cpu=False):
    import torch
    from transformers import DataCollatorWithPadding
    device = 'cpu' if cpu else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Scoring job requires CUDA')
    model.to(device).eval()
    pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    result = []
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = pad([{k: r[k] for k in INPUT_KEYS} for r in encoded[start:start + batch_size]])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=not cpu):
                values = model(**batch).logits.reshape(-1).float().cpu().tolist()
            result.extend(values)
    return result


def compare(pools, examples, neural, logistic, alpha, gold=None, resamples=10000):
    conditions, questions, rankings, contrasts, transitions = {}, {}, [], {}, {}
    for method, weight in [('logistic', -1), ('contextual', None), ('hybrid', alpha)]:
        cc, ss, rranks = rank_scores(pools, examples, neural, logistic, weight)
        for pool in pools:
            name = f'{pool}_{method}'
            conditions[name], questions[name] = cc[pool], ss[pool]
            if gold:
                cc[pool]['per_batch'] = {batch: {'question_count': len(subset), **{
                    metric: sum(r[metric] for r in subset) / len(subset) for metric in ev.METRICS}}
                    for batch in sorted({r['batch'] for r in gold.values()})
                    if (subset := [r for r in ss[pool] if gold[r['question_id']]['batch'] == batch])}
        rankings.extend({'condition': f"{r['pool']}_{method}", **r} for r in rranks)
    for pool in pools:
        if len({conditions[f'{pool}_{m}']['metrics']['coverage_full_pool'] for m in ('logistic', 'contextual', 'hybrid')}) != 1:
            raise ValueError('Ranking changed full-pool coverage')
        for method in ('contextual', 'hybrid'):
            before, after = f'{pool}_logistic', f'{pool}_{method}'
            delta = ev.paired_contrast(questions[before], questions[after], resamples=resamples, metrics=ev.METRICS)
            name = f'{pool}_{method}_minus_logistic'
            contrasts[name] = {'before': before, 'after': after, 'metrics': {m: {
                'difference': v['sft_minus_base'], 'paired_bootstrap_95ci': v['paired_bootstrap_95ci']}
                for m, v in delta.items()}}
            aa = {r['question_id']: r for r in questions[after]}
            transitions[name] = {'gained': [r['question_id'] for r in questions[before] if not r['coverage_at5'] and aa[r['question_id']]['coverage_at5']],
                'lost': [r['question_id'] for r in questions[before] if r['coverage_at5'] and not aa[r['question_id']]['coverage_at5']]}
    return {'question_count': len(examples), 'conditions': conditions, 'paired_contrasts': contrasts,
            'coverage_at5_transitions': transitions, 'blend_alpha': alpha, 'resamples': resamples}, questions, rankings


def evaluate(submission, training, cpu=False, resamples=10000):
    state = validate_submission(submission)
    trained = ev.read(training / 'manifest.json')
    if (trained['status'] != 'complete' or trained['smoke'] or
            trained['prepared_sha256'] != ev.digest(submission / 'prepared.json')):
        raise ValueError('Need completed full training from this submission')
    verify_files(training / 'model', trained['model_files'])
    if ev.digest(training / 'selection.json') != trained['selection_sha256']:
        raise ValueError('Frozen validation selection changed')
    selection = ev.read(training / 'selection.json')
    output = submission / ('comparison-' + os.environ.get('PBS_JOBID', 'manual'))
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'training': str(training), 'selection_sha256': trained['selection_sha256']}
    ev.write(output / 'manifest.json', status)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(training / 'model', local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(training / 'model', local_files_only=True)
        encoded = ev.records(submission / 'test_inputs.jsonl')
        pools = ev.read(submission / 'test_pools.json')
        dataset_records(encoded, pools)  # Validate candidate alignment; labels are never model inputs.
        examples = {r['question_id']: r for r in ev.records(submission / 'test_examples.jsonl')}
        logits = predict(model, tokenizer, encoded, state['config']['batch_size'], cpu)
        neural = split_predictions(encoded, logits, pools)
        ev.write(output / 'neural_scores.json', neural)
        result, questions, rankings = compare(pools, examples, neural,
            ev.read(submission / 'test_logistic_scores.json'), selection['hybrid']['alpha'],
            ev.read(submission / 'test_gold.json'), resamples)
        result['limitations'] = ['Historically inspected test95; exploratory, not a fresh confirmatory test.',
            'Fitting and validation use four sampled draws; test uses ten, equally for both rerankers.',
            'Encoder reads selected evidence windows within 512 tokens, not complete evidence.',
            'Hybrid includes the entire frozen logistic score, including TF-IDF, metadata and relation features.',
            'Official nonacceptance is not a scientific incorrectness label.']
        result['evidence_summary'] = state['evidence_summary']['test']
        result['generation_reused'] = True
        ev.write(output / 'comparison_summary.json', result)
        ev.write(output / 'comparison_per_question.json', questions)
        ev.write_jsonl(output / 'rankings.jsonl', rankings)
        verify_files(training / 'model', trained['model_files'])
        status['status'] = 'complete'
        for name, value in result['conditions'].items():
            m = value['metrics']
            print(f"{name}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}, full pool={m['coverage_full_pool']:.4f}", flush=True)
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        ev.write(output / 'manifest.json', status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'train', 'evaluate'], required=True)
    p.add_argument('--submission', type=Path, required=True)
    p.add_argument('--repository', type=Path, default=ROOT)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--comparison', type=Path)
    p.add_argument('--training', type=Path)
    p.add_argument('--download-model', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--resamples', type=int, default=10000)
    a = p.parse_args()
    if a.mode == 'prepare':
        if not a.baseline or not a.comparison:
            p.error('--baseline and --comparison required')
        prepare(a.repository.resolve(), a.baseline.resolve(), a.comparison.resolve(), a.submission.resolve(), a.download_model)
    elif a.mode == 'train':
        train(a.submission.resolve(), a.smoke, a.cpu)
    else:
        if not a.training:
            p.error('--training required')
        evaluate(a.submission.resolve(), a.training.resolve(), a.cpu, a.resamples)
