#!/usr/bin/env python3
"""Prepare/train/evaluate UniTor's released UT3 answer-generation baseline on Gadi."""
from __future__ import annotations
import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_replication import (PINS, read, save, digest, records, panel, sources,
                                       audit_training, unitor_parse, score)

SETTINGS = {
    'llama31': {'r': 32, 'alpha': 32, 'batch': 8, 'learning_rate': 6e-4},
    'phi4': {'r': 16, 'alpha': 16, 'batch': 2, 'learning_rate': 2e-4},
}
MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def llama_text(row, eos='', training=False):
    text = (f"{row['instruction']}\n\n# Question: {row['input_1']}\n# PubMed resources: "
            f"{row['input_2']}\n{row['input_3']}\n{row['input_4']}\n# Answer:")
    return text + f" {row['output']} {eos}" if training else text


def phi_messages(row, training=False):
    messages = [{'role': 'system', 'content': row['instruction'].strip()}, {'role': 'user',
        'content': f"Question: {row['input_1'].strip()}\nPubMed resources: {row['input_2']}\n{row['input_3']}\n{row['input_4']}"}]
    if training:
        messages.append({'role': 'assistant', 'content': f"Answer: {row['output']}"})
    return messages


def labels_after_marker(ids, marker):
    positions = [i for i in range(len(ids) - len(marker) + 1) if ids[i:i + len(marker)] == marker]
    if not marker or not positions:
        return [-100] * len(ids)
    start = positions[-1] + len(marker)
    return [-100] * start + ids[start:]


def validate_snapshot(submission):
    state = read(submission / 'prepared.json')
    for relative, expected in state['files_sha256'].items():
        path = (submission / relative).resolve()
        if not path.is_relative_to(submission.resolve()) or digest(path) != expected:
            raise ValueError('Frozen experiment file changed: ' + relative)
    # Re-check downloaded backbone weights/config/tokenizers before every GPU stage.
    for model in state['models'].values():
        snapshot = Path(model['snapshot'])
        for relative, expected in model['files_sha256'].items():
            if digest(snapshot / relative) != expected:
                raise ValueError('Pinned backbone changed: ' + relative)
    return state


def prepare(source_dir, bundle, submission, models, download, download_models):
    if not models or len(set(models)) != len(models) or not set(models) <= set(SETTINGS):
        raise ValueError('Select unique supported UniTor models')
    config = sources(source_dir, download)
    inputs, _ = panel(bundle)
    data = source_dir / 'unitor/data/dataset/answer_generation'
    train, validation = read(data / 'train_set_answGEN.json'), read(data / 'dev_set_answGEN.json')
    audit = audit_training(train, validation, inputs)
    from huggingface_hub import snapshot_download
    resolved = {}
    for model in models:
        info = config['models'][model]
        try:
            snapshot = Path(snapshot_download(info['model_id'], revision=info['revision'], local_files_only=not download_models))
        except Exception as exc:
            raise RuntimeError(f"Pinned {model} backbone is not cached; prepare with --download-models on Gadi") from exc
        weights = list(snapshot.glob('*.safetensors'))
        if not weights or not (snapshot / 'config.json').exists():
            raise ValueError('Incomplete backbone snapshot')
        resolved[model] = {**info, 'snapshot': str(snapshot.resolve()),
            'files_sha256': {p.name: digest(p) for p in sorted(snapshot.iterdir()) if p.is_file()},
            'settings': SETTINGS[model]}
    submission.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'test_inputs.jsonl', 'test_gold.jsonl'):
        shutil.copy2(bundle / name, submission / ('test_' + name if name == 'manifest.json' else name))
    for name in ('train_set_answGEN.json', 'dev_set_answGEN.json'):
        shutil.copy2(data / name, submission / name)
    instruction = next(r['instruction'] for r in train if r['type'] == 'factoid')
    test_rows = []
    for row in inputs:
        snippets = [s['text'] for s in row['snippets'][:3]]
        snippets += [''] * (3 - len(snippets))
        test_rows.append({'id': row['question_id'], 'type': 'factoid', 'instruction': instruction,
            'input_1': row['question'], **{f'input_{i+2}': s for i, s in enumerate(snippets)}})
    save(submission / 'test_rows.json', test_rows)
    code = submission / 'code'
    needed = [ROOT / 'scripts/paper_replication.py', Path(__file__), PINS,
              *sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'src/utility/java').glob('*')),
              ROOT / 'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar']
    for path in needed:
        if path.is_file():
            target = code / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2(ROOT / 'gadi_sft_8b_starter/jobs/unitor_replication.pbs', submission / 'job.pbs')
    files = {p.relative_to(submission).as_posix(): digest(p) for p in submission.rglob('*') if p.is_file()}
    save(submission / 'prepared.json', {'models': resolved, 'source_pins': config, 'source_audit': audit,
        'files_sha256': files, 'test_questions': 95, 'training_max_length': 1024,
        'inference_allocation': 1408, 'max_new_tokens': 256,
        'context': 'UT3: first three gold snippets, original order, no synthetic snippets or evidence reranking',
        'limitations': ['Fresh fine-tuning, not authors unreleased trained weights.',
            'Preserves released multi-task data and hyperparameters; uses Transformers Trainer with explicit completion masks for current runtime compatibility.',
            'Only released three-context UT3 variant; not the leaderboard UTR5 SBERT variant.',
            'Training includes authors context annotations; test resources are literal provided gold snippets.',
            'Historical project test95 has already been examined.']})
    print('Prepared pinned UniTor training and UT3 test95 evaluation:', ', '.join(models), flush=True)


def load_model(state, model_key, training, adapter=None):
    from unsloth import FastLanguageModel
    entry = state['models'][model_key]
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(adapter or entry['snapshot']),
        max_seq_length=1024 if training else 1408, dtype=None, load_in_4bit=True)
    if model_key == 'phi4':
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template='phi-4')
    if training:
        setting = entry['settings']
        model = FastLanguageModel.get_peft_model(model, r=setting['r'], lora_alpha=setting['alpha'],
            target_modules=MODULES, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth',
            random_state=3407, use_rslora=False, loftq_config=None)
    else:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def model_cache(source_dir, models):
    """Explicit one-time backbone download, separate from job submission."""
    config = sources(source_dir, True)
    from huggingface_hub import snapshot_download
    for model in models:
        info = config['models'][model]
        snapshot = snapshot_download(info['model_id'], revision=info['revision'])
        print(model, snapshot, flush=True)


def tokenize_training(rows, tokenizer, model_key):
    packed, counts = [], Counter()
    marker = (tokenizer.encode('\n# Answer:', add_special_tokens=False)[2:] if model_key == 'llama31'
              else tokenizer.encode('<|im_start|>assistant<|im_sep|>', add_special_tokens=False))
    for row in rows:
        text = llama_text(row, tokenizer.eos_token, True) if model_key == 'llama31' else tokenizer.apply_chat_template(
            phi_messages(row, True), tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=True)['input_ids']
        counts['truncated_rows'] += len(ids) > 1024
        ids = ids[:1024]
        labels = labels_after_marker(ids, marker)
        if all(v == -100 for v in labels):
            counts['zero_target_rows'] += 1
        packed.append({'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels})
    counts['rows'] = len(packed)
    return packed, dict(counts)


def train(submission, model_key, smoke):
    state = validate_snapshot(submission)
    output = submission / 'runs' / model_key / ('smoke' if smoke else 'train')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'manifest.json', {'status': 'running', 'smoke': smoke})
    model, tokenizer = load_model(state, model_key, True)
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    datasets, diagnostics = {}, {}
    for split, name, seed in [('train', 'train_set_answGEN.json', 42), ('validation', 'dev_set_answGEN.json', 34)]:
        rows = read(submission / name)
        if smoke:
            rows = rows[:16]
        packed, diagnostics[split] = tokenize_training(rows, tokenizer, model_key)
        # Published completion masking silently ignores marker-free rows. A port must not
        # silently change that into full-prompt training or train on empty labels.
        if diagnostics[split].get('zero_target_rows', 0):
            save(output / 'tokenization_audit.json', diagnostics)
            raise ValueError('Released 1024-token truncation/mask leaves empty targets; inspect tokenization audit before choosing a protocol change')
        datasets[split] = Dataset.from_list(packed)
        if model_key == 'llama31':
            datasets[split] = datasets[split].shuffle(seed=seed)
    save(output / 'tokenization_audit.json', diagnostics)
    setting = state['models'][model_key]['settings']
    args = TrainingArguments(output_dir=str(output / 'checkpoints'), per_device_train_batch_size=setting['batch'],
        per_device_eval_batch_size=8, gradient_accumulation_steps=4, warmup_steps=5,
        num_train_epochs=2, max_steps=1 if smoke else -1, learning_rate=setting['learning_rate'],
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=1,
        optim='adamw_8bit', weight_decay=.01, lr_scheduler_type='linear', seed=3407,
        eval_strategy='epoch', save_strategy='epoch', report_to='none', save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        remove_unused_columns=False)
    kwargs = {'model': model, 'args': args, 'train_dataset': datasets['train'], 'eval_dataset': datasets['validation'],
              'data_collator': DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)}
    kwargs['processing_class' if 'processing_class' in inspect.signature(Trainer.__init__).parameters else 'tokenizer'] = tokenizer
    trainer = Trainer(**kwargs)
    trainer.train()
    model.save_pretrained(str(output / 'adapter'))
    tokenizer.save_pretrained(str(output / 'adapter'))
    save(output / 'manifest.json', {'status': 'complete', 'smoke': smoke, 'global_steps': trainer.state.global_step,
        'best_eval_loss': trainer.state.best_metric, 'best_checkpoint': trainer.state.best_model_checkpoint,
        'adapter_files_sha256': {p.name: digest(p) for p in (output / 'adapter').iterdir() if p.is_file()},
        'runtime': {name: __import__(name).__version__ for name in ('transformers', 'torch', 'unsloth')},
        'tokenization_audit': diagnostics})


def evaluate(submission, model_key):
    state = validate_snapshot(submission)
    training = submission / 'runs' / model_key / 'train'
    trained = read(training / 'manifest.json')
    if trained['status'] != 'complete' or trained['smoke']:
        raise ValueError('Need completed full training')
    for name, expected in trained['adapter_files_sha256'].items():
        if digest(training / 'adapter' / name) != expected:
            raise ValueError('Trained adapter changed')
    model, tokenizer = load_model(state, model_key, False, training / 'adapter')
    output = submission / 'runs' / model_key / 'test95'
    output.mkdir(parents=True, exist_ok=False)
    saved = []
    import torch
    with (output / 'generations.jsonl').open('w', encoding='utf-8') as stream:
        for row in read(submission / 'test_rows.json'):
            text = llama_text(row) if model_key == 'llama31' else tokenizer.apply_chat_template(
                phi_messages(row), tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(text, return_tensors='pt', add_special_tokens=(model_key == 'llama31')).to('cuda')
            prompt_length = encoded['input_ids'].shape[-1]
            if prompt_length + 256 > 1408:
                raise ValueError('UT3 inference exceeds released 1408-token allocation; refusing silent truncation')
            with torch.inference_mode():
                result = model.generate(**encoded, max_new_tokens=256, do_sample=False, use_cache=True)
            tokens = result[0, prompt_length:]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            record = {'question_id': row['id'], 'raw_response': raw, 'prompt_tokens': prompt_length,
                'output_tokens': len(tokens), 'finish_reason': 'length' if len(tokens) >= 256 else 'stop'}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            saved.append(record)
            if len(saved) % 10 == 0:
                print(f'{model_key}: completed {len(saved)}/95', flush=True)
    # Scorer expects the standard bundle manifest name in a dedicated folder.
    bundle = output / 'bundle'
    bundle.mkdir()
    for source, target in [('test_manifest.json', 'manifest.json'), ('test_inputs.jsonl', 'test_inputs.jsonl'), ('test_gold.jsonl', 'test_gold.jsonl')]:
        shutil.copy2(submission / source, bundle / target)
    result = score(bundle, output / 'generations.jsonl', output, unitor_parse)
    result['limitations'] = state['limitations']
    result['protocol'] = state['context']
    save(output / 'comparison_summary.json', result)
    save(output / 'manifest.json', {'status': 'complete', 'question_count': 95, 'model': model_key})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['download-models', 'prepare', 'train', 'evaluate'])
    p.add_argument('--submission', type=Path)
    p.add_argument('--sources', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--models', nargs='+', choices=SETTINGS, default=['llama31', 'phi4'])
    p.add_argument('--model', choices=SETTINGS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--download', action='store_true')
    p.add_argument('--download-models', action='store_true')
    a = p.parse_args()
    if a.mode == 'download-models':
        if not a.sources:
            p.error('--sources is required')
        model_cache(a.sources, a.models)
    elif a.mode == 'prepare':
        if not a.sources or not a.bundle:
            p.error('--sources and --bundle are required for preparation')
        if not a.submission:
            p.error('--submission is required')
        prepare(a.sources, a.bundle, a.submission, a.models, a.download, a.download_models)
    elif a.mode == 'train':
        if not a.submission or not a.model:
            p.error('--submission and --model are required')
        train(a.submission.resolve(), a.model, a.smoke)
    else:
        if not a.submission or not a.model:
            p.error('--submission and --model are required')
        evaluate(a.submission.resolve(), a.model)
