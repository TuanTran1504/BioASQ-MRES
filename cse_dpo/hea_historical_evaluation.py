"""Historical full-dev greedy evaluation for HEA-DPO adapter snapshots.

Training/reference precision is independent of this NF4/FP16 evaluator.
Uses the same rendering, prepared gold, generation and scorer as the standalone
evaluation notebook. Never updates adapter weights or truncates supplied inputs.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cse_dpo.generated_bioasq_eval import generate_answer_from_prompt, load_gold_examples
from src.utility.bioasq_official import evaluate_with_bioasq_java
from src.utility.eval_dataset import load_eval_examples, render_prompt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_historical_dev(tokenizer, dev_input, registry_path, prompt_id,
                           max_seq_length=9000, max_new_tokens=64, limit=None):
    registry = json.loads(Path(registry_path).read_text())
    instruction = registry['prompts'][prompt_id]['instructions']['factoid']
    source_args = SimpleNamespace(question_types=['factoid'], max_resources=0,
        max_resource_chars=0, resource_selection='first', resource_granularity='document',
        resource_window_mode='single', local_files_only=True, limit=None)
    examples = load_eval_examples([Path(dev_input)], source_args, {'factoid': instruction})
    gold = load_gold_examples([Path(dev_input)])
    assert len(examples) == len(gold) and {e.question_id for e in examples} == set(gold)
    rows = []
    for example in examples:
        prompt = render_prompt(tokenizer, example, chat_template='qwen-2.5', prompt_format='chat')
        prompt_ids = tokenizer(prompt, return_tensors=None)['input_ids']
        if len(prompt_ids) + max_new_tokens > max_seq_length:
            raise ValueError(f'{example.question_id}: context overflow; refusing truncation')
        rows.append({'question_id': example.question_id, 'prompt': prompt,
            'canonical': example.gold_output, 'example': gold[example.question_id],
            'resources': list(example.resources), 'question': example.body, 'prompt_id': prompt_id})
    return rows if limit is None else rows[:limit]


def canonical_nll(model, tokenizer, prompt, answer, max_seq_length):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + ' ' + answer, add_special_tokens=False)
    assert full[:len(prefix)] == prefix
    full.append(tokenizer.eos_token_id)
    if len(full) > max_seq_length:
        raise ValueError('Canonical scoring overflow; refusing truncation')
    n = len(full) - len(prefix)
    ids = torch.tensor([full], device=model.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
        logits_to_keep=n + 1, use_cache=False).logits[:, :-1].float()
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
        ids[:, -n:].reshape(-1), reduction='mean'))


def valid_single_answer(text):
    match = re.fullmatch(r'\s*\[BE\]((?:(?!\[BE\]|\[EE\]).)+)\[EE\]\s*', text, re.DOTALL)
    return bool(match and match.group(1).strip())


def evaluate_historical_adapter(adapter_path, base_model_path, rows, output_dir,
                                max_seq_length=9000, max_new_tokens=64,
                                progress_factory=None, description='Historical dev'):
    if not torch.cuda.is_available():
        raise RuntimeError('Historical NF4 evaluation requires CUDA')
    adapter_path, base_model_path, output_dir = map(Path, (adapter_path, base_model_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = None
    metrics = []
    try:
        model = AutoModelForCausalLM.from_pretrained(str(base_model_path),
            quantization_config=quantization, device_map={'': torch.cuda.current_device()},
            low_cpu_mem_usage=True, local_files_only=True)
        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
        model.config.use_cache = False
        model.eval()
        settings = {'quantization': '4-bit NF4', 'double_quantization': True,
            'compute_dtype': 'float16', 'loaded_base_dtype': str(model.dtype),
            'attention_implementation': getattr(model.config, '_attn_implementation', None),
            'decoding': 'greedy', 'do_sample': False, 'num_generations': 1,
            'aggregation_strategy': 'first', 'use_cache': False,
            'max_seq_length': max_seq_length, 'max_new_tokens': max_new_tokens,
            'gold_source': 'prepared development gold_output',
            'prediction_normalization': 'Historical generate_answer_from_prompt cleaning',
            'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
            'adapter_path': str(adapter_path),
            'adapter_sha256': sha256(adapter_path / 'adapter_model.safetensors')}
        (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
        inputs = [{'question_id': q['question_id'], 'question': q['question'],
            'resources': q['resources'], 'rendered_prompt': q['prompt'],
            'gold_output': q['canonical'], 'prompt_id': q['prompt_id']} for q in rows]
        (output_dir / 'evaluation_inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inputs))
        iterable = progress_factory(rows, desc=description, leave=False) if progress_factory else rows
        with torch.no_grad():
            for q in iterable:
                # This is the historical evaluator's generation function, including
                # tokenizer defaults, no KV cache and prediction cleaning.
                prediction = generate_answer_from_prompt(model, tokenizer, q['prompt'],
                    max_seq_length=max_seq_length, max_new_tokens=max_new_tokens, do_sample=False)
                valid = valid_single_answer(prediction)
                row = {'question_id': q['question_id'], 'prediction': prediction,
                    'question_type': 'factoid', 'body': q['example'].body,
                    'source_path': q['example'].source_path,
                    'canonical': q['canonical'],
                    'canonical_nll': canonical_nll(model, tokenizer, q['prompt'], q['canonical'], max_seq_length),
                    'canonical_exact_match': float(prediction.strip() == q['canonical']),
                    'single_answer_format_valid': float(valid),
                    'answer_span_count': prediction.count('[BE]'),
                    'prediction_tokens': len(tokenizer.encode(prediction, add_special_tokens=False))}
                assert math.isfinite(row['canonical_nll'])
                metrics.append(row)
        official = evaluate_with_bioasq_java(
            prediction_rows=metrics,
            examples_by_key={(q['question_id'], 'factoid'): q['example'] for q in rows},
            model_label=output_dir.name,
            model_dir=output_dir,
            args=SimpleNamespace(
                bioasq_java_jar=str(Path(__file__).resolve().parents[1] /
                    'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar'),
                bioasq_java_heap='512m', bioasq_java_version=5),
            include_per_question=True,
        )
        official_by_id = {row['question_id']: row for row in official['per_question']}
        with (output_dir / 'per_question_metrics.jsonl').open('w') as f:
            for row in metrics:
                score = official_by_id[row['question_id']]
                row['mrr'] = float(score['mrr'])
                row['top1'] = float(score['strict_accuracy'])
                row['format_valid_top1'] = row['single_answer_format_valid'] * row['top1']
                row['scoring_backend'] = 'bioasq_java'
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        keys = ['canonical_nll', 'canonical_exact_match',
                'single_answer_format_valid', 'format_valid_top1', 'prediction_tokens']
        summary = {k: sum(r[k] for r in metrics) / len(metrics) for k in keys}
        factoid = official['aggregate']['by_type']['factoid']['metrics']
        summary.update(mrr=factoid['mrr'], top1=factoid['strict_accuracy'],
                       lenient_accuracy=factoid['lenient_accuracy'])
        summary.update(questions=len(metrics), evaluation_protocol='historical_full_dev_greedy_nf4_fp16',
                       scoring_backend='bioasq_java')
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        assert sha256(adapter_path / 'adapter_model.safetensors') == settings['adapter_sha256']
        return summary, metrics
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
