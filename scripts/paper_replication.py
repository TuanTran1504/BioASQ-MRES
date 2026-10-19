#!/usr/bin/env python3
"""Pinned sources, gold-blind prompting and common test95 scoring for paper baselines."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'gadi_sft_8b_starter/configs/paper_replication_sources.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def normalized(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def fetch(repository, commit, relative, target, expected, download):
    if not target.exists():
        if not download:
            raise FileNotFoundError(f'Missing pinned source: {target}; prepare with --download')
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/{repository}/{commit}/{relative}', timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Downloaded upstream source differs from pinned hash')
        target.write_bytes(content)
    if digest(target) != expected:
        raise ValueError(f'Pinned source changed: {relative}')


def sources(directory, download=False):
    config = read(PINS)
    for relative, info in config['unitor']['files'].items():
        fetch(config['unitor']['repository'], config['unitor']['commit'], relative,
              directory / 'unitor' / relative, info['sha256'], download)
    for info in (config['gptprompt']['request_file'], config['gptprompt']['notebook']):
        fetch(config['gptprompt']['repository'], config['gptprompt']['commit'], info['path'],
              directory / 'gptprompt' / info['path'], info['sha256'], download)
    return config


def panel(bundle):
    state = read(bundle / 'manifest.json')
    for name, expected in state['file_hashes'].items():
        if digest(bundle / name) != expected:
            raise ValueError('Test bundle hash differs')
    inputs, gold = records(bundle / 'test_inputs.jsonl'), records(bundle / 'test_gold.jsonl')
    ids = {r['question_id'] for r in inputs}
    if len(inputs) != 95 or len(ids) != 95 or len(gold) != 95 or {r['question_id'] for r in gold} != ids:
        raise ValueError('Expected identical 95-question input/gold membership')
    if any(set(r) != {'question_id', 'question', 'snippets'} or not r['snippets'] for r in inputs):
        raise ValueError('Generation inputs must contain only question ID, question and evidence')
    labels = {r['question_id']: r for r in gold}
    if any(labels[r['question_id']]['question'] != r['question'] for r in inputs):
        raise ValueError('Input/gold question text differs')
    if sorted(Counter(r['batch'] for r in gold).values()) != [20, 22, 26, 27]:
        raise ValueError('Wrong official four-batch membership')
    return inputs, labels


def audit_training(train, validation, inputs):
    test_ids = {r['question_id'] for r in inputs}
    test_text = {normalized(r['question']) for r in inputs}
    for rows in (train, validation):
        if test_ids & {r['id'] for r in rows} or test_text & {normalized(r['input_1']) for r in rows}:
            raise ValueError('Released training/validation data overlaps test95')
    return {
        'training_rows': len(train), 'validation_rows': len(validation),
        'training_types': dict(Counter(r['type'] for r in train)),
        'validation_types': dict(Counter(r['type'] for r in validation)),
        'train_validation_repeated_question_texts': sorted(
            {normalized(r['input_1']) for r in train} & {normalized(r['input_1']) for r in validation}),
        'test95_overlap_by_id_or_question': 0,
        'split_policy': 'Preserve authors released splits; repeated texts disclosed, not silently removed',
    }


def gpt_template(source_dir, inputs):
    """Read inert notebook AST and released request data; never execute upstream code."""
    config = read(PINS)['gptprompt']
    notebook = read(source_dir / 'gptprompt' / config['notebook']['path'])
    formats = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for function in tree.body:
            if not isinstance(function, ast.FunctionDef) or function.name != 'generate_format':
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "ordered by decreasing confidence" in node.value:
                    formats.append(node.value)
    if len(set(formats)) != 1:
        raise ValueError('Cannot resolve unique released factoid formatting prompt')
    request = records(source_dir / 'gptprompt' / config['request_file']['path'])[0]['body']
    prefix = request['messages'][:-1]
    if request['model'] != 'gpt-4-turbo' or [m['role'] for m in prefix] != ['system', 'user', 'assistant', 'user', 'assistant']:
        raise ValueError('Released model/example structure differs')
    example_questions = [m['content'].split('Question:', 1)[1].split('\n', 1)[0].strip()
                         for m in prefix if m['role'] == 'user']
    if {normalized(q) for q in example_questions} & {normalized(r['question']) for r in inputs}:
        raise ValueError('Few-shot example overlaps test question')
    return {'prefix': prefix, 'factoid_format': formats[0], 'example_questions': example_questions,
            'model': 'gpt-4-turbo', 'max_tokens': 100,
            'decoding': 'Released batch request defaults (temperature omitted)',
            'nominal_shots': 1, 'actual_examples': 2,
            'provenance': config,
            'limitation': 'Batch-3 Style-2 notebook; no identifiable batch-4 submission checkpoint/request released. '
                          'The two exact-answer examples are preserved from the released batch-2 request.'}


def gpt_payload(example, template):
    # No held-out answer labels or ideal-answer hints enter this function.
    passage = '\n\n'.join(s['text'] for s in example['snippets'])
    return {'model': template['model'], 'max_tokens': template['max_tokens'],
            'messages': template['prefix'] + [{'role': 'user',
                'content': f"Passage: {passage}\n\n Question:{example['question']} \n\n {template['factoid_format']}"}]}


def gpt_parse(raw):
    # Literal released parser: commas inside an entity are split as well.
    found = re.search(r'\[([^\]]+)\]', raw)
    return [s.strip().strip('"').strip("'") for s in found.group(1).split(',')] if found else []


def unitor_parse(raw):
    # Accept the [BI] alternate opening tag explicitly present in the authors' prompt.
    return [m.strip() for m in re.findall(r'\[(?:BE|BI)\](.*?)\[EE\]', raw, re.S) if m.strip()]


def score(bundle, generations, output, parser):
    """Score all 95 questions, including empty/failed parses, in generated order."""
    inputs, gold = panel(bundle)
    saved = records(generations)
    by_id = {r['question_id']: r for r in saved}
    if len(saved) != 95 or len(by_id) != 95 or set(by_id) != set(gold):
        raise ValueError('Only complete, unique 95-question runs may be scored')
    candidates, ordered = [], {}
    for row in inputs:
        qid = row['question_id']
        answers = parser(by_id[qid]['raw_response'])
        ordered[qid] = answers
        candidates.extend({'question_id': qid, 'answer': a} for a in dict.fromkeys(answers) if a)
    sys.path.insert(0, str(ROOT))
    from src.notebook_workflows.coverage_comparison import official_candidate_matches
    output.mkdir(parents=True, exist_ok=True)
    official = output / 'official'
    official.mkdir(exist_ok=True)
    matches = official_candidate_matches([{**r, 'gold_aliases': gold[r['question_id']]['gold_aliases']} for r in inputs],
        candidates, official, jar_path=ROOT / 'third_party/Evaluation-Measures/flat/BioASQEvaluation/dist/BioASQEvaluation.jar') if candidates else {}
    per_question = []
    for qid, answers in ordered.items():
        rank = next((i for i, a in enumerate(answers, 1) if matches.get((qid, a), False)), None)
        per_question.append({'question_id': qid, 'batch': gold[qid]['batch'], 'answers': answers,
            'first_accepted_rank': rank, 'mrr_at5': 1 / rank if rank and rank <= 5 else 0.,
            'coverage_at5': float(rank is not None and rank <= 5), 'coverage_full_pool': float(rank is not None)})
    def metrics(rows):
        return {'question_count': len(rows), **{k: sum(r[k] for r in rows) / len(rows)
                for k in ('mrr_at5', 'coverage_at5', 'coverage_full_pool')}}
    batches = {b: metrics([r for r in per_question if r['batch'] == b]) for b in sorted({r['batch'] for r in per_question})}
    result = {'metrics': metrics(per_question), 'per_batch': batches,
        'macro_batch_mrr_at5': sum(r['mrr_at5'] for r in batches.values()) / 4,
        'empty_parse_count': sum(not r['answers'] for r in per_question),
        'output_ceiling_count': sum(r.get('finish_reason') == 'length' for r in saved),
        'scorer': 'Existing official BioASQ Java candidate matcher; first five in generated order; no reranker',
        'exposure': 'Historically examined project test95, not a fresh untouched holdout'}
    save(output / 'comparison_per_question.json', per_question)
    save(output / 'comparison_summary.json', result)
    return result


def prepare(source_dir, bundle, output, download=False):
    config = sources(source_dir, download)
    inputs, _ = panel(bundle)
    base = source_dir / 'unitor/data/dataset/answer_generation'
    audit = audit_training(read(base / 'train_set_answGEN.json'), read(base / 'dev_set_answGEN.json'), inputs)
    template = gpt_template(source_dir, inputs)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'source_audit.json', audit)
    save(output / 'gptprompt_template.json', template)
    save(output / 'prepared.json', {'sources': config, 'bundle': str(bundle.resolve()),
        'source_dir': str(source_dir.resolve()), 'source_audit': audit,
        'test_inputs_sha256': digest(bundle / 'test_inputs.jsonl'),
        'test_gold_sha256': digest(bundle / 'test_gold.jsonl'),
        'gpt_template_sha256': digest(output / 'gptprompt_template.json')})
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.sources, a.bundle, a.output, a.download), indent=2))
