#!/usr/bin/env python3
"""Run the released GPTPrompt factoid method within the separately authorized $10 cap."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_replication import read, save, digest, panel, gpt_payload, gpt_parse, score

INPUT_RATE, OUTPUT_RATE = 10 / 1_000_000, 30 / 1_000_000
AUTHORIZED_CAP = 10.0


def request_key(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reserve(payload):
    # Byte-BPE cannot require more tokens than UTF-8 bytes. Extra framing allowance.
    return (len(json.dumps(payload, ensure_ascii=False).encode()) + 2048) * INPUT_RATE + payload['max_tokens'] * OUTPUT_RATE


def affordable(spent, amount, cap):
    return math.isfinite(cap) and 0 < cap <= AUTHORIZED_CAP and spent + amount <= cap


def run(prepared, output, keyfile, cap, max_calls=95):
    if not math.isfinite(cap) or not 0 < cap <= AUTHORIZED_CAP:
        raise ValueError('This experiment has an authorized total cap of US$10')
    config = read(prepared / 'prepared.json')
    bundle = Path(config['bundle'])
    if any(digest(bundle / f'test_{kind}.jsonl') != config[f'test_{kind}_sha256'] for kind in ('inputs', 'gold')):
        raise ValueError('Prepared test files changed')
    if digest(prepared / 'gptprompt_template.json') != config['gpt_template_sha256']:
        raise ValueError('Prepared prompting template changed')
    inputs, _ = panel(bundle)
    template = read(prepared / 'gptprompt_template.json')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'prepared_sha256': digest(prepared / 'prepared.json'), 'pricing': {'input': INPUT_RATE, 'output': OUTPUT_RATE},
                'authorization': 'User approved separate US$10 GPTPrompt replication cap', 'maximum_usd': cap}
    if (output / 'experiment.json').exists() and read(output / 'experiment.json') != identity:
        raise ValueError('Existing experiment identity or spending cap differs')
    save(output / 'experiment.json', identity)
    lock = output / 'paid-run.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    try:
        ledger_path = output / 'spend-ledger.json'
        ledger = read(ledger_path) if ledger_path.exists() else {'attempts': []}
        spent = sum(a['charged_or_reserved_usd'] for a in ledger['attempts'])
        count, rows, key = 0, [], None
        for example in inputs:
            payload = gpt_payload(example, template)
            sha = request_key(payload)
            cached = output / 'cache' / (sha + '.json')
            if cached.exists():
                envelope = read(cached)
                if envelope['payload'] != payload:
                    raise ValueError('Cached request differs')
            else:
                # Never repeat a possibly billed request automatically, including across restarts.
                if any(a['request_sha256'] == sha for a in ledger['attempts']):
                    print('Stopped: unresolved previous attempt; inspect ledger before retrying.')
                    break
                amount = reserve(payload)
                if count >= max_calls or not affordable(spent, amount, cap):
                    print('Stopped before next call: call limit or reserved spending cap.')
                    break
                if key is None:
                    from src.utility.eval_openai import read_api_key
                    key = read_api_key(keyfile)
                attempt = {'question_id': example['question_id'], 'request_sha256': sha, 'status': 'in_flight',
                           'charged_or_reserved_usd': amount, 'started_at': datetime.now(timezone.utc).isoformat()}
                ledger['attempts'].append(attempt)
                save(ledger_path, ledger)
                spent += amount
                count += 1
                try:
                    import requests
                    response = requests.post('https://api.openai.com/v1/chat/completions', json=payload,
                        headers={'Authorization': 'Bearer ' + key}, timeout=(15, 180))
                    if response.status_code != 200:
                        attempt.update(status='http_error', http_status=response.status_code)
                        save(ledger_path, ledger)
                        print(f'Stopped: API HTTP {response.status_code}. No automatic retry or substitute model.')
                        break
                    data = response.json()
                    save(output / 'responses' / (sha + '.json'), data)
                    usage = data.get('usage', {})
                    charge = usage['prompt_tokens'] * INPUT_RATE + usage['completion_tokens'] * OUTPUT_RATE
                    if charge > amount:
                        raise ValueError('Reported usage exceeded conservative reservation')
                    envelope = {'payload': payload, 'response': data}
                    save(cached, envelope)
                    attempt.update(status='complete', usage=usage, charged_or_reserved_usd=charge)
                    spent += charge - amount
                    save(ledger_path, ledger)
                except Exception as exc:
                    attempt.update(status='unresolved_error', error_type=type(exc).__name__)
                    save(ledger_path, ledger)
                    print('Stopped after an unresolved request; conservative charge retained.')
                    break
            data = envelope['response']
            choice = data['choices'][0]
            rows.append({'question_id': example['question_id'], 'request_sha256': sha,
                'raw_response': choice['message'].get('content') or '', 'finish_reason': choice['finish_reason'],
                'returned_model': data['model'], 'usage': data['usage']})
            if len(rows) % 10 == 0:
                print(f'Completed {len(rows)}/95; charged/reserved US${spent:.4f}', flush=True)
        (output / 'generations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        state = {'status': 'complete' if len(rows) == 95 else 'incomplete', 'completed_questions': len(rows),
                 'charged_or_reserved_usd': spent, 'maximum_usd': cap, 'model': template['model'],
                 'actual_examples': template['actual_examples'], 'limitation': template['limitation']}
        save(output / 'manifest.json', state)
        print(json.dumps(state, indent=2))
    finally:
        lock.unlink()
    if len(rows) == 95:
        print(json.dumps(score(bundle, output / 'generations.jsonl', output, gpt_parse), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api-key-file', type=Path, default=ROOT / 'open_ai_api.txt')
    p.add_argument('--max-usd', type=float, default=10.)
    p.add_argument('--max-new-calls', type=int, default=95)
    a = p.parse_args()
    run(a.prepared, a.output, a.api_key_file, a.max_usd, a.max_new_calls)
