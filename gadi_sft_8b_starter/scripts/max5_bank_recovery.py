"""Validate saved max-five draws before copying them into a fresh recovery run."""
import json
import math
from pathlib import Path

from expansion_dpo_data import response_id
from prepare_matched_8b_sft import digest, read
from run_single_answer_sampling import sample_seed


def response_key(row):
    return row.get('split'), row.get('question_id'), row.get('draw')


def recover_rows(directory, config, source, panel, model, config_sha256):
    """Accept an exact prefix of the frozen protocol, including part of a question.

    Keep malformed model completions: parsing success must not determine which
    draws survive. Only an interrupted, unterminated JSONL tail may be skipped.
    The original files are read-only throughout recovery.
    """
    directory = Path(directory).resolve()
    manifest_path, path = directory/'manifest.json', directory/'responses.jsonl'
    manifest_sha256 = digest(manifest_path)
    manifest = read(manifest_path)
    if (manifest.get('status') not in ('running', 'failed')
            or manifest.get('model_key') != model
            or manifest.get('smoke_test') is not False
            or manifest.get('full_dataset') is not True
            or manifest.get('gold_blind_generation') is not True
            or manifest.get('formulation') != 'expansion'
            or manifest.get('configuration') != config
            or manifest.get('dpo_config_sha256') != config_sha256
            or manifest.get('source') != source):
        raise ValueError('Recovery source is not a matching interrupted full max-five bank')
    question_ids = {s:[r['question_id'] for r in rr] for s,rr in panel.items()}
    if manifest.get('question_ids') != question_ids:
        raise ValueError('Recovery question partitions or ordering changed')
    expected = [(s,r['question_id'],d,r['messages'][:2])
                for s,rr in panel.items() for r in rr
                for d in range(config['sampled_responses']+1)]
    content = path.read_bytes()
    import hashlib
    content_sha256 = hashlib.sha256(content).hexdigest()
    lines = content.splitlines(keepends=True)
    rows, ignored_tail_bytes = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines)-1 and not line.endswith(b'\n'):
                ignored_tail_bytes = len(line)
                break
            raise ValueError(f'Malformed saved JSONL record {i+1}; refusing recovery') from None
        if len(rows) >= len(expected) or not isinstance(row, dict):
            raise ValueError('Unexpected saved response record')
        split, qid, draw, prompt = expected[len(rows)]
        raw = row.get('raw_response')
        if (response_key(row) != (split,qid,draw) or row.get('prompt') != prompt
                or row.get('model_key') != model or not isinstance(raw,str)
                or type(row.get('draw')) is not int or type(row.get('seed')) is not int
                or row['seed'] != sample_seed(qid,draw+1,config['seed'])
                or row.get('decoding') != ('sampling' if draw else 'greedy')
                or row.get('response_id') != response_id(model,qid,draw,raw)):
            raise ValueError(f'Changed or out-of-order saved response {i+1}')
        for field, maximum in [('prompt_tokens',config['max_seq_length']-config['max_new_tokens']),
                               ('output_tokens',config['max_new_tokens'])]:
            if type(row.get(field)) is not int or not 0 <= row[field] <= maximum:
                raise ValueError(f'Invalid saved {field} at record {i+1}')
        seconds = row.get('generation_seconds')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid saved timing at record {i+1}')
        rows.append(row)
    if not rows:
        raise ValueError('No complete saved draws to recover')
    if digest(path) != content_sha256 or digest(manifest_path) != manifest_sha256:
        raise ValueError('Recovery source changed while reading; stop the old job first')
    return rows, {'directory':str(directory), 'manifest_sha256':manifest_sha256,
                  'responses_sha256':content_sha256, 'reused_responses':len(rows),
                  'remaining_responses':len(expected)-len(rows),
                  'ignored_unterminated_tail_bytes':ignored_tail_bytes,
                  'generation_worker_sha256':manifest.get('worker_sha256')}
