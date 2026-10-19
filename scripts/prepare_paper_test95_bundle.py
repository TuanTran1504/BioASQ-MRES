#!/usr/bin/env python3
"""Create a portable test95 bundle from the project's existing official golden files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gadi_sft_8b_starter/scripts'))
from run_sft_test_evaluation import test_panel
sys.path.insert(0, str(ROOT))
from scripts.paper_replication import save, digest, panel


def prepare(output, test_dir):
    inputs, gold, batches = test_panel(test_dir, set())
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('test_inputs.jsonl', inputs), ('test_gold.jsonl', gold)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    save(output / 'manifest.json', {'question_count': 95, 'batches': batches,
        'file_hashes': {name: digest(output / name) for name in ('test_inputs.jsonl', 'test_gold.jsonl')}})
    panel(output)
    print('Prepared 95 factoids across four official batches:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-dir', type=Path, default=ROOT / 'data/Task13BTest')
    a = p.parse_args()
    prepare(a.output, a.test_dir)
