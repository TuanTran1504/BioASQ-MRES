# Expansion-generation snapshots

This directory contains lightweight outputs from three complete 160-question
Gadi equivalent-expansion runs. It excludes model weights, adapters,
checkpoints, prompts with BioASQ source text, snippets, gold answers and
`examples.jsonl`.

| Run | Schema-compliant responses | Accepted candidates | Invalid candidates |
| --- | ---: | ---: | ---: |
| Qwen3-8B base | 153 / 160 | 1,225 | 258 |
| Gemma-3-27B base | 142 / 160 | 605 | 24 |
| Qwen3-8B expansion SFT | 159 / 160 | 665 | 78 |

Each run directory contains:

- `generations.jsonl`: model responses and parsing diagnostics with the
  original question text removed;
- `candidates.jsonl`: accepted parsed answer candidates;
- `invalid_candidates.jsonl`: rejected candidates and rejection reasons; and
- `status.json`: aggregate generation and parsing status.

Candidate acceptance is structural and does not mean that an answer is
correct. Official BioASQ coverage must be computed against the separately held
local evaluation examples. Source and exported file hashes are recorded in
`manifest.json`.

Regenerate the snapshot from locally retrieved ignored artifacts:

```bash
python scripts/export_gadi_generation_results.py
```
