# Matched 8B SFT evaluation

Evaluate the completed Llama-3.1-8B and Qwen3-8B pairs on the same 160 development
questions, with all scientific snippets retained. This launcher performs inference
only. It does not retrain models or submit Ministral jobs.

| Condition | Adapter and prompt | Decoding | Requests per question |
|---|---|---|---|
| `original_greedy` | Matched single-answer SFT, its training prompt | Greedy | 1 |
| `original_sampling10` | Same single-answer adapter and prompt | Ten seeded draws, temperature 0.8, top-p 0.95, top-k 0 | 10 |
| `expansion_greedy` | Matched expansion SFT, its training prompt | Greedy equivalent expressions | 1 |

All arms use native chat templates, a 6,144-token total limit and 512 output tokens.
Qwen thinking stays disabled. Single-answer sampling preserves the exact unmarked
snippet strings used by the new matched training, unlike the historical Qwen
sampler's default `[BS]`/`[ES]` markers. Gold aliases remain in archived examples
for offline scoring and never enter generation prompts. Each attempted draw,
including parse failures, is retained. Seeds derive from seed 3407, question ID
and draw index. Candidates are deduplicated by case and whitespace; ranking is
generated/draw order, with at most five alternatives for final metrics and
coverage at ten as an offline diagnostic.

## Completed training runs

The configuration `configs/matched_8b_evaluation.json` points to these full runs:

| Model | Single-answer training job | Expansion training job |
|---|---|---|
| Llama-3.1-8B | 180794568 | 180794570 |
| Qwen3-8B | 180794572 | 180794574 |

The launcher validates both completed adapters, their nonempty weights and saved
configuration, pinned training matrix and shared base revision. It revalidates
the 1,296/144/160 question separation and pins the dev export's SHA256. It saves
adapter file hashes and training provenance in each evaluation manifest. Both
formulations use the same fitting/validation questions and evidence, with their
own internal validation loss for checkpoint selection. Absolute losses between
formulations are not comparable.

## Submit on Gadi

From the bundle directory, after pulling the repository:

```bash
bash scripts/submit_matched_8b_evaluation.sh
```

This validates both model pairs before submission and prints a submission TSV.
For each backbone it submits a four-question smoke job and an eight-hour full
evaluation job dependent on successful smoke exit. Every job runs all three arms
in separate processes. If an arm produces zero parseable answers in the smoke,
the dependent full evaluation does not run. Smoke results validate execution and
format, not answer quality. There are four jobs in total: two smokes and two full
evaluations. To submit just one backbone, pass `llama31` or `qwen3`.

Monitor the printed job IDs:

```bash
qstat -swx <smoke-ID> <full-ID>
```

Outputs are under:

```text
outputs/matched_8b_evaluation/<llama31|qwen3>-8b-evaluation-<PBS_JOBID>/
```

Check `manifest.json` for status `complete`, `smoke_test: false`, and
`expected_questions: 160`. Child arm statuses must also be complete. Original
arms store files directly below their condition directory; expansion has one
timestamped child. Failed runs preserve their available raw outputs and manifest
error. Inspect logs before any retry; every new job gets a fresh output directory.

## Score completed full evaluations

In your interactive shell, load the environment before analysis:

```bash
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
java -version
```

The repository's bundled per-question adapter supports Java 8+. A JDK is only
needed if its bundled source/evaluator hashes no longer match and recompilation
is necessary. No GPU is needed for analysis.

Replace the two job IDs with the full evaluation IDs printed at submission:

```bash
python ../scripts/analyze_matched_8b_evaluation.py \
  outputs/matched_8b_evaluation/llama31-8b-evaluation-<FULL_JOB_ID>.gadi-pbs \
  outputs/matched_8b_evaluation/qwen3-8b-evaluation-<FULL_JOB_ID>.gadi-pbs
```

The analyzer checks identical archived question/snippet records and complete
generation denominators, native prompt hashes, adapter identity, shared backbone,
decoding settings, draw counts/seeds and absence of evidence truncation. Complete
experiment directories can also be copied locally for scoring; archived prompts
are used rather than requiring the original Gadi prompt path.

Each experiment receives `comparison_summary.json` and
`comparison_per_question.json`. Scores use the official BioASQ Java candidate
matcher: MRR@5, strict/lenient accuracy and coverage at 1/5/10. The summary includes
parse success, candidate counts and generation request/token/time totals (time
excludes loading and preflight). It directly reports all three differences:

- Ten-sample original minus greedy original.
- Greedy expansion minus greedy original: equal request budget.
- Greedy expansion minus ten-sample original: quality against a larger request budget.

All contrasts have 10,000-resample paired question bootstrap intervals with seed
3407. Sampling diagnostics also distinguish coverage in the first 1/5/10 draws
from coverage among unique ranked candidates. Zero-parse arms receive an explicit
warning. These are one-seed development comparisons; they do not establish unseen
test improvement or a DPO effect. Raw outputs, weights and dataset text stay in
ignored `outputs/`; permitted aggregate results can be published separately.
