# Ten-sample rerankers for the fixed compactness configurations

The completed test95 experiment used the original frozen SFT logistic reranker
on both SFT and DPO-generated pools. This continuation trains **two new logistic
rerankers** on their matching generator configurations, then applies them to the
**same saved test95 pools**. There is no new generator training or paid judging.

| Reranker | Llama fitting bank | Qwen3 fitting bank | Ministral fitting bank |
|---|---|---|---|
| Strict | Native loss-selected max-five SFT | Compactness DPO epoch 1 | Native loss-selected max-five SFT |
| Alternative | Compactness DPO epoch 1 | Compactness DPO epoch 1 | Compactness DPO epoch 1 |

The existing SFT reranker remains the baseline. No additional baseline reranker
is fitted. Both new rerankers are also scored on all three configurations to
make transfer effects visible, without choosing a winner on test95.

## Generation and reuse

Each required model/policy supplies **one greedy and ten sampled responses for
each of 1,296 fitting and 144 validation questions**. Each response contains at
most five answer candidates. Greedy and sampling form separate candidate pools;
greedy is not silently added to the ten-sample pool. Duplicates are merged.

Sampling uses temperature 0.8, top-p 0.95, top-k 0, seed 3407, at most 512 output
tokens per response and the native max-five SFT prompt. Complete evidence is
retained. Input tokens are checked against the native tokenizer and the existing
8,704-token bank allocation; oversized evidence fails preparation rather than
being truncated. Gold answers are separate and never enter generation prompts.

All three DPO banks are needed. Only Llama and Ministral require additional SFT
companion samples; Qwen3 is DPO in both new configurations. Existing SFT banks
already contain greedy plus four samples on fitting and validation. Completed
epoch-one selection runs contain greedy plus four DPO validation samples.
These responses are validated and reused byte-for-byte in their raw text.

The old four samples used stochastic **seed indices 2–5**. The new ten-sample
budget uses indices **1–10**, as test generation does. Consequently the missing
six indices are **1 and 6–10**, not 5–10. Greedy separately uses seed index 1.
All previously missing DPO fitting responses are generated from the verified
epoch-one adapters. Repeated answers, parse failures and capped draws remain in
the requested-draw denominator; no failed draw is replaced by an extra attempt.

This requires 62,640 new full-run responses, excluding smoke checks: 42,768 DPO
fitting responses, 2,592 additional DPO validation responses and 17,280 additional
SFT companion responses. Saved responses provide the rest. This is a larger GPU
experiment than the previous four-sample bank generation.

Generation is split into chunks of at most 64 questions. Each completed chunk is
validated and hashed. Recovery reuses committed chunks, preserves interrupted
chunks for inspection and regenerates only an unfinished chunk. Generation jobs
request up to 48 hours each; the launcher also performs a short technical smoke
check for each model before its full bank job.

## Training and evaluation

Features, relation labels, pooling and the logistic model are unchanged. Source
frequency is presence in a draw divided by **ten**, at most once per answer per
draw. Model votes count models, not responses. Official Java acceptance provides
the binary training labels; it does not imply that an unmatched answer is
semantically incorrect.

Each classifier fits only the 1,296 fitting questions. Its regularization value
is selected from C = 0.1, 1, 10 by the mean validation MRR@5 over its greedy and
ten-sample pools, equally weighted. Ties keep the first C. Validation questions
are not added to fitting after selection. The 160-question dev panel and the 95
test questions are excluded from both fitting and regularization selection.

The CPU scorer reads the completed comparison
`outputs/compactness_test95/comparison-181558664.gadi-pbs`. Candidate strings,
source support, frequencies and official acceptance labels are pinned and reused.
It first reproduces all six old frozen-reranker results exactly, then scores both
new classifiers on the six existing pools. It reports per-batch scores, paired
bootstrap intervals and coverage gain/loss IDs.

Primary comparisons distinguish:

1. New versus old reranker **on the identical strict or alternative pool**.
2. Each matching DPO pipeline versus the original all-SFT baseline pipeline.

Full-pool coverage cannot change when only the reranker changes. Coverage@5 and
MRR@5 can change. The new rerankers also differ from the old baseline in training
sample budget (ten versus four), so an improvement cannot be attributed solely
to matching DPO source models. A separate ten-sample SFT baseline would be needed
to isolate that factor. Test95 has already been inspected; these results remain
exploratory. No checkpoints, C values or deployment choices are selected on it.

## Submit on Gadi

From the starter directory:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_compactness_matched_rerankers.sh
```

The launcher validates sources before submitting eight jobs: three GPU smoke/full
chains, one dependent CPU job fitting both rerankers, and one CPU scoring job
depending on successful training. The submission contains frozen code and job
files. Later checkout edits do not change queued workers.

To monitor:

```bash
record=$(ls -t outputs/compactness_matched_rerankers/submission-*/jobs-*.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

If a bank fails or times out, wait until the earlier chain is inactive, then
resume the **same** submission, replacing the example path below with the printed
path. The launcher refuses concurrent writers. It creates a new job record and
new training/scoring output directories while retaining the saved bank chunks.

```bash
bash scripts/submit_compactness_matched_rerankers.sh --resume-submission \
  outputs/compactness_matched_rerankers/submission-TIMESTAMP-PID
```

When scoring has exit status 0:

```bash
module load python3/3.12.13
python3 - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('outputs/compactness_matched_rerankers').glob(
    'submission-*/comparison-*/comparison_summary.json'), key=lambda p: p.stat().st_mtime)
if not paths:
    raise SystemExit('Scoring summary not ready; inspect the score job logs.')
p = paths[-1]
d = json.loads(p.read_text())
print('Results:', p)
for mode in ('greedy', 'sampling10'):
    print('\n' + mode)
    for setting, selector in (('baseline', 'max5'), ('strict', 'strict_matched'),
                              ('alternative', 'alternative_matched')):
        m = d['conditions'][f'{setting}_{mode}_{selector}_reranker']['metrics']
        print(f"  {setting}: MRR@5={m['mrr_at5']:.6f}, coverage@5={m['coverage_at5']:.4f}")
    for setting in ('strict', 'alternative'):
        c = d['paired_contrasts'][f'{setting}_{mode}_new_minus_old_same_pool']['metrics']['mrr_at5']
        print(f"  {setting}, new minus old: {c['difference']:+.6f}, CI={c['paired_bootstrap_95ci']}")
PY
```
