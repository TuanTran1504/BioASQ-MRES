# Pairwise ranking and conservative redundancy control

This experiment asks whether better selection can recover accepted answers
already present in the original and max-five SFT ensemble pools. It reuses
the frozen logistic rerankers from **181350943**, their cached 1,296 fitting
and 144 validation questions, and the scored test comparison **181287698**
(95 questions). It does not generate answers, edit candidate strings, create
new official labels, call an API, or train DPO.

One CPU job produces a controlled comparison:

| Training objective | Final selection |
| --- | --- |
| Archived pointwise logistic baseline | Original score order |
| Archived pointwise logistic baseline | Validation-selected typography penalty |
| New pairwise linear logistic ranker | Original score order |
| New pairwise linear logistic ranker | Validation-selected typography penalty |

The primary comparisons use original-trained rankers on original pools and
max-five-trained rankers on max-five pools, for greedy and ten-sample
generation. All three selectors (original-only, max-five-only, shared) are
also scored on all four pools for transfer diagnostics. These additional
results do not select a winner on the test data.

## Pairwise objective

The word TF-IDF, character TF-IDF and metadata features match the current
baseline, including relation labels, frequency, source ranks and evidence
overlap. Vectorizers fit only fitting candidates. No validation/test text or
gold aliases enter their vocabulary or model inputs.

For each fitting question and pool separately, form accepted/nonaccepted
candidate pairs. The model learns a positive score difference for the accepted
candidate. Both orientations enter logistic regression, with no intercept or
class balancing. Each eligible question/pool has equal total training weight;
weights are then rescaled to mean one. Questions/pools without both label
classes cannot contribute a ranking comparison and are counted as skipped.
Official nonacceptance does **not** mean biomedical incorrectness.

Use all such pairs up to **128 pairs per question/pool**. Larger groups are
sampled uniformly without replacement using a stable question/pool seed
derived from 3407. Pair counts, capped groups and skipped groups are exported.
Greedy and sampled pools are never mixed inside a pair. No hard negatives are
mined from inspected test cases.

For each selector, choose C from **0.1, 1, 10** using mean validation MRR@5
over that selector's declared greedy and four-sample validation pools.
Ties keep the first C. This is a validation-selected objective comparison,
not a fixed-C feature ablation. The baseline remains the original archived
validation-selected logistic model.

## Redundancy control

Start with a deliberately narrow comparison signature: case/whitespace,
typographic hyphens, between-letter hyphen versus space, and terminal full
stops. For example, `X-inactivation` and `X inactivation.` can be considered
redundant for this selection policy. All digits, internal decimal points,
percent signs, slashes, Greek letters and word order remain in the signature.
It does not infer that `F508del` and its long genetic description are aliases,
or that two numerical ranges or Turner karyotypes are equivalent.

Select the first five by reciprocal base rank minus a penalty if an expression
with the same signature has already been selected. The original first-ranked
candidate remains first. All unselected candidates follow in their original
score order. No strings or candidates are added, deleted or rewritten;
full-pool coverage must stay constant. An accepted punctuation variant may be
demoted by this policy: that tradeoff is measured rather than assumed safe.

Choose the penalty from **0, 0.1, 0.3, 1.0**, separately for each selector and
objective, using only its declared validation pools. Zero is evaluated first
and wins ties, allowing validation to disable redundancy control. Select C
on plain ranking first, then the penalty; this is not a joint C/penalty search.

## Run on Gadi

From `gadi_sft_8b_starter`:

```bash
git pull --ff-only
bash scripts/submit_pairwise_sft_rerankers.sh
```

The launcher verifies cached pools, source checkpoints, hashes, runtime
compatibility and question separation. It freezes the Python code and PBS
script and submits **one normal-queue job: 1 CPU, 16 GB, 2 hours**. It uses
existing scikit-learn/SciPy dependencies; no XGBoost installation is required.
Alternative baseline and comparison directories may be passed as its first
and second arguments.

Check the actual returned job ID, or retrieve it from the submission record:

```bash
record=$(ls -td outputs/pairwise_sft_rerankers/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Models and validation selection rules are saved before test scoring. Outputs
are under `outputs/pairwise_sft_rerankers/run-JOB_ID.gadi-pbs`:

- `manifest.json`: complete/failed state and frozen source provenance;
- `validation_selection.json`: selected redundancy penalties and model hashes;
- `*_pairwise_training.json`: C trials and fitting pair counts;
- `*.joblib`: three new pairwise models, in a distinct payload schema;
- `validation_summary.json`: selection diagnostics;
- `test_summary.json`: MRR@5, coverage@5, coverage@10, full-pool coverage,
  per-batch scores, paired bootstrap differences and coverage gains/losses;
- `*_per_question.json` and `*_rankings.jsonl`: outcomes and complete rankings.

Print the primary same-pool comparisons:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = sorted(Path('outputs/pairwise_sft_rerankers').glob('run-*'),
               key=lambda p: p.stat().st_mtime)
if not paths:
    raise SystemExit('No run output found.')
directory = paths[-1]
state = json.loads((directory / 'manifest.json').read_text())
print('Run:', directory, 'Status:', state['status'])
if state['status'] != 'complete':
    raise SystemExit('Inspect the job logs before reading results.')
d = json.loads((directory / 'test_summary.json').read_text())
for pool, selector in [('original_greedy', 'original'),
                       ('original_sampling10', 'original'),
                       ('max5_greedy', 'max5'),
                       ('max5_sampling10', 'max5')]:
    print('\n' + pool)
    baseline = f'{pool}_{selector}_logistic_plain'
    for objective in ('logistic', 'pairwise'):
        for policy in ('plain', 'diverse'):
            name = f'{pool}_{selector}_{objective}_{policy}'
            m = d['conditions'][name]['metrics']
            print(f"  {objective}_{policy}: MRR={m['mrr_at5']:.6f}, "
                  f"coverage@5={m['coverage_at5']:.4f}")
            if name != baseline:
                c = d['paired_contrasts'][f'{name}_minus_{baseline}']
                print('    MRR difference:', c['metrics']['mrr_at5'])
                print('    Coverage gains/losses:', c['coverage_at5_transitions'])
PY
```

## Interpretation

The experiment evaluates the whole saved pool, including candidates below
rank 10. Full-pool coverage is an invariant, not an improvement metric here.
Prefer gains in MRR@5 and coverage@5 supported by per-question gains/losses;
report differences with 10,000 paired bootstrap resamples (seed 3407).

Test95 has already been inspected, so these are exploratory results. Bootstrap
intervals are pointwise, without multiple-comparison adjustment. Training
banks have four sampled draws per model whereas test banks have ten. Neither
this linear objective nor typography redundancy adds deep evidence reasoning.
Evidence-aware neural verification and compactness DPO remain separate later
experiments; this job tests the two inexpensive reranking changes first.
