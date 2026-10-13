# Accepted-answer forms and validation ordering rules

Audited the pinned expansion-SFT source on 14 October 2026. All **1,802 recorded
accepted aliases** were retained across **1,296 fitting / 144 validation
questions**. Source hashes and split membership were checked. No development or
test questions, model weights, new generation, or paid judge calls were used.

## Provisional surface-form audit

| Form | Fitting aliases | Validation aliases |
|---|---:|---:|
| Abbreviation/symbol-like | 188 | 15 |
| Plain name/expression, including full names | 1169 | 140 |
| Full-name-plus-abbreviation form | 58 | 4 |
| Numerical expression | 87 | 9 |
| Descriptive phrase | 122 | 10 |
| Total | 1624 | 178 |

These are **automatic surface classifications**, not verified linguistic labels.
In particular, the internal `full_name` category is a fallback for ordinary names
and expressions: `Anakinra` and `Women` fall there, but neither is necessarily
an expanded abbreviation. Symbol-like forms can include gene symbols and
chromosome names. Descriptive expressions with abbreviations receive a
descriptive primary label and overlapping flags where detected. Ambiguous entries
are preserved in a review queue, not removed or silently treated as certain.

There are **204 fitting and 18 validation questions with multiple accepted
aliases**. **101 fitting and 8 validation questions** span multiple detected
primary form categories. Question-level availability counts overlap and must
not be summed as if they partition the dataset. Alias counts above are mutually
exclusive by primary classification.

The result does not establish that full-name-first is the best ordering policy.
It describes annotation forms, not whether the models generate those forms.
Sorting the accepted aliases themselves would give perfect acceptance by
construction and would not be a valid model evaluation.

## Saved outputs

`Artifacts/answer_form_audit_v1/accepted_alias_forms.jsonl` records every alias,
original alias index, question, split, primary form, overlapping flags, and
review reasons. `question_forms.jsonl` records all accepted aliases together and
their available forms per question. `review_queue.jsonl` contains ambiguous
classifications. `summary.json` includes form counts, question counts and hashes.

Reproduce the local audit from the repository root:

```bash
python scripts/audit_answer_forms.py
```

## Compare simple rules on real saved validation generations

The max-five internal-validation generations are still on Gadi, not in the
downloaded dev160 archive. Their loss-selected checkpoints are epoch 2 for all
three models. Reuse those previously generated **144-question, ten-sample**
validation banks to compare:

1. Original candidate order.
2. Plain/full-name-first.
3. Abbreviation-first.
4. Combined-form-first.

All rules use only candidate strings; no gold aliases enter classification or
ordering. Sorting is stable within each category. Candidate strings and full
pool membership stay unchanged. Official gold matching is used only afterwards
to score MRR@5, exact top-one acceptance, coverage@5, and full-pool coverage.
Paired bootstrap intervals compare each rule with original order. An invariant
full-pool score checks that this is a selection experiment, not generation.

The report separately evaluates the **first stochastic draw** (up to five
expressions) and the **ten-sample pooled candidates**. First draw is not greedy
inference; these results must not be labelled greedy max-five scores. The first
draw's coverage@5 cannot improve through ordering alone because all candidates
are already submitted. Ordering can change MRR and top-one acceptance. Pooled
coverage@5 can change as candidates move across the submission cutoff.

On Gadi, after pulling the script, run from `gadi_sft_8b_starter`:

```bash
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
python ../scripts/audit_answer_forms.py \
  --output outputs/answer_form_audit_v1 \
  --validation-run llama31=outputs/sft_max5_8b/llama31-expansion-sft-181038912.gadi-pbs/selection-181038913.gadi-pbs/epoch-2/sampling10 \
  --validation-run qwen3=outputs/sft_max5_8b/qwen3-expansion-sft-181038916.gadi-pbs/selection-181038917.gadi-pbs/epoch-2/sampling10 \
  --validation-run ministral3=outputs/sft_max5_8b/ministral3-expansion-sft-181038920.gadi-pbs/selection-181038921.gadi-pbs/epoch-2/sampling10 \
  > outputs/answer-form-audit-20261014.log
```

Results are under `outputs/answer_form_audit_v1/MODEL/ordering_summary.json`.
These are internal-validation diagnostics for choosing a rule, not independent
test evidence. They do not label semantic equivalence or harmful expansion.
The classifiers should be reviewed before using them as training labels.

## Verification

Tests check that biomedical symbols are not classified as numerical values,
parenthesised brand names are not assumed abbreviation expansions, stable rules
preserve candidate strings, every accepted alias is retained, and a controlled
144-question ordering comparison recovers the expected MRR change while full
pool coverage remains fixed. The comparer verifies question/alias/evidence
membership, all ten draws, raw candidate parsing, candidate budgets and pool
reconstruction before official scoring.
