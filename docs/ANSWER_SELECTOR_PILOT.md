# Tie-aware supervised final-answer selector pilot

The pilot trains a **new selector adapter** on the cached Llama-3.1-8B-Instruct
four-bit backbone. It does not modify the three answer generators or initialize
from their answer-generation SFT/DPO adapters. There is no model download, paid
judging, DPO training, new biomedical answer generation, or test95 evaluation.

## Run on Gadi

From `gadi_sft_8b_starter`:

```bash
git pull --ff-only
bash scripts/submit_answer_selector_pilot.sh
```

Default input:
`outputs/answer_selector_data/audit-181824213.gadi-pbs`.
Pass another complete export of the same schema as the first argument if needed.
All source hashes, question/pool membership and labels must validate. Preflight
resolves the already pinned Llama instruction checkpoint using existing SFT
lineage metadata. It reads checkpoint metadata, not dev answers or dev scores.

Preparation tokenizes all input views with that exact cached tokenizer, including
all candidates and all supplied snippets. It separately reserves the precise
number of constrained output tokens. The **8,704-token allocation** is checked for
every input and every epoch's training sequence; overflow stops submission rather
than truncating evidence, candidates or answer strings. Candidate codes must each
be one token under this tokenizer; otherwise submission fails explicitly.

Four jobs are recorded in
`outputs/answer_selector_pilot/submission-*/jobs.tsv`:

| Stage | Resource | What it does |
|---|---|---|
| baseline | CPU, 2-hour allocation | Refit the logistic reranker on exactly these prospective verbatim fitting pools; select C on all144 validation questions. |
| smoke | One V100, 1-hour allocation | Two optimizer steps on long fitting inputs, plus constrained selection on long validation inputs. |
| train | One V100, 12-hour allocation | Depends on successful baseline and smoke; train three epochs and select using all144 validation questions after each epoch, with epoch zero included. |
| report | CPU, 2-hour allocation | Compare the chosen selector with same-pool logistic and export validation mistake cases for inspection. |

CPU baseline and GPU smoke can run independently. A preparation failure submits no
jobs. A failed prerequisite prevents full training/report jobs from running. Partial
submission records remain available if a later `qsub` call fails.

## Supervision and ties

The model sees question, supplied evidence and numbered candidate answers. It
outputs up to five existing IDs. All candidate surfaces are preserved byte for byte;
only their display IDs and positions are remapped. The model never receives gold
aliases, official acceptance flags, relation tags or review annotations.

Three-digit string codes (`"001"`, `"002"`, ...) let each candidate choice be a
single token. JSON framing and selection without replacement are constrained at
inference, so the selector cannot fabricate an answer, duplicate an ID or change
answer spelling. This is an ID-selection policy, not unconstrained free-text QA.

For a fixed input and previously selected teacher-prefix IDs, let `L` be remaining
legal candidate tokens and `A` remaining officially accepted candidate tokens:

```text
selection loss = logsumexp(logits[L]) - logsumexp(logits[A])
```

This is the negative log probability of choosing **any** member of the accepted
set, after normalizing over legal candidates. Both accepted synonyms count equally
at that choice. The loss does not impose an order within either class. Fitting
input positions/IDs and positive teacher prefixes vary reproducibly by epoch.

Only accepted teacher choices, up to five, contribute supervised choice losses.
Once accepted candidates are exhausted, there is **no negative-tail ranking loss**.
JSON punctuation is forced, so it does not need an invented target ranking either.
The loss averages supervised choices within each question/pool view; all views have
equal weight. This is marginal supervised ID learning with partial labels, not
ordinary teacher forcing of a single fully ordered five-answer target.

Pools with no accepted candidate have no positive training signal. Pools whose
candidates are all accepted have zero ranking discrimination signal. Both remain in
validation generation/scoring. For the current export, training uses **2,021
informative views across 1,100 fitting questions**; the 99 all-accepted SFT views are
not used for ranking loss. This is why the count differs from the audit's 2,120
format-ready SFT views across 1,114 questions.

The labels mean **official acceptance**, not biomedical truth. Several reviewed
nonaccepted alternatives are valid synonyms or even preserve evidence qualifiers
better than gold. This pilot must not be described as teaching the model that all
other expressions are medically wrong.

## Training and selection

- Fresh language-only LoRA: rank16, alpha32, dropout0; base remains four-bit/frozen.
- Three epochs, learning rate `1e-4`, batch size1, accumulation16, AdamW on LoRA
  parameters, weight decay0.01, linear schedule with 10% warmup, clipping1.0.
- FP16 computation on V100; FP32 LoRA gradients/state; seed3407.
- All144 validation questions in each of greedy and sampling4 pools: **288 input
  views**, including uncovered pools. No filtering by answer correctness at inference.
- Constrained greedy ID selection before training (epoch0) and after each epoch.
- Select maximum mean MRR@5 across the two validation modes. Earliest epoch wins
  ties, including epoch0. Epoch0 means the instruction selector with a fresh,
  zero-effect LoRA, not one of the answer generators and not the logistic baseline.
- All epoch adapters and predictions are retained. Full-pool membership cannot
  change because inference only selects IDs from the fixed supplied pool.

The CPU baseline is refitted because exact-string pooling differs from historical
normalized pools. Its features, C grid (0.1/1/10), fitting-only vectorizers and
validation selection follow the existing logistic implementation. Compare both
methods on the same candidate pools; do not substitute old test scores.

Four stochastic responses/model is the audited generation budget. Results do not
establish transfer to ten-sample candidate pools. The generator fitting questions
were seen by generator SFT; these are stacked-selector training inputs, not
out-of-fold generator predictions.

## Monitor and inspect results

```bash
record=$(ls -t outputs/answer_selector_pilot/submission-*/jobs.tsv 2>/dev/null | head -n 1)
if [[ -n "$record" ]]; then
    cat "$record"
    ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
    qstat -swx $ids
    qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
else
    echo "No supervised selector submission found."
fi
```

After report completion:

```bash
submission=$(dirname "$record")
report_job=$(awk -F '\t' '$1=="report" {print $2}' "$record")
cat "$submission/report-${report_job}/report.md"
```

Training output is `submission-*/train-TRAINJOB.gadi-pbs/`: `checkpoint_scores.json`,
`selection.json`, `training_log.jsonl` and `epoch-N/adapter/` plus validation
predictions/per-question scores. The report contains `comparison_summary.json` and
`validation_mistakes.jsonl`, separating absent-from-pool answers from missed top-five
selections and lower-rank accepted answers. These are validation diagnostics used in
selection, so bootstrap intervals are descriptive, not independent-test evidence.

## DPO is a later stage

The report does **not** construct preferences or train DPO. Validation mistakes must
never be copied into DPO training. First inspect whether the selector helps, then
collect its actual alternative rankings on fitting questions and audit score gaps,
ties, supported synonyms and repeated-question counts. Future DPO likelihoods must
match this constrained ID policy (including legal-ID normalization), rather than
silently using unconstrained full-vocabulary sequence likelihoods. No such DPO
trainer is part of this pilot.
