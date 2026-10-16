# Reranker ablation: remove model-generated relation labels

The preference audit found a case where changing only `synonym` to
`nomenclature_variant` improved ensemble rank while preserving every answer.
This experiment asks whether the reranker benefits from those self-reported
tags, or whether they create an unreliable shortcut.

It reuses the completed rerankers from job **181350943** and the scored test
comparison from **181287698**. No new GPU generation, semantic judging or DPO
training is involved. One CPU job retrains the original-only, max-five-only
and shared logistic selectors using their exact cached candidate pools.

## Controlled comparison

Only `relation_types` is cleared, at both fitting and scoring. Answer strings,
candidate membership, official acceptance labels, question evidence, source
identity, source ranks, draw frequencies, text features, sample weights, solver
and random seed stay the same. Position information is retained: this is not
an ablation of candidate ordering or of the expansion prompt.

For each selector the job reports:

1. **Fixed C:** remove tags and refit with that selector's archived C. This is
   the direct feature ablation.
2. **Validation-selected C:** remove tags and select C from the original grid
   (0.1, 1, 10) using the same validation conditions and first-C tie rule.

All fitting uses 1,296 questions; C selection uses 144 separate validation
questions. Test membership is checked to be disjoint from both. All six models
are saved before test scoring. The old rerankers and source archives are never
overwritten. Ablated payloads have a distinct schema and must be scored through
this workflow, which enforces the tag-blind input policy.

Every model is compared with its corresponding archived reranker on identical
original/max-five greedy/ten-sample test pools. Report MRR@5, coverage@5,
coverage@10 and full-pool coverage, plus paired bootstrap differences (10,000
resamples, seed 3407). Full-pool coverage must stay unchanged. Separate per-batch
results and per-question outcomes are saved.

## Run on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_reranker_relation_ablation.sh
```

The launcher validates source hashes, runtime compatibility and question
membership, then snapshots the worker and PBS script before submitting one
normal-queue CPU job (1 CPU, 16 GB, 2 hours). Alternative baseline/comparison
directories may be passed as its first and second arguments.

```bash
record=$(ls -td outputs/reranker_relation_ablation/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Outputs are under `outputs/reranker_relation_ablation/run-JOB_ID.gadi-pbs`:

- `validation_selection.json`: C trials and frozen model hashes;
- `validation_summary.json`: selection diagnostics, not independent evidence;
- `test_summary.json`: all before/after metrics and paired contrasts;
- `validation_per_question.json`, `test_per_question.json`: paired outcomes;
- six `.joblib` payloads and `manifest.json`.

## Interpretation and next step

Start with the max-five selector's fixed-C comparison on the max-five sampled
pool. This directly tests the feature that affected the audited preferences.
Also report all other conditions; do not select a winning model using test95.

The 95 test questions have already informed project discussions, so this is an
exploratory ablation. Intervals are pointwise and unadjusted for multiple
comparisons; fitting uses four-sample pools while test uses ten-sample pools.
Removing a feature guarantees invariance to tag-only edits, not higher MRR.

If useful, a separate next experiment can rebuild out-of-fold feedback without
tags and recount semantically valid preference pairs. This job does not rebuild
the DPO audit or launch DPO training.
