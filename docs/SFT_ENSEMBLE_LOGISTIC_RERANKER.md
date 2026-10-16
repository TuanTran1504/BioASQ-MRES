# Three logistic rerankers for original and max-five SFT

Train three candidate selectors on three-model fitting pools and freeze each.
This step trains only the rerankers; it does
not generate new responses, train DPO, judge with an API, or score the test set.

## Data and labels

The completed original banks are `llama31-mixed-generate-181032357.gadi-pbs`,
`qwen3-mixed-generate-181032359.gadi-pbs`, and
`ministral3-mixed-generate-181032361.gadi-pbs` under
`gadi_sft_8b_starter/outputs/original_sft_dpo_8b`.
The launcher finds the newest full max-five bank submission for each model
under `outputs/max5_dpo_banks_8b`, retaining other models when only one is retried.
All six banks must pass their existing full-panel, hash, prompt, draw and seed
validators. The paired SFT datasets enforce 1296 fitting and 144 validation IDs,
disjoint from dev160 and the project test panel. Evidence must agree across arms.

Banks contain **one greedy plus four stochastic responses**, not ten samples.
Build separate greedy and sampling-four pools; the latter excludes the greedy
draw. Pool only original SFT outputs together, and only max-five outputs together.
Use the existing inference parsers, cap expansion responses at five, deduplicate
case/whitespace conservatively, and add no artificial variants or gold answers.
All questions remain in scoring denominators even when their pools are empty.

Official BioASQ Java matching supplies a binary *accepted form* label for every
retained candidate. A negative label does not assert biomedical incorrectness.
LLM semantic judging and its paid budget are unrelated to this selector training.

## Features and selection

Reuse question/candidate word TF-IDF, candidate character TF-IDF, and metadata
(snippet overlap, candidate size, source ranks, relation types, model agreement).
Model identity denotes Llama/Qwen/Ministral; formulation identity is not a feature.
Sample support is normalised by the requested draws per model, so four repeated
answers from one model still count as one model agreeing. Source rank is first
unique occurrence in draw order, then within-response order.

Each selector has its own vectorizers and classifier, fitted on fitting questions only.
Candidate weights give equal total weight to each nonempty question/formulation/
decoding stratum. Select C independently from `[0.1, 1, 10]`:

| Selector | Fitting pools | Validation pools used to select C |
| --- | --- | --- |
| Original-only | Original SFT, greedy and sampling-four | Two original-SFT conditions |
| Max-five-only | Max-five SFT, greedy and sampling-four | Two max-five-SFT conditions |
| Shared | Both formulations, greedy and sampling-four kept as separate pools | All four conditions |

Use equal-weight mean validation MRR@5 over the declared conditions. Exact ties
retain the first (smallest) C. Do not refit on validation after selection.
After selecting C, evaluate each frozen selector on all four validation pools,
including the opposite formulation. These cross-formulation scores do not influence
the specialised selectors' C selection. Validation results are selection diagnostics,
not independent held-out performance.

On the later test panel, compare the original-only reranker on original pools with
the max-five-only reranker on max-five pools as a complete-system comparison.
Use the shared reranker on both pools to hold selection constant. Applying all
three rerankers to each identical pool also isolates reranker training differences.
The shared reranker sees more candidate rows, although all three use the same fitting
question IDs; this is a difference in available training data, not an equal-data ablation.

Validation also reports round-robin, distinct-model voting, reciprocal-source-rank
voting, and logistic selection on identical pools. Reranking cannot change full-pool
coverage. This uses in-sample fitting banks from the trained generators, not
cross-fitted generator outputs; the held-out 144 questions assess the resulting
selector. A later ten-sample inference experiment must reconstruct the same
features and report the four-versus-ten sampling distribution difference.

## Submission on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_sft_ensemble_reranker.sh
```

One normal-queue CPU job (1 CPU, 16 GB, two hours) trains all three selectors sequentially
and waits with `afterok` dependencies
on outstanding max-five bank jobs. Already completed banks need no dependency.
Missing, failed or unverifiable jobs stop preparation before submission. Python
3.12 and the existing venv are used. Code and the Java adapter are frozen under
the submission directory, protecting queued jobs from later script changes.
Data and bank hashes are revalidated by the worker. No user credentials are read.

Check the printed job ID:

```bash
qstat -fx JOB_ID | grep -E 'Job Id|job_state|Exit_status|comment'
```

Outputs appear under `outputs/sft_ensemble_reranker/train-JOB_ID.gadi-pbs`:

- `original/reranker.joblib`: original-only frozen selector;
- `max5/reranker.joblib`: max-five-only frozen selector;
- `shared/reranker.joblib`: shared frozen selector;
- `reranker.joblib`: identical copy of the shared selector for existing consumers;
- each selector's `validation_selection.json`: fitting arms, selection conditions and C trials;
- `manifest.json`: provenance, completion status, source hashes and diagnostics;
- `validation_summary.json`: all three selectors' C selection and 24 validation conditions
  (four candidate pools, each with three baselines and three rerankers);
- `validation_per_question.json`, `validation_rankings.jsonl`: diagnostic outcomes;
- fitting/validation candidate files and official matcher reports.

If a job was already submitted using the earlier shared-only workflow, its frozen
code still trains only that selector. Submit the updated launcher once for the three-model
workflow; avoid resubmitting if a current three-selector job is already queued.

Final evaluation on 95 official factoids must load these frozen artifacts without
fitting or tuning it on test labels. Existing greedy outputs can be reused; new
ten-sample test generation is a separate GPU experiment. No jobs are submitted
from the local machine by preparing these files.

## Evaluate the completed selectors on the saved test pools

The three selectors completed in `train-181350943.gadi-pbs`. Their printed
validation-selection means are 0.578356 (original, C=1), 0.603935 (max-five,
C=0.1), and 0.588223 (shared, C=1). These average different declared validation
conditions and do not establish which selector performs best on an identical
pool. They are not 95-question test scores.

The three-model greedy and ten-sample test pools are already scored in
`outputs/sft_ensemble_test/comparison-181287698.gadi-pbs`. Evaluate all three
selectors on each of these four pools, alongside all four fixed ordering rules:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python ../scripts/evaluate_sft_ensemble_rerankers.py \
  --comparison outputs/sft_ensemble_test/comparison-181287698.gadi-pbs \
  --rerankers outputs/sft_ensemble_reranker/train-181350943.gadi-pbs \
  --output "outputs/sft_ensemble_reranker/test95-$(date -u +%Y%m%d-%H%M%S)"
```

This is CPU-only rescoring. It performs no fitting, GPU generation, Java
evaluation, or paid API calls. It verifies the saved test pools and their cached
official labels by reproducing all 12 original conditions, checks frozen model
and artifact hashes and scikit-learn compatibility, confirms that the rerankers'
fitting and validation IDs are disjoint from the test set, and verifies the
generator checkpoint/prompt identities against the bank sources used in training.
Predictions receive no official labels or gold aliases. Candidate counts,
full-pool coverage and generation costs stay unchanged. Failed/empty questions
remain in the denominators. Existing output directories are never overwritten.

The new directory contains `comparison_summary.json`, `comparison_per_question.json`,
`rankings.jsonl`, and a provenance `manifest.json`. It reports **28 conditions**:
four candidate pools, each evaluated by round-robin, model voting, reciprocal-rank
voting, sample-frequency voting, and the three frozen logistic selectors. Each
condition includes aggregate metrics and per-batch results. Paired question-level
95% bootstrap intervals (10000 resamples, seed 3407) compare each reranker with
each fixed rule; same-selector formulation contrasts and the specialised
original-versus-max-five complete-system contrast are also included.

No selector is chosen or refitted from test results. Report the specialised
selectors on their own formulations, and the shared selector on both formulations
to hold selection constant; cross-formulation applications are additional
diagnostics. The ten-sample test pools differ from the four-sample training pools
in candidate diversity and source-rank distributions, despite normalised draw
support. Interpret these as exploratory results on previously inspected test
batches, with uncorrected intervals across multiple comparisons.
