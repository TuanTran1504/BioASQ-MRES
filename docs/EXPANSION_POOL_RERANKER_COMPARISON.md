# Paired SFT/DPO pool reranking experiment

This experiment uses the existing logistic reranker design: question/candidate
word TF-IDF, candidate character TF-IDF and inference-available metadata,
including snippet overlap, source agreement, ranks and relation types. No gold
answer text or official labels enter features. The earlier all-development
`final_model.joblib` is not reused.

Three-model pools combine Llama-3.1-8B, Qwen3-8B and Ministral-3-8B within each
policy. The primary mode is ten sampled expansion responses per model. Greedy
expansion is a separate secondary experiment. SFT and DPO use the same source
composition and generation budget in each mode. No additional surface variants,
gold answers or candidates are inserted. Case and whitespace deduplication keeps
the first form in fixed Llama/Qwen/Ministral source order. Conflicting archived
official labels for a deduplicated form cause validation to fail.

## Comparisons

| Reranker training pool | Held-out SFT pool | Held-out DPO pool |
| --- | --- | --- |
| SFT | SFT pipeline | SFT-trained selector on DPO |
| DPO | DPO-trained selector on SFT | DPO pipeline |
| SFT + DPO | Shared selector on SFT | Shared selector on DPO |

The two diagonal conditions answer which separately trained pipeline performs
better. Each row holds the fitted selector constant across candidate pools,
allowing generator-pool differences to be measured. Each column holds candidate
pools identical while comparing reranker training data. The mixed row is a
shared-selection diagnostic and uses more training rows than either separate
reranker. Source-balanced round-robin ordering supplies an untrained baseline.

## Separation and scoring

Five folds split sorted question IDs with seed 3407, identically for both
policies. In each fold the reranker trains on 128 questions and predicts the
remaining 32. All candidates and all policy versions of a question stay in the
same fold. Every vectorizer and the classifier are fitted only on training
questions. Neither hyperparameters nor folds are tuned using evaluation scores.

Training uses equal total sample weight per nonempty question/policy, scaled to
mean one; this prevents questions with large pools dominating the loss.
Logistic settings retain the earlier pilot defaults: C=1, balanced classes,
liblinear, 2,000 maximum iterations and seed 3407. Policy identity is not a
feature. Empty pools still count among all 160 evaluation questions.

Rank all existing candidates by held-out classifier score, using fixed
source/rank/text ordering for exact ties. Report MRR@5, coverage@1/5/10 and
full-pool oracle coverage. Reranking cannot improve full-pool coverage; it can
move existing accepted answers into the top five. For the sampled union the
previous full-pool counts were 121 SFT and 123 DPO questions. Greedy counts were
101 and 101. Validate these against the new summary.

Paired question bootstrap intervals use 10,000 resamples. These exploratory
intervals condition on out-of-fold predictions, exclude retraining uncertainty
and have no multiplicity correction. This is reused-development-set
cross-validation, not an unseen-test result. Confirmatory evaluation requires a
reranker fitted/tuned on fitting/validation pools and frozen before an unseen
test evaluation.

## Run on Gadi

From the repository root, pull the committed workflow; then enter the starter
directory. The three completed, officially scored evaluation folders must be
present. Scikit-learn is included in `requirements-gadi.txt`.

```bash
git pull --ff-only
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
qsub jobs/compare_expansion_pool_rerankers.pbs
```

This submits one CPU job for both modes; it does not request a GPU or make API
calls. Keep the job ID printed by `qsub`. Check its status with:

```bash
qstat -swx JOB_ID
```

After completion, the output directory is
`outputs/expansion_pool_rerankers/cv-JOB_ID.gadi-pbs`. It contains:

- `manifest.json`: completion status, source hashes, settings and limitations;
- `comparison_summary.json`: all eight conditions per mode and paired contrasts;
- `sampling10_per_question.json` and `greedy_per_question.json`: all 160 question
  scores per condition;
- `sampling10_rankings.jsonl` and `greedy_rankings.jsonl`: held-out candidate
  scores, ranks, labels and fold IDs.

No full-development final checkpoint is exported. These outputs are ignored
experiment artifacts and may contain permitted dataset text in candidate
strings. To inspect the summary after replacing the job ID:

```bash
cat outputs/expansion_pool_rerankers/cv-JOB_ID.gadi-pbs/manifest.json
cat outputs/expansion_pool_rerankers/cv-JOB_ID.gadi-pbs/comparison_summary.json
```

The equivalent Python command can run locally after downloading all three
complete evaluation folders, including their archived official analyses.
The small outcome-selected `case_audit` exports alone are insufficient. The
script validates raw generation, candidate reconstruction and archived scores
before training; `--validate-only` performs these checks without fitting.
