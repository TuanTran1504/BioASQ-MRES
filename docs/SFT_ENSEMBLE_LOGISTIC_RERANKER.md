# Shared logistic reranker for original and max-five SFT

Train a single candidate selector on three-model fitting pools, freeze it, and
use it for both SFT formulations. This step trains only the reranker; it does
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

Every vectorizer and logistic classifier is fitted on fitting questions only.
Candidate weights give equal total weight to each nonempty question/formulation/
decoding stratum. Select C from `[0.1, 1, 10]` by the equal-weight mean MRR@5 over
the four validation conditions. Exact ties retain the first (smallest) C.
Do not refit on validation after selection. The selected model is shared across
all conditions. These validation results are selection diagnostics, not final
held-out performance.

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

One normal-queue CPU job (1 CPU, 16 GB, two hours) waits with `afterok` dependencies
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

- `reranker.joblib`: selected, frozen feature transformers and classifier;
- `manifest.json`: provenance, completion status, source hashes and diagnostics;
- `validation_summary.json`: C selection and all validation comparisons;
- `validation_per_question.json`, `validation_rankings.jsonl`: diagnostic outcomes;
- fitting/validation candidate files and official matcher reports.

Final evaluation on 95 official factoids must load this frozen artifact without
fitting or tuning it on test labels. Existing greedy outputs can be reused; new
ten-sample test generation is a separate GPU experiment. No jobs are submitted
from the local machine by preparing these files.
