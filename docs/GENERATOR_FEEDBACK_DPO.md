# Direction 1: train answer generation with ensemble-ranking feedback

The reranker stays frozen. We construct preferences between complete, naturally
generated max-five responses according to their effect on final ensemble
ranking. This is different from the pairwise-reranker experiment: its trainable
component was the reranker; the intended trainable component here is the SFT
answer generator.

Start with **Llama max-five SFT** as a pilot. The completed three-model banks
already contain 1 greedy and 4 sampled responses per question and model. No
additional GPU generation is needed to prepare the pilot. Qwen3 and Ministral
remain unchanged companions. The CPU preparation command can optionally prepare
each of the other focal generators separately too.

## What this stage does

One CPU job reuses the frozen feedback models from audit **181379068**, the
validated response banks, and the official acceptance labels cached by
reranker training **181350943**. It does not refit the reranker and does not
submit a GPU training job or paid judge requests.

For each question, one deterministic sampled slot of the focal model is
replaced with each of its five saved responses in turn. Eleven companion
responses remain identical. The pool is rebuilt with the original provenance,
draw support, deduplication and source-rank rules. Gold labels are applied only
after prediction to calculate final MRR@5 and coverage.

- **1,296 fitting questions:** each is scored by its existing out-of-fold
  reranker, whose vectorizers and weights were fitted without that entire fold.
- **144 validation questions:** scored by the archived final max-five logistic
  reranker, which was fitted on fitting questions. Validation is kept separate
  for preference evaluation and subsequent checkpoint selection.
- **dev160 and test95:** excluded from preference construction.

The generator itself saw the fitting questions during SFT; only the reranker
feedback is out of fold. The final reranker's regularization was previously
selected using validation, so those scores remain selection diagnostics.

## Separate preference objectives

| Export | Preferred response | Purpose |
| --- | --- | --- |
| `ranking_gain` | Strictly better final MRR@5, with accepted focal forms retained | Direct downstream-ranking feedback |
| `compactness` | Proper ordered subset of the rejected list, unchanged first candidate and retained relation tags, at least one focal accepted answer, and no reduction in final MRR@5 | Fewer expansions without losing accepted answers |
| `retain_needed_expansion` | Longer nested response that retains an accepted focal answer absent from the shorter response, without worse final ranking | Controls against always returning fewer answers |

For every objective, the preferred response must retain **all** officially
accepted focal expressions in the rejected response and preserve final full-pool
coverage. Comparisons that change only relation tags, case or whitespace are
excluded. Responses must pass strict max-five parsing and finish below the
output-token ceiling. No edited or synthetic completions are substituted for
the original raw responses.

Ranking preferences may change the initial answer; their categories explicitly
separate same accepted initial, same unaccepted initial, and changed initial
expression. They must not be described collectively as expansion-only
corrections. An unaccepted initial expression may still be semantically correct.

Each objective exports at most two content-deduplicated pairs per question,
prioritizing top-five coverage gains, then reciprocal-rank gains. The objectives
can overlap: do not concatenate files and count repeated pairs as additional
independent examples. Compactness allows tied MRR and therefore must not be
reported as demonstrated performance improvement. Deleting a valid synonym is
also not evidence that the synonym was a harmful or incorrect expansion.

## Submit preparation on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_generator_feedback_preferences.sh
```

The default prepares **Llama only**, using the existing bank and feedback
directories. To prepare all three focal generators in the same CPU job:

```bash
bash scripts/submit_generator_feedback_preferences.sh llama31 qwen3 ministral3
```

Use one of these commands, rather than submitting both. Optional environment
variables `FEEDBACK_RERANKERS` and `FEEDBACK_AUDIT` override source directories.
The launcher validates source hashes, bank identity, fold membership and
preparation provenance before submitting a frozen code/PBS snapshot. It requests
1 CPU, 16 GB and 2 hours in the normal queue.

Monitor the returned job ID, or recover it from its submission record:

```bash
record=$(ls -td outputs/generator_feedback_dpo/submission-*/jobs.tsv | head -n 1)
cat "$record"
job=$(awk -F '\t' 'NR==2 {print $2}' "$record")
qstat -swx "$job"
qstat -fx "$job" | grep -E 'Job Id|job_state|Exit_status|comment'
number=${job%%.*}
tail -n 60 *o"$number"
tail -n 60 *e"$number"
```

The output directory is `outputs/generator_feedback_dpo/preferences-JOB_ID.gadi-pbs`.

```bash
cat "outputs/generator_feedback_dpo/preferences-${job}/summary.json"
```

It contains:

- `manifest.json` and `summary.json`: counts, feedback provenance and limitations;
- `llama31/train_*_review_candidates.jsonl`: separate fitting preference sets;
- `llama31/validation_*_review_candidates.jsonl`: separate validation sets;
- `llama31/train_outcomes.jsonl` and `validation_outcomes.jsonl`: all saved
  alternatives and ranking outcomes, including first accepted rank and top five;
- `review_responses.jsonl`: deduplicated original completions, prompts, snippets,
  candidate-level official acceptance and empty semantic-review fields.

Other requested focal models have their own subdirectories. No training-ready
`train.jsonl` or `validation.jsonl` is written.

## Review and training follow-up

Preference packages are marked `training_ready=false` and `review_required`.
The earlier audit exposed tag shortcuts and unsupported numerical alternatives,
so official acceptance plus better rank is insufficient to approve every
expansion. Review chosen candidate correctness, evidence support, equivalence
and relation labels before assembling a training set. For an expansion-only
arm, require a semantically correct initial answer and keep it unchanged.

The current paid max-five judge CLI expects full three-model bank packages;
this targeted review export must not be passed to it as if it were that full
package. No new judging spend is initiated by this preparation job.

After reviewing yield and quality, build separate validated training arms:

1. Ranking-feedback DPO using reliable strict-gain pairs.
2. Compactness DPO with retention controls and question-balanced sampling.

Do not automatically train if a reliable fitting or validation set is empty or
has too few distinct questions. Freeze the original SFT reference and the
logistic reranker. Select DPO settings on validation, then compare the unchanged
SFT generator and DPO generator using matched generation budgets and the same
frozen reranker. Report MRR@5, coverage@5, full-pool coverage, per-question
gains/losses, candidate counts and generation cost. Generation/evaluation
jobs and a training-ready preference validator are later stages; they are not
silently launched by this CPU preparation command.

## Reviewed Llama compactness pilot (17 October 2026)

The targeted semantic review is complete. GPT-4.1-mini judged all 726 unique
candidate/original/relation units from 377 shortlisted responses in 174 cached
requests, costing **US$0.2470376** under the previously authorized US$10 max-five
judging cap. No gold aliases, official labels, model identity or intended
preference direction were supplied to the judge. Every support quote must
match a supplied snippet. Low confidence, unresolved attributes, repeated-label
conflicts and official/semantic conflicts abstain.

Assistant screening adds explicit vetoes for the ALD lower-bound omission,
screening start ages mixed with age intervals, an androsterone/androstenone
question/evidence mismatch, and a lamin-A/LMNA protein/gene abbreviation issue.
The automated annotations and assistant screening are **not independent expert
certification**. Some discarded expressions are valid aliases; this experiment
optimizes compactness, not a C3-over-incorrect-answer objective.

The final package is `generator_feedback_dpo_compactness_v2`:

| Split | Compactness | Retention controls | Total pairs | Distinct questions |
| --- | ---: | ---: | ---: | ---: |
| Fitting | 24 | 12 | 36 | 35 |
| Validation | 3 | 1 | 4 | 4 |

Require every preferred candidate to be high-confidence correct, supported,
equivalent and correctly tagged; require a supported, correct unchanged initial
answer on both sides. Retention controls must be clean on both sides. Keep at
most one pair per question/objective. Cap fitting compactness examples at twice
the number of retained control questions, using deterministic seed-3407 order.
Validation keeps all qualified distinct question/objective examples instead of
downsampling. The four-question validation set is very small: checkpoint loss
selection is an exploratory diagnostic, not reliable proof of generalization.
Ranking-gain pairs are excluded from this training package.

### Training settings

Continue the **exact loss-selected max-five Llama SFT adapter** used to produce
the banks. Update its existing language LoRA weights with standard whole-response
DPO, summed completion-token log probabilities, beta **0.1**, learning rate
**2e-6**, **3 epochs**, batch size **1**, accumulation **4**, and warm-up ratio
**0.1**. This gives **27 optimizer steps and 3 warm-up steps**. Freeze reference
completion log probabilities from the initial SFT weights, with dropout disabled.
Refuse silent prompt/completion truncation (sequence ceiling 8,704 tokens).
Select the saved adapter by **validation preference loss**, not MRR.

The training package validator reconstructs every selected pair from the copied
source candidates and semantic annotations. On Gadi it also links reviewed raw
responses back to the original complete banks, verifies bank hashes, verifies
the SFT checkpoint source, and pins portable training helpers with the submitted
code. Source models, preference files and code remain checked for edits. An
optional validator callback in the existing DPO trainer supports this distinct
schema; legacy expansion/original DPO validation remains unchanged.

### Upload the reviewed package and submit

The prepared local archive is
`Artifacts/generator-feedback-dpo-compactness-v2.tar.gz` (about 5.3 MB).

In Windows PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
scp .\Artifacts\generator-feedback-dpo-compactness-v2.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

Then on Gadi:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
mkdir -p outputs/generator_feedback_dpo/reviewed-compactness-v2
tar -xzf /scratch/nl78/$USER/generator-feedback-dpo-compactness-v2.tar.gz \
  -C outputs/generator_feedback_dpo/reviewed-compactness-v2
bash scripts/submit_generator_feedback_dpo.sh \
  outputs/generator_feedback_dpo/reviewed-compactness-v2
```

This submits **two GPU jobs**: a training smoke test, then a full pilot dependent
on smoke success. It does not submit new generation, paid judging or final test
evaluation. Do not pass this package to the legacy `submit_expansion_dpo_8b.sh`;
its historical max-ten adapters and preference schema differ.

```bash
record=$(ls -td outputs/generator_feedback_dpo_training/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Training output is
`outputs/generator_feedback_dpo_training/llama31-compactness-train-JOB_ID.gadi-pbs`.
Inspect `manifest.json`, `training_log.json` and `adapter/`. A successful training
job establishes that optimization completed; downstream improvement still needs
matched-budget generation and scoring with the original frozen reranker.

### Evaluate the completed compactness continuation on dev160

Training job **181456254.gadi-pbs** completed successfully: 27 optimizer steps,
36 fitting and four validation pairs, best validation preference loss
**0.6769970655441284**, and a saved adapter. These are training diagnostics,
not evidence of better answering or ranking.

Submit the matched development comparison:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_generator_feedback_evaluation.sh \
  outputs/generator_feedback_dpo_training/llama31-compactness-train-181456254.gadi-pbs
```

This submits **three jobs**: a one-hour GPU smoke test, one dependent GPU job
(eight-hour allowance) generating Llama DPO greedy and ten-sample responses,
then a dependent two-hour CPU scoring job. It does not train anything, refit a
reranker, use paid judging, or evaluate the official test batches.

All six completed **loss-selected max-five SFT dev160 archives** are reused:
Llama from evaluation 181038914, Qwen3 from 181038918, and Ministral from
181038923, each with greedy and ten-sample decoding. Their native raw responses,
parsing, complete evidence and source checkpoints are verified before reuse.
Only the Llama adapter changes. In each decoding mode, the identical archived
Qwen3 and Ministral responses participate in both ensembles. Ten-sample decoding
uses temperature 0.8, top-p 0.95, top-k 0, seed 3407 and 512 output tokens; each
response has at most five candidates. The original prompt and every snippet are
retained. DPO tokenization and sequence allocation must match the starting SFT;
no input truncation is permitted.

The primary selector is the **existing max-five logistic reranker with relation
labels**, from the exact artifact used for feedback. It is applied unchanged to
both pools. Sample-frequency voting is reported as a secondary comparator, and
Llama alone is scored in draw/within-response order. There is no dev-based
choice of selector or checkpoint. Fitting and internal-validation question IDs
must be disjoint from dev160. Submitted orchestration, inference helpers, data,
checkpoints, official evaluator JAR and reranker artifacts are pinned by hashes;
later checkout edits do not change the submitted code.

Reports include:

- Llama candidate count and output tokens per response, unique Llama candidates,
  and unique combined-pool size. Failed draws remain in denominators and parse
  failures, output ceilings and candidate-cap violations are reported explicitly.
- Official Java candidate acceptance across the **entire pool**, followed by
  coverage@1/5/10 and MRR@5. All empty/failed question pools remain in the denominator.
- Paired bootstrap differences, full-pool and top-five gains/losses with question
  IDs, raw rankings, and per-question compactness. A smaller pool alone is not a
  demonstrated ranking or semantic-quality improvement.

```bash
record=$(ls -td outputs/generator_feedback_dpo_evaluation/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Generation output is `outputs/generator_feedback_dpo_evaluation/generation-JOB_ID`.
CPU scoring writes
`outputs/generator_feedback_dpo_evaluation/comparison-JOB_ID/comparison_summary.json`,
`comparison_per_question.json`, `compactness_per_question.json`, `rankings.jsonl`
and the labeled candidate pools. Its standard output prints SFT/DPO MRR,
coverage, compactness and full-pool gains/losses for each decoding mode.

This is an exploratory, historically inspected development panel. The DPO
checkpoint was selected on only four preference-validation questions; training
feedback used four samples, while the multi-generation evaluation uses ten.
There is one training and decoding seed. Archived GPU timings are descriptive
because source jobs ran separately. Confirm any promising behavior on an
untouched evaluation panel before making a confirmatory performance claim.

### CPU-only recovery for scoring job 181458735

GPU generation **181458734.gadi-pbs** succeeded. CPU scoring
**181458735.gadi-pbs** failed before Java matching because the scorer's
`official/` report directory had not been created. This is an output-folder
bug, not an inference failure. The scorer now creates that directory; a
regression test exercises the actual Java candidate matcher and its file writer.

Keep the completed generation, original submitted code, and failed scoring
directory unchanged. Submit just one CPU recovery job:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_generator_feedback_rescore.sh \
  outputs/generator_feedback_dpo_evaluation/submission-20261017-023654-862209 \
  outputs/generator_feedback_dpo_evaluation/generation-181458734.gadi-pbs
```

The recovery validates the original input/code/checkpoint hashes and completed
generation, copies the original submitted code to a new recovery folder, and
applies **only** the missing-directory correction to its scorer. It records
both scorer hashes and verifies that every other copied code file is unchanged.
It does not change the original submission manifest to bypass integrity checks.
The frozen recovery runner, copied code, and original responses are checked
again at execution. No GPU jobs, generation, training or paid calls are needed.

The submission record is
`outputs/generator_feedback_dpo_evaluation/scoring-recovery-*/jobs.tsv`.
The new results go to
`outputs/generator_feedback_dpo_evaluation/comparison-rescore-JOB_ID/comparison_summary.json`.
The result manifest identifies the recovery and its recorded correction.
