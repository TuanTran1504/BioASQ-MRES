# Reranker-guided DPO: fitting-bank preference audit

The first stage measures whether different saved Llama responses produce useful
final-ranking preferences. It submits **one CPU audit job**, not DPO training.
It makes no new generations, Java-scoring requests, or paid judging calls.

## Available data and fixed protocol

The max-five banks contain 1296 fitting and 144 internal-validation questions,
each with one greedy plus four temperature-0.8 sampled responses from each of
Llama, Qwen3 and Ministral. The audit uses only fitting questions. Bank validators
check question partitions, native prompts, seeds, response IDs, complete draws,
hashes and the pinned data's exclusion of dev160 and official test questions.
It reuses the cached official candidate labels saved by the completed reranker
job `181350943.gadi-pbs`. Nonaccepted forms are not labelled semantically wrong.

The pilot configuration is
`gadi_sft_8b_starter/configs/reranker_guided_dpo_pilot.json`:

- focal generator: Llama max-five SFT;
- companions: unchanged Qwen3 and Ministral max-five responses;
- five question folds, sorted IDs and seed 3407;
- logistic C=0.1, fixed for this pilot, with no new search on validation or test;
- reranker fitting conditions: max-five greedy and max-five sampling-four;
- at most two provisional primary pairs per question.

The final reranker has already trained on fitting questions and therefore must
not supply in-sample feedback for those questions. Instead, this audit fits five
additional copies. Each question is scored by the copy that excluded its entire
fold, including all candidates and question-derived vectorizer inputs. Each
copy's fitting/held-out IDs, weights and hash are saved. Existing final selectors
are not modified. Generator banks are still in-sample SFT fitting outputs; only
the **reranker feedback** is cross-fitted. This distinction must be reported.

## One response changes; eleven stay fixed

For each fitting question, the sampled ensemble has twelve responses, four per
model. A SHA256 rule using question ID and seed chooses one Llama sample slot.
The other eleven response IDs stay identical for every comparison. Each of the
five saved Llama responses (the greedy response and the four samples) is tried
in that slot. The greedy response is an alternative, not a thirteenth response.

Rebuild the pool with the existing parsing, case/whitespace deduplication,
first-unique source ranks and per-model draw support divided by four. Failed
draws remain in denominators. The original bank rows and response IDs stay
unchanged; projected draw positions apply only inside the counterfactual pool.
No new strings, aliases or edited completions are inserted.

Each option is scored by its question's held-out reranker. The audit records
the first accepted answer's rank, reciprocal rank at five, fixed-budget and
full-pool coverage, pool size, response length and top-five predictions. Labels
are used after prediction to evaluate outcomes and never as reranker features.
The baseline is the unchanged response originally occupying the chosen slot.

## Provisional preferences and separate diagnostics

Prefer a complete response when its final reciprocal rank is strictly higher.
Both responses must have strict, unrepaired, duplicate-free max-five JSON and
finish below the output-token ceiling. The chosen response may not reduce the
number of officially accepted focal forms or full-pool coverage. Ties abstain;
duplicate response pairs do not increase yield. Deterministic ordering picks
the largest gains before the two-pair per-question cap.

**Primary candidates** additionally require the same normalised first answer
in both responses, with that initial answer officially accepted in both.
This isolates changes to expansion/metadata while holding the initial answer
fixed. Cases with different accepted initial expressions, changed initial
acceptance, or neither initial accepted are exported separately as diagnostics;
they are not silently mixed into the primary set.

All pairs have `training_ready=false` and `semantic_review_status=pending`.
An official accepted initial answer does not certify the other expansions as
valid or evidence-supported. Primary chosen/rejected responses are exported for
review using the existing max-five annotation schema. No `train.jsonl` or
`validation.jsonl` training files are written, and no DPO trainer is launched.
The targeted review export is not the full three-bank package expected by the
current paid judge CLI; use a validated targeted-review integration in the next
stage rather than passing it to that CLI directly.

## Submit and inspect on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_reranker_guided_dpo_audit.sh
```

The launcher checks the completed reranker artifacts and banks before submission,
then freezes code and hashes under `outputs/reranker_guided_dpo/submission-*`.
One normal-queue job requests one CPU, 16 GB and two hours. No earlier jobs are
resubmitted. Its job ID is recorded in that submission's `jobs.tsv`.

```bash
record=$(ls -td outputs/reranker_guided_dpo/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Outputs are under `outputs/reranker_guided_dpo/audit-JOB_ID.gadi-pbs`:

- `summary.json`: primary pair/question counts, exclusions, length differences,
  baseline OOF MRR and oracle MRR across the five saved alternatives;
- `alternative_outcomes.jsonl`: every question's five outcomes and fixed context;
- `preference_candidates.jsonl`: capped primary provisional preferences;
- `other_diagnostic_candidates.jsonl`: other initial-answer categories;
- `review_responses.jsonl`: deduplicated primary responses needing semantic review;
- `folds.json` and `fold-*.joblib`: feedback membership and frozen fold selectors;
- `manifest.json`: completion/provenance with `review_required` or
  `insufficient_primary_pairs`, never training-ready status.

The oracle-over-alternatives score uses fitting gold to choose a response offline.
It is a diagnostic upper bound, not an inference procedure or demonstrated test
gain. Useful yield and semantic quality must be established before DPO training.

## Follow-up: generator preference preparation

The [generator-feedback preparation workflow](GENERATOR_FEEDBACK_DPO.md) reuses
these frozen fold models instead of refitting them. It adds separate validation
preferences, filters tag-only changes, exports exact raw responses for review,
and separates strict ranking gains from compactness and retention controls.
Its default is a Llama pilot; optional focal generators are supported. It is
still a data-preparation stage, not an automatic DPO training submission.
