# Qwen3 and Ministral compactness DPO

Extend the expanded Llama experiment to **Qwen3-8B and Ministral-3-8B**. Each
model uses its own naturally generated max-five responses and its own exact
loss-selected SFT adapter/reference. Do not train these models on the Llama-only
preference package. Reviewed companion packages are now available locally; see
the completed review below before GPU submission.

## Protocol

1. Reuse the existing banks: 1,296 fitting and 144 internal-validation questions,
   with one greedy and four stochastic responses per model/question.
2. Prepare separate Qwen3 and Ministral preferences. Substitute one focal
   response into a fixed pool of the other 11 sampled response slots. Fitting
   feedback uses the existing five question-disjoint reranker folds; validation
   feedback uses the existing frozen max-five logistic reranker. The generator
   has seen the fitting questions during SFT; only reranker feedback is out of fold.
3. Review compactness and answer-retention candidates with the same blinded
   GPT-4.1-mini rubric. Official acceptance alone is insufficient for semantic
   approval. Keep the initial expression and all accepted focal expressions,
   require a shorter ordered subset for compactness, and retain longer-answer
   controls. Pool-size ties are allowed, as in expanded Llama.
4. Build **separate model-specific reviewed packages**. The training ratio cap
   remains 8:1 compactness/retention with at most one pair per question/objective.
   Use all eligible question-capped validation pairs. Minimums remain 64 distinct
   fitting and 12 distinct validation questions, with both objectives required.
   Insufficient reviewed data stops preparation; it is not silently padded.
5. Train each model separately with beta 0.1, learning rate 2e-6, three epochs,
   effective batch size four, and 10% warmup. Actual optimizer steps depend on
   that model's reviewed pair count. Retain all epoch checkpoints.
6. Score epochs 0/1/2/3 on all 144 internal-validation questions, in greedy and
   four-sample conditions. Only the focal generator changes. Both companion
   generators remain at their starting SFT checkpoints, and the reranker stays
   fixed. Dev160 and official test questions are excluded.

The strict primary selection remains comparable to Llama: preserve focal-model
and ensemble full-pool coverage, ensemble MRR and compactness, with no additional
empty, output-ceiling or surplus-candidate responses. Epoch zero is the fallback.
The report additionally records `ensemble_only_candidate_epoch`: highest MRR
under the same guards except the individual-model coverage requirement. This is
an explicitly separate diagnostic; it does not overwrite the strict selected
adapter or retrospectively change Llama's completed selection.

Do not assume separately successful generator checkpoints work together. If
multiple generators omit the same accepted form, companions may stop providing
it. A combined-checkpoint evaluation is a subsequent experiment.

## First Gadi job: preference preparation

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_generator_feedback_preferences.sh qwen3 ministral3
```

This submits **one CPU job for both models**, using existing generation and
frozen feedback models. It does not generate answers, make paid calls, train
DPO or evaluate test data. It checks bank, feature-code and five-fold membership
provenance before submission and at execution. The default source paths are:

- Reranker: `outputs/sft_ensemble_reranker/train-181350943.gadi-pbs`.
- Existing OOF fits: `outputs/reranker_guided_dpo/audit-181379068.gadi-pbs`.

The new submission snapshots its code. It automatically creates a verified
transfer archive under `/scratch/nl78/$USER/` when preference preparation succeeds.
Its exact name, containing the CPU job ID, is printed in the job log.

```bash
record=$(ls -t outputs/generator_feedback_dpo/submission-*/jobs.tsv | head -n 1)
cat "$record"
job=$(awk -F '\t' 'NR==2 {print $2}' "$record")
qstat -swx "$job"
qstat -fx "$job" | grep -E 'Job Id|job_state|Exit_status|comment'
number=${job%%.*}
tail -n 60 *o"$number"
tail -n 60 *e"$number"
```

Completed output is `outputs/generator_feedback_dpo/preferences-JOB_ID/`, with
one directory per focal model and a shared `review_responses.jsonl`. These are
**provisional candidates, not approved training pairs**. Transfer the archive
for semantic review before GPU submission; the trainer refuses this raw package.

## Review budget and package construction

The original max-five US$10 budget ledger is
`Artifacts/generator_feedback_judge_mini_v1`. The Llama reviews consumed
US$1.0278268. New companion review must reuse that cumulative ledger rather
than start another US$10 allowance. The judging tool accepts an explicit
`--budget-directory` pointing to the existing ledger, while its new output
directory records the companion review provenance. Cached identical requests
are reused, and pending requests remain unresolved if the shared cap is reached.
No new paid requests were made while preparing this workflow.

After the companion candidate archive is downloaded, use review scope
`all_compactness`, perform the same semantic screening, and build each package
using its corresponding configuration:

- `configs/generator_feedback_dpo_expanded_qwen3.json`
- `configs/generator_feedback_dpo_expanded_ministral3.json`

The package validator filters the shared review to the focal model, rejects
cross-model response pairs, reconstructs question-capped preferences and checks
all source hashes. Existing Llama review packages remain valid and unchanged.

## GPU submission after review

Once model-specific reviewed archives have been prepared and transferred to
these paths on Gadi, submit separately:

```bash
bash scripts/submit_expanded_generator_dpo.sh \
  outputs/generator_feedback_dpo/reviewed-expanded-qwen3-v2 qwen3
bash scripts/submit_expanded_generator_dpo.sh \
  outputs/generator_feedback_dpo/reviewed-expanded-ministral3-v2 ministral3
```

These commands are **for the post-review stage**, after transferring the v2
packages described below. Each submits smoke, training and validation-selection jobs with
success dependencies. Both models use the existing output root
`outputs/generator_feedback_dpo_expanded`, with model-specific run names and
separate submission records. All epoch adapters remain available regardless
of the strict winner. The `selected_adapter/` directory contains only that
winner, while `ensemble_only_candidate_epoch` is reported separately.

Llama-only condition names remain unchanged. New individual-model conditions
are named `qwen3_epochE_greedy`, `qwen3_epochE_sampling4` and the corresponding
`ministral3_...` names. Ensemble conditions retain
`epochE_sampling4_max5_reranker`. Each checkpoint report includes the focal-model
coverage loss count, ensemble transitions, MRR, candidate and token counts.

These validation questions previously selected the frozen reranker's
regularization. They remain selection diagnostics and do not establish unseen
performance. No automatic combined-model, dev160 or official-test evaluation
is included.

## Completed companion review: 19 October 2026

CPU preparation job `181513120.gadi-pbs` produced the verified source export.
Blinded GPT-4.1-mini review covered 1,939 shortlisted pairs, 3,242 responses and
4,703 distinct annotation units. The API successfully judged 4,699 units. Four
units repeatedly produced malformed, truncated Unicode output; they were
explicitly abstained, and all five affected responses were excluded from pair
selection. They were never assigned semantic approval. The original partial
API review and caches remain unchanged.

A separate, logged derivative removed boundary ellipses from 58 evidence quotes
across 46 units, only where the remaining quotation is an exact substring of its
cited snippet. No successful correctness, support, equivalence, relation or
confidence label changed. The derivative manifest carries the original review
hashes, abstentions and complete quotation-repair log.

Targeted assistant inspection also excluded additional numerical and scope
ambiguities, including strict percentage bounds, carrier frequency versus
disease incidence, and infusion duration versus dose-normalized infusion rate.
These vetoes are confined to the companion configurations; Llama's data and
selection decision remain unchanged.

| Model | Fitting compactness / retention | Fitting pairs / questions | Preference-validation compactness / retention | Validation pairs |
|---|---:|---:|---:|---:|
| Qwen3 | 104 / 13 | 117 / 116 | 24 / 1 | 25 |
| Ministral | 64 / 8 | 72 / 72 | 17 / 1 | 18 |

One fitting question contributes to both objectives for Qwen3, as permitted by
the existing one-pair-per-question-per-objective rule. Both packages meet the
unchanged minima and 8:1 fitting ratio. Checkpoint selection still evaluates all
144 internal-validation questions, not just these preference-validation subsets.

These are compactness preferences, not exclusively harmful-expansion examples.
For example, the selected Qwen3 compactness pairs delete 87 expressions judged
valid and 69 judged harmful; Ministral deletes 61 valid and 33 harmful expressions.
Only 12 Qwen3 and 9 Ministral fitting compactness pairs shrink the fixed ensemble
pool; the others can reduce focal response length while companions retain those
expressions. Generalization and actual MRR preservation require the GPU runs.

The cumulative max-five review ledger is US$2.3860584, including the previous
Llama reviews, within the original US$10 allowance. No GPU jobs were submitted
by the assistant. Aggregate counts, archive hashes and verification are recorded
in `results/companion_compactness_dpo_preferences_20261019.json`.

Use the final **v2** archives in local `Artifacts/`; earlier v1 files are superseded.
Upload from local PowerShell:

```powershell
cd C:\Users\Dustin\Projects\BioASQ-MRES
scp .\Artifacts\generator-feedback-dpo-expanded-qwen3-v2.tar.gz .\Artifacts\generator-feedback-dpo-expanded-ministral3-v2.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

Then on Gadi:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
for model in qwen3 ministral3; do
  preferences="outputs/generator_feedback_dpo/reviewed-expanded-${model}-v2"
  mkdir -p "$preferences" || break
  tar -xzf "/scratch/nl78/$USER/generator-feedback-dpo-expanded-${model}-v2.tar.gz" -C "$preferences" || break
  bash scripts/submit_expanded_generator_dpo.sh "$preferences" "$model" || break
done
```

This submits six jobs: a smoke, DPO training and validation-selection chain for
each model. The two training experiments keep their other two SFT generators
fixed; they do not combine the separately trained DPO models yet.
