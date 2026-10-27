# Coverage-focused generator DPO

This experiment asks whether preferences for successful natural responses improve
accepted-answer coverage at a fixed generation budget. It is distinct from the
previous compactness DPO: neither shorter responses nor fewer candidates define
the preference here. The first stage is a fitting-only CPU audit and review
export, not model training.

## Source and isolation

The source is the completed selector data audit
`outputs/answer_selector_data/audit-181824213.gadi-pbs`, which pins the three native
loss-selected max-five SFT response banks and official Java candidate labels.
Each question has one greedy and four stochastic responses per model. Greedy
and sampling4 are separate inference modes; this is not a ten-sample bank.

All 1,296 fitting questions remain in coverage denominators, including failed
requests and questions with no accepted candidate. All accepted gold aliases
were considered by the existing official matcher. Exact retained surfaces and
their response positions are preserved. Officially unaccepted does not mean
medically incorrect, so evidence, alias, numerical-bound and scope review is
still required before using a pair for training.

Validation144, dev160 and test95 IDs are checked for exclusion. No responses or
labels from these panels are used to construct preferences. These fitting
diagnostics are not estimates of unseen-question performance.

## Two separate preference objectives

1. **Sampled response coverage.** A natural sampled response with an accepted
   answer is preferred to a natural sampled response without one, from the same
   question, model and prompt. These pairs can improve response reliability but
   do not necessarily add coverage to the combined multi-sample pool.
2. **Greedy recovery.** A successful natural sampled response is preferred to
   that model's unsuccessful saved greedy response. This targets greedy failures
   and is exported as a separate objective, not silently mixed into training.

The cap is two pairs per question/model/objective, with content-identical pairs
deduplicated. The cap uses a deterministic seeded order, not output length.
Failed, repaired, capped, excluded and token-ceiling requests remain in baseline
coverage statistics but cannot form this strict pilot's preference pairs.

Three notions must remain separate:

- **Response-level success/failure:** useful supervision even when companions
  already supply an accepted answer.
- **Companions miss:** neither of the other two models supplies an accepted
  answer in its four sampled responses; the focal model can be complementary.
- **Fixed-context coverage gain:** for sampled pairs, remove the chosen response
  from its original slot and hold the other eleven draws fixed. Insert either
  the chosen or rejected response. A gain is possible only when the fixed draws
  lack an accepted answer. For greedy-recovery pairs, hold the other two greedy
  responses fixed instead. These are counterfactuals over saved outputs, not
  observed gains from a trained DPO model.

Questions where every saved response fails need additional positive examples;
the audit never inserts gold aliases into generated outputs or fabricates a
preferred response. First-answer acceptance is recorded separately from the
presence of an accepted expansion.

## Initial portable audit: 27 October 2026

The downloaded audit supplies exact candidate occurrences and official labels,
allowing the following counts without new generation or scoring:

| Model | Mixed sampled success/failure questions | Of these, companions miss | Sampled pairs | Greedy-recovery pairs |
|---|---:|---:|---:|---:|
| Llama | 393 | 24 | 774 | 158 |
| Qwen | 375 | 5 | 738 | 154 |
| Ministral | 391 | 54 | 765 | 150 |

There are 2,739 provisional pairs over 673 distinct fitting questions, involving
3,913 distinct responses. The sampling-only arm has 2,277 pairs. Fixed-eleven
sampled coverage gains are available on 13/4/37 questions for Llama/Qwen/Ministral;
greedy-ensemble recovery on 55/23/72. Counts across models/objectives are not
necessarily independent questions and should not be added as coverage gains.

Fitting source coverage is 1,015/1,296 for the greedy ensemble and 1,105/1,296 for
sampling4. No claim is made about whether DPO will improve those figures or
generalize. The portable audit lacks the original raw JSON responses/prompts;
its files contain IDs and labelled surfaces, not trainable text. The Gadi stage
below binds all original text and exact prompts from the hash-pinned native banks.

## Submit the full review export on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_generator_coverage_dpo_audit.sh
```

An alternative completed selector audit can be passed as the first argument.
Preparation verifies the source hashes, checks fitting-only membership, binds
original prompts/raw outputs to labelled response positions, and freezes code
and inputs. One CPU job writes:

- `summary.json`, `report.md`: model-level coverage, pair counts and limitations.
- `preference_candidates.jsonl`: provisional natural pairs with objective,
  first-answer labels, exact prompts, raw chosen/rejected outputs and fixed-context
  metadata. `training_ready` is false.
- `review_responses.jsonl`: unique responses for semantic/evidence inspection.
- `review_sample.jsonl`: up to 20 pairs per model/objective for initial review.
- `question_outcomes.jsonl`: all fitting questions/model outcomes, including
  all-failure cases.
- `manifest.json`: provenance and artifact hashes.

Outputs go to `outputs/generator_coverage_dpo/audit-JOBID.gadi-pbs` and a portable
archive at `/scratch/nl78/$USER/generator-coverage-dpo-audit-JOBID.tar.gz`.

```bash
record=$(ls -t outputs/generator_coverage_dpo/submission-*/jobs.tsv | head -n 1)
cat "$record"
job=$(awk -F '\t' 'NR==2 {print $2}' "$record")
qstat -fx "$job" | grep -E 'Job Id|job_state|Exit_status|comment'
cat "outputs/generator_coverage_dpo/audit-${job}/report.md"
```

No GPU generation/training, reranker fitting, paid judging or test evaluation is
submitted. There are no dependent training jobs. Existing archives are unchanged.

## After review

Freeze a coverage objective, model-specific reviewed pairs, and an additional-SFT
control using the same preferred examples. Start from the original max-five SFT
adapter rather than continuing compactness DPO. Keep inference prompts and
budgets fixed. Select checkpoints on validation by coverage and quality criteria
defined before training, retaining epoch zero as a baseline. Measure full-pool
and top-five coverage, MRR@5, gains/losses and repeated-generation diversity; more
successful individual responses do not guarantee a more useful ensemble pool.
Test inference can use ten samples, but the four-sample preference source must
be disclosed. The existing larger matched banks can support a later extension.

This audit produces no training-approved dataset or DPO checkpoint. Review the
actual natural responses before constructing the training files, particularly
where official acceptance differs only by wording.
