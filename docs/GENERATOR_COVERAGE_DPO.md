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

## Blinded semantic review

The full Gadi export `audit-182367619.gadi-pbs` contains 2,739 pairs across 673
fitting questions and 3,913 distinct natural responses. An offline inspection
verified all artifact hashes, raw prompts/response parsing, accepted labels and
the fixed-context coverage calculations against the original selector audit.
Seven targeted evidence cases were recorded locally; this was not a semantic
review of every pair or a prevalence estimate.

The paid reviewer is `gpt-4.1-mini-2025-04-14`, with structured candidate-level
correctness, evidence support, original/expansion equivalence and relation-label
judgments. Requests contain the question, all supplied snippets, original and
candidate strings and declared relation. Model identity, draw number, gold
aliases, official acceptance and chosen/rejected direction are hidden. Identical
candidate contexts are reviewed once across models/responses. The current plan
has 9,756 unique contexts in 814 requests.

The official model page lists US$0.40/million input and US$1.60/million output
tokens, verified 28 October 2026:
[OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The conservative whole-plan reserve is US$9.535421, allowing for maximum output
tokens and input/framing overhead without cache discounts. One new authorized
US$10 cumulative cap applies to this batch, including resumptions. Estimates do
not guarantee all requests succeed. Model availability must not trigger silent
substitution of another model.

Prepare without calls, locally:

```powershell
.venv\Scripts\python.exe scripts/judge_generator_coverage_dpo.py `
  Artifacts/generator_coverage_dpo_audit_182367619 `
  --selector-audit Artifacts/answer_selector_data_v1 `
  --output Artifacts/generator_coverage_dpo_judge_mini_20261028 `
  --max-usd 10
```

When authorized and API credit is available, add `--run`. A small initial batch
can use `--max-new-calls 8 --workers 2`. Resume with the exact same output
directory and cap, without the small call limit. Saved successful requests are
reused; partial failures remain unresolved. The shared exclusive session lock
and conservative spending ledger prevent a new process from resetting the cap.
Do not delete or replace the ledger to retry a failed batch. Credentials are
read from the existing local key file at request time and never printed.

The first attempted small batch was refused with `credit_balance_exhausted`:
zero valid judgments were obtained, and queued calls stopped. No further calls
were made until the user restored credit and explicitly requested a retry. The
retry succeeded; the full review then resumed using the same ledger and cap.
The ledger retains a conservative US$0.0146784 for two failed requests; this is
not a claim that the provider billed that amount. Successful requests are cached.
Concurrency can be 1..12 workers; a restart to increase throughput preserves
completed caches and conservatively retains reservations for interrupted calls.

Outputs keep raw text unchanged and separate provisional categories:

- `semantic_coverage_repair_candidate`: every preferred expression is clean and
  every rejected expression has a high-confidence incorrect/contradicted
  judgment with a validated evidence quote; no supported correct rejected
  answer remains.
- `benchmark_form_preference`: both complete responses are clean supported
  answers, distinguished by official acceptance; this is not a factual repair.
- `rejected_contains_supported_answer`: a valid rejected answer remains, even
  if other rejected expansions have problems.
- `preferred_not_clean`, `evidence_support_review`, `unresolved`, `pending`:
  not eligible for a clean semantic pilot.
- `assistant_evidence_hold`: explicit question-level holds for numerical,
  receptor/ligand, insufficient-popularity or conflicting-evidence examples.
  These holds are applied after blind annotation, not shown to the judge.

Repeated contradictory candidate judgments abstain, and quoted snippets are
checked literally. Neither self-reported high confidence nor an official gold
match establishes expert correctness. All reviewed files retain
`training_ready: false`; no training job is submitted by this reviewer.

### Completed automated review: 28 October 2026

After API credit was restored, all 814 request groups and 9,756 unique candidate
contexts were reviewed, covering the 3,913 natural responses and 2,739 pairs.
One wrong-length indexed response was retried; successful caches were reused.
Recorded estimated costs plus conservative failed/interrupted-call reservations
total US$2.277577 under the original US$10 cap. This is not an account billing
statement; successful-request token costs at the listed standard rates total
US$2.217937, before any actual cache discounts.

| Provisional category | Pairs |
|---|---:|
| Both complete responses judged correct/supported; official form differs | 825 |
| Preferred response not fully clean | 377 |
| Rejected response still contains a supported correct answer | 166 |
| Unresolved, inconsistent or invalid judgments | 1,355 |
| Assistant evidence hold | 12 |
| Evidence-support review | 4 |
| Meets strict automatic semantic-repair criterion | 0 |

The benchmark-form subset spans 296 distinct fitting questions: 257 Llama pairs,
282 Qwen pairs and 286 Ministral pairs. It contains 671 sampled-response pairs
and 154 greedy-recovery pairs. Ninety-two pairs over 44 questions have a
counterfactual saved-output official-coverage gain, not measured DPO improvement.

Zero pairs passing the strict semantic-repair filter does not establish zero
genuine errors. Full-response requirements, citation validation, unresolved
attributes and model inconsistencies can exclude real corrections. The judge
itself was too strict about rounding in the approximate-count example and
sometimes confused self-equivalence with correctness. Those decisions remain
recorded and held, rather than silently corrected or accepted.

The 825 cases are a possible **official-answer-form coverage** arm, not 825
scientific corrections. Inspect the subset and retain this objective distinction
before training. The source audit, paid plan, ledger, reviewed responses and
post-review files remain local under `Artifacts/`; aggregate reproducibility
metadata are in `results/generator_coverage_dpo_review_20261028.json`. No GPU job,
training-approved dataset or DPO model was produced by this review.

## Inspected answer-form pilot (28 October 2026)

The completed judge's 825 `benchmark_form_preference` pairs are a shortlist,
not 825 verified factual corrections. The follow-up CPU inspection verifies
the source/selector/review hashes and reconstructs the official-form category
from the reviewed native responses. It samples questions deterministically
without using preference gains or response length: the first 24 hash-ordered
questions per model yield 49 distinct questions and 128 pair records. This is
exploratory assistant inspection, not a preregistered study or expert review.

The inspection retained 20 questions and 51 unique native preferences. It held
76 sampled records for scope, quantity, precision, answer-type or evidence issues;
one retained duplicate was collapsed. The other 697 records were outside the
inspected sample and are excluded without claiming they are wrong. Remaining
judge errors include treating `10–15%` and `10–28%`, or `NSCLC` and
`EGFR-mutant NSCLC`, as pure wording alternatives.

| Generator | Training pairs / questions | Fitting-only inner holdout pairs / questions |
|---|---:|---:|
| Llama | 14 / 8 | 9 / 5 |
| Qwen | 9 / 5 | 4 / 2 |
| Ministral | 11 / 6 | 4 / 2 |

The shared inner holdout uses fitting question IDs only, consistently across
models. It does **not** consume validation144. Prompts and full native responses
are unchanged; gold aliases, acceptance labels and review notes do not enter
model messages. Preferred-only SFT files have the same preferred-response
exposure as DPO, including intentional repeats where one preferred completion
participates in different comparisons. Identical native preferences collapse
across draw/objective metadata with all original pair IDs retained.

This is an **official-answer-form coverage** dataset. Preferred responses are
shorter in 28 pairs, longer in 22 and equal in one; they have fewer candidates in
12, more in 17 and the same count in 22. No compactness rule is applied.
Only three representative saved-response swaps on one PAR2 question improve
fixed-companion official coverage. That example changes `320 kbp` to accepted
`320kbp`, not the quantity. These counts do not demonstrate new-question
coverage or improvement after training.

The local export includes model-specific DPO and matched additional-SFT files,
the question split, full inspected cases, exclusions, source hashes and a pilot
plan. It remains `training_ready=false`: native Gadi source/adapter verification,
tokenizer/context preflight without silent truncation, and GPU smoke execution
are not performed by this data-preparation command. Given the small question
counts, expand the inspection before a substantive GPU experiment. No new
generation, API calls, model training, job submission or test evaluation occurs.

```powershell
.venv\Scripts\python.exe scripts/prepare_generator_form_pilot.py `
  --source Artifacts/generator_coverage_dpo_audit_182367619 `
  --reviewed Artifacts/generator_coverage_dpo_judge_mini_20261028 `
  --selector-audit Artifacts/answer_selector_data_v1 `
  --output Artifacts/generator_form_pilot_20261028_v1 `
  --archive Artifacts/generator-form-pilot-20261028-v1.tar.gz
```

The export is immutable: choose a new output name when rerunning. Inspection
decisions and literal evidence quotes are tracked in
`configs/generator_form_pilot_inspection.json`, bound to this exact review.
Compare each generator's unchanged max-five SFT, additional preferred-only SFT
and DPO from that same starting SFT/reference, with fixed prompts, decoding
budget, companion models and reranker. Evaluate official full-pool coverage,
MRR@5, top-five coverage, per-question changes and output costs on validation144,
retaining epoch zero as an option. These unequal model-specific preference
datasets cannot establish that one backbone is inherently better.
