# Strict semantic evaluation of the current validation144 generator pools

Supplement the official metrics with Semantic MRR@5, coverage@5, coverage@10
and full-pool coverage. The rankings and official scores remain unchanged.
Evaluate the original SFT baseline, selected model-specific DPO and selected
shared-data DPO, each with greedy and four-sample ensemble generation: six
conditions, all 144 validation questions. No fitting questions, test95 scoring,
new generation, checkpoint reselection or training preferences are involved.

## Labels and uncertainty

- C3 is the existing official acceptance label; GPT does not assign or re-judge it.
- C2 means a complete, high-confidence equivalent answer to this question,
  interchangeable with at least one supplied reference alias. Formatting,
  established abbreviations, synonyms and harmless paraphrases can differ.
- C1 requires a clear failure to answer the question, not merely an unaccepted
  string. A valid alternative not equivalent to a reference remains unresolved.
- Uncertain, missing, contradictory, malformed or lower-confidence judgments
  remain unresolved. They are never silently relabelled C1 or removed from the
  question denominator. The report gives confirmed semantic matches plus
  optimistic bounds for unresolved candidates. These bounds are conditional on
  treating the resolved labels as correct, not calibrated statistical intervals.

`>54000` and `greater than 54,000` preserve a strict bound; `>=54000`,
`approximately 54000` and `exactly 54000` do not. Scope, species, population,
units, qualifiers, numerical claims, relationships and negation must retain their
meaning. The number of word edits does not determine equivalence. Question
context determines whether extra details are substantive.

The GPT judge receives all gold aliases, the full question and snippets and
indexed candidate surfaces. It sees no rank, model, condition, official labels,
vote counts or preference direction. Each unique question/context/answer gets
one cached judgment reused across conditions. Exact case/spacing variants remain
distinct. Evidence support is recorded separately: missing snippet support is
not itself proof of incorrectness. Contradiction, inconsistent equivalence or
invalid evidence quotes prevent a C2 label. Earlier broad C2 labels are not reused.

GPT labels are automated judgments, not independent biomedical expert review.
`review_candidates.jsonl` contains every unmatched candidate and its full context;
`review_sample.jsonl` selects up to 30 examples per resulting class for checking.
Results are internal validation diagnostics after checkpoint/design selection;
semantic rescoring does not make them independent test evidence. Validation
annotations must not become DPO training examples.

## Export the actual saved validation pools on Gadi

This command uses only Python's standard library and makes no API calls. It
verifies the completed recovery's frozen files, all 144 question memberships,
full candidate rankings, official labels and per-question/aggregate metrics.
Gold aliases are recovered from the original official evaluator inputs, not
inferred from the candidate pool. All six selected conditions retain every rank.

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
module load python3/3.12.13
python3 ../scripts/semantic_pool_evaluation.py export \
  --comparison outputs/generator_form_designs/scoring-recovery-20261029-115854-1119354/run-182573009.gadi-pbs \
  --output "outputs/semantic_evaluation/validation144-$(date -u +%Y%m%d-%H%M%S)" \
  --archive /scratch/nl78/$USER/semantic-val144-pools.tar.gz
```

Transfer and extract in local PowerShell:

```powershell
$artifactRoot = 'C:\Users\Dustin\Projects\BioASQ-MRES\Artifacts'
New-Item -ItemType Directory -Force $artifactRoot | Out-Null
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/semantic-val144-pools.tar.gz "$artifactRoot\semantic-val144-pools.tar.gz"
if ($LASTEXITCODE -eq 0) {
    New-Item -ItemType Directory -Force "$artifactRoot\semantic_val144_pools" | Out-Null
    tar -xzf "$artifactRoot\semantic-val144-pools.tar.gz" -C "$artifactRoot\semantic_val144_pools"
}
```

## Plan before paid annotation

From the local repository root, using the project's Python environment:

```powershell
.venv\Scripts\python.exe scripts/judge_semantic_answer_equivalence.py `
  Artifacts/semantic_val144_pools `
  --output Artifacts/semantic_val144_judge
```

This prints the actual number of unique unmatched candidates, planned requests
and a conservative cost reserve. It does not read an API key or call GPT.
Earlier spending caps cover separate experiments: obtain a new explicit cap
before using `--run --max-usd AMOUNT --max-new-calls COUNT`. Start with a small
request pilot, inspect the annotations, then resume the same frozen plan and
spending ledger. Successful cached requests are reused; failures do not trigger
unbounded automatic retries. Source/rubric/model/code changes cannot silently
resume the same output directory. Partial runs remain marked incomplete.

The initial pinned judge is `gpt-4.1-mini-2025-04-14`, temperature zero, with
strict JSON-schema output. The [official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
lists structured outputs and US$0.40/M input, US$1.60/M output tokens. It also
marks this snapshot deprecated; verify account availability before paid use.
The pipeline does not silently switch models. [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat)
constrain the schema, not the factual correctness of judgments. Requests, raw
responses, usage, validation issues and charged/reserved costs are retained.

The first 20-request pilot and a separate revised-rubric mini pilot exposed
over-acceptance of unverified identifiers and inconsistent handling of reference
scope. These partial outputs are diagnostic pilots, not final semantic scores.
Rubric version 2 additionally requires an explicit same-scope/bidirectional
equivalence decision and identification of unverified added text. Invalid or
unverified whole-answer equivalence remains uncertain even if a correct core
name appears in the candidate.

A stronger-model comparison can be planned with `--model gpt-4.1-2025-04-14`.
Its [official model page](https://developers.openai.com/api/docs/models/gpt-4.1)
lists US$2/M input and US$8/M output tokens. This is a separate explicit model
choice, not an automatic fallback, and requires authorization for that model.
Use a new versioned output directory and `--budget-directory ORIGINAL_JUDGE_DIR`
to preserve the original cumulative batch cap across rubric/model pilots.
This directory must have the original source/model metadata, approved cap and
spending ledger. All prior charges/reserves count against the same limit.
The report identifies the chosen model and hashes its exact rubric and code;
cached judgments from different payloads/models are not mixed.

Successful execution writes `evaluation/report.md`, `semantic_summary.json`,
per-question metrics, all annotated candidates, uncertainty cases and official
versus semantic gain cases. Confirmed semantic matching is C3 or strict C2.
No generator, reranker or DPO training is run.

## Conservative post-annotation review

Keep the raw GPT outputs unchanged and write reviewed results to a separate
directory. The review layer only changes questionable C1/C2 decisions to
`uncertain`; it never invents positive labels or changes C3. All six conditions
share the same candidate-level judgments and review holds.

```powershell
.venv\Scripts\python.exe scripts/review_semantic_pool_annotations.py `
  Artifacts/semantic_val144_pools `
  --judge Artifacts/semantic_val144_judge_full `
  --output Artifacts/semantic_val144_reviewed `
  --holds Artifacts/semantic_val144_manual_holds.jsonl
```

Review hints flag contradictory C1/C2 labels for case/spacing variants, positive
rationales admitting scope differences, identifiers absent from the supplied
references/evidence, replacement characters and quantitative mismatches. The
numerical hint recognizes common digit/word forms and flags changed bounds,
ranges, approximation or units. It is not a general mathematical-equivalence
solver: valid conversions or contextual answers may also be withheld for review.
None of these mechanical hints proves an answer incorrect. Assistant holds
record specific scope/type/completeness concerns with explicit reasons; they
are not independent expert labels. The original GPT scores remain available
alongside the conservative reviewed scores. Do not report either as official
BioASQ scores or as validated clinical correctness.

The reviewed report includes all 144 questions. `review_audit.json` gives
question-level metric uncertainty and paired changes from the SFT baseline;
`review_holds.jsonl` preserves every abstention and the original GPT class.
These are post-hoc validation diagnostics. Resolving the retained uncertainty
requires further adjudication, not silently treating it as C1.
