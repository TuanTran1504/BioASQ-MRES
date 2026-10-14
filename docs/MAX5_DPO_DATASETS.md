# Separate initial-answer and expansion-error preferences

This phase generates max-five banks, labels first answers and expansions
independently, and constructs two model-specific preference datasets. It does
not start DPO training. The user authorised GPT-4.1-mini with a **new US$10 cap**
on 14 October 2026; no paid calls were made during workflow preparation.

## Sources and generation

Each model starts from its frozen **best validation-loss max-five SFT adapter**,
epoch 2 in all three completed runs. Generation checks training/selection
metadata, actual adapter/tokenizer hashes, cached backbone and dataset
provenance. Completed dev-evaluation manifests locate frozen checkpoints only;
no dev questions, answers or metrics are used to construct the banks.

Use all **1,296 fitting / 144 validation questions**, disjoint from dev160 and
the audited official test questions. Generator prompts contain questions and
all snippets, without gold labels or teacher targets. Generate one greedy and
four stochastic responses per question (temperature 0.8, top-p 0.95, seed 3407).
Each response requests at most five equivalent expressions. This gives **7,200
responses per model / 21,600 total**. Raw responses, seeds, prompts, source
hashes, token counts and timings are retained.

The total allocation is 8,704 tokens with 512 output tokens, accommodating the
source SFT's 8,192-token inputs plus an output allowance. Oversized inputs fail
instead of truncating snippets. Strict JSON failures, over-five responses and
outputs at the token ceiling remain archived but do not supply primary pairs.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_max5_dpo_banks.sh
```

Six jobs: smoke and dependent full generation per model. All source checks run
before any submission. Append model names to restrict the submission. Smoke
generates two fitting and two validation questions with all five draws; full
generation preflights all input lengths. Existing runs are preserved.

```bash
record=$(ls -t outputs/max5_dpo_banks_8b/submission-*.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

After all three full jobs succeed:

```bash
bash scripts/prepare_max5_dpo_review.sh
```

This verifies bank membership, prompts, seeds, draws and hashes; applies the
official Java matcher independently to candidates; and packages a review at
`outputs/max5_dpo_banks_8b/review-v1`. The portable archive is
`/scratch/nl78/$USER/max5-dpo-review-v1.tar.gz`, with raw banks embedded and no
adapter weights. No semantic labels or paid calls occur in this step. A specific
submission record and a new output directory can be supplied as arguments.

Download in Windows PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
New-Item -ItemType Directory -Force .\Artifacts\max5_dpo_review_v1 | Out-Null
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/max5-dpo-review-v1.tar.gz .\Artifacts\max5-dpo-review-v1.tar.gz
if ($LASTEXITCODE -eq 0) {
    tar -xzf .\Artifacts\max5-dpo-review-v1.tar.gz -C .\Artifacts\max5_dpo_review_v1
}
```

## Independent candidate judgments

For every first/additional candidate, judge correctness as an answer, evidence
support, equivalence to the first answer, relation validity and confidence.
The first answer is equivalent to itself even if wrong. Model identity, draw,
official acceptance and intended preference direction are hidden. Equivalent
contexts are deduplicated. Exact quote checks, caches, a process lock and a
shared spend ledger constrain paid calls. Uncertain/failed judgments abstain;
official mismatches are not automatically incorrect. Conflicts, invalid quotes,
relation-only disagreements and absence-only incorrectness do not create
primary negative labels. Validation requests precede deterministically shuffled
fitting requests, protecting validation yield if the cap is reached. A capped
or failed run remains explicitly partial.

Use `gpt-4.1-mini-2025-04-14`. Rates checked against
[official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
are US$0.40 input / US$1.60 output per million tokens; budgeting assumes no cache
discount. From the local repository root, prepare without spending:

```powershell
.venv\Scripts\python.exe scripts/judge_max5_dpo_banks.py Artifacts/max5_dpo_review_v1 --output-dir Artifacts/max5_dpo_judge_mini_v1
```

Execute/resume under the authorised cap:

```powershell
.venv\Scripts\python.exe scripts/judge_max5_dpo_banks.py Artifacts/max5_dpo_review_v1 --output-dir Artifacts/max5_dpo_judge_mini_v1 --run --max-usd 10
```

Outputs are `reviewed-llm.jsonl`, `first_answer_labels.jsonl` and
`expansion_labels.jsonl`. All responses remain represented, including excluded
ones. These are LLM labels without an independent human audit.

## Preference construction

**Initial-answer objective:** naturally generated responses from the same
model/question, with equal candidate count and identical relation-type sequence.
Chosen first answer is semantically correct and supported; all its variants
are valid equivalents. Rejected first answer and its coherent variants are
incorrect, while equivalent to their own wrong core. Abstain if matched shapes
are unavailable. Supported C2 answers can be positives; do not manufacture
C3>C2 format preferences. The core and its variants necessarily change together;
this controls response shape, not the cognitive cause of error.

**Expansion-error objective:** the same verified correct and supported first
answer stays at position one. Negatives add decisively incorrect,
non-equivalent or contradicted candidates. Harmful candidates must not be
officially accepted, preserving accepted coverage in these conservative pairs.

- Deletion: remove only verified harmful additions and preserve valid remaining
  candidates in their order. The chosen completion is explicitly edited.
- Equal-count correction: replace harmful positions using independently judged
  valid alternatives naturally generated by the same model/question with the
  same first answer. Record donor response IDs. Do not invent replacements or
  insert gold aliases. Candidate count is controlled; token length can differ.

Wrong/unsupported first answers and unresolved candidates cannot supply
expansion-error pairs. Relation-label disagreement alone is outside this arm.
Deduplicate completions, cap at two pairs per model/question/objective, and
round-robin across correction/deletion types where both are available.

Construct locally after judging:

```powershell
.venv\Scripts\python.exe gadi_sft_8b_starter/scripts/max5_dpo_data.py --mode pairs --review-dir Artifacts/max5_dpo_review_v1 --annotations Artifacts/max5_dpo_judge_mini_v1/reviewed-llm.jsonl --output Artifacts/max5_dpo_preferences_v1
```

Output: `MODEL-initial-train.jsonl`, `MODEL-expansion-train.jsonl` and separate
validation files. Raw banks, labels, source IDs and hashes accompany pairs.
Official labels and raw candidate order are rechecked against the review.
Cross-model pairs and fitting/validation overlap are forbidden.

Optional `MODEL-ARM-matched-SPLIT.jsonl` files restrict both objectives to the
same questions and equal pair counts per question. The manifest records full
and matched yields, construction types and exclusions. Empty required splits
mark the package `insufficient_pairs`; empty matched subsets cannot support an
objective comparison. Full arms can differ in pair counts, questions and token
budgets, qualifying any comparison attributed to objective choice.

DPO training comes after inspecting yield and label quality. Initialise each
objective from its own frozen source SFT checkpoint/reference. Measure first
answer correctness, valid alternatives retained, harmful expansions, MRR@5 and
coverage gains/losses. Fewer alternatives alone is not success.

## Verification

Focused tests check partition/draw integrity, teacher labels excluded from
generation, C2 positives, wrong-core exclusion, conflict/uncertainty abstention,
validated replacement provenance, matched response shape, accumulation across
questions, model/split isolation, output ceilings, blind mini payloads, budget
safeguards and mocked smoke/full submission dependencies. Actual GPU feasibility
and source loading require the new Gadi smoke tests.
