# Original single-answer 8B SFT → DPO

Continue the original single-answer SFT adapters, not the expansion adapters:
Llama `180794568`, Qwen3 `180794572`, Ministral `180878745`. Each model's own
initial original SFT adapter is its frozen DPO reference. Original prompts,
native templates, full evidence and `Answer: [BE]expression[EE]` completions are
retained. Qwen thinking is disabled; Ministral uses its proven eager backend.

## Generate fresh banks on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_original_sft_dpo_8b.sh generate
```

This submits six jobs: generation smoke and dependent full generation per model.
Full banks cover the same 1,296 fitting and 144 internal-validation questions.
For each question each model produces one greedy and four sampled answers
(temperature 0.8, top-p 0.95), giving 7,200 attempted responses per model.
No development/test aliases enter generation prompts or preference training.
The 8,704 total / 128 output token limits leave room for the full training-input
contexts previously checked at 8,192 tokens. Overflow fails instead of truncating.
Generation/scoring scripts are locally tested; CUDA feasibility is verified by smoke.

```bash
record=$(ls -t outputs/original_sft_dpo_8b/submission-generate-*.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $4}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Full bank names are `MODEL-mixed-generate-FULL_JOB_ID.gadi-pbs` under
`outputs/original_sft_dpo_8b`. The `mixed` name is a default strategy marker;
generation is shared across all preference strategies. Generation jobs make no
paid judge requests and do not train DPO automatically.

## Official scoring and semantic review

After all banks complete, replace the example job IDs below with real full IDs:

```bash
python scripts/single_answer_dpo_data.py --mode review \
  --banks outputs/original_sft_dpo_8b/llama31-mixed-generate-LLAMA_JOB_ID.gadi-pbs \
          outputs/original_sft_dpo_8b/qwen3-mixed-generate-QWEN_JOB_ID.gadi-pbs \
          outputs/original_sft_dpo_8b/ministral3-mixed-generate-MINISTRAL_JOB_ID.gadi-pbs \
  --output outputs/original_sft_dpo_8b/review-v1
```

The official Java matcher identifies accepted C3 strings. Unmatched responses
remain unresolved. Invalid single-answer formats are excluded from preferences
and remain visible in the bank. The review package embeds raw responses and bank
manifests so it can be downloaded for local judging without adapter weights.

The user authorised **GPT-4.1 mini with a new US$10 cap** for this bank. The judge
uses `gpt-4.1-mini-2025-04-14`, structured output, exact-request caching and a
resumable spend ledger. It checks correctness independently of official matching,
requires high confidence and real snippet quotes for supported decisions, and
excludes uncertain judgments or official/semantic conflicts. Identical question/
answer judgments are shared across models; generator IDs and intended preference
directions are hidden. These are automatically labelled data, without a human audit.

The official model page lists $0.40 input / $1.60 output per million tokens:
[OpenAI GPT-4.1 mini documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The cap includes conservative reserves for failed/unknown-usage requests, uses
full input pricing without assuming cache discounts, and is shared across all
three banks. Concurrent paid processes in the same output folder are refused.
Reuse the same judge folder on resume; do not create multiple US$10 runs.
Ledger writes are atomic and serialized, with retries for brief Windows reader
locks. DNS failures stop queued requests and are unbilled because no connection
reached the API; other unknown-usage failures retain their conservative reserves.

From the full repository root, prepare a no-cost plan first. The example assumes
the package remains in the starter folder; use its actual local path if downloaded:

```bash
python scripts/judge_single_answer_dpo.py \
  gadi_sft_8b_starter/outputs/original_sft_dpo_8b/review-v1 \
  --output-dir Artifacts/original_sft_dpo_judge_mini_v1
```

The explicitly paid invocation, authorised for this bank only, is:

```bash
python scripts/judge_single_answer_dpo.py \
  gadi_sft_8b_starter/outputs/original_sft_dpo_8b/review-v1 \
  --output-dir Artifacts/original_sft_dpo_judge_mini_v1 \
  --run --max-usd 10 --max-new-calls 2000
```

Use the configured local API key file or provide `--api-key-file` explicitly.
Do not place credentials in Git or job variables. Partial judgments stay excluded;
the cap is not a promise that every answer will be judged. A subsequent human
sample audit is needed for publication-quality label claims.

Incomplete batches can be resumed with `--retry-singletons --output-tokens-per-answer 640`.
This retains exact cached requests and retries only missing answers individually,
checking that question, evidence, aliases, model and judging instructions match.
Up to eight workers share the same cap. Unsupported negative judgments with no
quotes whose rationale only cites missing evidence are excluded: absent support
does not establish scientific incorrectness. Excluded decisions remain available
for audit alongside their unchanged raw answers.

## Pair construction

C3 is officially accepted, C2 scientifically correct but unaccepted, C1 incorrect.
For C3>C1 prefer the supported accepted answer over a clearly incorrect answer.
For C3>C2 both answers must be supported; this teaches accepted wording rather
than a concept correction. Only naturally generated, complete tagged responses
are used. No teacher/gold answers are inserted to guarantee a chosen response.
Pairs stay within their question and partition and preserve raw completion text.
Case/format-only duplicates do not earn preferences. Each question supplies at
most two pairs per strategy; mixed interleaves concept/form pairs where available.

Transfer the judged `reviewed-llm.jsonl` back to Gadi if judging ran locally, then:

```bash
python scripts/single_answer_dpo_data.py --mode pairs \
  --banks outputs/original_sft_dpo_8b/llama31-mixed-generate-LLAMA_JOB_ID.gadi-pbs \
          outputs/original_sft_dpo_8b/qwen3-mixed-generate-QWEN_JOB_ID.gadi-pbs \
          outputs/original_sft_dpo_8b/ministral3-mixed-generate-MINISTRAL_JOB_ID.gadi-pbs \
  --annotations /path/to/reviewed-llm.jsonl \
  --output outputs/original_sft_dpo_8b/preferences-mini-v1
```

The package embeds bank responses and hashes, and recomputes all pairs when
validated. Report reliable/excluded response counts, class counts, pair yield
and contributing question counts. Concept/form subsets can have different
questions and pair counts; comparisons across those strategies are exploratory
unless question sets and update budgets are explicitly matched later.

## Train and evaluate

```bash
bash scripts/submit_original_sft_dpo_8b.sh train \
  outputs/original_sft_dpo_8b/preferences-mini-v1 all
```

`all` requests concept-only, form-only and mixed DPO for all three models: nine
conditions, each with training smoke, full training and a dependent paired greedy
evaluation (27 jobs). To reduce scope choose `mixed`, `concept` or `form`, then
optionally append model keys. Empty strata or fewer than ten planned optimizer
updates are refused before any qsub; inspect pair yield and select viable arms.

Two epochs, learning rate 0.000005, beta 0.1 and accumulation 16 are starting
settings. Warm-up is **5% of planned updates**, rounded up, and must finish before
training ends. For 159 pairs this gives 20 updates with one warm-up update, rather
than the previous ten-update/ten-warm-up pilot. Native completion masks, cached
initial-SFT reference likelihoods and dropout-disabled scoring reuse the tested
DPO implementation. Checkpoints are selected by internal-validation preference
loss; this is an exploratory setting, not selection by answer MRR. A continued-SFT
control and training-seed repeats are still needed to attribute any gain robustly.

The evaluation jobs regenerate both original SFT and DPO on the same 160 dev
questions using identical greedy one-answer inference. Official MRR equals exact
accuracy here; paired confidence intervals and question gains/losses are saved.
These dev questions have already informed earlier design decisions; they are
not an unseen confirmatory test. Reports live under
`outputs/original_sft_dpo_evaluation/MODEL-STRATEGY-evaluation-JOB_ID.gadi-pbs`.

Training records are `submission-train-*.tsv`; monitor them with column four as
above. If generation finished but Java scoring failed in an evaluation, rerun
`python ../scripts/analyze_original_sft_dpo.py ACTUAL_EVALUATION_DIRECTORY`.
The three full banks completed on Gadi as jobs `181032357`, `181032359` and
`181032361`. Their officially scored review package was downloaded locally on
13 October 2026 for the authorized mini judging and preference construction.
Gadi training submission requires the user's authenticated session; local SSH
access is unavailable.

## Completed mini-labelled dataset — 13 October 2026

All **8,224 unique question/answer units** were judged. The 21,600 generated
responses retain their raw text; **17,361** pass the training-label filters:
8,844 C3, 7,595 C2 and 922 C1. Exclusions comprise 3,770 uncertain/invalid judgments,
276 official/semantic conflicts, 191 absence-only incorrectness judgments and
two invalid single-answer formats. No judgments remain pending.

The shared ledger totals **US$2.2814128 billed or conservatively reserved**, below
the authorized US$10 cap; this is an upper estimate at full input rates, including
unknown-usage reserves, rather than a billing receipt. Failed DNS attempts never
reached the API and are documented as unbilled. Completed requests are cached;
remaining batch omissions and truncations were retried individually.

| Strategy | Train pairs | Train questions | Validation pairs | Validation questions | Updates / warm-up |
| --- | ---: | ---: | ---: | ---: | ---: |
| Concept: C3 > C1 | 200 | 144 | 25 | 18 | 26 / 2 |
| Form: C3 > C2 | 965 | 557 | 97 | 57 | 122 / 7 |
| Mixed | 1,021 | 583 | 106 | 61 | 128 / 7 |

Mixed training contains 151 concept and 870 form contrasts, rather than equal
numbers of each. These packages are shared across all three target models.
All three strategies passed hash, raw-response, question-partition, pair
reconstruction and training-schedule validation. The concept arm has only 18
validation questions, and strategy datasets/update budgets differ. These are
exploratory automatically labelled preferences without an independent human audit.

Aggregate counts and provenance are saved in
[`results/original_sft_dpo_preferences_mini_20261013.json`](../results/original_sft_dpo_preferences_mini_20261013.json).
The local portable archive is `Artifacts/original-sft-dpo-preferences-mini-v1.tar.gz`
(14,712,603 bytes; SHA256
`a4b65ef7e2572799b17503fb3637c2d4d2959fee654075be0f0c61a2eba72397`).
It embeds the banks, reviewed annotations and all six pair files; all embedded
content hashes were verified after compression. Raw data and API caches stay
outside Git.

Upload from Windows PowerShell at the repository root:

```powershell
scp .\Artifacts\original-sft-dpo-preferences-mini-v1.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

Then in the authenticated Gadi session:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
mkdir -p outputs/original_sft_dpo_8b/preferences-mini-v1
tar -xzf /scratch/nl78/$USER/original-sft-dpo-preferences-mini-v1.tar.gz \
  -C outputs/original_sft_dpo_8b/preferences-mini-v1
python scripts/single_answer_dpo_data.py --mode validate \
  --output outputs/original_sft_dpo_8b/preferences-mini-v1 --strategy mixed
bash scripts/submit_original_sft_dpo_8b.sh train \
  outputs/original_sft_dpo_8b/preferences-mini-v1 mixed
```

`mixed` submits all three models with smoke → training → dev160 evaluation
(nine jobs total). Use `all` instead only when requesting all nine strategy/model
conditions (27 jobs). These commands are the next Gadi step; this dataset export
does not establish that DPO training has run.
