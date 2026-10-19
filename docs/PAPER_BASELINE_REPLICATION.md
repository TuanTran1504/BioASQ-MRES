# UniTor and GPTPrompt factoid replications

This experiment evaluates the publicly released answer-generation methods on the
same 95 official BioASQ 13B factoid questions already used in this project. It does
not reproduce Phase A retrieval, ideal answers, or the complete challenge systems.
The test questions have already been examined during project analysis; they are
not a new untouched holdout. No test answers enter training or generation prompts.

## Completed GPTPrompt run, 19 October 2026

The authorized local run completed all 95 questions with returned model
`gpt-4-turbo-2024-04-09`. It cost **US$3.88723** of the US$10 cap.

| Batch | Questions | MRR@5 | Coverage@5 |
|---|---:|---:|---:|
| 13B1 | 26 | 0.397436 | 12/26 |
| 13B2 | 27 | 0.592593 | 16/27 |
| 13B3 | 20 | 0.375000 | 8/20 |
| 13B4 | 22 | 0.568182 | 13/22 |
| All questions | 95 | **0.487719** | **49/95** |

The equal-batch MRR mean is 0.483303. No empty parses or output-ceiling responses
occurred. Full public metadata is in
`results/paper_replication_gptprompt_test95_20261019.json`; private raw artifacts
are in `Artifacts/gptprompt_replication_test95_v1`. This is the released-code
reconstruction described below, not the exact winning Batch-4 run. UniTor GPU
training has been prepared but has not been submitted or validated on Gadi.

## What is reproduced

| Setting | UniTor UT3 | GPTPrompt released factoid code |
|---|---|---|
| Authors' repository | [crux82/BioASQ2025-UNITOR](https://github.com/crux82/BioASQ2025-UNITOR) | [dsgt-arc/bioasq-2025-1b-1](https://github.com/dsgt-arc/bioasq-2025-1b-1) |
| Pinned commit | `eea59cf1b3727584abc876acbcccb3ddf0b2a957` | `6eb8d3bfdf526b44b598fe2c4546f2995cc9dd20` |
| Backbone | Released Llama-3.1-8B-Instruct 4-bit; optional Phi-4 14B 4-bit | `gpt-4-turbo`, exactly as in the released requests |
| Training | Fresh multi-task SFT on authors' released data | None; released few-shot examples |
| Test evidence | First three provided gold snippets, original order | All provided gold snippets, original order |
| Generation | One greedy response, 256 output tokens | One response, 100 output tokens; decoding parameters omitted as in released Batch requests |
| Ranking | Generated answer order, first five | Generated answer order, first five |
| Evaluation | Existing official BioASQ Java matcher | Same matcher |

Sources and model revisions are recorded in
`gadi_sft_8b_starter/configs/paper_replication_sources.json`. Downloads are verified
against SHA-256 hashes. Upstream notebooks are read as JSON/AST, never executed.

### UniTor training protocol

The released datasets contain 7,588 fitting rows and 956 validation rows, including
yes/no, factoid, list and summary examples. Fitting factoids: 1,274; validation
factoids: 157. Neither split overlaps test95 by ID or normalized question text.
Two normalized question texts occur in both released fitting and validation sets;
the replication preserves that split and discloses the overlap.

| Setting | Llama | Phi-4 |
|---|---|---|
| LoRA rank / alpha | 32 / 32 | 16 / 16 |
| Learning rate | 0.0006 | 0.0002 |
| Training batch / accumulation | 8 / 4 | 2 / 4 |
| Epochs | 2 | 2 |
| Maximum training length | 1,024 | 1,024 |
| Warm-up | 5 steps | 5 steps |
| Optimizer / schedule | 8-bit AdamW / linear | Same |
| Weight decay / seed | 0.01 / 3407 | Same |
| Selection | Lowest released-validation loss | Same |

Both use the seven published projection targets, zero LoRA dropout, completion-only
loss and no packing. The Llama prompt is plain text, not the project's chat prompt.
Phi-4 uses the authors' Phi-4 chat template. A Transformers Trainer port replaces
the old TRL API; it is a method reproduction, not a bitwise reproduction of the
authors' software environment. Dataset and mask statistics are saved. Empty
completion targets cause a stop rather than silently training on prompt tokens.

Local tokenizer checks found two truncated Llama fitting rows and no truncated
Llama validation rows; no rows lost all answer targets. Phi-4's native tokenizer
check found two truncated fitting rows and one validation row, again without empty
targets. The Gadi training job repeats the audit with its actual Unsloth template.

This implements **UT3**, the released three-resource inference path. It does not
claim to reproduce **UTR5**, which selects five snippets using SBERT. The paper
also switched from Llama to Phi-4 for batches 3–4. Evaluating both backbones on all
four batches will provide separate per-model results; do not select a backbone
after inspecting each test batch. No released trained checkpoint was located.

### GPTPrompt fidelity and budget

The paper calls its model “GPT-4o-turbo”; the actual released requests use
**`gpt-4-turbo`**. We retain that model, without substitution. Its returned dated
model ID is archived. The official model page currently marks it deprecated;
account access is tested by the real request.

The nominal one-shot notebook takes `examples[:n+1]`, so `n=1` uses **two** examples.
We preserve the exact two example messages from a released request. The factoid
format prompt is extracted from the Batch-3 Style-2 notebook. No identifiable
Batch-4 request for `GPTPrompt1sStyle2` was found, so this is an explicitly labelled
released-code replication, not a claim to regenerate that exact leaderboard run.
Exact-answer hints used for ideal-answer generation are not used here.

The released comma-splitting answer parser is preserved, including its treatment
of commas inside entity names. Empty responses/parses and output-ceiling responses
remain in the 95-question denominator. No test-guided repairs or reranking are
introduced.

The user authorized a **new US$10 cap**, separate from previous judging budgets.
At the verified list prices, input costs US$10/million tokens and output costs
US$30/million. Each request reserves a conservative UTF-8-byte token bound plus
framing and maximum output cost before sending. Completed calls settle to reported
usage; ambiguous failures retain their reservation. Requests are cached, a process
lock prevents concurrent spending in the run directory, and uncertain requests
are not automatically retried. Reuse the same run directory when resuming; do not
start a second paid experiment under the same authorization.

## Run UniTor on Gadi

The user submits Gadi jobs. First update the repository and activate the existing
8B environment:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

Download pinned checkpoints once if they are not cached. Phi-4 is a separate 14B
download; this command does not train or use paid APIs:

```bash
python3 ../scripts/run_unitor_replication.py download-models \
  --sources data/paper_replication_sources --models llama31 phi4
```

Then submit the two smoke → training → test95 chains:

```bash
bash scripts/submit_unitor_replication.sh llama31 phi4
```

For an initial Llama-only replication, use `llama31` alone in both commands. The
submission script uses the existing portable test95 bundle, or creates it from
`../data/Task13BTest/13B*_golden.json`. It verifies all inputs before submitting.
The prepared code, data, job file and pinned backbone hashes are frozen. Do not
change an already submitted experiment to revive a failed job.

Monitor all stages from the printed record:

```bash
record=$(ls -t outputs/unitor_replication/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Each completed evaluation writes
`SUBMISSION/runs/MODEL/test95/comparison_summary.json`, with overall MRR@5,
coverage, per-batch results, equal-batch mean, parsing diagnostics and limitations.
Only full 95-question generation sets can be scored.

## Local GPTPrompt run

Preparation and the authorized paid run use these commands. The current experiment
directory must be reused to avoid buying duplicate calls:

```powershell
.\.venv\Scripts\python.exe scripts/paper_replication.py `
  --sources Artifacts/paper_replication_sources `
  --bundle Artifacts/official_factoid_test95_v1 `
  --output Artifacts/paper_replication_prepared_v1 --download

.\.venv\Scripts\python.exe scripts/run_gptprompt_replication.py `
  --prepared Artifacts/paper_replication_prepared_v1 `
  --output Artifacts/gptprompt_replication_test95_v1 --max-usd 10
```

Do not rerun preparation into an existing prepared directory. API credentials are
read through the project's existing key-file helper and never written to output.
Raw prompts, responses, usage and the spending ledger remain in ignored Artifacts.

## Interpretation

These baselines differ in model size, training data, context selection, output
budget and API cost. They help reproduce published approaches; they are not a
controlled comparison of only our SFT formulation. Published leaderboard maxima
are reference scores, not guaranteed targets for these reconstructed runs.

References: [UniTor paper](https://ceur-ws.org/Vol-4038/paper_10.pdf),
[GPTPrompt paper](https://ceur-ws.org/Vol-4038/paper_50.pdf),
[official GPT-4-turbo support and pricing](https://developers.openai.com/api/docs/models/gpt-4-turbo).
