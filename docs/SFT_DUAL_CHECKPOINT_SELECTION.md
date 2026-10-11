# Fresh three-model expansion SFT with coverage and loss selection

Train Llama-3.1-8B, Qwen3-8B and Ministral-3-8B afresh from the same pinned
instruction checkpoints used by the completed matched SFT experiment. This
workflow uses expansion SFT. It does not initialize from an existing SFT or DPO
adapter and does not retrain the single-answer formulation.

Each model trains once for three epochs using the existing 1,296 fitting and
144 internal-validation questions, seed 3407 and unchanged SFT settings. All
three epoch checkpoints are retained. From that identical training trajectory,
export two adapters independently:

| Export | Selection criterion | Tie-break |
| --- | --- | --- |
| `best_coverage` | Highest validation full-pool official coverage | Highest coverage@10, then earliest epoch |
| `best_eval_loss` | Lowest teacher-forced validation loss | Earliest epoch |

These are two selection rules, not a weighted combination of loss and coverage.
They may select the same epoch. The historical loss-selected adapters remain
unchanged. New epochs, validation generations and selected adapters use the
separate `outputs/sft_dual_selection_8b` directory.

## Coverage protocol

Each saved epoch generates ten expansion responses for each of the 144
validation questions: temperature 0.8, top-p 0.95, top-k 0, seed 3407 with the
existing question/draw seed hashing. Native chat templates, the trained expansion
system prompt and all supplied snippets are used. Qwen thinking is disabled;
Ministral uses the successful eager execution mode.

Validation inference uses the training-aligned 8,192-token total limit and 512
output tokens, leaving 7,680 tokens for the full prompt. The earlier 6,144 total
limit copied from dev inference failed on longer internal-validation questions.
Prompt preflight checks all 144 questions during smoke; the four longest rendered
prompts receive generations. Overflow fails instead of truncating evidence. A response
can contain up to ten unique candidates; ten responses can provide up to 100
candidate slots before deduplication. The existing parser and draw/within-draw
ordering are reused.

Gold aliases come from the source dataset's hashed audit file, not from teacher
variants. Only question/snippet fields enter the generation prompt; aliases are
used offline by the official BioASQ Java matcher. Report full-pool coverage,
coverage@1/5/10, MRR@5, candidate counts and parse-success rate. Failed parses
remain in the 144-question denominator. A higher full-pool score does not imply
fewer incorrect or unsupported candidates.

Validation MRR is diagnostic and does not decide either winner. No reranker is
used during selection. The outer 160 development questions and official test
questions do not select checkpoints. A later fixed-selection evaluation of both
exports is necessary to establish effects on final answering. Repeated reuse of
internal validation and one sampled-generation seed limit conclusions.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_sft_dual_selection_8b.sh
```

To run only one backbone, append its key (`llama31`, `qwen3` or `ministral3`).
Before any submission the helper checks the teacher dataset, official gold
audit, completed historical runs, exact base snapshot provenance and Java scorer.
It writes immutable preparation files in
`outputs/sft_dual_selection_8b/submission-TIMESTAMP-PID`.

The full submission contains nine GPU jobs, three per model:

1. Smoke: short training, all-question prompt preflight, ten responses on the four longest
   validation questions and official scoring. Two diagnostic smoke exports are
   marked as smoke and are not usable final models.
2. Full training: three fresh epochs with all snapshots retained, after successful
   smoke. The ordinary run-root `adapter` remains the trainer's loss-selected
   export.
3. Validation selection: generate and score each epoch on all 144 questions, then
   save the two independently selected adapters, after successful full training.

Full training requests 12 hours; validation selection requests 24 hours because
it evaluates ten draws per question for each of three checkpoints. Each job uses
one V100, 12 CPUs and 48 GB host memory. GPU runtime has not been verified locally;
the Gadi smoke stage exercises the actual CUDA stack and scorer. No paid judge
requests or new training labels are required.

## Retry smoke coverage after the 11 October context-budget failure

Smoke training jobs 180987381 (Llama), 180987384 (Qwen) and 180987387 (Ministral)
completed training and saved epoch adapters. Validation prompt preflight then
failed on question `65f774a4c4010b4d78000029`, exceeding the old 5,632-token prompt
budget. Their dependent full training and selection jobs did not run.

Reuse their completed smoke training and retry generation/scoring with the corrected
8,192-token limit. Save the old submission record path before launching a new one:

```bash
git pull --ff-only
failed_record=$(ls -t outputs/sft_dual_selection_8b/submission-*/jobs.tsv | head -n 1)
cat "$failed_record"
bash scripts/submit_sft_dual_selection_8b.sh --reuse-smoke-record "$failed_record"
```

Check that this record lists those three failed smoke jobs before running the last
command. The helper validates every saved smoke run and its checkpoint against the
new prepared matrix, input hashes and base pins before any qsub. Reuse is restricted
to smoke training; each full run still starts afresh from the pinned base checkpoint.
Nine replacement jobs are submitted: three generation/scoring smoke retries, three
dependent full training jobs and three dependent checkpoint-selection jobs. Existing
failed outputs are preserved; retries create new selection directories named by
the retry PBS job ID. The smoke record's `training_run` field records reused runs.

## Monitor and inspect

The submission prints its `jobs.tsv` path. Its columns are model, stage, job ID,
dependency and full training directory. To inspect the latest submission:

```bash
record=$(ls -t outputs/sft_dual_selection_8b/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

After selection completes, each full training directory contains
`selection-SELECTION_JOB_ID.gadi-pbs/manifest.json`. Its `checkpoint_scores`
records all three epochs and both metrics; `selected_adapters` records the chosen
epochs, paths and hashes. The adapters are at:

```text
outputs/sft_dual_selection_8b/MODEL-expansion-sft-TRAIN_JOB_ID.gadi-pbs/
  selection-SELECT_JOB_ID.gadi-pbs/
    selected_adapters/best_coverage/
    selected_adapters/best_eval_loss/
```

Epoch validation outputs preserve raw responses, samples/seeds, candidate pools,
official analyses and per-question selection scores. Exports carry native
tokenizer files and the selected checkpoint's actual weights. Existing outputs
are never overwritten. If selection fails, full training remains available;
resubmitting the selection PBS job with the original `MODEL_KEY`,
`SUBMISSION_DIR` and `TRAINING_RUN` creates a new selection directory and does
not repeat training.

These exports have their own selection provenance. Do not substitute them into
the historical matched-evaluation configuration, whose validators pin the old
training runs. Configure a new comparison when these jobs complete.
