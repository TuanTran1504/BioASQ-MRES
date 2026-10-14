# Original and max-five SFT on official factoid test data

Evaluate six frozen SFT checkpoints: original single-answer and max-five
expansion SFT for Llama-3.1-8B, Qwen3-8B and Ministral-3-8B. Original models use
their saved validation-loss-selected adapters; max-five models use the frozen
`best_eval_loss` epoch-2 adapters. No test score selects or trains a checkpoint.

The panel is **95 factoid questions** from the four official test batches:
13B1 (26), 13B2 (27), 13B3 (20), 13B4 (22). Fitting, internal-validation and
dev160 IDs must have zero overlap with this panel. All test questions and
snippets are retained. Inputs contain only question IDs, question text and
snippets; gold aliases are stored separately and supplied only to scoring.

Both formulations use their own native trained system prompts, greedy decoding,
one generation per question and 512 output tokens. Full input allocation is
sized to the longest actual complete prompt plus the output allowance, bounded
only by the backbone's native context. Evidence is never silently truncated.
Original submits one tagged answer; max-five submits at most five equivalent
expressions in generated order. This compares the two trained formulations,
not equal candidate budgets or identical prompts.

## Test provenance

These batches are already configured in historical project evaluation pipelines
and are **not certified previously unseen**. Results are official-test scores,
distinct from dev160 scores, but must not be described as a fresh blind test
without an exposure audit. Checkpoint decisions are already frozen on internal
validation. Technical smoke uses four long internal-validation questions for
each formulation; it does not use test questions or gold answers.

## Data transfer, if needed

Raw `data/Task13BTest` files are not tracked in Git. A portable factoid-only
archive was prepared locally at `Artifacts/official-factoid-test95-v1.tar.gz`
(207,739 bytes; SHA256
`76ce941dc0f809e52e3aa9ec7d17c7f15400a3c4f845efa1ea205f05868f4f46`).
It contains gold-free test inputs, separate accepted aliases and source hashes.
If the raw four batch files already exist on Gadi, the workflow can use them.
Otherwise upload the small bundle from Windows PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
scp .\Artifacts\official-factoid-test95-v1.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

On Gadi:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
mkdir -p data/official_factoid_test95_v1
tar -xzf /scratch/nl78/$USER/official-factoid-test95-v1.tar.gz -C data/official_factoid_test95_v1
```

The portable bundle is preferred when present; its hashes, 95 unique IDs,
question/gold correspondence and split disjointness are checked before inference.

## Submit

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_sft_test_evaluation.sh
```

All six source adapters and complete input lengths are checked before submitting
any jobs. Inference code and portable helper modules are frozen in each
submission so later Git pulls do not change queued generation workers.
Submit **six jobs total**: one validation-only smoke and one dependent full
test evaluation per backbone. Each full job evaluates both SFT formulations
sequentially, then scores them. Append model keys to restrict the submission.
No training, sampling-ten banks or paid judging occurs.

```bash
record=$(ls -t outputs/sft_test_evaluation/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

## Results

Reports are saved under
`outputs/sft_test_evaluation/MODEL-sft-test-JOB_ID.gadi-pbs/`:

- `comparison_summary.json`: official MRR@5, strict/lenient accuracy,
  coverage@1/5/10, paired max-five-minus-original intervals, parsing failures,
  output-token ceilings and per-batch metrics.
- `comparison_per_question.json`: paired scores on all 95 questions.
- Original/max-five raw generations, ordered candidates and input/config hashes.

Failures remain in the denominator. With one original answer, MRR equals exact
accuracy. Test gold is merged only after generation for the official Java
matcher; it is never inserted into model prompts or candidate pools.

Scoring alone can be repeated without GPUs:

```bash
python ../scripts/analyze_sft_test_evaluation.py PATH_TO_COMPLETED_TEST_EVALUATION
```

Local tests check the actual 95-question panel, raw snippet preservation,
gold/input separation, training/dev overlap rejection, portable bundle hashes,
failed-answer denominators, candidate ranks, and smoke/full submission chains.
GPU execution is verified by the submitted smoke jobs.
