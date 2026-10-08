# Original Qwen2.5 SFT: expansion and ten sampled answers

This evaluates the exact original single-answer SFT adapters previously reported
at dev160 MRR 0.40625 (0.5B) and 0.44375 (3B). It does not train new models.

Each checkpoint receives two inference arms on the same 160 questions and all
2,004 snippets, without gold in prompts:

| Arm | Prompt | Decoding | Requests per question |
| --- | --- | --- | --- |
| Original SFT expansion | Existing equivalent-expansion-v1, identical to the completed expansion SFT evaluations | Greedy; up to ten candidates | 1 |
| Original SFT sampling | Original extractive single-answer instructions; `[BE] answer [EE]` | Ten independently seeded draws; temperature 0.8, top-p 0.95, top-k disabled | 10 |

Both arms use a 6,144-token sequence limit, at most 512 new tokens per request,
and full-evidence preflight checks. The sampling prompt presents the same
snippet strings inside `[BS]`/`[ES]` markers. Its question/evidence JSON layout is
new, so this is not a reproduction of the historical single-answer evaluation.
There are no retries or gold-guided choices. Invalid sampled responses consume
their draw and remain saved. The single-answer parser accepts the optional
`Answer:` prefix used in the historical SFT targets, while still requiring exactly
one nonempty `[BE]`/`[EE]` answer and rejecting other surrounding text.
Case-insensitive duplicate answers are removed;
the first five unique candidates in response/draw order form the ranked result.
This ranking is a deterministic baseline, not a learned selector.

The scorer reports official-matcher MRR@5, strict/lenient accuracy, candidate
coverage at 1/5/10, paired bootstrap intervals, per-draw parse success, and
coverage within the first 1/5/10 actual draws. It also records input/output tokens,
request counts and generation time. Ten requests have a larger total token budget
than one expansion response; neither request cost nor total token cost is matched.
Coverage at ten is an offline diagnostic, not an official ten-answer submission.

## Transfer the historical adapters once

The adapters are ignored model files, not GitHub content. Their exact weights,
tokenizer, completion marker and fitting question IDs are packaged separately.
The committed identity hashes pin both historical runs:

- 0.5B: `20260927-143545-67f7a338`, selected `adapter`.
- 3B: `20260927-143545-3fca1c3d`, selected `adapter`.

From Windows PowerShell in the local repository:

```powershell
python scripts/export_original_qwen25_adapters.py
scp -r "C:\Users\Dustin\Projects\BioASQ-MRES\Artifacts\gadi_transfer\original_sft" dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/BioASQ-MRES/gadi_sft_8b_starter/outputs/
```

The destination is `outputs/original_sft/{05b,3b}/adapter/`. If the exact package
already exists on Gadi, the validator can use it directly; uploading it again is
unnecessary. A custom `--adapter` path must contain the same pinned package.
Do not use `adapter_best_eval_loss`, a DPO checkpoint, or an expansion SFT adapter.

## Pull, cache-check, then submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES
git pull --ff-only
cd gadi_sft_8b_starter
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
python scripts/run_original_qwen25_inference.py --model-size 05b --mode validate
python scripts/run_original_qwen25_inference.py --model-size 3b --mode validate
```

The historical adapters refer to `unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit`
and `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`. These differ from the newer
backbone names in expansion training. If either historical backbone is missing,
download it on the login node before submitting (skip this if validation passes):

```bash
python scripts/run_original_qwen25_inference.py --model-size 05b --mode prepare-cache
python scripts/run_original_qwen25_inference.py --model-size 3b --mode prepare-cache
```

GPU jobs use offline loading and never replace a historical backbone reference.
Submit both sizes, or only 3B:

```bash
bash scripts/submit_original_qwen25_evaluation.sh
# Alternatively, submit only 3B:
# bash scripts/submit_original_qwen25_evaluation.sh 3b
```

The helper validates checksums and question separation before any submission.
For each size it submits a four-question smoke job (both inference arms), then
an eight-hour full job conditional on smoke success. Two sizes mean four jobs,
not four independent full evaluations. Submission IDs are saved under
`outputs/original_qwen25/submission-*.tsv`. Avoid resubmitting active jobs.
The full run contains 160 expansion responses and 1,600 sampled responses.
Smoke jobs also require at least one parseable answer in each arm. A completed
process with zero parseable answers fails this check and does not release its
dependent full job.

## Score after completion

Use the printed full-job directory, replacing `<JOBID>` with the real PBS ID:

```bash
python ../scripts/analyze_original_qwen25_inference.py \
  outputs/original_qwen25/qwen25-3b-original-sft-<JOBID> \
  --expansion-sft-experiment outputs/matched_qwen25/qwen25-3b-expansion-dev160-180709562.gadi-pbs
```

For 0.5B use its original full-job directory and expansion SFT experiment
`outputs/matched_qwen25/qwen25-05b-expansion-dev160-180709560.gadi-pbs`.
The optional flag compares original SFT and expansion SFT with identical expansion
inputs/settings. It re-scores the stored expansion SFT outputs without regenerating.
Reports are saved as `comparison_summary.json` and `comparison_per_question.json`.
Python 3.12 plus a Java runtime is required; the bundled official adapter avoids a
JDK requirement.

## Recover the October 8 sampling results without new generation

The original sampling parser rejected the trained `Answer:` prefix. Saved raw
responses confirmed this issue (for example, `Answer: [BE]15[EE]`). The resulting
zero sampling scores are parser failures and must not be interpreted as zero
answer accuracy. Reparse both completed jobs using the corrected parser:

```bash
python ../scripts/reparse_original_qwen25_sampling.py \
  outputs/original_qwen25/qwen25-05b-original-sft-180768374.gadi-pbs
python ../scripts/reparse_original_qwen25_sampling.py \
  outputs/original_qwen25/qwen25-3b-original-sft-180768376.gadi-pbs
python ../scripts/analyze_original_qwen25_inference.py \
  outputs/original_qwen25/qwen25-05b-original-sft-180768374.gadi-pbs-sampling-parser-v2 \
  --expansion-sft-experiment outputs/matched_qwen25/qwen25-05b-expansion-dev160-180709560.gadi-pbs
python ../scripts/analyze_original_qwen25_inference.py \
  outputs/original_qwen25/qwen25-3b-original-sft-180768376.gadi-pbs-sampling-parser-v2 \
  --expansion-sft-experiment outputs/matched_qwen25/qwen25-3b-expansion-dev160-180709562.gadi-pbs
```

Run these after pulling the updated repository, in the existing Python/Java
scoring environment. No GPU job or model loading is required. Each recovery
creates a new sibling directory, preserves all ten raw draws and their seeds,
records the previous parsing errors and source-file hash, and retains measured
generation costs. Existing files are not overwritten; an existing recovery
directory causes an error. Use `--output-dir` for a different new destination.

The expansion arm remains unchanged, including its strict JSON parsing. Its
zero scores need separate raw-output inspection; the sampling repair does not
establish that the model followed the expansion format.

## Interpretation

These are development comparisons, not unseen-test confirmation. The original
checkpoint was selected on this development set. Original SFT had 1,363 fitting
examples from 1,137 distinct questions; expansion SFT used 1,296 fitting questions.
Backbone snapshots may also differ. The same-prompt contrast therefore compares
these trained checkpoints, not the expansion training objective in isolation.
Identical fitting data and backbone revisions would be needed for that claim.

Candidate diagnostic coverage now uses consecutive submission ranks after
deduplication, matching the primary ranked scorer. Old raw positions remain in
the audit. This fixes the earlier 3B base diagnostic coverage@5 disagreement
(64 versus 65), without changing its already reported primary MRR@5.
