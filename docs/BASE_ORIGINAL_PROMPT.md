# Matched base versus original single-answer SFT

This comparison reruns each base checkpoint and its completed original SFT
adapter on the same 160 development questions. Both use the exact trained
system prompt (`prompts/matched_single_answer_v1.txt`), including
`Answer: [BE]expression[EE]`, the same question/snippet serialization, native
template, disabled Qwen thinking, greedy decoding, seed 3407, unmarked snippets
and existing tagged-answer parser. This evaluates inference only, with no
training or paid API requests.

Every input snippet is retained. There is no fixed 6,144/8,192-token input cap:
preparation tokenizes every full prompt for both checkpoint tokenizers, verifies
identical input token sequences, and sets model allocation to the longest actual
prompt plus the original 512 output-token allowance. The backbone's native
context remains a physical constraint. Oversized evidence or changed runtime
token counts cause errors; nothing is silently truncated. Smoke checks all 160
inputs but generates the first four. Full jobs evaluate all 160 including format
failures. Both arms are fresh runs, so the changed allocation is held constant.

The earlier base one-answer JSON experiment remains a separate answer-count
comparison. Its MRR should not be labelled a matched before/after original SFT
result. The new base/original-SFT MRR, paired confidence intervals, coverage gains
and losses, parsing rates and cost are written together. With one answer, MRR
equals exact accuracy. This is exploratory development evidence from one SFT
training seed, not a confirmatory unseen-test result.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_base_original_prompt.sh
```

This submits a smoke and dependent full evaluation for each of Llama-3.1-8B,
Qwen3-8B and Ministral-3-8B. Cached pinned backbones, existing original/expansion
training manifests, matched input data, and Java JDK/scorer are validated before
any submission. No additional dataset transfer is needed. Actual GPU loading
and generation are verified by the smoke jobs on Gadi.

Each new submission stores a hash-checked copy of the inference worker and its
bundled Python helpers under `submission-*/inference_code`. Both base and SFT
use that same saved code. Later Git pulls changing the live inference worker
therefore do not invalidate queued jobs. Inputs, prompts, adapters and evaluation
configuration remain separately checked; changing those still requires a new
submission. The installed Python/CUDA libraries are not copied.

The initial smoke jobs 181035641, 181035646 and 181035649 stopped before
generation because a later update changed the live worker's hash. Their full
jobs never started. Pull this fix and submit a fresh batch with the command
above; do not edit old manifests or their recorded hashes.

The replacement Llama smoke 181046667 generated four responses such as
`Answer: 15` without the required `[BE]`/`[EE]` tags. This was a model-format
outcome, not a model-loading failure. Smoke now warns about zero parsed answers
and proceeds when generation completed, so full evaluation can measure that
outcome. Generation, input integrity and official-scoring failures still fail
the smoke job. Full evaluation retains every question in the denominator.

Strict tagged metrics remain under `conditions`. A separate
`format_recovered_conditions` section applies one identical gold-blind recovery
rule to both base and SFT: accept a single `Answer:` line, optionally removing a
sole opening or closing boundary tag. Keep the entire remaining expression;
never split aliases or trim explanations. Unrecoverable/multiple-tag responses
remain failures. Both views use the official matcher and paired intervals.
Recovery is a post-hoc diagnostic motivated by observed format errors, not a
replacement primary result or evidence of semantic correctness by itself.
Raw generations, strict candidates and their archived hashes are preserved.

Only Llama needs a fresh GPU batch for this format-gate failure:

```bash
git pull --ff-only
bash scripts/submit_base_original_prompt.sh llama31
```

Completed Qwen and Ministral generations can receive the additional diagnostic
by CPU reanalysis without new generation:

```bash
python ../scripts/analyze_base_original_prompt.py \
  outputs/base_original_prompt/qwen3-base-original-181046670.gadi-pbs \
  outputs/base_original_prompt/ministral3-base-original-181046672.gadi-pbs
```

## Monitor and read results

```bash
record=$(ls -t outputs/base_original_prompt/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Full jobs print paired MRR and score differences automatically. Each saves:

```text
outputs/base_original_prompt/MODEL-base-original-JOB_ID.gadi-pbs/
  comparison_summary.json
  comparison_per_question.json
  original_prompt.txt
  base/
  sft/
```

Exit status zero plus a summary confirms generation and official scoring. To
retry scoring without new GPU generation, substitute the actual directory:

```bash
python ../scripts/analyze_base_original_prompt.py \
  outputs/base_original_prompt/MODEL-base-original-JOB_ID.gadi-pbs
```
