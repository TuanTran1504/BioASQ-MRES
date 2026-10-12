# Base one-answer MRR paired with five/ten-answer expansion

Generate one greedy answer for Llama-3.1-8B, Qwen3-8B and Ministral-3-8B using the
exact base snapshots, native templates and all 160 development questions/snippets
from completed base-budget jobs 180988387, 180988389 and 180988391 respectively.
The new prompt retains their grounding instructions and JSON answer schema, then
requests exactly one concise expression marked `original`, without alternatives.
It does not use the historical tagged-answer SFT prompt or load trained adapters.

Use the same 6,144 total / 512 output token limits, greedy decoding, disabled Qwen
thinking and Ministral eager backend. Keeping the output token allowance fixed
isolates prompt/output strategy; shorter actual outputs determine measured cost.
Only the one-answer arm needs new GPU generation. Five/ten raw generations are
validated and reused; the official Java matcher scores all arms, adding fresh
analysis directories without replacing archived source files.

With one submitted answer, MRR@5 equals exact accuracy. Reports include MRR@5,
accuracy@1, coverage@5/10, generation cost, paired five-minus-one and ten-minus-one
MRR confidence intervals, and covered-question gains/losses. Failures remain in
denominators. The one arm is capped at one parsed candidate if the model ignores
the instruction; surplus answers are retained in raw/rejected outputs.
These are exploratory development comparisons, not unseen-test claims. The new
one-answer prompt differs in more instructions than just replacing a numeral.

## Submit

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_base_single_answer.sh
```

This submits six jobs: a four-question generation/scoring smoke and a dependent
160-question evaluation for each model. No training or paid judge requests occur.
Before any submission all chosen baseline archives, backbone pins, inputs, prompt
hashes and the Java scorer are checked. GPU execution is verified by Gadi smoke.

## Monitor

```bash
record=$(ls -t outputs/base_single_answer/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Each full job prints all three arms' scores automatically. Results are saved at:

```text
outputs/base_single_answer/MODEL-base-single-JOB_ID.gadi-pbs/
  comparison_summary.json
  comparison_per_question.json
  max1/TIMESTAMP-max1/
```

An inference manifest marked complete indicates generation finished; final exit
status zero and `comparison_summary.json` confirm scoring succeeded. Smoke scores
use four questions and are explicitly diagnostic. To retry scoring after a Java
issue without new GPU generation, use the actual full-run directory:

```bash
python ../scripts/analyze_base_single_answer.py \
  outputs/base_single_answer/MODEL-base-single-JOB_ID.gadi-pbs
```
