# Base-model greedy expansion: at most five versus at most ten

Evaluate the original pinned Llama-3.1-8B, Qwen3-8B and Ministral-3-8B instruction
backbones without SFT or DPO adapters. Each model receives the same 160 development
questions, all snippets, native chat template and greedy decoding in both arms.
The established expansion prompt allows up to ten equivalent expressions of one
answer concept; the new prompt changes TEN to FIVE and the corresponding padding
instruction. It does not request five unrelated answer concepts.

Both arms are generated afresh, using identical 6,144-token total and 512 output
token limits. The smaller answer budget does not reduce the output token ceiling.
Qwen thinking is disabled and Ministral uses the proven eager execution mode.
Completed historical SFT run metadata supplies exact base snapshot pins and
loader settings; the worker loads the **base snapshot**, never the adapter.

The parser retains at most five or ten unique candidates respectively. Surplus
candidates are recorded as rejected and the original raw responses are retained.
Official matching scores both arms, with parse failures in the denominator.
Reports include candidate-cap violations, MRR@5, coverage@1/5/10, candidate counts,
output tokens and generation time. Coverage@10 equals coverage@5 in the five arm.

Two paired coverage contrasts answer different questions:

- **Same five-candidate budget:** five-prompt coverage@5 minus ten-prompt coverage@5.
  This measures whether asking for fewer candidates changes the submitted five.
- **Total generation coverage:** five-prompt coverage@5 minus ten-prompt coverage@10.
  This measures coverage retained or lost when reducing the full answer pool.

Both report gained/lost question IDs, net change, 10,000-resample paired bootstrap
95% intervals and exact McNemar p-values. Results are exploratory, development-only
and unadjusted across comparisons. Exact acceptance is not a semantic/evidence audit.
There is no model training, paid LLM judging or checkpoint selection in this run.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_base_candidate_budget.sh
```

This submits six jobs: one four-question generation/scoring smoke and a dependent
160-question evaluation for each model. Append one or more model keys to restrict
the submission. Preparation checks all selected historical model provenance,
data hashes, cached base snapshots and the official Java scorer before any qsub.
The scripts have local tests; actual CUDA execution is verified by Gadi smoke.

## Monitor and read results

```bash
record=$(ls -t outputs/base_candidate_budget/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Each full job generates and scores both arms automatically. The stdout prints
coverage gains and losses. Complete reports and per-question scores are saved as:

```text
outputs/base_candidate_budget/MODEL-base-budget-JOB_ID.gadi-pbs/
  comparison_summary.json
  comparison_per_question.json
  max10/TIMESTAMP-max10/
  max5/TIMESTAMP-max5/
```

The inference manifest becoming complete means generation finished. Final job
success and the existence of `comparison_summary.json` confirm scoring completed.
Smoke outputs have a `-smoke` suffix and explicitly marked diagnostic scores.
To retry scoring after resolving a Java problem without repeating GPU generation:

```bash
python ../scripts/analyze_base_candidate_budget.py \
  outputs/base_candidate_budget/MODEL-base-budget-JOB_ID.gadi-pbs
```

Replace the model and job ID with the directory printed by the actual full job.
