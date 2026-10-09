# Ten-sample expansion SFT evaluation

This adds one inference condition for each completed Llama-3.1-8B and Qwen3-8B
expansion SFT adapter. It reuses the original/expansion baseline evaluations
180848513 and 180848515, without retraining or repeating their inference.

Each of the same 160 development questions receives ten independently seeded
expansion requests: temperature 0.8, top-p 0.95, top-k 0, seed 3407 with the same
question/draw seed derivation as original SFT sampling. Each request uses the
trained expansion system prompt and unchanged snippets, native chat template,
6,144-token total limit and 512 output tokens. Qwen thinking remains disabled.
Input preflight rejects evidence overflow rather than truncating snippets.

Each response supplies at most ten candidates. The parser and recovery rules
match greedy expansion; raw responses, rejected candidates, schema diagnostics,
budget clipping, seeds and parse failures remain available per draw. A failed
draw consumes a request and remains in the denominator. The combined pool keeps
the first occurrence of each string, ignoring case and whitespace. Ordering is
draw 1's candidates in response order, then draw 2's new candidates, and so on.
It does not use frequency, a reranker or gold answers for selection.

The analyzer reports MRR@5, first-answer accuracy, coverage@5, coverage@10 and
coverage across the entire unique pool. Full-pool coverage may examine up to
100 candidates before deduplication, whereas original SFT ten-sample inference
supplies at most ten. Compare coverage@10 at a fixed candidate limit and report
full-pool coverage separately. Ten requests is a matched request budget, not a
matched token or runtime budget. Generation tokens, time, requests, per-draw
parse success and coverage within the first 1/5/10 draws accompany the scores.

The analyzer validates the archived baseline, exact adapter hashes/provenance,
questions/evidence, trained prompt, inference limits and seeds. It checks that
saved pool candidates reproduce deduplicated draw/within-response order. All
questions, including complete parse failures, remain in the official BioASQ
evaluation. Paired question bootstrap intervals use 10,000 resamples and seed
3407; secondary intervals are unadjusted. One training seed and a reused
development set still limit the conclusions.

## Submit on Gadi

From the repository's bundle directory:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES
git pull --ff-only
cd gadi_sft_8b_starter
bash scripts/submit_expansion_sampling_8b.sh
```

This submits one smoke job and one dependent full job per model (four jobs).
The full jobs run only after their smoke jobs succeed. To submit one model,
append `llama31` or `qwen3`. Smoke uses four questions with all ten draws; full
uses 160. Both use the existing GPU queue and an eight-hour full-job limit.
The submission record is under `outputs/expansion_sampling_8b/`.

## Score completed jobs

Replace `LLAMA_JOB_ID` and `QWEN_JOB_ID` with the full job numbers in the record:

```bash
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
python ../scripts/analyze_expansion_sampling_8b.py \
  outputs/expansion_sampling_8b/llama31-8b-evaluation-LLAMA_JOB_ID.gadi-pbs \
  outputs/expansion_sampling_8b/qwen3-8b-evaluation-QWEN_JOB_ID.gadi-pbs
```

Each experiment gets `comparison_summary.json` and `comparison_per_question.json`
with all four inference conditions and paired contrasts. Raw generation files
remain unmodified. After copying experiments to another machine, supply one
experiment at a time with `--baseline /path/to/copied/baseline` to override the
original absolute Gadi path; the saved baseline manifest hash is still checked.
