# Original and max-five SFT ensembles without a learned reranker

Compare the original single-answer SFT and max-five expansion SFT checkpoints
of Llama-3.1, Qwen3 and Ministral on the same 95 official test factoids.
The models and their native prompts remain frozen. This experiment generates
responses and applies three fixed ordering rules; it performs no training or paid judging.

## Conditions

| Formulation | Decoding per model/question | Ensemble requests/question | Maximum raw candidates/question |
| --- | --- | --- | --- |
| Original SFT | One greedy answer | 3 | 3 |
| Max-five SFT | One greedy expansion | 3 | 15 |
| Original SFT | Ten stochastic answers | 30 | 30 |
| Max-five SFT | Ten stochastic expansions | 30 | 150 |

Actual unique candidate counts can be much lower. Sampling uses temperature 0.8,
top-p 0.95, seed 3407, and independently reproducible question/draw seeds.
Each response has 512 output tokens; max-five parsing retains at most five
unique candidates per response. All snippets are included, with the same
preflight token checks as the completed test experiment. There is no additional
evidence truncation. Original and max-five have different native prompt lengths
and output lengths; equal request counts do not imply equal inference cost.

Each of the four pools receives three selectors, giving **12 ensemble conditions**:

- **Round-robin:** interleave source positions in the fixed Llama, Qwen3, Ministral
  order, skipping duplicate strings already emitted. Duplicate shared answers
  can mean unequal numbers of unique contributions from the sources.
- **Model voting:** rank by the number of distinct models producing the candidate.
  Ten repetitions by one model still count as one model vote. Candidates supported
  by only one model remain eligible.
- **Reciprocal-rank voting:** sum `1/source_rank` across supporting models. Source
  rank is the first unique occurrence in draw order and then within-response order;
  repeated occurrences do not increase the score. For example, ranks 1 and 2 from
  two models contribute `1 + 1/2`. This uses source ordering, not calibrated probabilities.

Exact ties use the same fixed source order and source rank, then the normalised
string. No official labels or semantic judgments participate in ordering.
Deduplication is conservative case/whitespace normalisation; no synonyms, synthetic
variants or gold aliases are inserted into candidate pools. Formulations are pooled
separately: there is no six-checkpoint mixed inference pool.

## Reuse and integrity

Greedy generation is reused from these completed jobs under `outputs/sft_test_evaluation`:

| Model | Greedy test job |
| --- | --- |
| Llama | 181236678.gadi-pbs |
| Qwen3 | 181236680.gadi-pbs |
| Ministral | 181236682.gadi-pbs |

Preparation checks the full archives, native prompts, adapter hashes, pinned
backbones, complete snippets and exactly matching questions/gold. Sources are
the previously frozen loss-selected original and max-five SFT checkpoints.
These use 1296 fitting / 144 validation questions, with dev160 and the 95-question
test panel excluded from fitting. Preparation stops before submission if an
archive is missing or incompatible; it does not silently regenerate different baselines.

GPU smoke uses four longest internal-validation inputs for each formulation,
with ten samples each. It must produce a usable response in both arms. Full
generation uses only the 95 official test questions. Code and inference helpers
are copied and hashed at preparation, protecting queued jobs from later Git pulls.
The full jobs save raw completions even for parse failures, which remain in score denominators.

## Submit and monitor on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_sft_ensemble_test.sh
```

This submits **seven jobs**: three GPU smoke jobs (one hour), three dependent
full GPU jobs (12 hours), and one dependent CPU scoring job (two hours).
Each full GPU job evaluates both formulations for one backbone sequentially.
The CPU job waits for all three full jobs. There is no dependency on max-five
fitting/validation bank generation or logistic-reranker training.
Do not submit the same experiment again while its jobs are queued/running.

Find the latest submission record and check all its jobs:

```bash
record=$(ls -td outputs/sft_ensemble_test/submission-*/jobs.tsv | head -n 1)
cat "$record"
ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $ids
qstat -fx $ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Generation archives are `outputs/sft_ensemble_test/MODEL-sampling10-JOB_ID.gadi-pbs`.
Combined scores are `outputs/sft_ensemble_test/comparison-SCORE_JOB_ID.gadi-pbs`:

- `comparison_summary.json`: all 12 conditions, per-batch scores, paired contrasts and costs;
- `comparison_per_question.json`: all question-level outcomes, including empty pools;
- `rankings.jsonl`: full ordered pools with model/draw provenance;
- `original_greedy_candidates.jsonl`, `max5_greedy_candidates.jsonl`,
  `original_sampling10_candidates.jsonl`, `max5_sampling10_candidates.jsonl`;
- `manifest.json` and `official_scores/`: archive hashes, completion status and official matcher files.

## Interpretation

Final submissions use the top five strings and report official MRR@5, strict
accuracy and coverage@5. Coverage@10 is a **ten-candidate** comparison, not ten
draws. Full-pool coverage uses all distinct strings and reports candidate-generation
potential; it is not an official top-five submission score. The maximum sampling
pool is 30 raw strings for original SFT or 150 for max-five before deduplication.
All three selectors must have identical full-pool coverage within a generation condition.

Report aggregate and per-batch results for 13B1–13B4 (26, 27, 20, 22 factoids).
Paired question-level bootstrap intervals use 10000 resamples, seed 3407, for
max-five minus original and sampling-ten minus greedy under each fixed selector.
These exploratory comparisons use one generation seed and no multiplicity correction.
The batches have already been inspected in this project; they are not certified unseen.

Costs include request counts, input/output tokens, and summed per-response generation
seconds across all models, plus parse failures, output-ceiling counts and candidate-cap
violations. Summed generation seconds are aggregate generation effort, not elapsed
wall time of parallel jobs. Larger pools and higher coverage must be interpreted alongside
these budgets. A later comparison can reuse these archives with frozen logistic selectors.
