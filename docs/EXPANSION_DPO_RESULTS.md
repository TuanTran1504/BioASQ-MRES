# Matched expansion DPO pilot results

On 11 October 2026, the user supplied the official-scoring summary for all three completed Gadi evaluations. DPO produced very small changes beyond expansion SFT. Every paired 95% MRR interval includes zero, so this pilot provides no clear evidence of an improvement. It does not establish that DPO cannot help under another training protocol.

Each backbone received the same 159 fitting and 48 validation preference pairs, with ten optimiser updates and seed 3407. The labels were produced by the automated GPT-4.1 workflow. Each DPO policy uses its own initial expansion SFT checkpoint as its frozen reference. Evaluation uses the same 160 disjoint development questions, full snippets, native expansion prompt and candidate ordering on both sides.

| Backbone | Inference | SFT MRR@5 | DPO MRR@5 | DPO minus SFT | Paired 95% CI |
|---|---|---:|---:|---:|---|
| Llama-3.1-8B | Greedy | 0.501042 | 0.505208 | +0.004167 | [0.000000, 0.010417] |
| Llama-3.1-8B | Ten samples | 0.457396 | 0.455521 | -0.001875 | [-0.010000, 0.005000] |
| Qwen3-8B | Greedy | 0.526042 | 0.526042 | 0.000000 | [0.000000, 0.000000] |
| Qwen3-8B | Ten samples | 0.469479 | 0.474688 | +0.005208 | [-0.001979, 0.014167] |
| Ministral-3-8B | Greedy | 0.534375 | 0.533854 | -0.000521 | [-0.001562, 0.000000] |
| Ministral-3-8B | Ten samples | 0.483021 | 0.485000 | +0.001979 | [-0.008333, 0.017188] |

MRR and differences are copied from six-decimal console output. Last-decimal subtraction discrepancies reflect rounding. The aggregate JSON retains confidence intervals as supplied. Bootstrap resampling uses 10,000 paired question-level resamples with seed 3407; these intervals do not capture training-seed variation.

| Backbone | Inference | SFT coverage@10 | DPO coverage@10 |
|---|---|---:|---:|
| Llama-3.1-8B | Greedy | 0.5500 | 0.5625 |
| Llama-3.1-8B | Ten samples | 0.6312 | 0.6312 |
| Qwen3-8B | Greedy | 0.5687 | 0.5687 |
| Qwen3-8B | Ten samples | 0.6250 | 0.6375 |
| Ministral-3-8B | Greedy | 0.5750 | 0.5750 |
| Ministral-3-8B | Ten samples | 0.6750 | 0.6813 |

Coverage is copied at the four-decimal precision supplied. It is candidate coverage among the first ten unique expressions, not an official ten-answer submission score or coverage anywhere in the unrestricted pool.

## Interpretation

Llama gains slightly with greedy decoding but loses slightly with sampling. Qwen's greedy MRR is unchanged and sampled MRR gains slightly. Ministral loses slightly with greedy decoding and gains slightly with sampling. None of those intervals excludes zero. Ministral expansion SFT with greedy decoding retains the highest observed MRR among these conditions, 0.534375; that cross-backbone ordering is descriptive, not a tested significance claim.

For all three backbones, ten sampled expansion responses provide greater coverage at ten while yielding lower MRR at five than greedy expansion, for both SFT and DPO. Sampling uses temperature 0.8, top-p 0.95 and top-k 0. Pools retain draw order, then within-response order, with conservative deduplication and the first five unique candidates submitted. Generation coverage does not guarantee early ranking success. The difference is consistent with a need to examine selection, but it could also reflect stochastic changes in initial answer quality; a fixed-pool reranker comparison and per-question analysis are needed to distinguish explanations.

The current DPO objective rewards supported equivalent-expression coverage and error removal; it does not directly optimize candidate ordering. A separate selection comparison should hold candidate pools fixed and use only fitting/validation data for ranker training.

## Per-question diagnostics

The user subsequently supplied these counts from the saved per-question scores. Each row contains 160 questions.

| Backbone | Inference | MRR improved | MRR worsened | MRR unchanged |
|---|---|---:|---:|---:|
| Llama-3.1-8B | Greedy | 2 | 0 | 158 |
| Llama-3.1-8B | Ten samples | 2 | 2 | 156 |
| Qwen3-8B | Greedy | 0 | 0 | 160 |
| Qwen3-8B | Ten samples | 5 | 2 | 153 |
| Ministral-3-8B | Greedy | 0 | 1 | 159 |
| Ministral-3-8B | Ten samples | 1 | 3 | 156 |

Greedy MRR changed in only 3 of 480 backbone–question cases; sampled MRR changed in 15 of 480. These are outcome counts: unchanged MRR does not establish identical responses, aliases or evidence support. Ministral's sampled aggregate increases despite more losses than gains because reciprocal-rank changes have different magnitudes.

The following counts identify questions with an accepted candidate in the sampled pool but none among the submitted first five. The full-pool column includes the first-ten cases.

| Backbone | SFT: within first ten | SFT: anywhere in pool | DPO: within first ten | DPO: anywhere in pool |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 13 | 20 | 12 | 19 |
| Qwen3-8B | 11 | 20 | 11 | 21 |
| Ministral-3-8B | 11 | 14 | 11 | 15 |

A reranker restricted to the first ten DPO candidates could potentially recover the 12, 11 and 11 first-ten misses. There are another 7, 10 and 4 full-pool misses respectively, requiring a larger candidate slate. These counts show oracle selection headroom, not achievable gains from a trained selector. They also omit possible rank improvements on questions where the first five already contain an accepted answer.

## Inspect actual responses

From `gadi_sft_8b_starter`, pull the helper and export cases using existing official labels:

```bash
git pull --ff-only
python ../scripts/export_expansion_dpo_cases.py \
  outputs/expansion_dpo_evaluation/llama31-dpo-evaluation-180936011.gadi-pbs \
  outputs/expansion_dpo_evaluation/qwen3-dpo-evaluation-180936013.gadi-pbs \
  outputs/expansion_dpo_evaluation/ministral3-dpo-evaluation-180936015.gadi-pbs
```

This requires no new generation, Java scoring, GPU or paid judge requests. It verifies the saved protocol, candidate ordering and metrics against archived official matches. Each experiment receives a `case_audit` directory containing `cases.jsonl`, `audit_summary.json` and `source_hashes.json`. Existing exports are preserved; a repeated export stops instead of overwriting them.

The export contains every MRR change plus two deterministic examples per policy and selection-miss stratum, merged by question ID. Each case retains all four conditions, raw responses, ordered candidates, official matches, gold aliases and full snippets. The summary additionally counts changed raw completions, which can exceed the number of changed MRR outcomes. These are selected diagnostic cases, not a representative semantic-quality audit. Official rejection alone does not distinguish scientifically correct unaccepted expressions from incorrect candidates.

Download the three `case_audit` directories for local inspection. Compare changed cases for concept corrections, useful aliases, removed invalid variants and order changes before expanding training. Use the selection misses to assess whether evidence permits a selector to distinguish the accepted candidate. Raw exports remain in ignored output directories; the repository stores the helper and aggregate counts.

The 40 exported cases were subsequently downloaded and [inspected locally](EXPANSION_DPO_CASE_INSPECTION.md). All 18 changed-MRR outcomes are present. Fourteen of the 15 sampled changed cases retain the same normalized accepted-answer set in the full pool, with different first accepted ranks. Examples reveal expression/order effects and remaining numerical and answer-type failures; this selected sample cannot estimate population semantic quality.

## Full-pool gains and losses

Ten expansion draws can produce up to 100 candidate slots before conservative deduplication. To test whether DPO generates accepted answers for more questions while preserving previously covered questions, compare `coverage_full_pool` across all 160 questions. Report DPO-only questions (gained), SFT-only questions (lost), both covered and neither covered. Net coverage changes alone hide regressions. Coverage@10 remains a separate fixed candidate-budget comparison; MRR@5 remains submission performance.

For example, Llama's sampled incidence case moves its first accepted answer from rank 7 to 11, and Qwen's sampled polycystic kidney disease case moves from rank 10 to 11. Both lose coverage@10 while retaining full-pool coverage. Selected case exports cannot supply complete full-pool gain/loss counts, because they do not contain every unchanged-MRR question.

The user subsequently ran the full report on Gadi and supplied the following complete 160-question counts. These measure all unique candidates from ten complete expansion draws.

| Backbone | SFT full-pool covered | DPO full-pool covered | Gained | Lost | Net | Paired 95% coverage-difference CI | McNemar p |
|---|---:|---:|---:|---:|---:|---|---:|
| Llama-3.1-8B | 108/160 (67.5%) | 108/160 (67.5%) | 0 | 0 | 0 | [0.00000, 0.00000] | 1.0000 |
| Qwen3-8B | 109/160 (68.125%) | 112/160 (70%) | 3 | 0 | +3 | [0.00000, 0.04375] | 0.2500 |
| Ministral-3-8B | 111/160 (69.375%) | 113/160 (70.625%) | 2 | 0 | +2 | [0.00000, 0.03125] | 0.5000 |

On this sampled development evaluation, Qwen and Ministral show gains without losing any question that SFT covered; Llama retains the same covered-question set. This concerns question-level official acceptance somewhere in the pool. It does not imply that every SFT expression is retained, that all generated expressions are scientifically correct, or that those answers reach the submitted first five. Gains of three and two questions are small, and their intervals include zero with McNemar p-values of 0.25 and 0.5. They are not statistically conclusive evidence of general improvement.

At the first-ten cutoff, the changes include regressions:

| Backbone, ten draws | SFT covered@10 | DPO covered@10 | Gained | Lost | Net |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 101 | 101 | 1 | 1 | 0 |
| Qwen3-8B | 100 | 102 | 3 | 1 | +2 |
| Ministral-3-8B | 108 | 109 | 2 | 1 | +1 |

Since no backbone loses full-pool coverage, each first-ten loss still has an accepted expression somewhere in its DPO pool. The coverage@10 regressions therefore concern expression position at the cutoff. The [saved transition aggregate](../results/expansion_dpo_coverage_transitions_20261011.json) also records all intervals and p-values for these comparisons and the separate MRR@5 results.

For greedy expansion, Llama improves from 88 to 90 full-pool-covered questions (two gains, no losses; CI [0, 0.03125], p = 0.5). Qwen remains at 91 and Ministral at 92, each with zero gains and losses. The small greedy Ministral MRR decline occurs despite unchanged coverage, reflecting rank rather than coverage.

From `gadi_sft_8b_starter`, run:

```bash
git pull --ff-only
python ../scripts/summarize_expansion_dpo_coverage.py \
  outputs/expansion_dpo_evaluation/llama31-dpo-evaluation-180936011.gadi-pbs \
  outputs/expansion_dpo_evaluation/qwen3-dpo-evaluation-180936013.gadi-pbs \
  outputs/expansion_dpo_evaluation/ministral3-dpo-evaluation-180936015.gadi-pbs \
  --output outputs/expansion_dpo_evaluation/coverage-transitions-20261011.json
```

The helper uses the saved official candidate labels, verifies the complete experiment and candidate order, and recomputes aggregate coverage and MRR from all question-level rows. It reports paired question-level bootstrap intervals (10,000 resamples, seed 3407), exact two-sided McNemar p-values for binary coverage, and gained/lost question IDs. Failed parses remain in denominators. P-values are unadjusted exploratory diagnostics, and intervals do not capture training-seed uncertainty. No new generation, Java scoring, paid requests or GPU work are needed. Source files and existing report destinations are preserved; use a fresh output filename when repeating the report.

The combined JSON is small enough to download for local interpretation:

```powershell
Set-Location 'C:\Users\Dustin\Projects\BioASQ-MRES'
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/BioASQ-MRES/gadi_sft_8b_starter/outputs/expansion_dpo_evaluation/coverage-transitions-20261011.json .\Artifacts\coverage-transitions-20261011.json
```

## Provenance and limits

All smoke and full evaluation jobs exited with status zero. Full jobs were 180936011 (Llama), 180936013 (Qwen) and 180936015 (Ministral). Their elapsed times cover all four arms plus setup and are not isolated SFT/DPO generation-time measurements.

The [saved aggregate](../results/expansion_dpo_dev160_20261011.json) records the user-pasted scores, intervals, per-question diagnostic counts, protocol, job identifiers and limits. Full Gadi summaries and complete raw generation/scoring files have not been downloaded or independently inspected locally; selected case exports and their audit summaries have. The full summaries' expected locations are `outputs/expansion_dpo_evaluation/<model>-dpo-evaluation-<job>.gadi-pbs/comparison_summary.json` and `comparison_per_question.json`.

This is a small, single-seed pilot on a development set already used to inform the design. There is no independent human label audit or unseen final-test result. Scientific correctness and evidence support cannot be inferred from exact scores or parser validity. The subsequent console report supplies full-pool transitions; the combined source JSON with gained/lost question IDs has not been downloaded locally. The compact outputs do not supply token/time costs per condition or population candidate-quality diagnostics. Confirmatory claims would additionally need matched optimisation controls, a predeclared validation protocol, seed repeats and unseen evaluation.
