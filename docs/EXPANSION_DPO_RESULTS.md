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

The next diagnostic should inspect changed question outcomes, original-concept correctness, unsupported/non-equivalent variants and diversity before enlarging the DPO experiment. A separate selection comparison should hold candidate pools fixed and use only fitting/validation data for ranker training. The current DPO objective rewards supported equivalent-expression coverage and error removal; it does not directly optimize candidate ordering.

## Provenance and limits

All smoke and full evaluation jobs exited with status zero. Full jobs were 180936011 (Llama), 180936013 (Qwen) and 180936015 (Ministral). Their elapsed times cover all four arms plus setup and are not isolated SFT/DPO generation-time measurements.

The [saved aggregate](../results/expansion_dpo_dev160_20261011.json) records the user-pasted scores, intervals, protocol, job identifiers and limits. Full Gadi summaries and raw per-question artifacts have not been downloaded or independently inspected locally. Their expected locations are `outputs/expansion_dpo_evaluation/<model>-dpo-evaluation-<job>.gadi-pbs/comparison_summary.json` and `comparison_per_question.json`.

This is a small, single-seed pilot on a development set already used to inform the design. There is no independent human label audit or unseen final-test result. Scientific correctness and evidence support cannot be inferred from exact scores or parser validity. The compact output does not supply full-pool coverage, token/time costs per condition or candidate-quality diagnostics. Confirmatory claims would additionally need matched optimisation controls, a predeclared validation protocol, seed repeats and unseen evaluation.
