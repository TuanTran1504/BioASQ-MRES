# Expansion DPO annotation pilot

The three completed response banks were automatically annotated with pinned GPT-4.1, producing shared whole-response preference pairs. These are exploratory LLM labels; no independent human audit or held-out DPO answer result is available. Subsequent successful Gadi training is recorded separately in the [training diagnostics](../results/expansion_dpo_training_20261010.json).

| Model | Saved responses | Included after filtering | Excluded |
|---|---:|---:|---:|
| llama31 | 660 | 443 | 217 |
| qwen3 | 660 | 372 | 288 |
| ministral3 | 660 | 459 | 201 |

The same 100 fitting and 32 internal-validation questions produced 1,980 saved responses. Deduplicating identical original/candidate/relation contexts left 3,598 judge units in 242 successful final-pass requests. Across calibration and final annotation, 541 request attempts were recorded. No outer-development questions supplied annotations.

Successful responses reported an estimated API cost of US$6.7987 across both passes, using token usage and cached-input pricing. Failed attempts without usage retain US$2.9450 in conservative reserves; these reserves are not evidence of actual billing. Charging successful input at the uncached rate and retaining those reserves gives total budget accounting of US$9.8785, within the authorized US$10 cap. The final pass paused when prepaid credits were exhausted and resumed from its cache after credits were added. Rates follow the [official GPT-4.1 documentation](https://developers.openai.com/api/docs/models/gpt-4.1).

| Partition | Preference pairs | Represented questions |
|---|---:|---:|
| train | 159 | 83 |
| validation | 48 | 25 |

## Method and exclusions

The judge sees the exact question, full supplied snippets and accepted fitting/validation aliases. Generator identity, draw identifiers, official match flags and preference direction are hidden. Candidate order is deterministically shuffled. Structured decisions assess correctness, support, equivalence and declared relation validity independently. Supported judgments require exact quotes from existing snippet IDs. Raw generation responses are preserved.

Official matching determines C3. Correct nonmatching expressions become C2 and incorrect nonmatching expressions C1. Included responses require high-confidence resolved decisions for every candidate. Invalid quotes, conflicting repeated judgments and official/semantic disagreements exclude the complete response. Wrong relation labels are retained as potential negatives when decisive correctness/support/equivalence errors already establish a contrast; otherwise relation-only disagreements are excluded. Exclusion reasons can overlap.

The existing deterministic preference rule selects a correct supported original with equivalent, error-free variants and Pareto improvements in accepted/correct expression counts or error removal. Text length alone earns no preference; pair count is capped at two per question across all models. This does not train ordering or a reranker.

The final preference archive is preferences-llm-pilot-v4. A fitting-question spot-check exposed the first judge rubric conflating added specificity with scientific incorrectness. The clarified rubric distinguishes scientific correctness from strict equivalence: a supported more specific answer can remain C2 while non-equivalent to its original. Final labels were regenerated in a second paid pass; both passes count toward the budget. Earlier local archives remain preserved. Raw completions and the deterministic preference objective are unchanged.

Candidate-count comparisons: {'chosen_more_candidates': 141, 'chosen_fewer_candidates': 22, 'equal_candidate_counts': 44}. Pair categories (overlapping): {'accepted_expression_gain': 48, 'correct_expression_gain': 162, 'incorrect_removal': 45, 'non_equivalent_removal': 74, 'unsupported_removal': 38}.

## Practical limits

The technical pair validator passed, but automatic labels have not been independently human-validated. Model confidence is not calibrated and literal quote validity cannot establish semantic correctness. GPT-4.1 also verified the expansion SFT teacher targets, so teacher and judge errors can be correlated. Treat subsequent pilot results as exploratory and measure held-out scores before claiming a DPO benefit. LLM judges have documented biases and limitations: [Zheng et al.](https://arxiv.org/abs/2306.05685).

The [aggregate JSON](../results/expansion_dpo_judge_pilot_20261010.json) retains source hashes, costs, class counts, exclusions and pair diagnostics. Raw questions, snippets, aliases, judge responses and annotations remain in ignored local Artifacts directories.

To use locally annotated preferences on Gadi, transfer reviewed-llm.jsonl and rebuild preferences there with the original Gadi bank paths. Local absolute-path manifests are not portable. See [the workflow](EXPANSION_DPO_8B.md).
