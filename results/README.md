# Main saved results

Snapshot: 6 October 2026. [main_results.json](main_results.json) contains the
aggregate scores, evaluation settings, DPO epoch history and source-file SHA-256
hashes. Regenerate it from a workspace containing the original local artifacts:

```bash
python scripts/export_main_results.py
```

The subsequent [matched Llama/Qwen 8B SFT results](../docs/MATCHED_8B_SFT_RESULTS.md)
from 9 October 2026 are saved separately in
[matched_8b_sft_dev160_20261009.json](matched_8b_sft_dev160_20261009.json). They compare
greedy single-answer SFT, ten sampled single answers and greedy expansion SFT,
with paired confidence intervals and measured generation costs. The original
6 October snapshot below retains its historical scope.

The [three-backbone matched SFT snapshot](matched_8b_sft_dev160_20261010.json)
adds the Ministral evaluation from 10 October 2026 while preserving the earlier
Llama/Qwen scores and separate source hashes. Ministral expansion SFT has MRR@5
0.534375, original greedy SFT 0.48125 and original ten-sample SFT 0.509271.
Its expansion-minus-greedy MRR interval excludes zero before multiplicity
correction; these remain development results from one training seed.

The subsequent [ten-sample expansion SFT results](../docs/EXPANSION_SAMPLING_8B_RESULTS.md)
add the fourth inference condition and are saved in
[expansion_sampling_8b_dev160_20261009.json](expansion_sampling_8b_dev160_20261009.json).
This snapshot preserves all four conditions, fixed top-ten versus full-pool
coverage, paired intervals and generation costs; its reused baseline scores
agree with the earlier matched export.

Sanitized per-question model responses and parsed candidates from the Qwen3-8B
base, Gemma-3-27B base and Qwen3-8B expansion-SFT Gadi runs are preserved in
[expansion_generations](expansion_generations/README.md). They omit BioASQ
question text, snippets and gold answers.

## Single-answer factoid SFT and DPO

All evaluations below use the fixed 160-question development set and the official
BioASQ matcher. With one submitted answer, MRR equals exact accuracy.

| Evaluation | MRR | Correct / 160 | LLM semantic accuracy |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B SFT, standalone evaluation | 0.40625 | 65 | 0.74375 |
| Qwen2.5-3B SFT, standalone evaluation | 0.44375 | 71 | 0.81250 |
| Qwen2.5-0.5B, matched pre-DPO evaluation (step 0) | 0.40000 | 64 | unavailable |
| Qwen2.5-0.5B DPO, selected epoch 2 / step 52 | **0.41875** | **67** | unavailable |
| Qwen2.5-3B DPO | unavailable | unavailable | unavailable |

The selected 0.5B DPO checkpoint gains 0.01875 MRR over its own step-zero
baseline: four newly correct answers and one lost answer. Later epochs fall to
0.40625. This run stopped after concept-learning DPO; it is not a completed
three-stage curriculum. The 3B run has configuration and logs but no saved
evaluation history or completed summary. Its saved `running` status does not
establish that a process is currently active.

The SFT training-checkpoint selection scores are 0.40625 for 0.5B and 0.46875 for
3B. The separate standalone 3B evaluation scores 0.44375. Likewise, the 0.5B DPO
step-zero score differs from standalone SFT. These discrepancies are retained
explicitly; their cause has not been established. Use matched evaluation runs
for claims about training gains. Semantic scores are LLM judgments, not audited
human accuracy. These development scores do not establish unseen-test gains.

## Candidate-generation pilots

Coverage at ten means at least one candidate is accepted by the official matcher.
It is an offline pool diagnostic; it is not final-answer MRR or an official
ten-answer submission. All rows contain 160 development questions.

| Method | Covered at ten | Coverage at ten |
| --- | ---: | ---: |
| GPT-4.1 mini, ten independent samples | 78 | 48.750% |
| GPT-4.1 mini, equivalent-expression expansion | 88 | 55.000% |
| GPT-4.1 mini, extractive expansion v1 | 72 | 45.000% |
| GPT-4.1 mini, extractive expansion v2 | 71 | 44.375% |
| Llama-3.1-8B, equivalent-expression expansion | 84 | 52.500% |
| Qwen3-8B, equivalent-expression expansion | 85 | 53.125% |

The equivalent GPT expansion and sampling comparison records 16 questions covered
only by expansion and six only by sampling. The JSON retains request/token counts,
candidate diversity, parsing diagnostics and the distinct extractive comparisons.

## Pooled reranker pilot

The seven-source pool contains 7,362 candidates, including gold-blind formatting
variants, and an accepted candidate for 112/160 questions (70% oracle coverage).
The TF-IDF/metadata logistic reranker uses five-fold question-grouped
cross-validation over the development questions.

| Selection | MRR at five | Coverage at five |
| --- | ---: | ---: |
| Fixed source order | 0.483750 | 58.125% |
| Source-balanced round robin | 0.503125 | 61.250% |
| Consensus heuristic | 0.517083 | 62.500% |
| Cross-validated logistic reranker | **0.527500** | 61.250% |

The final reranker fitted to all 160 questions has no independent evaluation
score. These pool sizes and evaluation protocols differ from the individual
generator pilots, so the scores are not a controlled generator-training contrast.

## Historical list result

The recorded Llama-3.1-8B whole-response DPO system scores **0.5311819 mean F1**
on 83 BioASQ Task 13B list questions, with mean precision 0.5808109 and recall
0.5165600. See the [reproduction instructions](../reproducibility/best_list_system/README.md)
and [original expected result](../reproducibility/best_list_system/expected_result.json).

The export includes aggregate results and portable provenance. Source paths under
`Artifacts/` identify local evidence files; those ignored artifacts, model weights,
raw questions, gold aliases and API caches are not distributed in this result snapshot.
