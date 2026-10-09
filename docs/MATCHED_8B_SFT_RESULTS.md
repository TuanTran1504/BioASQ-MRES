# Matched Llama and Qwen 8B SFT results

Recorded 9 October 2026 from the Gadi scoring output supplied by the user.
The later [ten-sample expansion results](EXPANSION_SAMPLING_8B_RESULTS.md) add a
fourth condition and reuse these baseline scores.
[Complete aggregate reports](../results/matched_8b_sft_dev160_20261009.json) retain
all scores, paired intervals, diagnostics, generation costs and the source-output
SHA256. Absolute Gadi paths are converted to repository-relative paths. Raw
responses and per-question comparison files remain on Gadi; they have not been
retrieved or independently re-scored locally.

The comparison uses 160 development questions and the official BioASQ Java
candidate matcher. Within each backbone, freshly trained single-answer and
expansion SFT use identical 1,296 fitting and 144 internal-validation questions,
evidence and base snapshot. Checkpoint selection uses each formulation's own
internal validation loss. Both execution modes were `default`. This is separate
from the historical Qwen2.5 SFT comparison.

| Backbone | Condition | MRR@5 | First-answer accuracy | Coverage@5 | Coverage@10 | Generation minutes |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | Original SFT, greedy | 0.48125 | 48.125% (77) | 48.125% (77) | 48.125% (77) | 2.41 |
| Llama-3.1-8B | Original SFT, ten samples | 0.47021 | 37.500% (60) | 61.250% (98) | 65.000% (104) | 23.95 |
| Llama-3.1-8B | Expansion SFT, greedy | **0.50104** | 46.250% (74) | 55.000% (88) | 55.000% (88) | 9.33 |
| Qwen3-8B | Original SFT, greedy | 0.51875 | 51.875% (83) | 51.875% (83) | 51.875% (83) | 2.72 |
| Qwen3-8B | Original SFT, ten samples | 0.49656 | 41.250% (66) | 60.625% (97) | 61.250% (98) | 27.00 |
| Qwen3-8B | Expansion SFT, greedy | **0.52604** | 50.000% (80) | 56.875% (91) | 56.875% (91) | 15.29 |

Parentheses show questions covered out of 160. Generation time is the sum of
recorded generation durations, excluding model loading and preflight. Greedy
conditions use 160 requests; ten-sample conditions use 1,600. Expansion uses a
longer training prompt and emits longer responses, so equal requests do not mean
equal token budgets or runtime. Ten-sample inference uses temperature 0.8,
top-p 0.95 and top-k 0. All arms keep the exact snippet strings, native templates,
6,144-token total limit and 512 output tokens; Qwen thinking is disabled.

All six conditions have 160/160 question-level parse success and zero snippet
truncation. Both sampling arms parsed all 1,600 attempted draws. Mean unique
candidates per question are 1.00/4.09/3.22 for Llama and 1.00/3.47/4.12 for Qwen
(greedy original / ten-sample original / greedy expansion). Candidate relation
types and literal occurrence are diagnostics, not semantic correctness labels.

## Paired MRR differences

Intervals use 10,000 paired question-bootstrap resamples with seed 3407. Positive
differences favour the first method named in each contrast.

| Contrast | Llama difference [95% CI] | Qwen difference [95% CI] |
|---|---:|---:|
| Expansion minus greedy original | +0.01979 [-0.02188, +0.05938] | +0.00729 [-0.02292, +0.03542] |
| Expansion minus ten-sample original | +0.03083 [-0.02333, +0.08375] | +0.02948 [-0.01188, +0.07167] |
| Ten-sample minus greedy original | -0.01104 [-0.05865, +0.03781] | -0.02219 [-0.06719, +0.02333] |

Every MRR interval includes zero. Expansion's higher point estimates therefore
do not establish an MRR improvement or equivalence between methods.

Expansion increases the number of questions with accepted top-five coverage by
11 for Llama and eight for Qwen relative to greedy original. The respective paired coverage differences
are +0.06875 [0.01250, 0.12500] and +0.05000 [0.00625, 0.09375]. Both point estimates
for first-answer accuracy decrease by three questions. These question-bootstrap
intervals are unadjusted secondary comparisons; no family-wise multiplicity
correction was applied by this analyzer.

Ten-sample original produces the highest top-five and top-ten coverage for both
backbones, while its MRR point estimates are below greedy original. With answers
ranked in draw order, extra coverage does not automatically become better ranked
answering. Expansion uses approximately 39% of sampling's generation time for
Llama and 57% for Qwen, but approximately 3.88/5.62 times the generation time of
greedy original. These are observed timing ratios, not general speed
guarantees.

## Interpretation and provenance

The strongest observed expansion result is improved coverage over one greedy
original answer at the same request count. An MRR benefit remains uncertain.
Candidate ordering and the coverage/cost tradeoff warrant separate analysis;
these aggregates do not identify which aliases or biomedical concepts changed.
There is no expansion-DPO result in this comparison.

Both backbones use one training seed and a previously reused development set.
Question-level confidence intervals do not capture training-seed uncertainty.
These results do not establish unseen-test gains or statistically establish that
Qwen is better than Llama; no between-backbone paired contrast was computed here.

| Backbone | Original training job | Expansion training job | Full evaluation job |
|---|---|---|---|
| Llama-3.1-8B | 180794568 | 180794570 | 180848513 |
| Qwen3-8B | 180794572 | 180794574 | 180848515 |

Reproduction and scoring commands are in the
[matched 8B evaluation workflow](../gadi_sft_8b_starter/docs/MATCHED_8B_EVALUATION.md).
The full Gadi experiments contain `comparison_summary.json` and
`comparison_per_question.json` under
`gadi_sft_8b_starter/outputs/matched_8b_evaluation/<backbone>-8b-evaluation-<job>.gadi-pbs/`.
