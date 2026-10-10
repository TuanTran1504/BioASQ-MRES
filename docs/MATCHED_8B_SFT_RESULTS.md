# Matched Llama, Qwen and Ministral 8B SFT results

Recorded 9 October 2026 for Llama/Qwen and 10 October 2026 for Ministral from
the Gadi scoring output supplied by the user.
The later [ten-sample expansion results](EXPANSION_SAMPLING_8B_RESULTS.md) add a
fourth condition and reuse these baseline scores.
[Complete aggregate reports](../results/matched_8b_sft_dev160_20261010.json) retain
all scores, paired intervals, diagnostics, generation costs and the source-output
SHA256. The earlier Llama/Qwen-only snapshot is retained unchanged. Absolute
Gadi paths are converted to repository-relative paths. Raw
responses and per-question comparison files remain on Gadi; they have not been
retrieved or independently re-scored locally.

The comparison uses 160 development questions and the official BioASQ Java
candidate matcher. Within each backbone, freshly trained single-answer and
expansion SFT use identical 1,296 fitting and 144 internal-validation questions,
evidence and base snapshot. Checkpoint selection uses each formulation's own
internal validation loss. Llama and Qwen use `default` execution; Ministral
uses `eager` for both training formulations and all inference arms. This is separate
from the historical Qwen2.5 SFT comparison.

| Backbone | Condition | MRR@5 | First-answer accuracy | Coverage@5 | Coverage@10 | Generation minutes |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | Original SFT, greedy | 0.48125 | 48.125% (77) | 48.125% (77) | 48.125% (77) | 2.41 |
| Llama-3.1-8B | Original SFT, ten samples | 0.47021 | 37.500% (60) | 61.250% (98) | 65.000% (104) | 23.95 |
| Llama-3.1-8B | Expansion SFT, greedy | **0.50104** | 46.250% (74) | 55.000% (88) | 55.000% (88) | 9.33 |
| Qwen3-8B | Original SFT, greedy | 0.51875 | 51.875% (83) | 51.875% (83) | 51.875% (83) | 2.72 |
| Qwen3-8B | Original SFT, ten samples | 0.49656 | 41.250% (66) | 60.625% (97) | 61.250% (98) | 27.00 |
| Qwen3-8B | Expansion SFT, greedy | **0.52604** | 50.000% (80) | 56.875% (91) | 56.875% (91) | 15.29 |
| Ministral-3-8B | Original SFT, greedy | 0.48125 | 48.125% (77) | 48.125% (77) | 48.125% (77) | 4.72 |
| Ministral-3-8B | Original SFT, ten samples | 0.50927 | 41.250% (66) | 65.000% (104) | 66.250% (106) | 47.76 |
| Ministral-3-8B | Expansion SFT, greedy | **0.53438** | 50.625% (81) | 57.500% (92) | 57.500% (92) | 18.65 |

Parentheses show questions covered out of 160. Generation time is the sum of
recorded generation durations, excluding model loading and preflight. Greedy
conditions use 160 requests; ten-sample conditions use 1,600. Expansion uses a
longer training prompt and emits longer responses, so equal requests do not mean
equal token budgets or runtime. Ten-sample inference uses temperature 0.8,
top-p 0.95 and top-k 0. All arms keep the exact snippet strings, native templates,
6,144-token total limit and 512 output tokens; Qwen thinking is disabled.

All nine conditions have 160/160 question-level parse success and zero snippet
truncation. All three sampling arms parsed all 1,600 attempted draws. Mean unique
candidates per question are 1.00/4.09/3.22 for Llama and 1.00/3.47/4.12 for Qwen
(greedy original / ten-sample original / greedy expansion), and 1.00/3.99/2.96
for Ministral. Candidate relation
types and literal occurrence are diagnostics, not semantic correctness labels.

## Paired MRR differences

Intervals use 10,000 paired question-bootstrap resamples with seed 3407. Positive
differences favour the first method named in each contrast.

| Contrast | Llama difference [95% CI] | Qwen difference [95% CI] | Ministral difference [95% CI] |
|---|---:|---:|---:|
| Expansion minus greedy original | +0.01979 [-0.02188, +0.05938] | +0.00729 [-0.02292, +0.03542] | +0.05313 [+0.00521, +0.10313] |
| Expansion minus ten-sample original | +0.03083 [-0.02333, +0.08375] | +0.02948 [-0.01188, +0.07167] | +0.02510 [-0.02333, +0.07375] |
| Ten-sample minus greedy original | -0.01104 [-0.05865, +0.03781] | -0.02219 [-0.06719, +0.02333] | +0.02802 [-0.02104, +0.07865] |

Every Llama/Qwen MRR interval includes zero. Ministral's expansion-minus-greedy
MRR interval excludes zero before multiplicity correction; its advantage over
ten-sample original remains uncertain. All intervals are unadjusted, so this is
nominal question-level evidence on the evaluated development condition, not a
confirmatory claim across models, comparisons or training seeds.

Expansion increases the number of questions with accepted top-five coverage by
11 for Llama and eight for Qwen relative to greedy original. The respective paired coverage differences
are +0.06875 [0.01250, 0.12500] and +0.05000 [0.00625, 0.09375]. Both point estimates
for first-answer accuracy decrease by three questions. These question-bootstrap
intervals are unadjusted secondary comparisons; no family-wise multiplicity
correction was applied by this analyzer.

For Ministral, expansion adds 15 covered top-five questions relative to greedy
original (+0.09375, interval [0.03750, 0.15000]) and four first-answer matches.
Its first-answer difference interval includes zero. Original ten-sample SFT
covers 106 questions at ten versus expansion's 92, while its MRR point estimate
is lower. Expansion uses 39% of sampling's recorded generation time and 3.95
times greedy original's time. Extra accepted candidates and better top-ranked
answers remain separate outcomes.

Ten-sample original produces the highest top-five and top-ten coverage for all
three backbones. Its MRR point estimates are below greedy original for Llama
and Qwen, and above greedy original for Ministral. With answers
ranked in draw order, extra coverage does not automatically become better ranked
answering. Expansion uses approximately 39% of sampling's generation time for
Llama and 57% for Qwen, but approximately 3.88/5.62 times the generation time of
greedy original. These are observed timing ratios, not general speed
guarantees.

## Interpretation and provenance

Expansion improves coverage over one greedy original answer at the same request
count for all three backbones. Ministral also has nominal paired evidence for an
MRR gain over greedy original; the Llama/Qwen MRR gains remain uncertain.
Candidate ordering and the coverage/cost tradeoff warrant separate analysis;
these aggregates do not identify which aliases or biomedical concepts changed.
There is no expansion-DPO result in this comparison.

All three backbones use one training seed and a previously reused development set.
Question-level confidence intervals do not capture training-seed uncertainty.
These results do not establish unseen-test gains or a statistically supported
ranking between backbones; no between-backbone paired contrast was computed here.
Ministral's expansion MRR point estimate is highest, but its eager backend also
limits cross-model runtime interpretation.

| Backbone | Original training job | Expansion training job | Full evaluation job |
|---|---|---|---|
| Llama-3.1-8B | 180794568 | 180794570 | 180848513 |
| Qwen3-8B | 180794572 | 180794574 | 180848515 |
| Ministral-3-8B | 180878745 | 180872635 | 180897174 |

Reproduction and scoring commands are in the
[matched 8B evaluation workflow](../gadi_sft_8b_starter/docs/MATCHED_8B_EVALUATION.md).
The full Gadi experiments contain `comparison_summary.json` and
`comparison_per_question.json` under
`gadi_sft_8b_starter/outputs/matched_8b_evaluation/<backbone>-8b-evaluation-<job>.gadi-pbs/`.
