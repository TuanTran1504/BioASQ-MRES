# Ten-sample expansion SFT results

Recorded from user-supplied Gadi scoring output on 9 October 2026. The
[complete aggregate snapshot](../results/expansion_sampling_8b_dev160_20261009.json)
retains all four conditions, paired intervals, diagnostics, costs and the source
SHA256. Raw generation and per-question files remain on Gadi; they have not been
retrieved or independently re-scored locally. All six reused baseline arms agree
with the [previous result snapshot](MATCHED_8B_SFT_RESULTS.md).

The comparison uses the same 160 development questions and completed matched
Llama-3.1-8B/Qwen3-8B adapters. Both training execution modes are `default`.
Sampling uses ten independently seeded requests at temperature 0.8, top-p 0.95
and top-k 0. It retains each formulation's trained system prompt, identical
snippet strings, native templates, a 6,144-token total limit and 512 output tokens
per request. Qwen thinking is disabled. There is no DPO or learned reranker here.

## Quality and coverage

| Backbone | Condition | MRR@5 | First-answer accuracy | Coverage@5 | Coverage@10 | Full-pool coverage |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | Original SFT, greedy | 0.48125 | 48.125% | 48.125% | 48.125% | 48.125% |
| Llama-3.1-8B | Original SFT, ten samples | 0.47021 | 37.500% | 61.250% | **65.000%** | 65.000% |
| Llama-3.1-8B | Expansion SFT, greedy | **0.50104** | 46.250% | 55.000% | 55.000% | 55.000% |
| Llama-3.1-8B | Expansion SFT, ten samples | 0.45740 | 40.000% | 55.000% | 63.125% | **67.500%** |
| Qwen3-8B | Original SFT, greedy | 0.51875 | 51.875% | 51.875% | 51.875% | 51.875% |
| Qwen3-8B | Original SFT, ten samples | 0.49656 | 41.250% | **60.625%** | 61.250% | 61.250% |
| Qwen3-8B | Expansion SFT, greedy | **0.52604** | 50.000% | 56.875% | 56.875% | 56.875% |
| Qwen3-8B | Expansion SFT, ten samples | 0.46948 | 42.500% | 55.625% | **62.500%** | **68.125%** |

Ten-sample expansion covers 108/160 questions in Llama's full pool and 109/160 in
Qwen's. At the fixed ten-candidate limit, these fall to 101 and 100 respectively;
seven and nine questions have an accepted answer only beyond rank ten. Original
SFT sampling covers 104 and 98 questions. Its full pool cannot exceed ten strings,
so full-pool coverage equals coverage@10.

Relative to greedy expansion, sampling increases top-ten coverage by a net 13
questions for Llama and nine for Qwen, and full-pool coverage by 20 and 18. Its
first-answer accuracy decreases by ten and twelve questions. The observed MRR
also decreases. More pool coverage has not translated into better default-order
ranked answering.

Pools are ordered by draw, then within-response order, with case/whitespace
duplicates removed. There is no frequency sorting. A draw's variants can occupy
the first five slots before another draw's original answer is considered. The
first draw is stochastic; these pools do not explicitly include the greedy
response. Consequently this experiment changes both the available candidates
and the first answer, and does not isolate ordering as the sole cause of MRR
changes. Selection on frozen pools would test that separately.

## Paired uncertainty

Differences are ten-sample expansion minus the named comparator. Intervals use
10,000 paired question-bootstrap resamples and seed 3407.

| Backbone / metric | Minus original ten samples [95% CI] | Minus greedy expansion [95% CI] |
|---|---:|---:|
| Llama MRR@5 | -0.01281 [-0.05531, +0.02896] | -0.04365 [-0.09063, +0.00250] |
| Llama coverage@10 | -0.01875 [-0.07500, +0.03750] | +0.08125 [+0.03125, +0.13750] |
| Llama full-pool coverage | +0.02500 [-0.02500, +0.07500] | +0.12500 [+0.07500, +0.18125] |
| Qwen MRR@5 | -0.02708 [-0.08104, +0.02490] | -0.05656 [-0.10240, -0.01323] |
| Qwen coverage@10 | +0.01250 [-0.03125, +0.05625] | +0.05625 [+0.00625, +0.10625] |
| Qwen full-pool coverage | +0.06875 [+0.02500, +0.11875] | +0.11250 [+0.06250, +0.16250] |

Fixed top-ten coverage differences between the two sampling strategies span
zero for both backbones. Their MRR differences also span zero. Qwen expansion
sampling's MRR decline relative to greedy expansion excludes zero in the reported
unadjusted interval; Llama's does not. Qwen's full-pool increase over original
sampling also excludes zero, but compares different candidate budgets. These
secondary intervals have no family-wise multiplicity correction and should be
interpreted with that qualification.

## Costs and diagnostics

| Backbone | Sampling condition | Mean unique pool size | Output tokens | Generation minutes |
|---|---|---:|---:|---:|
| Llama-3.1-8B | Original SFT | 4.09 | 24,179 | 23.95 |
| Llama-3.1-8B | Expansion SFT | 13.64 | 105,788 | 84.13 |
| Qwen3-8B | Original SFT | 3.47 | 23,467 | 27.00 |
| Qwen3-8B | Expansion SFT | 17.74 | 135,701 | 128.68 |

Both sampling strategies use 1,600 requests. Expansion sampling uses about
3.51 times the measured generation time of original sampling for Llama and 4.77
times for Qwen, along with longer input/output sequences. Time excludes model
loading and preflight. Equal requests do not mean equal tokens, runtime or
candidate budgets. A full expansion pool can contain up to 100 candidates before
deduplication; fixed coverage@10 and unrestricted coverage remain separate.

All eight conditions parse at question level and have zero snippet truncation.
All 1,600 expansion draws per backbone supply usable candidates. Strict schema
compliance is 1,596/1,600 for Llama and 1,593/1,600 for Qwen; Qwen recovers two
incomplete top-level responses. Three draws per model hit the candidate limit.
The 41/293 rejected candidates include parser-level issues such as duplicates
or budget violations; those counts are not labels of biomedical incorrectness.
Likewise, literal snippet occurrence and reported relation types do not establish
semantic equivalence or evidence support.

One training seed, one sampled inference realization and a reused development
set limit the conclusions. Question-bootstrap intervals do not cover retraining
or resampling variability. This result supports investigating candidate selection
and cost tradeoffs; it does not establish a DPO benefit or unseen-test improvement.

Full sampling jobs are Llama `180878412` and Qwen `180878414`. They reuse baseline
jobs `180848513` and `180848515`. Commands and validation rules are in the
[sampling evaluation workflow](../gadi_sft_8b_starter/docs/EXPANSION_SAMPLING_8B.md).
