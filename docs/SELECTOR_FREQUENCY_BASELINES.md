# Frequency rules on the exact selector validation pools

This CPU-only analysis compares untrained generator-candidate ranking rules on
the same validation144 question/pool inputs used by the three supervised selector
pilots. It reuses the completed answer-selector data audit, all official aliases
and exact candidate labels; no new scoring API, generation or training is used.

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
python3 ../scripts/score_selector_frequency_baselines.py \
  --audit outputs/answer_selector_data/audit-181824213.gadi-pbs \
  --output "outputs/answer_selector_data/frequency-validation-$(date -u +%Y%m%d-%H%M%S)"
```

The same analysis can run locally on the downloaded audit. It verifies the audit
hashes, fitting/validation/held-out membership, complete request denominators,
every candidate occurrence, per-draw support and source ranks. Empty/failed draws
remain in denominators. Four sampling draws per generator are separate from the
greedy draw. Gold labels affect scoring only, never candidate ordering.

Rules: round robin interleaves the generators' first-unique candidate orders;
model voting counts distinct supporting generators; rank voting sums reciprocal
first-unique source ranks; sample-frequency voting counts distinct supporting
draws (equivalent to summed normalized frequencies with equal draw budgets).
Ties use fixed Llama/Qwen/Ministral source priority and first source rank, as in
the previous heuristics. No parameters are selected on validation.

Unlike the historical normalized round-robin helper, this implementation
deduplicates only the already unique source IDs. It preserves case/whitespace
variants and every exact candidate surface, matching the selector-pilot policy.
These rules operate over full generator pools, not selectors' top-five lists.

## Results: 27 October 2026

| Method | Greedy MRR@5 | Greedy hits | Sampling4 MRR@5 | Sampling4 hits |
|---|---:|---:|---:|---:|
| Round robin | 0.555787 | 93/144 | 0.538657 | 95/144 |
| Model voting | 0.545949 | 88/144 | 0.539583 | 92/144 |
| Reciprocal-rank voting | 0.572569 | 93/144 | 0.580671 | 97/144 |
| Sample-frequency voting | 0.545949 | 88/144 | 0.531366 | 96/144 |

Full-pool coverage stays 96/144 greedy and 107/144 sampling4. On these same pools,
the previously reported logistic baseline was 0.586806/0.599884, and the strongest
individual LLM selector by mean, Ministral, was 0.579861/0.575694. Thus logistic
exceeds every fixed heuristic in observed MRR, while rank voting exceeds all
individual LLM selectors on sampling4. These are descriptive comparisons;
paired uncertainty versus the trained systems requires their saved per-question
predictions. No assertion of statistical superiority is made.

Selector-plus-logistic model voting previously reached 0.587384/0.607523 on these
pools, with 96/102 top-five hits. It is a different fusion rule and retains one
fewer sampling4 question than logistic's 103 hits despite higher observed MRR.

The exported summary pins validation input and supervision hashes. Per-question
scores and source-ID predictions are saved for future paired comparison. No DPO
preferences are created; validation cases must not become training examples.
