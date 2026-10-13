# Max-five SFT diagnosis before DPO

Inspected the three completed dev160 archives downloaded on 13 October 2026.
Primary comparisons use the epoch-2 loss-selected adapters, greedy generation,
the trained max-five prompt, original candidate order, and official BioASQ
acceptance. Epoch-1 coverage-selected and ten-sample outputs are secondary.
This is exploratory diagnosis on a repeatedly inspected development set.

## Loss-selected greedy outputs

| Model | MRR@5 | Accepted first | Accepted at ranks 2–5 | No accepted answer | Parse failures | Greedy misses recovered by ten-sample full pool |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.502604 | 75 | 13 | 72 | 0 | 24 |
| Qwen3-8B | 0.516146 | 78 | 11 | 71 | 0 | 22 |
| Ministral-3-8B | 0.510104 | 75 | 16 | 69 | 0 | 25 |

All category counts sum to 160. A greedy max-five response cannot hide an
accepted candidate after rank five. Its coverage@5 is therefore its full-pool
coverage: 88/160, 89/160 and 91/160 respectively. Moving existing accepted
candidates to rank one would give oracle MRR values of 0.55000, 0.55625 and
0.56875, without generating any new candidates. These are bounds using gold,
not achievable reranker results.

Ten-sample generation recovers accepted expressions for 24, 22 and 25 questions
missed by greedy generation. These are paired recovery counts, not net coverage
gains: sampling can also lose questions covered by greedy generation. Each
sample requests at most five expressions; its pooled candidate budget is larger.
For loss-selected sampling, accepted candidates lie beyond the submitted first
five in 17, 15 and 18 questions respectively.

## Examples that distinguish preference objectives

These are selected diagnostic examples, not a representative semantic audit.
Their questions must not become training examples from dev160.

1. **Answer ordering / accepted form:** For the gefitinib target question
   (`53188480b166e2b806000018`), all three models generate
   `epidermal growth factor receptor`, `EGFR`, then
   `epidermal growth factor receptor (EGFR)`. Only the third is accepted by the
   archived matcher. This is not evidence of choosing the wrong biological
   target. A same-candidate-set ordering preference can isolate the MRR objective.
2. **Scientifically supported output rejected by exact scoring:** For hepcidin
   disulfide bridges (`54f2210164850a5854000001`), all models generate `four` or
   `4`. Multiple supplied snippets explicitly state four bridges. The recorded
   gold alias is the complete sentence
   `Hepcidin contains eight cysteine residues that form four disulfide bridges`.
   All short outputs receive no official match. These should not be labelled C1.
3. **Expansion changes specificity:** For transcription elongation
   (`53372ba3d6d3ac6a34000058`), a supplied snippet explicitly links H3K36
   trimethylation to elongation. Llama leads with `H3K36me3`. Qwen puts accepted
   `H3K36 trimethylation` fourth and also emits dimethylation. Ministral lists
   `H3K36me1`, `H3K36me2`, and accepted `H3K36me3` fifth. These methylation states
   are not interchangeable expressions. This is a useful candidate-validity
   preference pattern, pending semantic/evidence labels for training examples.
4. **An answer mixed with a different biochemical process:** For UBC9
   (`58a5b1fe60087bc10a000024`), supplied snippets identify the sole E2 enzyme
   required for sumoylation. Llama includes accepted `SUMO-conjugating enzyme`
   but also `E2 ubiquitin-conjugating enzyme`. The latter should receive explicit
   semantic review rather than be assumed an equivalent expression because
   it occurs in an otherwise accepted response.

An official nonmatch can be a correct synonym, an incomplete answer, an incorrect
concept, or a scope-changing variant. The archive's relation-type strings are
model predictions, not independently validated labels. No complete C1/C2 counts
or harmful-expansion rates are claimed. No paid judge calls were made.

## Recommended first max-five DPO experiment

Generate several complete max-five responses from each model's own frozen
loss-selected SFT checkpoint on the 1,296 fitting questions. Use the 144
internal-validation questions for validation only; do not derive training pairs
from these dev160 examples. Keep model-specific banks and the corresponding
frozen SFT reference policy.

Start with an independently labelled **concept and candidate-validity** arm:

- Prefer a supported accepted answer over genuinely incorrect responses.
- Prefer a clean supported response over one that adds incorrect or
  meaning-changing variants, retaining accepted coverage. The shorter response
  may win; length itself is not a reward.
- Exclude unresolved comparisons; official rejection alone does not establish C1.
- Preserve C2 answers as semantically correct. Keep exact-form C3>C2 preferences
  out of this first arm, consistent with the current preference experiment.

Keep an **ordering-only** arm separate if studied later: same validated candidates,
accepted expression earlier. It tests a different objective and can be compared
with supervised reranking of identical pools. Its official ranking signal often
reflects answer form rather than better biomedical reasoning.

Select checkpoints by greedy validation MRR@5 after each epoch and also retain
the best validation-loss checkpoint. Report dev160 paired MRR@5, coverage@5 gains
and losses, supported/incorrect candidate rates after semantic judging, and
cost. Reserve an unseen test set for confirmatory conclusions.

## Saved artifacts and verification

Local `Artifacts/max5_sft_audit/diagnosis/summary.json` contains all four conditions
per model. `cases.jsonl` exports all 960 greedy question-condition records,
including candidate lists, accepted matches, snippets and raw responses.
Semantic labels remain explicitly unreviewed.

The existing evaluation validator passed all three archives: four-arm membership,
question/evidence equality, hashes, raw-response parsing, candidate pools, sample
seeds, candidate limits and untruncated snippets. The diagnostic exporter also
recomputed per-question MRR and full-pool coverage from archived official matches.
This checks consistency of saved scoring, without rerunning Java scoring or GPU
generation. It does not validate missing remote adapter weight files.

```bash
python scripts/audit_max5_sft_outputs.py Artifacts/max5_sft_audit \
  --output Artifacts/max5_sft_audit/diagnosis
```
