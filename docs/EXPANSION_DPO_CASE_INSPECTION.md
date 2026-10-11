# Expansion DPO case inspection

On 11 October 2026, 40 exported model–question cases were downloaded from Gadi: 13 Llama, 15 Qwen and 12 Ministral. The exports contain every question with a changed MRR outcome plus small deterministic samples of selection misses. They are diagnostic examples, not a representative sample for estimating semantic error rates.

Local checks replayed the saved raw-response parser and candidate pooling for all 160 case–arm records (40 cases, four arms each). Candidate strings, draw seeds, order, ranks and saved MRR/coverage agree. All 18 reported changed outcomes are present: three greedy and 15 sampled. Official acceptance flags were read from the archived export; Java scoring was not rerun. Full source generation files and official scorer reports remain on Gadi, so their recorded source hashes cannot be independently recomputed from these excerpts.

## What explains the MRR changes?

Fourteen of the 15 sampled MRR changes retain the same accepted-answer set in the complete pool after case/whitespace normalization. Their first accepted ranks change. This is a direct observation about pool order and accepted strings; it does not establish that every generated candidate has the same scientific meaning or quality.

| Model | Case | First accepted rank: SFT → DPO | Observation |
|---|---|---|---|
| Llama, greedy | CADASIL gene | absent → 3 | Both start with `NOTCH3`; DPO adds the accepted string `Notch3 gene`. Snippets 1.1 and 2.1 explicitly mention the gene. |
| Llama, greedy | Costello hormone deficiency | absent → 3 | Both start with `growth hormone`; DPO adds `growth hormone deficiency`. Snippets 8.1 and 8.2 support the deficiency wording. This improves question-aligned expression rather than identifying a different hormone. |
| Qwen, ten samples | Turner chromosome | 7 → 2 | Both start with `X chromosome`; the accepted string `X` appears earlier after DPO. |
| Qwen, ten samples | Decernotinib target | absent → 5 | Both already generate `Janus kinase 3` and `JAK3`; DPO adds the accepted combined string `Janus kinase 3 (JAK3)`. This is the one sampled changed case with a different normalized accepted-answer set. |
| Llama, ten samples | Cep135 location | 5 → 6 | Additional descriptions in the first response push the accepted `Centrosome` beyond the submission limit. The concept is already present in earlier wording. |
| Ministral, ten samples | MMP requirement | 8 → 1 | `Zinc ion` is replaced as the first candidate by accepted `Zinc`; both policies already contain `zinc` somewhere. Snippet 7.1 describes a zinc ion at the active site. |
| Ministral, ten samples | BCG treatment disease | 2 → 5 | Additional broad terms (`carcinoma`, `malignancy`, `neoplasm`) delay accepted `bladder cancer`. |
| Ministral, ten samples | BRAF treatment expression | 2 → 4 | Longer expressions with a performance-status qualifier precede the concise accepted regimen after DPO. The qualifier is present in snippet 1.1; exact rejection does not establish scientific error. |
| Ministral, greedy | RBFOX2/PTBP1 association | 3 → 4 | The accepted string `Colorectal cancer metastasis` moves later; the same accepted answer remains in the pool. |

Ministral's sampled MRR gain is explained by magnitudes: the zinc case gains 1.0 reciprocal rank, while the three losses sum to approximately 0.683333. Dividing the net gain by 160 gives approximately +0.001979. Counting wins and losses alone would miss this.

These examples support an answer-expression and ordering explanation for the changed official scores. They do not show a consistent improvement in biomedical reasoning, semantic precision or evidence support. Textual response changes on other questions were counted by the full Gadi export, but most unchanged-MRR questions were not included for local inspection.

## Why selection misses persist

Examples show both strict answer-form failures and substantive candidate problems:

- A polycystic kidney disease question already has `polycystic kidney disease` and `PKD` near the front. The officially accepted combined form `polycystic kidney disease (PKD)` is later: rank 9 for Llama, rank 7 for Ministral, and rank 10 → 11 for Qwen in the sampled pools.
- In Qwen's mutation-question pool, the accepted `G380R` is rank 6 for both policies. An additional sampled response contributes `FGFR3` as an original answer at pool rank 9. The supplied snippet 1.1 distinguishes the FGFR3 protein from the residue-380 mutation. A gene-only answer therefore does not answer the requested mutation.
- In Llama's DPO incidence pool, draw 1 labels `1:3.5 to 1:5 male live births` as `numerically_equivalent` at pool rank 4. Its original candidate uses denominators 5,000 and 3,500. Dropping those thousands changes the ratios by a factor of 1,000. This is an arithmetic fidelity failure, independently of official acceptance. The accepted `1:5,000 live male births` is at rank 11.
- Longer descriptive answers can consume early slots while concise accepted expressions appear later. This is visible for Cep135, the UC2 tool and AUROC. A selector must consider the requested answer type as well as snippet support.

The current conservative string deduplication preserves these alternate forms as separate candidates. Their accumulation can delay an accepted expression even when the correct concept is available. Semantic clustering could also remove a gold-compatible expression if it keeps only one representative, so any clustering comparison should preserve expression-level candidates or evaluate that tradeoff explicitly.

## Next comparison

The most directly motivated next experiment is a selector comparison on fixed pools. Train only on fitting questions, choose settings on internal validation and hold the selector fixed across SFT and DPO. First compare the same first ten candidates per question with the original ordering and an evidence-aware selector, submitting at most five. A separate full-pool comparison can measure larger-slate headroom; its candidate and computational budgets differ.

Preference-data revisions should explicitly test numeric/unit fidelity, requested entity type and useful concise expressions. Ordering preferences require their own declared objective: the current generator pairs reward coverage and error removal without directly optimizing rank. Pair inspection and semantic/evidence checks are still needed before scaling training. The present cases do not justify assuming that a larger DPO run will improve accuracy.

## Saved evidence

[Derived inspection results](../results/expansion_dpo_case_inspection_20261011.json) preserve export hashes, audit summaries, checks and all 18 changed-rank records. Raw responses, questions, snippets and gold aliases remain in the ignored local `Artifacts/<model>-dpo-evaluation-<job>.gadi-pbs/case_audit` directories. The [aggregate evaluation report](EXPANSION_DPO_RESULTS.md) retains full-development MRR, intervals and diagnostic counts; no population metric was recalculated from the 40 selected cases.
