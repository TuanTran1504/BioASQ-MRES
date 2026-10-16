# Inspect covered answers omitted from the reranker's top five

A selection miss is a question with at least one officially accepted candidate
in the full pool, but no accepted candidate in the first five positions.
An accepted candidate at rank two is a ranking opportunity, not a top-five
selection miss. Empty pools and pools with no accepted candidate are separate
generation/acceptance failures and are not exported as selection misses.

The existing specialised rerankers' reported test95 results imply:

| Pool | Accepted somewhere | Accepted in top five | Selection misses |
|---|---:|---:|---:|
| Original greedy | 56 | 56 | 0 |
| Original ten-sample | 71 | 65 | 6 |
| Max-five greedy | 60 | 58 | 2 |
| Max-five ten-sample | 75 | 65 | 10 |

These are counts derived from the reported completed evaluation. The exporter
checks the saved rankings and recomputes all question-level and aggregate
metrics before identifying the actual questions.

## Export on Gadi

With the project environment active, run from `gadi_sft_8b_starter`:

```bash
git pull --ff-only
python ../scripts/export_sft_reranker_misses.py \
  --evaluation outputs/sft_ensemble_reranker/test95-20261016-021644 \
  --output outputs/sft_ensemble_reranker/selection-misses-v1 \
  --archive /scratch/nl78/$USER/sft-reranker-selection-misses.tar.gz
```

No PBS submission is needed: this reads saved results and scores individual
candidates with existing CPU logistic models. It does not fit models, generate
answers, invoke the Java scorer, call a paid judge, or construct DPO pairs.
Existing export directories and archives are not overwritten.

The baseline is the completed evaluation **with relation labels**. The separate
no-relation-label ablation is not silently substituted or pooled into this
export. All three frozen logistic selectors on all four pools are inspected;
all 28 saved methods are retained for comparison within each selected question.

## Contents

- `summary.json`: counts and question IDs for all 12 logistic conditions;
- `report.md`: count table and the prioritised expansion ten-sample questions;
- `cases.jsonl`: every missed question, merged across conditions; full evidence,
  gold aliases, top-five candidates, all accepted candidates with their ranks,
  scores, relation metadata, source ranks and draw support;
- `source_generations.jsonl`: original saved greedy and sampled responses from
  both SFT formulations and all three models for those selected questions;
- `manifest.json`: input/output hashes and inference-only provenance.

For each missed logistic condition, exact score explanations are exported for
the top five and the highest-ranked accepted answer. They include word-feature,
character-feature and metadata totals; the strongest individual contributions;
and the score gap between rank five and the best accepted answer. The frozen
weights must reproduce every explained score. These explain the model's
arithmetic, not scientific truth or a causal effect of changing a feature.

Source hashes, generator identity, fitting/validation/test disjointness and
runtime compatibility are checked using the existing frozen-evaluation loader.
Rankings must preserve every cached candidate and its official label and must
reproduce the saved 95-question metrics, including empty questions.

## Download to this machine

In local PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
New-Item -ItemType Directory -Force .\Artifacts\sft_reranker_selection_misses | Out-Null
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/sft-reranker-selection-misses.tar.gz .\Artifacts\sft-reranker-selection-misses.tar.gz
if ($LASTEXITCODE -eq 0) {
    tar -xzf .\Artifacts\sft-reranker-selection-misses.tar.gz -C .\Artifacts\sft_reranker_selection_misses
}
```

## Review questions

Start with the ten max-five ten-sample misses under the max-five-trained
reranker, then compare the six original ten-sample misses. Record:

1. Are the top-five answers semantically correct alternatives that the official
   gold aliases do not accept? Official nonacceptance is not a wrong-answer label.
2. Is the accepted candidate actually supported by the supplied snippets, and
   does it have the right entity, scope, numerical value and qualifiers?
3. Is the accepted form rare or split across multiple aliases? Which models and
   draws supplied it? Do not merge expressions as synonyms without review.
4. Does frequency, source position, relation metadata, literal evidence overlap
   or lexical similarity contribute to the score gap?
5. Do frequency voting or another frozen reranker recover the same question?

These are outcome-selected, previously inspected test cases. Use them to diagnose
failure modes; do not insert them into fitting data or DPO preferences. Future
training interventions should use fitting questions and validation selection.
