# Max-five sampled ensemble: full-pool absence audit

The existing 95-question test comparison has 75 questions covered somewhere in
the max-five ten-sample three-model pool. This audit selects **all 20 remaining
questions**, where no parsed, capped pool candidate has a positive cached
official-acceptance label. The selection is independent of reranker order.

The matching max-five logistic reranker covers 65 questions in its top five.
That gives three disjoint groups:

| Group | Questions |
|---|---:|
| Accepted answer in the top five | 65 |
| Accepted answer in the pool but outside the top five | 10 |
| Accepted answer absent from the entire pool | 20 |

The earlier `sft-reranker-selection-misses` download contains the middle group
and other selectors' ranking failures. Its 25 exported questions have **zero**
overlap with the last group. It cannot support an inspection of these 20 cases.

## Export on Gadi

From the project environment:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate

python3 ../scripts/export_sft_ensemble_pool_absences.py \
  --comparison outputs/sft_ensemble_test/comparison-181287698.gadi-pbs \
  --output outputs/sft_ensemble_test/pool-absences-test95-v1 \
  --archive /scratch/nl78/$USER/max5-ensemble-pool-absences-test95-v1.tar.gz
```

This is a CPU export, not a new PBS job. It reuses the existing raw generations
and labels, validates reconstructed pools against all 12 original condition
scores, and requires exactly 20 absences. It performs no generation, fitting,
new official scoring or paid judging. Existing output directories/archives are
preserved; use new names for a second export.

The archive includes:

- `cases.jsonl`: all 20 complete questions, batches, accepted aliases, evidence,
  all four original/max-five greedy/sampling pools and complete candidate
  metadata. Coverage by the original prompt is retained for comparison.
- `source_generations.jsonl`: 240 question-level saved generation records:
  20 questions × three models × two prompts × two generation modes. Each sampled
  record contains its ten original draws, including raw text and parse errors.
- `summary.json`: complete membership, per-batch counts and comparator coverage.
- `report.md`: a readable question/alias overview.
- `manifest.json`: input and output hashes, code identity and provenance.

Any frequency top-five lists in this export are explicitly labeled fixed
baselines; they are not presented as the logistic reranker's output. Full-pool
absence itself does not require loading or rerunning a reranker.

## Download on the local machine

```powershell
cd C:\Users\Dustin\Projects\BioASQ-MRES
New-Item -ItemType Directory -Force .\Artifacts\max5_ensemble_pool_absences_test95_v1 | Out-Null
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/max5-ensemble-pool-absences-test95-v1.tar.gz .\Artifacts\
if ($LASTEXITCODE -eq 0) {
    tar -xzf .\Artifacts\max5-ensemble-pool-absences-test95-v1.tar.gz -C .\Artifacts\max5_ensemble_pool_absences_test95_v1
}
```

## Review after transfer

All 20 are official pool-absence failures; keep that status even if an unaccepted
expression later proves semantically correct. For each observed case, inspect:

1. The full question and every accepted alias.
2. Each model's initial answers across ten draws and every pooled expansion.
3. Whether a supported equivalent expression is present but officially unaccepted.
4. Whether the generated concept is wrong or insufficient for the actual question.
5. Whether supplied evidence establishes an accepted answer, is insufficient,
   conflicts, or the reference has a wording/scope issue.
6. Whether raw text contains an expression that parsing or the five-candidate
   limit excluded, and whether the original-prompt pool covers the question.

The export does not assign these semantic diagnoses. Counts and concrete examples
must come from the downloaded real cases, not synthetic illustrations. These
historically inspected test cases remain diagnostic and are excluded from future
preference construction and fitting.

## Completed inspection: 19 October 2026

The real exported files were downloaded and all hashes verified. All four pools
for all 20 questions reconstructed exactly from the 240 saved generation rows;
sample seeds, parsed answers and evidence counts were independently checked.
The 600 target max-five sampled responses contained no empty parses and no
output-token-ceiling cases. Three responses had recoverable schema issues; the
one response exceeding the candidate cap discarded Fontan wording that did not
contain an accepted alias.

Primary, source-evidence-based manual diagnoses are:

| Diagnosis | Questions |
|---|---:|
| Supported answer form present, but exact reference wording/format absent | 16 |
| Exact accepted raw answer lost during whitespace normalization | 1 |
| Numerical/statistical reference and evidence alignment needs adjudication | 2 |
| Anatomical specificity gap relative to the reference | 1 |

These are diagnostic categories, not a new semantic-accuracy metric or expert
adjudication. They do not approve every initial answer or expansion. Historical
official pool coverage remains 75/95, and production code was not changed.

The processing case is SPDL1, question `67df1a1d18b1e36f2e000065`.
The exact reference answer containing U+2009 thin spaces appears as the initial
answer in 11/30 raw max-five sampled responses: Llama 5, Qwen3 5, Ministral 1.
The expansion parser preserves those characters. The ensemble `merge_pool`
stores `clean(candidate["answer"])`, and `clean()` collapses Unicode whitespace
to ordinary ASCII spaces. The reference is not normalized by that helper.
An isolated check with the official Java scorer accepted the original thin-space
answer and rejected the cleaned ordinary-space answer. This check did not alter
historical labels, rankings or scores.

The two numerical cases are MS survival and systemic-sclerosis onset age. Their
supplied snippets describe different cohorts, time origins and statistical
quantities; their generated values must not be declared equivalent to the
reference numbers without adjudication. Fontan initial answers name the heart,
but the pool never names the vena cava required by the more specific reference.
That is a relative specificity gap, not proof of absent anatomical knowledge.

Only two of these 20 cases are covered in the original ten-sample pool: the
Calabash chalk question (exact alias `Antiemetic`) and epilepsy exome count
(exact alias `>54,000 human exomes`). Neither greedy pool covers any of the 20.

The complete observed-case table, exact evidence quotes, initial-answer
distributions and official Unicode reproduction are saved under local
`Artifacts/max5_ensemble_pool_absences_test95_v1/inspection/`. Aggregate findings
and provenance are recorded in
`results/max5_ensemble_pool_absence_audit_test95_20261019.json`.

The immediate processing follow-up is a separately labeled, gold-blind
surface-preservation comparison on all 95 questions. Keep normalized dedup keys
separate from submission surfaces and assess any new gains/losses before changing
the production pipeline. Do not infer a revised aggregate score from this single
case, and do not modify the frozen feature code used by queued experiments in
place.
