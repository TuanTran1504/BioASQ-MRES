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
