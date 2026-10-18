# Expanded Llama compactness DPO

Prepared on 18 October 2026. This is a larger controlled follow-up to the
36-pair Llama pilot, **not a completed training result or a full 1,296-question
preference dataset**. The user submits the jobs on Gadi.

The [Qwen3 and Ministral extension](COMPANION_COMPACTNESS_DPO.md) reuses this
recipe with model-specific reviewed pairs, references and independent runs.

The pilot reduced output tokens but changed dev160 ensemble MRR by only
+0.00125 (95% paired bootstrap CI -0.008125 to +0.010625). The KN-93 regression
showed that useful accepted wording can become less frequent or appear later.
Those development cases are diagnostic only and are not added to training.

## Data

The original review admitted compactness pairs only when substituting a shorter
saved Llama response also reduced the combined pool. The expanded review also
admits pool-size ties where another model already supplies the omitted wording,
and records the actual pool effect separately. The source remains the existing
fitting/validation max-five banks and the question-disjoint generator-feedback
preference audit, job **181434386**. No new fitting-bank generation was needed.

| Partition | Compactness pairs | Retention controls | Pairs / distinct questions |
|---|---:|---:|---:|
| Fitting | 96 | 12 | 108 / 108 |
| Internal validation | 21 | 1 | 22 / 22 |

The fitting selection caps compactness at eight pairs per retained control,
with at most one pair per question per objective and deterministic seed 3407.
This **8:1 cap is an experimental choice**, not an established optimal ratio.
It keeps all 12 eligible retention-control questions and limits deletion-only
examples. Validation uses all eligible question-capped pairs. Before the cap,
the clean fitting shortlist had 388 compactness pairs across 275 questions and
16 retention pairs across 12 questions.

Of the 96 fitting compactness pairs, 10 shrink the ensemble pool and 86 leave
its size unchanged. Of the 21 validation compactness pairs, 3 shrink it and 18
leave it unchanged. Retention controls prefer a longer response preserving an
accepted expression absent from its shorter alternative. They are recorded
separately from deletion examples.

Each pair contains two **naturally generated whole responses** for the same
question and prompt. Compactness keeps the same initial answer and relation
tag, prefers a strict ordered subset, and preserves every officially accepted
expression in the rejected response. The recorded frozen-reranker replacement
cannot reduce full-pool coverage or MRR; the other 11 response slots stay fixed.
These are local guarantees on saved responses, not guarantees about a trained model.

Preferred expressions require high-confidence correctness, evidence support,
equivalence and relation judgments. Unresolved/conflicting reviews are excluded.
The rejected initial answer must also be correct and supported. Compactness can
delete **valid optional expressions**, not just harmful ones: the judge assessed
82 removed fitting expressions as valid and 59 as harmful. Describe this as
compactness with answer retention, rather than pure factual-error correction.

## Review and provenance

GPT-4.1-mini reviewed 1,040 candidate pairs, represented by 1,736 responses and
3,016 unique judgment units. All 726 pilot units were reused. The same US$10
budget ledger now records **US$1.0278268 total** across both runs; the additional
cost was US$0.7807892. Two length-truncated attempts and their successful recovery
are included. Successful cached requests were reused. No credentials or API
response caches are included in the transfer package.

The judge saw the question, evidence and relations without gold aliases, model
identity or preference direction. The prior four assistant vetoes remain.
Five additional exclusions hold out population-scope mismatch, paralogon versus
gene-copy counts, conflicting HEp-2 origin descriptions, and missing numerical
approximation qualifiers. IDs and reasons are pinned in the expanded config.
These are automated labels plus assistant screening, **not independent
biomedical expert certification**.

Local package: `Artifacts/generator_feedback_dpo_expanded_v1`.
Archive: `Artifacts/generator-feedback-dpo-expanded-v1.tar.gz`.
All source files, annotations, splits and configuration are hashed. Before
submission, validation reconstructs the pairs and checks them against the
original Gadi bank strings and exact starting SFT adapter.

## Training and selection

Only Llama is trained, starting again from the exact loss-selected max-five SFT
adapter used for the banks. Its initial SFT is the frozen DPO reference.
Qwen3, Ministral and the existing max-five logistic reranker stay fixed.

| Setting | Value |
|---|---|
| Objective | Standard whole-response sigmoid DPO |
| Beta / learning rate | 0.1 / 2e-6 |
| Epochs / effective pair batch size | 3 / 4 |
| Planned optimizer steps / warmup | 81 / 9 |
| Allocation / output ceiling | 8,704 total tokens / 512 new tokens |
| Preference-loss validation | 22 reviewed pairs; evaluated each epoch |
| Downstream selection | All 144 internal-validation questions |

All three complete epoch checkpoints are retained. A separate GPU job evaluates
them after training, with the saved starting-SFT and companion banks as epoch
zero. Each DPO epoch generates one greedy and four stochastic responses per
validation question, up to five candidates per response. Sampling uses
temperature 0.8, top-p 0.95 and native-bank seed indices 2 through 5. Complete
evidence and the original 8,704-token allocation are retained; overflow fails
instead of truncating evidence. Gold aliases are supplied only to scoring.

The predeclared rule selects using the four-sample ensemble. An eligible epoch:

- loses no previously covered full-pool question, for Llama alone or the ensemble;
- has ensemble MRR at least as high as epoch zero;
- uses no more mean Llama output tokens than epoch zero;
- has no additional empty, output-ceiling or surplus-candidate responses.

Among eligible epochs, choose highest MRR, then fewest output tokens, then the
earliest epoch. **Epoch zero remains the fallback**. These guards apply only to
the validation draws. Greedy scores and preference losses are reported separately.

The reranker's regularization was already selected on these 144 questions, so
these are internal selection diagnostics, not independent performance estimates.
Dev160 and the 95 official-test questions are excluded from training and this
selection. Four-sample selection does not establish ten-sample improvement.
No dev/test evaluation is automatically submitted.

## Transfer and submit

Local PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
scp .\Artifacts\generator-feedback-dpo-expanded-v1.tar.gz dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/
```

On Gadi:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
mkdir -p outputs/generator_feedback_dpo/reviewed-expanded-v1
tar -xzf /scratch/nl78/$USER/generator-feedback-dpo-expanded-v1.tar.gz \
  -C outputs/generator_feedback_dpo/reviewed-expanded-v1
bash scripts/submit_expanded_generator_dpo.sh \
  outputs/generator_feedback_dpo/reviewed-expanded-v1
```

Three dependent GPU jobs: training-and-selection smoke (one-hour limit),
training (four hours), then validation selection (12 hours). Failure prevents
dependants from running. No paid calls occur on Gadi. Code and inference worker
are snapshotted, so later pulls do not change queued workers.

Monitor from the starter directory:

```bash
record=$(ls -t outputs/generator_feedback_dpo_expanded/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $2}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Training output is under
`outputs/generator_feedback_dpo_expanded/llama31-compactness-train-TRAIN_JOB.gadi-pbs/`.
Its `validation-selection-SELECT_JOB.gadi-pbs/comparison_summary.json` reports
all four epoch scores, eligibility, coverage transitions, token counts and the
selected epoch. `selected_adapter/` contains the winner, including the original
SFT if epoch zero wins. The training `adapter/` remains the best preference-loss
checkpoint and is not overwritten by downstream selection.

## Inspect the epoch-1 Llama coverage loss

Completed selection job **181507871** retained epoch zero. On validation144,
epoch 1 had ensemble MRR 0.611227 versus starting SFT 0.606944, with mean Llama
tokens 46.80 versus 61.56. It lost one previously covered Llama question while
losing none at ensemble level. These are internal selection diagnostics.

Export that loss without changing selection, running models or judging again:

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
python3 ../scripts/export_expanded_dpo_coverage_loss.py \
  --comparison outputs/generator_feedback_dpo_expanded/llama31-compactness-train-181507870.gadi-pbs/validation-selection-181507871.gadi-pbs \
  --epoch 1 \
  --output outputs/generator_feedback_dpo_expanded/epoch1-coverage-loss-181507871 \
  --archive /scratch/nl78/$USER/expanded-dpo-epoch1-loss-181507871.tar.gz
```

The exporter derives the question ID from the saved per-question full-pool
coverage transitions. It checks source hashes, draws, seeds, evidence and cached
official candidate labels, and reconstructs the focal Llama and ensemble metrics.
It loads the exact hashed parser from the original submission without writing
bytecode there. No Java rerun or optional Python packages are needed.

`report.md` shows the four paired sampled responses and frozen ensemble top five.
`cases.jsonl` adds full evidence, gold aliases, the greedy response pair and
candidate acceptance flags. `source_responses.jsonl` includes the unchanged
companions. Comparing normalized initial strings distinguishes an initial-wording
change from deletion of an accepted later expression with the same initial
answer. Neither event establishes a semantic error without examining evidence.

On local PowerShell:

```powershell
Set-Location C:\Users\Dustin\Projects\BioASQ-MRES
New-Item -ItemType Directory -Force .\Artifacts\expanded_dpo_epoch1_loss_181507871 | Out-Null
scp dt9536@gadi.nci.org.au:/scratch/nl78/dt9536/expanded-dpo-epoch1-loss-181507871.tar.gz .\Artifacts\
if ($LASTEXITCODE -eq 0) {
    tar -xzf .\Artifacts\expanded-dpo-epoch1-loss-181507871.tar.gz -C .\Artifacts\expanded_dpo_epoch1_loss_181507871
}
```

The original selection directory and selected adapter are read-only inputs.
Choose a new export path if repeating the command; existing exports are not
overwritten. Outcome-selected validation cases are diagnostic and not approved
new fitting preferences.
