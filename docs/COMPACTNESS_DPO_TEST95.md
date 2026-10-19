# Three fixed compactness ensembles on test95

The follow-up [ten-sample matching-reranker workflow](COMPACTNESS_MATCHED_RERANKERS.md)
generates fitting/validation banks with one greedy plus ten sampled responses,
trains two configuration-specific rerankers and reuses this experiment's fixed
test95 pools. The results below continue to refer to the original frozen reranker.

This experiment compares the existing max-five SFT ensemble against two fixed
configurations on the same 95 factoid test questions from BioASQ 13B batches 1–4.
Each configuration is scored under greedy and ten-sample generation, using the
unchanged max-five logistic reranker. Sample-frequency voting is a secondary
comparison on the same pools.

| Configuration | Llama | Qwen3 | Ministral |
|---|---|---|---|
| `baseline` | Starting SFT | Starting SFT | Starting SFT |
| `strict` | Starting SFT | DPO epoch 1 | Starting SFT |
| `alternative` | DPO epoch 1 | DPO epoch 1 | DPO epoch 1 |

The strict choices come from completed validation144 selection. The alternative
allows individual coverage losses and is fixed at epoch 1 for all models before
this evaluation; it is not selected using test results. Starting SFT means the
exact loss-selected max-five SFT used to generate the preference banks.

## Pinned inputs

The default launcher validates these existing Gadi outputs:

| Model | Expanded DPO training job | Validation selection job | Strict selected epoch |
|---|---|---|---:|
| Llama | 181507870 | 181507871 | 0 |
| Qwen3 | 181529747 | 181529748 | 1 |
| Ministral | 181529764 | 181529765 | 0 |

Epoch-one adapters are read from each completed selection run's
`epoch-1/adapter`, checked against the retained epoch-one training checkpoint,
and pinned with their native tokenizer. The final loss-selected training adapter
is not substituted for epoch one. Existing strict selection outputs are unchanged.

The original Llama-only selection format omitted `focal_model`. The reader accepts
that legacy format only for Llama when its linked training record and pinned
configuration both identify Llama. All completion, validation-size, source,
selection and checkpoint-hash checks still apply. Missing companion identities
or explicit conflicting identities are rejected; historical manifests are not edited.

Historical SFT ten-sample configurations omitted `top_k`, while their inference
worker hardcoded `top_k=0`. Baseline reading accepts that absent field only via
an explicit compatibility option; a conflicting recorded value remains an error.
New epoch-one DPO configurations must record `top_k=0`. Archived files are not
rewritten, and all other decoding, source, evidence, draw and seed checks remain.

The baseline response archives come from
`outputs/sft_ensemble_test/comparison-181287698.gadi-pbs`.
The reranker is `max5/reranker.joblib` from
`outputs/sft_ensemble_reranker/train-181350943.gadi-pbs` (C=0.1).
All six SFT archives are reused; each model's DPO epoch-one responses are generated
once per decoding mode and reused across configurations.

Every source, preference partition, adapter, selection decision, tokenizer input
plan and reranker is checked before any job is submitted. Test IDs must be
disjoint from all 1,296 fitting and 144 validation questions. Gold answers are
written to a separate scoring file; inference receives question IDs, questions
and full supplied snippets only. The launcher preflights the official Java scorer.
Source files, job files and inference helpers are snapshotted for queued runs.

## Submit on Gadi

No new data upload is required if the completed outputs above remain on Gadi.

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_compactness_test95.sh
```

The launcher submits **seven jobs**: a smoke/full GPU chain for each of the three
models, followed by one CPU score job depending on all three full generations.
Preparation errors stop submission. Each smoke tests the four longest prompts
under both decoding modes. Full jobs evaluate all 95 questions. Generation jobs
request eight hours; smoke overrides this to one hour; scoring requests two hours.
No DPO training, new reranker fitting, paid judging or dev160 generation occurs.

Decoding matches the existing test archives: seed 3407, greedy or 10 samples,
temperature 0 or 0.8, top-p 0.95, top-k 0, up to five candidates per response and
512 output tokens. Test sampled draws use seed indices 1–10, matching the test
baseline, not validation's bank-aligned offset. Full evidence is preserved and
context allocation must match the baseline token IDs and allocated length.

Optional path overrides are `--comparison`, `--rerankers`,
`--llama31-selection`, `--qwen3-selection` and `--ministral3-selection`.
They do not change the declared configurations and must pass the same validation.

## Monitor

```bash
record=$(ls -t outputs/compactness_test95/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Each full generation saves
`outputs/compactness_test95/MODEL-generation-JOBID.gadi-pbs/manifest.json`.
The final score job saves `outputs/compactness_test95/comparison-JOBID.gadi-pbs`.
If a job fails, inspect its stderr using its recorded numeric job ID; dependent
jobs will not run successfully until that failure is resolved.

## Read results

The score job prints all six primary conditions, token/candidate means, paired
MRR deltas, confidence intervals and full-pool gains/losses. To inspect the saved
summary after a successful scoring job:

```bash
module load python3/3.12.13
python3 - <<'PY'
import json
from pathlib import Path

runs = sorted(Path("outputs/compactness_test95").glob("comparison-*"),
              key=lambda p: p.stat().st_mtime, reverse=True)
complete = [p for p in runs if (p / "manifest.json").exists()
            and json.loads((p / "manifest.json").read_text())["status"] == "complete"]
if not complete:
    raise SystemExit("No completed test95 comparison yet.")
root = complete[0]
d = json.loads((root / "comparison_summary.json").read_text())
print("Run:", root)
for mode in ("greedy", "sampling10"):
    print("\n" + mode)
    for setting in ("baseline", "strict", "alternative"):
        name = f"{setting}_{mode}"
        m = d["conditions"][name + "_max5_reranker"]["metrics"]
        z = d["ensemble_generation"][name]
        print(f"  {setting}: MRR={m['mrr_at5']:.6f}, "
              f"top5={round(m['coverage_at5']*95)}/95, "
              f"full pool={round(m['coverage_full_pool']*95)}/95, "
              f"tokens/response={z['mean_output_tokens_per_response']:.2f}")
    for setting in ("strict", "alternative"):
        name = f"{setting}_minus_baseline_{mode}_max5_reranker"
        c = d["paired_contrasts"][name]["metrics"]["mrr_at5"]
        t = d["coverage_transitions"][name]["coverage_full_pool"]
        print(f"  {setting}: delta={c['difference']:+.6f}, "
              f"95% CI={c['paired_bootstrap_95ci']}, "
              f"coverage gained={t['gained']}, lost={t['lost']}")
print("Warnings:", d["quality_warnings"])
PY
```

Outputs include per-batch results, per-question metrics, ranked candidates,
coverage gains/losses and MRR changes, per-model SFT/DPO generated-order results,
model output-token/candidate counters and configuration-level token costs.
Empty or unparsable outputs remain in denominators. New official matches must
reproduce every historical baseline candidate label and baseline round-robin
metric before results are marked complete.

## Interpretation

This test95 set has already been inspected repeatedly; this is an exploratory
comparison, not a fresh untouched final test. Report every fixed configuration,
including regressions. Joint all-DPO behaviour cannot be inferred from the three
separate one-model validation replacements.

Pooling, relation features and surface handling are deliberately identical to the
historical experiment to isolate checkpoint changes. In particular, the known
Unicode-whitespace surface issue is not silently fixed here; any surface-preserving
rescore must be a separate, declared comparison. Official nonacceptance does not
establish semantic incorrectness. Token savings do not guarantee proportional
walltime savings, especially when comparing generation times from archived runs.
