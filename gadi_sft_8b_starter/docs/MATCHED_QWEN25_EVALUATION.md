# Evaluate the completed Qwen2.5 expansion SFT adapters

This compares each completed SFT adapter with its matching unadapted backbone on
the same 160 development questions. It uses the equivalent-expansion v1 prompt,
every snippet, greedy decoding, a 6,144-token total limit, a 512-token output limit,
and the existing parser's ten-candidate cap. Final top-five answers retain generated
order. No learned selector, pooling, DPO or official test data is involved.

The adapters are pinned to completed Gadi training jobs:

| Size | Training job | Adapter directory under `outputs/expansion_sft/` |
| --- | --- | --- |
| 0.5B | 180598547.gadi-pbs | `qwen25-05b-expansion-sft-180598547.gadi-pbs/adapter` |
| 3B | 180598549.gadi-pbs | `qwen25-3b-expansion-sft-180598549.gadi-pbs/adapter` |

Each evaluation job runs its base condition and then SFT in separate processes,
freeing GPU memory between conditions. It checks completed full-training metadata,
the original training/validation hashes, and disjointness from all 160 development
IDs before loading a model. Model downloads are disabled.

## Validate from a Gadi login node

```bash
cd /scratch/nl78/$USER/BioASQ-MRES
git pull --ff-only
cd gadi_sft_8b_starter
module load python3/3.12.13
source /scratch/nl78/$USER/venvs/bioasq-8b/bin/activate
python3 scripts/run_matched_qwen25_expansion.py --model-size 05b --mode validate
python3 scripts/run_matched_qwen25_expansion.py --model-size 3b --mode validate
```

The ignored files `data/expansion_dev160.jsonl` in the bundle and the original
`../data/BioASQ_expansion_sft/seed3407_teacher_strict_v1/` dataset must already be
present on Gadi. Validation reports missing or mismatched inputs explicitly.

## Submit smoke tests

To validate both models and submit all four jobs at once, run:

```bash
bash scripts/submit_matched_qwen25_evaluation.sh
```

This prints and saves every submitted job ID under `outputs/matched_qwen25/`.
Each full job has a PBS `afterok` dependency on its matching smoke test, so it
starts only if that smoke test exits successfully. See the
[PBS dependency documentation](https://www.nas.nasa.gov/hecc/support/kb/commonly-used-qsub-command-options_175.html).
Run the submission command once; each invocation submits a new set of jobs.

Alternatively, submit and inspect the smoke tests manually:

```bash
qsub jobs/evaluate_qwen25_05b_expansion_smoke.pbs
qsub jobs/evaluate_qwen25_3b_expansion_smoke.pbs
```

Each smoke test generates four questions for both conditions. After both finish,
inspect the scheduler exit codes, stderr and these manifests:

```bash
find outputs/matched_qwen25 -maxdepth 2 -path '*smoke*' \
  -name manifest.json -print -exec cat {} \;
```

Require `status: complete` for each smoke manifest, then submit the full jobs:

```bash
qsub jobs/evaluate_qwen25_05b_expansion.pbs
qsub jobs/evaluate_qwen25_3b_expansion.pbs
```

Each full job requests one V100 GPU and four hours. Its directory is
`outputs/matched_qwen25/qwen25-<size>-expansion-dev160-<PBS_JOBID>/`.
The outer manifest lists portable relative paths to each generation run. Both
conditions must contain all 160 generation records; parse failures remain in the
denominator. Check completion with:

```bash
find outputs/matched_qwen25 -maxdepth 2 -path '*dev160*' \
  -name manifest.json -print -exec cat {} \;
```

## Score completed full runs

From the full repository on a machine with the Python dependencies, Java and the
BioASQ evaluator JAR, run the following for each completed experiment directory:

```bash
python scripts/analyze_matched_qwen_expansion.py \
  gadi_sft_8b_starter/outputs/matched_qwen25/EXPERIMENT_DIRECTORY
```

This can run on a Gadi login node for these small offline evaluations if the
environment and Java are available, or locally after retrieving the complete
experiment directories. For local retrieval, run from PowerShell in the repository:

```powershell
New-Item -ItemType Directory -Force gadi_sft_8b_starter/outputs/matched_qwen25 | Out-Null
scp -r 'dt9536@gadi-dm.nci.org.au:/scratch/nl78/dt9536/BioASQ-MRES/gadi_sft_8b_starter/outputs/matched_qwen25/qwen25-*-expansion-dev160-*' gadi_sft_8b_starter/outputs/matched_qwen25/
```

The scorer verifies matching input and prompt hashes, generation settings, full
question/snippet records and denominators. It writes `comparison_summary.json`
and `comparison_per_question.json`. The summary reports strict/lenient accuracy,
MRR at five, coverage at 1/5/10, candidate diagnostics and paired SFT-minus-base
differences with 10,000-resample question-bootstrap intervals (seed 3407).

MRR uses the first accepted candidate among the first five in generated order.
Coverage at ten is an offline diagnostic. These are development comparisons;
they do not establish unseen-test improvement. The saved question-level generation
files include dataset text and gold for offline scoring and remain under ignored
`outputs/`; publish permitted aggregate summaries separately.
