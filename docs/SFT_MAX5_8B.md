# Five-answer expansion SFT on three 8B backbones

Freshly train Llama-3.1-8B, Qwen3-8B and Ministral-3-8B from the same pinned base
snapshots as the ten-answer experiment. Use its validated teacher examples and
unchanged 1,296 fitting / 144 validation / 160 disjoint development split.

The derived dataset changes the system prompt from up to TEN to up to FIVE and
retains at most the first five target expressions in their existing order. The
first target is a recorded gold alias in every example and remains unchanged.
Question/snippet messages are identical. No new teacher generation or paid API
calls are needed. Existing data, training outputs and models are preserved.

Source checks found 111 fitting and 12 validation targets exceeding five. All
questions remain; only those targets lose trailing variants. Derived files live
under `gadi_sft_8b_starter/data/matched_8b_sft_max5`. The source hashes and
deterministic transformation are checked on every build and training preflight.

Training settings match ten-answer SFT: three epochs, seed 3407, rank/alpha 32,
dropout 0.05, learning rate 0.0002, batch one with accumulation 16, all epoch
snapshots retained and no early stopping. Ministral uses eager execution.

Each model independently exports the lowest validation-loss epoch and the
highest validation full-pool coverage epoch. Coverage uses ten sampled responses
per question with up to five candidates each (up to 50 slots before deduplication),
temperature 0.8, top-p 0.95, 8,192 total tokens and 512 output tokens. All 144
validation questions/snippets are retained; smoke checks all prompt lengths and
generates the four longest. Official BioASQ matching supplies coverage labels,
with failed parses retained in denominators. Candidate-cap violations are archived.

These selection scores are not development/test results. For a later comparison
of training formulations, evaluate both the ten-answer SFT and five-answer SFT
checkpoints with the same five-answer prompt, candidate budget and selection rule.
Full-pool comparisons of ten versus five candidates per draw have unequal budgets.

## Submit on Gadi

```bash
cd /scratch/nl78/$USER/BioASQ-MRES/gadi_sft_8b_starter
git pull --ff-only
bash scripts/submit_sft_max5_8b.sh
```

This builds the derived dataset on Gadi, validates all selected historical base
pins and the Java scorer, then submits nine jobs: smoke, full training and
checkpoint selection per model. Full stages depend on preceding success. To
restrict backbones append `llama31`, `qwen3` or `ministral3`. Fresh five-answer
smoke training is required; previous ten-answer smoke adapters are incompatible.
Full training requests 12 hours and selection 24 hours, one V100 per job. Local
tests cover data, provenance and dependencies; actual CUDA execution is checked
by the Gadi smoke jobs. No jobs were submitted from the local machine because
Gadi rejected its noninteractive SSH authentication.

## Monitor

The first max-five submission's smoke jobs 181021908, 181021911 and 181021914
failed before model loading with `No module named src.utility.training_execution`.
The max-five data converter imported an evaluation launcher that put the
repository root ahead of the portable bundle in Python's search path. The
repository's different `src` package then hid the bundled GPU helpers. Prompt
conversion now imports a pure helper, and training explicitly prioritises the
bundle root. Targets and training settings are unchanged. The dependent train
and selection jobs never started. After pulling the fix, resubmit with
`bash scripts/submit_sft_max5_8b.sh` to obtain fresh smoke/train/select jobs.

```bash
record=$(ls -t outputs/sft_max5_8b/submission-*/jobs.tsv | head -n 1)
cat "$record"
job_ids=$(awk -F '\t' 'NR>1 {print $3}' "$record")
qstat -swx $job_ids
qstat -fx $job_ids | grep -E 'Job Id|job_state|Exit_status|comment'
```

Full outputs are saved under `outputs/sft_max5_8b/MODEL-expansion-sft-TRAIN_JOB_ID.gadi-pbs`.
The selection directory `selection-SELECT_JOB_ID.gadi-pbs` contains its manifest,
epoch metrics, validation generations and `selected_adapters/best_coverage` and
`selected_adapters/best_eval_loss`. Generation/scoring smoke must finish with
exit status zero before full training begins. Existing run names remain intact.
