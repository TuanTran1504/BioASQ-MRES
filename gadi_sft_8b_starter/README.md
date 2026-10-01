# BioASQ 8B Experiments for Gadi

This is a small, portable package for Gadi. It supports the original SFT jobs and a gold-blind exact-span expansion experiment. It deliberately excludes prior adapters, DPO data, candidate banks, notebooks, test evaluations, and historical artifacts.

It trains `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` with 4-bit LoRA and generated-dev BioASQ MRR checkpoint selection.

The expansion job runs the unchanged instruction checkpoint without training. It gives the model each of the fixed 160 development questions and all of its snippets, asks for up to ten literal answer spans, and writes candidates in the same artifact format used by the local 3B experiment.

## Included variants

| Variant | Training examples | Dev questions | Purpose |
| --- | ---: | ---: | --- |
| `full_resources_per_alias` | 1,804 | 160 | Full-resource per-gold-alias SFT, matching the full-resource notebook design. |
| `evidence_per_supported_alias` | 1,352 | 160, plus a 129-question supported subset | Evidence-grounded per-supported-alias SFT, matching the evidence-grounded notebook design. |

Both variants use a batch size of 1, gradient accumulation of 32, sequence length 4,096, rank-32 LoRA with dropout 0.05, linear warmup of 5 updates, and generated-dev MRR model selection. Training artifacts are written to `outputs/runs/<timestamp>-<run-name>/` and are never overwritten by a new run name.

## First use on Gadi

1. Copy this complete directory to `/scratch/nl78/$USER/`.
2. Load modern Python, then validate the bundle: `module load python3/3.12.13` followed by `python3 scripts/verify_bundle.py`.
3. Create the environment once on a login node:

```bash
PYTHON_MODULE=python3/3.12.13 bash scripts/setup_environment.sh
```

4. Accept Meta's Llama license using your Hugging Face account, export `HF_TOKEN`, then cache the model once on a login node:

```bash
export HF_HOME=/scratch/nl78/$USER/hf_cache
export HF_TOKEN=<your-token>
python3 scripts/download_base_model.py
unset HF_TOKEN
```

Alternatively, keep the token in a permission-protected file outside the repository and avoid exporting it:

```bash
python3 scripts/download_base_model.py \
  --token-file /scratch/nl78/$USER/.secrets/hf_token.txt
```

5. Edit the `#PBS -q` line in a job file if `gpuvolta` is not the GPU queue available to `nl78`. Check with `qstat -Q` or submit the smoke test first.
6. Submit the smoke test, then a full run:

```bash
qsub jobs/00_gpu_smoke_test.pbs
qsub jobs/train_full_resources_8b.pbs
# or
qsub jobs/train_evidence_grounded_8b.pbs
```

Monitor with `qstat -swx <job-id>`. PBS output and errors are retained by Gadi; model, checkpoints, logs, dev predictions, metrics, and manifests are stored under `outputs/`.

## Important notes

- Run `scripts/download_base_model.py` on a login node, not a compute node. Training defaults to `--local-files-only` to prevent compute-node downloads.
- The first GPU job should be the smoke test. It validates CUDA, the environment, data, model cache, and the BioASQ Java evaluator with only eight training/dev examples.
- Do not use `--resume-from-checkpoint auto` with a reused run directory. This launcher creates timestamped managed run folders, so every normal submission starts a distinct experiment.
- The PBS resource requests are a conservative one-GPU starting point for 8B 4-bit LoRA. If the smoke test reports out-of-memory, reduce `max_seq_length` or increase the requested GPU-memory class if your allocation permits it.

## Exact-span expansion experiment

Before transfer, create the fixed dev-set export from the full repository:

```bash
python scripts/export_gadi_expansion_dev.py
python gadi_sft_8b_starter/scripts/verify_expansion_bundle.py
```

The generated `gadi_sft_8b_starter/data/` directory is ignored by Git but must be included in the transfer to Gadi. It contains gold aliases for offline scoring; the runner sends only the question and snippets to the model.

On Gadi, after the environment and model cache have been prepared, submit the four-question smoke test and inspect its status before starting all 160 questions:

```bash
cd /scratch/nl78/$USER/gadi_sft_8b_starter
qsub jobs/01_expansion_smoke_test.pbs
qstat -swx <job-id>
cat outputs/expansion/*smoke*/status.json

qsub jobs/run_extractive_expansion_8b.pbs
```

The full job uses greedy decoding, a 6,144-token total sequence limit, 512 output tokens, and fails if all snippets do not fit. It writes progress after every question under `outputs/expansion/`. To resume an interrupted directory without repeating completed questions, run the same script in a GPU job with `--resume-run outputs/expansion/<run-directory>`.

After copying the completed run directory back to this repository, score it with the existing official BioASQ analyzer:

```bash
python scripts/analyze_gadi_expansion.py <run-directory>
```

See [docs/TRANSFER_TO_GADI.md](docs/TRANSFER_TO_GADI.md) for the transfer and operating procedure.
